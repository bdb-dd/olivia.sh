#!/usr/bin/env python3
"""Offline end-to-end smoke test: the eval harness driving a model *through the
durable model router* (``model_router.py``, the ``add-small-partition-proxy``
work this branch now sits on).

It stands up the **real** router and the **real** ``anthropic_proxy`` in front of
a stub vLLM backend, then exercises the exact path the L0–L2 evals use:

    eval client ──/v1/messages──► anthropic_proxy ──/v1/chat/completions──►
        model_router ──(routes by `model`)──► stub vLLM (/v1/models, /v1/chat/completions)

No cluster, no GPU: the router's squeue-based discovery is the only piece that
can't run on a laptop, so we shim *only* ``_server_nodes`` to point at the local
stub and let the real ``_probe`` / ``resolve`` / forward logic run unchanged.

What it proves:
  * a request whose ``model`` is a **preset name / alias** (``laguna`` /
    ``laguna-m1``) is routed to the live backend and **rewritten** to the served
    repo id the backend actually advertises (the router's core contract);
  * ``/v1/models`` reports the live preset;
  * a **known** preset that isn't serving → 503, an **unknown** model → 404
    (so the harness gets actionable errors, not hangs);
  * streaming SSE survives the hop;
  * the full ``anthropic_proxy → router`` chain works, i.e. the harness's real
    entrypoint reaches the backend through the router.

Run (needs aiohttp — use the repo .venv):

    .venv/bin/python evals/router_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Error: aiohttp required (pip install -r requirements.txt, or use .venv).",
          file=sys.stderr)
    sys.exit(2)

import model_router  # noqa: E402  (real router under test)

# The stub advertises exactly what the `laguna` preset's repo id is, so the
# router's preset→served_name resolution has something live to match.
SERVED_ID = "poolside/Laguna-M.1-FP8"
STUB_REPLY = "hello from stub"


# --------------------------------------------------------------------------- #
# Tiny PASS/FAIL harness (mirrors evals/protocol/selftest.py style)           #
# --------------------------------------------------------------------------- #
_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}{('  — ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
# Stub vLLM backend                                                           #
# --------------------------------------------------------------------------- #
def make_stub(recorder: dict) -> web.Application:
    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list",
             "data": [{"id": SERVED_ID, "object": "model", "owned_by": "stub"}]})

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        recorder["last_chat"] = body
        if body.get("stream"):
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            first = {"id": "stub", "object": "chat.completion.chunk",
                     "choices": [{"index": 0,
                                  "delta": {"role": "assistant", "content": STUB_REPLY},
                                  "finish_reason": None}]}
            last = {"id": "stub", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            await resp.write(f"data: {json.dumps(first)}\n\n".encode())
            await resp.write(f"data: {json.dumps(last)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response({
            "id": "stub", "object": "chat.completion", "created": 0,
            "model": body.get("model"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": STUB_REPLY},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        })

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    return app


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
async def start_app(app: web.Application) -> tuple:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_router_app(stub_port: int) -> web.Application:
    """The real router, with squeue discovery shimmed to the local stub."""
    async def _fake_server_nodes(_self) -> list:
        return ["127.0.0.1"]
    model_router.BackendDiscovery._server_nodes = _fake_server_nodes

    class NS:
        backend_port = stub_port
        discovery_ttl = 15.0
        batch_tokens = 15
        batch_chars = 100
        batch_delay_ms = 150
        no_flush_newline = False
        empty_timeout = 0          # disable the spindown watchdog in-test
        token = None

    # create_app is async (sets up the ClientSession on startup).
    return NS


# --------------------------------------------------------------------------- #
# Stage 1 — drive the router directly                                         #
# --------------------------------------------------------------------------- #
async def test_router_direct(session: aiohttp.ClientSession, base: str,
                             recorder: dict) -> None:
    print("Stage 1 — router routing contract (model field → backend)")

    async with session.get(f"{base}/v1/models") as r:
        body = await r.json()
    ids = [m.get("id") for m in body.get("data", [])]
    presets_seen = [m.get("olivia_preset") for m in body.get("data", [])]
    check("/v1/models lists the live served id", SERVED_ID in ids, str(ids))
    check("/v1/models tags it with its preset", "laguna" in presets_seen,
          str(presets_seen))

    # preset name → routed + rewritten to the served repo id
    payload = {"model": "laguna", "stream": False, "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]}
    async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
        status = r.status
        body = await r.json()
    check("preset name 'laguna' routes (200)", status == 200, f"status={status}")
    check("router rewrote model → served repo id",
          recorder.get("last_chat", {}).get("model") == SERVED_ID,
          str(recorder.get("last_chat", {}).get("model")))
    check("backend reply flows back to client",
          body.get("choices", [{}])[0].get("message", {}).get("content") == STUB_REPLY)

    # alias → same backend
    recorder.pop("last_chat", None)
    payload["model"] = "laguna-m1"
    async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
        status = r.status
    check("alias 'laguna-m1' routes (200)", status == 200, f"status={status}")
    check("alias also rewritten to served id",
          recorder.get("last_chat", {}).get("model") == SERVED_ID)

    # known preset, not serving → 503
    payload["model"] = "glm52"
    async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
        status503 = r.status
        body = await r.json()
    check("known-but-idle preset → 503", status503 == 503, f"status={status503}")
    check("503 names the preset", body.get("error", {}).get("preset") == "glm52",
          json.dumps(body))

    # unknown model → 404
    payload["model"] = "no-such-model-xyz"
    async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
        status404 = r.status
    check("unknown model → 404", status404 == 404, f"status={status404}")

    # streaming hop survives
    recorder.pop("last_chat", None)
    payload = {"model": "laguna", "stream": True, "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]}
    async with session.post(f"{base}/v1/chat/completions", json=payload) as r:
        status = r.status
        text = await r.text()
    check("streaming routes (200)", status == 200, f"status={status}")
    check("streamed SSE carries the backend tokens", STUB_REPLY in text)


# --------------------------------------------------------------------------- #
# Stage 2 — full harness entrypoint: anthropic_proxy → router                 #
# --------------------------------------------------------------------------- #
async def test_through_anthropic_proxy(router_port: int, recorder: dict) -> None:
    print("Stage 2 — harness path: anthropic_proxy(/v1/messages) → router → backend")
    proxy_port = free_port()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO_ROOT, "anthropic_proxy.py"),
         "--upstream", f"http://127.0.0.1:{router_port}",
         "--model", "laguna",                       # router resolves preset → backend
         "--listen-port", str(proxy_port),
         "--keepalive-interval", "0"],               # no idle pings at the stub
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=REPO_ROOT)
    base = f"http://127.0.0.1:{proxy_port}"
    try:
        recorder.pop("last_chat", None)
        async with aiohttp.ClientSession() as session:
            # wait for the proxy to accept connections
            up = False
            for _ in range(50):
                if proc.poll() is not None:
                    break
                try:
                    async with session.get(f"{base}/", timeout=aiohttp.ClientTimeout(total=2)) as r:
                        if r.status < 500:
                            up = True
                            break
                except Exception:
                    await asyncio.sleep(0.2)
            check("anthropic_proxy came up", up,
                  "exited early" if proc.poll() is not None else "no health within timeout")
            if not up:
                return

            body = {"model": "laguna", "max_tokens": 64, "stream": False,
                    "messages": [{"role": "user", "content": "hi"}]}
            async with session.post(f"{base}/v1/messages", json=body,
                                    headers={"x-api-key": "dummy",
                                             "anthropic-version": "2023-06-01"}) as r:
                status = r.status
                data = await r.json()
            check("/v1/messages through the router (200)", status == 200, f"status={status}")
            texts = [b.get("text", "") for b in data.get("content", [])
                     if b.get("type") == "text"]
            check("Anthropic reply carries the backend text",
                  any(STUB_REPLY in t for t in texts), json.dumps(data)[:300])
            check("router saw the routed request (rewritten to served id)",
                  recorder.get("last_chat", {}).get("model") == SERVED_ID,
                  str(recorder.get("last_chat", {}).get("model")))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
async def amain() -> int:
    recorder: dict = {}
    stub_runner, stub_port = await start_app(make_stub(recorder))
    NS = make_router_app(stub_port)
    router_app = await model_router.create_app(NS)
    router_runner, router_port = await start_app(router_app)
    base = f"http://127.0.0.1:{router_port}"
    try:
        async with aiohttp.ClientSession() as session:
            await test_router_direct(session, base, recorder)
        await test_through_anthropic_proxy(router_port, recorder)
    finally:
        await router_runner.cleanup()
        await stub_runner.cleanup()

    print()
    print(f"router smoke: {_passed} checks passed, {_failed} failed")
    return 1 if _failed else 0


def main() -> int:
    print("Router e2e smoke — anthropic_proxy → model_router → stub vLLM\n")
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
