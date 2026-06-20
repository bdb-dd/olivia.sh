#!/usr/bin/env python3
"""
Model router — multi-model vLLM front-end for the `small` partition
==================================================================

A durable, CPU-only reverse proxy that fans a single stable endpoint out to
whichever GPU (`accel`) inference jobs are currently running. It is the
in-cluster half of the "durable proxy on the small partition" design
(``plans/proposed/small_partition_proxy.md``).

Architecture::

    laptop ──SSH tunnel──► small-node:8080 (this router) ──internal Slingshot──► accel GPU node : vLLM
            (stable 7 days)        │  routes by request `model`               (each serves on :8000)
                                   ▼
                       squeue → running vLLM jobs → probe each /v1/models

Backend discovery is **job-name-independent**: the router lists the user's
running ``vllm-*`` jobs via ``squeue``, probes each head node's
``/v1/models`` over the internal fabric, and builds a {served model id → node}
map. So it doesn't matter whether servers are named ``vllm-server`` or, under the
per-branch scheme, ``vllm-<branch>`` — the router asks each server what it
actually serves. A client selects a model by the request's ``model`` field — a
preset name (``glm51``, ``kimi27``), an alias, or the served repo id — which is
resolved to a preset via ``presets.json`` and thus to a live backend.

Auto-spindown: a CPU job on the `small` partition bills its reservation while up,
so to respect NRIS fair-use the router **shuts itself down after a configurable
window with no GPU servers running** (default 30 min; ``--empty-timeout 0`` to
disable). Exiting ends the SLURM job and frees the allocation.

Usage (typically launched by run_proxy.sh inside the SLURM job)::

    python model_router.py --listen-port 8080 --backend-port 8000 --empty-timeout 1800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Optional

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Error: aiohttp required. Install with: pip install aiohttp",
          file=sys.stderr)
    sys.exit(1)

# Local modules (same directory). Ensure they're importable when launched by an
# absolute path from a SLURM job.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import presets  # noqa: E402
from vllm_proxy import BatchConfig, forward_stream, forward_nonstream  # noqa: E402

# Running vLLM server jobs are named with this prefix (`vllm-server`, or
# `vllm-<branch>` under the per-branch scheme). Build jobs are `build-vllm-gh200`
# (prefix `build-`), so this prefix cleanly excludes them.
SERVER_JOB_PREFIX = "vllm-"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BackendDiscovery:
    """Maps served model id -> live head node by probing running vLLM servers.

    Job-name-independent: ``squeue`` only tells us *which nodes* run a vLLM
    server; the authoritative "what does it serve" comes from each server's
    ``/v1/models``. SLURM is the liveness source of truth (a dead job is absent
    from squeue), and a server still loading simply isn't in the map yet.
    """

    def __init__(self, user: str, backend_port: int = 8000,
                 ttl: float = 15.0, probe_timeout: float = 3.0):
        self.user = user
        self.backend_port = backend_port
        self.ttl = ttl
        self.probe_timeout = probe_timeout
        self._served: dict = {}     # served model id (original case) -> node
        self._cache_at: float = 0.0

    async def _run(self, *cmd: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", errors="replace") if out else ""

    async def _expand_head_node(self, nodelist: str) -> str:
        """First hostname of a (possibly compressed) SLURM nodelist.

        Multi-node jobs report e.g. ``gpu-1-[90-91]``; the head node (where vLLM
        serves) is the first. Single-node lists pass straight through.
        """
        nodelist = nodelist.strip()
        if not nodelist or nodelist in ("(null)", "n/a"):
            return ""
        if "[" in nodelist or "," in nodelist:
            out = await self._run("scontrol", "show", "hostnames", nodelist)
            hosts = [h for h in out.split("\n") if h.strip()]
            return hosts[0].strip() if hosts else nodelist
        return nodelist

    async def _server_nodes(self) -> list:
        """Head nodes of the user's RUNNING vLLM server jobs (any naming)."""
        out = await self._run("squeue", "-u", self.user, "-h", "-o", "%j|%N|%T")
        nodes = []
        seen = set()
        for line in out.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            name, nodelist, state = (p.strip() for p in parts)
            if state != "RUNNING" or not name.startswith(SERVER_JOB_PREFIX):
                continue
            head = await self._expand_head_node(nodelist)
            if head and head not in seen:
                seen.add(head)
                nodes.append(head)
        return nodes

    async def _probe(self, session: aiohttp.ClientSession, node: str) -> list:
        """Served model ids reported by the vLLM server on `node` (empty if down/loading)."""
        url = f"http://{node}:{self.backend_port}/v1/models"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.probe_timeout)) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

    async def refresh(self, session: aiohttp.ClientSession) -> dict:
        nodes = await self._server_nodes()
        results = await asyncio.gather(*(self._probe(session, n) for n in nodes))
        served: dict = {}
        for node, ids in zip(nodes, results):
            for mid in ids:
                served[mid] = node   # last writer wins if a model runs twice
        self._served = served
        self._cache_at = time.monotonic()
        return served

    async def served_map(self, session: aiohttp.ClientSession, force: bool = False) -> dict:
        if force or (time.monotonic() - self._cache_at) > self.ttl:
            return await self.refresh(session)
        return self._served

    def invalidate(self) -> None:
        self._cache_at = 0.0

    async def resolve(self, session: aiohttp.ClientSession, model_string: str,
                      force: bool = False):
        """Resolve a request `model` to (node, served_id_to_send) or None.

        Tries the preset's served name first (so ``glm51`` → the repo id vLLM
        actually serves), then the raw string. Case-insensitive against the
        live served ids.
        """
        served = await self.served_map(session, force=force)
        if not served:
            return None
        candidates = []
        p = presets.match_model(model_string)
        if p and p.served_name:
            candidates.append(p.served_name)
        candidates.append(model_string)
        lowered = {sid.lower(): sid for sid in served}
        for cand in candidates:
            sid = lowered.get(cand.lower())
            if sid:
                return served[sid], sid
        return None


class Router:
    def __init__(self, discovery: BackendDiscovery, batch_config: BatchConfig,
                 token: Optional[str] = None):
        self.discovery = discovery
        self.batch_config = batch_config
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        # No total/read timeout: generations stream for minutes. Keep a connect
        # timeout so a dead backend fails fast instead of hanging the client.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self):
        if self.session:
            await self.session.close()

    # -- helpers ----------------------------------------------------------- #

    def _auth_ok(self, request: web.Request) -> bool:
        if not self.token:
            return True
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            auth = auth[7:]
        return auth == self.token or request.headers.get("x-api-key", "") == self.token

    @staticmethod
    def _err(status: int, message: str, **extra) -> web.Response:
        body = {"error": {"message": message, "type": "router_error", **extra}}
        return web.json_response(body, status=status)

    # -- handlers ---------------------------------------------------------- #

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_models(self, request: web.Request) -> web.Response:
        """List models currently served (OpenAI /v1/models shape)."""
        if not self._auth_ok(request):
            return self._err(401, "missing or invalid credentials")
        served = await self.discovery.served_map(self.session)
        data = []
        for sid in sorted(served):
            p = presets.match_model(sid)
            data.append({
                "id": sid,
                "object": "model",
                "owned_by": "olivia",
                "olivia_preset": p.canonical if p else None,
                "olivia_aliases": p.aliases if p else [],
            })
        return web.json_response({"object": "list", "data": data})

    async def _route(self, request: web.Request, path: str) -> web.StreamResponse:
        if not self._auth_ok(request):
            return self._err(401, "missing or invalid credentials")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return self._err(400, "invalid JSON body")

        model = body.get("model", "")
        target = await self.discovery.resolve(self.session, model)
        if target is None:
            # Distinguish "no such model" (404) from "known model, not running" (503).
            p = presets.match_model(model)
            if p is None:
                available = sorted(x.canonical for x in presets.registry().all())
                return self._err(404, f"unknown model {model!r}; send a preset name, "
                                       "alias, or served repo id", available_presets=available)
            served = await self.discovery.served_map(self.session)
            return self._err(503, f"preset {p.canonical!r} is not currently serving",
                             preset=p.canonical, serving=sorted(served))
        node, served_id = target
        body["model"] = served_id   # exact id the backend serves
        url = f"http://{node}:{self.discovery.backend_port}{path}"

        is_stream = bool(body.get("stream", False))
        try:
            if is_stream:
                return await forward_stream(self.session, url, request, body, self.batch_config)
            return await forward_nonstream(self.session, url, body)
        except aiohttp.ClientConnectorError:
            # Stale node (job moved/restarted). Re-discover once and retry.
            self.discovery.invalidate()
            target = await self.discovery.resolve(self.session, model, force=True)
            if target is None:
                return self._err(503, f"backend for {model!r} went away")
            node, served_id = target
            body["model"] = served_id
            url = f"http://{node}:{self.discovery.backend_port}{path}"
            _log(f"retrying {model!r} at fresh node {node} after connect error")
            if is_stream:
                return await forward_stream(self.session, url, request, body, self.batch_config)
            return await forward_nonstream(self.session, url, body)

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        return await self._route(request, "/v1/chat/completions")

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._route(request, "/v1/completions")

    async def handle_unknown(self, request: web.Request) -> web.Response:
        return self._err(
            404,
            f"router does not implement {request.method} {request.path}; "
            "supported: /v1/chat/completions, /v1/completions, /v1/models",
        )

    # -- auto-spindown watchdog ------------------------------------------- #

    async def idle_watchdog(self, empty_timeout: float, poll: float = 60.0):
        """Shut the router down after `empty_timeout`s with no GPU servers up.

        Respects NRIS fair-use: an idle CPU job still bills its small
        reservation, so when nothing is being fronted we free the allocation.
        Starting the router gives a full grace window before any server exists,
        and transient squeue/probe errors do NOT count toward the timeout.
        """
        if empty_timeout <= 0:
            _log("auto-spindown disabled (--empty-timeout 0)")
            return
        last_seen = time.monotonic()   # grace window at startup
        was_empty = False
        interval = max(5.0, min(poll, empty_timeout))
        while True:
            await asyncio.sleep(interval)
            try:
                served = await self.discovery.served_map(self.session, force=True)
            except Exception as e:
                _log(f"watchdog: discovery error ({e}); not counting toward idle")
                last_seen = time.monotonic()
                continue
            if served:
                if was_empty:
                    _log(f"GPU servers back up: {sorted(served)}")
                was_empty = False
                last_seen = time.monotonic()
                continue
            if not was_empty:
                _log(f"no GPU servers up — auto-shutdown in {int(empty_timeout)}s if none start")
                was_empty = True
            if (time.monotonic() - last_seen) >= empty_timeout:
                _log(f"no GPU servers for >= {int(empty_timeout)}s — shutting down to free the small allocation")
                signal.raise_signal(signal.SIGTERM)   # web.run_app exits gracefully
                return


async def create_app(args) -> web.Application:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    discovery = BackendDiscovery(
        user=user, backend_port=args.backend_port, ttl=args.discovery_ttl)
    batch_config = BatchConfig(
        max_tokens=args.batch_tokens,
        max_chars=args.batch_chars,
        max_delay_ms=args.batch_delay_ms,
        flush_on_newline=not args.no_flush_newline,
    )
    token = args.token or os.environ.get("OLIVIA_PROXY_TOKEN") or None
    router = Router(discovery, batch_config, token=token)

    async def on_startup(app):
        await router.start()
        served = await discovery.served_map(router.session, force=True)
        _log(f"router up; user={user!r}; serving: {sorted(served) or '(none yet)'}")
        app["watchdog"] = asyncio.create_task(router.idle_watchdog(args.empty_timeout))

    async def on_cleanup(app):
        task = app.get("watchdog")
        if task:
            task.cancel()
        await router.stop()

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/chat/completions", router.handle_chat)
    app.router.add_post("/v1/completions", router.handle_completions)
    app.router.add_get("/v1/models", router.handle_models)
    app.router.add_route("GET", "/", router.handle_health)
    app.router.add_route("HEAD", "/", router.handle_health)
    app.router.add_route("GET", "/health", router.handle_health)
    app.router.add_route("*", "/{tail:.*}", router.handle_unknown)
    return app


def main():
    parser = argparse.ArgumentParser(description="Multi-model vLLM router for the small partition")
    parser.add_argument("--listen-host", default="0.0.0.0",
                        help="Bind interface (default 0.0.0.0 — reachable from the login node "
                             "over the internal network; compute nodes are not internet-facing)")
    parser.add_argument("--listen-port", type=int, default=8080,
                        help="Router listen port (default 8080)")
    parser.add_argument("--backend-port", type=int, default=8000,
                        help="Port each vLLM server listens on (default 8000)")
    parser.add_argument("--discovery-ttl", type=float, default=15.0,
                        help="Seconds to cache squeue+probe backend lookups (default 15)")
    parser.add_argument("--empty-timeout", type=float, default=1800.0,
                        help="Auto-shutdown after this many seconds with no GPU servers up "
                             "(default 1800 = 30 min; 0 disables)")
    parser.add_argument("--token", default=None,
                        help="Require this bearer token / x-api-key (default: OLIVIA_PROXY_TOKEN env, else open)")
    parser.add_argument("--batch-tokens", type=int, default=15)
    parser.add_argument("--batch-chars", type=int, default=100)
    parser.add_argument("--batch-delay-ms", type=int, default=150)
    parser.add_argument("--no-flush-newline", action="store_true")
    args = parser.parse_args()

    print("=" * 56)
    print("Olivia model router (small partition)")
    print("=" * 56)
    print(f"Listen:        {args.listen_host}:{args.listen_port}")
    print(f"Backends:      vLLM :{args.backend_port}  (discovered via squeue + /v1/models)")
    print(f"Discovery TTL: {args.discovery_ttl}s")
    print(f"Spindown:      {'after ' + str(int(args.empty_timeout)) + 's idle (no servers)' if args.empty_timeout > 0 else 'disabled'}")
    print(f"Auth:          {'token required' if (args.token or os.environ.get('OLIVIA_PROXY_TOKEN')) else 'open (internal network only)'}")
    print("=" * 56)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(create_app(args))
    web.run_app(app, host=args.listen_host, port=args.listen_port, print=None)


if __name__ == "__main__":
    main()
