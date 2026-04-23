#!/usr/bin/env python3
"""
Anthropic-compatible proxy for vLLM / GLM
=========================================

Translates Anthropic Messages API (``POST /v1/messages``) into OpenAI
``POST /v1/chat/completions`` against an upstream vLLM server (or the
``vllm_proxy`` batching proxy) and rewrites the response back into
Anthropic's block-oriented event model.

Primary use case: point Claude Code CLI at a local GLM-5.1 deployment.

Architecture (default — direct to vLLM via the tunnel)::

    Claude Code  ──/v1/messages──►  anthropic_proxy  ──/v1/chat/completions──►  vllm:8000
    (local)       (Anthropic)        (this script,                              (GLM-5.1,
                                      localhost:8002)                            via SSH tunnel)

Optionally chain through ``vllm_proxy`` for SSE batching by setting
``--upstream http://localhost:8001`` (requires ``ENABLE_PROXY=1`` on the
server when starting vLLM).

Usage::

    python anthropic_proxy.py --model cyankiwi/GLM-5.1-AWQ-4bit

    export ANTHROPIC_BASE_URL=http://localhost:8002
    export ANTHROPIC_AUTH_TOKEN=sk-any-nonempty-string
    claude

Scope notes (intentional omissions):

- ``cache_control`` blocks are dropped silently (GLM has no prompt cache).
- ``thinking``/extended-thinking request params are dropped (GLM's
  reasoning parser always emits ``reasoning_content`` when present).
- ``image`` content blocks are replaced with a ``[image omitted]`` text
  placeholder (GLM deployments here are text-only).
- All Anthropic model names map to the single configured upstream model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from typing import Any, Optional

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Error: aiohttp required. Install with: pip install aiohttp",
          file=sys.stderr)
    sys.exit(1)

log = logging.getLogger("anthropic_proxy")


# --------------------------------------------------------------------------- #
# ID helpers                                                                  #
# --------------------------------------------------------------------------- #

def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _finish_to_stop_reason(finish: Optional[str]) -> Optional[str]:
    if not finish:
        return None
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "stop_sequence": "stop_sequence",
    }.get(finish, "end_turn")


# --------------------------------------------------------------------------- #
# Request translation: Anthropic /v1/messages -> OpenAI /v1/chat/completions  #
# --------------------------------------------------------------------------- #

def _flatten_system(system: Any) -> str:
    """Anthropic's ``system`` field may be a string or a list of content blocks."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _tool_result_to_text(content: Any) -> str:
    """Collapse an Anthropic ``tool_result.content`` payload to a plain string.

    The payload can be: a raw string, a list of text blocks, or arbitrary JSON.
    OpenAI's ``tool`` message only accepts string content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            elif isinstance(blk, dict) and blk.get("type") == "image":
                parts.append("[image omitted]")
            else:
                parts.append(json.dumps(blk))
        return "\n".join(parts)
    return json.dumps(content)


def anthropic_to_openai(body: dict, model_override: str) -> dict:
    out: dict[str, Any] = {
        "model": model_override,
        "stream": bool(body.get("stream", False)),
    }

    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]

    messages: list[dict] = []

    sys_text = _flatten_system(body.get("system"))
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
                continue
            if not isinstance(content, list):
                continue
            # User blocks may interleave text + tool_result. OpenAI needs the
            # tool results as separate ``role: tool`` messages, so we flush
            # text first, then emit each tool result as its own message.
            text_parts: list[str] = []
            tool_msgs: list[dict] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type")
                if btype == "text":
                    text_parts.append(blk.get("text", ""))
                elif btype == "image":
                    text_parts.append("[image omitted]")
                elif btype == "tool_result":
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": blk.get("tool_use_id"),
                        "content": _tool_result_to_text(blk.get("content")),
                    })
            if text_parts:
                messages.append({
                    "role": "user",
                    "content": "\n".join(p for p in text_parts if p),
                })
            messages.extend(tool_msgs)

        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            if not isinstance(content, list):
                continue
            text_parts = []
            tool_calls = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type")
                if btype == "text":
                    text_parts.append(blk.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": blk.get("id") or _tool_id(),
                        "type": "function",
                        "function": {
                            "name": blk.get("name", ""),
                            "arguments": json.dumps(blk.get("input") or {}),
                        },
                    })
                # ``thinking`` blocks are dropped from replayed history —
                # GLM's reasoning is regenerated each turn and including
                # prior thinking just eats context.
            asst: dict[str, Any] = {"role": "assistant"}
            asst["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                asst["tool_calls"] = tool_calls
            messages.append(asst)

    out["messages"] = messages

    tools = body.get("tools")
    if tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema")
                                  or {"type": "object", "properties": {}},
                },
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]

    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        ttype = tc.get("type")
        if ttype == "auto":
            out["tool_choice"] = "auto"
        elif ttype == "any":
            out["tool_choice"] = "required"
        elif ttype == "none":
            out["tool_choice"] = "none"
        elif ttype == "tool" and tc.get("name"):
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }

    if out["stream"]:
        # Final chunk carries real usage counts so we can report
        # output_tokens in ``message_delta``.
        out["stream_options"] = {"include_usage": True}

    return out


# --------------------------------------------------------------------------- #
# Response translation: non-streaming                                         #
# --------------------------------------------------------------------------- #

def openai_to_anthropic(resp: dict, model_label: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content_blocks: list[dict] = []

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    text = message.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or _tool_id(),
            "name": fn.get("name") or "",
            "input": args,
        })

    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model_label,
        "content": content_blocks,
        "stop_reason": _finish_to_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# --------------------------------------------------------------------------- #
# Response translation: streaming                                             #
# --------------------------------------------------------------------------- #

class AnthropicStreamWriter:
    """Emit Anthropic SSE events from OpenAI-style streaming deltas.

    OpenAI streams flat deltas (``content`` / ``reasoning_content`` /
    ``tool_calls[i].function.arguments``); Anthropic streams block-open /
    delta / block-close events. This class tracks the currently-open block
    and transparently closes and reopens blocks when the delta category
    changes (e.g. thinking -> text, or one tool_call -> the next).
    """

    def __init__(self, response: web.StreamResponse, model_label: str):
        self.response = response
        self.model_label = model_label
        self.message_id = _msg_id()
        self.started = False
        self.closed = False
        # Set when the client hangs up mid-stream (socket reset while we're
        # writing). After this, every _emit becomes a no-op, and translate_stream
        # uses it to break out of the upstream-read loop early.
        self.client_gone = False
        self.current_block_index: Optional[int] = None
        self.current_block_type: Optional[str] = None  # "thinking" | "text" | "tool_use"
        self.current_tool_call_idx: Optional[int] = None
        self.next_index = 0
        self.stop_reason: Optional[str] = None
        self.completion_tokens = 0

    async def _emit(self, event_name: str, payload: dict) -> None:
        if self.client_gone:
            return
        frame = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
        try:
            await self.response.write(frame.encode())
        except (ConnectionError, aiohttp.ClientConnectionResetError):
            # Client closed the SSE connection before we finished streaming
            # (e.g. upstream stalled and the Anthropic SDK hit its timeout).
            # Nothing useful to send anymore; mark and suppress so the rest
            # of finalize() / the upstream loop can wind down cleanly.
            self.client_gone = True
            log.warning("client disconnected mid-stream; suppressing further emits")

    async def ensure_started(self) -> None:
        if self.started:
            return
        self.started = True
        # ``input_tokens`` at start is a placeholder (0) — the real count
        # only arrives in the upstream's final usage chunk, by which point
        # we've already committed to the Anthropic event order. We report
        # the accurate ``output_tokens`` in ``message_delta`` at the end.
        await self._emit("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model_label,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    async def _close_current(self) -> None:
        if self.current_block_index is None:
            return
        await self._emit("content_block_stop", {
            "type": "content_block_stop",
            "index": self.current_block_index,
        })
        self.current_block_index = None
        self.current_block_type = None
        self.current_tool_call_idx = None

    async def _open_block(self, block: dict, block_type: str,
                          tool_call_idx: Optional[int] = None) -> None:
        idx = self.next_index
        self.next_index += 1
        self.current_block_index = idx
        self.current_block_type = block_type
        self.current_tool_call_idx = tool_call_idx
        await self._emit("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": block,
        })

    async def on_text_delta(self, text: str) -> None:
        if not text:
            return
        await self.ensure_started()
        if self.current_block_type != "text":
            await self._close_current()
            await self._open_block({"type": "text", "text": ""}, "text")
        await self._emit("content_block_delta", {
            "type": "content_block_delta",
            "index": self.current_block_index,
            "delta": {"type": "text_delta", "text": text},
        })

    async def on_thinking_delta(self, text: str) -> None:
        if not text:
            return
        await self.ensure_started()
        if self.current_block_type != "thinking":
            await self._close_current()
            await self._open_block({"type": "thinking", "thinking": ""}, "thinking")
        await self._emit("content_block_delta", {
            "type": "content_block_delta",
            "index": self.current_block_index,
            "delta": {"type": "thinking_delta", "thinking": text},
        })

    async def on_tool_call_delta(
        self,
        tc_idx: int,
        tc_id: Optional[str],
        tc_name: Optional[str],
        args_fragment: Optional[str],
    ) -> None:
        await self.ensure_started()
        needs_new_block = (
            self.current_block_type != "tool_use"
            or self.current_tool_call_idx != tc_idx
        )
        if needs_new_block:
            await self._close_current()
            await self._open_block(
                {
                    "type": "tool_use",
                    "id": tc_id or _tool_id(),
                    "name": tc_name or "",
                    "input": {},
                },
                "tool_use",
                tool_call_idx=tc_idx,
            )
        if args_fragment:
            await self._emit("content_block_delta", {
                "type": "content_block_delta",
                "index": self.current_block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": args_fragment,
                },
            })

    async def on_finish(self, finish_reason: Optional[str]) -> None:
        stop = _finish_to_stop_reason(finish_reason)
        if stop:
            self.stop_reason = stop

    def on_usage(self, usage: dict) -> None:
        ct = usage.get("completion_tokens")
        if isinstance(ct, int):
            self.completion_tokens = ct

    async def finalize(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.ensure_started()
        await self._close_current()
        await self._emit("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": self.stop_reason or "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": self.completion_tokens},
        })
        await self._emit("message_stop", {"type": "message_stop"})


async def translate_stream(
    upstream: aiohttp.ClientResponse,
    response: web.StreamResponse,
    model_label: str,
) -> None:
    writer = AnthropicStreamWriter(response, model_label)
    pending = b""

    try:
        async for chunk in upstream.content.iter_any():
            if writer.client_gone:
                # Client hung up — stop reading upstream so the underlying
                # connection can be released by the ``async with`` in the
                # caller. Otherwise we'd drain the remainder into /dev/null.
                break
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                stripped = raw.strip()
                if not stripped or not stripped.startswith(b"data: "):
                    continue
                data_str = stripped[6:].decode("utf-8", errors="replace")
                if data_str == "[DONE]":
                    pending = b""
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if isinstance(usage, dict):
                    writer.on_usage(usage)

                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                # Reasoning is emitted by GLM's ``--reasoning-parser``
                # under either name depending on the vLLM version.
                for fld in ("reasoning_content", "reasoning"):
                    val = delta.get(fld)
                    if isinstance(val, str) and val:
                        await writer.on_thinking_delta(val)

                content = delta.get("content")
                if isinstance(content, str) and content:
                    await writer.on_text_delta(content)

                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    await writer.on_tool_call_delta(
                        tc.get("index", 0),
                        tc.get("id"),
                        fn.get("name"),
                        fn.get("arguments"),
                    )

                if choice.get("finish_reason"):
                    await writer.on_finish(choice["finish_reason"])
    finally:
        await writer.finalize()


# --------------------------------------------------------------------------- #
# aiohttp server                                                              #
# --------------------------------------------------------------------------- #

def _error_response(status: int, err_type: str, message: str) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status=status,
    )


class AnthropicProxy:
    def __init__(self, upstream_url: str, model: str,
                 keepalive_interval: int = 180):
        self.upstream_url = upstream_url.rstrip("/")
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None
        # Keepalive: 0 disables. Otherwise, every N seconds of idle, send a
        # trivial completion upstream to keep vLLM's Ray compiled-DAG warm.
        # Multi-node PP has an instability pattern where the engine wedges
        # on idle → active transitions; a lightweight ping avoids long idle.
        self.keepalive_interval = keepalive_interval
        self._keepalive_task: Optional[asyncio.Task] = None
        self._last_activity: float = 0.0

    def _mark_activity(self) -> None:
        self._last_activity = asyncio.get_event_loop().time()

    async def _keepalive_loop(self) -> None:
        log.info("keepalive: pinging upstream every %ds while idle",
                 self.keepalive_interval)
        while True:
            try:
                await asyncio.sleep(self.keepalive_interval)
                idle_for = asyncio.get_event_loop().time() - self._last_activity
                if idle_for < self.keepalive_interval:
                    # Real traffic is flowing; skip this tick.
                    continue
                try:
                    async with self.session.post(
                        f"{self.upstream_url}/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "temperature": 0,
                            "stream": False,
                        },
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        await resp.read()
                        log.debug("keepalive: upstream %d after %ds idle",
                                  resp.status, int(idle_for))
                except asyncio.TimeoutError:
                    log.warning("keepalive: upstream did not respond within 60s "
                                "(engine may be wedged — consider restart)")
                except aiohttp.ClientError as e:
                    log.warning("keepalive: upstream unreachable: %s", e)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("keepalive: unexpected error (continuing)")

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self._mark_activity()
        if self.keepalive_interval > 0:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        if self.session:
            await self.session.close()

    async def handle_messages(self, request: web.Request) -> web.StreamResponse:
        self._mark_activity()
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error_response(400, "invalid_request_error", "Invalid JSON body")

        openai_body = anthropic_to_openai(body, self.model)
        is_streaming = openai_body.get("stream", False)
        model_label = body.get("model") or self.model

        if not is_streaming:
            try:
                async with self.session.post(
                    f"{self.upstream_url}/v1/chat/completions",
                    json=openai_body,
                    headers={"Content-Type": "application/json"},
                ) as upstream:
                    if upstream.status != 200:
                        text = await upstream.text()
                        return _error_response(upstream.status, "api_error", text)
                    data = await upstream.json()
            except aiohttp.ClientError as e:
                return _error_response(502, "api_error", f"Upstream unreachable: {e}")
            return web.json_response(openai_to_anthropic(data, model_label))

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        async def _try_write_error(message: str) -> None:
            """Best-effort error-event emission. The client may have already
            hung up (if upstream stalled past its timeout), so swallow any
            reset instead of cascading into a 500."""
            err = {"type": "error",
                   "error": {"type": "api_error", "message": message}}
            try:
                await response.write(
                    f"event: error\ndata: {json.dumps(err)}\n\n".encode()
                )
            except (ConnectionError, aiohttp.ClientConnectionResetError):
                log.warning("client already gone; dropped error event: %s",
                            message[:120])

        try:
            async with self.session.post(
                f"{self.upstream_url}/v1/chat/completions",
                json=openai_body,
                headers={"Content-Type": "application/json"},
            ) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    await _try_write_error(text)
                    return response
                await translate_stream(upstream, response, model_label)
        except aiohttp.ClientError as e:
            await _try_write_error(str(e))
        return response

    async def handle_health(self, request: web.Request) -> web.Response:
        """Root health probe. Claude Code sends HEAD / on startup to verify the
        base URL is reachable; returning 200 keeps its logs quiet."""
        return web.Response(status=200, text="ok")

    async def handle_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "data": [{
                "type": "model",
                "id": self.model,
                "display_name": self.model,
            }],
            "has_more": False,
            "first_id": self.model,
            "last_id": self.model,
        })

    async def handle_unknown(self, request: web.Request) -> web.Response:
        """Catch-all. Logs the path so we can see what a client probes for."""
        body_preview = ""
        if request.can_read_body:
            try:
                raw = await request.read()
                body_preview = raw[:500].decode("utf-8", errors="replace")
            except Exception:
                body_preview = "<unreadable>"
        log.warning("unhandled %s %s (headers=%s) body=%r",
                    request.method, request.path_qs,
                    {k: v for k, v in request.headers.items()
                     if k.lower() in ("user-agent", "anthropic-version",
                                      "anthropic-beta", "content-type",
                                      "x-api-key", "authorization")},
                    body_preview)
        return _error_response(404, "not_found_error",
                               f"Proxy does not implement {request.method} {request.path}")


@web.middleware
async def _logging_middleware(request: web.Request, handler):
    """Log every request: method, path, status, and any exception."""
    try:
        response = await handler(request)
    except web.HTTPException as e:
        log.info("%s %s -> %d", request.method, request.path_qs, e.status)
        raise
    except Exception:
        log.exception("%s %s -> 500", request.method, request.path_qs)
        raise
    log.info("%s %s -> %d", request.method, request.path_qs,
             getattr(response, "status", 0))
    return response


def create_app(upstream_url: str, model: str,
               keepalive_interval: int = 180) -> web.Application:
    proxy = AnthropicProxy(upstream_url, model,
                           keepalive_interval=keepalive_interval)

    async def on_startup(app: web.Application) -> None:
        await proxy.start()
        log.info("Anthropic proxy forwarding to %s (model=%s)",
                 proxy.upstream_url, proxy.model)

    async def on_cleanup(app: web.Application) -> None:
        await proxy.stop()

    app = web.Application(middlewares=[_logging_middleware])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_post("/v1/messages", proxy.handle_messages)
    app.router.add_get("/v1/models", proxy.handle_models)
    app.router.add_route("HEAD", "/", proxy.handle_health)
    app.router.add_route("GET", "/", proxy.handle_health)
    # Catch-all so any unknown Claude Code probe shows up in the log
    # instead of silently 404'ing with no hint about what was requested.
    app.router.add_route("*", "/{tail:.*}", proxy.handle_unknown)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anthropic-compatible proxy for vLLM / GLM (Claude Code bridge)",
    )
    parser.add_argument("--listen-host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--listen-port", type=int, default=8002,
                        help="Proxy listen port (default: 8002)")
    parser.add_argument("--upstream", default="http://localhost:8000",
                        help="Upstream OpenAI-compatible URL. Default is vLLM "
                             "directly (8000). Set to http://localhost:8001 to "
                             "chain through vllm_proxy, which requires "
                             "ENABLE_PROXY=1 on the server side.")
    parser.add_argument("--model", required=True,
                        help="Model name to send upstream "
                             "(e.g. cyankiwi/GLM-5.1-AWQ-4bit)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Log request/response bodies (noisy, but useful for debugging Claude Code handshakes)")
    parser.add_argument("--keepalive-interval", type=int, default=180,
                        help="Seconds of upstream idle before sending a "
                             "trivial completion to keep the Ray compiled-DAG "
                             "warm on multi-node PP (default: 180). "
                             "Set to 0 to disable.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Anthropic proxy -> OpenAI / GLM bridge")
    print("=" * 60)
    print(f"Listening on   : {args.listen_host}:{args.listen_port}")
    print(f"Upstream       : {args.upstream}")
    print(f"Model override : {args.model}")
    print(f"Keepalive      : {args.keepalive_interval}s"
          if args.keepalive_interval > 0 else "Keepalive      : disabled")
    print()
    print("Point Claude Code at this proxy:")
    print(f"  export ANTHROPIC_BASE_URL=http://{args.listen_host}:{args.listen_port}")
    print("  export ANTHROPIC_AUTH_TOKEN=sk-any-nonempty-string")
    print("  claude")
    print("=" * 60)

    app = create_app(args.upstream, args.model,
                     keepalive_interval=args.keepalive_interval)
    web.run_app(app, host=args.listen_host, port=args.listen_port, print=None)


if __name__ == "__main__":
    main()
