#!/usr/bin/env python3
"""
vLLM Batching Proxy
===================

A thin proxy layer that batches SSE tokens from vLLM before forwarding to clients.
This reduces network overhead when accessing vLLM through high-latency connections (SSH tunnels).

Architecture:
    Client <--[batched SSE]--> Proxy <--[per-token SSE]--> vLLM
                (slow tunnel)           (localhost, fast)

Usage:
    python vllm_proxy.py --vllm-port 8000 --proxy-port 8001

    Then connect your client to port 8001 instead of 8000.
"""

import argparse
import asyncio
import json
import time
from typing import Optional
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web


@dataclass
class BatchConfig:
    """SSE event batching configuration."""
    max_tokens: int = 15        # Flush after N events (one per delta chunk)
    max_chars: int = 100        # Flush after N bytes accumulated
    max_delay_ms: int = 150     # Flush after N milliseconds
    flush_on_newline: bool = True  # Flush when any buffered chunk contains newline


@dataclass
class EventBuffer:
    """Buffer for accumulating raw SSE event lines before sending.

    Format-agnostic: works equally for `delta.content` (vanilla models),
    `delta.reasoning` (GLM-4.5/4.7/5.1 with --reasoning-parser), `delta.tool_calls`,
    or any other delta field. The proxy used to extract `delta.content` and
    rebuild a single batched event from it — that approach silently dropped any
    chunk whose delta wasn't `content`, which broke completely on GLM-style
    reasoning streams (no content tokens emitted to client).
    """
    events: list = field(default_factory=list)         # raw SSE lines (bytes)
    bytes_total: int = 0
    saw_newline: bool = False
    first_event_time: Optional[float] = None

    def add(self, sse_line: bytes, content_text: str = ""):
        if self.first_event_time is None:
            self.first_event_time = time.time()
        self.events.append(sse_line)
        self.bytes_total += len(sse_line)
        if content_text and "\n" in content_text:
            self.saw_newline = True

    def should_flush(self, config: BatchConfig) -> bool:
        if not self.events:
            return False
        if len(self.events) >= config.max_tokens:
            return True
        if self.bytes_total >= config.max_chars:
            return True
        if self.first_event_time:
            elapsed_ms = (time.time() - self.first_event_time) * 1000
            if elapsed_ms >= config.max_delay_ms:
                return True
        if config.flush_on_newline and self.saw_newline:
            return True
        return False

    def flush(self) -> bytes:
        """Return concatenated SSE bytes and reset buffer."""
        out = b"".join(self.events)
        self.events = []
        self.bytes_total = 0
        self.saw_newline = False
        self.first_event_time = None
        return out


class VLLMProxy:
    """Proxy server that batches vLLM SSE tokens."""

    def __init__(self, vllm_host: str, vllm_port: int, batch_config: BatchConfig):
        self.vllm_url = f"http://{vllm_host}:{vllm_port}"
        self.batch_config = batch_config
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    async def proxy_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Proxy /v1/chat/completions with token batching for streaming requests."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        is_streaming = body.get("stream", False)

        if not is_streaming:
            # Non-streaming: pass through directly
            async with self.session.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            ) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)

        # Streaming: batch raw SSE bytes before forwarding. We do NOT inspect
        # or rewrite delta payloads — that lets the proxy work for any delta
        # field (content, reasoning, tool_calls, ...) without per-format code.
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        await response.prepare(request)

        buffer = EventBuffer()

        async def flush_buffer():
            if buffer.events:
                await response.write(buffer.flush())

        try:
            async with self.session.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            ) as upstream:
                # iter_chunked instead of line-by-line: reading raw bytes lets
                # us forward exact upstream framing, and lines are
                # newline-terminated SSE frames so we still split correctly.
                pending = b""
                async for chunk in upstream.content.iter_any():
                    pending += chunk
                    # Split into complete SSE lines (keep trailing partial line
                    # in `pending` for the next iteration).
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        # Re-attach the newline so downstream parsers see
                        # framing exactly as vLLM emits it.
                        line_with_nl = line + b"\n"

                        # Inspect the line just enough to (a) detect [DONE],
                        # (b) detect finish events that should flush+forward
                        # immediately, (c) sniff content/reasoning text for
                        # newline-flush heuristic. No payload rewriting.
                        stripped = line.strip()
                        if not stripped or not stripped.startswith(b"data: "):
                            # Forward blank lines / comments as-is (preserves
                            # SSE inter-event spacing).
                            buffer.add(line_with_nl)
                            continue

                        data_str = stripped[6:].decode("utf-8", errors="replace")

                        if data_str == "[DONE]":
                            await flush_buffer()
                            await response.write(line_with_nl)
                            # Drain any remaining bytes (usually empty) and
                            # exit the outer chunk loop on next iteration.
                            if pending.strip():
                                await response.write(pending)
                            pending = b""
                            break

                        # For known JSON events: peek to extract finish flag
                        # and content text; if parse fails, just batch as-is.
                        text_for_newline_check = ""
                        is_finish_event = False
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta") or {}
                                # Sample any text-bearing delta field for the
                                # newline-flush heuristic.
                                for fld in ("content", "reasoning", "reasoning_content"):
                                    v = delta.get(fld)
                                    if isinstance(v, str):
                                        text_for_newline_check += v
                                if choices[0].get("finish_reason"):
                                    is_finish_event = True
                        except json.JSONDecodeError:
                            pass

                        buffer.add(line_with_nl, text_for_newline_check)

                        if is_finish_event:
                            # Flush immediately so the client sees finish_reason
                            # without waiting for the timer.
                            await flush_buffer()
                        elif buffer.should_flush(self.batch_config):
                            await flush_buffer()
                    else:
                        # No newline yet; check if a time-based flush should
                        # fire so we don't sit on partially-batched data while
                        # the upstream is mid-chunk.
                        if buffer.should_flush(self.batch_config):
                            await flush_buffer()
                # Final drain in case the upstream ended without [DONE].
                if pending:
                    buffer.add(pending)
                await flush_buffer()

        except Exception as e:
            error_event = {"error": str(e)}
            await response.write(f"data: {json.dumps(error_event)}\n\n".encode())

        return response

    async def proxy_passthrough(self, request: web.Request) -> web.Response:
        """Pass through non-streaming endpoints directly."""
        path = request.path
        method = request.method

        # Build upstream URL
        url = f"{self.vllm_url}{path}"
        if request.query_string:
            url += f"?{request.query_string}"

        # Forward request
        kwargs = {"headers": dict(request.headers)}
        if method in ("POST", "PUT", "PATCH"):
            kwargs["data"] = await request.read()

        async with self.session.request(method, url, **kwargs) as resp:
            body = await resp.read()
            return web.Response(
                body=body,
                status=resp.status,
                headers={k: v for k, v in resp.headers.items()
                        if k.lower() not in ('transfer-encoding', 'content-encoding')}
            )


async def create_app(vllm_host: str, vllm_port: int, batch_config: BatchConfig) -> web.Application:
    """Create the proxy application."""
    proxy = VLLMProxy(vllm_host, vllm_port, batch_config)

    async def on_startup(app):
        await proxy.start()
        print(f"Proxy connected to vLLM at {proxy.vllm_url}")

    async def on_cleanup(app):
        await proxy.stop()

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Route chat completions through batching proxy
    app.router.add_post("/v1/chat/completions", proxy.proxy_chat_completions)

    # Pass through all other endpoints
    app.router.add_route("*", "/{path:.*}", proxy.proxy_passthrough)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="vLLM batching proxy for high-latency connections"
    )
    parser.add_argument(
        "--listen-host", default="127.0.0.1",
        help="Host/interface for the proxy to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--vllm-host", default="localhost",
        help="vLLM server host (default: localhost)"
    )
    parser.add_argument(
        "--vllm-port", type=int, default=8000,
        help="vLLM server port (default: 8000)"
    )
    parser.add_argument(
        "--proxy-port", type=int, default=8001,
        help="Proxy listen port (default: 8001)"
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=15,
        help="Flush after N tokens (default: 15)"
    )
    parser.add_argument(
        "--batch-chars", type=int, default=100,
        help="Flush after N characters (default: 100)"
    )
    parser.add_argument(
        "--batch-delay-ms", type=int, default=150,
        help="Flush after N milliseconds (default: 150)"
    )
    parser.add_argument(
        "--no-flush-newline", action="store_true",
        help="Don't flush on newlines"
    )
    args = parser.parse_args()

    batch_config = BatchConfig(
        max_tokens=args.batch_tokens,
        max_chars=args.batch_chars,
        max_delay_ms=args.batch_delay_ms,
        flush_on_newline=not args.no_flush_newline
    )

    print("=" * 50)
    print("vLLM Batching Proxy")
    print("=" * 50)
    print(f"Upstream vLLM:  {args.vllm_host}:{args.vllm_port}")
    print(f"Proxy port:     {args.proxy_port}")
    print(f"Batch config:")
    print(f"  Max tokens:   {batch_config.max_tokens}")
    print(f"  Max chars:    {batch_config.max_chars}")
    print(f"  Max delay:    {batch_config.max_delay_ms}ms")
    print(f"  Flush newline: {batch_config.flush_on_newline}")
    print("=" * 50)

    app = asyncio.get_event_loop().run_until_complete(
        create_app(args.vllm_host, args.vllm_port, batch_config)
    )

    web.run_app(app, host=args.listen_host, port=args.proxy_port, print=None)


if __name__ == "__main__":
    main()
