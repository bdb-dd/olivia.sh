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
    """Token batching configuration."""
    max_tokens: int = 15        # Flush after N tokens
    max_chars: int = 100        # Flush after N characters
    max_delay_ms: int = 150     # Flush after N milliseconds
    flush_on_newline: bool = True  # Flush when content contains newline


@dataclass
class TokenBuffer:
    """Buffer for accumulating tokens before sending."""
    tokens: list = field(default_factory=list)
    content: str = ""
    first_token_time: Optional[float] = None
    usage: Optional[dict] = None

    def add(self, token_content: str, usage: Optional[dict] = None):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.tokens.append(token_content)
        self.content += token_content
        if usage:
            self.usage = usage

    def should_flush(self, config: BatchConfig) -> bool:
        if not self.tokens:
            return False

        # Check token count
        if len(self.tokens) >= config.max_tokens:
            return True

        # Check character count
        if len(self.content) >= config.max_chars:
            return True

        # Check time elapsed
        if self.first_token_time:
            elapsed_ms = (time.time() - self.first_token_time) * 1000
            if elapsed_ms >= config.max_delay_ms:
                return True

        # Check for newlines
        if config.flush_on_newline and '\n' in self.content:
            return True

        return False

    def flush(self) -> tuple[str, Optional[dict]]:
        """Return accumulated content and reset buffer."""
        content = self.content
        usage = self.usage
        self.tokens = []
        self.content = ""
        self.first_token_time = None
        self.usage = None
        return content, usage


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

        # Streaming: batch tokens before forwarding
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

        buffer = TokenBuffer()
        chunk_id = None
        model = None

        async def flush_buffer():
            """Send batched tokens as single SSE event."""
            nonlocal buffer
            if not buffer.content:
                return

            content, usage = buffer.flush()

            # Construct batched SSE event (same format as vLLM)
            event_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }]
            }
            if usage:
                event_data["usage"] = usage

            sse_line = f"data: {json.dumps(event_data)}\n\n"
            await response.write(sse_line.encode())

        try:
            async with self.session.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            ) as upstream:
                async for line in upstream.content:
                    line = line.decode('utf-8').strip()

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        # Flush any remaining tokens
                        await flush_buffer()
                        await response.write(b"data: [DONE]\n\n")
                        break

                    try:
                        data = json.loads(data_str)

                        # Capture metadata from first chunk
                        if chunk_id is None:
                            chunk_id = data.get("id")
                            model = data.get("model")

                        # Extract token content
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            # Check for finish
                            finish_reason = choices[0].get("finish_reason")
                            if finish_reason:
                                await flush_buffer()
                                # Forward the finish event as-is
                                await response.write(f"data: {data_str}\n\n".encode())
                                continue

                            if content:
                                usage = data.get("usage")
                                buffer.add(content, usage)

                                if buffer.should_flush(self.batch_config):
                                    await flush_buffer()

                    except json.JSONDecodeError:
                        # Forward malformed data as-is
                        await response.write(f"{line}\n\n".encode())

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

    web.run_app(app, host="0.0.0.0", port=args.proxy_port, print=None)


if __name__ == "__main__":
    main()
