#!/usr/bin/env python3
"""
Interactive Chat Client for vLLM Devstral Server (Rich Edition)
================================================================

Features:
- Beautiful terminal UI with rich formatting
- Interactive multi-turn conversation
- Token usage and generation speed metrics
- Streaming with live display
- Markdown rendering for responses

Usage:
    python chat_devstral_rich.py <hostname> [--port 8000] [--stream]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional, Tuple, Dict, Any

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better formatting: pip install rich")

console = Console() if RICH_AVAILABLE else None


class DevstralChat:
    def __init__(self, host: str, port: int = 8000, stream: bool = True):
        self.base_url = f"http://{host}:{port}"
        self.stream = stream
        self.conversation_history: list = []
        self.model: Optional[str] = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.turn_count = 0
        
    def check_server(self) -> bool:
        """Check if server is healthy and get model info."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code != 200:
                return False
            
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model = models[0]["id"]
            return True
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False
    
    def chat(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Send a message and get a response with metrics."""
        self.turn_count += 1
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        payload = {
            "model": self.model or "default",
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": self.stream
        }
        
        start_time = time.time()
        first_token_time = None
        response_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            if self.stream:
                response_content, metrics = self._stream_response(payload, start_time)
                first_token_time = metrics.get("first_token_time")
                prompt_tokens = metrics.get("prompt_tokens", 0)
                completion_tokens = metrics.get("completion_tokens", 0)
            else:
                # Show spinner while waiting
                if RICH_AVAILABLE:
                    with console.status("[bold green]Thinking...", spinner="dots"):
                        resp = requests.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                            timeout=300
                        )
                else:
                    print("Thinking...", end="", flush=True)
                    resp = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        timeout=300
                    )
                    print("\r", end="")
                
                if resp.status_code != 200:
                    self.conversation_history.pop()
                    return None
                
                data = resp.json()
                response_content = data["choices"][0]["message"]["content"]
                
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
        
        except requests.exceptions.Timeout:
            self.conversation_history.pop()
            return None
        except Exception as e:
            self.conversation_history.pop()
            return None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        
        return {
            "turn": self.turn_count,
            "response": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time_sec": round(total_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "time_to_first_token": round(first_token_time, 2) if first_token_time else None
        }
    
    def _stream_response(self, payload: dict, start_time: float) -> Tuple[str, dict]:
        """Handle streaming response with token batching for improved throughput."""
        response_content = ""
        first_token_time = None
        prompt_tokens = 0
        completion_tokens = 0

        # Token batching settings - reduces I/O overhead over SSH tunnel
        BATCH_SIZE = 10  # Number of tokens to batch before printing
        BATCH_CHARS = 50  # Or flush when buffer exceeds this many chars
        BATCH_TIMEOUT = 0.1  # Max seconds to hold tokens before flushing

        token_buffer = ""
        last_flush_time = time.time()
        tokens_in_buffer = 0

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300
        )

        if resp.status_code != 200:
            return "", {}

        if RICH_AVAILABLE:
            console.print("\n[bold blue]Assistant:[/bold blue] ", end="")
        else:
            print("\nAssistant: ", end="", flush=True)

        def flush_buffer():
            nonlocal token_buffer, last_flush_time, tokens_in_buffer
            if token_buffer:
                print(token_buffer, end="", flush=True)
                token_buffer = ""
                tokens_in_buffer = 0
                last_flush_time = time.time()

        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        flush_buffer()  # Flush any remaining tokens
                        break
                    try:
                        data = json.loads(data_str)

                        if first_token_time is None and data.get("choices"):
                            first_token_time = time.time() - start_time

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            token_buffer += content
                            tokens_in_buffer += 1
                            response_content += content

                            # Flush buffer if: enough tokens, enough chars, or timeout
                            current_time = time.time()
                            should_flush = (
                                tokens_in_buffer >= BATCH_SIZE or
                                len(token_buffer) >= BATCH_CHARS or
                                (current_time - last_flush_time) >= BATCH_TIMEOUT or
                                '\n' in content  # Always flush on newlines for readability
                            )

                            if should_flush:
                                flush_buffer()

                        usage = data.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        pass

        # Final flush in case anything remains
        flush_buffer()
        print()

        if completion_tokens == 0:
            completion_tokens = int(len(response_content.split()) * 1.3)

        return response_content, {
            "first_token_time": first_token_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.turn_count = 0

    def _bench_one(self, prompt: str, max_tokens: int) -> Optional[Dict[str, Any]]:
        """Single silent streaming request — TTFT + raw token counts. No history."""
        payload = {
            "model": self.model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.time()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload, stream=True, timeout=600,
            )
            if resp.status_code != 200:
                return None
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if first_token_time is None and data.get("choices"):
                    delta = data["choices"][0].get("delta", {})
                    # GLM-5.1 with --reasoning-parser glm45 emits reasoning tokens via
                    # `delta.reasoning`; non-reasoning models use `delta.content`.
                    # Either counts as the first generated token for TTFT purposes.
                    if delta.get("content") or delta.get("reasoning"):
                        first_token_time = time.time() - start
                usage = data.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        except Exception:
            return None
        total = time.time() - start
        return {
            "ttft": first_token_time or 0.0,
            "total": total,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def benchmark(self, prompt: str, n: int, max_tokens: int, warmup: int = 1) -> int:
        """Run prompt N times, report per-run and aggregate metrics. Returns exit code."""
        if not self.model:
            print(f"Error: not connected to {self.base_url}")
            return 1
        print(f"Server: {self.base_url}")
        print(f"Model:  {self.model}")
        print(f"Prompt ({len(prompt)} chars): {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Runs:   {n}  (+ {warmup} warmup)   max_tokens={max_tokens}")
        print()
        # Warmup — captures prefix-cache and JIT warm states without affecting averages
        for i in range(warmup):
            r = self._bench_one(prompt, max_tokens)
            if r is None:
                print(f"warmup {i+1}: FAILED")
                return 1
            print(f"warmup {i+1}: ttft={r['ttft']*1000:.0f}ms  decode={(r['completion_tokens']-1)/max(r['total']-r['ttft'], 1e-9):.2f} tok/s  ({r['completion_tokens']} out)")
        print()
        header = f"{'run':>4}  {'prompt':>7}  {'output':>7}  {'TTFT':>9}  {'decode tok/s':>13}  {'total':>9}"
        print(header)
        print("-" * len(header))
        results = []
        for i in range(1, n + 1):
            r = self._bench_one(prompt, max_tokens)
            if r is None:
                print(f"{i:>4}  FAILED")
                continue
            decode_toks = (r["completion_tokens"] - 1) / max(r["total"] - r["ttft"], 1e-9)
            print(f"{i:>4}  {r['prompt_tokens']:>7}  {r['completion_tokens']:>7}  {r['ttft']*1000:>7.0f}ms  {decode_toks:>11.2f}    {r['total']:>7.2f}s")
            results.append((r, decode_toks))
        if not results:
            return 1
        n_ok = len(results)
        avg_ttft = sum(r["ttft"] for r, _ in results) / n_ok
        avg_decode = sum(d for _, d in results) / n_ok
        avg_total_toks_per_s = sum(r["completion_tokens"] / r["total"] for r, _ in results if r["total"] > 0) / n_ok
        avg_out = sum(r["completion_tokens"] for r, _ in results) / n_ok
        print("-" * len(header))
        print(f"avg over {n_ok} run(s):")
        print(f"  TTFT:               {avg_ttft*1000:.0f}ms")
        print(f"  decode (post-TTFT): {avg_decode:.2f} tok/s")
        print(f"  end-to-end:         {avg_total_toks_per_s:.2f} tok/s  ({avg_out:.0f} out tokens avg)")
        return 0
    
    def get_stats_table(self) -> Table:
        """Get stats as a rich table."""
        table = Table(title="Session Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total turns", str(self.turn_count))
        table.add_row("Prompt tokens", f"{self.total_prompt_tokens:,}")
        table.add_row("Completion tokens", f"{self.total_completion_tokens:,}")
        table.add_row("Total tokens", f"{self.total_prompt_tokens + self.total_completion_tokens:,}")
        table.add_row("Messages in history", str(len(self.conversation_history)))
        
        return table


def print_response(response: str):
    """Print response with optional markdown rendering."""
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold blue]Assistant[/bold blue]",
            border_style="blue"
        ))
    else:
        print(f"\nAssistant: {response}")


def print_metrics(metrics: dict):
    """Print turn metrics."""
    if RICH_AVAILABLE:
        text = Text()
        text.append(f"Turn {metrics['turn']}", style="bold")
        text.append(" │ ")
        text.append(f"Prompt: {metrics['prompt_tokens']}", style="cyan")
        text.append(" │ ")
        text.append(f"Response: {metrics['completion_tokens']}", style="green")
        text.append(" │ ")
        text.append(f"Time: {metrics['total_time_sec']}s", style="yellow")
        text.append(" │ ")
        text.append(f"Speed: {metrics['tokens_per_second']} tok/s", style="magenta")
        if metrics.get('time_to_first_token'):
            text.append(" │ ")
            text.append(f"TTFT: {metrics['time_to_first_token']}s", style="blue")
        console.print(text)
    else:
        print(f"[Turn {metrics['turn']}] "
              f"Prompt: {metrics['prompt_tokens']} | "
              f"Response: {metrics['completion_tokens']} | "
              f"Time: {metrics['total_time_sec']}s | "
              f"Speed: {metrics['tokens_per_second']} tok/s")


def print_help():
    """Print available commands."""
    if RICH_AVAILABLE:
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        
        table.add_row("/help", "Show this help message")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/stats", "Show session statistics")
        table.add_row("/history", "Show conversation history")
        table.add_row("/stream", "Toggle streaming mode")
        table.add_row("/quit", "Exit the chat")
        
        console.print(table)
    else:
        print("""
Commands:
  /help     - Show this help message
  /clear    - Clear conversation history
  /stats    - Show session statistics
  /history  - Show conversation history
  /stream   - Toggle streaming mode
  /quit     - Exit the chat
""")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Devstral via vLLM")
    parser.add_argument("host", help="vLLM server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses")
    parser.add_argument("--bench", metavar="PROMPT", help="Run a non-interactive benchmark with PROMPT and exit")
    parser.add_argument("-n", "--runs", type=int, default=3, help="Bench: number of measured runs (default 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Bench: max output tokens per run (default 256)")
    parser.add_argument("--warmup", type=int, default=1, help="Bench: warmup runs excluded from averages (default 1)")
    args = parser.parse_args()

    # Benchmark mode: non-interactive, exit when done.
    if args.bench:
        chat = DevstralChat(args.host, args.port, stream=True)
        if not chat.check_server():
            print(f"Error: Cannot connect to server at {args.host}:{args.port}")
            sys.exit(1)
        sys.exit(chat.benchmark(args.bench, args.runs, args.max_tokens, args.warmup))
    
    # Probe the server first so the banner shows the model we're actually
    # talking to (this client is used across multiple vLLM deployments —
    # Devstral, GLM-4.7, GLM-5.1, etc. — so a hardcoded "Devstral" title was
    # misleading).
    chat = DevstralChat(args.host, args.port, args.stream)
    connected = chat.check_server()

    if RICH_AVAILABLE:
        if connected:
            console.print(Panel.fit(
                "[bold cyan]vLLM Chat Client[/bold cyan]\n"
                f"Model:  [green]{chat.model}[/green]\n"
                f"Server: [green]{args.host}:{args.port}[/green]",
                border_style="cyan"
            ))
        else:
            console.print(Panel.fit(
                "[bold cyan]vLLM Chat Client[/bold cyan]\n"
                f"Server: [green]{args.host}:{args.port}[/green]",
                border_style="cyan"
            ))
    else:
        print("=" * 60)
        print("  vLLM Chat Client")
        print("=" * 60)
        if connected:
            print(f"Model:  {chat.model}")
        print(f"Server: {args.host}:{args.port}")

    if not connected:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] Cannot connect to server at {args.host}:{args.port}")
        else:
            print(f"Error: Cannot connect to server at {args.host}:{args.port}")
        sys.exit(1)

    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] Streaming: [cyan]{'enabled' if chat.stream else 'disabled'}[/cyan]")
        console.print("\nType [cyan]/help[/cyan] for commands, [cyan]/quit[/cyan] to exit\n")
    else:
        print(f"Streaming: {'enabled' if chat.stream else 'disabled'}")
        print(f"Streaming: {'enabled' if chat.stream else 'disabled'}")
        print("\nType /help for commands, /quit to exit\n")
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
            else:
                user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/quit", "/exit", "/q"):
                    if RICH_AVAILABLE:
                        console.print(chat.get_stats_table())
                        console.print("[yellow]Goodbye![/yellow]")
                    else:
                        print("Goodbye!")
                    break
                elif cmd == "/help":
                    print_help()
                elif cmd == "/clear":
                    chat.reset_conversation()
                    if RICH_AVAILABLE:
                        console.print("[yellow]Conversation cleared.[/yellow]")
                    else:
                        print("Conversation cleared.")
                elif cmd == "/stats":
                    if RICH_AVAILABLE:
                        console.print(chat.get_stats_table())
                    else:
                        print(f"\nStats: {chat.turn_count} turns, "
                              f"{chat.total_prompt_tokens + chat.total_completion_tokens} total tokens")
                elif cmd == "/history":
                    if RICH_AVAILABLE:
                        for i, msg in enumerate(chat.conversation_history):
                            role = msg["role"].capitalize()
                            style = "green" if role == "User" else "blue"
                            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                            console.print(f"[{style}]{i+1}. [{role}][/{style}]: {content}")
                    else:
                        for i, msg in enumerate(chat.conversation_history):
                            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                            print(f"{i+1}. [{msg['role']}]: {content}")
                elif cmd == "/stream":
                    chat.stream = not chat.stream
                    if RICH_AVAILABLE:
                        console.print(f"Streaming: [cyan]{'enabled' if chat.stream else 'disabled'}[/cyan]")
                    else:
                        print(f"Streaming: {'enabled' if chat.stream else 'disabled'}")
                else:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Unknown command:[/red] {user_input}")
                    else:
                        print(f"Unknown command: {user_input}")
                continue
            
            # Regular chat
            metrics = chat.chat(user_input)
            
            if metrics:
                if not chat.stream:
                    print_response(metrics["response"])
                print_metrics(metrics)
            
        except KeyboardInterrupt:
            print()
            if RICH_AVAILABLE:
                console.print(chat.get_stats_table())
                console.print("[yellow]Interrupted. Goodbye![/yellow]")
            else:
                print("Goodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()

