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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from chat_storage import ConversationStore, Summarizer, default_db_path

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
        self.host_label = f"{host}:{port}"
        self.stream = stream
        self.conversation_history: list = []
        self.model: Optional[str] = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.turn_count = 0
        # Storage is wired in by attach_storage() after check_server() so we
        # know the model name; left as None when --no-store is set or the chat
        # is invoked from --bench mode.
        self.store: Optional[ConversationStore] = None
        self.summarizer: Optional[Summarizer] = None
        self.conversation_id: Optional[int] = None

    def attach_storage(self, db_path: Path) -> None:
        """Open the conversation DB and start the background summarizer.

        Safe to call repeatedly; subsequent calls are no-ops.
        """
        if self.store is not None:
            return
        self.store = ConversationStore(db_path)
        if self.model:
            self.summarizer = Summarizer(self.store, self.base_url, self.model)


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
            "stream": self.stream,
            "stream_options": {"include_usage": True},
        }
        
        start_time = time.time()
        first_token_time = None
        response_content = ""
        reasoning_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            if self.stream:
                response_content, reasoning_text, metrics = self._stream_response(payload, start_time)
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
                msg = data["choices"][0]["message"]
                response_content = msg.get("content") or ""
                # Some vLLM builds expose reasoning on non-stream responses too
                # (e.g. GLM-5.1 with --reasoning-parser glm45). If absent, this
                # is just the empty string.
                reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""

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

        # Only the visible answer goes back into conversation_history — the
        # model doesn't expect its prior chain-of-thought as input on the next
        # turn. Reasoning is persisted separately for searchability only.
        self.conversation_history.append({
            "role": "assistant",
            "content": response_content
        })

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

        # Persist + enqueue background summarization. Failures here must not
        # break the chat — log and move on.
        if self.store is not None:
            try:
                if self.conversation_id is None:
                    self.conversation_id = self.store.start_conversation(
                        self.model, self.host_label
                    )
                self.store.append_turn(
                    conversation_id=self.conversation_id,
                    turn=self.turn_count,
                    user_message=user_message,
                    assistant_content=response_content,
                    assistant_reasoning=reasoning_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                if self.summarizer is not None:
                    if self.turn_count == 1:
                        self.summarizer.enqueue_title(
                            self.conversation_id, user_message, response_content
                        )
                    if self.turn_count >= 5 and self.turn_count % 5 == 0:
                        # Summary worker pulls fresh state from the DB at job
                        # execution time, so successive enqueues chain
                        # correctly even if a prior job is still in flight.
                        self.summarizer.enqueue_summary(
                            self.conversation_id, self.turn_count
                        )
            except Exception as e:
                sys.stderr.write(f"[storage] persist failed: {e}\n")

        return {
            "turn": self.turn_count,
            "response": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time_sec": round(total_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "time_to_first_token": round(first_token_time, 2) if first_token_time else None
        }

    def _stream_response(self, payload: dict, start_time: float) -> Tuple[str, str, dict]:
        """Handle streaming response with token batching for improved throughput.

        Returns (visible_content, reasoning_text, metrics). The two text
        buffers are kept separate so reasoning can be persisted in its own
        searchable column without being replayed back to the model.
        """
        import queue
        import threading

        content_text = ""
        reasoning_text = ""
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
            return "", "", {}

        if RICH_AVAILABLE:
            console.print("\n[bold blue]Assistant:[/bold blue] ", end="")
        else:
            print("\nAssistant: ", end="", flush=True)

        # Decouple stdout from the SSE receive loop. Without this, every
        # `print(..., flush=True)` blocks the reader until the terminal has
        # flushed; on slow terminals (or terminals with heavy scrollback),
        # this back-pressures the upstream socket and roughly halves observed
        # tok/s vs. a silent receiver. The writer thread drains a queue and
        # does the actual I/O; the receiver only enqueues bytes.
        write_q: queue.Queue = queue.Queue()
        WRITER_DONE = object()

        def _writer():
            while True:
                item = write_q.get()
                if item is WRITER_DONE:
                    return
                sys.stdout.write(item)
                sys.stdout.flush()

        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()

        def flush_buffer():
            nonlocal token_buffer, last_flush_time, tokens_in_buffer
            if token_buffer:
                write_q.put(token_buffer)
                token_buffer = ""
                tokens_in_buffer = 0
                last_flush_time = time.time()

        # Track whether we're currently inside a "reasoning" run so we can
        # render it visibly distinct from the final answer (GLM-5.1 with
        # --reasoning-parser glm45 streams its chain-of-thought as
        # `delta.reasoning`; without explicit handling here, reasoning was
        # silently dropped and the client appeared to hang during long
        # chain-of-thought even though the server was generating fine).
        in_reasoning = False
        REASONING_OPEN = "\033[2;3m<thinking>\n"   # dim italic
        REASONING_CLOSE = "\n</thinking>\033[0m\n"

        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        if in_reasoning:
                            flush_buffer()
                            write_q.put(REASONING_CLOSE)
                            in_reasoning = False
                        flush_buffer()  # Flush any remaining tokens
                        break
                    try:
                        data = json.loads(data_str)

                        if first_token_time is None and data.get("choices"):
                            delta_peek = data["choices"][0].get("delta", {})
                            if delta_peek.get("content") or delta_peek.get("reasoning"):
                                first_token_time = time.time() - start_time

                        # The final chunk under stream_options.include_usage=True
                        # has `choices: []` (empty) and only carries usage —
                        # don't try to read a delta from it.
                        choices = data.get("choices") or []
                        delta = choices[0].get("delta", {}) if choices else {}
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning", "")

                        # Handle reasoning: render in a dim italic block so
                        # the user can see thinking happen, but it's visually
                        # distinct from the final answer.
                        if reasoning:
                            if not in_reasoning:
                                flush_buffer()  # close any pending content batch
                                write_q.put(REASONING_OPEN)
                                in_reasoning = True
                            token_buffer += reasoning
                            tokens_in_buffer += 1
                            reasoning_text += reasoning

                        if content:
                            if in_reasoning:
                                flush_buffer()  # drain reasoning before switching
                                write_q.put(REASONING_CLOSE)
                                in_reasoning = False
                            token_buffer += content
                            tokens_in_buffer += 1
                            content_text += content

                        if reasoning or content:
                            # Flush buffer if: enough tokens, enough chars, or timeout.
                            # Newlines-in-content trigger a flush for prose
                            # readability (no mid-sentence freezes). Newlines-in-
                            # reasoning don't — GLM chain-of-thought is heavily
                            # newline-delimited (numbered steps, line breaks
                            # between thoughts), so flushing per-newline there
                            # would collapse the effective batch size to ~1 token
                            # and add unnecessary writer wake-ups.
                            current_time = time.time()
                            should_flush = (
                                tokens_in_buffer >= BATCH_SIZE or
                                len(token_buffer) >= BATCH_CHARS or
                                (current_time - last_flush_time) >= BATCH_TIMEOUT or
                                '\n' in content
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
        # Shut down the writer thread and wait for it to drain.
        write_q.put(WRITER_DONE)
        writer_thread.join()
        print()

        if completion_tokens == 0:
            # Approximate when usage wasn't included; counts both visible and
            # reasoning tokens so the rate calc downstream isn't deflated for
            # reasoning-heavy models.
            completion_tokens = int(len((content_text + reasoning_text).split()) * 1.3)

        return content_text, reasoning_text, {
            "first_token_time": first_token_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    
    def reset_conversation(self):
        """Close the current conversation; the next turn opens a new one."""
        self.conversation_history = []
        self.turn_count = 0
        self.conversation_id = None

    def load_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Replace in-memory state with a stored conversation. Returns metadata
        dict on success, None if the id doesn't exist or storage isn't attached.
        """
        if self.store is None:
            return None
        meta, msgs = self.store.load_conversation(conversation_id)
        if meta is None:
            return None
        # Only replay visible content back to the model — reasoning is stored
        # for search but not part of the chat input.
        self.conversation_history = [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]
        self.conversation_id = conversation_id
        self.turn_count = meta["turn_count"]
        return {
            "id": meta["id"],
            "title": meta["title"],
            "model": meta["model"],
            "turn_count": meta["turn_count"],
            "summary": meta["summary"],
        }

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
        table.add_row("/new", "Start a new conversation (alias: /clear)")
        table.add_row("/list [N]", "List N most recent stored conversations (default 20)")
        table.add_row("/load <id>", "Resume a stored conversation by id")
        table.add_row("/search <query> [--in answer|reasoning|both]",
                      "Full-text search across stored messages (default: both)")
        table.add_row("/title <text>", "Manually set the title of the active conversation")
        table.add_row("/summary", "Show the rolling summary of the active conversation")
        table.add_row("/stats", "Show session statistics")
        table.add_row("/history", "Show in-memory conversation history")
        table.add_row("/stream", "Toggle streaming mode")
        table.add_row("/quit", "Exit the chat")

        console.print(table)
    else:
        print("""
Commands:
  /help                                          - Show this help message
  /new                                           - Start a new conversation (alias: /clear)
  /list [N]                                      - List N most recent stored conversations
  /load <id>                                     - Resume a stored conversation by id
  /search <query> [--in answer|reasoning|both]   - Full-text search (default scope: both)
  /title <text>                                  - Manually set conversation title
  /summary                                       - Show rolling summary
  /stats                                         - Show session statistics
  /history                                       - Show in-memory history
  /stream                                        - Toggle streaming mode
  /quit                                          - Exit the chat
""")


def _human_age(iso_str: str) -> str:
    """Render an ISO 8601 UTC timestamp as a relative '5m ago' string."""
    try:
        then = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def print_conversation_list(rows) -> None:
    """Render /list output: one line per stored conversation."""
    if not rows:
        if RICH_AVAILABLE:
            console.print("[yellow](no stored conversations yet)[/yellow]")
        else:
            print("(no stored conversations yet)")
        return
    if RICH_AVAILABLE:
        table = Table(title="Stored Conversations", show_header=True)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("When", style="yellow")
        table.add_column("Turns", justify="right")
        table.add_column("Model", style="dim")
        table.add_column("Title", style="white")
        for r in rows:
            table.add_row(
                str(r["id"]),
                _human_age(r["updated_at"]),
                str(r["turn_count"]),
                (r["model"] or "")[:32],
                r["title"] or "[dim](untitled)[/dim]",
            )
        console.print(table)
    else:
        for r in rows:
            print(f"#{r['id']:>4}  {_human_age(r['updated_at']):>10}  "
                  f"turns={r['turn_count']:<3}  "
                  f"{(r['model'] or '')[:24]:<24}  "
                  f"{r['title'] or '(untitled)'}")


def print_search_results(hits, query: str, scope: str) -> None:
    if not hits:
        msg = f"No matches for {query!r} in {scope}."
        if RICH_AVAILABLE:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        return
    if RICH_AVAILABLE:
        console.print(f"[bold]{len(hits)}[/bold] match(es) for "
                      f"[cyan]{query}[/cyan] in [magenta]{scope}[/magenta]:")
        for h in hits:
            title = h.title or "(untitled)"
            console.print(
                f"\n[cyan]#{h.conversation_id}[/cyan] "
                f"[white]{title}[/white]  "
                f"[dim]{_human_age(h.updated_at)}  "
                f"turn {h.turn} {h.role} ({h.matched_in})[/dim]"
            )
            # Snippet already contains [bold]...[/bold] markers from FTS5.
            console.print(f"  {h.snippet}")
    else:
        print(f"{len(hits)} match(es) for {query!r} in {scope}:")
        for h in hits:
            title = h.title or "(untitled)"
            print(f"\n#{h.conversation_id}  {title}  "
                  f"[{_human_age(h.updated_at)}, turn {h.turn} {h.role}, "
                  f"matched in {h.matched_in}]")
            # Strip Rich markup for plain output
            snippet = h.snippet.replace("[bold]", "*").replace("[/bold]", "*")
            print(f"  {snippet}")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Devstral via vLLM")
    parser.add_argument("host", help="vLLM server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses")
    parser.add_argument("--bench", metavar="PROMPT", help="Run a non-interactive benchmark with PROMPT and exit")
    parser.add_argument("-n", "--runs", type=int, default=3, help="Bench: number of measured runs (default 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Bench: max output tokens per run (default 256)")
    parser.add_argument("--warmup", type=int, default=1, help="Bench: warmup runs excluded from averages (default 1)")
    parser.add_argument("--resume", nargs="?", const=-1, type=int, metavar="ID",
                        help="Resume most recent conversation (no value), or specific id")
    parser.add_argument("--no-store", action="store_true",
                        help="Disable conversation persistence (storage is on by default)")
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

    # Storage attaches after the server probe so the summarizer can use the
    # detected model. --no-store skips this for ephemeral sessions.
    if not args.no_store:
        try:
            chat.attach_storage(default_db_path())
        except Exception as e:
            sys.stderr.write(f"[storage] init failed, continuing without persistence: {e}\n")

    if args.resume is not None:
        if chat.store is None:
            if RICH_AVAILABLE:
                console.print("[red]--resume requires storage; remove --no-store to use it.[/red]")
            else:
                print("--resume requires storage; remove --no-store to use it.")
        else:
            target = args.resume if args.resume > 0 else chat.store.most_recent_id()
            if target is None:
                if RICH_AVAILABLE:
                    console.print("[yellow]No stored conversations to resume.[/yellow]")
                else:
                    print("No stored conversations to resume.")
            else:
                meta = chat.load_conversation(target)
                if meta is None:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Conversation #{target} not found.[/red]")
                    else:
                        print(f"Conversation #{target} not found.")
                else:
                    title = meta["title"] or "(untitled)"
                    msg = (f"Resumed conversation #{meta['id']}: {title}  "
                           f"(turns: {meta['turn_count']})")
                    if RICH_AVAILABLE:
                        console.print(f"[green]✓[/green] {msg}")
                    else:
                        print(msg)

    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] Streaming: [cyan]{'enabled' if chat.stream else 'disabled'}[/cyan]")
        console.print("\nType [cyan]/help[/cyan] for commands, [cyan]/quit[/cyan] to exit\n")
    else:
        print(f"Streaming: {'enabled' if chat.stream else 'disabled'}")
        print("\nType /help for commands, /quit to exit\n")
    
    def _say(text: str) -> None:
        if RICH_AVAILABLE:
            console.print(text)
        else:
            # Strip simple [tag]...[/tag] markup for plain output
            import re
            print(re.sub(r"\[/?[a-zA-Z0-9 _#]+\]", "", text))

    try:
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
                    parts = user_input.split(None, 1)
                    head = parts[0].lower()
                    rest = parts[1].strip() if len(parts) > 1 else ""

                    if head in ("/quit", "/exit", "/q"):
                        if RICH_AVAILABLE:
                            console.print(chat.get_stats_table())
                            console.print("[yellow]Goodbye![/yellow]")
                        else:
                            print("Goodbye!")
                        break
                    elif head == "/help":
                        print_help()
                    elif head in ("/clear", "/new"):
                        chat.reset_conversation()
                        _say("[yellow]Started a new conversation.[/yellow]")
                    elif head == "/stats":
                        if RICH_AVAILABLE:
                            console.print(chat.get_stats_table())
                        else:
                            print(f"\nStats: {chat.turn_count} turns, "
                                  f"{chat.total_prompt_tokens + chat.total_completion_tokens} total tokens")
                    elif head == "/history":
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
                    elif head == "/stream":
                        chat.stream = not chat.stream
                        _say(f"Streaming: [cyan]{'enabled' if chat.stream else 'disabled'}[/cyan]")
                    elif head == "/list":
                        if chat.store is None:
                            _say("[red]Storage is disabled (--no-store).[/red]")
                            continue
                        try:
                            n = int(rest) if rest else 20
                        except ValueError:
                            _say("[red]Usage: /list [N][/red]")
                            continue
                        print_conversation_list(chat.store.list_conversations(n))
                    elif head == "/load":
                        if chat.store is None:
                            _say("[red]Storage is disabled (--no-store).[/red]")
                            continue
                        try:
                            cid = int(rest)
                        except ValueError:
                            _say("[red]Usage: /load <conversation-id>[/red]")
                            continue
                        meta = chat.load_conversation(cid)
                        if meta is None:
                            _say(f"[red]Conversation #{cid} not found.[/red]")
                        else:
                            title = meta["title"] or "(untitled)"
                            _say(f"[green]✓[/green] Resumed #{meta['id']}: {title}  "
                                 f"(turns: {meta['turn_count']})")
                            if meta["model"] and chat.model and meta["model"] != chat.model:
                                _say(f"[yellow]Note: stored model was "
                                     f"{meta['model']!r}, current is {chat.model!r}.[/yellow]")
                    elif head == "/search":
                        if chat.store is None:
                            _say("[red]Storage is disabled (--no-store).[/red]")
                            continue
                        if not rest:
                            _say("[red]Usage: /search <query> [--in answer|reasoning|both][/red]")
                            continue
                        # Split off a trailing `--in <scope>` if present.
                        scope = "both"
                        query = rest
                        marker = " --in "
                        if marker in query:
                            query, _, scope_arg = query.rpartition(marker)
                            scope = scope_arg.strip().lower()
                            query = query.strip()
                        if scope not in ("answer", "reasoning", "both"):
                            _say("[red]--in must be one of: answer, reasoning, both[/red]")
                            continue
                        if not query:
                            _say("[red]Search query is empty.[/red]")
                            continue
                        try:
                            hits = chat.store.search(query, scope=scope)
                        except Exception as e:
                            _say(f"[red]Search failed: {e}[/red]")
                            continue
                        print_search_results(hits, query, scope)
                    elif head == "/title":
                        if chat.store is None:
                            _say("[red]Storage is disabled (--no-store).[/red]")
                            continue
                        if chat.conversation_id is None:
                            _say("[yellow]No active conversation yet — send a message first.[/yellow]")
                            continue
                        if not rest:
                            _say("[red]Usage: /title <text>[/red]")
                            continue
                        chat.store.update_title(chat.conversation_id, rest)
                        _say(f"[green]✓[/green] Title set: {rest}")
                    elif head == "/summary":
                        if chat.store is None:
                            _say("[red]Storage is disabled (--no-store).[/red]")
                            continue
                        if chat.conversation_id is None:
                            _say("[yellow]No active conversation yet.[/yellow]")
                            continue
                        summary, through = chat.store.get_summary(chat.conversation_id)
                        if summary:
                            _say(f"[bold]Summary[/bold] (covers turns 1-{through}):\n{summary}")
                        else:
                            _say("[dim](no summary yet — first one is generated at turn 5)[/dim]")
                    else:
                        _say(f"[red]Unknown command:[/red] {user_input}")
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
    finally:
        # Drain background summarizer (5s budget) so an in-flight title or
        # summary lands before we exit. Anything still in flight is abandoned.
        if chat.summarizer is not None:
            chat.summarizer.shutdown(drain_timeout_s=5.0)
        if chat.store is not None:
            chat.store.close()


if __name__ == "__main__":
    main()

