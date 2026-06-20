"""The L1 micro-agent loop.

Drives a model through multiple turns against ``anthropic_proxy.py``: send
messages + tools, execute any tool_use the model emits against the sandbox, feed
the tool_result back, repeat until the model stops (no tool call) or the turn cap
is hit. Then a deterministic :mod:`oracle` judges the final sandbox state.

Reuse:
* ``client.call_nonstream`` for each turn (non-streaming — the loop only needs
  the final message; the SSE grammar is L0's job);
* ``conformance.check_invariants`` to score *tool-call validity per turn*, so a
  protocol break mid-loop (malformed/hallucinated tool call) is caught here too.

This also exercises the proxy's history translation (assistant ``tool_use`` +
user ``tool_result`` -> OpenAI ``tool_calls`` + ``role:tool``) every turn.
"""
from dataclasses import dataclass, field
import json
import time

from evals.protocol import client, conformance
from evals.micro import oracle, sandbox, tools

DEFAULT_SYSTEM = (
    "You are an autonomous coding agent working inside a sandbox directory. "
    "Use the provided tools to complete the task. Inspect files before changing "
    "them. When the task is fully done, stop and reply with a brief confirmation "
    "and no further tool call. Do not ask the user questions — act on your own."
)


@dataclass
class Outcome:
    id: str
    category: str
    success: bool = False
    turns_used: int = 0
    tool_calls: int = 0
    invalid_tool_turns: int = 0      # turns whose response broke an L0 invariant
    thrash_count: int = 0            # repeated identical (tool,args) calls
    terminated: str = "stopped"      # "stopped" | "max_turns" | "error"
    premature_stop: bool = False     # stopped on its own but oracle failed
    error: str = ""
    wall_s: float = 0.0
    oracle: list = field(default_factory=list)
    transcript: list = field(default_factory=list)


def _assistant_history(content: list) -> list:
    # Keep text + tool_use for history; drop thinking (the proxy drops prior
    # thinking anyway, and not all upstreams accept it back).
    return [b for b in content if b.get("type") in ("text", "tool_use")]


def run_task(task: dict, sandbox_dir: str, *, base_url: str, model: str,
             max_turns: int = 12, bash_timeout: float = 30.0,
             max_tokens: int = 2048, call_fn=None) -> Outcome:
    call = call_fn or client.call_nonstream
    out = Outcome(id=task.get("id", "?"), category=task.get("category", "uncategorized"))
    tool_names = task.get("tools") or list(tools.TOOL_SCHEMAS.keys())
    tool_definitions = tools.tool_defs(tool_names)
    system = task.get("system") or DEFAULT_SYSTEM

    messages = [{"role": "user", "content": task["prompt"]}]
    seen = set()
    t0 = time.perf_counter()

    for turn in range(max_turns):
        out.turns_used = turn + 1
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": messages,
            "tools": tool_definitions,
        }
        res = call(base_url, body)
        if not res.ok:
            out.terminated = "error"
            out.error = res.error or "call failed"
            break

        resp = res.response or {}
        if any(not c.passed for c in conformance.check_invariants(resp, tool_definitions)):
            out.invalid_tool_turns += 1

        tus = conformance.tool_uses(resp)
        out.transcript.append({"turn": turn + 1,
                               "tool_calls": [{"name": t.get("name"), "input": t.get("input")} for t in tus],
                               "text": conformance.text_of(resp)[:200]})

        if not tus:
            out.terminated = "stopped"
            break

        messages.append({"role": "assistant", "content": _assistant_history(resp["content"])})

        results = []
        for tu in tus:
            out.tool_calls += 1
            key = (tu.get("name"), json.dumps(tu.get("input"), sort_keys=True, default=str))
            if key in seen:
                out.thrash_count += 1
            seen.add(key)
            tr = tools.execute(sandbox_dir, tu.get("name"), tu.get("input") or {}, bash_timeout)
            block = {"type": "tool_result", "tool_use_id": tu.get("id"), "content": tr.output}
            if tr.is_error:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})
    else:
        out.terminated = "max_turns"

    # L2: materialize hidden grading files only now — they are kept out of the
    # agent's view during the loop (written after it ends), so the agent solves
    # from the problem statement, not by reading the test (SWE-bench style).
    sandbox.materialize(task.get("oracle_files", {}), sandbox_dir)
    out.oracle = oracle.evaluate(sandbox_dir, task.get("oracle", []), bash_timeout=max(bash_timeout, 60))
    out.success = oracle.succeeded(out.oracle)
    out.premature_stop = (out.terminated == "stopped" and not out.success)
    out.wall_s = round(time.perf_counter() - t0, 2)
    return out
