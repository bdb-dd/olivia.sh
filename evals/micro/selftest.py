#!/usr/bin/env python3
"""Offline self-test for the L1 micro-agent harness.

Proves the sandbox tools, the deterministic oracle, and the full multi-turn
agent loop (turn counting, thrash detection, premature-stop / runaway
classification, oracle wiring) all work — with NO GPU and NO proxy, by driving
the loop with scripted fake "models".

    python evals/micro/selftest.py     # or: python -m evals.micro.selftest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.protocol.client import CallResult  # noqa: E402
from evals.micro import agent, oracle, sandbox, tools  # noqa: E402

_passed = 0
_failed = 0


def check(cond, label):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  SELFTEST FAIL: {label}")


# --- scripted fake models --------------------------------------------------- #

def _resp_tool(name, inp, tid="t1"):
    return CallResult(True, "nonstream", status=200, response={
        "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 5}})


def _resp_text(txt):
    return CallResult(True, "nonstream", status=200, response={
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": txt}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}})


def _has_assistant(body):
    return any(m["role"] == "assistant" for m in body["messages"])


WRITE_TASK = {
    "id": "t-write", "category": "file-ops",
    "prompt": "create greeting.txt containing Hello, World!",
    "files": {}, "tools": ["write_file", "read_file", "list_dir"],
    "oracle": [{"check": "file_equals", "path": "greeting.txt", "content": "Hello, World!"}],
}


def solver(base, body):
    if not _has_assistant(body):
        return _resp_tool("write_file", {"path": "greeting.txt", "content": "Hello, World!"})
    return _resp_text("Done.")


def premature(base, body):
    return _resp_text("I already handled it.")  # stops with no tool call


def thrasher(base, body):
    return _resp_tool("read_file", {"path": "a.txt"})  # same call forever, never stops


# --- tests ------------------------------------------------------------------ #

def test_tools():
    with sandbox.sandbox({"a.txt": "hello\nworld\n"}) as sb:
        check(not tools.write_file(sb, "b.txt", "data").is_error, "write_file ok")
        check(tools.read_file(sb, "b.txt").output == "data", "read_file roundtrip")
        check(tools.read_file(sb, "missing").is_error, "read_file missing -> error")
        r = tools.run_bash(sb, "echo hi")
        check((not r.is_error) and "hi" in r.output, "run_bash echo")
        check(tools.run_bash(sb, "exit 3").is_error, "run_bash nonzero -> error")
        g = tools.grep(sb, "world", "a.txt")
        check("a.txt:" in g.output and "world" in g.output, "grep finds match")
        check("a.txt" in tools.list_dir(sb, ".").output, "list_dir lists files")
        check(tools.read_file(sb, "../../../etc/hosts").is_error, "path jail blocks escape")


def test_oracle():
    with sandbox.sandbox({"out.txt": "50\n", "code.py": "x = 1\n"}) as sb:
        ok = lambda specs: oracle.succeeded(oracle.evaluate(sb, specs))
        check(ok([{"check": "file_equals", "path": "out.txt", "content": "50"}]), "file_equals pass")
        check(not ok([{"check": "file_equals", "path": "out.txt", "content": "51"}]), "file_equals fail")
        check(ok([{"check": "file_exists", "path": "code.py"}]), "file_exists pass")
        check(not ok([{"check": "file_exists", "path": "nope"}]), "file_exists fail")
        check(ok([{"check": "bash_exit_zero", "command": "true"}]), "bash_exit_zero pass")
        check(not ok([{"check": "bash_exit_zero", "command": "false"}]), "bash_exit_zero fail")
        check(ok([{"check": "stdout_contains", "command": "echo OK", "text": "OK"}]), "stdout_contains pass")
        check(not ok([{"check": "evaluate", "path": "x"}]) and True, "unknown oracle -> not succeeded")


def test_loop_solves():
    with sandbox.sandbox(WRITE_TASK["files"]) as sb:
        out = agent.run_task(WRITE_TASK, sb, base_url="mock://", model="m", call_fn=solver)
    check(out.success, "solver: oracle success")
    check(out.terminated == "stopped", "solver: terminated stopped")
    check(out.turns_used == 2, f"solver: 2 turns (got {out.turns_used})")
    check(out.tool_calls == 1, f"solver: 1 tool call (got {out.tool_calls})")
    check(out.premature_stop is False, "solver: not premature")
    check(out.thrash_count == 0, "solver: no thrash")
    check(out.invalid_tool_turns == 0, "solver: no invalid turns")


def test_loop_premature_stop():
    with sandbox.sandbox(WRITE_TASK["files"]) as sb:
        out = agent.run_task(WRITE_TASK, sb, base_url="mock://", model="m", call_fn=premature)
    check(out.terminated == "stopped", "premature: terminated stopped")
    check(out.success is False, "premature: oracle fails")
    check(out.premature_stop is True, "premature: flagged premature_stop")
    check(out.tool_calls == 0 and out.turns_used == 1, "premature: 1 turn, 0 calls")


def test_loop_thrash_and_runaway():
    task = {"id": "t-thrash", "category": "x", "prompt": "p", "files": {"a.txt": "hi"},
            "tools": ["read_file"], "oracle": [{"check": "file_exists", "path": "never.txt"}]}
    with sandbox.sandbox(task["files"]) as sb:
        out = agent.run_task(task, sb, base_url="mock://", model="m", max_turns=3, call_fn=thrasher)
    check(out.terminated == "max_turns", "thrash: hits turn cap (runaway)")
    check(out.tool_calls == 3, f"thrash: 3 calls (got {out.tool_calls})")
    check(out.thrash_count == 2, f"thrash: 2 repeats flagged (got {out.thrash_count})")
    check(out.success is False, "thrash: oracle fails")
    check(out.premature_stop is False, "thrash: runaway is not premature_stop")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"L1 micro-agent selftest — {len(tests)} test groups")
    for t in tests:
        t()
    print(f"\nselftest: {_passed} checks passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
