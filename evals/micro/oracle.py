"""Deterministic success oracles for L1 tasks.

Evaluated against the *final sandbox state* after the agent loop ends — never
against the model's own claim of success. A task passes iff every oracle check
passes. Available checks::

    {"check": "file_exists",       "path": "out.txt"}
    {"check": "file_contains",     "path": "out.txt", "text": "ok", "case_insensitive": true}
    {"check": "file_not_contains", "path": "a.py",    "text": "oldName"}
    {"check": "file_equals",       "path": "out.txt", "content": "42", "strip": true}
    {"check": "file_matches",      "path": "out.txt", "pattern": "^\\d+$"}
    {"check": "bash_exit_zero",    "command": "python -m pytest -q", "timeout": 60}
    {"check": "stdout_contains",   "command": "python main.py", "text": "OK", "timeout": 30}
"""
from dataclasses import dataclass
import os
import re

from evals.micro import tools


@dataclass
class OracleResult:
    name: str
    passed: bool
    detail: str = ""


def _read(sandbox: str, path: str):
    full = os.path.join(sandbox, path)
    if not os.path.isfile(full):
        return None
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def evaluate(sandbox: str, specs, bash_timeout: float = 60.0) -> list:
    out = []
    for s in specs or []:
        c = s.get("check")

        if c == "file_exists":
            ok = os.path.isfile(os.path.join(sandbox, s["path"]))
            out.append(OracleResult(f"file_exists:{s['path']}", ok))

        elif c in ("file_contains", "file_not_contains"):
            body = _read(sandbox, s["path"])
            if body is None:
                out.append(OracleResult(f"{c}:{s['path']}", False, "file missing"))
                continue
            text = s["text"]
            hay, needle = (body, text)
            if s.get("case_insensitive"):
                hay, needle = hay.lower(), needle.lower()
            present = needle in hay
            ok = present if c == "file_contains" else not present
            out.append(OracleResult(f"{c}:{s['path']}", ok, f"text={text!r}"))

        elif c == "file_equals":
            body = _read(sandbox, s["path"])
            if body is None:
                out.append(OracleResult(f"file_equals:{s['path']}", False, "file missing"))
                continue
            want = s["content"]
            if s.get("strip", True):
                body, want = body.strip(), want.strip()
            out.append(OracleResult(f"file_equals:{s['path']}", body == want,
                                    f"got={body[:60]!r}"))

        elif c == "file_matches":
            body = _read(sandbox, s["path"])
            if body is None:
                out.append(OracleResult(f"file_matches:{s['path']}", False, "file missing"))
                continue
            ok = re.search(s["pattern"], body, re.MULTILINE) is not None
            out.append(OracleResult(f"file_matches:{s['path']}", ok, f"pattern={s['pattern']!r}"))

        elif c == "bash_exit_zero":
            r = tools.run_bash(sandbox, s["command"], timeout=s.get("timeout", bash_timeout))
            out.append(OracleResult(f"bash_exit_zero:{s['command'][:40]}", not r.is_error,
                                    r.output[-200:]))

        elif c == "stdout_contains":
            # stdout only (not stderr) and exit 0 — a SyntaxError traceback echoes
            # the offending source line to stderr, which must not count as output.
            r = tools.run_bash(sandbox, s["command"], timeout=s.get("timeout", bash_timeout))
            ok = (r.returncode == 0) and (s["text"] in r.stdout)
            out.append(OracleResult(f"stdout_contains:{s['command'][:40]}", ok,
                                    f"want={s['text']!r} exit={r.returncode}"))

        else:
            out.append(OracleResult(f"unknown_oracle:{c}", False, "unknown check"))

    return out


def succeeded(results: list) -> bool:
    return bool(results) and all(r.passed for r in results)
