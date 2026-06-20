"""Sandbox tools for the L1 micro-agent harness.

A small fixed tool set the model drives to complete a task inside a throwaway
directory: ``read_file``, ``write_file``, ``list_dir``, ``grep``, ``run_bash``.
Each tool has an Anthropic tool definition (sent to the model) and a local
executor (run against the sandbox dir).

SECURITY: ``run_bash`` executes model-generated shell in the sandbox cwd with a
timeout. File tools are jailed to the sandbox via realpath checks, but bash is
not fully jailed — it is the user's machine, bounded only by ``cwd`` + timeout.
For untrusted models run the whole harness inside the cluster container or a
disposable VM; ``--no-bash`` drops the tool entirely for read/write-only tasks.
"""
from dataclasses import dataclass
import os
import re
import subprocess

MAX_OUTPUT = 4000  # truncate tool output fed back to the model


@dataclass
class ToolResult:
    output: str                 # combined stdout+stderr+exit, shown to the model
    is_error: bool = False
    stdout: str = ""            # bash only: separated streams for deterministic oracles
    stderr: str = ""
    returncode: int = 0


# --------------------------------------------------------------------------- #
# Anthropic tool definitions (what the model sees)                            #
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = {
    "read_file": {
        "name": "read_file",
        "description": "Read and return the full contents of a file in the working directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path."}},
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write (create or overwrite) a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "list_dir": {
        "name": "list_dir",
        "description": "List files and directories under a path (default '.').",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    "grep": {
        "name": "grep",
        "description": "Search files for a regular expression and return matching lines with file:line prefixes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "File or directory to search (default '.')."},
            },
            "required": ["pattern"],
        },
    },
    "run_bash": {
        "name": "run_bash",
        "description": "Run a bash command in the working directory and return stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def tool_defs(names) -> list:
    return [TOOL_SCHEMAS[n] for n in names if n in TOOL_SCHEMAS]


# --------------------------------------------------------------------------- #
# Executors                                                                    #
# --------------------------------------------------------------------------- #

def _safe_path(sandbox: str, path: str) -> str:
    root = os.path.realpath(sandbox)
    full = os.path.realpath(os.path.join(sandbox, path or "."))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"path escapes sandbox: {path!r}")
    return full


def _trunc(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + "\n[...truncated...]"


def read_file(sandbox: str, path: str) -> ToolResult:
    try:
        with open(_safe_path(sandbox, path), "r", encoding="utf-8", errors="replace") as f:
            return ToolResult(_trunc(f.read()))
    except FileNotFoundError:
        return ToolResult(f"error: no such file: {path}", is_error=True)
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"error: {e}", is_error=True)


def write_file(sandbox: str, path: str, content: str) -> ToolResult:
    try:
        full = _safe_path(sandbox, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")
        return ToolResult(f"wrote {path} ({len(content or '')} bytes)")
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"error: {e}", is_error=True)


def list_dir(sandbox: str, path: str = ".") -> ToolResult:
    try:
        full = _safe_path(sandbox, path)
        entries = sorted(os.listdir(full))
        lines = [(e + "/" if os.path.isdir(os.path.join(full, e)) else e) for e in entries]
        return ToolResult("\n".join(lines) or "(empty)")
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"error: {e}", is_error=True)


def grep(sandbox: str, pattern: str, path: str = ".") -> ToolResult:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(f"error: bad pattern: {e}", is_error=True)
    root = _safe_path(sandbox, path)
    targets = []
    if os.path.isfile(root):
        targets = [root]
    else:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                targets.append(os.path.join(dirpath, fn))
    hits = []
    for t in targets:
        rel = os.path.relpath(t, os.path.realpath(sandbox))
        try:
            with open(t, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{i}:{line.rstrip()}")
        except Exception:  # noqa: BLE001 — skip binaries/unreadable
            continue
    return ToolResult(_trunc("\n".join(hits)) if hits else "(no matches)")


def run_bash(sandbox: str, command: str, timeout: float = 30.0) -> ToolResult:
    try:
        p = subprocess.run(command, shell=True, cwd=sandbox, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(f"[timed out after {timeout}s]", is_error=True, returncode=124)
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"error: {e}", is_error=True, returncode=1)
    parts = []
    if p.stdout:
        parts.append(p.stdout.rstrip())
    if p.stderr:
        parts.append("[stderr]\n" + p.stderr.rstrip())
    parts.append(f"[exit={p.returncode}]")
    return ToolResult(_trunc("\n".join(parts)), is_error=(p.returncode != 0),
                      stdout=p.stdout or "", stderr=p.stderr or "", returncode=p.returncode)


def execute(sandbox: str, name: str, args: dict, bash_timeout: float = 30.0) -> ToolResult:
    args = args or {}
    if name == "read_file":
        return read_file(sandbox, args.get("path", ""))
    if name == "write_file":
        return write_file(sandbox, args.get("path", ""), args.get("content", ""))
    if name == "list_dir":
        return list_dir(sandbox, args.get("path", "."))
    if name == "grep":
        return grep(sandbox, args.get("pattern", ""), args.get("path", "."))
    if name == "run_bash":
        return run_bash(sandbox, args.get("command", ""), timeout=bash_timeout)
    return ToolResult(f"error: unknown tool {name!r}", is_error=True)
