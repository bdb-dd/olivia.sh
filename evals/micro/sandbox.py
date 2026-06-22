"""Throwaway per-run sandbox for L1 tasks.

Each task run gets a fresh temp directory seeded from the task's ``files`` map,
so runs are isolated and need no git-clean reset between them. Removed after the
run (keep with ``cleanup=False`` for debugging a failure).
"""
from contextlib import contextmanager
import os
import shutil
import tempfile


def materialize(files: dict, dest: str) -> None:
    for rel, content in (files or {}).items():
        full = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(full) or dest, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


@contextmanager
def sandbox(files: dict, cleanup: bool = True):
    path = tempfile.mkdtemp(prefix="olivia-l1-")
    try:
        materialize(files, path)
        yield path
    finally:
        if cleanup:
            shutil.rmtree(path, ignore_errors=True)
