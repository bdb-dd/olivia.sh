#!/usr/bin/env python3
"""Offline test for the L2-real runner's ``--concurrency`` machinery.

The real value of concurrency (vLLM batching N request streams) needs the
cluster, but the *orchestration* — N isolated clone/venv slots, each instance
run exactly once, true N-way parallelism, slot reuse, and resume-skipping —
is pure Python and tested here with stubbed clone + ``run_instance`` (no git,
no model, no GPU).

    .venv/bin/python evals/swe_real/test_concurrency.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import evals.swe_real.runner as R  # noqa: E402

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}{('  — ' + detail) if detail else ''}")


class _Cfg:
    def __init__(self, work):
        self.work = work
        self.concurrency = 1
        self.gold = False


def _instances(n):
    return [{"instance_id": f"t-{i:02d}", "version": "x", "base_commit": "deadbeef"}
            for i in range(n)]


def _install_stubs(probe):
    """Replace clone + run_instance with fast in-memory stubs.

    ``probe`` accumulates: max observed concurrency, the set of (repo,venv)
    slots seen, and the per-instance call count.
    """
    # ensure_clone: no real git/venv — just hand back this slot's canonical paths.
    R.ensure_clone = lambda work, slot=0, seed_repo=None: R.slot_paths(work, slot)

    lock = threading.Lock()

    def fake_run_instance(inst, repo, venv, cfg):
        with lock:
            probe["cur"] += 1
            probe["max"] = max(probe["max"], probe["cur"])
            probe["slots"].add((repo, venv))
            probe["calls"][inst["instance_id"]] = probe["calls"].get(inst["instance_id"], 0) + 1
        time.sleep(0.05)  # hold the slot long enough for real overlap
        with lock:
            probe["cur"] -= 1
        return {"instance_id": inst["instance_id"], "version": inst["version"],
                "resolved": inst["instance_id"].endswith(("2", "5", "8")),
                "turns": 3, "terminated": "stopped", "wall_s": 0.05}

    R.run_instance = fake_run_instance


def test_slot_paths():
    print("slot_paths — isolated trees per slot")
    s0 = R.slot_paths("/w", 0)
    s1 = R.slot_paths("/w", 1)
    s2 = R.slot_paths("/w", 2)
    check("slot 0 is the canonical work/django+venv", s0 == ("/w/django", "/w/venv"), str(s0))
    check("higher slots get isolated cc<N> trees", s1 == ("/w/cc1/django", "/w/cc1/venv"), str(s1))
    check("every slot is a distinct (repo,venv)", len({s0, s1, s2}) == 3)


def test_parallelism():
    print("run_concurrent — N-way parallel, each instance once, slots isolated")
    orig_ec, orig_ri = R.ensure_clone, R.run_instance
    probe = {"cur": 0, "max": 0, "slots": set(), "calls": {}}
    _install_stubs(probe)
    try:
        with tempfile.TemporaryDirectory() as work:
            resdir = os.path.join(work, "results")
            os.makedirs(resdir)
            insts = _instances(12)
            R.run_concurrent(insts, _Cfg(work), resdir, n_workers=4)

            files = [f for f in os.listdir(resdir) if f.endswith(".json")]
            check("a result file per instance", len(files) == 12, f"{len(files)} files")
            check("each instance run exactly once",
                  all(probe["calls"].get(i["instance_id"]) == 1 for i in insts),
                  str(probe["calls"]))
            check("achieved true 4-way concurrency", probe["max"] == 4, f"max={probe['max']}")
            check("used exactly 4 isolated slots", len(probe["slots"]) == 4,
                  f"{len(probe['slots'])} slots")
    finally:
        R.ensure_clone, R.run_instance = orig_ec, orig_ri


def test_resume():
    print("run_concurrent — resumable (skips finished instances)")
    orig_ec, orig_ri = R.ensure_clone, R.run_instance
    probe = {"cur": 0, "max": 0, "slots": set(), "calls": {}}
    _install_stubs(probe)
    try:
        with tempfile.TemporaryDirectory() as work:
            resdir = os.path.join(work, "results")
            os.makedirs(resdir)
            insts = _instances(6)
            # Pre-seed 2 instances as already done.
            for done in insts[:2]:
                with open(R._result_path(resdir, done), "w") as f:
                    f.write('{"resolved": true}')
            R.run_concurrent(insts, _Cfg(work), resdir, n_workers=3)
            ran = set(probe["calls"])
            check("pre-existing results are skipped",
                  ran == {i["instance_id"] for i in insts[2:]}, str(sorted(ran)))
            check("all 6 result files present afterward",
                  len([f for f in os.listdir(resdir) if f.endswith(".json")]) == 6)
    finally:
        R.ensure_clone, R.run_instance = orig_ec, orig_ri


def test_workers_capped_to_todo():
    print("run_concurrent — workers capped to instance count")
    orig_ec, orig_ri = R.ensure_clone, R.run_instance
    probe = {"cur": 0, "max": 0, "slots": set(), "calls": {}}
    _install_stubs(probe)
    try:
        with tempfile.TemporaryDirectory() as work:
            resdir = os.path.join(work, "results")
            os.makedirs(resdir)
            insts = _instances(2)
            R.run_concurrent(insts, _Cfg(work), resdir, n_workers=8)  # more workers than work
            check("never provisions more slots than instances", len(probe["slots"]) <= 2,
                  f"{len(probe['slots'])} slots for 2 instances")
            check("both instances still processed", len(probe["calls"]) == 2)
    finally:
        R.ensure_clone, R.run_instance = orig_ec, orig_ri


def main():
    print("L2-real runner concurrency selftest\n")
    test_slot_paths()
    test_parallelism()
    test_resume()
    test_workers_capped_to_todo()
    print()
    print(f"selftest: {_passed} checks passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
