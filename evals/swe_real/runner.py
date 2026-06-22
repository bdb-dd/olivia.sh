#!/usr/bin/env python3
"""L2-real — SWE-bench Verified runner (django slice), built for Olivia.

Runs entirely inside a python:3.11 apptainer on the `small` CPU partition
(x86 + internet + reaches the vLLM node), so it stays within Sigma2 usage policy
(no login-node compute, no GPU waste). For each instance:

  1. reset the django clone to the instance's base_commit (clone once, reuse);
  2. drive the model (via anthropic_proxy) through the agent loop over the repo —
     read/write/grep/run_bash with the venv on PATH — to fix the issue described
     in the problem statement (the agent never sees the evaluation tests);
  3. apply the gold `test_patch` (the hidden FAIL_TO_PASS/PASS_TO_PASS tests),
     run the affected test modules, parse django's output, and verify the agent's
     change makes FAIL_TO_PASS pass without breaking PASS_TO_PASS.

Resumable: one result JSON per instance; re-running skips finished instances.

Usage (inside the container, from the repo root):
  python evals/swe_real/runner.py --base-url http://localhost:8002 \
      --model poolside/Laguna-M.1-FP8 --preset laguna --work /cluster/work/.../swe
"""
import argparse
import concurrent.futures
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.protocol import client, conformance  # noqa: E402
from evals.micro import tools  # noqa: E402

REPO_URL = "https://github.com/django/django"
HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = os.path.join(HERE, "instances")

SWE_SYSTEM = (
    "You are an expert software engineer working in a checkout of the django/django "
    "repository at a specific past commit. You are given a bug report. Make the minimal "
    "change to the library source (under django/) that fixes the described bug. Use the "
    "tools to explore and edit files and to run shell commands; `python` is the repo's "
    "virtualenv and the test runner is `python tests/runtests.py --settings=test_sqlite "
    "<module>`. Do NOT edit files under tests/ — a separate hidden test suite will judge "
    "your fix. When the fix is complete, stop with a brief summary and no further tool call."
)

STATUS_RE = re.compile(r"^(.*?) \.\.\. (ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)$")


def sh(cmd, cwd=None, env=None, timeout=2400):
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# --------------------------------------------------------------------------- #
# django environment                                                          #
# --------------------------------------------------------------------------- #

# Optional packages django's test suite needs; without them the relevant tests
# SKIP (e.g. Pillow -> ImageField tests), which fails verification even on a
# correct fix. Lightweight subset (no selenium/geoip2/redis/pywatchman).
TEST_DEPS = ("Pillow PyYAML pytz tzdata sqlparse jinja2 docutils bcrypt "
             "argon2-cffi numpy tblib aiosmtpd")


def slot_paths(work, slot):
    """(repo, venv) for a worker slot. Slot 0 is the canonical ``work/django`` +
    ``work/venv`` (so a prior sequential run's clone is reused); higher slots get
    isolated ``work/cc<N>/`` trees so concurrent instances never share a checkout
    or venv (each ``reset_instance`` does a hard ``git checkout`` + ``pip -e``)."""
    if slot == 0:
        return os.path.join(work, "django"), os.path.join(work, "venv")
    base = os.path.join(work, f"cc{slot}")
    return os.path.join(base, "django"), os.path.join(base, "venv")


def ensure_clone(work, slot=0, seed_repo=None):
    repo, venv = slot_paths(work, slot)
    os.makedirs(os.path.dirname(repo), exist_ok=True)
    if not os.path.isdir(os.path.join(repo, ".git")):
        # Slot 0 clones from GitHub; higher slots clone from the local slot-0 repo
        # (fast, no network) when it exists, so K slots aren't K network clones.
        src = seed_repo if (seed_repo and os.path.isdir(os.path.join(seed_repo, ".git"))) else REPO_URL
        rc, out = sh(f"git clone --quiet {src} {repo}", timeout=900)
        if rc:
            raise RuntimeError(f"clone failed (slot {slot}): {out[-300:]}")
    if not os.path.isdir(venv):
        sh(f"python -m venv {venv}", timeout=180)
    # idempotent (cheap when satisfied); ensures django's optional test deps exist
    sh(f"{venv}/bin/python -m pip install -q -U pip", env=venv_env(venv), timeout=300)
    sh(f"{venv}/bin/python -m pip install -q {TEST_DEPS}", env=venv_env(venv), timeout=1200)
    return repo, venv


def venv_env(venv):
    env = dict(os.environ)
    env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = venv
    return env


def reset_instance(repo, venv, base_commit):
    env = venv_env(venv)
    sh(f"git checkout -f {base_commit}", cwd=repo, env=env)
    sh("git clean -fdxq", cwd=repo, env=env)
    # editable install (cheap once satisfied; keeps django importable at this checkout)
    rc, out = sh(f"{venv}/bin/python -m pip install -q -e .", cwd=repo, env=env, timeout=1800)
    if rc:
        raise RuntimeError(f"pip install failed: {out[-300:]}")
    return env


def test_modules_from_patch(test_patch):
    mods = set()
    for line in test_patch.splitlines():
        m = re.match(r"\+\+\+ b/tests/(.+)\.py", line)
        if m:
            mods.add(m.group(1).replace("/", "."))
    return sorted(mods)


def parse_django(output):
    res = {}
    for line in output.splitlines():
        m = STATUS_RE.match(line.rstrip())
        if not m:
            continue
        desc, st = m.group(1).strip(), m.group(2)
        status = ("PASSED" if st == "ok"
                  else "SKIPPED" if st.startswith("skipped") else "FAILED")
        res[desc] = status
        # django >=4.1 prints "method (module.Class.method)" — the method is
        # repeated inside the parens. SWE-bench ids store "method (module.Class)".
        # Register that normalized form too so lookups match either way.
        mm = re.match(r"^(\w+) \((.+)\.(\w+)\)$", desc)
        if mm and mm.group(1) == mm.group(3):
            res.setdefault(f"{mm.group(1)} ({mm.group(2)})", status)
    return res


def verify(output, fail_to_pass, pass_to_pass):
    parsed = parse_django(output)
    f2p = {t: parsed.get(t, "MISSING") for t in fail_to_pass}
    p2p = {t: parsed.get(t, "MISSING") for t in pass_to_pass}
    ok = all(v == "PASSED" for v in f2p.values()) and all(v == "PASSED" for v in p2p.values())
    return ok, {"fail_to_pass": f2p, "pass_to_pass": p2p}


# --------------------------------------------------------------------------- #
# agent loop over the repo                                                     #
# --------------------------------------------------------------------------- #

def run_agent(repo, prompt, *, base_url, model, env, max_turns, bash_timeout, max_tokens=4096):
    tdefs = tools.tool_defs(["read_file", "write_file", "list_dir", "grep", "run_bash"])
    messages = [{"role": "user", "content": prompt}]
    seen = set()
    m = {"turns": 0, "tool_calls": 0, "invalid_tool_turns": 0, "thrash": 0,
         "terminated": "stopped", "error": ""}
    for turn in range(max_turns):
        m["turns"] = turn + 1
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "system": SWE_SYSTEM, "messages": messages, "tools": tdefs}
        res = client.call_nonstream(base_url, body, timeout=900)
        if not res.ok:
            m["terminated"], m["error"] = "error", res.error or "call failed"
            break
        resp = res.response or {}
        if any(not c.passed for c in conformance.check_invariants(resp, tdefs)):
            m["invalid_tool_turns"] += 1
        tus = conformance.tool_uses(resp)
        if not tus:
            m["terminated"] = "stopped"
            break
        messages.append({"role": "assistant",
                         "content": [b for b in resp["content"] if b.get("type") in ("text", "tool_use")]})
        results = []
        for tu in tus:
            m["tool_calls"] += 1
            key = (tu.get("name"), json.dumps(tu.get("input"), sort_keys=True, default=str))
            if key in seen:
                m["thrash"] += 1
            seen.add(key)
            tr = tools.execute(repo, tu.get("name"), tu.get("input") or {}, bash_timeout, env=env)
            blk = {"type": "tool_result", "tool_use_id": tu.get("id"), "content": tr.output}
            if tr.is_error:
                blk["is_error"] = True
            results.append(blk)
        messages.append({"role": "user", "content": results})
    else:
        m["terminated"] = "max_turns"
    return m


def build_prompt(inst):
    return (f"# Bug report for django/django (commit {inst['base_commit'][:12]})\n\n"
            f"{inst['problem_statement']}\n\n"
            "Fix the bug in the library source. Do not edit the tests.")


# --------------------------------------------------------------------------- #
# orchestration                                                                #
# --------------------------------------------------------------------------- #

def run_instance(inst, repo, venv, cfg):
    t0 = time.time()
    rec = {"instance_id": inst["instance_id"], "version": inst["version"]}
    try:
        env = reset_instance(repo, venv, inst["base_commit"])
        if getattr(cfg, "gold", False):
            # Harness self-test: apply the gold patch instead of running the model.
            # This MUST resolve — if it doesn't, the bug is in setup/verify, not the model.
            with open(os.path.join(repo, ".goldpatch.diff"), "w") as f:
                f.write(inst["patch"])
            rc_g, out_g = sh("git apply -v .goldpatch.diff", cwd=repo, env=env)
            rec.update({"mode": "gold", "gold_applied": rc_g == 0, "turns": 0, "terminated": "gold"})
        else:
            agent = run_agent(repo, build_prompt(inst), base_url=cfg.base_url, model=cfg.model,
                              env=env, max_turns=cfg.max_turns, bash_timeout=cfg.bash_timeout)
            rec.update(agent)
            rec["mode"] = "agent"
        # capture the agent's diff (the fix), then apply the hidden test patch
        _, diff = sh("git diff", cwd=repo, env=env)
        rec["patch_size"] = len(diff)
        with open(os.path.join(repo, ".testpatch.diff"), "w") as f:
            f.write(inst["test_patch"])
        rc_tp, out_tp = sh("git apply -v .testpatch.diff", cwd=repo, env=env)
        rec["test_patch_applied"] = (rc_tp == 0)
        mods = test_modules_from_patch(inst["test_patch"])
        rec["test_modules"] = mods
        cmd = (f"{venv}/bin/python tests/runtests.py --settings=test_sqlite --verbosity 2 "
               + " ".join(mods))
        _, out = sh(cmd, cwd=repo, env=env, timeout=cfg.test_timeout)
        ok, detail = verify(out, inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"])
        rec["resolved"] = bool(ok and rec["test_patch_applied"])
        rec["verify"] = detail
    except Exception as e:  # noqa: BLE001 — record per-instance failures, keep going
        rec["resolved"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _result_path(resdir, inst):
    return os.path.join(resdir, inst["instance_id"] + ".json")


def run_sequential(insts, cfg, resdir):
    """One clone, instances one at a time (the original behaviour)."""
    repo, venv = ensure_clone(cfg.work, 0)
    for i, inst in enumerate(insts, 1):
        rpath = _result_path(resdir, inst)
        if os.path.isfile(rpath):
            cached = json.load(open(rpath))
            print(f"[{i}/{len(insts)}] {inst['instance_id']} — cached "
                  f"({'RESOLVED' if cached.get('resolved') else 'unresolved'})", flush=True)
            continue
        print(f"[{i}/{len(insts)}] {inst['instance_id']} (django {inst['version']}) — running...", flush=True)
        rec = run_instance(inst, repo, venv, cfg)
        with open(rpath, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"    -> {'RESOLVED' if rec.get('resolved') else 'UNRESOLVED'}  "
              f"turns={rec.get('turns','?')} term={rec.get('terminated','?')} "
              f"wall={rec.get('wall_s','?')}s {rec.get('error','')}", flush=True)


def run_concurrent(insts, cfg, resdir, n_workers):
    """Instances in parallel across N isolated clone/venv slots.

    Each in-flight instance holds one slot; the blocking model call
    (``urllib``) and the django test run (``subprocess``) both release the GIL,
    so N threads keep N requests in flight at once → the vLLM host batches them.
    Pair with ``anthropic_proxy --no-serialize`` (the serialize lock would
    otherwise force them back to one-at-a-time). Resumable: finished instances
    (a result file already on disk) are skipped, same as the sequential path.
    """
    todo = [inst for inst in insts if not os.path.isfile(_result_path(resdir, inst))]
    cached = len(insts) - len(todo)
    if cached:
        print(f"[cache] {cached}/{len(insts)} already done", flush=True)
    if not todo:
        return
    n_workers = max(1, min(n_workers, len(todo)))

    # Provision N (clone, venv) slots: slot 0 from GitHub, the rest cloned from
    # slot 0 locally and in parallel (so setup is ~one network clone, not N).
    print(f"[setup] provisioning {n_workers} clone/venv slots…", flush=True)
    seed_repo, _ = ensure_clone(cfg.work, 0)
    slots = [slot_paths(cfg.work, 0)]
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers - 1) as ex:
            slots.extend(ex.map(lambda k: ensure_clone(cfg.work, k, seed_repo),
                                range(1, n_workers)))
    print(f"[setup] {len(slots)} slots ready; dispatching {len(todo)} instances "
          f"on {n_workers} workers", flush=True)

    slot_q = queue.Queue()
    for s in slots:
        slot_q.put(s)
    plock = threading.Lock()
    done = {"n": 0}

    def work(inst):
        repo, venv = slot_q.get()
        try:
            rec = run_instance(inst, repo, venv, cfg)
        finally:
            slot_q.put((repo, venv))
        with open(_result_path(resdir, inst), "w") as f:
            json.dump(rec, f, indent=2)
        with plock:
            done["n"] += 1
            print(f"  [{done['n']}/{len(todo)}] {inst['instance_id']:<26} -> "
                  f"{'RESOLVED' if rec.get('resolved') else 'UNRESOLVED'}  "
                  f"turns={rec.get('turns','?')} term={rec.get('terminated','?')} "
                  f"wall={rec.get('wall_s','?')}s {rec.get('error','')}", flush=True)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(work, todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("OLIVIA_PROXY_URL", "http://localhost:8002"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="generic")
    ap.add_argument("--work", required=True, help="scratch dir on /cluster/work")
    ap.add_argument("--instances", default=INSTANCES)
    ap.add_argument("--only", default=None, help="comma list of instance_ids")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--bash-timeout", type=float, default=180.0)
    ap.add_argument("--test-timeout", type=float, default=1200.0)
    ap.add_argument("--gold", action="store_true",
                    help="apply the gold patch instead of the model (harness self-test)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="run N instances in parallel, each in its own clone/venv "
                         "(exploits vLLM batching for slow multi-node models; pair "
                         "with anthropic_proxy --no-serialize)")
    cfg = ap.parse_args()
    tag = cfg.preset + ("-gold" if cfg.gold else "")

    insts = [json.load(open(os.path.join(cfg.instances, f)))
             for f in sorted(os.listdir(cfg.instances)) if f.endswith(".json")]
    if cfg.only:
        want = {s.strip() for s in cfg.only.split(",")}
        insts = [i for i in insts if i["instance_id"] in want]
    os.makedirs(cfg.work, exist_ok=True)
    resdir = os.path.join(cfg.work, f"results-{tag}")
    os.makedirs(resdir, exist_ok=True)

    conc = max(1, cfg.concurrency)
    print(f"L2-real SWE-bench (django) — tag={tag} model={cfg.model} "
          f"base={cfg.base_url} instances={len(insts)} mode={'GOLD' if cfg.gold else 'agent'}"
          f"{f' concurrency={conc}' if conc > 1 else ''}", flush=True)

    if conc > 1:
        run_concurrent(insts, cfg, resdir, conc)
    else:
        run_sequential(insts, cfg, resdir)

    # Rebuild the ledger from the per-instance result files, in instance order
    # (concurrent runs finish out of order; this keeps the summary deterministic).
    records = []
    for inst in insts:
        rpath = _result_path(resdir, inst)
        if os.path.isfile(rpath):
            records.append(json.load(open(rpath)))

    n = len(records)
    res = sum(1 for r in records if r.get("resolved"))
    print(f"\n=== L2-real summary: {res}/{n} resolved "
          f"({round(100*res/n) if n else 0}%) ===", flush=True)
    for r in records:
        print(f"  {'OK ' if r.get('resolved') else 'XX '} {r['instance_id']:<26} "
              f"turns={r.get('turns','?'):<3} term={r.get('terminated','?'):<10} "
              f"{r.get('wall_s','?')}s", flush=True)
    summary = {"tag": tag, "preset": cfg.preset, "model": cfg.model, "gold": cfg.gold,
               "n": n, "resolved": res, "results": records}
    with open(os.path.join(cfg.work, f"swe_real-{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {cfg.work}/swe_real-{tag}.json", flush=True)


if __name__ == "__main__":
    main()
