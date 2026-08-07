#!/usr/bin/env python3
"""L2-real via mini-swe-agent — drive the model with a *real* open agent harness
(SWE-agent's mini-swe-agent, ~74% on Verified, the exact class Laguna was trained
on) instead of our deliberately-minimal loop, on the SAME django instances + the
SAME gold-validated verify. Isolates the one variable that matters: agent quality.

mini-swe-agent uses bash-in-markdown (no tool-call API), so litellm talks straight
to vLLM's OpenAI endpoint — no anthropic_proxy needed. Runs with `LocalEnvironment`
*inside* our python:3.11 apptainer (no nested containers, no per-instance image
pulls); the agent edits the repo in place and we verify the resulting state.

Run (in the apptainer, with the miniswe-venv that has mini-swe-agent):
  OPENAI_API_BASE=http://<gpu>:8000/v1 OPENAI_API_KEY=x no_proxy=...,<gpu> \
    miniswe-venv/bin/python evals/swe_real/mini_runner.py \
      --model openai/poolside/Laguna-M.1-FP8 --preset laguna-mini --work /cluster/work/.../swe
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
import yaml  # noqa: E402  (mini-swe-agent dep)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evals.swe_real import runner as R  # noqa: E402  reuse setup + gold-validated verify

from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402


def load_swebench_yaml():
    path = os.path.join(str(builtin_config_dir), "benchmarks", "swebench.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run_instance(inst, repo, venv, cfg, swecfg):
    t0 = time.time()
    rec = {"instance_id": inst["instance_id"], "version": inst["version"]}
    try:
        env = R.reset_instance(repo, venv, inst["base_commit"])

        # mini-swe-agent runs bash via LocalEnvironment in OUR repo, venv on PATH.
        menv = dict((swecfg.get("environment") or {}).get("env") or {})
        menv.pop("BASH_ENV", None)  # docker-image-specific (conda activate testbed)
        menv["PATH"] = env["PATH"]
        menv["VIRTUAL_ENV"] = venv
        environment = LocalEnvironment(
            cwd=repo, env=menv,
            timeout=int((swecfg.get("environment") or {}).get("timeout", 60)))

        model = get_model(config={"model_name": cfg.model})

        agent_cfg = dict(swecfg.get("agent") or {})
        agent_cfg["step_limit"] = cfg.step_limit
        agent_cfg["cost_limit"] = 0.0  # local model has no litellm price
        if cfg.wall_cap:
            agent_cfg["wall_time_limit_seconds"] = cfg.wall_cap
        agent = DefaultAgent(model, environment, **agent_cfg)

        result = agent.run(inst["problem_statement"])
        rec["exit_status"] = result.get("exit_status")
        rec["steps"] = len(getattr(agent, "messages", []) or [])

        # the agent edited files in place; verify the repo state (our gold path)
        _, diff = R.sh("git diff", cwd=repo, env=env)
        rec["patch_size"] = len(diff)
        with open(os.path.join(repo, ".testpatch.diff"), "w") as f:
            f.write(inst["test_patch"])
        rc_tp, _ = R.sh("git apply -v .testpatch.diff", cwd=repo, env=env)
        rec["test_patch_applied"] = rc_tp == 0
        mods = R.test_modules_from_patch(inst["test_patch"])
        cmd = (f"{venv}/bin/python tests/runtests.py --settings=test_sqlite --verbosity 2 "
               + " ".join(mods))
        _, out = R.sh(cmd, cwd=repo, env=env, timeout=cfg.test_timeout)
        ok, detail = R.verify(out, inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"])
        rec["resolved"] = bool(ok and rec["test_patch_applied"])
        rec["verify"] = detail
    except Exception as e:  # noqa: BLE001
        rec["resolved"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="litellm model id, e.g. openai/poolside/Laguna-M.1-FP8")
    ap.add_argument("--preset", default="laguna-mini")
    ap.add_argument("--work", required=True)
    ap.add_argument("--instances", default=R.INSTANCES)
    ap.add_argument("--only", default=None)
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument("--wall-cap", type=int, default=900, help="per-instance agent wall-clock cap (s)")
    ap.add_argument("--test-timeout", type=float, default=1200.0)
    cfg = ap.parse_args()

    insts = [json.load(open(os.path.join(cfg.instances, f)))
             for f in sorted(os.listdir(cfg.instances)) if f.endswith(".json")]
    if cfg.only:
        want = {s.strip() for s in cfg.only.split(",")}
        insts = [i for i in insts if i["instance_id"] in want]
    resdir = os.path.join(cfg.work, f"results-{cfg.preset}")
    os.makedirs(resdir, exist_ok=True)
    swecfg = load_swebench_yaml()
    repo, venv = R.ensure_clone(cfg.work)

    print(f"L2-real via mini-swe-agent — preset={cfg.preset} model={cfg.model} "
          f"instances={len(insts)} step_limit={cfg.step_limit}", flush=True)
    records = []
    for i, inst in enumerate(insts, 1):
        rpath = os.path.join(resdir, inst["instance_id"] + ".json")
        if os.path.isfile(rpath):
            records.append(json.load(open(rpath)))
            print(f"[{i}/{len(insts)}] {inst['instance_id']} — cached "
                  f"({'RESOLVED' if records[-1].get('resolved') else 'unresolved'})", flush=True)
            continue
        print(f"[{i}/{len(insts)}] {inst['instance_id']} ({inst['version']}) — mini-swe-agent...", flush=True)
        rec = run_instance(inst, repo, venv, cfg, swecfg)
        with open(rpath, "w") as f:
            json.dump(rec, f, indent=2)
        records.append(rec)
        print(f"    -> {'RESOLVED' if rec.get('resolved') else 'UNRESOLVED'}  "
              f"patch={rec.get('patch_size','?')} exit={rec.get('exit_status','?')} "
              f"steps={rec.get('steps','?')} {rec.get('error','')} {rec.get('wall_s')}s", flush=True)

    n = len(records)
    res = sum(1 for r in records if r.get("resolved"))
    print(f"\n=== mini-swe-agent summary: {res}/{n} resolved "
          f"({round(100*res/n) if n else 0}%) ===", flush=True)
    for r in records:
        print(f"  {'OK ' if r.get('resolved') else 'XX '} {r['instance_id']:<24} "
              f"patch={r.get('patch_size','?'):<6} exit={r.get('exit_status','?')}", flush=True)
    with open(os.path.join(cfg.work, f"swe_real-{cfg.preset}.json"), "w") as f:
        json.dump({"preset": cfg.preset, "model": cfg.model, "harness": "mini-swe-agent",
                   "n": n, "resolved": res, "results": records}, f, indent=2)
    print(f"\nwrote {cfg.work}/swe_real-{cfg.preset}.json", flush=True)


if __name__ == "__main__":
    main()
