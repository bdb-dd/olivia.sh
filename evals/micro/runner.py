"""Run the L1 micro-agent task set against a live proxy and score it.

Each task runs in a fresh sandbox, ``--repeat`` times. The headline is
oracle-verified task success; alongside it we report behavioural signals the
strategy doc calls for: turn efficiency, thrash, premature-stop, runaway, and
per-turn tool-call validity (an L0 invariant break seen mid-loop).
"""
from dataclasses import asdict, dataclass, field
from statistics import median
import json
import os

from evals.micro import agent, sandbox


@dataclass
class RunConfig:
    base_url: str
    model: str
    preset: str = "generic"
    max_turns: int = 12
    repeat: int = 1
    bash_timeout: float = 30.0
    no_bash: bool = False
    keep_sandbox: bool = False


def _prepare_task(task: dict, cfg: RunConfig) -> dict:
    if cfg.no_bash:
        task = dict(task)
        task["tools"] = [t for t in (task.get("tools") or []) if t != "run_bash"]
    return task


def run(cfg: RunConfig, tasks: list, out_path=None) -> dict:
    records = []
    skipped = []
    for task in tasks:
        needs_bash = "run_bash" in (task.get("tools") or [])
        if cfg.no_bash and needs_bash and _task_requires_bash_oracle(task):
            skipped.append(task.get("id"))
            continue
        t = _prepare_task(task, cfg)
        for r in range(cfg.repeat):
            with sandbox.sandbox(task.get("files", {}), cleanup=not cfg.keep_sandbox) as sb:
                outcome = agent.run_task(
                    t, sb, base_url=cfg.base_url, model=cfg.model,
                    max_turns=cfg.max_turns, bash_timeout=cfg.bash_timeout)
            rec = asdict(outcome)
            rec["run"] = r
            rec["oracle"] = [{"name": o.name, "passed": o.passed, "detail": o.detail}
                             for o in outcome.oracle]
            records.append(rec)

    summary = _summarize(records, skipped, cfg)
    payload = {"config": asdict(cfg), "summary": summary, "records": records}
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summary["out_path"] = out_path
    _print_scorecard(summary, records, cfg)
    return summary


def _task_requires_bash_oracle(task: dict) -> bool:
    return any(o.get("check") in ("bash_exit_zero", "stdout_contains")
              for o in task.get("oracle", []))


def _summarize(records: list, skipped: list, cfg: RunConfig) -> dict:
    n = len(records)
    succ = [r for r in records if r["success"]]
    by_cat = {}
    for r in records:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[1] += 1
        if r["success"]:
            c[0] += 1
    return {
        "preset": cfg.preset,
        "base_url": cfg.base_url,
        "n_runs": n,
        "n_success": len(succ),
        "success_rate": round(len(succ) / n, 3) if n else 0.0,
        "median_turns_success": median([r["turns_used"] for r in succ]) if succ else None,
        "premature_stop": sum(r["premature_stop"] for r in records),
        "runaway_max_turns": sum(r["terminated"] == "max_turns" for r in records),
        "errors": sum(r["terminated"] == "error" for r in records),
        "total_thrash": sum(r["thrash_count"] for r in records),
        "invalid_tool_turns": sum(r["invalid_tool_turns"] for r in records),
        "by_category": by_cat,
        "skipped": skipped,
    }


def _print_scorecard(summary: dict, records: list, cfg: RunConfig) -> None:
    print()
    print(f"L1 micro-agent — preset={summary['preset']}  base={summary['base_url']}  "
          f"tasks={len(records)} runs  max_turns={cfg.max_turns}")
    if summary["skipped"]:
        print(f"  skipped (need bash, --no-bash set): {', '.join(summary['skipped'])}")
    print()
    print(f"  {'task':<26} {'category':<14} {'ok':<3} {'turns':<6} "
          f"{'calls':<6} {'thrash':<7} {'invalid':<8} {'term':<10} {'wall':<6}")
    for r in records:
        mark = "ok" if r["success"] else "XX"
        print(f"  {r['id']:<26} {r['category']:<14} {mark:<3} {r['turns_used']:<6} "
              f"{r['tool_calls']:<6} {r['thrash_count']:<7} {r['invalid_tool_turns']:<8} "
              f"{r['terminated']:<10} {r['wall_s']:<6}")
    s = summary
    print()
    print(f"SUCCESS: {s['n_success']}/{s['n_runs']}"
          f" ({round(100 * s['success_rate'])}%)   "
          f"median turns (solved): {s['median_turns_success']}")
    print(f"  premature-stop: {s['premature_stop']}   runaway(max_turns): {s['runaway_max_turns']}   "
          f"errors: {s['errors']}   thrash: {s['total_thrash']}   "
          f"invalid-tool-turns: {s['invalid_tool_turns']}")
    print("  by category: " + "  ".join(
        f"{c} {p}/{t}" for c, (p, t) in sorted(s["by_category"].items())))
    if summary.get("out_path"):
        print(f"\nwrote {summary['out_path']}")
