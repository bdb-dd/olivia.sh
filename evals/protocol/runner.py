"""Run the L0 protocol fixtures against a live ``anthropic_proxy.py`` endpoint
and produce a scorecard.

Separation of concerns mirrors :mod:`conformance`:

* **invariants** are aggregated into a single pass/total *gate* — any failure is
  a proxy regression and makes the run "dirty" (non-zero exit);
* **expectations** are aggregated into per-category *pass rates* — model
  behaviour, reported but never gating.

Each fixture runs in both ``nonstream`` and ``stream`` modes (the streaming path
additionally validates the SSE event grammar) and ``--repeat`` times, since
behavioural checks are stochastic.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional
import json
import os

from evals.protocol import client, conformance, loader


@dataclass
class RunConfig:
    base_url: str
    model: str
    preset: str = "generic"
    modes: tuple = ("nonstream", "stream")
    repeat: int = 1
    stall: float = 45.0
    timeout: float = 600.0
    concurrency: int = 1
    think: str = "default"          # "on" | "off" | "default" — gates requires_thinking
    default_max_tokens: int = 512


def _build_body(fx: loader.Fixture, mode: str, cfg: RunConfig) -> dict:
    req = fx.request
    body = {
        "model": cfg.model,
        "max_tokens": req.get("max_tokens", cfg.default_max_tokens),
        "temperature": req.get("temperature", 0),
        "messages": req.get("messages", []),
        "stream": (mode == "stream"),
    }
    if req.get("system"):
        body["system"] = req["system"]
    if req.get("tools"):
        body["tools"] = req["tools"]
    if "tool_choice" in req:
        body["tool_choice"] = req["tool_choice"]
    return body


def _run_one(fx: loader.Fixture, mode: str, run_idx: int, cfg: RunConfig) -> dict:
    rec = {"id": fx.id, "category": fx.category, "mode": mode, "run": run_idx,
           "ok": False, "error": None, "skipped": False,
           "latency_s": 0.0, "ttft_s": None,
           "invariants": [], "expects": []}

    body = _build_body(fx, mode, cfg)
    tools = fx.tools

    if mode == "stream":
        res = client.call_stream(cfg.base_url, body, stall=cfg.stall)
    else:
        res = client.call_nonstream(cfg.base_url, body, timeout=cfg.timeout)

    rec["latency_s"] = round(res.latency_s, 2)
    rec["ttft_s"] = round(res.ttft_s, 2) if res.ttft_s is not None else None

    if not res.ok:
        rec["error"] = res.error
        return rec

    if mode == "stream":
        sr = conformance.validate_sse(res.events or [])
        resp = sr.response
        checks = list(sr.checks)
        checks += conformance.check_invariants(resp, tools)
        checks += conformance.check_expectations(resp, tools, fx.expect)
    else:
        resp = res.response or {}
        checks = conformance.check_invariants(resp, tools)
        checks += conformance.check_expectations(resp, tools, fx.expect)

    rec["ok"] = True
    rec["invariants"] = [asdict(c) for c in checks if c.kind == "invariant"]
    rec["expects"] = [asdict(c) for c in checks if c.kind == "expect"]
    return rec


def run(cfg: RunConfig, fixtures: list, out_path: Optional[str] = None) -> dict:
    work = []
    skipped = []
    for fx in fixtures:
        if fx.requires_thinking and cfg.think != "on":
            skipped.append(fx.id)
            continue
        modes = tuple(fx.modes) if fx.modes else cfg.modes
        for mode in modes:
            for r in range(cfg.repeat):
                work.append((fx, mode, r))

    records: list = []
    if cfg.concurrency <= 1:
        for fx, mode, r in work:
            records.append(_run_one(fx, mode, r, cfg))
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futs = [pool.submit(_run_one, fx, mode, r, cfg) for fx, mode, r in work]
            records = [f.result() for f in futs]

    summary = _summarize(records, skipped, cfg)
    payload = {"config": asdict(cfg), "summary": summary, "records": records}
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summary["out_path"] = out_path
    _print_scorecard(summary, records, skipped, cfg)
    return summary


# --------------------------------------------------------------------------- #
# Aggregation + printing                                                       #
# --------------------------------------------------------------------------- #

def _summarize(records: list, skipped: list, cfg: RunConfig) -> dict:
    inv_pass = inv_total = 0
    inv_failures: list = []
    exp_by_cat: dict = {}        # category -> [pass, total]
    exp_by_fixture: dict = {}    # id -> [pass, total]
    errors: list = []

    for rec in records:
        if not rec["ok"]:
            errors.append({"id": rec["id"], "mode": rec["mode"],
                           "error": rec["error"]})
            continue
        for c in rec["invariants"]:
            inv_total += 1
            if c["passed"]:
                inv_pass += 1
            else:
                inv_failures.append({"id": rec["id"], "mode": rec["mode"],
                                     "name": c["name"], "detail": c["detail"]})
        cat = rec["category"]
        cp = exp_by_cat.setdefault(cat, [0, 0])
        fp = exp_by_fixture.setdefault(rec["id"], [0, 0])
        for c in rec["expects"]:
            cp[1] += 1
            fp[1] += 1
            if c["passed"]:
                cp[0] += 1
                fp[0] += 1

    # de-dup invariant failures (same fixture/mode/check across repeats)
    seen = set()
    uniq_fail = []
    for f in inv_failures:
        key = (f["id"], f["mode"], f["name"])
        if key not in seen:
            seen.add(key)
            uniq_fail.append(f)

    return {
        "preset": cfg.preset,
        "base_url": cfg.base_url,
        "modes": list(cfg.modes),
        "repeat": cfg.repeat,
        "n_calls": len(records),
        "n_errors": len(errors),
        "errors": errors,
        "invariants_pass": inv_pass,
        "invariants_total": inv_total,
        "invariants_clean": not uniq_fail and not errors,
        "invariant_failures": uniq_fail,
        "expect_by_category": exp_by_cat,
        "expect_by_fixture": exp_by_fixture,
        "skipped": skipped,
    }


def _bar(p, t):
    return f"{p}/{t}" + (f" {round(100 * p / t)}%" if t else "")


def _print_scorecard(summary: dict, records: list, skipped: list, cfg: RunConfig) -> None:
    print()
    print(f"L0 protocol conformance — preset={summary['preset']}  "
          f"base={summary['base_url']}  modes={','.join(summary['modes'])}  "
          f"repeat={summary['repeat']}")
    if skipped:
        print(f"  skipped {len(skipped)} (requires_thinking, --think!=on): "
              f"{', '.join(skipped)}")
    print()

    # per-fixture line
    by_fx = summary["expect_by_fixture"]
    inv_fail_ids = {(f["id"]) for f in summary["invariant_failures"]}
    for fid in sorted(by_fx):
        p, t = by_fx[fid]
        cat = next((r["category"] for r in records if r["id"] == fid), "")
        mark = "x" if (fid in inv_fail_ids or (t and p < t)) else "."
        print(f"  [{mark}] {fid:<34} {cat:<20} exp {_bar(p, t)}")

    print()
    print(f"INVARIANTS (proxy regression gate): "
          f"{_bar(summary['invariants_pass'], summary['invariants_total'])}  "
          f"-> {'CLEAN' if summary['invariants_clean'] else 'DIRTY'}")
    for f in summary["invariant_failures"]:
        print(f"  x [{f['id']}/{f['mode']}] {f['name']}  {f['detail']}")

    print()
    print("EXPECTATIONS by category (model behaviour, not gating):")
    for cat in sorted(summary["expect_by_category"]):
        p, t = summary["expect_by_category"][cat]
        print(f"  {cat:<24} {_bar(p, t)}")

    if summary["errors"]:
        print()
        print(f"ERRORS (transport / stall): {summary['n_errors']}")
        for e in summary["errors"]:
            print(f"  ! [{e['id']}/{e['mode']}] {e['error']}")

    if summary.get("out_path"):
        print(f"\nwrote {summary['out_path']}")
