#!/usr/bin/env python3
"""Fetch + cache a slice of SWE-bench Verified instances for the L2-real runner.

Pulls instances from the HF datasets-server (no `datasets` dep needed — plain
HTTP), filters to a tractable slice (django on python-3.11-compatible versions),
and writes one JSON per instance to evals/swe_real/instances/.

Why django + 4.2/5.0: django is ~46% of Verified and the only big repo that's
pure-Python (no C builds) — fast `pip install -e .`. Versions 4.2 and 5.0 run on
python 3.11, so a single python:3.11 apptainer covers the whole slice (avoids the
per-instance python-version problem the official Docker harness bakes in).

Usage:
  python evals/swe_real/fetch_instances.py                 # default slice
  python evals/swe_real/fetch_instances.py --versions 4.2,5.0 --max 12
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

DATASET = "princeton-nlp/SWE-bench_Verified"
BASE = "https://datasets-server.huggingface.co/rows"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "instances")


def _ftp_count(v):
    if isinstance(v, list):
        return len(v)
    try:
        return len(json.loads(v))
    except Exception:
        return 0


def fetch_all_rows(split="test"):
    rows = []
    offset = 0
    while True:
        q = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": split,
            "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{BASE}?{q}", timeout=30) as r:
            d = json.loads(r.read())
        batch = [x["row"] for x in d.get("rows", [])]
        rows.extend(batch)
        total = d.get("num_rows_total", 0)
        offset += len(batch)
        if not batch or offset >= total:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="django/django")
    ap.add_argument("--versions", default="4.2,5.0",
                    help="comma list of repo versions to keep (python-3.11 compatible)")
    ap.add_argument("--max", type=int, default=12, help="cap instances (easiest first)")
    args = ap.parse_args()

    keep_versions = {v.strip() for v in args.versions.split(",")}
    print(f"fetching {DATASET} ...")
    rows = fetch_all_rows()
    print(f"  {len(rows)} total instances")

    sel = [r for r in rows if r["repo"] == args.repo and r["version"] in keep_versions]
    sel.sort(key=lambda r: (_ftp_count(r["FAIL_TO_PASS"]), _ftp_count(r["PASS_TO_PASS"])))
    sel = sel[:args.max]

    os.makedirs(OUT, exist_ok=True)
    # clear stale
    for f in os.listdir(OUT):
        if f.endswith(".json"):
            os.remove(os.path.join(OUT, f))
    for r in sel:
        # normalize list fields that the dataset stores as JSON strings
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if isinstance(r[k], str):
                r[k] = json.loads(r[k])
        with open(os.path.join(OUT, f"{r['instance_id']}.json"), "w") as f:
            json.dump(r, f, indent=2)

    print(f"\ncached {len(sel)} {args.repo} instances ({'/'.join(sorted(keep_versions))}) -> {OUT}")
    for r in sel:
        print(f"  {r['instance_id']:<26} ver={r['version']} "
              f"F2P={_ftp_count(r['FAIL_TO_PASS'])} P2P={_ftp_count(r['PASS_TO_PASS'])} "
              f"diff={r.get('difficulty','?')}")


if __name__ == "__main__":
    main()
