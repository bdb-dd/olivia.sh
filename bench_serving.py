#!/usr/bin/env python3
"""bench_serving.py — TTFT + decode-throughput benchmark for an OpenAI-compatible
chat endpoint (e.g. vLLM).

Streams responses to measure time-to-first-token (TTFT) and decode throughput
(tok/s), single-stream first, then at increasing concurrency to probe the
multi-node PP decode wedge (where a stream emits a few tokens then stalls at
0 tok/s). A per-token read-timeout (--stall) makes a wedged stream error fast
instead of hanging, so the probe terminates quickly.

Usage:
  python bench_serving.py                              # :8010, Kimi, conc 1,2,4
  python bench_serving.py --concurrency 1,2,4,8 --max-tokens 256
  python bench_serving.py --thinking                   # leave Kimi thinking on
"""
import argparse
import json
import statistics
import threading
import time

import requests

PROMPT = (
    "Write a detailed, technical ~300-word explanation of how pipeline "
    "parallelism differs from tensor parallelism in large language model "
    "inference. Be specific about communication patterns and trade-offs."
)


def fmt(x):
    return f"{x:.2f}s" if x is not None else "n/a"


def one_request(args, idx, out):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.thinking:
        # instant mode: measure pure answer decode, not reasoning
        body["chat_template_kwargs"] = {"thinking": False}

    t0 = time.time()
    ttft = None
    last = t0
    max_gap = 0.0
    completion_tokens = 0
    err = None
    try:
        # timeout=(connect, read): the read timeout is the *inter-token* budget,
        # so a stalled (wedged) stream raises ReadTimeout after --stall seconds.
        with requests.post(
            f"{args.base}/v1/chat/completions",
            json=body, stream=True, timeout=(10, args.stall),
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ch = chunk.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content") or ""
                    if piece:
                        now = time.time()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            max_gap = max(max_gap, now - last)
                        last = now
                u = chunk.get("usage")
                if u and u.get("completion_tokens"):
                    completion_tokens = u["completion_tokens"]
    except requests.exceptions.ReadTimeout:
        err = f"ReadTimeout: no token for >{args.stall:.0f}s (possible wedge)"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:100]}"

    decode_t = (last - t0 - ttft) if ttft else 0.0
    tok_s = (completion_tokens - 1) / decode_t if (decode_t > 0 and completion_tokens > 1) else 0.0
    out[idx] = {
        "err": err, "ttft": ttft, "tokens": completion_tokens,
        "wall": time.time() - t0, "tok_s": tok_s, "max_gap": max_gap,
        "stalled": bool(err and "ReadTimeout" in err),
    }


def run_level(args, conc):
    out = {}
    threads = [threading.Thread(target=one_request, args=(args, i, out)) for i in range(conc)]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    rs = [out[i] for i in range(conc)]
    print(f"\n=== concurrency {conc} ===")
    for i, r in enumerate(rs):
        if r["err"]:
            print(f"  req{i}: ERROR {r['err']}  (ttft={fmt(r['ttft'])}, got {r['tokens']} tok before stall)")
        else:
            print(f"  req{i}: ttft={fmt(r['ttft'])}  tokens={r['tokens']}  "
                  f"decode={r['tok_s']:.1f} tok/s  wall={r['wall']:.1f}s  max_gap={r['max_gap']:.2f}s")
    ok = [r for r in rs if not r["err"] and r["ttft"]]
    stalled = [r for r in rs if r["stalled"]]
    if ok:
        print(f"  -- {len(ok)}/{conc} ok | aggregate decode={sum(r['tok_s'] for r in ok):.1f} tok/s "
              f"| per-stream avg={statistics.mean(r['tok_s'] for r in ok):.1f} tok/s "
              f"| avg ttft={statistics.mean(r['ttft'] for r in ok):.2f}s | level wall={wall:.1f}s")
    if stalled:
        print(f"  !! WEDGE SUSPECTED: {len(stalled)}/{conc} streams stalled "
              f"(emitted then went silent >{args.stall:.0f}s) — the glm51-style PP decode wedge")
    return rs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8010")
    ap.add_argument("--model", default="moonshotai/Kimi-K2.6")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", default="1,2,4", help="comma list of concurrency levels")
    ap.add_argument("--runs", type=int, default=3, help="single-stream runs at conc=1 (run 0 is warmup)")
    ap.add_argument("--thinking", action="store_true", help="leave Kimi thinking mode on (default: off)")
    ap.add_argument("--stall", type=float, default=45.0, help="inter-token timeout (s) that flags a wedge")
    args = ap.parse_args()

    print(f"benchmark -> {args.base}  model={args.model}  max_tokens={args.max_tokens}  "
          f"thinking={'on' if args.thinking else 'off'}  stall={args.stall:.0f}s")
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    if 1 in levels:
        print(f"\n### single-stream ({args.runs} runs; run 0 = warmup) ###")
        for run in range(args.runs):
            out = {}
            one_request(args, 0, out)
            r = out[0]
            tag = " (warmup)" if run == 0 else ""
            if r["err"]:
                print(f"  run{run}: ERROR {r['err']}{tag}")
            else:
                print(f"  run{run}: ttft={fmt(r['ttft'])}  tokens={r['tokens']}  "
                      f"decode={r['tok_s']:.1f} tok/s  wall={r['wall']:.1f}s{tag}")
        levels = [lv for lv in levels if lv != 1]

    if levels:
        print("\n### concurrency / wedge probe ###")
        for conc in levels:
            run_level(args, conc)


if __name__ == "__main__":
    main()
