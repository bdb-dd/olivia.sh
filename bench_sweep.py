#!/usr/bin/env python3
"""Concurrency sweep for a vLLM OpenAI endpoint — stdlib only (threads + urllib,
streaming SSE), so it runs on a bare login node with no extra deps.

One CSV line per concurrency level:
  concurrency,agg_tok_s,median_per_stream_tok_s,median_ttft_s,p95_ttft_s,failures,total_tokens,wall_s

Usage:
  bench_sweep.py --url http://gpu-1-86:8000 --model RedHatAI/GLM-5.2-FP8 \
      --levels 1,2,4,8,16,32 --max-tokens 256
"""
import argparse, json, statistics, threading, time, urllib.request, urllib.error

PROMPT = ("Explain in depth how a B-tree database index works, including node "
          "splits, rebalancing on insert and delete, and why fan-out matters.")


def one_request(url, model, prompt, max_tokens, idx, out):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt} (variant {idx})"}],
        "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; toks = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                s = raw.decode("utf-8", "ignore").strip()
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                ch = chunk.get("choices") or []
                if ttft is None and ch:
                    d = ch[0].get("delta", {})
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        ttft = time.perf_counter() - t0
                if chunk.get("usage"):
                    toks = chunk["usage"].get("completion_tokens", toks)
        total = time.perf_counter() - t0
        out[idx] = {"ok": True, "ttft": ttft or total, "total": total, "tokens": toks}
    except Exception as e:
        out[idx] = {"ok": False, "err": str(e), "total": time.perf_counter() - t0}


def run_level(url, model, prompt, max_tokens, concurrency):
    out = {}
    threads = [threading.Thread(target=one_request,
                                args=(url, model, prompt, max_tokens, i, out))
               for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    res = [out.get(i, {"ok": False, "err": "missing"}) for i in range(concurrency)]
    ok = [r for r in res if r.get("ok")]
    fail = len(res) - len(ok)
    total_tokens = sum(r["tokens"] for r in ok)
    agg = total_tokens / wall if wall > 0 else 0
    ps = [r["tokens"] / r["total"] for r in ok if r["total"] > 0]
    ttfts = sorted(r["ttft"] for r in ok if r.get("ttft"))
    med_ps = statistics.median(ps) if ps else 0
    med_ttft = statistics.median(ttfts) if ttfts else 0
    p95_ttft = ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))] if ttfts else 0
    print(f"{concurrency},{agg:.1f},{med_ps:.1f},{med_ttft:.2f},{p95_ttft:.2f},{fail},{total_tokens},{wall:.1f}",
          flush=True)
    errs = [r.get("err") for r in res if not r.get("ok")]
    if errs:
        print(f"  # {fail} failures: {errs[:3]}", flush=True)
    return fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    print("concurrency,agg_tok_s,median_per_stream_tok_s,median_ttft_s,p95_ttft_s,failures,total_tokens,wall_s",
          flush=True)
    for lvl in [int(x) for x in a.levels.split(",")]:
        run_level(a.url, a.model, a.prompt, a.max_tokens, lvl)
