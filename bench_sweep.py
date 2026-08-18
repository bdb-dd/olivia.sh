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


def build_prompt(base, prompt_tokens, idx):
    """Prompt for request `idx`, padded to ~prompt_tokens.

    The per-request marker goes FIRST and the filler is seeded from idx, so two
    concurrent requests diverge at token zero. That matters: vLLM v1 enables
    prefix caching BY DEFAULT (and several presets pass --enable-prefix-caching
    explicitly). With a shared prefix and the variant marker appended at the end
    — as this script used to do — requests 2..N would hit the cache and skip
    prefill almost entirely. At ~600 tokens that is noise; at 100K+ it makes
    prefill look free and turns TTFT and aggregate throughput into fiction.
    """
    if prompt_tokens <= 0:
        return f"(variant {idx}) {base}"
    head = f"Session {idx * 7919}, shard {idx}. "
    filler, i = [head], idx * 1000003
    approx_chars = prompt_tokens * 4
    while sum(len(x) for x in filler) < approx_chars:
        i += 1
        filler.append(f"Record {i}: node {i * 7 % 977} holds key {i * 31 % 4093} "
                      f"with fanout {i % 17 + 2} and depth {i % 5 + 1}. ")
    return ("".join(filler))[:approx_chars] + "\n\n" + base


def one_request(url, model, prompt, max_tokens, idx, out, ctk=None, timeout=900,
                prompt_tokens=0):
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": build_prompt(prompt, prompt_tokens, idx)}],
        "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
        "stream": True, "stream_options": {"include_usage": True},
    }
    # Optional chat-template kwargs (e.g. {"enable_thinking": false}) to toggle
    # reasoning per-request, overriding the server's --default-chat-template-kwargs.
    if ctk:
        body["chat_template_kwargs"] = ctk
    payload = json.dumps(body).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; toks = 0; ptoks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                    ptoks = chunk["usage"].get("prompt_tokens", ptoks)
        total = time.perf_counter() - t0
        out[idx] = {"ok": True, "ttft": ttft or total, "total": total,
                    "tokens": toks, "prompt_tokens": ptoks}
    except Exception as e:
        out[idx] = {"ok": False, "err": str(e), "total": time.perf_counter() - t0}


def run_level(url, model, prompt, max_tokens, concurrency, ctk=None, timeout=900,
              prompt_tokens=0):
    out = {}
    threads = [threading.Thread(target=one_request,
                                args=(url, model, prompt, max_tokens, i, out, ctk,
                                      timeout, prompt_tokens))
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
    med_ptok = statistics.median([r.get("prompt_tokens", 0) for r in ok]) if ok else 0
    print(f"{concurrency},{int(med_ptok)},{agg:.1f},{med_ps:.1f},{med_ttft:.2f},{p95_ttft:.2f},{fail},{total_tokens},{wall:.1f}",
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
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad the prompt to roughly this many tokens so CONTEXT LENGTH "
                         "becomes a measured axis. The built-in prompt is only ~35 "
                         "tokens, so an unpadded sweep measures throughput at ~600 "
                         "tokens of context and says nothing about long-context "
                         "behaviour. Padding is filler prose (~4 chars/token); the "
                         "ACTUAL prompt_tokens the server reports is emitted in the CSV, "
                         "so the approximation never has to be trusted.")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON dict passed as chat_template_kwargs, e.g. '
                         '\'{"enable_thinking": false}\' to disable reasoning')
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-request timeout (s); lower it for wedge-prone "
                         "presets (e.g. glm51) so a hung level fails fast")
    a = ap.parse_args()
    ctk = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    print("concurrency,prompt_tokens,agg_tok_s,median_per_stream_tok_s,median_ttft_s,p95_ttft_s,failures,total_tokens,wall_s",
          flush=True)
    for lvl in [int(x) for x in a.levels.split(",")]:
        run_level(a.url, a.model, a.prompt, a.max_tokens, lvl, ctk, a.timeout,
                  a.prompt_tokens)
