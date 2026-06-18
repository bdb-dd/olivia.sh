# GLM-5.2 (glm52) — performance & concurrency sweep

**Date:** 2026-06-18 · **Model:** `RedHatAI/GLM-5.2-FP8` (744B FP8, ~40B active, DSA)
**Deploy:** 3 nodes × 4 GH200 (TP=4 + PP=3), vLLM main + PR#45895, NGC 26.05,
DeepGEMM 88965b0, **RayExecutorV2 engine-as-actor**, **eager** (`CUDAGRAPH_MODE=NONE`).
**Tool:** `bench_sweep.py` (stdlib threads + streaming SSE), prompt = B-tree explainer,
`max_tokens=256`, `temperature=1.0`, `top_p=0.95`, thinking on (default).

| concurrency | agg tok/s | per-stream tok/s | median TTFT s | failures |
|---|---|---|---|---|
| 1  | 5.6   | 5.6 | 13.9 | 0 |
| 2  | 11.1  | 5.6 | 14.2 | 0 |
| 4  | 22.5  | 5.6 | 14.9 | 0 |
| 8  | 43.1  | 5.4 | 16.5 | 0 |
| 16 | 81.1  | 5.1 | 18.8 | 0 |
| 32 | 130.7 | 4.1 | 16.5 | 0 |
| 48 | 224.5 | 4.7 | 18.5 | 0 |
| 64 | 419.0 | 6.6 | 6.2  | 0 |

## Takeaways
- **Stable concurrency: ZERO failures 1→64.** No decode wedge. This is the payoff of
  the V2/engine-as-actor path — glm51's legacy executor wedges at `Running≥2`; glm52
  batches cleanly. (Tested at `max_tokens=256`; long-context concurrency is KV-bound —
  the engine reports ~3.5x max concurrency at the full 131K context.)
- **Aggregate throughput scales hard:** ~5.6 → ~420 tok/s (≈75×). Good batched/multi-user
  server.
- **Per-stream holds ~4–7 tok/s** across all levels (no collapse under load).
- **Weak spots:** single-stream is **slow (~5.6 tok/s end-to-end, eager)** and **TTFT is
  high (~14 s single-stream — PP=3 pipeline + prefill)**. CUDAGraph capture would likely
  lift decode speed but is untried (likely IMAs on NGC 26.05 per the Kimi findings).
- **Reliability caveat — flaky multi-node init:** 1 of 3 cold starts (job 1305748) loaded
  all weights then HUNG on a cross-node sync (GPUs at 0% util, silent; workers split across
  two Slingshot segments 10.63.0.x / 10.63.1.x). A `scancel` + restart recovered. Likely
  NCCL multi-node init flakiness — candidate fix: NCCL env tuning / node-placement. Watch
  on cold start; `olivia.sh server watch` + the hang signature (weights loaded, 0% util,
  no "Application startup complete") flags it.

## Repro
```bash
TIME_LIMIT=3:00:00 HF_HOME=/cluster/work/projects/nn10104k/huggingface \
  CUDAGRAPH_MODE=NONE ./olivia.sh server start glm52 -d        # baked recipe
# once /health is 200 on the head node, warm up one request (DeepGEMM JIT), then:
python3.12 bench_sweep.py --url http://<head>:8000 --model RedHatAI/GLM-5.2-FP8 \
  --levels 1,2,4,8,16,32,48,64 --max-tokens 256
```
