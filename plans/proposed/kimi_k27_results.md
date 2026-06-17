# Kimi K2.7-Code — CUDAGraph capture verdict + serving stress test

**Date:** 2026-06-15 · **Branch:** add-kimi-k26 · **Model:** `moonshotai/Kimi-K2.7-Code`
(1T MoE, native int4 compressed-tensors, same `KimiK25ForConditionalGeneration` arch as K2.6)
**Serving shape:** 2× GH200 nodes, TP=4 + PP=2, Ray MQ executor, `--max-model-len 131072`.

---

## 1. CUDAGraph capture campaign — VERDICT: exhausted (deploy eager)

**Goal:** recover CUDAGraph decode speedup (eager ~3–6× slower per stream). The blocker is a
`CUDA error: an illegal memory access` (`cudaErrorIllegalAddress`) raised **during CUDAGraph
capture**, inside a miscompiled inductor kernel:

```
_capture_cudagraphs (vllm/v1/worker/gpu_model_runner.py)
  -> torch/_inductor/runtime/triton_heuristics.py:1690 run
  -> autotune_to_one_config
RuntimeError: CUDA error: an illegal memory access was encountered
```

Two axes were explored to dodge it:

- **Config axis — exhausted (earlier).** `inductor_compile_config.combo_kernels=false`,
  `use_inductor_graph_partition`, `max_cudagraph_capture_size`, PIECEWISE vs FULL_AND_PIECEWISE —
  all still IMA. It is a miscompiled kernel, not a config knob.
- **Torch axis — exhausted (this round).** Hypothesis: a newer NGC PyTorch (newer inductor) might
  emit a correct kernel. Built vLLM 0.21 on **NGC 26.05** (`build_vllm_gh200.sh`, job 1260572,
  container `vllm-kimi-6-sandbox`, torch `2.12.0a0+…nv26.05`). Result (job 1261420, FAP):
  weights loaded, capture reached **30/51 PIECEWISE graphs**, then the **identical IMA**.
  **NGC 26.05's inductor does not fix it.**

**Conclusion:** Ship **eager** (`CUDAGRAPH_MODE=NONE`) for Kimi multi-node on GH200 — the proven,
stable config (8010 K2.6 and 8020 K2.7 both run eager). Remaining unexplored options, both
low-priority: (a) upstream bug report to vLLM/NVIDIA with the capture repro; (b) a non-NGC stable
torch wheel (risky — forfeits NGC's tuned GH200 kernels and CUDA 13.2 stack).

---

## 2. Build-infra findings (NGC 26.05) — fixed + to-codify

NGC 26.05 ships torch as a **local prerelease** (`2.12.0a0+5aff3928d8.nv26.05`). pip will not match
that against a normal `torch>=X` requirement (PEP 440 excludes pre-releases from `>=` ranges), so the
whole runtime-deps batch fails `ResolutionImpossible`. NGC 26.03 (`2.11.0a0`) happened not to trip it.

- **Build bug (fixed):** the deps install ended with `… 2>&1 | tail -30 || true`, which **silently
  swallowed the fatal ResolutionImpossible** and shipped a container that passed `import vllm` but
  could not serve (missing ray, transformers, compressed-tensors, …). Fix in `build_vllm_gh200.sh`:
  `compressed-tensors` moved to a separate `--no-deps` install; the `|| true` masking removed so the
  build now **aborts loudly** on a resolver failure (`set -euo pipefail` is active).

- **Still to codify in `build_vllm_gh200.sh`** (found while salvaging `vllm-kimi-6` so it could serve;
  these are transitive deps lost when packages are pulled to `--no-deps`):
  - `xgrammar` (vLLM 0.21 **hard-imports** it at startup) → install `xgrammar==0.2.2 --no-deps`
    **plus** its FFI runtime `apache-tvm-ffi>=0.1.9` (latest xgrammar wants `torch>=2.10.0` and hits
    the same prerelease wall; 0.2.2 wants only `torch>=1.10.0`).
  - `loguru` (imported by `vllm/model_executor/models/kimi_k25.py`) → add to the batch.
  - The pip "incompatible" notices (torch==2.11.0, flashinfer pin, numba, cudnn-frontend, setuptools)
    are vLLM's own metadata pins vs NGC's newer versions — **benign** for a `--no-deps` NGC build.
  - **FastAPI/Starlette incompatibility (vllm-kimi-6 dead end):** the fresh 26.05 batch resolved
    `fastapi 0.137.0` + a newer Starlette, and vLLM 0.21's API server then 500s **every** request with
    `'_IncludedRouter' object has no attribute 'path'`. So even with capture aside, `vllm-kimi-6`
    cannot serve. Combined with the capture IMA, the 26.05 container is abandoned. To make a 26.05
    build serviceable you'd also need to pin fastapi/starlette to the versions in the working
    `vllm-kimi-4` (NGC 26.03) container. **Not pursued** — eager on the proven container is the answer.

---

## 3. Stress + concurrency sweep (eager K2.7, deployable container vllm-kimi-4, job 1261695)

_(Run on `vllm-kimi-4` — the proven NGC-26.03 eager container that serves 8010/8020 — because the
26.05 build `vllm-kimi-6` can't serve, per §2. This is the deployable config, so these are the
numbers that matter.)_

_Methodology:_ concurrency sweep against the live server (login node → head:8000), streaming
chat/completions, fixed prompt + `max_tokens`. Metrics per level: aggregate output tok/s,
median per-stream tok/s, median TTFT, failures.

_Run 2026-06-15, 256 output tokens/request, distinct prompt per request (defeats prefix cache),
streaming, login-node load gen → head:8000._

| Concurrency | Aggregate tok/s | Per-stream tok/s | Failures |
|---|---|---|---|
| 1  | 16.9  | 16.9 | 0 |
| 2  | 37.0  | 18.6 | 0 |
| 4  | 78.0  | 19.5 | 0 |
| 8  | 134.6 | 16.8 | 0 |
| 16 | 267.3 | 16.7 | 0 |
| 32 | 602.6 | 18.8 | 0 |

- **TTFT** ≈ **1.8 s** single-stream (validated separately; chunks then arrive ~50–60 ms apart).
  The per-level TTFT from the sweep script was a measurement bug (didn't latch first token) — ignore it.
- **0 failures** at every level.

### Findings
- **Per-stream throughput is flat** (~17–19 tok/s) from 1→32 concurrency — it does **not** degrade
  under load. Eager decode is dominated by per-step overhead (kernel launches + cross-node PP bubble),
  not GPU compute, so adding requests amortizes that fixed cost essentially for free.
- **Aggregate scales near-linearly:** 16.9 → 602.6 tok/s = **35.7× at 32×** concurrency. More headroom
  above 32 — the live K2.6 (same arch, eager) sustains ~830 tok/s at 48 concurrent in production.
- **Scenario guidance:**
  - **Max total throughput:** push concurrency high — 32 → 602 tok/s, and the K2.6 production data
    shows ~830 tok/s at 48. Throughput keeps climbing until KV cache saturates (was only ~28–31% at 48).
  - **Balanced:** because per-stream stays ~18 tok/s regardless of load, there is **no real
    throughput-vs-latency trade-off** here — you can batch aggressively without hurting individual
    streams. For interactive use, 8–16 concurrent keeps TTFT/latency tight while already delivering
    135–267 tok/s aggregate.
- **Implication for CUDAGraph:** the flat-per-stream / overhead-bound profile is exactly what
  CUDAGraph would fix (it removes per-step launch overhead) — but capture IMAs (§1), so eager's
  per-step overhead is the price we currently pay. Single-stream ~17 tok/s is the eager ceiling.
