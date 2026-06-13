# Kimi K2.6 serving performance — re-enable CUDAGraph for single-stream decode

**Status:** proposed · **Created:** 2026-06-12 · **Branch:** `add-kimi-k26`

A fresh-context pickup doc. The Kimi K2.6 multi-node deployment works end to end,
but single-stream decode is slow (~10 tok/s) because we run **eager** (CUDAGraph
disabled). This plan is about getting CUDAGraph back on. None of this is urgent —
the server is fully usable for batched/multi-user work today.

---

## 1. Where things stand (so a cold session can orient)

- **What's deployed:** `vllm-kimi-1-sandbox` (vLLM `0.19.2.dev0`, NGC torch
  `2.11.0a0…nv26.03`), serving `moonshotai/Kimi-K2.6` on **2 nodes, TP=4 + PP=2**,
  **eager**. Started via `./olivia.sh server start kimi`.
- **Why it's the kimi preset's job:** `run_vllm_server.sh` defaults
  `CUDAGRAPH_MODE=NONE` when it detects Kimi (commit `62f2f32`), which emits
  `--compilation-config {"mode": "NONE"}` to vLLM. glm51 has the same eager
  workaround for its own reasons.
- **Reproduce a server + benchmark:** `./olivia.sh server start kimi`,
  `./olivia.sh server watch`, `./olivia.sh tunnel up` (or `LOCAL_PORT=8010`),
  then `python3 bench_serving.py` (committed: `2e78168`).

## 2. The problem (measured)

`bench_serving.py` against the eager server (instant mode, `max_tokens=256`):

| Concurrency | Per-stream | TTFT | Aggregate |
|---|---|---|---|
| 1 | 10.2 tok/s | 0.19s | 10.2 |
| 4 | 10.1 | 0.20s | 40.4 |
| 16 | 9.4 | 0.31s | 150.8 |

- **TTFT (prefill) is great** (~0.2s). The issue is purely **decode**: ~10 tok/s
  single-stream = ~100 ms/token. A 500-token reply ≈ 50s — sluggish for solo
  interactive use.
- **No wedge** at any concurrency (probed to 16) — unlike glm51. Throughput scales
  near-linearly, per-stream stays ~flat. **That flat-per-stream / linear-aggregate
  signature means decode is launch/overhead-bound, not compute- or bandwidth-bound**
  — the GPU has huge headroom at batch 1; it's just not kept busy.

## 3. Root cause

Two contributors, in order of how fixable they are:

1. **Eager mode (the big, fixable one).** With CUDAGraph/torch.compile off, every
   decode step re-issues the model's full sequence of kernel launches (hundreds,
   across 61 layers + attention + MoE routing) through Python → PyTorch dispatch →
   CUDA launch. That per-token CPU launch overhead dominates the step; the GPU
   idles between launches. CUDAGraph replays one captured graph per step instead —
   vLLM routinely gets **2–5× decode speedup** from it.
2. **Multi-node PP bubble at batch 1 (structural).** PP=2 hands each token from
   stage 0 (node A) → stage 1 (node B) over Slingshot. At batch 1 there are no
   other microbatches to fill the bubble, so one node idles while the other runs,
   plus per-token cross-node latency. Inherent to fitting a 1T model on 8 GPUs;
   largely hidden once there are concurrent users. CUDAGraph still helps the
   per-stage work.

## 4. Why CUDAGraph is currently off (the blocker to fix)

Turning compilation back on (`CUDAGRAPH_MODE` unset → vLLM default
`FULL_AND_PIECEWISE`, `mode=VLLM_COMPILE`) crashes at startup during the memory
**`profile_run`**:

```
TypeError: Multiple dispatch failed for 'torch._ops.vllm.min_latency_fused_qkv_a_proj.default';
all __torch_dispatch__ handlers returned NotImplemented
```
Trace: `gpu_model_runner.profile_run` → `_dummy_run` → `model()` →
`vllm/model_executor/models/kimi_k25.py:440 forward` → `vllm/compilation/cuda_graph.py`.

Interpretation: this Kimi MLA fused op (`vllm.min_latency_fused_qkv_a_proj` — the
q/kv latent down-projection) has **no fake/meta implementation**, so torch.compile's
FakeTensor tracing can't trace through it and aborts. It works fine in pure eager
(real-tensor dispatch), which is why `mode=NONE` runs.

## 5. Approaches (try in this order)

1. **Cheap experiment — `CUDAGRAPH_MODE=PIECEWISE`.** The crash was under the
   default `FULL_AND_PIECEWISE`. The compile config lists `unified_mla_attention`
   et al. as `splitting_ops`; PIECEWISE captures graphs *between* those splits. It
   probably still hits the same FakeTensor trace failure (PIECEWISE is still
   `VLLM_COMPILE`/inductor), but it's a one-command test:
   `CUDAGRAPH_MODE=PIECEWISE ./olivia.sh server start kimi` (CUDAGRAPH_MODE is
   forwarded by olivia.sh). If it boots, benchmark it.
2. **Register a fake/meta for the op (most likely the real fix).** Find where
   `vllm.min_latency_fused_qkv_a_proj` is registered (vLLM csrc / the kimi modeling
   code) and add a `torch.library.register_fake` (meta kernel) so compile can trace
   it. Can be done as a **container patch** in `build_vllm_gh200.sh` (like the other
   vLLM patches there), or by marking it an opaque custom op excluded from the
   compiled region. Then rebuild (`./olivia.sh build kimi --force`) and re-test.
3. **Check upstream / bump vLLM.** See whether a newer vLLM (0.20+/main) already
   ships a meta impl for this op or otherwise compiles Kimi. Kimi needs ≥0.19.1; a
   newer pin may also need a newer `NGC_PYTORCH_TAG` (see the comment block at the
   top of `build_vllm_gh200.sh`). Validate that the stable-ABI ops still build.
4. **Fallback — accept eager.** If none land soon, document ~10 tok/s as the known
   single-user latency floor; batched throughput (~150 tok/s @ 16) is already good.

## 6. Validation

For any approach: server must reach SERVING without the `Multiple dispatch failed`
error (watch the head log), then re-run `python3 bench_serving.py`. **Target:
single-stream decode ~25–50 tok/s** (the 2–5× CUDAGraph uplift), TTFT unchanged,
and **re-confirm no wedge** (`--concurrency 1,2,4,8,16`) since capture can change
multi-rank dispatch behavior (cf. `vllm#24689` full-CUDAGraph cross-rank wedge).

## 7. Notes / related

- **glm51 carries the same eager workaround** (its own CUDAGraph TODO). Kimi and
  glm51 share the multi-node TP=4+PP=2 infra. A fix here may or may not transfer
  (different model arch — Kimi MLA vs glm51 DSA).
- **Interesting datapoint:** Kimi does **not** wedge under concurrency, but glm51
  does (`Running≥2`). Same PP shape, different model. If anyone revisits the glm51
  wedge, that contrast is a clue (the wedge is model/arch-specific, not pure
  PP+Slingshot).
- Don't forget the storage split: glm51 weights live on `/cluster/work` now;
  Kimi on `/cluster/projects` (1 TiB quota). See the HF storage-split note.

## 8. References

- Commits (branch `add-kimi-k26`): `62f2f32` (eager default), `2e78168`
  (bench_serving.py), `5a4f0c5` (kimi preset).
- Files: `run_vllm_server.sh` (IS_KIMI block + CUDAGRAPH_MODE handling, ~line 260
  & ~620), `build_vllm_gh200.sh` (vLLM patch section, NGC tag comment), the failing
  op in vLLM `kimi_k25.py`.
- vLLM trackers worth checking: `vllm#24689` / PR `#25906` (full-CUDAGraph
  cross-rank dispatch), and search vLLM issues for `min_latency_fused_qkv_a_proj`
  / Kimi K2.6 compile.
