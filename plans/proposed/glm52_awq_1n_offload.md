# GLM-5.2 AWQ single-node via Grace weight offload — characterization

**Goal:** find a working *single-node* (1×4 GH200, TP=4, **no PP**) serving config
for `cyankiwi/GLM-5.2-AWQ-INT4` by offloading part of each weight shard into
coherent Grace LPDDR5X (`--cpu-offload-gb`), freeing HBM for KV → long context.
The prize: **no multi-node PP decode wedge** and a 1-node allocation that schedules
easily. Preset: `glm52_awq_1n` (reuses the `glm52` container).

## Why offload is mandatory here
~415 GiB AWQ ÷ 4 GPUs ≈ **~104 GB/GPU**, over the 96 GB HBM *before any KV*. So a
single node cannot even load without `--cpu-offload-gb`. Rough per-GPU split
(total offloaded = 4 × CPU_OFFLOAD_GB):

| CPU_OFFLOAD_GB (/GPU) | resident weights /GPU | HBM freed for KV /GPU |
|---|---|---|
| 24 | ~80 GB | ~16 GB (tight) |
| 40 | ~64 GB | ~32 GB |
| 55 | ~49 GB | ~47 GB |

## Correction that reshapes the budget (vs the original sketch)
`--kv-cache-dtype fp8_e4m3` **fails on GH200 for GLM-5.2**: DSA needs a sparse-MLA
backend; the only sm_90 one is `FLASHMLA_SPARSE`, which rejects fp8 KV
(`No valid attention backend found … FLASHMLA_SPARSE: [kv_cache_dtype not supported]`
— the exact wall the FP8 preset hit). So **KV is bf16**, ~2× the sketch's per-seq
figure → expect ~1 concurrent 500K stream (not 3); 1M is single-stream/tight. The
startup line `GPU KV cache size: N tokens` is ground truth.

### BUT — `fp8_ds_mla` may unlock fp8 KV on GH200 (high-value, untried)
Only *standard* `fp8_e4m3` is dead on Hopper (a hard SM90 CUTLASS GMMA limit —
vLLM #27604 closed "not planned"; the fp8-capable `FLASHINFER_MLA_SPARSE` is
Blackwell-only). But the current vLLM support matrix shows **`FLASHMLA_SPARSE`
(SM90–SM100) accepts KV dtype `fp8_ds_mla`** — a DeepSeek-specific fp8 MLA KV
format — on Hopper. Our FP8 preset only ever tried `fp8_e4m3`, so this was never
tested. Confirmed present in our container's `flashmla_sparse.py` + `config/cache.py`.
GLM-5.2's DSA already routes through `FLASHMLA_SPARSE` on GH200, so if it accepts
`fp8_ds_mla` the KV ~halves → **doubles the budget above**, and it retro-improves
the existing FP8 3-node preset AND the 2-node AWQ preset. `KV_CACHE_DTYPE` is a
forwarded knob → test `KV_CACHE_DTYPE=fp8_ds_mla` early.

## Method (stepwise: safest first, one variable at a time)
Container `vllm-glm52-1-sandbox` (already built, DeepGEMM 88965b0 for the DSA
indexer). Eager (IS_GLM52 default). Single stream unless noted. Record the
`GPU KV cache size: N tokens` line + single-stream decode tok/s for every run.

1. **Composition smoke test** — does single-node DSA + offload even stand up?
   `CPU_OFFLOAD_GB=40 MAX_MODEL_LEN=131072 ENABLE_EXPERT_PARALLEL=0` @ util 0.90.
   Low KV pressure so any OOM points at weights/offload, not KV.
   - OOM at load → raise offload (55). DSA/offload incompatible → try EP on / debug.
1b. **fp8 KV probe (high value)** — same as (1) + `KV_CACHE_DTYPE=fp8_ds_mla`. Does
   the DSA/FLASHMLA_SPARSE path accept fp8 KV on GH200? If yes, KV ~halves for every
   GLM-5.2 config. Check the model still decodes coherently (fp8 KV can hurt quality).
   **DE-RISKED 2026-07-02 (source read of our container):** `flashmla_sparse.py` lists
   `fp8_ds_mla` (+ `"fp8"` alias) in `supported_kv_cache_dtypes`, has a full FP8
   prefill/decode path, and `get_supported_head_sizes()==[576]` = GLM-5.2's MLA layout;
   `config/cache.py` accepts `fp8_ds_mla` as a CLI value. So acceptance/serving is
   high-confidence; open unknowns are only scale metadata (fp8_ds_mla is self-scaling,
   likely no `--calculate-kv-scales` needed) and numerical coherence.
2. **Push context** — `MAX_MODEL_LEN=512000`, sweep `CPU_OFFLOAD_GB ∈ {40, 55}`,
   util 0.95. Read KV tokens; confirm N ≥ max-model-len for ≥1 stream.
3. **Offload→latency curve** — at a fixed 131072, sweep `CPU_OFFLOAD_GB ∈ {24,40,55}`,
   measure single-stream decode tok/s (the C2C streaming tax).
4. **Expert-parallel variable** — repeat the best point with `ENABLE_EXPERT_PARALLEL=1`.
5. **1M stretch** — `MAX_MODEL_LEN=1024000`, offload ~55–60, `--max-num-seqs 1–2`.

Launch (per-branch deploy key `add-glm52-awq`; `server start` does NOT auto-deploy):
```bash
./olivia.sh server deploy
CPU_OFFLOAD_GB=40 MAX_MODEL_LEN=131072 ENABLE_EXPERT_PARALLEL=0 \
  ./olivia.sh server start glm52_awq_1n --no-tail
./olivia.sh server watch
```

## Results
| # | offload/GPU | max_model_len | EP | KV dtype | util | GPU KV tokens | single-stream tok/s | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 131072 | off | bf16 | 0.90 | — | — | **FAILED @ engine init: MTP layer-78 weights absent from checkpoint** (deepseek_mtp.py:480) — MTP had auto-enabled. ~90 min burned (load + drafter). |
| 2 | 40 | 131072 | off | bf16 | 0.90 | _tbd_ | _tbd_ | _CANCELED (job 1427786) — paused sweep overnight; pivoted to the MTP graft below_ |
| 3 | 40 | 131072 | off | bf16 | 0.90 | **173,568** | **~6.8** | **SERVES ✅ (job 1456591, 2026-07-02).** Weights 65.68 GiB/GPU, KV 14.97 GiB/GPU. Single-node offload path validated end-to-end. |

| 4 | 40 | 131072 | off | **fp8_ds_mla** | 0.90 | **284,032** | **~6.8** | **SERVES ✅ + COHERENT (job 1461175).** 1.64× KV vs bf16 (fp8_ds_mla ≈ 0.58 bytes/tok of bf16). Reasoning+answers correct. Decode SAME speed → KV isn't the bottleneck; benefit is capacity. → ~256K single-stream / 2× 131K. |

### fp8_ds_mla validated (Run 4) — cross-cutting win
`--kv-cache-dtype fp8_ds_mla` **works on GH200 for GLM-5.2** (FLASHMLA_SPARSE), serves + stays
coherent, 1.64× KV. No `--calculate-kv-scales` needed. Benefit is **KV capacity, not decode
speed** (eager+offload is the decode bottleneck, not KV bandwidth). **Also applies to the FP8
3-node `glm52` and 2-node `glm52_awq` presets** → free context/concurrency headroom there too.

| 5 | 55 | 393216 | off | fp8_ds_mla | 0.90 | **587,200** | **~6.3** | **SERVES ✅ — single-node ~500K CLEARED (job 1462694).** KV 29.5 GiB/GPU (2× offload-40), weights 52.65 GiB/GPU. 587K > 512K → one 500K stream fits (~15% headroom). Decode slightly slower (heavier offload). |

### HEADLINE: single-node ~500K achieved
`fp8_ds_mla` + `offload 55` → **587K-token KV pool > 500K**, so a single 500K-context stream serves
on 1 node (4×GH200, no PP wedge). Ladder: bf16/off40=173K(~131K) → fp8/off40=284K(~256K) →
fp8/off55=587K(~500K). **1M single-stream is impractical single-node** (would need offload ~75+ →
heavy tax, resident weights ~29 GB/GPU) — better via 2-node or KV-tiering (LMCache). Decode is
~6–7 tok/s eager across all configs (offload/eager bound, not KV) → **MTP is the speed lever (Run 6).**

### Budget analysis from Run 3 (bf16 baseline)
- **173.5K KV tokens** at offload-40/bf16/util-0.90 → fits ~131K single-stream comfortably, but
  **cannot fit even one 500K stream** (needs ~3×) nor 256K (needs ~1.5×). Confirms the "bf16 →
  ~1 short stream" prediction.
- Rough model: resident weights/GPU ≈ 103.75 − offload_GB; KV/GPU ≈ 86·util_frac − resident − ~5.
  Levers to reach 500K single-stream: **fp8_ds_mla** (~2× KV, Run 2/1b), **offload 55–60** (~2–2.5×),
  **util 0.95** (marginal). 500K likely needs fp8_ds_mla **+** offload ~55–60 (compounding), at the
  cost of a heavier C2C decode tax. Decode is already slow (~6.8 tok/s eager single-stream).

### Observations — job 1424649 (2026-07-01)
- **Offload composes end-to-end.** `UVAOffloader`, `Total CPU offloaded parameters: 40.36`/GPU,
  `Using FLASHMLA_SPARSE attention backend`, MoE = `CompressedTensorsWNA16MarlinMoEMethod`
  (MARLIN WNA16 — the cyankiwi repo is compressed-tensors WNA16, not classic AWQ, so the
  FLASHINFER_MOE_FP16 env swap is a no-op here; harmless).
- **Cold load is the pain: main model = 3208 s (~53 min)** from Lustre (`/cluster/projects`),
  single-stream ~131 MB/s, auto-prefetch skipped (410 GiB > 90% of 269 GiB RAM). Every cold
  start pays this. Mitigations to explore: stage to node-local NVMe, or more load parallelism.
- **MTP auto-enabled unintentionally** (PP=1 bypassed the PP>1 disable guard) → the MTP draft
  (`DeepSeekMTPModel`) loaded as a SECOND full 83-shard pass (`Loading drafter model...`),
  ~doubling the load. FIXED: GLM-5.2 MTP is now opt-in (`ENABLE_SPECULATIVE=1`); the rest of
  the sweep runs MTP-off. (This run kept MTP to salvage the allocation + get a first datapoint.)

## MTP re-enabled via graft (dnhkng) — checkpoint BUILT (2026-07-01, no GPU)
cyankiwi/GLM-5.2-AWQ-INT4 dropped the layer-78 MTP weights, so MTP was impossible on the
plain repo. `dnhkng/GLM-5.2-AWQ-INT4-FP8-MTP-delta` is **tooling (27 KB, no weights)**: a graft
script that extracts the FP8 `model.layers.78.*` head from GLM-5.2-FP8 and merges it onto the
AWQ body, plus a vLLM patch so layer 78 loads as FP8 while the body stays AWQ.

**Built (all login-node, no GPU):**
- **Merged checkpoint:** `/cluster/projects/nn10104k/models/GLM-5.2-AWQ-INT4-MTP-FP8`. Extracted
  1569 layer-78 tensors from our LOCAL `RedHatAI/GLM-5.2-FP8` (`--extract-local-mtp`, no download;
  RedHatAI ≡ zai-org). config gains `mtp_quantization_config` + `num_nextn_predict_layers=1`;
  index gains the 1569 entries. Presents as **419.5 GiB** but costs only **9.4 GiB** new disk
  (base shards symlinked to the persistent AWQ cache). Tooling+venv at `/cluster/projects/nn10104k/.mtp-graft/`.
- **GRAFT GOTCHA (fixed):** the script's `hardlink` mode `os.link`'d the HF snapshot's *symlinks*
  (relative `../../blobs/...`) → **90 broken links** in the out-dir. Fixed by repointing each to the
  AWQ snapshot's absolute path (0 broken now, all shards resolve). Re-graft must redo this fix.
- **vLLM patch** (`patches/vllm-awq-fp8-mtp-quant-config.patch`, vendored): only `deepseek_mtp.py`
  + `deepseek_v2.py` (pure Python). **Applies cleanly to `091386a`** (0 rejects) and is a **safe
  no-op** for non-MTP checkpoints (`_get_mtp_quant_config`→None when `mtp_quantization_config` absent),
  so the FP8 preset + plain AWQ are unaffected → live-patchable, no rebuild.
- **Preset `glm52_awq_mtp`** (local-path model, served `glm52-awq-mtp`, 1×4 TP=4). MTP auto-enables
  because the path contains "MTP".

**LOCAL-PATH BIND GOTCHA (fixed 2026-07-02, job 1463473→1463561):** serving the local-dir
checkpoint failed FAST at `create_engine_config`→`maybe_override_with_speculators`→
`get_config_dict(path)` → `HFValidationError: Repo id must be in the form 'namespace/repo_name'`.
Cause: the merged dir (`/cluster/projects/nn10104k/models/...`) is NOT in the container bind list
(only HF_HOME + CONTAINER_DIR), so inside the job `os.path.isdir(model)` is false → transformers
treats it as a HF repo id. FIX: run_vllm_server.sh now auto-binds a local-dir MODEL
(`--bind $MODEL:$MODEL`; its symlinks resolve into the already-bound HF_HOME). Verified: job 1463561
cleared config + loaded 86 shards (83 base + 3 FP8-MTP).

**MTP loads & composes (job 1463561, 2026-07-02): the graft+patch+bind WORK end-to-end** — full
2-pass load (main + drafter, ~80 min), no layer-78 error, `speculative-config {method:mtp,
num_speculative_tokens:3}` accepted. It then failed only on a KV-budget check: the MTP drafter
eats ~5 GiB HBM (KV 14.97→10.09 GiB at offload40), and bf16 KV @131072 needs 11.46 GiB → est. max
115328. **Not fundamental** — fix by fp8_ds_mla (halves KV), or lower max_model_len, or util 0.95.
Relaunched as **job 1464356 = MTP + fp8_ds_mla + offload40 + 131072** (the production-optimal
combo; fp8 KV ~199K capacity at the MTP-reduced 10 GiB → 131K fits).

**To serve (next GPU session):**
```bash
# 1. live-patch the shared glm52 container (pure-Python; no rebuild; no-op for FP8/plain-AWQ)
SB=/cluster/work/projects/nn10104k/containers/vllm-glm52-1-sandbox
cd $SB/usr/local/lib/python3.12/dist-packages && \
  patch -p1 < <deployed>/patches/vllm-awq-fp8-mtp-quant-config.patch && \
  find vllm/model_executor/models \( -name 'deepseek_mtp*.pyc' -o -name 'deepseek_v2*.pyc' \) -delete
# 2. serve (MTP auto-on via 'MTP' in path; offload auto-defaults 40)
CPU_OFFLOAD_GB=40 MAX_MODEL_LEN=131072 ./olivia.sh server start glm52_awq_mtp
```
(For reproducibility, also fold the patch into the build via a local-patch hook — TODO.)

### MTP THROUGHPUT — VALIDATED (job 1464356, 2026-07-02): MTP+fp8_ds_mla serves + speeds decode
Config: `glm52_awq_mtp` (grafted checkpoint), offload 40, fp8_ds_mla, 131072, eager, MTP num_spec=3.
KV pool 184,128 tok (fp8 gave the headroom bf16 lacked → 131K fits with the MTP drafter's ~5 GiB HBM;
weights 68.91 GiB/GPU incl. drafter). ~79 min 2-pass load. `bench_sweep.py` (temp 1.0, max-tokens 128):

Two `bench_sweep.py` runs (temp 1.0, max-tokens 128): first COLD (compile/DeepGEMM caches building,
TTFT ~2.5s), second WARM to conc-64 (steady state, TTFT ~0.4s). Warm is the representative table:

| concurrency | agg tok/s | per-stream tok/s | TTFT s | fail | (cold agg) |
|---|---|---|---|---|---|
| 1  | 14.8  | 14.8 | 0.39 | 0 | 11.9 |
| 2  | 28.9  | 15.2 | 0.67 | 0 | 26.4 |
| 4  | 46.5  | 12.0 | 0.81 | 0 | 43.1 |
| 8  | 67.0  | 8.9  | 0.88 | 0 | 65.4 |
| 16 | 96.8  | 6.4  | 1.11 | 0 | 84.0 |
| 32 | 130.5 | 4.3  | 1.33 | 0 | 111.8 |
| 64 | **201.8** | 3.4 | 3.60 | 0 | — |

**HEADLINE: MTP single-stream 11.9 (cold) → 14.8 (warm) tok/s vs the ~6.8 tok/s MTP-off baseline =
~1.75–2.2×.** The graft (dnhkng) + FP8-MTP quant patch + local-model bind all pay off. **Aggregate
scales cleanly to ~202 tok/s @64-way and is STILL CLIMBING (no saturation), 0 failures at every level
1→64** (single-node = no PP wedge — the multi-node presets can't do this). Per-stream degrades under
load (14.8→3.4) and TTFT rises (0.4→3.6s; eager+offload saturating) — capture would lift this but IMAs
on this stack. Served name defaulted to the local PATH (SERVED_MODEL_NAME not applied for local-path
presets — minor: set it explicitly if routing by name matters).

### CUDAGraph CAPTURE — WORKS on this path; CAPTURE+MTP is the peak config (2026-07-03)
(Supersedes the "capture IMAs on this stack" note above — that was untested for the single-node AWQ+DSA
path.) Eager is NOT forced here: `CAPTURE_EXPERIMENT=1` (PIECEWISE + `combo_kernels=false` + NCCL
all-reduce) captures cleanly in ~40s, 0 IMA (jobs 1466375, 1469123, 1469631). But capture only amortizes
kernel-launch overhead — the dominant single-node cost is the C2C weight-streaming (offload) tax — so
it's ~2×, NOT the ~4.5× of glm51. Warm single-stream / aggregate@64 (fp8_ds_mla, offload40, 131K):

| config | single-stream tok/s | agg@64 tok/s |
|---|---|---|
| eager (baseline) | 6.8 | — |
| capture only | 15.1 (~2.2×) | 157.9 |
| MTP only | 14.8 (~2.2×) | 201.8 |
| **capture + MTP** | **21.8 (~3.2×)** | **241.3** |

**Capture + MTP PARTIALLY STACK** (job 1469631): capture composes with MTP (captures in 36s with the
drafter loaded, KV pool 175,936), and single-stream 21.8 beats either lever alone (~15) by ~1.45× —
different overheads (launch vs forward-passes) partly add, the shared C2C ceiling caps it below a
fully-additive 4×. c64 shows the combo still ahead (241 vs 202 vs 158), 0 fail 1→64. **Peak single-node
config = capture + MTP: ~22 tok/s single-stream / ~241 @64.** Untested: Option B (low-offload, no-MTP)
to attack the C2C tax that bounds all of these. Cost: MTP double-load ~84 min (main ~56 + drafter ~29).

## Option 1 investigation — FlashInfer-MLA on sm_90 — CLOSED (dead end)
**Finding (2026-07-01):** `FLASHINFER_MLA_SPARSE` (the fp8-capable sparse-MLA backend)
is Blackwell-only by a HARD kernel gap, not a liftable vLLM guard, so there is no
cheap unlock on GH200:
- vLLM `platforms/cuda.py::_get_backend_priorities`: for `use_mla` + `device_capability.major==10`
  (Blackwell) the sparse list has BOTH FlashInfer+FlashMLA (FlashInfer preferred for fp8 KV);
  the `else` (Hopper, major==9) branch lists **`FLASHMLA_SPARSE` only**. FlashInfer-MLA-sparse
  is structurally absent on sm_90.
- FlashInfer #1466: `trtllm_batch_decode_with_kv_cache_mla` → RuntimeError **"Unsupported
  architecture" on H200 (SM90)**. FlashInfer #3111: HiSparse sparse attn (V3.2 / GLM) is a
  **Blackwell** feature. So even patching the vLLM gate would just hit a runtime kernel error.
- TRT-LLM = same wall (it's the source of the trtllm-gen MLA kernels; Blackwell-only) AND
  won't have GLM-5.2's brand-new DSA arch. Dynamo = serving/orchestration layer, adds no
  kernels → relevant to KV tiering (option 2), not fp8 kernels.
- **Redirect:** the real sm_90 fp8-KV lever is `fp8_ds_mla` via **FlashMLA** (FlashMLA's own
  fp8 sparse kernel; DeepSeek-V3.2 used it on Hopper). Already wired as sweep **step 1b** —
  that's where fp8-KV effort goes, NOT porting FlashInfer to Hopper.

## Later options (parked — see chat)
2. Inverse tiering: bulk KV in LPDDR, hot pages in HBM (LMCache vLLM connector — keeps
   GLM-5.2 support; on GH200 the "CPU" tier IS coherent Grace, so paging is fast).
   The right architecture for many long sparse-attention sessions. Maturity w/ MLA = risk.
3. Others: 2-node `glm52_awq` already has ~350 GB aggregate KV WITHOUT offload (only cost
   = PP wedge + serialize) — may already serve 500K–1M multi-stream; A/B it vs 1-node.
   NVFP4 out (Blackwell). Check any INT8 KV the sparse backend accepts.

## Load-time — Run:ai Model Streamer — DONE (2026-07-03), now the GLM-5.2 default
The single-node offload path's worst UX cost was **cold load: ~56 min** for the
~415 GiB AWQ checkpoint (MTP double-loads the drafter → ~84 min). Diagnosis: the
load is **vLLM-pipeline-bound, not I/O-bound** — raw Lustre read is ~1.3 GB/s, and
page-cache reuse was a dead end (the bottleneck is vLLM's serial per-shard
processing, not the disk read).

**Fix: `LOAD_FORMAT=runai_streamer`** (the Run:ai Model Streamer, a parallel
safetensors loader). Job 1472308 (`vllm-glm52runai`, glm52_awq_1n + offload40 +
fp8_ds_mla + 131K on gpu-1-108):
- Loads at ~450–600 it/s → **`Model loading took 66.38 GiB and 569.2 s` (~9.5 min)** —
  **~6× faster** than the default loader. Confirms the pipeline-bound diagnosis.
- **Serves + decodes cleanly**: `Application startup complete`; two decode probes
  returned coherent output (`"The capital of Norway is Oslo."`, `finish_reason=stop`)
  with the reasoning-token patch reporting `reasoning_tokens` correctly. KV cache
  264,128 tokens (2.02× @131K), eager (capture IMAs on this stack, as expected).

**Packaging (durable):** runai ships as a pip package that can't be installed
offline into the container. It's staged under the persistent HF_HOME and imported
via PYTHONPATH — no container modification:
- aarch64 wheel + `humanize` extracted to `/cluster/projects/nn10104k/huggingface/runai-pkg/`
  (bound in the container because it's under HF_HOME; loaded via `CONTAINER_PYTHONPATH`).
- Source wheels cached at `/cluster/projects/nn10104k/.mtp-graft/runai-wheels/`.
- Re-stage from a login node: `pip download runai-model-streamer --no-deps
  --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12
  --abi cp312 --implementation cp`, unzip under HF_HOME/runai-pkg.

**Wired as default:** `run_vllm_server.sh` auto-sets `LOAD_FORMAT=runai_streamer`
+ `CONTAINER_PYTHONPATH=$HF_HOME/runai-pkg` for GLM-5.2 when that staged pkg
exists (guarded `-d "$HF_HOME/runai-pkg"`, so it no-ops when absent — e.g. the
work-tier FP8 `glm52` where the pkg isn't staged). Fully overridable. `olivia.sh`
forwards both env vars. Applies to the AWQ projects-tier path today; to enable it
for the FP8 `glm52` too, stage the pkg on the work tier as well.
