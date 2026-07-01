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
