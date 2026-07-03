# GLM-5.2 KV-tiering on GH200 — feasibility + plan (queue item (b))

**Goal:** bulk KV in coherent Grace LPDDR, hot pages in HBM → serve **many
concurrent / longer-context** sessions on **one node** (no multi-node PP wedge).
The handoff framed this as "LMCache", but the research below flips that: **vLLM's
native KV-offload is already in our container and is the right first move; LMCache
is a later, higher-risk option.**

Researched 2026-07-03 (no GPU — offline investigation of the deployed build + web).

---

## TL;DR / recommendation
1. **Start native, not LMCache.** Our pinned vLLM build (main `091386a`) ships a
   first-class CPU KV-offload path — `--kv-offloading-size <GiB>` +
   `--kv-offloading-backend` — with two registered specs: `CPUOffloadingSpec` and
   **`TieringOffloadingSpec`** (a real HBM↔CPU tiering manager). No dependency to
   stage, no external package. **This is the cheapest experiment that answers the
   gating question.**
2. **The gating question is arch compatibility, not availability.** Does the
   offload path initialize and actually move blocks for GLM-5.2's **DSA / sparse-MLA
   + `fp8_ds_mla` KV layout**? The scheduler is explicitly MLA/hybrid-aware (it
   reasons about "the MLA full-attention group" and "DeepSeek V4 (MLA + SWA
   groups)"), so there's a real chance it just works — but sparse-MLA + fp8 KV is
   bleeding-edge and must be verified on-cluster.
3. **Mind which capability you're buying** (see below) — the available native
   offload is **prefix-reuse tiering**, not single-sequence context-beyond-HBM.
4. **LMCache = later/maybe.** It's not installed (needs offline arm64 staging like
   runai), and its DeepSeek-V3.2/sparse-MLA support is **actively buggy** today
   (sglang #15739; CacheBlend-for-MLA still an open feature request). Don't lead
   with it.

---

## What's already in the container (verified in `vllm-glm52-1-sandbox`)
- **CLI:** `--kv-offloading-size` / `--kv-offloading-backend` (arg_utils.py:1170/1173)
  and `--kv-transfer-config` (:1492). Example format: `--kv-transfer-config
  '{"cpu_bytes_to_use": 80m}'`.
- **Registered offload specs** (`v1/kv_offload/factory.py`):
  - `CPUOffloadingSpec` (`v1/kv_offload/cpu/spec.py`)
  - **`TieringOffloadingSpec`** (`v1/kv_offload/tiering/spec.py`) — full module:
    `manager.py`, `async_lookup.py`, `fs`/`obj` backends. This is the HBM/CPU
    tiering engine.
- **Connector suite** (`distributed/kv_transfer/kv_connector/v1/`): native
  `offloading_connector.py` (`OffloadingConnector`, implements `SupportsHMA` —
  Heterogeneous-Memory-Architecture aware, promising for GH200 coherent memory),
  `simple_cpu_offload_connector.py`, plus shims for `lmcache_connector.py`,
  `lmcache_mp_connector.py`, `lmcache_integration/`, nixl, mooncake, moriio, hf3fs.
- **Scheduler is MLA/hybrid-arch-aware** (`offloading/scheduler.py`): handles
  full-attention vs sliding-window groups, **DeepSeek-V4 MLA+SWA**, and even
  EAGLE/MTP draft groups (excludes the volatile trailing draft block). GLM-5.2's
  `GlmMoeDsa` sits squarely in this design space.
- **LMCache package: NOT installed** (only the vLLM-side shim exists).

## Two capabilities — don't conflate them (this decides whether (b) is even the right lever)
| | **Prefix-reuse offload** (available now) | **Single-sequence layerwise offload** (RFC #33398, NOT built) |
|---|---|---|
| What it does | Offload computed KV blocks to CPU tier; reload on a **prefix hit** across requests/turns | Page an **active** sequence's KV between HBM/CPU **during decode** to fit context > HBM |
| Serves | Many sessions **sharing prefixes** / long multi-turn (skip recompute) | One (or few) session whose **resident** KV exceeds HBM |
| In our build? | **Yes** (`CPUOffloadingSpec`/`TieringOffloadingSpec`) | No — upstream RFC #33398, "proposal stage", **explicitly targets sparse-attn like DeepSeek-V3.2** (our family) but unimplemented |

Our (b) goal ("many concurrent long-context sessions, bulk KV in LPDDR") maps to
**prefix-reuse** *if* those sessions share history/prefixes (agentic multi-turn,
shared system prompt) — the available path helps there. If instead each stream
needs independent resident KV beyond HBM, that's RFC #33398 territory (unbuilt) or
the 2-node no-offload path (queue item (a)).

## The GH200 tailwind
The literature's KV-offload latency worry ("CPU→GPU transfer per layer must be ≤
that layer's forward time" — RFC #33398) is a **PCIe** worry. On GH200 the CPU
tier is **coherent Grace LPDDR over 900 GB/s C2C** — ~30× a PCIe4 x16 link — so the
constraint is far easier to meet here than in any published benchmark. The native
connector advertising `SupportsHMA` suggests upstream already models coherent
memory. **This is the reason KV-tiering is more attractive on Olivia than the
generic numbers imply.**

## ⚠️ The C2C-contention catch (connects (b) to (a) and (d))
Our single-node path **already offloads *weights*** to Grace (`--cpu-offload-gb`),
and we measured that **C2C weight-streaming is the decode bottleneck** (why capture
only got ~2.2×, not glm51's ~4.5×). Adding **KV** offload on the *same* single node
makes weights **and** KV pages share the one C2C link during decode → they will
contend, likely capping the KV-tiering win. **Implication:** the better home for
KV-tiering may be the **2-node, no-weight-offload** shape (weights fully in HBM →
C2C free entirely for KV traffic). So (b) and (a) are coupled: run (a) first to
free C2C, *then* layer KV-tiering on top. Worth deciding before spending a job.

## LMCache assessment (why it's not first)
- Not installed → needs the same offline arm64 staging dance as runai (login-node
  `pip download --platform manylinux2014_aarch64 --python-version 3.12`, unzip under
  a bound dir, import via `PYTHONPATH`). Non-trivial (LMCache has compiled bits).
- **Sparse-MLA support is immature:** "Failed to deploy DeepSeek-V3.2 with LMCache
  0.3.10" (sglang #15739); CacheBlend-for-MLA is an open feature request. GLM-5.2's
  DSA is the same V3.2 sparse-MLA family → high chance of the same breakage.
- Upside if it matures: CacheBlend (partial-prefix blending), cross-instance KV
  sharing, richer eviction — none of which we need for the first result.

## Step 0 (offline) — DONE 2026-07-03: exact schema + the job-saving gotcha
Read the deployed source. Concrete findings:
- **Flag:** `--kv-offloading-size <GiB>` is the whole switch. Backend enum is
  `Literal["native","lmcache"]`, default **`native`**. Offload is active only when
  `kv_offloading_size` is set. Size is the **total across TP ranks** (÷world_size=4
  per worker on our TP=4).
- **Auto-wiring** (`config/vllm.py:785`): `native` → builds
  `KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
  kv_connector_extra_config={"cpu_bytes_to_use": size<<30})`. (Set
  `VLLM_USE_SIMPLE_KV_OFFLOAD=1` → the simpler `SimpleCPUOffloadConnector` fallback.)
  `lmcache` → `LMCacheMPConnector` talking to a standalone LMCache server on
  `tcp://localhost:5555` (so LMCache needs a *separate server process*, not just a
  library import — more than the runai-style staging).
- **`CPUOffloadingSpec` is byte-level arch-agnostic at the worker:** it sizes the CPU
  pool from raw `kv_cache_tensors` bytes / block, no per-head assumptions — so MLA's
  compressed-latent KV blocks should move as opaque bytes. The **attention-group
  awareness lives in the scheduler** (handles MLA / SWA / DeepSeek-V4 / MTP groups),
  which is the part that must grok DSA's skip-topk layout.
- **⛔ JOB-SAVER — `expandable_segments` hard-conflict.** `run_vllm_server.sh:227`
  hard-sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and vLLM
  (`_verify_kv_transfer_compat`) **raises `ValueError` and refuses to start with ANY
  KV connector configured under that setting** unless the **cumem allocator** is on
  (CuMemAllocator disables expandable_segments around its pool). So the smoke test
  will crash at config-validation instantly unless we first either (i) unset
  expandable_segments for the run, or (ii) enable the cumem allocator. **Line 227 is
  an unconditional `export`, not override-respecting** — so step 1 needs a one-line
  prep edit: `export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"`
  then pass `PYTORCH_CUDA_ALLOC_CONF=` to disable. (Weigh: expandable_segments guards
  against fragmentation OOM on the tight-HBM offload path — dropping it is a real risk;
  the cumem route avoids that but adds sleep-mode machinery. Try unset first, watch for
  frag OOM.)

## Recommended experiment ladder (each is one GPU job; stop when the goal is met)
1. **Native CPU offload smoke test (cheapest, answers gating question).** Confirm it
   (a) inits without rejecting sparse-MLA/`fp8_ds_mla`, (b) actually stores/loads
   blocks (offload metrics move), (c) still decodes coherently. One-shot form (note
   the expandable_segments prep from step 0):
   ```bash
   # after making run_vllm_server.sh:227 override-respecting:
   PYTORCH_CUDA_ALLOC_CONF= \
   EXTRA_VLLM_ARGS='--kv-offloading-size 120' \
   CPU_OFFLOAD_GB=40 KV_CACHE_DTYPE=fp8_ds_mla MAX_MODEL_LEN=131072 \
   ENABLE_EXPERT_PARALLEL=0 SERVER_JOB_NAME=vllm-glm52kv \
   ./olivia.sh server start glm52_awq_1n
   ```
   (`--kv-offloading-size 120` = 120 GiB total CPU tier ≈ 30 GiB/worker of Grace
   LPDDR; comfortable next to the ~40 GiB/GPU weight offload already resident.)
   Workload: a **prefix-reuse** bench (shared long system prompt, many turns) — that's
   what this capability accelerates; measure cache-hit rate + TTFT on reuse. If it
   crashes on fp8-KV or the DSA indexer, retry with bf16 KV to localize the cause.
2. **If it engages:** sweep offload size, measure concurrent long sessions vs the
   no-offload baseline; add a `run_vllm_server.sh` knob (`KV_OFFLOAD_GB` → emits
   `--kv-offloading-size`) mirroring `CPU_OFFLOAD_GB`, forwarded by `olivia.sh`.
3. **If C2C contention caps it** (expected on single-node offload — KV pages share the
   link with streamed weights): re-run on the **2-node no-weight-offload** shape
   (needs (a) first) so C2C is KV-only.
4. **Only if a real gap remains** (need cross-request sharing / CacheBlend, native
   tiering insufficient): stand up the LMCache MP server + stage the client, retest,
   gate on its sparse-MLA bugs being fixed.

## Open questions to resolve on-cluster (step 1)
- Does `expandable_segments=` (unset) cause fragmentation OOM on the tight-HBM
  offload path, or is cumem-allocator the necessary route? (Decided by step 1.)
- Does the offload path accept **`fp8_ds_mla`** KV, or force bf16 KV? (bf16 halves
  the per-token benefit — same wall as elsewhere.)
- Does the scheduler hash/group **DSA's skip-topk indexer** state correctly, or only
  the MLA full-attention blocks? (The novel bit vs the DeepSeek-V4 code it was built for.)

## Status
Research + offline step 0 **DONE** (schema pinned, expandable_segments blocker found
→ step 1 is now one-shot after a 1-line script edit). **No code changes, nothing
built.** Next action (step 1) is GPU-gated — needs an allocation. See
[[project_glm52_status]] for the broader ledger.
