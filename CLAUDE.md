# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for building and running vLLM on NVIDIA GH200 (GraceHopper) ARM64 GPUs on HPC clusters. The key challenge is preserving NGC's custom PyTorch build while installing vLLM and its dependencies.

## Local Python Setup

Local tooling (`anthropic_proxy.py`, `chat_devstral.py`) runs on your workstation, not inside the cluster container, and needs its own Python deps. Pick either path — both land on the same `.venv/` (gitignored).

**With mise** (auto-activated venv, recommended if you already use it):
```bash
mise trust       # one-time: approve mise.toml in this repo
mise install     # installs the pinned Python version
mise run install # pip install -r requirements.txt into .venv
```
After that, `cd`ing into the directory auto-activates the venv — no `source` needed.

**Without mise** (plain venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Activate with `source .venv/bin/activate` before running `python anthropic_proxy.py …` or `python chat_devstral.py …`.

Cluster-side scripts (`build_vllm_gh200.sh`, `run_vllm_server.sh`, `vllm_proxy.py`) use the Singularity container's Python and are unaffected by either path.

## Olivia CLI (`olivia.sh`)

Unified CLI for managing vLLM on an HPC cluster. Uses SSH ControlMaster for single 2FA authentication per session.

### Quick Start
```bash
./olivia.sh                    # Show help
./olivia.sh chat               # Connect to vLLM and start chat
./olivia.sh status             # Check cluster and connection status
```

### Modules

#### Chat Module
```bash
./olivia.sh chat               # Connect and start interactive chat
./olivia.sh chat --port 9000   # Use different local port
./olivia.sh chat --tunnel-only # Just set up tunnel, don't start chat
./olivia.sh chat --no-stream   # Disable streaming
```

**Note:** The chat module verifies a vLLM server is running before connecting. If no server is found, it will prompt you to start one.

#### Build Module
```bash
./olivia.sh build              # Show build help
./olivia.sh build --presets    # List available model presets
./olivia.sh build --list       # List existing containers on cluster
./olivia.sh build logs         # Tail logs of a currently running build job

# Build containers (deploys script and submits SLURM job)
./olivia.sh build glm51_v19    # Build GLM-5.1 container on vLLM v0.19.0 (alias: glm51)
./olivia.sh build glm51_v20    # Build GLM-5.1 container on vLLM v0.20.0 (quarantined — same wedge)
./olivia.sh build glm47        # Build GLM-4.7 container
./olivia.sh build kimi         # Build Kimi K2.6 container on vLLM v0.19.1
./olivia.sh build devstral     # Build Devstral container
./olivia.sh build llama        # Build Llama container

# Build options
./olivia.sh build glm47 --index 2    # Build second container (safe, won't touch existing)
./olivia.sh build glm47 --force      # Rebuild existing container (use with caution!)
./olivia.sh build glm47 --sif        # Create SIF image after build
./olivia.sh build glm47 --vllm v0.6.6  # Override vLLM version
./olivia.sh build glm47 --no-tail    # Don't tail logs after submitting
```

**Safety:** By default, builds will **fail** if a container with the same name already exists. Use `--index N` to create a new container, or `--force` to explicitly overwrite.

#### Server Module
```bash
./olivia.sh server                   # Show server help
./olivia.sh server list              # List available containers
./olivia.sh server status            # Show running server status

# Start a server (uses preset with default model)
./olivia.sh server start glm51_v19   # Start GLM-5.1 server on vLLM v0.19.0 (alias: glm51; 8 GPUs across 2 nodes, TP=4 + PP=2)
./olivia.sh server start glm51_v20   # Start GLM-5.1 server on vLLM v0.20.0 (quarantined — same wedge)
./olivia.sh server start glm47       # Start GLM-4.7 server
./olivia.sh server start kimi        # Start Kimi K2.6 server (8 GPUs across 2 nodes, TP=4 + PP=2)
./olivia.sh server start devstral    # Start Devstral server
./olivia.sh server start llama       # Start Llama server
./olivia.sh server start qwen        # Start Qwen server

# Start with options
./olivia.sh server start glm47 --index 2              # Use vllm-glm47-2-sandbox
./olivia.sh server start glm47 --model custom/model   # Override default model
./olivia.sh server start -c vllm-custom-1-sandbox -m my/model  # Explicit container

# Monitoring
./olivia.sh server watch             # Smart monitor: GPU loading -> health -> ready

# Other actions
./olivia.sh server restart glm47     # Cancel running job and restart
./olivia.sh server restart glm47 -d  # Deploy script and restart
./olivia.sh server logs              # Tail logs of running server
./olivia.sh server ssh               # Open shell on GPU node
./olivia.sh server cancel            # Cancel running vLLM job
./olivia.sh server deploy            # Upload run_vllm_server.sh to cluster
```

**Watch command phases:**
1. **WAITING** - Waits for SLURM job to be submitted
2. **PENDING** - Job queued, waiting for resources
3. **LOADING** - GPU memory increasing as weights load (progress bar)
4. **INIT** - Weights loaded, checking /health endpoint
5. **SERVING** - Live throughput monitoring (tok/s, active requests, KV cache, poll interval)

**Watch command features:**
- **Runs indefinitely** - Continues monitoring even after server is ready; automatically recovers if job stops
- **Desktop notifications** (macOS/Linux) - Notifies when:
  - Job transitions from PENDING → RUNNING
  - Server becomes READY
  - Job stops or fails
- **Linear back-off** - Poll interval starts at 3s, increases by 3s per iteration when idle, caps at 60s. Resets to 3s when activity is detected

**Server presets** (with default models):
| Preset | Default Model | GPUs | Notes |
|--------|---------------|------|-------|
| `glm51_v19` (alias `glm51`) | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | TP=4 + PP=2, vLLM v0.19.0, container index 1. **Defaults to PIECEWISE CUDAGraph capture** (NGC-26.03 rebuild): de-wedged via NCCL all-reduce, ~22 tok/s/stream, 0-fail 1→64. The old serialization workaround is no longer needed. |
| `glm51_v20` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | Same as glm51_v19 but on vLLM v0.20.0 + RayExecutorV2, container index 2. **Quarantined** — same wedge as v0.19.0; kept for diagnostic work only. |
| `glm52` | `RedHatAI/GLM-5.2-FP8` | 12 (3 nodes × 4) | TP=4 + PP=3. Block-FP8 (~755 GB) — does **not** fit 8 GPUs, hence 3 nodes. **Needs vLLM main `091386a` (pinned) + PR#45895** (new skip-topk DSA indexer); not in any release. Same multi-node PP wedge as glm51 → proxy serialization. fp8 KV cache + DeepGEMM (`VLLM_DEEP_GEMM_WARMUP=skip`). |
| `glm52_awq` | `cyankiwi/GLM-5.2-AWQ-INT4` | **8 (2 nodes × 4)** | TP=4 + **PP=2**. AWQ/compressed-tensors INT4 (~411 GB) fits 8 GPUs, so it costs **8 GPU-h/hour instead of 12**. Same DSA skip-topk indexer as the FP8, so it needs PR#45895 → runs on the **glm53** (vLLM v0.27.1) container where that is native. Weights live on the **persistent** tier and are already cached. Auto-applies `VLLM_PP_LAYER_PARTITION=38,40` (see below). **Not yet served.** |
| `glm53` | `zai-org/GLM-5.3-FP8` | 12 (3 nodes × 4) | GLM-5.2's **same base, re-post-trained** — identical arch, so it **reuses the glm52 container** (index 1, no rebuild) and inherits the whole 5.2 runtime profile. **Weights not public yet** (Z.ai promised them ~2 weeks after the 2026-08-14 launch); confirm the repo id when they land. |
| `glm53_v27` | `zai-org/GLM-5.3-FP8` | 12 (3 nodes × 4) | Same model on its **own** `vllm-glm53-1` container built from **vLLM v0.27.1 + NGC 26.07**, no PR graft. The upgrade path off glm52's pinned-main build — **unvalidated**. |
| `dsv41` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | **4 (1 node)** | **552B backbone + 196B Engram**, only 8B active per prompt token / 16B per output token. Arch `DeepseekV41ForCausalLM`, native from vLLM v0.30.0 -> **reuses the glm53flash container, no build**. vLLM sizes it at ~614 GB VRAM, which our 384 GB node cannot hold -- so `--engram-config {"cpu_offload":true}` moves the sparsely-accessed Engram tables (259.5 GiB, ~47 GiB/rank) into **pinned Grace host memory** and the cards hold only the backbone. **fp8 KV required** (inverse of GLM-5.3-Flash). ✅ **SERVES + BENCHED 2026-11-03** (job 2697324): 112 tok/s single, peak **2454 tok/s @64 streams**, decode flat across a 120x context range, 0 failures. Backend is **`FLASHMLA_SPARSE_DSV41`** (compiled into vLLM) — NOT the flashinfer variant; flashinfer is not needed. KV measured **1.08 GiB/rank = 229,815 tokens**, so the recipe's 890 B/token does not reconcile (~3.4-5 KB/token observed; the pool also funds the indexer, SWA and compressor caches). |
| `glm53_flash` | `zai-org/GLM-5.3-Flash` | **4 (1 node)** | ⚠️ **A DIFFERENT MODEL from `glm53`, not a variant of it.** 320B total / 18B active, new arch `Glm5NextForConditionalGeneration`, new pretrain, natively multimodal. 34 KDA linear-attn + 11 NoPE sparse-MLA layers, 288 experts, mHC, native FP8 ~306 GiB, 1M ctx, MIT. **One node at TP=4.** Own container from **source: vLLM v0.32.0 + NGC 26.07** (`glm5_next` is native from v0.30.0). **Hopper: BF16 KV only.** |
| `glm53_flash_prebuilt` | `zai-org/GLM-5.3-Flash` | 4 (1 node) | Same model from the **prebuilt** `vllm/vllm-openai:glm53-flash` image, container index 2. **Fallback only** — the primary path while `glm5_next` was in no public release; use if the source build regresses. |
| `glm52_awq` (aliases `glm-5.2-awq`, `glm52awq`) | `cyankiwi/GLM-5.2-AWQ-INT4` | 8 (2 nodes × 4) | TP=4 + PP=2. **AWQ-INT4 (~415 GiB)** — fits 8×GH200 (the preferred path over 3-node FP8), **persistent** (projects) tier. **Reuses the glm52 container** (same vLLM main + PR#45895 + DeepGEMM — the DSA indexer needs are architectural, not FP8-specific); build `glm52`, serve `glm52_awq`. Same GLM-5.2 sparse-MLA → **bf16 KV** (no fp8 KV on GH200), eager. **Untested on-cluster:** MoE-FP16 kernel + the PP=2 indexer-aligned layer partition (`38,40`) need first-serve verification; multi-node PP wedge applies → proxy serialization. |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | TP=4 + PP=2, native int4, MLA, multimodal, vLLM v0.19.1 |
| `laguna` | `poolside/Laguna-M.1-FP8` | 4 | TP=4, single node. FP8 (~225 GB), dense full attention (FLASH_ATTN), CUDAGraph capture on. vLLM v0.21.0, `poolside_v1` parsers. |
| `ornith` | `ornith-ai/Ornith-1.5-397B-FP8` | 8 (2 nodes × 4) | **Ornith 1.5 397B flagship** (org renamed from `deepreinforce-ai`). TP=4 + PP=2, FP8 W8A8 (418 GB). Qwen3.5 hybrid-attn MoE, 256K ctx (1M via `ORNITH_YARN_FACTOR=4`). **Runs on the `glm53flash` container (vLLM v0.32.1) — no build**; qwen3_5_moe is native since 0.19.1. **1.5's FP8 ships the MTP head** (1.0's did not) — spec decode is opt-in (`ENABLE_SPECULATIVE=1`, `ORNITH_SPEC_METHOD=mtp\|dflash`) |
| `ornith_1n` | `ornith-ai/Ornith-1.5-397B-FP8` | **4 (1 node)** | The same 397B on **one node**: 418 GB exceeds 384 GB of HBM, so `CPU_OFFLOAD_GB=20` (default, measured best) parks the excess in Grace LPDDR5X — restricted to the routed experts via `CPU_OFFLOAD_PARAMS=experts` (`--cpu-offload-params`, vLLM ≥0.32). The dsv41 lesson, refined. Half the GPUs of `ornith`, no cross-node PP. **Measured: 59 tok/s single-stream, 299 @64** |
| `ornith_122b` | `ornith-ai/Ornith-1.5-122B-FP8` | 4 (1 node) | **New in 1.5:** 122B-A10B, 48 layers, 256 experts top-8, FP8 W8A8 (133 GB, persistent tier), TP=4. **MTP on by default** (192 tok/s single, 3,300 @64). 1M via `ORNITH_YARN_FACTOR=4` + `MAX_MODEL_LEN=1048576` — ⚠️ NOT the model card's top-level `rope_scaling` override, which is a silent no-op (needle-verified). **No DFlash drafter** for this size |
| `ornith_122b_1m` | `ornith-ai/Ornith-1.5-122B-FP8` | 4 (1 node) | **1M-token window** (`MAX_MODEL_LEN=1048576 ORNITH_YARN_FACTOR=4`), **MTP on** (the 122B default). ✅ Needle-verified with MTP at 1,036,709 tokens (job 2851670) and without it (job 2759667); 3.85× @1M, 196 tok/s @1K. `ENABLE_SPECULATIVE=0` restores the MTP-off config |
| `ornith_1m` | `ornith-ai/Ornith-1.5-397B-FP8` | 8 (2 nodes × 4) | The 2-node 397B with the same 1M window. ⚠️ **Not yet verified on the 397B** (the override is shape-agnostic; the 4.42M-token pool fits ~4 × 1M) |
| `ornith_1n_lowlat` | `ornith-ai/Ornith-1.5-397B-FP8` | 4 (1 node) | Single-user latency shape of `ornith_1n`: MTP + `CPU_OFFLOAD_GB=22`. 75.6 tok/s single, but a 315K-token pool and 229 @64 |
| `ornith_gh200` | `deepreinforce-ai/Ornith-1.0-35B-FP8` | **1** | **35B on a single GH200 card** (TP=1). FP8 W8A8 (~35 GB, ~3B active), hybrid linear+full attn (auto backend), multimodal, 256K ctx — max single-user throughput. MTP off (no head in FP8). Still points at the (purge-hollowed) 1.0 `ornith` container — not moved to 1.5, `qwen3_xml`/`qwen3` parsers |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | TP=4 |

**Per-preset GPU allocation:** `olivia.sh server start` reads resources from `get_preset_resources()` in `olivia.sh` and passes corresponding `--nodes`, `--gpus-per-node`, `--cpus-per-task` overrides to `sbatch`. `glm51` (2 nodes × 4) and `glm52`/`glm53` (3 nodes × 4) cross node boundaries; everything else runs single-node 4 GPUs. The `cluster` snapshot reports start-time/feasibility for 1-, 2-, and 3-node × 4-GPU shapes.

### Preset names — `<family>-<version>-<params>-<ctx>[-Nn][-variant]` (since 2026-11-12)

Every preset name says what it serves: model family and version, parameter count,
the default max context (`32k` = 32768, `128k` = 131072, `256k` = 262144, `1m` =
1048576), then **`-Nn` = number of nodes** on every multi-node preset and on a
single-node shape of a model that also has a multi-node one (`…-397b-256k-2n` vs
`…-1n`). A behaviour variant adds a word (`-lowlat`, `-mtp`, `-prebuilt`,
`-vllm020`); a context variant changes `ctx` instead (`ornith-1.5-122b-1m`). The
name is derived from the preset's card fields in `presets.json` and
**`presets.py lint` enforces it** (run by `check_scripts.sh`), so it cannot drift.

**Every previous name still works** — old canonical names and all old aliases are
aliases now, so `./olivia.sh server start ornith_122b` and a router request for
`model: "kimi27"` behave exactly as before. Older sections of this file use the
old names. `./olivia.sh server card <preset>` prints a preset's model card (no SSH),
and `server start` prints it before submitting.

| New name | Was | Status |
|---|---|---|
| `glm-5.1-744b-128k-2n` | `glm51_v19` | benchmarked 2026-06-20 |
| `glm-5.1-744b-128k-2n-vllm020` | `glm51_v20` | quarantined (same PP wedge as v0.19) |
| `glm-5.2-744b-128k-3n` | `glm52` | benchmarked 2026-06-18 |
| `glm-5.2-744b-128k-2n` | `glm52_awq` | never served on 2 nodes (container proven by the 1-node shape, glm-5.2-744b-128k-1n) |
| `glm-5.3-744b-128k-3n` | `glm53` | not yet run (weights gated/unpublished) |
| `glm-5.3-744b-128k-3n-vllm027` | `glm53_v27` | not yet run (upgrade-path container) |
| `glm-5.3-flash-320b-128k` | `glm53_flash` | benchmarked 2026-11-06 (job 2732602) |
| `glm-5.3-flash-320b-128k-prebuilt` | `glm53_flash_prebuilt` | fallback image |
| `borealis-27b-128k` | `borealis` | benchmarked 2026-08-18 |
| `deepseek-4-flash-304b-128k` | `dsv4flash` | BLOCKED on this stack (2026-08-19) |
| `deepseek-4.1-flash-748b-128k` | `dsv41` | benchmarked + tuned 2026-11-06 |
| `deepseek-4.1-flash-748b-1m` | `dsv41_1m` | benchmarked 2026-11-06 (job 2731395) |
| `deepseek-4.1-flash-748b-128k-lowlat` | `dsv41_lowlat` | benchmarked 2026-11-06 (job 2731404) |
| `laguna-s-2.1-118b-128k` | `lagunas21` | benchmarked 2026-08-18 |
| `qwen-3.8-27b-256k` | `qwen38` | benchmarked 2026-08-18 |
| `glm-4.7-358b-32k` | `glm47` | no current benchmark |
| `kimi-k2.6-1t-128k-2n` | `kimi` | benchmarked 2026-06-20 (eager) |
| `kimi-k2.7-code-1t-128k-2n` | `kimi27` | benchmarked 2026-06-20 (eager) |
| `laguna-m.1-225b-128k` | `laguna` | benchmarked 2026-06-20 |
| `ornith-1.5-397b-256k-2n` | `ornith` | benchmarked 2026-11-09 (job 2777625) |
| `ornith-1.5-397b-1m-2n` | `ornith_1m` | NOT VERIFIED on the 397B (override needle-verified on the 122B) |
| `ornith-1.5-397b-256k-1n` | `ornith_1n` | benchmarked 2026-11-08 (job 2760309) |
| `ornith-1.5-397b-256k-1n-lowlat` | `ornith_1n_lowlat` | benchmarked 2026-11-08 (job 2766587) |
| `ornith-1.5-122b-256k` | `ornith_122b` | benchmarked 2026-11-08 (jobs 2754478, 2754533) |
| `ornith-1.5-122b-1m` | `ornith_122b_1m` | needle-verified with MTP 2026-11-12 (job 2851670) |
| `ornith-1.0-35b-256k` | `ornith_gh200` | benchmarked 2026-07-16 — its 1.0 container is purge-hollowed |
| `devstral-2-123b-32k` | `devstral` | no current benchmark |
| `llama-3.3-70b-32k` | `llama` | no current benchmark |
| `qwen-2.5-72b-32k` | `qwen` | no current benchmark |
| `glm-5.2-744b-128k-1n` | `glm52_awq_1n` | benchmarked 2026-07-03 |
| `glm-5.2-744b-128k-1n-mtp` | `glm52_awq_mtp` | experimental (local MTP-grafted checkpoint) |

#### Prefetch Module
```bash
./olivia.sh prefetch glm51_v19                  # preset -> its default repo
./olivia.sh prefetch cyankiwi/GLM-5.1-AWQ-4bit  # explicit HuggingFace repo id
./olivia.sh prefetch glm47 --revision <sha>     # pin a revision
./olivia.sh prefetch <model> --no-follow        # start, don't tail progress
```

Downloads model weights into the HuggingFace cache **from a login
node** — never a GPU/SLURM job, so the multi-hundred-GB transfer doesn't burn allocation. The
download is **detached** (`setsid`, survives closing the CLI) and **resumable** (skips
already-complete blobs). Run it once before `server start` so the server job serves
immediately instead of downloading inside the allocation.

Mechanism notes:
- The Olivia login node is **amd64** while the GH200 compute nodes are **arm64**, so the
  containers can't exec on the login node. Prefetch instead builds a one-time throwaway
  `python3.12` venv (under `<HF_HOME parent>/.prefetch/venv`) with `huggingface_hub` + `hf_xet`.
- **The cache tier is per-preset.** `prefetch`/`server start` resolve `HF_HOME` from the
  preset's `storage` field in `presets.json`: `projects` (persistent, the default) or `work`
  (the at-risk `/cluster/work` mirror, for models that don't fit the 1 TiB project quota —
  kimi/kimi27/glm51/glm52). You never hand-pass the override; both commands land on the same
  tier the server reads from, so weights aren't re-downloaded to the wrong place. An explicit
  repo id resolves through the same registry, and an explicit `HF_HOME=…` on the command line
  still overrides. See [[project_hf_storage_split]].
- `HF_HOME` (the persistent/projects root) is pinned explicitly from your local environment (see
  "Persistent model cache" under Container/Cache Structure); the `work` root is derived from it
  by swapping the `/cluster/projects/<proj>` prefix for `/cluster/work/projects/<proj>`.
  `HF_TOKEN` is forwarded over stdin for gated repos.

#### Tunnel Module
```bash
./olivia.sh tunnel             # Show tunnel status
./olivia.sh tunnel up          # Open tunnel to vLLM server
./olivia.sh tunnel refresh     # Re-point tunnel after the job moved nodes
./olivia.sh tunnel down        # Close tunnel
./olivia.sh reconnect          # Re-auth after a drop and restore the tunnel
```

**Connection durability (mitigating 2FA fragility):**
Olivia requires password + OTP on every *new* interactive SSH connection (SSH
keys do not bypass 2FA). ControlMaster pays that 2FA once and multiplexes; the
key to avoiding re-auth is keeping the master alive. The master is opened
detached (`ssh -f -N -M`) with a long `ControlPersist` and keepalives, so a
single 2FA covers a long idle window and survives the terminal closing.

- `SSH_CONTROL_PERSIST` (default `12h`) — how long the master persists when idle
- `SSH_ALIVE_INTERVAL` / `SSH_ALIVE_COUNT` — keepalive to prevent NAT/idle drops
- After a genuine drop, `./olivia.sh reconnect` re-auths (one OTP) and restores
  any recorded tunnel. Unattended auto-reconnect (autossh) is intentionally not
  used because a fresh connection needs an interactive OTP.

> For a *durable* endpoint that survives GPU job restarts (the old `--login-proxy`
> login-node relay, now removed), use the `proxy` module below — a router on the
> `small` partition, which is the NRIS-policy-correct place for a long-lived
> process. The direct `tunnel` is still the simplest path for a quick single-server
> session.

#### Proxy Module (durable multi-model router)
```bash
./olivia.sh proxy start        # Deploy router files + submit the router (small partition, 7-day walltime)
./olivia.sh proxy tunnel       # Forward localhost:8003 -> router node (point clients here)
./olivia.sh proxy status       # Router job + which model presets are currently serving
./olivia.sh proxy logs         # Tail the router log
./olivia.sh proxy stop         # Cancel the router job
./olivia.sh proxy deploy       # Upload router files only
```

A CPU-only reverse proxy (`model_router.py`) submitted to the **`small`**
partition that gives clients **one stable endpoint** routing to whichever GPU
(`accel`) server is live. The client selects a model by the request's `model`
field — a **preset name** (`glm51`, `kimi27`, ...), an alias, or the served repo
id — which the router resolves to a preset (`presets.json`) and thus to a live
backend. Clients never need to know the node or container index. See the full
design + policy analysis in `plans/proposed/small_partition_proxy.md`.

**Discovery is job-name-independent:** the router lists the user's running
`vllm-*` jobs via `squeue`, then **probes each node's `/v1/models`** to learn
what it actually serves, building a {served model → node} map. So it works
regardless of how servers are named (`vllm-server`, or the per-branch
`vllm-<DEPLOY_KEY>` from the concurrent-agent isolation scheme) — no per-preset
job naming required. A server still loading just isn't in the map yet.

Why the small partition: it is the policy-correct home for a long-lived process
(NRIS: *"always use the queue system"*), allows partial-node allocation (the job
asks for just 2 CPUs / 4 GiB), and permits a **7-day** walltime — far more
durable than the ~8 h GPU job, and the policy-clean replacement for the
(now-removed) login-node relay. Compute nodes are not internet-facing, so the
laptop still tunnels in through the login node — but the tunnel target (the
`small` node) is stable for the job's lifetime instead of moving every GPU job
restart.

**Auto-spindown:** an idle CPU job still bills its `small` reservation, so the
router **shuts itself down after 30 min with no GPU servers running** (default;
`ROUTER_EMPTY_TIMEOUT` seconds, `0` disables). Exiting ends the SLURM job and
frees the allocation. There's a full grace window at startup before any server
exists, and transient squeue/probe errors don't count toward the timeout.

Typical flow:
```bash
./olivia.sh proxy start                 # bring up the durable router
./olivia.sh server start glm51          # start one or more GPU servers as usual
./olivia.sh server start laguna         #   (router picks them up within ~15s)
./olivia.sh proxy tunnel                # forward localhost:8003 to the router
curl localhost:8003/v1/models           # see which models are live
python anthropic_proxy.py --model glm51 --upstream http://localhost:8003   # Claude Code
```

Notes:
- All servers serve on port 8000 (each job owns its node, so no collision); the
  router discovers them by probing `/v1/models`, not by job name or port.
- `OLIVIA_PROXY_TOKEN` (env) optionally requires a bearer token / `x-api-key`.
- `ROUTER_EMPTY_TIMEOUT` (env, default 1800) tunes the idle auto-spindown.
- **On-cluster validation pending** — see the plan doc's "Remaining on-cluster
  validation" checklist.

#### Global Options
```bash
./olivia.sh --kill-all         # Close tunnel and SSH connection
./olivia.sh --version          # Show version
```

## Direct Script Commands

The underlying scripts can still be used directly on the cluster:

### Build vLLM Container (`build_vllm_gh200.sh`)
```bash
# List available model presets
./build_vllm_gh200.sh
# or
MODEL_ID=help ./build_vllm_gh200.sh

# Build using a preset (applies recommended vLLM and transformers versions)
MODEL_ID=glm47 ./build_vllm_gh200.sh
MODEL_ID=devstral ./build_vllm_gh200.sh
MODEL_ID=llama ./build_vllm_gh200.sh

# Build second container of same type (different index)
MODEL_ID=glm47 BUILD_INDEX=2 ./build_vllm_gh200.sh

# Override preset defaults
MODEL_ID=glm47 VLLM_VERSION=v0.6.6 ./build_vllm_gh200.sh

# Submit as SLURM job (will FAIL if container exists - safe default)
MODEL_ID=glm47 sbatch build_vllm_gh200.sh

# Submit with explicit overwrite permission
MODEL_ID=glm47 OVERWRITE=1 sbatch build_vllm_gh200.sh

# Create SIF image after build
MODEL_ID=glm47 CREATE_SIF=1 ./build_vllm_gh200.sh

# Custom model ID (uses generic defaults)
MODEL_ID=my-custom-model ./build_vllm_gh200.sh
```

**Safety:** In batch mode (sbatch), the build will fail if a container already exists unless `OVERWRITE=1` is set. This prevents accidental overwrites of working containers.

### Run vLLM Server (`run_vllm_server.sh`)
```bash
# List available containers (run without CONTAINER set)
./run_vllm_server.sh

# Run specific container by name
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Run Devstral container
CONTAINER=vllm-devstral-1-sandbox MODEL=mistralai/Devstral-2-123B-Instruct-2512 ./run_vllm_server.sh

# Submit as SLURM job
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ sbatch run_vllm_server.sh
```

## Architecture

### Build Script (`build_vllm_gh200.sh`)
Five-phase build process using Singularity:
1. Create sandbox from NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`)
2. Verify NGC PyTorch installation
3. Build vLLM with pip constraints to preserve NGC PyTorch (uses `--no-deps` strategy)
4. Verify final installation (PyTorch version unchanged, vLLM functional)
5. Optionally convert sandbox to SIF image

**Critical constraint**: NGC PyTorch must not be replaced by pip. The build uses a constraints file and `--no-deps` installation to prevent this.

### Model Presets

The build script includes predefined configurations for common models. Run without `MODEL_ID` or with `MODEL_ID=help` to see available presets:

| Preset | Description | vLLM | Transformers |
|--------|-------------|------|--------------|
| `glm51_v19` (alias `glm51`) | GLM-5.1 (744B / 40B active) MoE+DSA flagship, recipe-default | v0.19.0 | >=5.4.0 |
| `glm51_v20` | GLM-5.1 on RayExecutorV2 — **quarantined**, same multi-node PP wedge as v0.19.0 | v0.20.0 | >=5.4.0 |
| `glm52` | GLM-5.2 (744B / 40B active) MoE+DSA, successor to 5.1 — FP8. Builds vLLM main pinned to `091386a` + auto-grafts PR#45895 snapshot (`VLLM_PATCHES`); not in any release | main `091386a` + PR#45895 | >=5.4.0 |
| `glm53flash` | **GLM-5.3-Flash (320B/18B active)** — a different model from `glm53`, on the new `glm5_next` arch. NGC 26.07 (torch 2.13, same as `glm53_v27`), vllm-project DeepGEMM `1e1842a8`. `glm5_next` is native from **v0.30.0** — *not* v0.29.0, despite the recipe page. Builds `vllm-glm53flash-1` | v0.32.0 | >=5.16.1,<5.20.0 |
| `glm53flash_prebuilt` | **Fallback.** Same model from the prebuilt arm64 image `vllm/vllm-openai:glm53-flash`, Phase 3 skipped. Was the primary path Aug–Oct 2026 while `glm5_next` was in no public release. Builds to **index 2** so it coexists with the source build | prebuilt | (prebuilt) |
| `glm53_v27` | GLM-5.3 on a **tagged** release. Same arch as 5.2, so this preset exists only to retire the pinned-main + PR-graft build (PR#45895 merged upstream in v0.24.0) and pick up the DSA/parser work through v0.27.1. NGC 26.07 (torch 2.13), vllm-project DeepGEMM `e21c821f`. Builds `vllm-glm53-1`; the serve preset `glm53` needs no build at all | v0.27.1 | >=5.5.3 |
| `glm47` | GLM-4.7 (358B) flagship model | main | >=5.0.0rc0 |
| `kimi` | Kimi K2.6 (1T / 32B active) MoE + MLA, multimodal | v0.19.1 | >=4.57.1,<5.0.0 |
| `laguna` | Laguna M.1 (Poolside, 225B / 23B active) MoE coding model — FP8, single-node TP=4, native vLLM Laguna support | v0.21.0 | >=5.7.0 |
| `ornith` | Ornith 1.0 MoE (Deep Reinforce, Qwen3.5 hybrid-attn + multimodal) coding models — FP8 W8A8. One `qwen3_5_moe` container serves both `ornith` (397B flagship, 2-node TP=4+PP=2) and `ornith_gh200` (35B, 1× GH200, TP=1). New arch → build from main, NGC 26.05 | main | >=5.8.1 |
| `devstral` | Devstral/Mistral models | main | >=4.45.0 |
| `llama` | Llama 3.x models | main | >=4.45.0 |
| `qwen` | Qwen 2.5 models | main | >=4.45.0 |
| `generic` | Generic build (default) | main | >=4.45.0 |

Custom MODEL_ID values are also supported - they will use generic defaults.

To add a new preset, edit the `apply_preset()` function in `build_vllm_gh200.sh`
(build-time vLLM/transformers versions) **and** add the entry to **`presets.json`**
(runtime model/aliases/container/shape/`storage`). `presets.json` is the single source of
truth for the runtime preset table: `olivia.sh` (via `presets.py`) and the
durable router (`model_router.py`) both read it, so the CLI and the router stay in
sync. The old `preset_field`/`normalize_preset` case statements in `olivia.sh`
were extracted into `presets.json` — don't re-add per-preset data to the bash.
**Preset variants (`"env"`).** A preset may carry an `"env"` object of server env
defaults, e.g. `"env": {"MAX_MODEL_LEN": "1048576", "ORNITH_YARN_FACTOR": "4"}`, so a
tested operating point is one name instead of a remembered recipe (`ornith_122b_1m`,
`ornith_1m`, `ornith_1n_lowlat`). `olivia.sh server start` exports each one **only if
the caller has not set it** — `ENABLE_SPECULATIVE=1 ./olivia.sh server start
ornith_122b_1m` still wins — and prints which value it used. `presets.py env <name>`
shows them. Names are validated on load (`^[A-Z][A-Z0-9_]*$`, nothing credential-like:
these values are echoed and ride in the sbatch command line), and they must also be on
`start_server_job`'s forwarding whitelist to reach the job. Variants share the base's
repo id, so a request routed by repo id resolves to the base preset (same weights,
same storage tier).

Set `"storage": "work"` for a model that lives on the `/cluster/work` mirror
(doesn't fit the 1 TiB project quota); it defaults to `projects` (persistent).
`olivia.sh` resolves `HF_HOME` from this for `prefetch`/`server start`, so the
right cache is picked without a hand-passed override — keep it in sync with the
[[project_hf_storage_split]] note when weights are relocated.

### Server Script (`run_vllm_server.sh`)
Runs vLLM server with GH200-optimized settings:
- Tensor parallelism across 4 GPUs by default
- NCCL optimizations for NVLink (`NCCL_P2P_LEVEL=NVL`)
- GPU reordering for better performance (`CUDA_VISIBLE_DEVICES=1,2,3,0`)
- Flash Attention backend
- Speculative decoding with ngram method (disabled by default, enable for repetitive workloads)

### Key Environment Variables

**Build:**) 
- `CONTAINER_DIR`: Output directory for containers (required)
- `MODEL_ID`: Model identifier for naming (default: `generic`, e.g., `glm47`, `devstral`)
- `BUILD_INDEX`: Build index for multiple builds of same model (default: `1`)
- `VLLM_VERSION`: vLLM version to build (default: `main`)
- `CREATE_SIF`: Set to `1` to create SIF image after build
- `OVERWRITE`: Set to `1` to allow overwriting existing containers in batch mode (default: `0`)

**Server:**
- `CONTAINER_DIR`: Directory to search for containers (required)
- `CONTAINER`: Container name or path (if not set, lists available containers)
- `MODEL`: HuggingFace model ID (default: `mistralai/Devstral-2-123B-Instruct-2512`)
- `TP_SIZE`: Tensor parallel size (default: 4)
- `GPU_MEM_UTIL`: GPU memory utilization (default: 0.90)
- `MAX_MODEL_LEN`: Maximum context length (default: `131072` for GLM-5.1 — native 205K context with ~260 GB KV cache headroom on 8×GH200 AWQ; `32768` elsewhere)
- `HF_HOME`: HuggingFace cache directory for model weights (**required**). The value in `mise.local.toml` is the **persistent/projects root** (e.g. `/cluster/projects/<proj>/huggingface`). `olivia.sh` forwards it from your local environment, but for a **work-tier preset** (`"storage": "work"` in `presets.json`) it first rewrites the root to the `/cluster/work` mirror — so `prefetch`/`server start` land a kimi/glm51/glm52 model on `/cluster/work` automatically, while laguna/glm47/etc. stay on the persistent root. `run_vllm_server.sh` errors if it is unset and (correctly, by design now) warns when it resolves under `/cluster/work` (auto-purged after 21–42 days — see "Persistent model cache" under Container/Cache Structure). An explicit `HF_HOME=…` on the command line overrides the per-preset resolution. `HF_CACHE` is a back-compat alias.
- `HF_TOKEN`: HuggingFace token for gated models. `olivia.sh` forwards it over stdin (kept off argv and out of logs), so it lives only in local config — not the cluster shell.
- `QOS`: SLURM QOS for the server job, passed through as `--qos=` by
  `olivia.sh server start` (default: unset → the account's `nn10104k` QOS).
  **`QOS=devel` is the most effective scheduling lever on this cluster.** Priority
  here is `QOS_component + 1 point per accrued minute` (fairshare/jobsize/partition
  all contribute 0), and the QOS component is
  `PriorityWeightQOS(20000) × (your QOS priority / cluster max 10000)` — so
  `nn10104k` (9880) scores **19760** while `devel` (10000) scores **20000**. That
  +240 head start is worth **four hours of queueing**: measured 2026-10-28, a devel
  job started in *seconds* while a normal-QOS build sat pending **7.5 h** on the same
  partition. `devel` caps walltime at **02:00:00**, allows **2** concurrent jobs, and
  carries **DenyOnLimit** (a >2 h request is *rejected at submit*, not killed later),
  so it fits a serve plus a benchmark sweep but **not** a from-scratch container
  build. `olivia.sh` warns if you combine `QOS=devel` with `TIME_LIMIT` over 2 h.
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: `FLASH_ATTN`)
- `VERBOSE`: Set to `1` for detailed logging including weight loading progress (default: `0`)
- `VLLM_LOGGING_LEVEL`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`, or `DEBUG` if VERBOSE=1)
- `CUDAGRAPH_MODE`: vLLM compilation / CUDAGraph mode (default: unset → vLLM auto-selects `PIECEWISE` or `FULL_AND_PIECEWISE`). Set `NONE` to disable graph capture (eager, slow — debugging escape hatch). Capture adds a multi-minute "Capturing cudagraph" phase at startup but typically yields 2-5× decode throughput.
- `LOAD_FORMAT`: passed through to vLLM's `--load-format`. Its headline use is `runai_streamer` — the Run:ai Model Streamer, a parallel safetensors loader that cuts GLM-5.2's cold load ~6× (validated: ~56 min default → **~9.5 min**; GLM-5.2 cold load is vLLM-pipeline-bound, not I/O-bound). **Auto-defaulted to `runai_streamer` for GLM-5.2** when the staged runai package exists at `$HF_HOME/runai-pkg` (see `CONTAINER_PYTHONPATH`); unset it (or pass another format) to opt out. No-ops for presets whose `HF_HOME` tier lacks the staged pkg (e.g. the work-tier FP8 `glm52`).
- `CONTAINER_PYTHONPATH`: prepended as `PYTHONPATH` inside the Singularity container (must be under an already-bound dir). Lets a pip package we can't install offline into the container still be importable — used to load the staged `runai_model_streamer` from `$HF_HOME/runai-pkg` (bound because it lives under `HF_HOME`). Auto-set alongside the `LOAD_FORMAT=runai_streamer` GLM-5.2 default. To re-stage the pkg (login node has internet): `pip download runai-model-streamer --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12 --abi cp312 --implementation cp`, unzip under `HF_HOME/runai-pkg` (+ the `humanize` dep). Source wheels cached at `/cluster/projects/nn10104k/.mtp-graft/runai-wheels/`.

**Speculative Decoding (ngram):**
- `ENABLE_SPECULATIVE`: Enable speculative decoding (default: `0`)
- `NUM_SPECULATIVE_TOKENS`: Number of tokens to speculate (default: 5)
- `PROMPT_LOOKUP_MAX`: Max n-gram window size (default: 4)

**GLM-4.7 / GLM-5.1 Specific (auto-detected when MODEL contains "GLM-4.7" or "GLM-5.1"):**
- `GLM_TOOL_PARSER`: Tool call parser (default: `glm47` — GLM-5.1 reuses the GLM-4.7 parser per the official vLLM recipe)
- `GLM_REASONING_PARSER`: Reasoning parser (default: `glm45`)
- `ENABLE_AUTO_TOOL_CHOICE`: Enable automatic tool selection (default: `1` for GLM-4.7/GLM-5.1, `0` elsewhere — GLM ships with a tool parser wired up, and OpenAI tool-using clients like Claude Code via `anthropic_proxy.py` need this on). Set `ENABLE_AUTO_TOOL_CHOICE=0` explicitly to disable on GLM.
- `SERVED_MODEL_NAME`: Custom model name for API (default: empty, uses model ID)
- `MTP_SPECULATIVE_TOKENS`: MTP speculative tokens (default: `3`)
- GLM-5.1 additionally passes `--trust-remote-code` and flips the MoE flashinfer kernel from `VLLM_USE_FLASHINFER_MOE_FP8=1` to `VLLM_USE_FLASHINFER_MOE_FP16=1` (matches the QuantTrio GLM-5-AWQ recipe)

**Kimi K2.6 Specific (auto-detected when MODEL contains "Kimi-K2"):**
- `KIMI_TOOL_PARSER`: Tool call parser (default: `kimi_k2`)
- `KIMI_REASONING_PARSER`: Reasoning parser (default: `kimi_k2`; set to empty to omit, e.g. instant mode). Thinking mode is on by default, so this is normally required.
- Kimi K2.6 also passes `--trust-remote-code` (custom `KimiK25ForConditionalGeneration`) and `--mm-encoder-tp-mode data` (MoonViT vision encoder replicated across the TP group), auto-selects an MLA attention backend, and enables expert parallel. No `--quantization` flag — the native int4 is `compressed-tensors`, which vLLM auto-detects.
- **CUDAGraph (decode speedup) — not available; Kimi runs eager (`CUDAGRAPH_MODE=NONE`, the default).** CUDAGraph *capture* is unrecoverable on the deployed vLLM 0.21 + NGC-torch GH200 stack and the recovery campaign is **exhausted** (2026-06-15); both K2.6 (`:8010`) and K2.7 (`:8020`) on `vllm-kimi-4-sandbox` run eager and stable. Eager single-stream decode is ~17 tok/s (overhead-bound), but per-stream stays flat under load and aggregate scales near-linearly (~267 tok/s at 16-way, ~600 at 32-way, ~830 at 48-way in production), so batched/multi-user throughput is good. **Two distinct failures — do not conflate them:**
    - *profile_run dispatch crash — FIXED (still required).* `torch.compile` profile_run aborts *before* capture with `Multiple dispatch failed for 'torch._ops.vllm.min_latency_fused_qkv_a_proj' … NotImplemented`: that MLA op has a registered fake, but under NGC's torch alpha it never lands in the FakeTensor dispatch table. `build_vllm_gh200.sh` (`PYPATCH_MINLATENCY`) forces `_use_min_latency_gemm=False` in `deepseek_v2.py`, so the op is never emitted and the layer falls back to `MergedColumnParallelLinear` (identical math). Baked into the build; it only lets compile *reach* capture — it is **not** the capture bug.
    - *CUDAGraph capture IMA — UNFIXED (the blocker).* With profile_run fixed, capture itself dies with `CUDA error: an illegal memory access` (`cudaErrorIllegalAddress`) inside a **miscompiled inductor kernel** (`_capture_cudagraphs` → `torch/_inductor/runtime/triton_heuristics.py:autotune_to_one_config`). It is **mode-independent** (`PIECEWISE` and `FULL_AND_PIECEWISE` both IMA) and both escape axes are exhausted: config (`combo_kernels=false`, `use_inductor_graph_partition`, `max_cudagraph_capture_size`) and torch (rebuilt on NGC 26.05 with a newer inductor — capture reached 30/51 graphs then the identical IMA; 26.05 separately breaks the FastAPI server). Remaining low-priority options: an upstream bug report, or a non-NGC stable torch (forfeits NGC's tuned GH200 kernels).

  Do **not** set `CUDAGRAPH_MODE=PIECEWISE`/`FULL_AND_PIECEWISE` on the current containers — capture will crash. (Historical note: PIECEWISE *did* capture and run ~34 tok/s on the earlier vLLM **0.19.1** container; moving to 0.21 for reasoning_tokens + K2.7 lost it to the capture IMA above.) Separately, idle multi-node servers can die on an idle→active NCCL abort regardless of mode — keep a keepalive pinging during idle windows (`anthropic_proxy.py --keepalive-interval`). Full account: `plans/proposed/kimi_k27_results.md`; superseded history: `plans/proposed/kimi_serving_perf.md`.

**Multi-node Distributed Inference (GLM-5.1):**
- `NUM_NODES`: Number of nodes to use (default: `1`). When `>1`, `run_vllm_server.sh` bootstraps a Ray cluster via `srun`, starts head on node 0, workers on remaining nodes, waits for the cluster to register the expected GPU count, then launches vLLM with `--distributed-executor-backend ray`
- `PP_SIZE`: Pipeline parallel size (default: `1`). For 2-node glm51 jobs this is `2`
- `TP_SIZE`: Tensor parallel size (default: `4`). For multi-node jobs, this is the *intra-node* tensor parallel size (per node)
- `RAY_PORT`: Ray GCS port (default: `6379`)
- `RAY_CGRAPH_GET_TIMEOUT`: Ray compiled-DAG per-step timeout in seconds (default: `1800`, applied to multi-node only). Ray's upstream default of `300` is too short for multi-node PP inference over Slingshot — a single engine step during long generations can exceed it and crash the raylet with `ChannelError: Channel closed`.
- Head node IP is auto-discovered via `ip -4 -o addr show hsn0` (Slingshot interface); falls back to `hostname -I`
- Ray temp dir is per-job: `${WORKDIR}/cache/ray_tmp_${SLURM_JOB_ID}`

**MoE (Mixture of Experts) / AWQ Settings:**
- `ENABLE_EXPERT_PARALLEL`: Enable expert parallel sharding (default: `auto` - auto-enables for AWQ MoE models including GLM-4.7-AWQ and GLM-5.1-AWQ)

**Batching Proxy (for SSH tunnel streaming):**
- `ENABLE_PROXY`: Enable the batching proxy (default: `0`)
- `PROXY_PORT`: Port for the proxy server (default: `8001`)
- `PROXY_BATCH_TOKENS`: Flush after N tokens (default: `15`)
- `PROXY_BATCH_CHARS`: Flush after N characters (default: `100`)
- `PROXY_BATCH_DELAY_MS`: Max delay before flush in milliseconds (default: `150`)

### GLM-4.7 Quantization Options

| Model | Size | GH200 Compatible | Notes |
|-------|------|------------------|-------|
| `QuantTrio/GLM-4.7-AWQ` | ~358GB | Yes | Tight fit on 4×96GB, reduce MAX_MODEL_LEN |
| `QuantTrio/GLM-4.7-AWQ` | ~181GB | **Yes (Recommended)** | AWQ 4-bit, leaves room for KV cache |
| `Salyut1/GLM-4.7-NVFP4` | ~179GB | **No** | Requires Blackwell GPUs (B100/B200) |

**Important:** NVFP4 quantization requires native FP4 tensor cores only available on Blackwell GPUs. GH200/Hopper GPUs will fail with `No compiled nvfp4 quantization kernel`.

### GLM-4.7 Usage

```bash
# Build GLM-4.7 container
MODEL_ID=glm47 ./build_vllm_gh200.sh

# Run GLM-4.7 with AWQ quantization (RECOMMENDED for GH200)
# Leaves ~200GB headroom for KV cache
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Run GLM-4.7 with FP8 quantization (tight fit, reduce context length)
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ MAX_MODEL_LEN=8192 ./run_vllm_server.sh

# Run GLM-4.7 AWQ with MTP speculative decoding
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_SPECULATIVE=1 ./run_vllm_server.sh

# Run GLM-4.7 with tool calling enabled
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_AUTO_TOOL_CHOICE=1 ./run_vllm_server.sh
```

**Memory requirements for GLM-4.7 (358B parameters):**
| Quantization | Model Size | 4×GH200 (384GB) | Notes |
|--------------|------------|-----------------|-------|
| AWQ 4-bit | ~181GB | ~200GB free for KV cache | Recommended |
| FP8 | ~358GB | ~26GB free | Reduce MAX_MODEL_LEN |
| BF16 | ~716GB | Won't fit | Needs 8+ GPUs |

### GLM-5.1 Quantization Options

GLM-5.1 is a 744B-parameter MoE model with DeepSeek Sparse Attention (DSA), activating ~40–44B parameters per token. It is the successor to GLM-5 and shares the same architecture scale. GLM-5.1 does **not** fit on 4×GH200 in any useful quantization — this preset is designed for **8 GPUs across 2 nodes** on Olivia.

| Model | Size | Olivia Compatible | Notes |
|-------|------|-------------------|-------|
| `cyankiwi/GLM-5.1-AWQ-4bit` | ~430 GB | **Yes (Recommended)** | AWQ 4-bit, fits on 8×GH200 with ~260 GB free for KV cache |
| `zai-org/GLM-5.1-FP8` | ~744 GB | Tight fit on 8 GPUs | Needs DeepGEMM and aggressive context reduction |
| `zai-org/GLM-5.1` (BF16) | ~1.5 TB | **No** | Would need 16+ GPUs |
| `QuantTrio/GLM-5.1-AWQ` | — | **Does not exist (yet)** | Community-requested; not yet released |

**Important:** On Olivia, glm51 is the only preset that crosses a node boundary today. Internode bandwidth is **HPE Slingshot at 200 Gbit/s (~25 GB/s)**, ~36× slower than intra-node NVLink C2C (900 GB/s). Naive TP=8 across 2 nodes would bottleneck on cross-node all-reduce for every transformer layer. The preset therefore uses **TP=4 intra-node + PP=2 cross-node**: pipeline parallelism only transfers hidden states between stages (not per-layer), so it degrades gracefully on Slingshot-class links.

### Known issue: multi-node PP decode wedge

**RESOLVED 2026-06-20 — the cause is the custom all-reduce, not capture (confirmed by experiment).** On the rebuilt NGC-26.03 `vllm-glm51-1-sandbox`, `CUDAGRAPH_MODE=PIECEWISE` (which auto-disables the custom all-reduce → graph-safe NCCL all-reduce) runs `bench_sweep.py` **1→64 with 0 failures** (~22 tok/s/stream through 8-way, ~895 tok/s @64; previously hung at `Running ≥ 2`).

**Attribution experiment (the decisive test):** serving **eager + `DISABLE_CUSTOM_ALL_REDUCE=1`** (NCCL all-reduce, *no capture*) → **0 failures at concurrency 1/2/4/8/16 = no wedge**. The only difference from the original wedging config (eager + *custom* all-reduce) is the all-reduce kernel, so **vLLM's custom all-reduce is the wedge cause; NCCL all-reduce fixes it.** Capture is **not** the wedge fix — it merely *requires* the custom kernel off (it isn't graph-safe), so enabling capture incidentally applies the real fix. (This overturned an earlier theory that capture's static-graph determinism fixed a cross-rank dispatch divergence — the experiment disproved it; lesson: validate mechanism claims.)

**Why the custom all-reduce wedged (mechanism):** vLLM's custom all-reduce (the intra-node TP=4 NVLink one/two-shot reduction, run every layer) uses a shared-memory + IPC signaling handshake. Under concurrent decode (`Running ≥ 2`) on this 2-node PP setup, that handshake races/deadlocks when overlapping decode steps contend on the signal flags → a rank blocks forever → step loop stalls, `Running` pinned at 0 tok/s, eventual raylet abort. NCCL's all-reduce uses a robust collective protocol (no shared-mem flag race) and doesn't hang. (Also why swapping the Ray executor — Compiled Graph → RayExecutorV2 — didn't help: the deadlock is in the all-reduce kernel, not the executor.)

**What capture *does* contribute — throughput (~4.5×), not de-wedging:** eager + NCCL all-reduce is only **~5 tok/s/stream** (NCCL's tiny per-decode-step all-reduces are launch/sync-bound in eager). PIECEWISE capture (capture sizes `[1,2,…,512]`; only attention + `sparse_attn_indexer` stay eager) amortizes the per-step overhead via graph replay → **~22 tok/s/stream**. So the recommended config is **capture + NCCL all-reduce**: NCCL de-wedges, capture makes it fast. The serialization workaround below is **superseded** on this container.

---
*Historical (pre-fix) account:*

GLM-5.1 on Olivia (2×4 GH200, TP=4+PP=2, Slingshot) reproducibly wedges during decode when `Running >= 2` concurrent sequences hit the engine. Prefill completes normally; the first few decode tokens may emit; then generation stops at `0 tok/s` while `Running` stays pinned and the engine step loop never advances. HTTP front-end stays responsive. Eventually Ray raylet aborts with `Fatal Python error: Aborted`.

Originally suspected to be Ray Compiled Graph's `MutableObjectProvider` deadlock (`ray#58426`), but **upgrading to vLLM v0.20.0 + RayExecutorV2 (which bypasses Compiled Graph entirely, confirmed in logs) reproduced the exact same wedge**. The actual root cause is likely deeper — candidates include DSA attention + PP decode sync, MoE expert-parallel routing across PP stages, or NCCL P2P behavior on Slingshot for decode-sized tensors. No upstream fix exists; the official vLLM GLM-5 recipe only validates single-node TP=8.

**Workaround:** run `anthropic_proxy.py` with request serialization (on by default), which holds an `asyncio.Lock` around the upstream `/v1/chat/completions` call so vLLM never sees `Running >= 2` from our proxy. Claude Code sends parallel `generate_session_title` + `repl_main_thread` on every turn — the second request queues invisibly behind the first. Cost: parallel client requests serialize; for single-user CLI usage this is invisible.

Upstream issue trackers to check for future fixes: `vllm#26318` (Slurm + Slingshot PP hang), `vllm#30044` (2-node GH200 TP=4/PP=2), `vllm#24689` / `PR #25906` (full-CUDAGraph cross-rank dispatch wedge).

### Known issue: empty tool_response in GLM-5.1 chat template (cyankiwi fork)

`cyankiwi/GLM-5.1-AWQ-4bit` ships a modified `chat_template.jinja` that **silently drops the content of `role:tool` messages**, rendering every tool result as literal `<tool_response><tools>\n</tools></tool_response>` with no content inside. The model sees "empty tools" for every MCP tool call (WebSearch, Tavily, etc.) and can't act on the results.

Root cause: vLLM's OpenAI entrypoint (`chat_utils.py:1519-1521`) auto-converts string `role:tool` content into `[{"type": "text", "text": "..."}]`. The cyankiwi template's tool-handler hits its `{%- else -%}` branch for non-string content and treats each item as a `tool_reference` lookup (expecting a `.name` field) — but text content parts have no `.name`, so the inner loop emits nothing. The base `zai-org/GLM-5.1` template has a third fallback branch (`visible_text(m.content)`) that cyankiwi removed.

**Fix:** we ship the base `zai-org/GLM-5.1/chat_template.jinja` at `templates/glm51_chat_template.jinja` and auto-pass `--chat-template <path>` to vLLM when running GLM-5.1. `olivia.sh server deploy` uploads the template alongside `run_vllm_server.sh`. No proxy change needed. Override via `CHAT_TEMPLATE_FILE=<path>` if you want to experiment with a different template.

### GLM-5.1 Usage

```bash
# Build GLM-5.1 container on vLLM v0.19.0 (alias glm51, container index 1)
./olivia.sh build glm51_v19
# Quarantined v0.20.0 + RayExecutorV2 build (container index 2)
./olivia.sh build glm51_v20
# or direct:
MODEL_ID=glm51 sbatch build_vllm_gh200.sh

# Start GLM-5.1 server (preset auto-allocates 2 nodes × 4 GPUs)
./olivia.sh server start glm51_v19

# Watch loading progress (handles multi-node GPU aggregation automatically)
./olivia.sh server watch

# Direct sbatch invocation for glm51 (multi-node overrides required)
CONTAINER=vllm-glm51-1-sandbox MODEL=cyankiwi/GLM-5.1-AWQ-4bit \
    NUM_NODES=2 TP_SIZE=4 PP_SIZE=2 \
    sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 \
    run_vllm_server.sh

# Override context length (GLM-5.1 has native 205K context window)
./olivia.sh server start glm51_v19
# or direct: MAX_MODEL_LEN=65536 ... run_vllm_server.sh
```

**Memory requirements for GLM-5.1 (744B params, ~40B active):**
| Quantization | Model Size | 8×GH200 (768 GB HBM) | Notes |
|--------------|------------|----------------------|-------|
| AWQ 4-bit (cyankiwi) | ~430 GB | ~260 GB free (@0.90 util) | Recommended, plenty of KV cache room |
| FP8 (zai-org) | ~744 GB | ~25 GB free | Tight, reduce MAX_MODEL_LEN |
| BF16 | ~1.5 TB | Won't fit | — |

**Multi-node architecture (glm51):**
```
                      ┌─────────────────────────────┐
                      │  SLURM allocation (2 nodes) │
                      └──────────────┬──────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                │                                         │
      Node 0 (head)                                Node 1 (worker)
      ┌──────────────────┐                        ┌──────────────────┐
      │ Ray head (6379)  │ ◄─── Slingshot ───►    │ Ray worker       │
      │ vllm serve       │      200 Gbit/s        │ (Ray only)       │
      │ TP=4 (NVLink)    │      PP activations    │ TP=4 (NVLink)    │
      │ GPUs 0-3         │                        │ GPUs 0-3         │
      └──────────────────┘                        └──────────────────┘
```

The Ray cluster is bootstrapped at job start via `srun`, torn down on exit via a trap. vLLM discovers GPUs across both nodes through Ray and assigns the pipeline stages accordingly.

### GLM-5.2 (preset `glm52`) — FP8, 3-node, bleeding edge

GLM-5.2 is the successor to GLM-5.1: same `GlmMoeDsaForCausalLM` MoE+DSA architecture and 744B / ~40B-active scale, but with a **1M-token native context** and a **new periodic / skip-topk DSA indexer** (`index_topk_freq=4`, `index_skip_topk_offset=3`, `index_topk_pattern`) that GLM-5.1 does not have.

**Upstream status (as of 2026-06-17): not in any vLLM release.** The skip-topk indexer path is implemented by upstream **PR#45895** ("Indexer init skip and MTP TopK share for iteration"), created 2026-06-17 and still unmerged. v0.23.0 was tagged two days before it. The build preset **pins `VLLM_VERSION` to the exact main commit the deployed container built from and the snapshot was validated against (`091386a`, 2026-06-17)** and **grafts PR#45895 at build time** via the `VLLM_PATCHES` mechanism (`PRESET_VLLM_PATCHES="45895"`), so `./olivia.sh build glm52` is **not blocked on the merge** and is **reproducible** — clone(pinned base) + apply(snapshot) is byte-identical every build. See "Patch-during-build" below. A newer vLLM main may also require `NGC_PYTORCH_TAG=26.04-py3` or later. Our multi-node PP decode wedge workaround (proxy serialization) is **not** superseded — the upstream wedge issues (#26318, #30044, #24689) all went stale/closed-unfixed.

**Patch-during-build (`VLLM_PATCHES`):** `build_vllm_gh200.sh` takes a space-separated list of vLLM PR numbers (`VLLM_PATCHES`, defaulting to the preset's `PRESET_VLLM_PATCHES`). In Phase 3, after cloning vLLM (at the pinned commit) and before the `pip install .` compile, it `git apply`s each PR's cumulative diff to `/opt/vllm` — from the committed `patches/` snapshot if present (preferred, reproducible), else fetched live from `github.com/vllm-project/vllm/pull/<N>.diff`. The step is **idempotent** (a reverse-apply check skips PRs already present — e.g. once one merges into main) and **fails the build loudly** if a diff no longer applies cleanly (rather than compiling a half-patched tree). PR#45895 is pure Python (9 files), so no kernel recompile is triggered. Validated 2026-06-17: the diff applies cleanly to the pinned base (main `091386a`, the commit the deployed glm52 container built from). A snapshot lives at `patches/vllm-pr45895-glm52-indexer.diff`; `olivia.sh build` deploys `patches/` alongside the build script and binds it read-only at `/opt/olivia-patches`, and the build **prefers the committed snapshot** (reproducible + offline), falling back to the **live** GitHub PR fetch only when no snapshot is bound in. **When PR#45895 merges:** drop `"45895"` from `PRESET_VLLM_PATCHES` (the idempotent apply tolerates it in the meantime), or build with `VLLM_PATCHES=""`.

**Why 3 nodes (not 2 like glm51):** the only quant available today is block-FP8 (`zai-org/GLM-5.2-FP8`, or the byte-identical re-host `RedHatAI/GLM-5.2-FP8`), ~755 GB. At ~94 GB/GPU that does **not** fit 8×GH200 (96 GB cards). Spreading across **3 nodes × 4 GPUs (TP=4 + PP=3 = 12 GPUs)** drops weights to ~63 GB/GPU, leaving ~18 GB/GPU for KV. **Now published (2026-07):** `cyankiwi/GLM-5.2-AWQ-INT4` (~415 GiB) restores the **8-GPU / 2-node** path (TP=4 + PP=2) — the preferred shape — wired as the separate **`glm52_awq`** preset (persistent tier, reuses the glm52 container). See "GLM-5.2 AWQ" below.

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| block-FP8 (e4m3, [128,128]) | `zai-org/GLM-5.2-FP8` / `RedHatAI/GLM-5.2-FP8` | ~755 GB | **3 nodes × 4 GH200** (TP=4 + PP=3); 8 GPUs won't fit |
| AWQ / compressed-tensors INT4 | `cyankiwi/GLM-5.2-AWQ-INT4` | **~411 GB** | **NOW EXISTS** (preset `glm52_awq`) — **2 nodes × 4 GH200** (TP=4 + PP=2), the cheaper path this table used to say was hypothetical. Verified on-cluster 2026-08-17: `GlmMoeDsaForCausalLM`, 78 layers, 1M ctx, `pack-quantized` 4-bit group_size 32, **same skip-topk indexer** (`index_topk_freq=4`, `index_skip_topk_offset=3`), MTP head present. Already fully cached on the **persistent** projects tier |
| AWQ-INT4 | `cyankiwi/GLM-5.2-AWQ-INT4` | ~415 GiB | **Yes — preset `glm52_awq`, 2 nodes × 4 GH200** (TP=4 + PP=2); the **preferred** path (fits 8 GPUs, persistent tier). Published by the same cyankiwi author as the GLM-5.1 AWQ; same `GlmMoeDsa` arch. |
| NVFP4 | `Lorbus/GLM-5.2-NVFP4` etc. | — | **No** — needs Blackwell FP4 tensor cores |
| BF16 | `zai-org/GLM-5.2` | ~1.5 TB | No — 16+ GPUs |

**Runtime specifics auto-applied by `run_vllm_server.sh` when `MODEL` contains `GLM-5.2`** (`IS_GLM52`): `--trust-remote-code`, glm47 tool parser + glm45 reasoning parser, auto tool choice, sparse-MLA attention backend (auto-select), **eager CUDAGraph** (`CUDAGRAPH_MODE=NONE` — capture IMAs during `capture_model` on this NGC stack, like Kimi; override to retry), `MAX_MODEL_LEN=131072` default, **block-FP8 path** — `VLLM_USE_DEEP_GEMM=1`, `VLLM_DEEP_GEMM_WARMUP=skip` (avoids the multi-minute DeepGEMM JIT warmup; override with `VLLM_DEEP_GEMM_WARMUP=""`), and `--kv-cache-dtype fp8_e4m3` (roughly doubles fittable context/concurrency). MTP speculative decoding is auto-disabled because `PP_SIZE>1` (the existing PP+MTP guard). The cyankiwi chat-template override is **5.1-only** and is skipped for 5.2 (the FP8 repos ship a correct base template). PR#45895's validated flag set is the source of these defaults. **Cold load: `LOAD_FORMAT=runai_streamer` is auto-defaulted** when the staged runai pkg is present at `$HF_HOME/runai-pkg` (~6× faster load, ~9.5 min vs ~56 min — see `LOAD_FORMAT`/`CONTAINER_PYTHONPATH` above); it no-ops when the pkg isn't staged on the resolved `HF_HOME` tier.

**`reasoning_tokens` on `/v1/chat/completions` usage (build patch):** upstream vLLM only surfaces `reasoning_tokens` on `/v1/responses` — chat/completions has no `completion_tokens_details` at all, and the glm45 reasoning parser (`DeepSeekV3ReasoningWithThinkingParser`) never counts reasoning tokens (the base `count_reasoning_tokens()` returns 0, the V3 wrapper doesn't forward it, and GLM emits `</think>` but omits the `<think>` start token so the inner R1 parser's start/end depth counter is 0 too). `build_vllm_gh200.sh` (`PYPATCH_REASONING_TOKENS`, applied in Phase 3 after the PR graft) fixes all three: it adds `count_reasoning_tokens` to **both** the glm45 (`DeepSeekV3ReasoningParser`) and `kimi_k2` parsers (count tokens before the first reasoning-end marker), adds `CompletionTokenUsageInfo` + `UsageInfo.completion_tokens_details` to `engine/protocol.py`, and populates it in the non-streaming and streaming usage paths of `chat_completion/serving.py` — tolerant of both vLLM 0.21 (`reasoning_parser` / `all_previous_token_ids`) and vLLM main (`parser`/`parsers` + its own `previous_token_ids` accumulator) serving layouts (0.21-specific anchors tried first under a guard, so the main-only `parser` var is never injected into 0.21). The patch is idempotent and defensive (a missing anchor logs + no-ops, so a future vLLM refactor never half-patches or fails the build) and the runtime guards leave non-reasoning models unaffected, so it runs on every build — **any reasoning model (glm47/glm51/glm52 and Kimi K2.x) gets it from this one block.** **Takes effect on the next container rebuild.**

**Weights storage:** lives on the at-risk-but-spacious work cache, not the quota-bound project area — the 703.7 GiB FP8 does not fit the ~462 GiB free on `/cluster/projects/nn10104k` (1 TiB quota). This is now **automatic**: the `glm52` preset is tagged `"storage": "work"` in `presets.json`, so `olivia.sh prefetch`/`server start` resolve `HF_HOME` to `/cluster/work/projects/nn10104k/huggingface` on their own — no hand-passed override (see [[project_hf_storage_split]]). The tier hint is portable (a `work`/`projects` flag, not an absolute path); the actual roots stay in `mise.local.toml`. To force a different cache, an explicit `HF_HOME=… ` on the command line still wins.
```bash
./olivia.sh prefetch glm52      # -> /cluster/work/... automatically (preset is work-tier)
```

### GLM-5.2 Usage

```bash
# 1. Prefetch FP8 weights to the work cache (detached, resumable). The glm52
#    preset is work-tier, so HF_HOME auto-resolves to /cluster/work/... — no
#    override needed (an explicit repo id like RedHatAI/GLM-5.2-FP8 also routes
#    there via the preset registry).
./olivia.sh prefetch glm52

# 2. Build the container. The glm52 preset builds vLLM main and auto-grafts
#    PR#45895 (VLLM_PATCHES=45895) for GLM-5.2's skip-topk indexer — no need to
#    wait for the upstream merge. Disable the graft with VLLM_PATCHES="" if the
#    PR has since merged into main.
./olivia.sh build glm52

# 3. Start the server (preset auto-allocates 3 nodes × 4 GPUs, TP=4 + PP=3;
#    HF_HOME auto-resolves to the work-tier cache).
./olivia.sh server start glm52

# Check whether a 3-node shape can schedule right now
./olivia.sh cluster
```

### GLM-5.3 (presets `glm53` / `glm53_v27`) — same base as 5.2, weights pending

GLM-5.3 (Z.ai, announced **2026-08-14**) is **not a new architecture and not a new
pretrain**. It is GLM-5.2's *same 744B / ~40B-active base* re-post-trained — Z.ai's
own framing is that "every reported gain comes from scaled post-training". Same
`GlmMoeDsaForCausalLM`, same MoE+DSA, same periodic/skip-topk indexer, same 1M
native context. The headline gains are agentic-coding and long-horizon (Terminal-Bench
3.0 4.6 → 28.3, DeepSWE v1.1 46.2 → 66.9) plus an unplanned jump in cyber-security
capability (CyberGym 84.5%) — which is exactly *why* the weights were held back.

> ⚠️ **The weights are not public yet (as of 2026-08-17).** GLM-5.3 shipped
> API-only (Z.ai API, GLM Coding Plan, ZCode); Z.ai said open weights would follow
> "in stages after safety evaluation, roughly two weeks" → **~end of August 2026**.
> There is **no `zai-org/GLM-5.3*` repo on HuggingFace** yet, no published license
> (5.2 was MIT; that does **not** carry over automatically), and no announced quant
> lineup. `presets.json` therefore points at the **expected** id
> `zai-org/GLM-5.3-FP8` (following 5.2's naming) — **confirm it, the license, and
> the actual quants before prefetching.** A `RedHatAI/GLM-5.3-FP8`-style re-host is
> likely too, as with 5.2.

**Because the architecture is unchanged, no new model support is needed anywhere —
the deployed glm52 container already serves it.** That is the whole design of the
`glm53` preset: it points `container_prefix` at `glm52` / index 1, so
`./olivia.sh server start glm53` runs the **already-validated** 3-node FP8 container
with **no build and no allocation spent**. The two presets:

| Preset | Container | vLLM | Why |
|--------|-----------|------|-----|
| `glm53` | `vllm-glm52-1-sandbox` (**shared**, no rebuild) | main `091386a` + PR#45895 graft | Day-0 serving the moment weights land. Zero build risk — this container already serves 744B DSA FP8 across 12 GPUs. |
| `glm53_v27` | `vllm-glm53-1-sandbox` (**its own**) | **v0.27.1** + NGC 26.07 | The upgrade path. Separate container name, so it structurally cannot clobber the working glm52 one. |

**Runtime:** `run_vllm_server.sh` detects `GLM-5.3` in `MODEL` (`IS_GLM53`) and
**lights `IS_GLM52` as the "5.2-family runtime profile"** rather than duplicating
five blocks — so 5.3 inherits verbatim: block-FP8 DeepGEMM path
(`VLLM_USE_DEEP_GEMM=1`, `VLLM_DEEP_GEMM_WARMUP=skip`), **eager CUDAGraph**
(`CUDAGRAPH_MODE=NONE`), no fp8-KV override (FLASHMLA_SPARSE on Hopper rejects it),
RayExecutorV2 engine-as-actor, the PP=3 layer partition `26,24,28`, sparse-MLA
auto-select backend, glm47/glm45 parsers, `--trust-remote-code`, 128K default
context. `IS_GLM53` stays a separate flag purely for reporting and for the day a
future 5.x *isn't* the 5.2 base — split them there first.

#### Is an updated vLLM worth it? (checked 2026-08-17)

Yes, but as a **separate gated experiment**, not as a prerequisite for serving 5.3.

*What has changed upstream since glm52 was pinned to main `091386a` (2026-06-17):*
- **PR#45895 merged** (`ab666069`, 2026-06-19) and is an ancestor of **v0.24.0**
  (2026-06-29) onward — verified against `v0.27.1` (`compare` → `behind_by: 0`). The
  whole reason glm52 pins an unreleased main commit + grafts a patch **evaporates**;
  a modern preset builds a **tag**.
- **v0.24.0** added the streaming parser engine for GLM-4.7/5.1/5.2 (tool-call
  streaming — relevant to Claude Code through `anthropic_proxy.py`).
- **v0.27.0** added *skip sparse indexer scoring for short dense prefills* (#48407),
  straight on the GLM-5.x DSA hot path, plus Quark GLM-5.2 checkpoint fixes (#48886).
- **v0.27.0 moved to torch 2.13.0 + Triton 3.7.1** — a breaking environment change.
  NGC 26.05 (glm52's base) ships torch 2.12.0a0; **NGC 26.07** ships 2.13.0a0 +
  CUDA 13.3.1 and is the newest tag, so the preset moves there. This matters because
  our `--no-deps` strategy keeps *NGC's* torch, so vLLM's pin and the base image
  should agree.
- **DeepGEMM moved:** vLLM no longer pins deepseek-ai — v0.27.1 pins
  `vllm-project/DeepGEMM` @ `e21c821f`. The build script gained a `DEEPGEMM_REPO`
  knob (default unchanged: deepseek-ai) so the preset can match vLLM's pin exactly.

*What it does **not** obviously buy:* nothing upstream addresses the CUDAGraph
capture IMA (still eager) or the multi-node PP decode wedge — #48720 is an RFC for
*detecting* TP deadlocks, not fixing them. glm52 already avoids the wedge via
engine-as-actor RayExecutorV2. A newer torch/inductor *might* fix the capture IMA
(that would be the big win — glm52's ~5.6 tok/s single-stream is eager-bound), which
is worth one `CUDAGRAPH_MODE=PIECEWISE` retry on the new container.

*The de-risking trick:* **validate `glm53_v27` against `GLM-5.2-FP8`, which is
already in the work-tier cache** — the arch is identical, so a clean 5.2 serve on
the new container proves the whole toolchain *before* 5.3 weights exist, and the
existing README numbers give an exact A/B baseline. If it regresses, `glm53` on the
glm52 container is the fallback and needs no rebuild.

### GLM-5.3 Usage

```bash
# --- Path A: serve on the existing glm52 container (no build) ---
# 0. FIRST confirm the real repo id + license once Z.ai publishes the weights;
#    presets.json currently holds the EXPECTED id zai-org/GLM-5.3-FP8.
./olivia.sh prefetch glm53            # ~755 GB -> /cluster/work (work-tier preset)
./olivia.sh cluster                   # check a 3-node × 4-GPU shape can schedule
./olivia.sh server start glm53        # runs vllm-glm52-1-sandbox, TP=4 + PP=3
./olivia.sh server watch

# --- Path B: the vLLM upgrade (own container, cannot clobber glm52) ---
./olivia.sh build glm53_v27           # vLLM v0.27.1 + NGC 26.07 + vllm-project DeepGEMM
# Validate the toolchain on GLM-5.2 weights we already have, BEFORE 5.3 lands:
./olivia.sh server start glm53_v27 --model RedHatAI/GLM-5.2-FP8
# Then A/B against README "## Performance" -> GLM-5.2 (glm52). Retry capture once:
CUDAGRAPH_MODE=PIECEWISE ./olivia.sh server start glm53_v27 --model RedHatAI/GLM-5.2-FP8
```

### DeepSeek-V4.1-Flash (preset `dsv41`) — 748B on ONE node, via Grace memory

The first preset where **non-HBM memory for cold parameters is the enabling trick**,
not an optimisation.

**The arithmetic.** vLLM's recipe sizes V4.1-Flash at **~614 GB of VRAM** — a GB200
NVL4 tray (768 GB) or an 8-GPU H200 node (1128 GB). Our node is **4 × 96 = 384 GB**.
It does not fit, and no quantisation choice closes that gap. What closes it is that
~196.6B of those parameters are **Engram**: an n-gram conditional memory held by just
**two layers (1 and 14)**, each a ~384M-row × 256 hash table, **accessed sparsely by
token lookup**. Resident they cost **~47.2 GiB per rank at TP4** — half of every
96 GB card. `--engram-config {"cpu_offload":true}` puts them in **pinned host
memory** instead, leaving the GPUs holding only the backbone.

**Why this suits GH200 particularly.** Grace sits behind **NVLink-C2C at ~900 GB/s**,
roughly an order of magnitude above a PCIe host hop, with ~800 GB of LPDDR5X per node
to hold the tables. A cold, sparsely-read structure behind a 900 GB/s link is close
to the ideal case for host offload. It is also *finer grained* than our
`CPU_OFFLOAD_GB` lever, which evicts by layer order and is blind to access frequency;
this moves exactly the cold structure. `CPU_OFFLOAD_GB` still stacks on top if the
backbone alone is tight.

**KV is almost free, which is unusual here.** CSA2 assigns each attention layer one of
three static modes (Full / Reindex / Reuse), sharing main KV and indexer K across
layers, and the main KV is **FP4** (E2M1, one E4M3 scale per 16 channels) → **890
bytes per token**, about a quarter of V4-Flash. A 1M-token prompt therefore holds
under 1 GB of global KV. **Weights, not cache, set capacity** — the opposite of every
other large preset here, and why raising `MAX_MODEL_LEN` is cheap on this one.

| | `dsv4flash` (V4) | `dsv41` (V4.1) |
|---|---|---|
| arch | `DeepseekV4ForCausalLM` | **`DeepseekV41ForCausalLM`** |
| KV per token | ~3.5 KB | **890 B** (FP4, shared across layers) |
| spec decode | MTP (layout-mismatched, off) | **DSpark only — V4.1 DROPPED MTP** |
| cold params | — | **196.6B Engram → host memory** |

⚠️ **`DeepSeek-V4.1` contains `DeepSeek-V4` as a substring**, so `IS_DSV41` is tested
first and `IS_DSV4` excludes it — the same trap GLM-5.3 vs GLM-5.3-Flash set.

**Runtime (`IS_DSV41`):** `--tokenizer-mode deepseek_v41`, `deepseek_v41` tool and
reasoning parsers, `--engram-config`, `--attention-config` (backend
**`FLASHMLA_SPARSE_DSV41`** — the compiled path; indexer dtype left at `auto`,
because `mxfp4` is Blackwell-only), **`--kv-cache-dtype fp8`
(REQUIRED — the inverse of GLM-5.3-Flash, where fp8 KV is Blackwell-only)**,
`--mm-encoder-tp-mode data`, `--max-num-seqs 128`, the DeepGEMM lane with warmup
skipped, and `VLLM_ENGINE_READY_TIMEOUT_S=3600` because the first load is long.
**This arch does not support torch.compile**, so the compilation config pins
`cudagraph_mode=FULL_DECODE_ONLY`; the default `FULL_AND_PIECEWISE` dies at capture.

> ✅ **NOT blocked — that claim was wrong, and three serve attempts proved it.**
> v0.32.1 registers **three** V4.1 sparse-MLA backends, not one:
> `FLASHMLA_SPARSE_DSV41` (vLLM's own compiled `_flashmla_C`),
> `FLASHINFER_MLA_SPARSE_DSV41` (needs flashinfer) and `FLASHMLA_MEGA_ATTN_DSV41`
> (same KV geometry — a drop-in A/B). **Only the middle one needs flashinfer**, and
> the preset now defaults to the compiled path, so flashinfer is irrelevant here.
>
> ⚠️ **Do NOT set the indexer knobs from vLLM's recipe.** `indexer_kv_dtype=mxfp4`
> requires **sm_10x (datacenter Blackwell)** and `indexer_sparse_logits` requires
> mxfp4 **plus** an SM100-class GPU — both fatal on Hopper (job 2694626 died 25 s in,
> *after* the Engram tables had offloaded). Leave `indexer_kv_dtype` unset so it
> resolves to `auto` → **fp8**, the Hopper-correct value. Blackwell-only knobs stay
> reachable via `DSV41_INDEXER_KV_DTYPE` / `DSV41_INDEXER_SPARSE_LOGITS`.
> Losing sparse logits is a real cost — it would make the indexer pay O(candidate
> blocks) instead of walking the whole context, and it "pays off roughly 32K and
> beyond" — but it is a **silicon** limit (FP4 tensor cores), not a version gate.
>
> 🧠 **The Engram offload works, measured:** all four ranks log
> `Engram table offloaded to pinned host memory: 96,002,514 rows x 256, 23.60 GiB
> per rank` — per table, two Engram layers, so **~47.2 GiB/rank and ~188.8 GiB of
> Grace LPDDR5X node-wide**, half the node's entire HBM. Read by **UVA** over
> NVLink-C2C, not staged copies, which is why GH200 suits this at all.

#### DeepSeek-V4.1-Flash Usage

```bash
# 1. Prefetch (~510 GB). Work tier: the persistent tier has ~173 GB free after Flash.
./olivia.sh prefetch dsv41

# 2. NO BUILD. DeepseekV41ForCausalLM is native from v0.30.0 and the glm53flash
#    container is v0.32.1.dev0, so it already carries the arch.

# 3. Serve on ONE node, TP=4, Engram in Grace memory. Use devel (see QOS above).
#    PICK AN OPERATING POINT — the axes are opposed and no config wins both.
#    All three measured 2026-11-06; see README "## Performance" for the numbers.

# (a) BATCH THROUGHPUT — the default. 2454 tok/s @64 streams, 128 streams OK.
#     Costs: 131K context, only 1.75x concurrency at the full window.
QOS=devel TIME_LIMIT=02:00:00 ./olivia.sh server start dsv41

# (b) LONG CONTEXT / CONCURRENCY — the FULL 1M native window, 11.3x at 128K.
#     Costs ~30% throughput. Both knobs are REQUIRED: MAX_MODEL_LEN alone fails
#     because the prefill workspace scales with it and eats the pool.
MAX_MODEL_LEN=1048576 DSV41_SPARSE_MLA_WORKSPACE_FACTOR=2 CPU_OFFLOAD_GB=5 \
  QOS=devel TIME_LIMIT=02:00:00 ./olivia.sh server start dsv41

# (c) LOW LATENCY — DSpark speculative decode. 132.0 tok/s single-stream at 131K,
#     the best measured. Costs 47% of aggregate at 16 streams, so NOT a default.
#     The offload is not optional: DSpark's draft stages need resident memory.
ENABLE_SPECULATIVE=1 CPU_OFFLOAD_GB=4 \
  QOS=devel TIME_LIMIT=02:00:00 ./olivia.sh server start dsv41

./olivia.sh server watch

# Fallback / tuning ladder, revised against measurement:
#   a) MORE CONCURRENCY. Short context is capped by our own MAX_NUM_SEQS=128, not
#      by the pool (which holds ~217 streams at 1K). Long context is pool-bound:
#      1.75x at 131072. gpu_memory_utilization is NOT a lever — it is already at
#      its ceiling (the allocator fails to map 20 MB with ~20 MB free).
#   b) LONGER CONTEXT. Raising MAX_MODEL_LEN alone FAILS: the sparse-MLA prefill
#      workspace scales with it (factor x max_model_len x 1152 B), so 262144 at
#      the default factor 5 shrank the pool 1.08 -> 0.59 GiB and vLLM refused
#      (0.84 GiB needed for one request). Pair it with
#      DSV41_SPARSE_MLA_WORKSPACE_FACTOR=2.
#   c) backbone/pool still tight -> CPU_OFFLOAD_GB (stacks on Engram; Grace has
#      ~800 GB). Treat the first run as measuring the weight-GB -> KV-GB rate.
#   d) capture problems -> DSV41_CUDAGRAPH_MODE=NONE
#   e) single-stream speed -> ENABLE_SPECULATIVE=1 (DSpark; never MTP, which V4.1
#      dropped and vLLM rejects). num_speculative_tokens must be 5, the trained
#      dspark_block_size — it is NOT derived; the pydantic validator demands it
#      before vLLM's own `num_speculative_tokens = block_size` line can run.
#      ⚠️ DSPARK IS MEMORY-GATED, SO IT MUST COME AFTER (c), NOT BEFORE.
#      Its draft stages (dspark_target_layer_ids [37,38,39], each a 128-expert
#      MoE) have to be resident. At the default budget job 2730626 loaded fine
#      and then died at 9m54s with `No available memory for the cache blocks`
#      — there is no KV headroom left to pay for the draft. Pair it with
#      CPU_OFFLOAD_GB rather than running it on a bare config.
```

### GLM-5.3-Flash (presets `glm53_flash` / `glm53_flash_prebuilt`) — 320B on ONE node

⚠️ **`GLM-5.3-Flash` is not a variant of `GLM-5.3`. They are different models, and
the substring `GLM-5.3` matches both** — which is why `run_vllm_server.sh` tests
`IS_GLM53_FLASH` *first* and `IS_GLM53` explicitly excludes it. Getting this wrong
hands Flash the 744B model's PP=3 layer partition, eager default,
`--trust-remote-code` and DSA assumptions.

| | `glm53` (744B) | `glm53_flash` |
|---|---|---|
| Params | 744B / 40B active | **320B / 18B active** |
| Arch | `GlmMoeDsaForCausalLM` | **`Glm5NextForConditionalGeneration`** (new) |
| Lineage | GLM-5.2's base, re-post-trained | **New base, new pretrain** |
| Attention | sparse MLA (DSA) everywhere | **hybrid: 34 KDA linear + 11 NoPE sparse-MLA** |
| Modality | text | **native image + video** |
| Shape | 12 GPUs, 3 nodes, TP=4+PP=3 | **4 GPUs, 1 node, TP=4** |
| Weights | ~755 GB FP8 | **~306 GiB FP8** (62 shards, MIT) |

Verified from the published `config.json` + `model.safetensors.index.json`
(2026-09-01), not the model card:
- 45 layers. `layer_types` = 34 `linear_attention` + 11 `deepseek_sparse_attention`;
  `linear_attn_config.kda_layers` lists the KDA ones. **So do not force an
  attention backend** — one forced choice cannot serve both halves. Same rule as
  Ornith/Qwen3.8, for the same reason.
- MoE: 288 routed experts, 8 active + 1 shared, `first_k_dense_replace=3`.
- **mHC** (`mhc: true`, `hc_mult`, `hc_sinkhorn_iters`) — Manifold-Constrained
  Hyper-Connections, a new residual mechanism. Nothing for us to configure, but it
  is why the arch is new rather than a GLM-5.2 reskin.
- Quant is **block-FP8** (`weight_scale_inv` tensors) → the **DeepGEMM lane**, same
  as glm52/qwen38/dsv4, not Ornith's channel/token W8A8.
- Vision tower: 347 `model.visual.*` tensors. Served for text; the encoder loads.
- **MTP head IS shipped — but grepping the weight_map for `mtp` finds ZERO
  tensors.** It is stored as language layer **45**, one past `num_hidden_layers=45`,
  with the `eh_proj`/`enorm`/`hnorm` nextn signature and its own 288-expert MoE.
  The Ornith trap in reverse: there, config claimed a head and the weights were
  absent; here, the name is absent and the weights are present.

**Two Hopper rules that will cost you a job if ignored:**
1. **BF16 KV cache, always.** vLLM's recipe is explicit that `--kv-cache-dtype fp8`
   is Blackwell-only for this model and Hopper must run BF16 KV. This is the
   *inverse* of DeepSeek-V4-Flash, which refuses anything but fp8 KV. That is now
   three different KV-dtype rules across four MLA-family models, so **never
   generalise a KV-dtype setting between them.** The server script clears a stray
   `KV_CACHE_DTYPE=fp8*` with a warning (`FORCE_KV_CACHE_DTYPE=1` overrides).
2. ~~**FlashInfer ≥ 0.6.18** is required for NoPE sparse MLA~~ — **MEASURED FALSE
   on Hopper.** The recipe says so and maps "Sparse-MLA initialization error" to an
   older FlashInfer, but job 2673772 loaded this model and captured CUDA graphs
   51/51 + 2/2 with **flashinfer entirely absent**: the selector logged `Using
   FLASH_ATTN_MLA_SPARSE attention backend out of potential backends:
   ['FLASH_ATTN_MLA_SPARSE', 'FLASHMLA_SPARSE']` — neither is a flashinfer backend —
   with PYNCCL all-reduce and DEEPGEMM FP8 MoE. The NoPE sparse-MLA path is served by
   **FlashAttention** here. What flashinfer's absence actually costs is FlashInfer
   all-reduce and the `FLASHINFER_*` backend variants, not the ability to serve.

3. **THE REAL BLOCKER WAS `max_num_seqs`.** `glm5_next` is a hybrid, and each of its
   34 KDA linear-attention layers holds a per-sequence recurrent state that vLLM
   accounts for as a **"Mamba cache block"** — one per decode sequence, drawn from
   the same budget as KV. At TP=4 / util 0.9 that budget is 2.52 GiB/rank = **202
   blocks**, against vLLM's default `max_num_seqs` of **1024**. So the engine died
   **8.6 minutes in, AFTER a full weight load and a clean capture**:
   `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (202)`.
   Capped at **128** (`GLM53_FLASH_MAX_NUM_SEQS`) — deliberately well under 202,
   because the count drifts with memory-profiling noise and the KV pool binds first
   at every usable context anyway (1.59x at 131072), so margin beats headroom we
   cannot reach.

**Version floor: v0.30.0, and the recipe page is wrong about it.** vLLM's recipe
says "vLLM 0.29.0+". That is incorrect — **v0.29.0's `registry.py` contains zero
`Glm5Next` entries.** Checked tag by tag (2026-10-28): v0.29.0 → 0, v0.30.0 → 4,
v0.31.0 → 4, v0.32.0 → 4. Trusting the recipe would have produced a preset that
builds cleanly and then fails at model load with an unsupported-architecture error.
**The preset builds v0.32.0** (newest tag, 2026-10-18): this model's upstream
support is only weeks old and several GH200-relevant bugs were filed right after
release (missing sparse-MLA paths on some SMs, hybrid mamba/attention KV page
alignment, a recurring CUDA IMA), so the later tag has had the most fixes land.
torch stays **2.13.0**, so **NGC 26.07** is still the matching base — no new base
image. FlashInfer arrives as **0.7.0.post1**, which clears the ≥0.6.18 NoPE
sparse-MLA floor on its own.

> **Historical note — why this was a prebuilt-image preset until 2026-10-28.**
> When GLM-5.3-Flash's weights landed (2026-08-31), `glm5_next` was in **no public
> vLLM release**: `main`'s registry had no entry and the recipe said, verbatim,
> *"use docker before the integration is included in the public repo"*. There was
> nothing to clone and build, so the preset pulled `vllm/vllm-openai:glm53-flash`
> (which does publish a **linux/arm64** manifest, hence usable on GH200). That is
> what `build_vllm_gh200.sh`'s **`PREBUILT_IMAGE`** mode was added for: Phase 1
> builds the sandbox from a docker image, **Phase 3 is skipped entirely**, and
> Phase 4 verifies what actually matters about an image we did not build — that
> `Glm5NextForConditionalGeneration` is registered and FlashInfer is ≥0.6.18 —
> instead of trusting the tag. v0.30.0 made that unnecessary. The mode is **kept**:
> it is generic, additive, has no effect on any other preset, and day-0 models are
> a recurring pattern here. `glm53_flash_prebuilt` (container index 2) keeps the
> image reachable as a **fallback** if the source build regresses.

**Memory budget.** ~306 GiB on 4×96 GB is tighter than `glm47`'s ~358 GB AWQ (which
serves), leaving ~40 GiB after weights at the default `GPU_MEM_UTIL=0.90` — the
first knob if the KV pool comes out too small. KV should nonetheless go far: only
11 of 45 layers cache anything, and those are MLA with `kv_lora_rank=512` (order
11 KB/token at BF16). Enabling MTP costs another ~7 GiB for layer 45's experts,
which is why it is **off by default** here despite being present.

**Reasoning effort.** The chat template takes `reasoning_effort` ∈ `low|high|max`
(default `max`) and thinking is always on. Pass it through
`chat_template_kwargs`. Parsers are the GLM family's: `--tool-call-parser glm47`,
`--reasoning-parser glm45`.

### GLM-5.3-Flash Usage

```bash
# 1. Prefetch (~306 GiB) to the PERSISTENT tier — the preset is storage=projects,
#    so HF_HOME resolves to /cluster/projects automatically. Deliberate: the
#    /cluster/work purge hollowed out 11 of 12 sandboxes once already.
./olivia.sh prefetch glm53_flash

# 2. Build the container from source (vLLM v0.32.0 + NGC 26.07; ~2 h, 1 GPU).
#    ASK FOR 2.5 h, NOT MORE. The job needs ~2 h (a v0.32.0 compile reached
#    1 h 37 m before the DeepGEMM failure, plus SIF conversion); the script's own
#    #SBATCH default of 2 h is too tight. Walltime is a backfill lever *when a
#    schedule gap exists* — a 4 h request needs a 4 h gap — so asking for less is
#    free insurance. But do NOT assume it buys speed: measured 2026-10-28 with
#    `sbatch --test-only`, the estimated start was IDENTICAL for 4 h, 2.5 h and
#    even 1 h, because the binding constraint was queue depth (~1476 pending jobs
#    also wanting a single GPU), not fragmentation.
BUILD_TIME_LIMIT=02:30:00 ./olivia.sh build glm53_flash --sif
#    Add --force to rebuild over an existing/partial sandbox (it builds into
#    .new.JOBID and swaps atomically, preserving the old one as .prev.TIMESTAMP).
#    Fallback if the source build regresses — pull the prebuilt image to index 2:
#    ./olivia.sh build glm53_flash_prebuilt && ./olivia.sh server start glm53_flash_prebuilt
#
#    Do NOT cancel-and-resubmit a queued build just to tweak flags. Measured on
#    this cluster: PriorityWeightAge=10080 with PriorityMaxAge=7d (=10080 min), so
#    the age factor is exactly ONE POINT PER MINUTE QUEUED, and a job's priority is
#    `QOS + AGE` with fairshare/jobsize/partition all contributing 0. Since every
#    job in our QOS shares the same 19760 base (= the median pending priority),
#    MINUTES QUEUED IS THE ONLY THING RANKING US. Example: 48 min of age put us at
#    rank 87 of 1680; resubmitting would have reset that to the median and cost a
#    top-5% slot for no gain. Check with `sprio -j <id> -l` and `sprio -w`.

# 3. Serve on ONE node, TP=4 — ON devel, or you may queue for hours. A 4-GPU
#    single-node shape is the hardest allocation we ask for (there were ZERO full
#    nodes free all of 2026-10-28), and devel's +240 priority is what gets it
#    promptly; its 2 h cap fits a smoke test plus a sweep.
./olivia.sh cluster                      # confirm a 1-node × 4-GPU shape schedules
QOS=devel TIME_LIMIT=02:00:00 ./olivia.sh server start glm53_flash
./olivia.sh server watch
#
#    ✅ ANSWERED (job 2673772): flashinfer is NOT needed. The model loads and
#    captures 51/51 + 2/2 without it, on FLASH_ATTN_MLA_SPARSE. The "cute-dsl skew"
#    was also our own bug — `--no-deps` silently dropped flashinfer's
#    nvidia-cutlass-dsl dependency — fixed in 36c7858, but it gates nothing.
#    What this preset still needs is one serve to validate the max_num_seqs cap.

# Fallback ladder, in the order to try it:
#   a) KV pool too small -> NOT GPU_MEM_UTIL. Measured: 0.90 is already the
#      ceiling — the allocator failed to map 20 MB with ~14 MB free on all four
#      ranks, so 0.93/0.94 OOM earlier. Lower MAX_MODEL_LEN instead (concurrency
#      is pool / context), or CPU_OFFLOAD_GB.
#   b) CUDAGraph capture IMA (upstream #54317, seen on 4×B200)
#                                        -> CUDAGRAPH_MODE=NONE (eager, like glm52/kimi)
#   c) once it serves clean, THEN try    -> ENABLE_SPECULATIVE=1   (MTP, ~7 GiB)
```

### Borealis 27B (preset `borealis`) — Norwegian-centric, Gemma-3, one GH200

Borealis is the **National Library of Norway**'s (NbAiLab) Norwegian-centric
instruct family — a fitting resident for a Norwegian national HPC cluster. We
serve the **27B** full release, `NbAiLab/borealis-27b`.

**Verified from the published config (2026-08-18), not the model card:**
- Arch **`Gemma3ForConditionalGeneration`**, `model_type: gemma3`, **62 layers**,
  **131072** max position embeddings, `torch_dtype: bfloat16`.
- **No `quantization_config` and no quantized release** — the only other artefacts
  are GGUF (llama.cpp, unusable here). So we serve **BF16 ~54 GB**, which still
  fits a **single GH200** (96 GB) at TP=1 with room for KV.
- **Multimodal**: SigLIP vision tower (27 layers, 1152 hidden, 896×896, patch 14).
  Served for **text**; the encoder loads dormant.
- Shares **`vllm-glm53-1-sandbox`** — the third model on that one build, after
  GLM-5.3 and Qwen3.8. Gemma 3 has been supported in vLLM for many releases.

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| BF16 | `NbAiLab/borealis-27b` | ~54 GB | **1 GH200 (TP=1)** — preset `borealis` |
| BF16 (open-licence variant) | `NbAiLab/borealis-open-27b` | ~54 GB | same shape; different licence terms |
| GGUF | `NbAiLab/borealis-27b{,-open}-gguf` | varies | **No** — llama.cpp, not vLLM |

**Runtime (`IS_BOREALIS`).** ⚠️ **Do NOT force `FLASH_ATTN`** — verified the hard
way on 2026-08-18 (job 2032118 died at engine init in 110 s with *`mm_prefix`
(PrefixLM bidirectional attention) requires FlashAttention v4, which does not
resolve for this head_size*). Gemma 3's **text** path is ordinary sliding-window +
full attention, but the **vision tower** makes the config multimodal PrefixLM —
giving the image prefix bidirectional attention — and vLLM only serves that
through FA4, an SM100/Blackwell path unavailable at this head size on Hopper. The
backend is therefore left **unset so vLLM auto-selects**, like the MLA/hybrid
models. This bites even though we only ever send text. Otherwise it is an instruct model that emits neither `<think>` blocks
nor a tool-call grammar vLLM has a parser for, so **no `--reasoning-parser`, no
`--tool-call-parser`, no `--enable-auto-tool-choice`**. The single override is
`MAX_MODEL_LEN=131072`: the generic fallback is 32768, which would silently
discard three quarters of the native window.

> **Status: weights cached (~54 GB, persistent tier), NEVER SERVED.**

```bash
./olivia.sh prefetch borealis                        # done
TIME_LIMIT=00:50:00 ./olivia.sh server start borealis  # 1 GH200, TP=1
```

### Qwen3.8 27B (preset `qwen38`) — dense multimodal on ONE GH200

Qwen3.8 (Alibaba, open weights 2026-08-13/14, **Apache 2.0**) shipped a 27B dense
multimodal model and a 2.4T-A95B MoE. We serve the **27B**, which is close to an
ideal fit for a single Olivia card.

**Verified from the published `Qwen/Qwen3.8-27B-FP8` config + the downloaded
checkpoint (2026-08-17) — not from the model card:**
- Arch **`Qwen3_5ForConditionalGeneration`**, `model_type: qwen3_5` — the **same
  Qwen 3.5 family as Ornith**, but **dense** where Ornith is
  `Qwen3_5MoeForConditionalGeneration`. Registered natively in **vLLM v0.27.1**
  (`registry.py:581`), which is why this needs **no new container** — it shares
  `vllm-glm53-1-sandbox` (vLLM 0.27.1, transformers 5.15.0, DeepGEMM 2.6.1).
- 27B dense, **64 layers**, hidden 5120, 24 q-heads / 4 KV-heads, **262144** native
  context (1M via YaRN, not enabled).
- **HYBRID attention like Ornith**: 48 linear-attn layers + 16 full (full every
  4th) → vLLM must **auto-select** the backend, never forced `FLASH_ATTN`.
- **Multimodal**: 27-layer vision tower (hidden 1152, patch 16), 333 vision
  tensors. Served for **text**; the encoder loads dormant.
- Quant is **DeepSeek-style block-FP8** (`[128,128]`, e4m3, dynamic) ≈ **29 GB**.
  ⚠️ This is the one place it **differs from Ornith**: Ornith is compressed-tensors
  channel/token W8A8 and never touches DeepGEMM, whereas Qwen3.8 rides the **same
  DeepGEMM lane as glm52**. Hence `VLLM_USE_DEEP_GEMM=1` + `WARMUP=skip` and the
  flashinfer FP8-MoE kernel forced off (it is dense — there is no MoE kernel).
- **MTP head genuinely present** — 22 `mtp.*` tensors (`mtp.fc.weight`,
  `mtp.layers.0.*`) of 1606 total, and vLLM registers `Qwen3_5MTP`. **The opposite
  of Ornith**, whose FP8 declared `mtp_num_hidden_layers=1` and shipped none. So
  MTP speculative decode should really work here.

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| block-FP8 | `Qwen/Qwen3.8-27B-FP8` | **~29 GB** | **1 GH200 (TP=1)** — preset `qwen38`; ~56 GB KV headroom |
| BF16 | `Qwen/Qwen3.8-27B` | ~54 GB | 1 GH200, ~32 GB KV — the no-quant-variable fallback |
| AWQ-INT4 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | ~15 GB | 1 GH200 — only worth it to pack several models per node |
| NVFP4 / GGUF / MLX | various | — | **No** — Blackwell FP4, llama.cpp, Apple |
| MoE flagship | `Qwen/Qwen3.8-2.4T-A95B(-FP8)` | ~2.4 T params | Far beyond 12 GPUs — not attempted |

**Runtime specifics (`IS_QWEN38`)**: `--tool-call-parser qwen3_xml`,
`--reasoning-parser qwen3` (same family as Ornith), `--enable-auto-tool-choice`,
`--trust-remote-code`, `--enable-prefix-caching`, `MAX_MODEL_LEN=262144`,
auto-select attention backend, DeepGEMM block-FP8 path. Knobs:
`QWEN38_TOOL_PARSER`, `QWEN38_REASONING_PARSER`, `QWEN38_ENABLE_PREFIX_CACHING`,
`QWEN38_GDN_PREFILL_BACKEND`.

> The GDN prefill backend is **empty (auto) by default here, unlike Ornith's
> forced `triton`**. Ornith pins Triton only because *its* container had a broken
> flashinfer that the build uninstalled; the glm53 container carries a **working
> flashinfer 0.6.17** ("imports cleanly — keeping"). If the flashinfer GDN path
> misbehaves, `QWEN38_GDN_PREFILL_BACKEND=triton` is the known-good fallback.

> **Status: weights cached, NEVER SERVED.** 29 GB on the persistent projects tier,
> 0 incomplete, blobs verified. Nothing below has been run on-cluster yet.

#### Qwen3.8 Usage

```bash
./olivia.sh prefetch qwen38            # already done — 29 GB, persistent tier
./olivia.sh server start qwen38        # 1 GH200, TP=1 (no build needed)
./olivia.sh server watch

# The first thing to try once it serves clean — the MTP head is really there:
ENABLE_SPECULATIVE=1 ./olivia.sh server start qwen38

# If the flashinfer GDN prefill path misbehaves, fall back to Ornith's kernel:
QWEN38_GDN_PREFILL_BACKEND=triton ./olivia.sh server start qwen38
### GLM-5.2 AWQ (preset `glm52_awq`) — INT4, 2-node 8×GH200 (preferred)

`cyankiwi/GLM-5.2-AWQ-INT4` (~415 GiB, 83 safetensors, MIT) is the AWQ-INT4
re-host from the same author as `cyankiwi/GLM-5.1-AWQ-4bit`. Same
`GlmMoeDsaForCausalLM` MoE+DSA arch as the FP8 variant (including the new
skip-topk indexer), so it needs the **same container** as `glm52` — vLLM main
`091386a` + PR#45895 + DeepGEMM `88965b0` — because those requirements are
**architectural (the DSA indexer), not FP8-specific**. So there is **no separate
build**: build `glm52`, serve `glm52_awq` (the kimi/kimi27 pattern). At ~415 GiB
it **fits 8×GH200** (~52 GB/GPU weights, ~44 GB/GPU free for KV), restoring the
**2-node TP=4 + PP=2** path — preferred over the 3-node FP8 — and it fits the
persistent 1 TiB project quota, so `storage: projects` (not the FP8's `work`).

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| AWQ-INT4 | `cyankiwi/GLM-5.2-AWQ-INT4` | ~415 GiB | **2 nodes × 4 GH200** (TP=4 + PP=2) — preferred |
| block-FP8 | `RedHatAI/GLM-5.2-FP8` | ~755 GB | 3 nodes × 4 (preset `glm52`) |

**Runtime deltas from the FP8 path** (auto-applied via `IS_GLM52` + `IS_AWQ`):
the FP8-only block (DeepGEMM, RayExecutorV2 engine-as-actor, PP=3 partition) is
guarded `IS_AWQ==0` and correctly **skips** for AWQ. AWQ instead gets the GLM-5
AWQ MoE-FP16 flashinfer kernel (`VLLM_USE_FLASHINFER_MOE_FP16=1`), expert
parallel (auto for AWQ GLM MoE), and the **PP=2** indexer-aligned layer
partition `VLLM_PP_LAYER_PARTITION=38,40` (a stage boundary must land on a
full-indexer layer or init dies with `KeyError: model.layers.<N>.self_attn.attn`;
the even 39/39 split lands on a skip-topk layer). It shares the FP8 path's
**bf16 KV** (GH200 sparse-MLA `FLASHMLA_SPARSE` rejects fp8 KV) and **eager**
default (capture IMAs on this NGC stack).

> ⚠️ **Unverified on-cluster (first-serve checklist).** Nothing below has run yet
> — the preset is wired from the GLM-5.1-AWQ recipe + the FP8 GLM-5.2 findings:
> 1. **MoE-FP16 kernel** is the right choice for GLM-5.2 AWQ (inherited from 5.1).
> 2. **PP=2 partition `38,40`** clears the skip-topk `KeyError` (fall back to `42,36`).
> 3. **Multi-node PP decode wedge** — expect it (as glm51/glm52); mitigate with
>    proxy serialization, or try glm51's fix (PIECEWISE capture + NCCL all-reduce)
>    vs glm52-FP8's (RayExecutorV2 engine-as-actor). Which one applies is untested.
> 4. Confirm the KV budget: `GPU KV cache size: N tokens` ≥ `max-model-len × concurrency`.

```bash
# 0. Weights already prefetched to the persistent tier (projects):
./olivia.sh prefetch glm52_awq            # cyankiwi/GLM-5.2-AWQ-INT4 -> /cluster/projects/...
# 1. Build once (the FP8 preset — same container):
./olivia.sh build glm52
# 2. Serve the AWQ variant (2 nodes × 4 GPUs, TP=4 + PP=2):
./olivia.sh server start glm52_awq
./olivia.sh server watch
```

### Kimi K2.6 Quantization Options

Kimi K2.6 (Moonshot) is a 1T-parameter MoE (~32B active) with 384 experts, MLA attention, a 262K context window, and a native MoonViT vision encoder (multimodal). It uses the same 2-node shape as glm51 on Olivia: it does **not** fit on a single 4-GPU node.

| Model | Size | Olivia Compatible | Notes |
|-------|------|-------------------|-------|
| `moonshotai/Kimi-K2.6` | ~640 GB | **Yes (Recommended)** | Moonshot's **native int4** (compressed-tensors); fits on 8×GH200. This base repo *is* the int4 checkpoint. |
| `nvidia/Kimi-K2.6-NVFP4` | ~smaller | **No** | Requires Blackwell FP4 tensor cores |
| `amd/Kimi-K2.6-MXFP4` | ~smaller | **No** | AMD MXFP4 (ROCm) |
| `unsloth/Kimi-K2.6-GGUF` | varies | **No** | GGUF is for llama.cpp, not vLLM |
| `moonshotai/Kimi-K2.6-AWQ` | — | **Does not exist** | There is no AWQ repo — use the native-int4 base repo above. |

**Important:**
- The model is the base repo `moonshotai/Kimi-K2.6` — its native int4 is `quant_method: compressed-tensors`, which vLLM auto-detects, so **no `--quantization` flag** is passed.
- **transformers must be `>=4.57.1,<5.0.0`** (a hard requirement from the model card, and incompatible with glm51's `>=5.4.0` — which is why Kimi gets its own container). vLLM **v0.19.1** is the manually-verified stable release; newer K2.6 support is nightly-only.
- Architecture `KimiK25ForConditionalGeneration` ships as `custom_code`, so `--trust-remote-code` is required.
- Per Moonshot's deploy guide the server auto-passes `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data --trust-remote-code`. **Thinking mode is on by default**, so the reasoning parser is required; for instant mode pass `chat_template_kwargs={"thinking": false}` (recommended `temperature` 1.0 thinking / 0.6 instant, `top_p` 0.95).
- The ~640 GB weights download from HF at **serve** time, so the cluster needs `HF_TOKEN` set then (not for the build).

### Kimi K2.6 Usage

```bash
# Build Kimi K2.6 container (vLLM v0.19.1, transformers >=4.57.1,<5.0.0)
./olivia.sh build kimi

# Start Kimi K2.6 server (preset auto-allocates 2 nodes × 4 GPUs, TP=4 + PP=2)
./olivia.sh server start kimi
./olivia.sh server watch

# Override context length (native 262K; default is 131072)
# or direct: MAX_MODEL_LEN=262144 ... run_vllm_server.sh

# Force instant mode (no reasoning extraction) by dropping the reasoning parser
KIMI_REASONING_PARSER="" CONTAINER=vllm-kimi-4-sandbox MODEL=moonshotai/Kimi-K2.6 \
    NUM_NODES=2 TP_SIZE=4 PP_SIZE=2 ./run_vllm_server.sh
```

**Memory requirements for Kimi K2.6 (1T params, ~32B active):**
| Quantization | Model Size | 8×GH200 (768 GB HBM) | Notes |
|--------------|------------|----------------------|-------|
| Native int4 | ~640 GB | ~50–120 GB free (@0.90 util) | Recommended; MLA keeps KV cache compact |
| BF16 | ~2 TB | Won't fit | — |

### Laguna M.1 (preset `laguna`) — FP8, single-node 4×GH200

Laguna M.1 (Poolside, released 2026-04-28, Apache 2.0) is a 225B-total /
23B-active MoE coding model: `LagunaForCausalLM`, 70 layers (3 dense + 67 sparse
MoE), 256 experts with top-k=16, **dense GQA full attention** (64 q-heads /
8 KV-heads, 262K context, YaRN), native reasoning + tool calling. It is
**natively integrated** into transformers (`model_type: laguna`, ≥5.7.0) and
into vLLM since **v0.21.0** (PR#41129 added the arch and the `poolside_v1`
tool/reasoning parsers). Unlike GLM-5.x (sparse-MLA/DSA) and Kimi (MLA), Laguna
uses ordinary attention — so it keeps the **`FLASH_ATTN` backend** and lets
**CUDAGraph capture run** (no eager override). The standard `CUDAGRAPH_MODE=NONE`
escape hatch still applies if capture misbehaves on the NGC-torch stack.

Default quant is **block-FP8** (`poolside/Laguna-M.1-FP8`, ~225 GB), which fits a
**single GH200 node** at TP=4 with room for KV — no cross-node PP, so none of the
multi-node PP decode wedge that affects glm51/glm52/kimi. FP8 is auto-detected
from `quantization_config`, so no `--quantization` flag and no special DeepGEMM
pin (ordinary block-FP8, not glm52's DSA-indexer `fp8_fp4_mqa_logits` path).

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| block-FP8 | `poolside/Laguna-M.1-FP8` | ~225 GB | **1 node × 4 GH200** (TP=4); ~150 GB free for KV @0.90 util |
| BF16 | `poolside/Laguna-M.1` | ~450 GB | 2 nodes × 4 (TP=4 + PP=2) — not the default preset |
| NVFP4 | `poolside/Laguna-M.1-NVFP4` | ~115 GB | **No** — needs Blackwell FP4 tensor cores (GH200 is Hopper) |

**Runtime specifics auto-applied by `run_vllm_server.sh` when `MODEL` contains
`Laguna`** (`IS_LAGUNA`): `--tool-call-parser poolside_v1`, `--reasoning-parser
poolside_v1`, `--enable-auto-tool-choice`, `--trust-remote-code`, thinking-mode
chat-template kwargs (`--default-chat-template-kwargs '{"enable_thinking":
true}'`), `MAX_MODEL_LEN=131072` default (native 262K), `FLASH_ATTN` backend, and
CUDAGraph capture left on. All parsers/knobs are overridable
(`LAGUNA_TOOL_PARSER`, `LAGUNA_REASONING_PARSER`, `LAGUNA_ENABLE_THINKING=0` for
instant mode). Expert parallel is **not** auto-enabled (single-node TP already
shards the 256 experts); set `ENABLE_EXPERT_PARALLEL=1` to try it.

> ⚠️ **Verify on first build/serve** (sourced from the model card / vLLM PR, not
> yet run on Olivia): the `poolside_v1` parser names, the
> `--default-chat-template-kwargs` flag, and that **v0.21.0 compiles on NGC
> 26.03** (the inherited default — the validated vLLM-0.21 Kimi container builds
> there). If the csrc/`libtorch_stable` ABI wall bites, bump
> `NGC_PYTORCH_TAG=26.05-py3` (the glm52 base).

### Laguna M.1 Usage

```bash
# 1. Prefetch the FP8 weights into the persistent HF cache (detached, resumable)
./olivia.sh prefetch poolside/Laguna-M.1-FP8

# 2. Build the container (vLLM v0.21.0, transformers >=5.7.0)
./olivia.sh build laguna

# 3. Start the server (single node × 4 GPUs, TP=4)
./olivia.sh server start laguna
./olivia.sh server watch

# Serve instant (no reasoning extraction)
LAGUNA_ENABLE_THINKING=0 ./olivia.sh server start laguna
# or direct: CONTAINER=vllm-laguna-1-sandbox MODEL=poolside/Laguna-M.1-FP8 ./run_vllm_server.sh
```

### Ornith 1.0 (presets `ornith` / `ornith_gh200`) — Qwen3.5 hybrid-attn MoE, FP8

Ornith 1.0 (Deep Reinforce, released 2026-06-25, MIT) is an agentic-coding model
family that learns its own RL scaffolds. It ships in four sizes — 9B Dense, 31B
Dense, 35B MoE, 397B MoE — with the 9B/35B/397B post-trained on **Qwen 3.5** and
the 31B on Gemma 4. We serve the two MoE sizes, arch
`Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`), **256K context**:
- **397B MoE** (`ornith`) — the flagship (~512 experts). *Config numbers inferred from the model card; not yet inspected on-cluster.*
- **35B MoE** (`ornith_gh200`) — 40 layers, 256 experts (top-8, **~3B active**), 16 q-heads / 2 KV-heads, shared expert. Deep Reinforce's speed/quality sweet spot.

> **⚠️ Architecture reality (verified on-cluster 2026-07-16 from the 35B FP8
> checkpoint — the model card is misleading here):**
> - **HYBRID attention, NOT plain GQA.** Every layer has BOTH a Gated-DeltaNet
>   `linear_attn` mixer (conv1d + `in_proj_a/b/qkv/z`) AND a full `self_attn` —
>   the Qwen3-Next/3.5 lineage. So vLLM must **auto-select** the backend (linear
>   layers need a mamba/GDN kernel + recurrent state); we do **not** force
>   `FLASH_ATTN`. CUDAGraph/KV handling is vLLM's hybrid path, not Laguna-like.
> - **MULTIMODAL.** Ships a vision tower (`model.visual.*`) + video preprocessor.
>   We serve it for **text** (coding); the encoder just loads (dormant).
> - **MTP declared but WEIGHTS ABSENT.** `config.text_config.mtp_num_hidden_layers=1`,
>   but the FP8 export contains **zero** MTP/nextn tensors (layers stop at 39). So
>   **MTP speculative decode does not work on the published FP8 checkpoints** and is
>   **off by default** (`ENABLE_SPECULATIVE=1` only against a checkpoint that has
>   the head — e.g. a community `-MTP` variant or possibly the BF16 base, untested).
> - **FP8 is compressed-tensors channel/token W8A8** (weights per-channel static,
>   activations per-token dynamic), **not** DeepSeek block-FP8 → **DeepGEMM is not
>   used** (the build pins it only for main compatibility). vLLM auto-detects it →
>   no `--quantization` flag.

Ornith emits `<think>` reasoning blocks and uses the `qwen3_xml` tool parser +
`qwen3` reasoning parser. The only official quants are **FP8**
(`Ornith-1.0-{35B,397B}-FP8`) + GGUF — **no AWQ/INT4** — so both presets serve FP8.

**Two deployments, matched to what each size actually needs** — sharing **one
container** (`vllm-ornith-1-sandbox`, since both are `qwen3_5_moe`), like
`kimi`/`kimi27`:

| Preset | Model | Shape | Rationale |
|--------|-------|-------|-----------|
| `ornith` | 397B FP8 (~400 GB) | **2 nodes × 4 GH200**, TP=4 + PP=2 | The flagship genuinely needs multi-GPU capacity: ~400 GB doesn't fit one node (384 GB); 8×GH200 = 768 GB holds it with ~280 GB KV (~18 seqs @256K). Cross-node PP over Slingshot (not TP), like glm51. |
| `ornith_gh200` | 35B FP8 (~35 GB) | **1× GH200 card**, TP=1 | The 35B only needs one GPU — reserving a whole node for it is waste. TP=1 (no cross-GPU comm) = max single-user throughput. ~35 GB weights + ~10 GB KV @256K on a 96 GB card. (MTP would be the natural extra speedup, but the FP8 checkpoint lacks the head.) |

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| FP8 W8A8 (channel/token) | `deepreinforce-ai/Ornith-1.0-397B-FP8` | ~400 GB | **2 nodes × 4 (TP=4 + PP=2)** — `ornith`; ~280 GB KV headroom |
| FP8 W8A8 (channel/token) | `deepreinforce-ai/Ornith-1.0-35B-FP8` | ~35 GB | **1 GH200 (TP=1)** — `ornith_gh200`; huge KV headroom |
| BF16 | `deepreinforce-ai/Ornith-1.0-{35B,397B}` | 2× | Fits at 2× the GPUs; may retain the MTP head (untested) |
| GGUF | `deepreinforce-ai/Ornith-1.0-{9B,35B}-GGUF` | varies | **No** — llama.cpp, not vLLM |

**Build (`build_vllm_gh200.sh`, preset `ornith`, one container for both sizes):**
`qwen3_5_moe` is a **new architecture** — older vLLM rejects it (`architectures
['Qwen3_5MoeForConditionalGeneration'] are not supported`, vllm#35344), and
transformers 5.x renamed the config to `Qwen3_5MoeTextConfig` which pre-5.x-aware
vLLM can't load (vllm#36236). So the preset builds **vLLM main** (like glm47) +
**transformers ≥5.8.1**, on **NGC 26.05** (main needs the newer `torch::stable`
ABI — the glm52 lesson; 26.03 fails the `csrc/libtorch_stable` CUDA compile). It
pins a **main-matching DeepGEMM** (`88965b0781`) for build compatibility only —
Ornith's channel/token FP8 does not use DeepGEMM. The model card claims vLLM
≥0.19.1 but that predates the transformers-5.x fix, so `main` is safer. All
`ornith*` aliases build the one shared container.

**Runtime specifics auto-applied by `run_vllm_server.sh` when `MODEL` contains
`Ornith`** (`IS_ORNITH`): `--tool-call-parser qwen3_xml`, `--reasoning-parser
qwen3`, `--enable-auto-tool-choice`, `--trust-remote-code`,
`--enable-prefix-caching`, `MAX_MODEL_LEN=262144`, **auto-select attention
backend** (hybrid linear+full attn — not forced `FLASH_ATTN`), and **MTP off** (the
FP8 checkpoint has no MTP head). Shape is set from the preset by `olivia.sh`
(`ornith` → NUM_NODES=2, TP=4, PP=2; `ornith_gh200` → 1 GPU, TP=1). Beyond that:

- **`ornith_gh200` (TP=1, single node):** CUDAGraph capture auto-selected. The
  single-GPU case also skips the 4-GPU `CUDA_VISIBLE_DEVICES` reorder so it isn't
  handed non-existent device ids.
- **`ornith` (TP=4 + PP=2, 2 nodes):** like glm51 this crosses a node boundary
  over Slingshot, so it inherits the **multi-node PP machinery**: PP (not TP)
  across nodes, **PIECEWISE CUDAGraph** default (trips `--disable-custom-all-reduce`
  → graph-safe NCCL, the glm51 de-wedge), and **engine-as-actor Ray**
  (`--data-parallel-backend=ray`, like glm52 — vLLM main's external Ray bootstrap
  needs it or init fails with `ActorHandleNotFoundError`). Same **multi-node PP
  decode wedge** risk as glm51/glm52 — if PIECEWISE doesn't de-wedge (or IMAs on
  the hybrid), fall back to `CUDAGRAPH_MODE=NONE` (eager) + `anthropic_proxy.py`
  request serialization.

Knobs override: `ORNITH_TOOL_PARSER`, `ORNITH_REASONING_PARSER`,
`ORNITH_ENABLE_PREFIX_CACHING`; `ENABLE_SPECULATIVE=1` (+ `ORNITH_MTP_METHOD`,
`ORNITH_MTP_SPECULATIVE_TOKENS`) only if you point at a checkpoint that has the MTP
head. Expert parallel is **not** auto-enabled; `ENABLE_EXPERT_PARALLEL=1` to try it.

**✅ `ornith_gh200` SERVES (validated 2026-07-16, vLLM main pinned `251f7e4`, NGC 26.05).**
Single GH200, TP=1, CUDAGraph FULL capture (captures cleanly on the hybrid, unlike
Kimi/glm52), **~174 tok/s single-stream**, 0 failures 1→64 (~4560 tok/s @64). See
README `## Performance`. Three things were required to get the Qwen3-Next hybrid up
on this stack, all now codified (build) or defaulted (serve):
> 1. **Drop skewed flashinfer.** vLLM main pulls flashinfer 0.6.14, whose eagerly-
>    imported Blackwell kernel references `cutlass.cute.nvgpu.OperandMajorMode`
>    (absent in the container's cute-dsl) → `import flashinfer` throws, crashing
>    engine init (both the GDN prefill and the FP8-MoE oracle probe it). The build
>    now import-tests flashinfer and **uninstalls it if broken** (self-guarding;
>    Hopper needs none of its Blackwell kernels).
> 2. **Skip the `ll_bf16` cute-dsl router-GEMM warmup.** Its availability check only
>    verified `cutlass.cute` (present) not `quack` (absent) → `ModuleNotFoundError:
>    quack` at warmup. The build pypatches the check to also require `quack`.
> 3. **Force the Triton/FLA GDN prefill kernel** (`run_vllm_server.sh` IS_ORNITH:
>    `--additional-config '{"gdn_prefill_backend":"triton"}'`, `ORNITH_GDN_PREFILL_BACKEND`)
>    — the default "auto" picks flashinfer on Hopper, which we removed.
>
> Result: an all-Triton/CUTLASS path, zero flashinfer. (Two shared build-script bugs
> were also fixed en route — `NGC_PYTORCH_TAG` not forwarded into the Phase-3
> container, and the verify importing vLLM from the `/opt/vllm` source tree instead
> of the install.) `VLLM_VERSION` is now **pinned to `251f7e4`** (the commit that
> built+served) in the build preset, and a **from-scratch gated rebuild re-confirmed**
> the flashinfer/quack fixes reproduce a working serve (no hand-patching).
>
**✅ The 2-node `ornith` (397B) also SERVES (validated 2026-07-17).** 8×GH200
(TP=4 + PP=2), ~400 GB W8A8 loads in ~106 s, **~81 tok/s single-stream, 0 failures
1→16, ~845 tok/s @16 — and NO multi-node PP decode wedge** (engine-as-actor
RayExecutorV2 + PIECEWISE capture, the glm52 lesson). Getting there took two
multi-node fixes on top of the shared serving fixes:
> 1. **Device-index (`env -u CUDA_VISIBLE_DEVICES`).** On `251f7e4` the
>    EngineCoreActor computes physical GPU ids for the WHOLE world (8 = TP4×PP2)
>    by indexing `CUDA_VISIBLE_DEVICES` — but that's the node-local 4 GPUs, so id 4
>    IndexErrors (`Error computing device indices ... local range [0,8) base
>    "0,1,2,3"`). Fixed by leaving CVD *unset in the container* so vLLM returns raw
>    ids and Ray places the 8 workers. NB **singularity leaks the host CVD**, so
>    omitting our `--env` isn't enough — `run_vllm_server.sh` prefixes the exec with
>    `env -u CUDA_VISIBLE_DEVICES` (+ `RAY_EXPERIMENTAL_NOSET`) for `NUM_NODES>1`.
>    Executor-independent (v1 runs EngineCore as a Ray actor for both the legacy and
>    engine-as-actor backends; `ENGINE_AS_ACTOR=0` forces the plain one but hits the
>    same bug). Empty CVD ("") ≠ unset — empty = zero GPUs, which breaks `ray start`.
> 2. **cutlass FP8 `weight_loader` double-set.** The compressed-tensors W8A8 FP8
>    cutlass linear sets `weight_loader` twice when linear dims need 16-alignment
>    padding (`pad_n>0` + channel-wise scale) → `AssertionError: Overwriting
>    existing tensor attribute: weight_loader`. Only the 397B's dims are unaligned
>    (the 35B's aren't), so 35B never hit it. Build pypatches out the redundant
>    re-set (`PYPATCH_CUTLASS_WEIGHTLOADER`).

#### Ornith Usage

```bash
# --- Flagship: 397B on 2 nodes (preset `ornith`) ---
# 1. Prefetch the FP8 weights (~400 GB, work tier — resolved automatically)
./olivia.sh prefetch ornith          # -> deepreinforce-ai/Ornith-1.0-397B-FP8 (/cluster/work)

# 2. Build the shared container (vLLM main + transformers>=5.8.1 + NGC 26.05)
./olivia.sh build ornith

# 3. Start (preset auto-allocates 2 nodes × 4 GPUs, TP=4 + PP=2)
./olivia.sh server start ornith
./olivia.sh server watch
./olivia.sh cluster                   # check a 2-node shape can schedule now

# If PIECEWISE capture misbehaves on the hybrid, go eager (+ proxy serialization)
CUDAGRAPH_MODE=NONE ./olivia.sh server start ornith

# --- 35B on a single GH200 card (preset `ornith_gh200`) — same container, no rebuild ---
./olivia.sh prefetch ornith_gh200     # -> deepreinforce-ai/Ornith-1.0-35B-FP8 (projects tier)
./olivia.sh server start ornith_gh200 # TP=1, max single-user throughput (MTP off — no head in FP8)
```

> The two presets now serve **distinct repo ids** (397B vs 35B), so the durable
> router (`model_router.py`) can tell them apart — you can even run both at once
> and route by `model: "ornith"` vs `"ornith_gh200"` (or the repo id / other
> aliases). They share the container, but a 2-node job and a 1-GPU job are
> independent SLURM allocations.

### Batching Proxy for SSH Tunnels

When accessing vLLM over SSH tunnels, streaming responses can be slow due to per-token network overhead. The batching proxy aggregates multiple tokens into single SSE events, reducing network round-trips by ~70%.

**Architecture:**
```
Client <--[batched SSE]--> Proxy:8001 <--[per-token SSE]--> vLLM:8000
         (SSH tunnel)                    (localhost, fast)
```

**Usage:**
```bash
# Enable proxy when starting server
ENABLE_PROXY=1 CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Then tunnel to proxy port instead of vLLM port
ssh -L 8001:localhost:8001 user@<cluster-login-host>...

# Connect chat client to proxy port
python chat_devstral.py localhost --port 8001 --stream
```

**Performance comparison:**
| Mode | Without Proxy | With Proxy |
|------|---------------|------------|
| Non-streaming | 17 tok/s | 17 tok/s |
| Streaming | ~5 tok/s | ~15 tok/s |

The proxy can also be run standalone:
```bash
python vllm_proxy.py --vllm-port 8000 --proxy-port 8001 --batch-tokens 15 --batch-delay-ms 150
```

### Anthropic Compatibility Proxy (Claude Code bridge)

`anthropic_proxy.py` lets Claude Code CLI (and any other Anthropic-Messages-API client) talk to our GLM deployment. It accepts `POST /v1/messages`, translates the request to OpenAI `/v1/chat/completions`, forwards to the upstream, and rewrites the streaming response from OpenAI deltas to Anthropic's block-oriented event model (`message_start` → `content_block_start`/`content_block_delta`/`content_block_stop` per block → `message_delta` → `message_stop`).

**Architecture** (default — direct to vLLM via the existing `olivia.sh tunnel up` forward):
```
Claude Code ──/v1/messages──► anthropic_proxy ──/v1/chat/completions──► vllm:8000
 (local)       (Anthropic)      (localhost:8002)                         (GLM-5.1,
                                                                          via SSH tunnel)
```

**Usage (local machine):**
```bash
# 1. Tunnel to the cluster (forwards localhost:8003 → vllm:8000 on the GPU node).
./olivia.sh tunnel up

# 2. Start the Anthropic proxy locally (default listen 127.0.0.1:8002).
python anthropic_proxy.py --model cyankiwi/GLM-5.1-AWQ-4bit

# 3. Point Claude Code at it.
export ANTHROPIC_BASE_URL=http://localhost:8002
export ANTHROPIC_AUTH_TOKEN=sk-any-nonempty-string   # value is ignored, must be set
claude
```

Optionally chain through the batching proxy for SSE compaction over the tunnel: start the server with `ENABLE_PROXY=1` (so `vllm_proxy` runs alongside vLLM on the cluster), tunnel 8001 as well, and run `python anthropic_proxy.py --upstream http://localhost:8001 --model …`.

**Options:**
- `--listen-host` / `--listen-port` — bind address (default `127.0.0.1:8002`)
- `--upstream` — OpenAI-compatible URL (default `http://localhost:8003`, the olivia.sh tunnel's local port → direct vLLM; `:8000` is avoided because another local dev service may bind it). Set to `http://localhost:8001` to chain through `vllm_proxy` (requires `ENABLE_PROXY=1` server-side).
- `--model` *(required)* — the model name forwarded to vLLM (e.g. `cyankiwi/GLM-5.1-AWQ-4bit`). All Anthropic model names in client requests are remapped to this single value.
- `-v` / `--verbose` — log request/response bodies for debugging.
- `--keepalive-interval N` — seconds of upstream idle before sending a 1-token dummy completion to keep vLLM's Ray compiled-DAG warm (default `180`, set `0` to disable). Works around a multi-node PP instability where the engine wedges on idle → active transitions; pinging the DAG periodically avoids long idle windows.
- `--no-serialize` — disable the request-serialization guard (on by default). Without this flag, the proxy holds an `asyncio.Lock` around every upstream `/v1/chat/completions` call so only one request is in flight at a time. This works around the glm51 multi-node PP decode wedge (see "Known issue" above) — Claude Code's parallel `generate_session_title` + `repl_main_thread` pattern reliably triggers the wedge without it. Disable only for testing or if the underlying vLLM bug has been fixed upstream.
- `--dump-requests DIR` — diagnostic: writes every `/v1/messages` request body (plus headers, with `authorization` / `x-api-key` redacted) to `DIR/req-<timestamp>-<id>.json`. Used to capture the exact Claude Code payload that triggers a wedge, so it can be replayed verbatim via curl to isolate the trigger. Off by default.

**What gets translated:**
- Request: `system` (string or content-block list) → `role: system` message; `messages` with `text`/`tool_use`/`tool_result` blocks → OpenAI messages + `tool_calls` + `role: tool` results; `tools` → OpenAI `function` tools; `tool_choice` (`auto`/`any`/`none`/`tool`) → OpenAI equivalents; `stop_sequences` → `stop`.
- Response: `reasoning_content` → `thinking` block; `content` → `text` block; `tool_calls` → `tool_use` block with incremental `input_json_delta` events; `finish_reason` → `stop_reason` (`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`).

**What is dropped silently (intentional):**
- `cache_control` fields (GLM has no prompt cache)
- `thinking` request param / extended-thinking config (GLM's reasoning parser always runs when configured server-side)
- `image` content blocks → replaced with `[image omitted]` text

**Known limitations:**
- `input_tokens` in the initial `message_start` event is reported as `0` (real count is only available in the upstream's final usage chunk). Accurate `output_tokens` are reported in `message_delta`.
- `thinking` blocks from prior assistant turns are dropped when replaying history — GLM regenerates reasoning each turn.

### Container/Cache Structure

**Shared containers on the cluster** (`CONTAINER_DIR`):
- `vllm-{model}-{index}-sandbox/`: Singularity sandbox (writable container)
- `vllm-{model}-{index}.sif`: Compressed Singularity image (optional)

Naming examples:
- `vllm-glm51-1-sandbox` - GLM-5.1 build #1
- `vllm-glm47-1-sandbox` - GLM-4.7 build #1
- `vllm-kimi-1-sandbox` - Kimi K2.6 build #1
- `vllm-devstral-1-sandbox` - Devstral build #1
- `vllm-generic-1-sandbox` - Generic build #1

**Persistent model cache (`HF_HOME`)**: HuggingFace model weights (hundreds of GB) live in a persistent project area — e.g. `/cluster/projects/<proj>/huggingface`. They must **not** sit on `/cluster/work`, which NRIS auto-purges after 21–42 days (this silently deleted a ~430 GB GLM-5.1 AWQ cache once, leaving only metadata + dangling symlinks).

> 🔥 **This happened again, to everything — verified 2026-08-17.** *Every* work-tier
> model cache had been purged to metadata-only: `RedHatAI--GLM-5.2-FP8` (632K),
> `cyankiwi--GLM-5.1-AWQ-4bit` (380K), `deepreinforce-ai--Ornith-1.0-397B-FP8`
> (568K), `moonshotai--Kimi-K2.6` (328K), `moonshotai--Kimi-K2.7-Code` (608K),
> `poolside--Laguna-M.1-FP8` (228K). `blobs/` was empty and every snapshot symlink
> dangled. Only the **projects**-tier copies survived. So **every `"storage":
> "work"` preset — `glm51`, `glm52`, `glm53`, `kimi`, `kimi27`, `ornith` — will
> silently re-download its weights inside the GPU allocation** unless you
> `prefetch` first. `du -sh $HF_HOME/hub/models--*` is the 2-second check: a
> multi-hundred-GB model reading as KB is a purged cache, not a present one.
> **Always `./olivia.sh prefetch <preset>` before `server start` on a work-tier
> preset that has been idle for more than ~3 weeks.** `HF_HOME` is set in `mise.local.toml` and forwarded to jobs by `olivia.sh`; populate it with `./olivia.sh prefetch`. Note `/cluster/work` and `/cluster/projects` are different Lustre filesystems, so migrating weights between them is copy-then-delete (no cross-FS hardlinks).

**Ephemeral compile/JIT caches**: Triton, DeepGEMM, TorchInductor and vLLM compile caches are regenerable and stay under `$PWD/cache/` on `/cluster/work` (`run_vllm_server.sh` sets `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`). They're large and high-churn, so auto-purge is harmless — but they must avoid the small home quota (defaulting them to `~/.triton` crashes jobs mid-profile with `Disk quota exceeded`).

**Deployed scripts — per-branch isolation (`deploys/<DEPLOY_KEY>/`)**: `olivia.sh` deploys `run_vllm_server.sh`, `build_vllm_gh200.sh`, the GLM-5.1 chat template, and `patches/` into `${CONTAINER_DIR}/deploys/<DEPLOY_KEY>/`, and submits jobs from there (`sbatch ${DEPLOY_DIR}/<script>`). `DEPLOY_KEY` defaults to the local **git branch**. This is the fix for the **concurrent-agent clobber**: when multiple agents develop different presets on different branches, their `run_vllm_server.sh`/`build` scripts diverge (each adds its own `IS_<preset>` detection); a single shared deployed copy let whoever deployed last before another's `sbatch` win (SLURM snapshots the batch script *at submit*), so jobs silently ran the wrong preset's config (e.g. generic 32K context, no reasoning parser). Per-branch dirs make collisions structurally impossible, and an agent on this scheme neither clobbers nor is clobbered by the old shared path. Sandboxes, `logs/`, and the `$PWD/cache` compile cache stay **shared** (submit cwd is still `CONTAINER_DIR`). Override `DEPLOY_KEY` to share (`DEPLOY_KEY=main`) or split deploy dirs explicitly. `server start` does **not** auto-deploy — run `server deploy` (or `server restart -d`) after editing a script.

**Job-name isolation (`vllm-<DEPLOY_KEY>`)** — the deploy-key isolation above fixed *script* clobber but the SLURM **job name** stayed hardcoded `vllm-server`, so two agents' concurrently-running servers were indistinguishable to `olivia.sh`'s squeue-based resolution: `watch`/`logs`/`tunnel` could target — and `server cancel` could **kill** — the other agent's server. Now `olivia.sh server start` names each branch's job `vllm-<DEPLOY_KEY>` (override `SERVER_JOB_NAME`), and **action** resolution (`find_job_id`/`find_vllm_node`) matches your branch's job by an **exact** first-field name match — never another agent's, and never an in-flight `build-vllm-gh200`. **Display** (`server status`, `cluster`) stays broad — it enriches every `vllm-*` server so you still see all agents' jobs. Legacy `vllm-server` jobs started before this change are still *shown* but must be **restarted** to be *targetable* under the new scheme.

## Troubleshooting

**Allocation failures / jobs stuck pending.** Before debugging a job that won't start, rule out a cluster maintenance window: run `scontrol show reservation` on the cluster. A `maintstop` reservation covers all nodes and only permits specific maintenance accounts; SLURM reports `ReqNodeNotAvail, Reserved for maintenance` without useful context. Multi-node jobs (e.g. the 2×4-GPU `glm51` or 3×4-GPU `glm52`/`glm53` presets) are especially sensitive.
