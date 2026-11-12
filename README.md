# vLLM for NVIDIA GH200 (GraceHopper) on HPC Clusters

Build and run [vLLM](https://github.com/vllm-project/vllm) on NVIDIA GH200 ARM64 GPUs, specifically optimized for the NRIS Olivia HPC cluster — from single-node 4-GPU models up to 1T-parameter MoE models sharded across 3 nodes / 12 GPUs. Includes full-featured CLI tooling, a streaming chat client, an Anthropic/Claude-Code bridge, and performance optimizations for high-latency SSH tunnel connections.

## Features

- **Preserves NGC PyTorch** - Builds vLLM without overwriting NVIDIA's custom PyTorch
- **Model Presets** - Build + serve recipes for GLM-4.7, GLM-5.1, GLM-5.2 (FP8 + INT4), GLM-5.3, Kimi K2.6/K2.7, Laguna M.1, Ornith 1.0, Qwen3.8-27B, Borealis-27B, Gemma-4, Devstral, Llama, and Qwen
- **Multi-node serving** - TP=4 intra-node + pipeline parallel across nodes over Slingshot, with an auto-bootstrapped Ray cluster (GLM-5.1/5.2 and Kimi span 2–3 nodes)
- **Reproducible builds** - Pin a vLLM commit and graft not-yet-released upstream PRs from committed snapshots (`VLLM_PATCHES`), so a container rebuilds byte-identically
- **GH200 Optimizations** - NCCL/NVLink tuning, optimal GPU ordering, Flash Attention, DeepGEMM/FP8 paths
- **Unified CLI** - Single command interface with SSH ControlMaster (one 2FA per session) + durable reconnect
- **Claude Code bridge** - `anthropic_proxy.py` serves the Anthropic Messages API (thinking/reasoning + tool calls) on top of the OpenAI endpoint
- **reasoning_tokens** - Reported on `/v1/chat/completions` usage for the reasoning models (Kimi, GLM-5.x)
- **Streaming Proxy** - Batches SSE tokens for ~3x faster streaming over SSH tunnels
- **Smart Monitoring** - Multi-phase server watch with GPU loading progress and live throughput

## Quick Start

```bash
# Show help and available commands
./olivia.sh

# Check cluster status
./olivia.sh status

# Build a GLM-4.7 container
./olivia.sh build glm47

# Start the server
./olivia.sh server start glm47

# Monitor server startup (GPU loading → health → serving)
./olivia.sh server watch

# Connect and chat
./olivia.sh chat
```

## Prerequisites

- SSH access to your HPC cluster with 2FA configured
- Python 3.8+ with `requests` and `rich` libraries (for chat client)
- HuggingFace token for gated models (Llama, etc.)

## Configuration

This repo ships with **safe generic defaults**. You must configure your cluster settings via environment variables.

### CLI (`olivia.sh`) configuration

Set these before using `./olivia.sh`:

```bash
export REMOTE_HOST=<your-cluster-login-host>
export REMOTE_CONTAINER_DIR=<path-on-cluster-for-containers>

# Optional (defaults shown)
export REMOTE_USER=$USER
export REMOTE_PORT=8000   # cluster-side vLLM port
export LOCAL_PORT=8003    # local forward; avoids :8000 (another local dev service may bind it)
```

### Direct scripts configuration

The direct scripts require `CONTAINER_DIR` (the directory containing your Singularity sandboxes/SIFs on the cluster):

```bash
export CONTAINER_DIR=<path-on-cluster-for-containers>
```

## CLI Reference

### `./olivia.sh`

Unified CLI for all operations. Uses SSH ControlMaster for single 2FA authentication per session.

| Command | Description |
|---------|-------------|
| `chat` | Connect to vLLM and start interactive chat |
| `build` | Build vLLM containers |
| `server` | Manage vLLM server (start, stop, logs) |
| `proxy` | Durable multi-model router on the small partition (start, tunnel, status) |
| `tunnel` | Manage SSH tunnel to GPU node |
| `status` | Show cluster and connection status |
| `--kill-all` | Close tunnel and SSH connection |

### Chat Module

```bash
./olivia.sh chat               # Connect and start interactive chat
./olivia.sh chat --port 9000   # Use different local port
./olivia.sh chat --tunnel-only # Just set up tunnel, don't start chat
./olivia.sh chat --no-stream   # Disable streaming responses
```

### Build Module

```bash
./olivia.sh build              # Show build help
./olivia.sh build --presets    # List available model presets
./olivia.sh build --list       # List existing containers on cluster

# Build containers (build + serve presets share the same name)
./olivia.sh build glm52        # GLM-5.2 (FP8, pinned vLLM commit + PR#45895 snapshot)
./olivia.sh build kimi         # Kimi K2.6/K2.7 (shared container)
./olivia.sh build glm51        # GLM-5.1
./olivia.sh build glm47        # GLM-4.7 (single node)
./olivia.sh build devstral     # Devstral

# Build options
./olivia.sh build glm47 --index 2    # Build second container (safe, won't touch existing)
./olivia.sh build glm47 --force      # Rebuild existing container
./olivia.sh build glm47 --sif        # Create SIF image after build
./olivia.sh build glm47 --vllm v0.6.6  # Override vLLM version (branch, tag, or commit SHA)
```

**Safety:** Builds fail by default if a container already exists. Use `--index N` to create a new container or `--force` to explicitly overwrite.

### Server Module

```bash
./olivia.sh server                   # Show server help
./olivia.sh server list              # List available containers
./olivia.sh server status            # Show running server status

# Start servers (preset with default model; multi-node presets auto-allocate nodes)
./olivia.sh server start glm52       # GLM-5.2 (3 nodes × 4, eager) — see HF_HOME note below
./olivia.sh server start kimi        # Kimi K2.6 (2 nodes × 4, eager)
./olivia.sh server start glm51       # GLM-5.1 (2 nodes × 4)
./olivia.sh server start glm47       # GLM-4.7 (single node, 4 GPUs)

# Start with options
./olivia.sh server start glm47 --index 2              # Use vllm-glm47-2-sandbox
./olivia.sh server start glm47 --model custom/model   # Override default model
./olivia.sh server start -c vllm-custom-1-sandbox -m my/model  # Explicit container

# Monitoring and management
./olivia.sh server watch             # Smart monitor with progress bars
./olivia.sh server logs              # Tail logs of running server
./olivia.sh server ssh               # Open shell on GPU node
./olivia.sh server restart glm47     # Cancel running job and restart
./olivia.sh server cancel            # Cancel running vLLM job
./olivia.sh server deploy            # Upload run_vllm_server.sh to cluster
```

**Watch command phases:**
1. **WAITING** - Waits for SLURM job to be submitted
2. **PENDING** - Job queued, waiting for resources
3. **LOADING** - GPU memory increasing as weights load (progress bar)
4. **INIT** - Weights loaded, checking /health endpoint
5. **SERVING** - Live throughput monitoring (tok/s, active requests, KV cache)

### Tunnel Module

```bash
./olivia.sh tunnel             # Show tunnel status
./olivia.sh tunnel up          # Open tunnel to vLLM server
./olivia.sh tunnel down        # Close tunnel
```

### Proxy Module (durable multi-model router)

A CPU-only reverse proxy on Olivia's **`small`** partition (up to a 7-day
walltime) that gives clients **one stable endpoint** routing to whichever GPU
server is live. Select a model by the request's `model` field — a preset name
(`glm51`, `kimi27`, ...), alias, or served repo id — and the router finds the
backend by listing running `vllm-*` jobs and **probing each `/v1/models`**, so it
works regardless of job naming; you never need to know the node or container
index. It replaces the (now-removed) login-node relay with a queue-system job
(the NRIS-policy-correct place for a long-lived process), and **auto-stops after
30 min with no GPU servers up** so an idle CPU job doesn't bill its reservation.
Full design + Sigma2 policy analysis:
[`plans/proposed/small_partition_proxy.md`](plans/proposed/small_partition_proxy.md).

```bash
./olivia.sh proxy start        # Deploy + submit the router (small partition)
./olivia.sh server start glm51 # Start GPU servers as usual; router picks them up (~15s)
./olivia.sh proxy tunnel       # Forward localhost:8003 -> router node
curl localhost:8003/v1/models  # See which presets are currently live
./olivia.sh proxy status       # Router job + live models
./olivia.sh proxy stop         # Cancel the router (it bills its small reservation while up)
```

> Compute nodes aren't internet-facing, so the laptop still tunnels in through
> the login node — but the tunnel target (the `small` node) is now stable for the
> job's lifetime instead of moving on every GPU job restart. Live-validated
> 2026-06-22: a cross-model eval sweep routed glm52 + kimi27 concurrently through
> one router endpoint (see the plan doc's checklist for what's left).

#### Using the proxy (clients)

The router is an **OpenAI-compatible** HTTP endpoint. Reach it at:

- **In-cluster** (e.g. an eval/batch job on another node): `http://<router-node>:8080`
  directly — no tunnel. Find the node with `./olivia.sh proxy status`.
- **From a laptop**: `./olivia.sh proxy tunnel`, then `http://localhost:8003`.

Pick the model with the request's **`model` field** — a preset name (`glm51`,
`kimi27`, `laguna`, ...), an alias, or the served repo id. `GET /v1/models` lists
what's live right now. Supported paths: `/v1/chat/completions`, `/v1/completions`,
`/v1/models`.

```bash
# Discover what's live
curl http://localhost:8003/v1/models

# OpenAI-compatible client (Python)
#   from openai import OpenAI
#   c = OpenAI(base_url="http://localhost:8003/v1", api_key="x")   # api_key unused unless OLIVIA_PROXY_TOKEN set
#   c.chat.completions.create(model="glm51", messages=[...])      # "glm51" -> live backend

# Anthropic clients / Claude Code — front the router with anthropic_proxy.py:
python anthropic_proxy.py --model glm51 --upstream http://localhost:8003
export ANTHROPIC_BASE_URL=http://localhost:8002 ANTHROPIC_AUTH_TOKEN=x && claude
```

> **In-cluster clients:** compute nodes inherit an `http_proxy` (squid) for
> internet egress — it will wrongly route `localhost` and the router/backend host
> through squid (→ 503). Set `no_proxy=localhost,127.0.0.1,<router-node>` (or
> unset `http_proxy`) in the client before calling the router. (The router job
> itself already does this.) If `OLIVIA_PROXY_TOKEN` is set on the router, send it
> as a `Bearer` token / `x-api-key`.

## Model Presets

| Preset | Default Model | GPUs | Container | Notes |
|--------|---------------|------|-----------|-------|
| `glm51_v19` (alias `glm51`) | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-1` | TP=4 + PP=2, vLLM v0.19.0. Multi-node PP decode wedge → serve behind `anthropic_proxy.py` serialization |
| `glm51_v20` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-2` | vLLM v0.20.0 + RayExecutorV2. **Quarantined** (same wedge) |
| `glm52` | `RedHatAI/GLM-5.2-FP8` | 12 (3 nodes × 4) | `vllm-glm52-1` | TP=4 + PP=3, block-FP8 (~755 GB). vLLM main pinned `091386a` + PR#45895 snapshot. Eager; fp8 KV + DeepGEMM |
| `glm52_awq` | `cyankiwi/GLM-5.2-AWQ-INT4` | **8 (2 nodes × 4)** | `vllm-glm53-1` | TP=4 + **PP=2**. AWQ/compressed-tensors INT4 (~411 GB) fits 8 GPUs — **8 GPU-h/hour vs the FP8's 12**. Same DSA skip-topk indexer, so it runs on the v0.27.1 container where PR#45895 is native. Weights on the **persistent** tier, already cached. **Not yet served** |
| `glm53` | `zai-org/GLM-5.3-FP8` *(expected id)* | 12 (3 nodes × 4) | `vllm-glm52-1` (shared) | GLM-5.2's **same base, re-post-trained** → identical arch, so it reuses the glm52 container (no rebuild) and its whole runtime profile. **Weights not public yet** (announced 2026-08-14, open weights promised ~2 weeks out) — confirm repo id + license before prefetching |
| `glm53_v27` | `zai-org/GLM-5.3-FP8` *(expected id)* | 12 (3 nodes × 4) | `vllm-glm53-1` | Same model on **vLLM v0.27.1 + NGC 26.07** (torch 2.13), no PR graft — PR#45895 merged upstream in v0.24.0. The upgrade path off glm52's pinned-main build; **unvalidated, not yet benchmarked** |
| `dsv41` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | **4 (1 node)** | `vllm-glm53flash-1` (shared) | **748B of parameters on one node, by putting the cold ones in Grace memory.** 552B backbone + 196.6B Engram; only 8B active per prompt token / 16B per output token. vLLM sizes it at ~614 GB VRAM vs our 384 GB -- `--engram-config {"cpu_offload":true}` moves the sparsely-accessed Engram hash tables (259.5 GiB, ~47 GiB/rank at TP4, held by only layers 1 and 14) into **pinned host memory** over NVLink-C2C. KV is **890 B/token** (FP4, shared across layers), so weights set capacity, not cache. Arch native from vLLM v0.30.0 -> **no build**. **fp8 KV required.** WARNING: **blocked on FlashInfer** (`FLASHINFER_MLA_SPARSE_DSV41`), the same blocker as `glm53_flash`. **Not yet served** |
| `glm53_flash` | `zai-org/GLM-5.3-Flash` | **4 (1 node)** | `vllm-glm53flash-1` | **A DIFFERENT MODEL from `glm53` above, despite the name.** GLM-5.3-Flash is 320B total / **18B active**, a new arch (`Glm5NextForConditionalGeneration`) and a new pretrain: 45 layers = **34 KDA linear-attention + 11 NoPE sparse-MLA**, 288 experts (8+1 active), Manifold-Constrained Hyper-Connections, natively **multimodal** (image+video), 1M ctx, native FP8 **~306 GiB**, MIT. Fits **ONE node** — tighter than `glm47`'s ~358 GB AWQ — and hybrid attention makes KV unusually cheap (only 11 layers cache, and those are MLA with `kv_lora_rank=512`). Own container built from **source: vLLM v0.32.0 + NGC 26.07** — `glm5_next` is native from **v0.30.0** (*not* v0.29.0, despite the recipe page). **Hopper: BF16 KV only.** MTP head present (as layer 45, and vLLM registers `Glm5NextMTPModel`) but **off by default**. **Not yet served** |
| `glm53_flash_prebuilt` | `zai-org/GLM-5.3-Flash` | 4 (1 node) | `vllm-glm53flash-2` | Same model from the **prebuilt** `vllm/vllm-openai:glm53-flash` image (arm64). **Fallback only** — this was the primary path from 2026-08-31 to 2026-10-28, while `glm5_next` existed in no public vLLM release. Use if the source build regresses |
| `borealis` | `NbAiLab/borealis-27b` | **1** | `vllm-glm53-1` (shared) | **Borealis 27B — National Library of Norway**, Norwegian-centric instruct. Gemma-3 arch (`Gemma3ForConditionalGeneration`), **BF16 ~54 GB, no quantized release**, SigLIP vision tower (served for text), 128K ctx. Single GH200 TP=1; shares the glm53 container. Backend must be **auto-selected, not FLASH_ATTN** (Gemma 3's vision tower makes it multimodal PrefixLM → `mm_prefix` wants FA4, unavailable on Hopper at this head size; forcing it killed engine init). No reasoning/tool parsers. |
| `qwen38` | `Qwen/Qwen3.8-27B-FP8` | **1** | `vllm-glm53-1` (shared) | **Qwen3.8 27B dense multimodal on a SINGLE GH200** (TP=1). Block-FP8 ~29 GB, hybrid linear+full attn (48+16), 262K ctx, Apache 2.0. Arch `Qwen3_5ForConditionalGeneration` — native in vLLM v0.27.1, so it shares the glm53 container with no rebuild. **MTP head confirmed present** (22 `mtp.*` tensors) → `ENABLE_SPECULATIVE=1` is the obvious win. Weights cached on the persistent tier. **Not yet served** |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | `vllm-glm47-1` | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | `vllm-kimi-4` | TP=4 + PP=2, native int4, MLA, multimodal, vLLM 0.21. Eager. reasoning_tokens on chat/completions |
| `kimi27` | `moonshotai/Kimi-K2.7-Code` | 8 (2 nodes × 4) | `vllm-kimi-4` (shared) | Same arch + container as K2.6 (no rebuild); thinking-only |
| `laguna` | `poolside/Laguna-M.1-FP8` | 4 | `vllm-laguna-1` | TP=4, single node. FP8 (~225 GB), dense attention (FLASH_ATTN), CUDAGraph on. vLLM v0.21.0, `poolside_v1` parsers |
| `ornith` | `ornith-ai/Ornith-1.5-397B-FP8` | 8 (2 nodes × 4) | `vllm-glm53flash-1` (shared) | **Ornith 1.5 397B flagship.** TP=4 + PP=2, FP8 W8A8 (418 GB), 256K ctx. No build — qwen3_5_moe is native in the v0.32.1 container. 1.5 ships the MTP head (opt-in). ✅ Serves: **79.6 tok/s single, 1,283 @64**, 4.42M-token pool, 0 failures (job 2777625) |
| `ornith_1n` | `ornith-ai/Ornith-1.5-397B-FP8` | **4 (1 node)** | `vllm-glm53flash-1` (shared) | **The 397B on ONE node**: 20 GiB/GPU of routed experts in Grace memory (`--cpu-offload-params experts`). ✅ Serves: **59 tok/s single-stream, 299 @64** at the 20 GiB default (458K-token pool); aggregate is offload-bandwidth-bound |
| `ornith_122b` | `ornith-ai/Ornith-1.5-122B-FP8` | 4 (1 node) | `vllm-glm53flash-1` (shared) | **New in 1.5.** 122B-A10B, FP8 W8A8 (133 GB, persistent tier), TP=4. ✅ **192 tok/s single-stream, 3,300 @64 with MTP (default on)**, 4.5M-token pool, 17× @256K |
| `ornith_gh200` | `deepreinforce-ai/Ornith-1.0-35B-FP8` | 1 | `vllm-ornith-1` (shared) | **35B on a single GH200 card** (TP=1), ~3B active, hybrid linear+full attn, multimodal, 256K ctx — max single-user throughput. MTP off (no head in FP8). Shares the `ornith` container, `qwen3_xml`/`qwen3` parsers |
| `gemma4` | Gemma 4 (31B, multimodal) | 1–2 | `vllm-gemma4-1` | vLLM v0.19.0, AWQ |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | `vllm-devstral-1` | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | — | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | — | TP=4 |
| `generic` | *(user specified)* | 4 | — | generic defaults |

> Build and serve presets are aligned by name (e.g. `./olivia.sh build glm52` then `./olivia.sh server start glm52`). The **Container** column is where each preset's server looks (`vllm-<name>-<index>-sandbox`); `kimi`/`kimi27` share index 4. See **[CLAUDE.md](CLAUDE.md)** for full per-model guides — memory layout, quant options, known issues, and multi-node architecture.

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
| `ornith-1.5-122b-1m` | `ornith_122b_1m` | needle-verified 2026-11-08 (job 2759667) |
| `ornith-1.0-35b-256k` | `ornith_gh200` | benchmarked 2026-07-16 — its 1.0 container is purge-hollowed |
| `devstral-2-123b-32k` | `devstral` | no current benchmark |
| `llama-3.3-70b-32k` | `llama` | no current benchmark |
| `qwen-2.5-72b-32k` | `qwen` | no current benchmark |
| `glm-5.2-744b-128k-1n` | `glm52_awq_1n` | benchmarked 2026-07-03 |
| `glm-5.2-744b-128k-1n-mtp` | `glm52_awq_mtp` | experimental (local MTP-grafted checkpoint) |

## Performance

Latest measured throughput / latency. **Update this section after every sweep** (with the date + config).

### Overview — peak aggregate throughput per preset, sorted by efficiency

Every figure below is the **largest aggregate** recorded in that preset's own section; the
`per GPU` column divides it by the allocation. Sorted by that, because GPU-hours are what we
actually spend.

| preset | model | GPUs | peak aggregate | per GPU |
|---|---|---:|---:|---:|
| `ornith_gh200` | 35B MoE FP8 | 1 | 4,621 tok/s | **4,621** |
| `qwen38` | 27B dense FP8 | 1 | 3,162 tok/s | **3,162** |
| `borealis` | 27B Gemma-3 BF16 | 1 | 2,666 tok/s | **2,666** |
| `lagunas21` | Laguna S 2.1 FP8 | 4 | 7,751 tok/s | **1,938** |
| `ornith_122b` | Ornith 1.5 122B-A10B FP8, MTP | 4 | 3,300 tok/s | **825** |
| `dsv41` | **748B** + Engram offload | 4 | 2,465 tok/s | **616** |
| `laguna` | Laguna M.1 FP8 | 4 | 2,056 tok/s | **514** |
| `glm53_flash` | 320B/18B `glm5_next` | 4 | 695 tok/s | **174** |
| `glm51` | 744B AWQ | 8 | 895 tok/s | **112** |
| `ornith` | Ornith 1.5 397B FP8, 2 nodes | 8 | 1,283 tok/s | **160** |
| `ornith` (1.0) | Ornith 1.0 397B MoE FP8, 2 nodes | 8 | 845 tok/s | **106** |
| `ornith_1n` | Ornith 1.5 397B, 1 node, 20 GiB experts in Grace | 4 | 299 tok/s | **75** |
| `kimi27` | 1T int4, eager | 8 | 591 tok/s | **74** |
| `glm52` | 744B FP8, eager | 12 | 419 tok/s | **35** |

⚠️ **Read this as a rough ranking, not a benchmark.** The sweeps differ in `max_tokens`
(256 vs 512), concurrency ceiling, prompt length and whether reasoning was on, so the numbers
are not strictly comparable across rows. They are, however, each reproducible from the
configuration recorded in that preset's section.

Two things the ranking makes plain: **small models on one card dominate throughput-per-GPU**,
and among the large models **`dsv41` is the standout** — 616 tok/s/GPU against `glm52`'s 35 for
a comparable parameter count, i.e. ~18× better use of the same allocation. `dsv41` is also the
only large preset serving a **1M context** (see its section for the three operating points).
Presets not listed have never served: `glm53`, `glm52_awq`, `dsv4flash` (blocked), `glm47`.

### Ornith 1.5 `ornith_122b` — 122B-A10B FP8, 1 node × 4 GH200, TP=4 · 2026-11-08 (job 2754478)

**New size in 1.5. Serves first time, no build** — `vllm-glm53flash-1-sandbox` (vLLM
v0.32.1, transformers 5.19) carries `qwen3_5_moe` natively. Load **51 s**, 30.01 GiB/GPU,
PIECEWISE + FULL CUDA graphs captured clean. **KV pool 52.95 GiB/GPU = 4,503,736 tokens,
17.18× concurrency at the full 262,144 window** — only 12 of 48 layers hold KV (3:1
GDN:full) with 2 KV heads, so the pool is ~20× dsv41's. Baseline config: spec decode off,
`max_num_seqs` 128, prefix caching on, Triton GDN prefill, native 256K (no YaRN).

`sweep_context_matrix.sh`, `max_tokens=200`, reasoning on, 0 failures in every cell:

| context (actual tokens) | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1,040 — per-stream tok/s | **119.2** | 104.4 | 95.6 | 89.6 | 70.1 | 56.6 | 44.0 |
| 1,040 — aggregate tok/s | 119.1 | 207.8 | 381.7 | 715.1 | 1,115 | 1,793 | **2,770** |
| 8,036 — per-stream | 116.9 | | 93.5 | | 68.6 | | 45.7 |
| 8,036 — aggregate | 116.0 | | 366.5 | | 1,049 | | 2,643 |
| 32,012 — per-stream | 111.6 | | 87.5 | | 54.2 | | |
| 127,922 — per-stream | 93.4 | | 69.7 | | | | |
| 255,799 — per-stream | 71.8 | 62.6 | | | | | |

- Above dsv41 (2,465 peak), below `lagunas21` on the same 4 GPUs (5,230 @64, 7,751 @128).
- Decode is **flat to 32K** (−6%) and keeps **60% of short-context speed at the full 256K**.
- ⚠️ Read **per-stream** at long context: `agg_tok_s` includes client-side prompt
  construction (a 256K prompt is ~1.4 MB of text), so it understates the server.
- ⚠️ TTFTs (0.18 s @1K → 0.95 s @256K) are **prefix-cache warm** — each cell's discarded
  warm-up pass sends the same prompt. They are not cold-prefill figures.

#### `ornith_122b` + MTP — the shipped head, now the default · 2026-11-08 (job 2754533)

`ENABLE_SPECULATIVE=1 ORNITH_SPEC_METHOD=mtp` (2 draft tokens). The head costs 1.2 GiB/GPU;
KV pool 4.50M → **3.89M tokens (14.85× @256K)**. **Mean acceptance length 2.17.** 0 failures.

| context | 1 stream: base → MTP | 4 streams (per-stream) | 16 streams (per-stream) | 64 streams (aggregate) |
|---|---|---|---|---|
| 1K | 119.2 → **192.1** (+61%) | 95.6 → 131.6 | 70.1 → 101.4 | 2,770 → **3,300** (+19%) |
| 8K | 116.9 → 191.5 (+64%) | 93.5 → 138.1 | 68.6 → 91.4 | 2,643 → 2,578 (−2.5%) |
| 32K | 111.6 → 183.2 (+64%) | 87.5 → 132.7 | 54.2 → 82.6 | |
| 128K | 93.4 → 142.5 (+53%) | 69.7 → 88.2 | | |
| 256K | 71.8 → 100.8 (+40%) | (2 streams) 62.6 → 84.5 | | |

**192 tok/s ties `lagunas21` (192.8) for the fastest single stream measured here**, ahead of ornith_gh200's 174. `lagunas21` remains far ahead on aggregate, but falls off faster with context (47 tok/s at 200K vs 101 here at 256K — ⚠️ not like-for-like: this sweep's prompts are prefix-cache warm, and the Laguna TTFTs were cold). And unlike dsv41's DSpark (−47% aggregate @16), MTP here is **neutral-to-positive at
batch** — 8K @64 is the only cell that regressed. So **MTP is now ON by default for the
122B** (`ENABLE_SPECULATIVE=0` to disable); other Ornith sizes stay opt-in until measured.

#### `ornith_122b` @ 1M via the model card's YaRN recipe — ❌ RUNS, BUT IS WRONG · 2026-11-08 (job 2754537)

`MAX_MODEL_LEN=1048576` + the card's `--hf-overrides '{"rope_scaling": {yarn, factor 4}}'`.
Pool 4,598,409 tokens (4.39× @1M), and the throughput matrix looks healthy all the way out —
**an 878,117-token request decoded at 36.2 tok/s with 0 failures** (511K: 41.5). It means
nothing. `needle_probe.py` (passcode planted at 2% depth, asked for at the end):

| prompt tokens | passcode recovered | answer |
|---|---|---|
| 33,975 | ✅ | correct |
| 207,148 | ✅ | correct (inside the native 262,144) |
| **621,676** | ❌ | `ductductductductduct…` |

**The card's recipe is a silent no-op on vLLM.** Ornith is `…ForConditionalGeneration`; the
rotary embedding is built from `text_config.rope_parameters`, and vLLM applies a top-level
`rope_scaling` override to the *outer* config, where nothing reads it. Only
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` lets the window grow, so every position past 262,144 runs on
untrained plain M-RoPE. `ORNITH_YARN_FACTOR` now overrides `text_config` instead (commit
`c7778ab`); its verification run is below. ⚠️ **Lesson: a throughput sweep cannot detect a
broken position encoding — verify long context with a needle, not with tok/s.**

#### `ornith_122b` @ 1M via the CORRECTED YaRN override — ✅ WORKS · 2026-11-08 (job 2759667)

`MAX_MODEL_LEN=1048576 ORNITH_YARN_FACTOR=4` (MTP off, so only YaRN differs from job
2754537) — now packaged as the preset **`ornith_122b_1m`**
(`./olivia.sh server start ornith_122b_1m`). The override now targets `text_config.rope_parameters`. Three independent signs it
took: the "greater than the derived max_model_len" warning is **gone** (derived =
`max_seq_len=1048576`), weights load **0.1 GiB larger** (30.11 vs 30.01 GiB — a 4× rotary
cache), and the needle:

| prompt tokens | card recipe (2754537) | corrected (2759667) |
|---|---|---|
| 33,975 | ✅ | ✅ |
| 207,148 | ✅ | ✅ |
| 621,675 | ❌ `ductduct…` | ✅ |
| **1,036,709** | — | ✅ (122.5 s, ~8.5K tok/s prefill) |

**A full 1M-token context, correct, on one node.** Pool 4,591,062 tokens = 4.38× @1M.
Per-stream decode (0 failures): 1K 119.9 · 128K 97.6 · 256K 77.9 · **512K 54.8 · 878K 36.2**;
16 streams @1K 1,134 aggregate. **Static YaRN costs nothing measurable at short context**
(119.9 vs 119.2 baseline) — the card's quality caveat for short inputs is untested here.

### Ornith 1.5 `ornith_1n` — 397B on ONE node, experts in Grace memory · 2026-11-08 (job 2754516)

The flagship's 418 GB (~97.4 GiB/GPU) exceeds a whole 96 GB card, so the dsv41 lesson
applies — but refined: `CPU_OFFLOAD_GB=30` restricted to **`--cpu-offload-params experts`**
(new in vLLM 0.32), so only routed-expert weights live in Grace LPDDR5X and are read by
zero-copy UVA over NVLink-C2C. Logged **30.17 GiB/GPU offloaded**, 64.83 GiB/GPU resident,
load 252 s, CUDA graphs clean, **KV 16.79 GiB/GPU = 1,142,125 tokens (4.36× @256K)**.
0 failures.

| context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1K — per-stream tok/s | **50.3** | 33.1 | 21.8 | 13.7 | 8.2 | 5.1 | 3.4 |
| 1K — aggregate | 50.3 | 66.0 | 87.2 | 109.3 | 130.7 | 164.0 | **218.8** |
| 8K — per-stream | 50.7 | | 22.3 | | 8.6 | | |
| 32K — per-stream | 49.7 | | 22.4 | 13.9 | | | |
| 128K — per-stream | 45.4 | 31.9 | | | | | |
| 256K — per-stream | 40.5 | | | | | | |

**What it shows — the expert-offload hypothesis was half right.**
- ✅ Single stream works: **50 tok/s**, and flat with context (−20% at 256K). At batch 1 each
  token touches 10 of 512 experts, so only their rows cross C2C. A layer-order offload of
  the same 30 GiB would stream *all* of it every step (~70 ms/step at ~450 GB/s → ≤14 tok/s).
- ❌ Batch does not scale: aggregate **131 @16, 219 @64**. With more sequences per step,
  top-10-of-512 routing touches nearly every expert, so each step reads most of the 30 GiB
  over C2C — the ~70 ms/step floor above, ≈8 tok/s/stream at 16. **Offloaded bytes per step
  are the ceiling, so the lever is the dsv41 rule: offload the minimum.** 20 GiB and
  20 GiB + MTP are queued next.
#### The layer-order control — 20 GiB, non-selective · 2026-11-08 (job 2759654)

An accident made useful: `CPU_OFFLOAD_PARAMS` was not forwarded by `olivia.sh` (fixed in
`a87141e`), so this run offloaded **20.04 GiB/GPU in layer order** — dsv41's mode — instead
of experts only. Resident 74.95 GiB/GPU, load 337 s, **KV 7.17 GiB = 453,354 tokens (1.73×
@256K)**: 10 GiB less offload bought back 9.6 GiB of pool, the same ~1:1 rate dsv41 measured.

| | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| layer-order 20 GiB — 1K aggregate | 38.7 | 60.5 | 89.8 | 124.5 | 158.7 | 208.3 | **273.6** |
| experts-only 30 GiB — 1K aggregate | **50.3** | 66.0 | 87.2 | 109.3 | 130.7 | 164.0 | 218.8 |

Single-stream by context, layer-order 20 GiB: 38.7 (1K) · 39.1 (8K) · 38.6 (32K) · 36.2
(128K) · 32.9 (256K). 0 failures.

- ❌ **My prediction was wrong.** I expected layer-order to cap at ~20 tok/s (all 20 GiB
  read every step at an assumed ~450 GB/s). It reached **38.7**, which implies **~800 GB/s
  effective over C2C** — about twice my assumption.
- The two runs differ in **two** variables (mode and amount), so this does not yet say which
  mode is better. Read only what is safe: experts-only wins **batch-1 latency** even while
  offloading 50% more; layer-order at 20 GiB wins **every batch ≥4** and gives up 60% of the
  pool. Experts-only at 20 GiB (queued) is the run that separates mode from amount.

#### Experts-only at 20 GiB — the separating run, and the new default · 2026-11-08 (job 2760309)

Same amount as the control (20.61 vs 20.04 GiB/GPU offloaded), so **only the mode differs**.
Resident 74.38 GiB/GPU, **pool 458,494 tokens (1.75× @256K)** — matches the control's
453,354. 0 failures.

| 1K aggregate | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| **experts-only 20 GiB** | **59.3** | **82.9** | **113.0** | **149.3** | **174.4** | **222.9** | **298.7** |
| layer-order 20 GiB | 38.7 | 60.5 | 89.8 | 124.5 | 158.7 | 208.3 | 273.6 |
| experts-only 30 GiB | 50.3 | 66.0 | 87.2 | 109.3 | 130.7 | 164.0 | 218.8 |

Single-stream by context: **59.3 (1K) · 59.2 (8K) · 58.0 (32K) · 52.0 (128K) · 45.3 (256K)**
vs layer-order's 38.7 → 32.9. 16 streams @8K: 186.8 vs 169.5.

- ✅ **Expert-targeted offload wins every cell** at equal bytes: **+53% single-stream**,
  +9% @64. The margin narrows with batch exactly as the mechanism predicts — a bigger batch
  touches more experts, so expert-only reads converge toward reading everything.
- ✅ **Amount matters more than I thought:** 30 → 20 GiB (both experts-only) is +18% single,
  +36% @64. The dsv41 rule holds: **offload the minimum**. `CPU_OFFLOAD_GB` default is now
  **20** (was 30); raise it only to buy pool (~1 GiB per GiB) for concurrent long contexts.
- The 397B on one node is now **59 tok/s single-stream** — 73% of the 1.0 397B's 81.5 on
  **twice the GPUs** — and holds 45 tok/s at 256K.

#### Experts-only 20 GiB + MTP — ❌ does not fit a 256K window · 2026-11-08 (job 2760314)

The MTP head and its speculator cost **+3.14 GiB/GPU** (77.52 vs 74.38 GiB resident), leaving
3.52 GiB of KV against the **4.21 GiB one 262,144-token request needs**:
`ValueError: To serve at least one request with the model's max seq len (262144), (4.21 GiB
KV cache is needed, ...)`. Died at startup after a full load + capture (~7 min). Retrying at
`CPU_OFFLOAD_GB=22`.

#### Experts-only 22 GiB + MTP — a latency option, not a default · 2026-11-08 (job 2766587)

22.12 GiB/GPU offloaded, resident 76.01, **pool 315,376 tokens (1.20× @256K)**, mean
acceptance length **2.33**. 0 failures. Compared with experts-only 20 GiB, no MTP (2760309):

| 1 stream | 1K | 8K | 32K | 128K | 256K |
|---|---|---|---|---|---|
| no MTP (20 GiB) | 59.3 | 59.2 | 58.0 | 52.0 | 45.3 |
| **MTP (22 GiB)** | **75.6** | 54.3 | 26.7 | **63.2** | **51.8** |
| MTP TTFT (s) | 0.40 | 1.36 | 5.09 | 0.71 | 1.28 |

Aggregate @1K: 113 @4 (= no-MTP), 149 @8 (=), 187 vs 174 @16, 217 vs 223 @32, **229 vs 299
@64 with p95 TTFT 45 s**.

- When the prompt is prefix-cached, MTP is worth **+14 to +27%** single-stream.
- ⚠️ The per-stream column includes TTFT, and the 8K/32K cells show TTFTs of 1.4–5.1 s where
  the no-MTP run had ~0.4 s — i.e. **prefix-cache misses, not slower decode**. The 3.1 GiB
  the head costs shrinks a pool that was already small; 8 streams × 32K is 256K of a 315K
  pool, so the warm-up's prompts do not survive. The cause is inferred from TTFT, not
  separately measured.
- **So: off by default on `ornith_1n`** (unlike the 122B, whose 3.9M-token pool has room).
  For single-user latency work use the preset **`ornith_1n_lowlat`** (= `ENABLE_SPECULATIVE=1
  CPU_OFFLOAD_GB=22`). At
  `CPU_OFFLOAD_GB=20` it does not start with a 256K window (job 2760314, above).

- For reference, the 1.0 397B on **2 nodes** (no offload) reached 81.5 single / 845 @16. One
  node halves the GPUs and avoids cross-node PP, but at batch the 2-node shape is far better
  per GPU (106 vs 33 tok/s/GPU @16). Use `ornith_1n` for **single-user / scarce-node** work.

### Ornith 1.5 `ornith` — 397B, 2 nodes × 4 GH200, TP=4 + PP=2, PIECEWISE · 2026-11-09 (job 2777625)

**On the `vllm-glm53flash-1` container (v0.32.1), no build.** 47.6–48.1 GiB/GPU, weights in
~127 s, **KV 4,418,411 tokens (16.9× @256K)**, PIECEWISE capture, engine-as-actor Ray. 0 failures,
**no PP decode wedge**. (First attempt, job 2762334, served too but got no requests — the sweep's
multi-node head-node bug, fixed in `5202f01`.)

| context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 1K — per-stream | **79.6** | 71.6 | 41.4 | 38.2 | 32.1 | 26.9 | 20.0 |
| 1K — aggregate | 79.6 | 142.9 | 165.4 | 305.0 | 512.6 | 857.4 | **1,269.7** |
| 8K — per-stream / agg | 73.4 | | 42.3 | | 31.1 / 485 | | 20.9 / **1,283** |
| 32K — per-stream | 76.0 | | 40.4 | | 29.2 | | |
| 128K — per-stream | 71.8 | | 36.2 | | | | |
| 256K — per-stream | 63.1 | 53.2 | | | | | |

- vs the 1.0 397B on the same shape (pinned main `251f7e4`): single **79.6 vs 81.5** (same), and
  it now scales to **64 streams** (1.0 was measured to 16: 845; 1.5 @16: 513 — the 1.0 sweep used
  `max_tokens=256`, this one 200, so the @16 gap is not like-for-like).
- vs `ornith_1n` (same weights, **half the GPUs**): +34% single (79.6 vs 59.3), **4.2× @64**
  (1,270 vs 299), 9.6× the pool. Per GPU @64: **159** vs 75. **Use 2 nodes for batch/serving;
  `ornith_1n` only when 2 nodes will not schedule.**

### Ornith 1.0 `ornith_gh200` — 35B MoE FP8 on 1× GH200, CUDAGraph · 2026-07-16/17
Concurrency sweep (`bench_sweep.py`, `max_tokens=512`, warm/JIT-cached pass on the pinned-commit rebuild):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Agg tok/s — reasoning **on**  | 173.7 | 309.8 | 569.5 | 1044.3 | 1697.5 | 2836.2 | 4620.6 |
| Per-stream tok/s — **on**     | 173.8 | 155.1 | 142.5 | 130.7 | 106.2 | 88.8 | 72.4 |
| Agg tok/s — reasoning **off** | 173.8 | 258.5 | 567.2 | 979.8 | 1645.2 | 2820.6 | 4210.4 |
| Per-stream tok/s — **off**    | 173.9 | 157.8 | 141.9 | 127.0 | 105.9 | 88.3 | 70.2 |
| p95 TTFT (s)                  | 0.04 | 0.07 | 0.07 | 0.08 | 0.16 | 0.19 | 0.28 |

**Single GH200 card, TP=1, 0 failures 1→64.** Single-stream **~174 tok/s** — the fastest single-stream of any preset here (vs Laguna ~63, GLM-5.2 ~5.6, Kimi ~17), exactly the "max single-user throughput" this preset targets; MoE (~3B active) + CUDAGraph FULL capture (captures cleanly on the hybrid, unlike eager Kimi/glm52). Per-stream degrades gracefully to ~72 tok/s @64; aggregate near-linear to ~4620 tok/s @64. TTFT sub-100 ms through 8-way. **Reasoning on vs off is the same decode rate** (per-stream tok/s within noise) — thinking (`chat_template_kwargs={"enable_thinking": false}` to disable) just emits more tokens per request, so a request is longer, not slower per token; the slightly lower *aggregate* off is only shorter answers finishing early (lower steady-state batch occupancy), same as Laguna. (A cold c=8 outlier — a 5.2 s triton JIT stall on a fresh shape — vanishes once kernels are cached.) vLLM main pinned `251f7e4`, transformers 5.8.1, NGC 26.05, 256K context; identical numbers on the hand-patched container and the from-scratch pinned rebuild.

**On-cluster reality (Qwen3-Next hybrid on the NGC stack — the model card is misleading):** the 35B is `Qwen3_5MoeForConditionalGeneration`, a **hybrid** model (Gated-DeltaNet `linear_attn` + full `self_attn`), multimodal (vision tower, served for text), **channel/token W8A8 FP8** (not block-FP8 → DeepGEMM unused), and the FP8 export ships **no MTP weights** (config declares `mtp_num_hidden_layers=1` but the head is absent → MTP off). vLLM main pulls **flashinfer 0.6.14**, version-skewed against the container's cute-dsl (its Blackwell kernel imports `cutlass.cute.nvgpu.OperandMajorMode`, absent here) → importing it crashes engine init. Serving it needed: **flashinfer removed** (Hopper doesn't need its Blackwell kernels), the `ll_bf16` cute-dsl router-GEMM warmup **skipped** (needs the absent `quack`), and GDN prefill forced to the **in-tree Triton/FLA** kernel (`--additional-config '{"gdn_prefill_backend":"triton"}'`) — an all-Triton/CUTLASS path, zero flashinfer. Two shared build-script bugs were also fixed en route (`NGC_PYTORCH_TAG` forwarding, verify-from-source-tree). See CLAUDE.md.

### Ornith 1.0 `ornith` — 397B flagship, 2 nodes × 4 GH200, TP=4 + PP=2, PIECEWISE capture · 2026-07-17
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, reasoning on, warm pass):

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Aggregate tok/s | 81.5 | 148.4 | 276.0 | 499.8 | 845.4 |
| Per-stream tok/s | 81.5 | 74.5 | 69.1 | 62.5 | 52.9 |
| p95 TTFT (s) | 0.07 | 0.12 | 0.12 | 0.13 | 0.29 |

**397B on 2 nodes (8×GH200), 0 failures 1→16 — and NO multi-node PP decode wedge** (the engine-as-actor RayExecutorV2 + PIECEWISE capture avoids glm51's wedge, the glm52 lesson paying off). Single-stream **~81 tok/s** is remarkable for a 397B over Slingshot PP (vs glm52's ~5.6 tok/s eager on 3 nodes) — PIECEWISE capture + only 2 PP stages + MoE. ~400 GB W8A8 loads in ~106 s; KV cache 3.99M tokens (15.2× concurrency @256K). vLLM main pinned `251f7e4`, NGC 26.05.

**Three fixes were needed for the multi-node path** (all now codified): the EngineCoreActor on `251f7e4` computes physical GPU ids for the whole world (8) by indexing `CUDA_VISIBLE_DEVICES`, which is only the node-local 4 GPUs → `IndexError` — fixed by stripping CVD from the container with **`env -u CUDA_VISIBLE_DEVICES`** (+ `RAY_EXPERIMENTAL_NOSET`), so vLLM uses raw ids and Ray places the 8 workers itself (singularity leaks the host CVD, so omitting the `--env` wasn't enough). And the compressed-tensors **W8A8 FP8 cutlass** linear double-sets `weight_loader` when linear dims need 16-alignment padding (the 397B's do, the 35B's don't) → `AssertionError` — patched (redundant re-set dropped). Both the legacy and engine-as-actor Ray executors hit the device-index bug (v1 runs EngineCore as a Ray actor either way), so the fix is executor-independent. See CLAUDE.md.

### Borealis 27B (`borealis`) — Norwegian-centric Gemma-3 on 1× GH200, BF16 · 2026-08-18
`bench_sweep.py`, `max_tokens=512`, **warm pass** (warmup pass discarded).

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 55.8 | 108.9 | 215.2 | 423.9 | 807.7 | 1535.1 | 2665.5 |
| Per-stream tok/s | 55.8 | 54.5 | 53.8 | 53.0 | 51.4 | 49.3 | 42.4 |
| p95 TTFT (s) | 0.03 | 0.04 | 0.05 | 0.04 | 0.05 | 0.07 | 0.12 |
| Failures | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**0 failures 1→64 and the flattest scaling curve of any preset here.** Per-stream barely moves from 1-way to 32-way (55.8 → 49.3, a 12% drop, versus qwen38's 86 → 61), and TTFT stays **under 0.12 s even at 64-way** — the lowest latency in this document. Aggregate scales 48× from 1→64. Single-stream ~55.8 tok/s is slower than qwen38's ~86, which is expected: this is **BF16** (no quantized release exists) against qwen38's block-FP8, and 62 dense layers. The trade is worth naming — Borealis gives up ~35% single-stream but holds concurrency far better, so on a shared card it overtakes on aggregate under load.

Loaded in **45 s to healthy** — the fastest startup here, since BF16 skips dequant setup. Ordinary Gemma-3 attention captures CUDAGraphs cleanly. Runs on the shared `vllm-glm53-1` container (vLLM v0.27.1 + NGC 26.07), the **third** model on that one build.

> ⚠️ **The attention backend must be auto-selected, never forced to `FLASH_ATTN`.** Job 2032118 died at engine init in 110 s with `ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid ... Reason: ['mm_prefix (PrefixLM bidirectional attention) requires FlashAttention v4, which does not resolve for this head_size']`. Gemma 3's **text** path is ordinary sliding-window + full attention, but its **vision tower** makes the config multimodal PrefixLM — the image prefix gets bidirectional attention — and vLLM only serves that via FA4, an SM100/Blackwell path unavailable at this head size on Hopper. **This bites even though we only ever send text.** Fixed by leaving `VLLM_ATTENTION_BACKEND` unset.

### DeepSeek-V4-Flash-0731 (`dsv4flash`) — BLOCKED, does not serve on this stack · 2026-08-19
Six attempts, five distinct blockers, ~1 GPU-h total. **It never served a token.** Recorded here because each blocker is real, four are fixed and committed, and the fifth is structural.

| # | Blocker | Status |
|---|---|---|
| 1 | `AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto` | **Fixed** — `KV_CACHE_DTYPE=fp8` default. Note this is the *exact inverse* of GLM-5.2, which on Hopper cannot use fp8 KV at all |
| 2 | `KeyError: 'model.layers.43.mtp_block.main_norm.weight'` | **Fixed** (MTP → opt-in). The head IS shipped — 4705 `mtp.*` tensors — but as `mtp.0.*` where v0.27.1 expects `model.layers.<N>.mtp_block.*`. **Tensor presence ≠ loadable** |
| 3 | `ImportError: tilelang is required for mhc` | **Fixed** — `tilelang==0.1.12` pinned in the build |
| 4 | `tvm::ffi::Error: TypeAttr __ffi_repr__ already registered` (C++ abort, no Python traceback) | **Fixed** — `apache-tvm-ffi==0.1.11`, the one version vLLM, tilelang and flashinfer all accept |
| 5 | `ModuleNotFoundError: No module named 'quack'` | **UNFIXABLE HERE** |

> 🚧 **Why blocker 5 is structural, not another patch.** There are two quack import paths. The first (`fused_indexer_q.py`) is guarded by `has_cutedsl()`, which under-reports — it checks `cutlass` but every cutedsl path also imports `quack` — so making the check honest routes it to an existing fallback (`PYPATCH_HAS_CUTEDSL_QUACK`, same bug class as Ornith's `ll_bf16` check). The second, `deepseek_v4/compressor.py:423`, has **no capability gate and no fallback**; its own comment states *"head=512 on CUDA always uses cutedsl"*. So quack is mandatory, and it cannot be installed: `quack-kernels` 0.6.4 pins `nvidia-cutlass-dsl==4.6.2`, vLLM v0.27.1 pins **4.6.0**, the container has **4.5.2**. No version satisfies all three, and bumping cutlass-dsl is what broke Ornith. **Retry on a vLLM release whose cutlass-dsl pin matches quack-kernels'** — the preset and all four fixes are committed and ready.

### Laguna S 2.1 (`lagunas21`) — 1 node × 4 GH200, FP8, TP=4 · 2026-08-18
`bench_sweep.py`, `max_tokens=512`, **warm pass**. **KV cache 3,190,414 tokens.**

**Concurrency at short context (~74 prompt tokens):**

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 192.7 | 339.0 | 614.8 | 1105.1 | 1867.9 | 3132.4 | 5229.5 | 6362.3 | **7750.9** |
| Per-stream tok/s | 192.8 | 169.7 | 153.8 | 138.3 | 116.9 | 98.2 | 82.3 | 66.9 | 61.6 |
| p95 TTFT (s) | 0.03 | 0.04 | 0.04 | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.22 |
| Failures | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**The fastest preset here: ~193 tok/s single-stream and 7751 tok/s aggregate at 128-way, 0 failures throughout, p95 TTFT ≤0.22 s.** For scale, qwen38 peaks at 3162 @64 and Borealis at 2666 @64 — though both are 1-GPU presets against this one's 4, so *per GPU* qwen38 is still ahead on aggregate. **128 is not the ceiling**: aggregate was still climbing (6362 → 7751) and per-stream fell only 8% from 96 to 128, so the top end has not saturated.

**Context ladder (the result that matters)**, `--prompt-tokens` calibrated to the real tokenizer, `MAX_MODEL_LEN=524288` for the 200K/500K rungs:

| Context (actual prompt tokens) | c=1 agg | c=1 per-stream | c=1 TTFT | c=16 agg | c=16 per-stream | c=16 p95 TTFT | fails |
|---|---|---|---|---|---|---|---|
| 74 | 192.7 | 192.8 | 0.03 s | 1867.9 | 116.9 | 0.05 s | 0 |
| 15,975 (16K) | 184.6 | 187.8 | 0.08 s | 807.7 | 53.7 | 4.49 s | 0 |
| 99,474 (100K) | 69.8 | 91.8 | 2.73 s | 118.3 | 11.9 | 36.95 s | 0 |
| 198,873 (200K) | 29.9 | 47.0 | 7.86 s | 36.6 | 4.3 | 111.54 s | 0 |
| 497,074 (500K) | 7.3 | 15.3 | 29.88 s | **7.6** | **1.1** | **465.62 s** | 0 |

> 🔥 **Context, not concurrency, is the dominant cost — and it invalidates reading any short-context headline as a general result.** Across the ladder at 16-way, aggregate throughput falls from **1867.9 → 7.6 tok/s** (246×) and p95 TTFT rises from **0.05 s → 465.6 s** (9300×, i.e. 7.8 minutes to first token). Single-stream falls 192.8 → 15.3 tok/s (−92%). The full spread between this preset's best number (7751 tok/s at 128-way, ~74 tokens) and its worst (7.6 tok/s at 16-way, 500K) is over **1000×**. **Zero failures anywhere** — it never breaks, it just degrades until it is a different service.

> ⚠️ **"Maximum viable length" has three distinct answers, and only one is a hard limit.**
> 1. **Window** — `max_model_len`. A request past it is **rejected with HTTP 400**, not truncated. This is the only genuine wall, and it is a *config* choice here: Laguna S is natively 1M, but the `IS_LAGUNA` default caps it at 131072, so a 100K-target request that tokenises to ~178K gets a 400 until you raise it.
> 2. **KV capacity** — 4,024,015 tokens at a 512K window; vLLM reports "Maximum concurrency for 524,288 tokens per request: 7.68x". **This is NOT an admission limit.** I predicted 500K×16 (8M needed) was arithmetically impossible; it ran anyway, with 0 failures, because vLLM schedules the excess in waves rather than refusing it. The cost surfaces as queueing latency, not errors.
> 3. **Latency tolerance** — the real operational limit. Laguna S will happily serve 500K×16 at 1.1 tok/s per stream with ~8 minutes to first token. Nothing fails; it is simply unusable for interactive work. Pick the rung by the latency you can accept, not by what the server will admit.

### Qwen3.8 27B (`qwen38`) — 27B dense on 1× GH200, block-FP8, PIECEWISE capture · 2026-08-18
`bench_sweep.py`, `max_tokens=512`, **warm pass** (a discarded warmup pass runs first — see the JIT note below).

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 85.9 | 162.7 | 318.0 | 613.9 | 1148.2 | 1960.8 | 3162.0 |
| Per-stream tok/s | 86.0 | 81.4 | 79.5 | 76.8 | 71.8 | 61.4 | 49.5 |
| p95 TTFT (s) | 0.07 | 0.14 | 0.11 | 0.16 | 0.17 | 0.28 | 0.48 |
| Failures | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**0 failures 1→64, sub-0.5 s TTFT throughout, aggregate scaling 37× from 1→64.** Single-stream **~86 tok/s** and per-stream degrades gently (86 → 49.5 at 64-way), so this is a genuinely good multi-user model on one card. For a *dense* 27B that compares well with the sparse models here — ornith_gh200's 35B MoE reaches ~174 tok/s but activates only ~3B/token, and Laguna's 225B MoE gets ~63 tok/s on 4× the GPUs. Weights **28.46 GiB in 21 s**; **KV cache 832,557 tokens** = 3.18× concurrency at the full 262,144-token context. **CUDAGraph PIECEWISE capture works** (51 graphs) — the hybrid linear+full attention captures cleanly like Ornith, unlike the GLM-5.x/Kimi MLA models that IMA. First workload ever run on the `vllm-glm53-1` container (vLLM v0.27.1 + NGC 26.07), so it also validates that build.

> ⚠️ **Always discard a cold pass on this model.** The first-touch run reported *137.6* agg / *8.55 s* TTFT at c=4 and *208.7* / *13.10 s* at c=8 — pure Triton JIT stalls on fresh shapes, not throughput. Warm, those are **318.0 / 0.11 s** and **613.9 / 0.16 s**. A cold sweep makes this model look like it collapses at moderate concurrency when it does the opposite. Same effect the Ornith sweep saw; `sweep_when_ready.sh` now runs a warmup pass and discards it.

**Not yet measured: MTP.** The checkpoint really ships the head (22 `mtp.*` tensors) and vLLM registers `Qwen3_5MTP`, so `ENABLE_SPECULATIVE=1` remains the likeliest large single-stream win.

**Reasoning not observed.** `reasoning_content` was empty on every prompt, including with `chat_template_kwargs={"enable_thinking": true}` — the kwarg does reach the template (prompt tokens 87→83) and the template references `enable_thinking`/`<think>`, but the model answered directly with no `<think>` block. Parsers (`qwen3_xml`/`qwen3`) configured but unexercised. Answers correct, including the bat-and-ball trick question.

### Laguna M.1 (`laguna`) — 1 node × 4 GH200, FP8, CUDAGraph · 2026-06-20
Concurrency sweep (`bench_sweep.py`, `max_tokens=512`), reasoning on (`enable_thinking=true`) vs off:

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Agg tok/s — reasoning on  | 62.7 | 118.6 | 229.7 | 346.6 | 616.3 | 1182.3 | 2055.7 |
| Per-stream tok/s — on     | 62.7 | 59.4 | 57.5 | 43.3 | 38.5 | 37.0 | 32.2 |
| Agg tok/s — reasoning off | 63.3 | 118.9 | 229.9 | 411.8 | 717.6 | 1147.6 | 2002.3 |
| Per-stream tok/s — off    | 63.3 | 59.5 | 57.6 | 51.5 | 44.9 | 35.9 | 31.5 |

Single node, TP=4 — no multi-node PP, so none of the glm51/glm52 decode wedge. **First preset to run with CUDAGraph capture** (Laguna's ordinary dense attention captures cleanly, unlike the eager Kimi/glm52): single-stream **~63 tok/s** (~3.7× the eager Kimi's ~17), aggregate near-linear to **~2050 tok/s at 64-way**, 0 failures 1→64, sub-second TTFT (one transient ~2 s blip at 8–16 reasoning-on). Reasoning on vs off is the same decode rate — thinking just emits more tokens per request (~300 reasoning tokens on a 400-token answer), so it's longer per request, not slower per token. vLLM v0.21.0, transformers 5.12, fastapi 0.136.3.

### GLM-5.2 (`glm52`) — 3 nodes × 4 GH200, eager, FP8 · 2026-06-18
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, thinking on):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 5.6 | 11.1 | 22.5 | 43.1 | 81.1 | 130.7 | 224.5 | 419.0 |
| Per-stream tok/s | 5.6 | 5.6 | 5.6 | 5.4 | 5.1 | 4.1 | 4.7 | 6.6 |

Stable 1→64 (0 failures, no decode wedge — RayExecutorV2). Single-stream is slow (~5.6 tok/s, eager) with high TTFT (~14 s, PP=3 prefill); strong batched throughput (~75× from 1→64). CUDAGraph capture IMAs on this NGC stack, so eager only.

> **This table is also the A/B baseline for `glm53_v27`** (vLLM v0.27.1 + NGC 26.07). GLM-5.3 is GLM-5.2's base re-post-trained, so the new container can — and should — be validated on these same `RedHatAI/GLM-5.2-FP8` weights before GLM-5.3's weights are published. Re-run this exact sweep there and record it below; the number to watch is whether a newer torch/inductor lets `CUDAGRAPH_MODE=PIECEWISE` capture (which would move single-stream off ~5.6 tok/s).

### GLM-5.3 (`glm53` / `glm53_v27`) — not yet run
No allocation spent. GLM-5.3's weights were still unpublished as of 2026-08-17 (announced 2026-08-14, open weights promised ~2 weeks out), so neither preset has been prefetched, built, or served. `glm53` is expected to match the GLM-5.2 row above exactly — it is the same base on the same container.
### DeepSeek-V4.1-Flash (`dsv41`) — ✅ **SERVING, BENCHMARKED AND TUNED** · 2026-11-03, tuning 2026-11-06
**748B parameters generating on four GPUs**, job 2697324 on gpu-1-50. Engine init 119.71 s, CUDA graphs captured 100%, zero errors, `OpenAI server is ready to accept requests`. A completion returned `dsv41 online` with reasoning split into its own field, so the `deepseek_v41` tokenizer/tool/reasoning parsers all resolve.

| streams | aggregate | per stream |
|---|---|---|
| 1 | **112.4 tok/s** (2 runs, 0.2% apart) | 112.4 |
| 4 | 357 tok/s | 89.2 |
| 16 | **1121 tok/s** | 70.0 |

**Measurement conditions, so these are not over-read:** short prompts, `max_tokens` 200–300, `temperature 0`, requests issued from the login node to the compute node. Not a long-context test and not a full `bench_sweep.py` run — the KV pool was never stressed, which matters given only 1.75x concurrency is available at the full 131072 window.

**Why this is the best throughput-per-GPU we have measured.** GLM-5.2 is a comparable 744B and needs **12 GPUs across 3 nodes** to reach ~895 tok/s @64, eager-only. This does **1121 tok/s @16 on 4 GPUs, with CUDA graphs captured.** Single-stream it beats Ornith's 397B on 8 GPUs (~81 tok/s) and Laguna M.1 on the same 4-GPU shape (~63 tok/s). The flat scaler — 112 → 70 per stream across a 16x fan-out — is the other notable part.

**And the KV result inverts the GLM-5.3-Flash story.** Measured at TP=4 / util 0.90:

| | dsv41 | glm53_flash |
|---|---|---|
| Available KV cache | **1.08 GiB/rank** | 2.52 GiB/rank |
| GPU KV cache size | **229,815 tokens** | 208,476 tokens |
| Max concurrency @131072 | **1.75x** | 1.59x |

More KV tokens out of **less than half** the memory — about 2.6x the tokens per GiB. That is CSA2 earning its design, and it is why `DSV41_MAX_NUM_SEQS=128` was never at risk: here the cap binds on **weights**, exactly as predicted, not on cache.

> ⚠️ **Unreconciled, and worth chasing:** the preset notes claim CSA2 costs **890 bytes/token**, but 229,815 tokens from 1.08 GiB is ~5 KB/token (~20 KB if that figure is per-rank and the pool is sharded). The likely explanation is that 890 B covers only the **main** KV while this pool also funds the indexer K cache, the sliding-window cache and the compressor — the four backends the log registers (`FLASHMLA_SPARSE_DSV41` / `DEEPSEEK_SPARSE_SWA` / `DEEPSEEK_V41_INDEXER` / `CompressorBackend`). **Unverified.** Do not plan context budgets off the 890 B figure until it is.

**Still on the table:** `DSV41_ENGRAM_USE_THP=true` (confirmed legal at DP=1 — the single most promising next knob, since ~259 GiB of table behind 4 KiB pages is ~68M PTEs under random hash lookup); a real `bench_sweep.py` run including long context; and DSpark speculative decode.

#### Tuning results — what each lever actually costs (2026-11-06)

All at TP=4 on one node. `median_per_stream_tok_s` is the defensible number; see the
sweep driver's header for why `agg_tok_s` is unusable at long context.

| config | KV pool | 1 stream | agg @64 | note |
|---|---|---|---|---|
| baseline (131072) | 1.08 GiB / 229,815 tok | **109.4** | **2454** | 1.75x concurrency at full window |
| `MAX_MODEL_LEN=262144` | 0.59 GiB | — | — | ❌ **refused** — pool *shrank*; see below |
| 262144 + `WORKSPACE_FACTOR=2` | 0.85 GiB / 264,522 tok | 109.3 | 2465 | ✅ **256K serves**; 255,817-tok request at **95.4** |
| `CPU_OFFLOAD_GB=10` | **11.64 GiB** | 64.0 | 831 | capacity ×10.8, throughput −66% |
| **1M**: `1048576` + factor 2 + offload 5 | 3.07 GiB / **1,479,244 tok** | 76.2 | — | ✅ **full native window**; 1,023,096-tok request at **49.7**; 11.3× at 128K |
| **DSpark** + offload 4 | — | **120.2** | 574 @16 | ✅ **132.0 at 131K** — best single-stream; aggregate @16 −47% |

### Three operating points, because the axes are genuinely opposed

No single config maximises both aggregate and single-stream throughput. Pick by workload:

| want | config | gets | costs |
|---|---|---|---|
| **batch throughput** | baseline (no offload, no DSpark) | **2454 tok/s @64**, 128 streams OK | 131K context, 1.75× at full window |
| **long context / concurrency** | `MAX_MODEL_LEN=1048576` `DSV41_SPARSE_MLA_WORKSPACE_FACTOR=2` `CPU_OFFLOAD_GB=5` | **1M window**, 1.41× at 1M, **11.3× at 128K** | −30% throughput |
| **low latency** | `ENABLE_SPECULATIVE=1` `CPU_OFFLOAD_GB=4` | **132.0 tok/s single-stream** at 131K | aggregate @16 −47% |

**DSpark pays for its own offload.** Acceptance length **2.27 → 3.00** of 5 proposed. Against
the *no-offload* baseline it is +27% at 131K and +10% at 1K, despite the 4 GB offload it needs
to fit its draft stages (layers 37–39, each a 128-expert MoE) — those alone would cost ~25%.
But it is a latency setting, not a default: at 16 streams aggregate falls 1081 → 574, the usual
speculative-decode shape where drafts become wasted work once the batch saturates the GPU.
One point does not fit — 32K measured −9% — and is recorded unexplained rather than rationalised.

**Context ladder, each step needing the one before:** 131,072 → 262,144 (needs factor 2) →
1,048,576 (needs factor 2 **and** offload 5). Raising `MAX_MODEL_LEN` alone fails at every step.


**Raising `MAX_MODEL_LEN` alone fails, and the reason is the prefill workspace.**
FlashMLA-sparse reserves `factor × max_model_len × 1152` bytes, default factor 5. Doubling
the window doubled the workspace (0.70 → 1.41 GiB) and the pool *fell* 1.08 → 0.59 GiB, so
vLLM refused: *"0.84 GiB KV cache is needed … available 0.59 GiB"*. It does **not** auto-fit
downward. Factor 2 returns enough to make 262144 fit.

> ⚠️ **The workspace factor is regime-dependent, and I got this wrong once.** Measured at
> `max_model_len=131072` it has **zero** effect, and on that basis it was recorded as "not a
> lever". It is decisive at 262144. vLLM's docstring — "lower values return memory to the KV
> cache **at long max_model_len**" — meant that literally. Testing one operating point and
> generalising was the error.

**`CPU_OFFLOAD_GB` converts weights→KV at ~1:1 but spends throughput at ~1:3.** 10 GB
offloaded returned 10.56 GiB of pool (~106%), enough for ~19 concurrent streams at 128K or
1M context — while costing 41% of single-stream and 66% of aggregate throughput. The cause
is documented behaviour: `--cpu-offload-gb` evicts **by layer order, blind to access
frequency**, so it moves hot layers and pays NVLink-C2C traffic every token. Contrast the
Engram offload, which is surgical and free. **Use the minimum offload that reaches the
target** — 10 GB was ~10× more than needed for most goals.

**128 concurrent streams work** (first tested here): 1089.6 tok/s aggregate, 0 failures.

**Conversion rates do not generalise between mechanisms.** Workspace factor → pool ran ~31%;
weight offload → pool ran ~106%. And tokens-per-GiB is not constant across windows: 212,792
at `max_model_len` 131072 vs 311,202 at 262144. **Linear extrapolation across regimes is
invalid here** — measure each operating point.

#### How it got here — three runs, three one-line faults, all ours
**Allocation spent: ~3.9 GPU-h for the whole campaign** (measured: `cost` reported `Used 9.10` GPU-h for the 2026.2 period afterwards, up from 4.95 before the first dsv41 serve). Breakdown worth keeping, because most of it was not the work:

| job | outcome | GPU-h |
|---|---|---|
| 2694557 | unbound variable, 1 s | 0.0 |
| 2694626 | Blackwell mxfp4, 25 s | ~0.03 |
| 2694691 | missing quack, ~7 min | 0.5 |
| 2697324 | ✅ **served**, 51:22 elapsed | **3.4** |

**Reaching a working server cost only ~0.6 GPU-h** (≈9 min to `ready` × 4 GPUs). The other ~2.8 GPU-h of job 2697324 was the server sitting **idle** until the 1200 s watchdog cancelled it. So an unattended successful serve costs ~1.3 GPU-h per idle window on this shape — cheap insurance against a forgotten job, but not free, and worth a shorter `IDLE_TIMEOUT` for a bench-and-exit run.

Weights prefetched and verified; **three failed serve attempts before the good one, each killed by a different one-line fault in our own code, each caught further along than the last.** The model itself never objected to anything.

| step | result |
|---|---|
| prefetch | ✅ **510 GB (475.3 GiB), 48/48 shards**, 0 incomplete, 0 dangling symlinks, 6m25s (~1.3 GB/s) to the work tier. Verified with `stat -L`, not `du` — content lives in a shared `hub/blobs` store behind two symlink hops, where `du` under-reports ~30,000x |
| submit | ❌ not a job — `sbatch: Unable to open file .../deploys/add-dsv41/run_vllm_server.sh`. `server start` does not upload the script without **`--deploy`**, and this branch had never deployed |
| serve (job 2694557) | ❌ **1 second** — `line 601: DSV41_CUDAGRAPH_MODE: unbound variable`. An informational echo referenced the knob 392 lines before its assignment; fatal under `set -euo pipefail` whenever the operator has not exported it. 0.0 GPU-h |
| serve (job 2694626) | ❌ **25 s**, *after* all four Engram tables had offloaded — `indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs (sm_10x)`. Copied from a recipe written for B200/GB200 |
| serve (job 2694691) | ❌ **~7 min**, in the attention *forward* — `ModuleNotFoundError: No module named 'quack'` from `fused_indexer_q_cutedsl.py:13`. **A build patch written in August to prevent exactly this had never executed** (it globbed `dist-packages` ~770 lines before vLLM is installed; empty glob, empty loop, exit 0, no output). 0.5 GPU-h |

**✅ What is now proven on-cluster — the thing the preset exists for.** All four ranks, twice over:

    Engram table offloaded to pinned host memory: 96,002,514 rows x 256, 23.60 GiB per rank

That is **per table**, across two Engram layers (1 and 14) → **~47.2 GiB per rank, ~188.8 GiB of Grace LPDDR5X node-wide**, matching the figure derived from the config beforehand. ~1.03 bytes/element, consistent with fp8 hash tables. **Half the node's entire HBM worth of parameters, resident in host memory** — read by **UVA lookup**, so the GPU addresses Grace directly over NVLink-C2C rather than staging copies. Without it there is no single-node configuration of this model. vLLM also confirmed `dp_shared_memory=False` resolves automatically at DP=1, which is the precondition `use_thp` needs — so transparent huge pages are a legal, de-risked next step.

**Also settled:** the backend is **`FLASHMLA_SPARSE_DSV41`**, compiled into vLLM (`_flashmla_C.abi3.so`), *not* the flashinfer variant the earlier note below assumed. v0.32.1 registers three V4.1 sparse-MLA backends and only one needs flashinfer. `util 0.90` is at its ceiling here too — the allocator logged the same `failed to map 20 MB with ~20 MB free` signature as GLM-5.3-Flash on all four ranks.

**❌ Known unavailable on Hopper:** `indexer_sparse_logits`, which scores only candidate blocks instead of walking the whole context — O(candidate blocks) per query, and vLLM notes it "pays off for long contexts (roughly 32K and beyond)", i.e. exactly our 131072 window. It needs MXFP4 **tensor cores** (SM100 / CDNA4); Hopper bottoms out at fp8. This is silicon, not a version gate, so **expect throughput below vLLM's published figures by design, with the gap widening as context grows.** Do not read that as underperformance.

> **Lesson, and it is the same one three times.** Every failure was ours, none was the model's, and each was a single line. The common thread is not carelessness about the model but **unverified assumptions about our own toolchain**: a knob assumed assigned, a dtype assumed portable, a patch assumed applied. Two now have permanent checks (`check_scripts.sh`, and a Phase 4 assertion on the installed copy); the dtype one has only a written rule — *a recipe's dtypes are valid only for the hardware the recipe was written on.*

Why it is worth the allocation once unblocked: **748B parameters on a single node.** vLLM sizes V4.1-Flash at ~614 GB VRAM (a GB200 tray, or an 8-GPU H200 node); our node has 384 GB. The gap closes because 196.6B of those parameters are Engram n-gram tables, held by only two layers and read sparsely by token lookup — `--engram-config {"cpu_offload":true}` parks them in pinned Grace memory over NVLink-C2C (~900 GB/s, ~800 GB available) and the cards hold only the backbone. KV is 890 B/token, so a 1M-token prompt costs under 1 GB of global KV. If it serves, it is the largest model we have run on four GPUs, and the first where cold-parameter offload is what makes it possible at all.

### GLM-5.3-Flash (`glm53_flash`) — ✅ **SERVES** · validated 2026-11-06 (job 2732602)

The `GLM53_FLASH_MAX_NUM_SEQS=128` cap holds, on the fifth serve attempt. Engine up,
`Application startup complete`, 0 failures across the matrix.

| streams | aggregate | per stream | p95 TTFT |
|---|---|---|---|
| 1 | 95.2 | 95.3 | 0.19s |
| 4 | 302.8 | 75.8 | 0.32s |
| 16 | **694.8** | 43.6 | 0.54s |
| 64 | 464.1 ⚠️ | 8.4 | **23.90s** ⚠️ |

Decode is flat across context like dsv41 — 95.3 / 94.4 / 93.5 / **90.3** tok/s per stream at
1K / 8K / 32K / 131K.

**⚠️ Run this preset at ≤16 concurrent.** Aggregate *falls* from 16 → 64 streams and p95 TTFT
blows out ~44× (0.54s → 23.90s). It degrades rather than saturates past ~16.

**The cap also grew the pool, which was not the goal.** KDA state blocks are reserved *per
sequence*, so requesting 128 slots instead of vLLM's default 1024 freed the difference:
**208,476 → 367,414 KV tokens (+76%)** and concurrency at 131072 **1.59× → 2.80×**. The default
was costing capacity as well as crashing the engine.

Backend is **`FLASH_ATTN_MLA_SPARSE`**, unchanged by the in-place `has_cutedsl` container patch
— a side effect that was flagged as a risk and is now ruled out.

> 📊 **Compared with `dsv41` on the same node shape, this preset loses on both axes:**
> glm53_flash (320B / 18B active) 95.3 single-stream, peak ~695 @16; dsv41 (748B) 109.4
> single-stream, peak **2454 @64**. dsv41 is the better engine here despite 2.3× the
> parameters — so prefer it unless you specifically need Flash's multimodality or MIT licence.

### How it got here — five attempts · updated 2026-11-06

**Allocation spent so far: ~5 GPU-h of 6000** (measured: `cost` reported `Used 4.95` GPU-h for the whole 2026.2 period on 2026-11-02, superseding the ~3 GPU-h estimate previously recorded here). Weights prefetched and verified; container built and verified; **three serve attempts**. The third got all the way through weight load **and full CUDA-graph capture** before dying on a vLLM *default* — so the model, the arch support and our container are all proven; what remained was one missing argument. No throughput numbers yet.

| step | result |
|---|---|
| prefetch | ✅ 306 G on the **persistent** tier, 74 files in **2m23s** (~2.1 GB/s), 62/62 shards matching the index, 0 incomplete. MTP head confirmed at language layer 45 (1760 tensors) |
| build (job 2663868) | ❌ failed 29 min in — `<stdin> line 16: SyntaxError`. Our own nested-heredoc bug, latent on `main` since ~2026-08-19 |
| build (job 2665731) | ❌ failed 1 h 37 m in — `fatal error: elfutils/libdwfl.h`. DeepGEMM's vendored `deep_jit` needs `libdw-dev`, which no NGC image ships |
| build (job 2666829) | ✅ **41 GB sandbox, vLLM 0.32.1.dev0 on NGC 26.07.** NGC torch preserved, serving stack verified, and all four `Glm5Next` entries present in the *installed* registry. 7.5 h queued, then 2 h 05 m running. SIF step OOM-killed (`mksquashfs` MaxRSS 131478 M vs 128 G) — harmless, the atomic swap precedes it |
| serve (job 2673535) | ❌ engine init, 2.5 min — `OSError: libz3.so.4.15`. tilelang's `z3-solver` dep was skipped by our `--no-deps` install |
| serve (job 2673772) | ❌ engine init, **8.6 min** — but it got *far*: weights loaded, backends selected, **CUDA graphs captured 51/51 PIECEWISE + 2/2 FULL**, then `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (202)`. A vLLM default, not a toolchain fault. 0.6 GPU-h. (Log read 2026-11-02 — the SSH master had dropped mid-run, so this sat unobserved for three days and the preset was wrongly recorded as FlashInfer-blocked) |

**What the attempts did confirm**, from job 2673535's log before it died: `Attention Backend: <auto-select>` (correctly not forced), `glm47`/`glm45` parsers, `max_model_len 131072`, `tensor_parallel_size 4`, and `kv_cache_dtype=auto` — i.e. the Hopper BF16-KV rule honoured. The preset's runtime block behaves as designed.

> ✅ **The FlashInfer question is settled: it is NOT required on Hopper.** Job 2673772 ran with flashinfer **absent** and the NoPE sparse-MLA layers initialised anyway — the selector logged `Using FLASH_ATTN_MLA_SPARSE attention backend out of potential backends: ['FLASH_ATTN_MLA_SPARSE', 'FLASHMLA_SPARSE']` (neither is a flashinfer backend), with all-reduce falling back to PYNCCL and the FP8 MoE on **DeepGEMM**. vLLM's recipe calls FlashInfer ≥0.6.18 *required* for `qk_rope_head_dim=0`; on this stack that is not so, exactly as GLM-5.2's DSA suggested. A fix for the underlying `--no-deps` dependency drop is committed anyway (`36c7858`, unvalidated), but **it blocks nothing** and does not justify a rebuild on its own.

> 🚨 **What actually blocked it, and why `GPU_MEM_UTIL` is the wrong knob.** `glm5_next` is a hybrid, and each of its **34 KDA linear-attention layers holds a per-sequence recurrent state** — one "Mamba cache block" per decode sequence, out of the *same* budget as KV. Measured at TP=4 / util 0.9: **2.52 GiB of KV per rank → 208,476 tokens → 202 state blocks**, against vLLM's default `max_num_seqs` of **1024**. The error suggests raising `--gpu-memory-utilization`; that is **not available here** — the same run logged the caching allocator failing to map 20 MB with ~14 MB free on all four ranks, so 0.9 is already the ceiling. Fixed with `GLM53_FLASH_MAX_NUM_SEQS=128` — deliberately well below the measured 202, because a higher cap buys nothing: the **KV pool binds first at every context we run** (208,476 tokens ÷ `max_model_len` → 1.6 concurrent at 131072, 6.4 at 32768, 25 at 8192), so 128 only starts to bind below ~1.6K context. Since the block count drifts with profiling noise, margin is worth more than unreachable headroom — a cap at 192 would sit 5% under the measured count and risk another full load-and-capture to discover it. The lever for *concurrency*, as opposed to startup, is **lowering `MAX_MODEL_LEN`**: at 131072 the pool holds only **1.59x**.

**Lesson worth more than the preset:** three of the four failures were latent faults in our own build path, not properties of this model — and the first two had silently blocked *every* container build since August. `--no-deps` protects NGC PyTorch but quietly drops legitimate dependencies, and nothing verified the result until a GPU was already allocated. Hence the build-time import probes now in `build_vllm_gh200.sh`, and the `--qos=devel` probe pattern (a 12-second job proved the `libdw-dev` fix; a 26-second one proved the `z3-solver` fix) instead of discovering such things 1.5 h into a build.

What the shape should be, and why it is worth the allocation:
- **One node, TP=4.** ~306 GiB of weights against 4×96 GB = 384 GB. That is a tighter fit than `glm47`'s ~358 GB AWQ, which already serves at TP=4, leaving roughly 40 GiB after weights at the default `GPU_MEM_UTIL=0.90`. For comparison, GLM-5.2 needs **12 GPUs across 3 nodes** — so if this serves, it is a 320B-class model at **4 GPU-h/hour instead of 12**.
- ~~**KV should be unusually cheap.**~~ **MEASURED WRONG, and badly — keep it on the record.** The reasoning was sound (only 11 of 45 layers cache, and those are NoPE MLA with `kv_lora_rank=512`) and we expected a large pool from the recipe's 14.92M tokens at TP=4 on a bigger card. Actual: **208,476 tokens, 2.52 GiB per rank** — roughly **70× less**. The per-token cost was never the issue; the *budget* was. 320B of FP8 weights over 4×96 GB leaves ~2.5 GiB per card, and cheap KV cannot help when there is no room to put it. Worse, the **state** cache competes for that same budget, which is what actually killed the run. **Lesson: on a tight single-node fit, size the pool from the weights, not from the per-token cost.**
- **Known risks — with the first two now settled, and the ranking corrected.** (1) The weight fit is genuinely tight: **confirmed**, and it is the dominant constraint. But `GPU_MEM_UTIL` was the wrong first knob — it is already at its ceiling; `MAX_NUM_SEQS` and `MAX_MODEL_LEN` are the real ones. (2) Upstream's recurring-CUDA-IMA report (vLLM #54317, 4×B200): **not hit** — capture completed 51/51 PIECEWISE and 2/2 FULL, so `CUDAGRAPH_MODE=NONE` stays unused and this model captures on first try, unlike Kimi. (3) Upstream support is only weeks old, which is why the preset builds the newest tag rather than the oldest one carrying the arch — still the right call.

> **Preset retargeted 2026-10-28.** It originally pulled the prebuilt `vllm/vllm-openai:glm53-flash` image because `glm5_next` was in no public vLLM release. It now builds **v0.32.0 from source** like every other preset. Note the recipe page's "vLLM 0.29.0+" is **wrong** — v0.29.0 carries zero `Glm5Next` registry entries; **v0.30.0** is the real floor. The prebuilt image stays reachable as `glm53_flash_prebuilt` (container index 2).

### GLM-5.2 AWQ (`glm52_awq_1n`) — 1 node × 4 GH200, AWQ-INT4, offload40 · 2026-07-03
Peak single-node config = **capture + MTP + fp8_ds_mla + offload40 + 131K**: ~22 tok/s single-stream (3.2× eager), ~241 tok/s @64-way, 0 failures. 3× the throughput of the 3-node FP8 path at 1/3 the GPU cost. Full sweep in `plans/proposed/glm52_awq_1n_offload.md`.

> **Cold load ~6× faster with the Run:ai Model Streamer** (validated 2026-07-03): the default loader takes **~56 min** for the ~415 GiB AWQ checkpoint (~84 min with MTP's double-load) — but that is **vLLM-pipeline-bound, not I/O-bound** (raw Lustre read is ~1.3 GB/s; page-cache reuse was a dead end). `LOAD_FORMAT=runai_streamer` parallelizes the safetensors read → **`Model loading took … 569 s` (~9.5 min)**, serves + decodes cleanly. Now the **default for GLM-5.2** when the staged runai pkg is present under `HF_HOME` (`run_vllm_server.sh` auto-sets `LOAD_FORMAT`/`CONTAINER_PYTHONPATH`; override either to opt out). The pkg is staged durably at `$HF_HOME/runai-pkg` (projects tier).

### Kimi K2.7 / K2.6 — 2 nodes × 4 GH200, eager, native int4 · 2026-06-20
Concurrency sweep (256 output tokens, distinct prompts):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Aggregate tok/s | 17.2 | 37.3 | 77.8 | 133.6 | 264.3 | 590.9 |
| Per-stream tok/s | 17.2 | 18.6 | 19.5 | 16.8 | 16.5 | 18.5 |

Per-stream flat ~17–19 tok/s; TTFT ~1.0 s single-stream; 0 failures (re-confirmed 2026-06-20, unchanged vs 2026-06-15 within noise). Production K2.6 sustains ~830 tok/s at 48 concurrent. Eager (CUDAGraph capture unrecoverable on this stack).

> **Cold-start ≈ 40 min** (measured 2026-06-20): the ~640 GB int4 checkpoint loads at ~38 s/shard × 64 shards off Lustre (~270 MB/s), during which the server sits at "weights reserved, 0 % util, `/health` 000" — that is loading, **not** a hang. `./olivia.sh server watch` and any health-wait must allow ~40+ min before the server answers. Cross-node NCCL runs over TCP (`NET/Socket`, no CXI/RDMA plugin), but loading — not NCCL — dominates cold-start.

### GLM-5.1 (`glm51`) — 2 nodes × 4 GH200, AWQ, **PIECEWISE CUDAGraph capture** · NGC 26.03 · 2026-06-20
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, streaming):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Aggregate tok/s  | 22.0 | 44.6 | 87.6 | 170.6 | 246.2 | 626.9 | 894.6 |
| Per-stream tok/s | 22.0 | 22.3 | 21.9 | 21.3 | 15.4 | 19.6 | 14.0 |
| p95 TTFT (s)     | 0.09 | 0.08 | 0.14 | 0.19 | 2.54 | 0.41 | 0.68 |

This config runs **0 failures across 1→64** (previously hung at `Running ≥ 2`, forcing `anthropic_proxy.py` serialization). An isolating experiment separates the two effects — on the freshly rebuilt NGC-**26.03** `vllm-glm51-1-sandbox`:
- **De-wedge = NCCL all-reduce, not capture.** `CUDAGRAPH_MODE=PIECEWISE` auto-disables vLLM's custom all-reduce → graph-safe NCCL. The isolating test — **eager + `DISABLE_CUSTOM_ALL_REDUCE=1`, no capture — also runs 0 failures at concurrency 1–16**, so the *custom all-reduce kernel* was the wedge cause; NCCL fixes it. Capture isn't needed to de-wedge (it just forces the custom kernel off, since it isn't graph-safe).
- **Throughput = capture.** PIECEWISE capture (51/51 graphs, no IMA — 26.03's inductor handles GLM-DSA capture where **26.05 IMAs** on glm52/Kimi) lifts single-stream from **~5 tok/s** (eager+NCCL) to **~22 tok/s** (~4.5×), holding ~22/stream through 8-way, ~895 tok/s @64.

Recommended config: **capture + NCCL all-reduce** (de-wedged *and* fast). The earlier "wedge → serialize" workaround is **superseded** for this container.

### GLM-4.7
- **GLM-4.7** (single node, 4 GPUs, AWQ): fast single-node, no multi-node wedge.

> Sweep tools: `bench_sweep.py` (concurrency, streaming SSE) and `bench_serving.py` (TTFT + decode). Re-run after any serving-config change and refresh the tables above.

## Direct Script Usage

The underlying scripts can be used directly on the cluster without the CLI:

### Build Container

```bash
# List presets
MODEL_ID=help ./build_vllm_gh200.sh

# Build using a preset
MODEL_ID=glm47 ./build_vllm_gh200.sh

# Build second container of same type
MODEL_ID=glm47 BUILD_INDEX=2 ./build_vllm_gh200.sh

# Submit as SLURM job
MODEL_ID=glm47 sbatch build_vllm_gh200.sh

# Override preset defaults
MODEL_ID=glm47 VLLM_VERSION=v0.6.6 ./build_vllm_gh200.sh

# Create SIF image after build
MODEL_ID=glm47 CREATE_SIF=1 ./build_vllm_gh200.sh
```

### Run Server

```bash
# List available containers
./run_vllm_server.sh

# Run specific container
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Submit as SLURM job
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ sbatch run_vllm_server.sh

# Enable batching proxy for faster streaming over SSH
ENABLE_PROXY=1 CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh
```

## GLM-4.7 Guide

GLM-4.7 is a 358B parameter Mixture-of-Experts model requiring careful memory management.

### Quantization Options

| Model | Size | GH200 Compatible | Notes |
|-------|------|------------------|-------|
| `QuantTrio/GLM-4.7-AWQ` | ~181GB | **Yes (Recommended)** | AWQ 4-bit, leaves ~200GB for KV cache |
| `QuantTrio/GLM-4.7-AWQ` | ~358GB | Yes | FP8, tight fit - reduce MAX_MODEL_LEN |
| `Salyut1/GLM-4.7-NVFP4` | ~179GB | **No** | Requires Blackwell GPUs (B100/B200) |

### Running GLM-4.7

```bash
# Build GLM-4.7 container
./olivia.sh build glm47

# Start with AWQ quantization (recommended)
./olivia.sh server start glm47

# Or with direct scripts:
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# FP8 with reduced context (tight memory fit)
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ MAX_MODEL_LEN=8192 ./run_vllm_server.sh

# Enable MTP speculative decoding
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_SPECULATIVE=1 ./run_vllm_server.sh

# Enable tool calling
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_AUTO_TOOL_CHOICE=1 ./run_vllm_server.sh
```

### Memory Requirements

| Quantization | Model Size | 4×GH200 (384GB) | Notes |
|--------------|------------|-----------------|-------|
| AWQ 4-bit | ~181GB | ~200GB free | Recommended |
| FP8 | ~358GB | ~26GB free | Reduce MAX_MODEL_LEN |
| BF16 | ~716GB | Won't fit | Needs 8+ GPUs |

## Batching Proxy

When accessing vLLM over SSH tunnels, streaming responses can be slow due to per-token network overhead. The batching proxy aggregates tokens into single SSE events, improving streaming throughput by ~3x.

```
Client <--[batched SSE]--> Proxy:8001 <--[per-token SSE]--> vLLM:8000
         (SSH tunnel)                    (localhost, fast)
```

### Performance

| Mode | Without Proxy | With Proxy |
|------|---------------|------------|
| Non-streaming | 17 tok/s | 17 tok/s |
| Streaming | ~5 tok/s | ~15 tok/s |

### Usage

```bash
# Enable proxy when starting server
ENABLE_PROXY=1 CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Tunnel to proxy port
ssh -L 8001:localhost:8001 user@<cluster-login-host>...

# Or run proxy standalone
python vllm_proxy.py --vllm-port 8000 --proxy-port 8001 --batch-tokens 15 --batch-delay-ms 150
```

## Environment Variables

### Build Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | *(required)* | Model preset or custom identifier |
| `BUILD_INDEX` | `1` | Build index for multiple containers |
| `VLLM_VERSION` | `main` | vLLM ref to build — branch, tag, **or commit SHA** (presets may pin a SHA for reproducible builds) |
| `VLLM_PATCHES` | *(preset)* | Space-separated vLLM PR numbers to graft at build time (committed `patches/` snapshot preferred, else live GitHub fetch) |
| `NGC_PYTORCH_TAG` | `26.03-py3` | NGC PyTorch base-image tag (a preset may pin, e.g. glm52 → `26.05-py3`) |
| `DEEPGEMM_REF` | `59f2c07` | DeepGEMM commit (a preset may pin, e.g. glm52) |
| `CREATE_SIF` | `0` | Create SIF image after build |
| `OVERWRITE` | `0` | Allow overwriting existing containers |
| `MAX_JOBS` | `8` | Parallel compilation jobs |
| `CONTAINER_DIR` | *(required)* | Output directory |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER` | *(required)* | Container name or path |
| `MODEL` | `mistralai/Devstral-2-123B-Instruct-2512` | HuggingFace model ID |
| `HF_HOME` | *(required)* | Persistent HF weights cache. Must be on project storage (auto-purge-safe), **not** `/cluster/work` — except glm52, whose ~700 GB FP8 exceeds the project quota, so override to the work cache. Normally forwarded from `mise.local.toml` |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models (forwarded over stdin) |
| `TP_SIZE` | `4` | Tensor parallel size (intra-node) |
| `NUM_NODES` | `1` | Nodes for multi-node serving (glm51/kimi = 2, glm52 = 3); auto-bootstraps Ray |
| `PP_SIZE` | `1` | Pipeline-parallel size across nodes (2 for glm51/kimi, 3 for glm52) |
| `CUDAGRAPH_MODE` | *(auto)* | `NONE` = eager. Kimi and glm52 default to eager (CUDAGraph capture IMAs on this NGC stack) |
| `GPU_MEM_UTIL` | `0.90` | GPU memory utilization |
| `MAX_MODEL_LEN` | `32768` | Max context length (131072 for GLM-5.x / Kimi) |
| `VERBOSE` | `0` | Enable detailed logging |

### Speculative Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SPECULATIVE` | `auto` | Enable speculative decoding (`auto`, `0`, `1`) |
| `NUM_SPECULATIVE_TOKENS` | `5` | Tokens to speculate (ngram) |
| `PROMPT_LOOKUP_MAX` | `4` | Max n-gram window size |
| `MTP_SPECULATIVE_TOKENS` | `3` | MTP tokens for GLM-4.7 |

### GLM-4.7 Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_TOOL_PARSER` | `glm47` | Tool call parser |
| `GLM_REASONING_PARSER` | `glm45` | Reasoning parser |
| `ENABLE_AUTO_TOOL_CHOICE` | `0` | Enable automatic tool selection |
| `SERVED_MODEL_NAME` | *(model ID)* | Custom model name for API |
| `ENABLE_EXPERT_PARALLEL` | `auto` | Expert parallel for AWQ MoE models |

### Batching Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PROXY` | `0` | Enable batching proxy |
| `PROXY_PORT` | `8001` | Proxy server port |
| `PROXY_BATCH_TOKENS` | `15` | Flush after N tokens |
| `PROXY_BATCH_CHARS` | `100` | Flush after N characters |
| `PROXY_BATCH_DELAY_MS` | `150` | Max delay before flush (ms) |

## Architecture

### Build Process (`build_vllm_gh200.sh`)

Five-phase build using Singularity:

1. **Create sandbox** from NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`)
2. **Verify NGC PyTorch** installation is intact
3. **Build vLLM** with pip constraints to preserve NGC PyTorch (`--no-deps` strategy)
4. **Verify final installation** (PyTorch version unchanged, vLLM functional)
5. **Optionally convert** sandbox to SIF image

**Key constraint:** NGC PyTorch must not be replaced by pip. The build uses a constraints file and `--no-deps` installation to prevent this.

### Server Optimizations (`run_vllm_server.sh`)

GH200-specific optimizations:

- **GPU Ordering:** `CUDA_VISIBLE_DEVICES=1,2,3,0` puts slowest GPU last
- **NCCL for NVLink:** `NCCL_P2P_LEVEL=NVL` enables NVLink peer-to-peer
- **GPU Direct RDMA:** `NCCL_NET_GDR_LEVEL=PHB`
- **Flash Attention:** Backend set to `FLASH_ATTN`
- **Memory:** Expandable PyTorch memory segments

### Directory Structure

```
vllm-ngc/
├── olivia.sh              # Unified CLI
├── build_vllm_gh200.sh    # Container build script
├── run_vllm_server.sh     # Server run script
├── chat_devstral.py       # Interactive chat client
├── vllm_proxy.py          # SSE batching proxy
├── patch_glm47_nvfp4.py   # NVFP4 compatibility patch
├── cache/                 # Local cache directories
│   ├── pip/
│   ├── huggingface/
│   └── vllm/
└── logs/                  # Build and server logs
```

**Shared containers on the cluster** (`CONTAINER_DIR`):

```
vllm-glm47-1-sandbox/      # GLM-4.7 build #1
vllm-devstral-1-sandbox/   # Devstral build #1
vllm-generic-1-sandbox/    # Generic build #1
vllm-glm47-1.sif           # Compressed SIF image (optional)
```

## Chat Client

Interactive chat client with rich terminal UI:

```bash
# Basic usage (port matches the tunnel's local port; default LOCAL_PORT=8003)
python chat_devstral.py localhost --port 8003 --stream

# Features:
# - Multi-turn conversation history
# - Token usage and generation speed metrics
# - Markdown rendering
# - Streaming with live display
```

Dependencies: `pip install requests rich`

## Troubleshooting

### Build fails with "container already exists"

Use `--index N` to create a new container or `--force` to overwrite:

```bash
./olivia.sh build glm47 --index 2   # Create vllm-glm47-2-sandbox
./olivia.sh build glm47 --force     # Overwrite vllm-glm47-1-sandbox
```

### Server won't start - out of memory

Reduce context length or use quantized models:

```bash
# Reduce context length
MAX_MODEL_LEN=8192 ./olivia.sh server start glm47

# Use AWQ quantization (recommended for GLM-4.7)
./olivia.sh server start glm47 --model QuantTrio/GLM-4.7-AWQ
```

### Slow streaming over SSH tunnel

Enable the batching proxy:

```bash
ENABLE_PROXY=1 ./olivia.sh server start glm47
# Then tunnel to port 8001 instead of 8000
```

### NVFP4 model fails to load

NVFP4 quantization requires Blackwell GPUs (B100/B200). Use AWQ instead:

```bash
# Don't use NVFP4 on GH200
# MODEL=Salyut1/GLM-4.7-NVFP4  # Won't work!

# Use AWQ instead
MODEL=QuantTrio/GLM-4.7-AWQ ./olivia.sh server start glm47
```

### 2FA prompt on every command

Ensure SSH ControlMaster is working:

```bash
./olivia.sh status  # Check connection status
./olivia.sh --kill-all && ./olivia.sh status  # Reset and reconnect
```

## License

MIT
