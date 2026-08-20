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
| `borealis` | `NbAiLab/borealis-27b` | **1** | `vllm-glm53-1` (shared) | **Borealis 27B — National Library of Norway**, Norwegian-centric instruct. Gemma-3 arch (`Gemma3ForConditionalGeneration`), **BF16 ~54 GB, no quantized release**, SigLIP vision tower (served for text), 128K ctx. Single GH200 TP=1; shares the glm53 container. Backend must be **auto-selected, not FLASH_ATTN** (Gemma 3's vision tower makes it multimodal PrefixLM → `mm_prefix` wants FA4, unavailable on Hopper at this head size; forcing it killed engine init). No reasoning/tool parsers. |
| `qwen38` | `Qwen/Qwen3.8-27B-FP8` | **1** | `vllm-glm53-1` (shared) | **Qwen3.8 27B dense multimodal on a SINGLE GH200** (TP=1). Block-FP8 ~29 GB, hybrid linear+full attn (48+16), 262K ctx, Apache 2.0. Arch `Qwen3_5ForConditionalGeneration` — native in vLLM v0.27.1, so it shares the glm53 container with no rebuild. **MTP head confirmed present** (22 `mtp.*` tensors) → `ENABLE_SPECULATIVE=1` is the obvious win. Weights cached on the persistent tier. **Not yet served** |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | `vllm-glm47-1` | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | `vllm-kimi-4` | TP=4 + PP=2, native int4, MLA, multimodal, vLLM 0.21. Eager. reasoning_tokens on chat/completions |
| `kimi27` | `moonshotai/Kimi-K2.7-Code` | 8 (2 nodes × 4) | `vllm-kimi-4` (shared) | Same arch + container as K2.6 (no rebuild); thinking-only |
| `laguna` | `poolside/Laguna-M.1-FP8` | 4 | `vllm-laguna-1` | TP=4, single node. FP8 (~225 GB), dense attention (FLASH_ATTN), CUDAGraph on. vLLM v0.21.0, `poolside_v1` parsers |
| `ornith` | `deepreinforce-ai/Ornith-1.0-397B-FP8` | 8 (2 nodes × 4) | `vllm-ornith-1` | **397B flagship.** TP=4 + PP=2, FP8 W8A8 (~400 GB). Qwen3.5 hybrid-attn MoE, 256K ctx, PIECEWISE capture + engine-as-actor Ray. MTP off (no head in FP8). Same multi-node PP wedge risk as glm51. vLLM main, transformers ≥5.8.1 |
| `ornith_gh200` | `deepreinforce-ai/Ornith-1.0-35B-FP8` | 1 | `vllm-ornith-1` (shared) | **35B on a single GH200 card** (TP=1), ~3B active, hybrid linear+full attn, multimodal, 256K ctx — max single-user throughput. MTP off (no head in FP8). Shares the `ornith` container, `qwen3_xml`/`qwen3` parsers |
| `gemma4` | Gemma 4 (31B, multimodal) | 1–2 | `vllm-gemma4-1` | vLLM v0.19.0, AWQ |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | `vllm-devstral-1` | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | — | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | — | TP=4 |
| `generic` | *(user specified)* | 4 | — | generic defaults |

> Build and serve presets are aligned by name (e.g. `./olivia.sh build glm52` then `./olivia.sh server start glm52`). The **Container** column is where each preset's server looks (`vllm-<name>-<index>-sandbox`); `kimi`/`kimi27` share index 4. See **[CLAUDE.md](CLAUDE.md)** for full per-model guides — memory layout, quant options, known issues, and multi-node architecture.

## Performance

Latest measured throughput / latency. **Update this section after every sweep** (with the date + config).

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
