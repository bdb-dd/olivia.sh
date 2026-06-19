# vLLM for NVIDIA GH200 (GraceHopper) on HPC Clusters

Build and run [vLLM](https://github.com/vllm-project/vllm) on NVIDIA GH200 ARM64 GPUs, specifically optimized for the NRIS Olivia HPC cluster — from single-node 4-GPU models up to 1T-parameter MoE models sharded across 3 nodes / 12 GPUs. Includes full-featured CLI tooling, a streaming chat client, an Anthropic/Claude-Code bridge, and performance optimizations for high-latency SSH tunnel connections.

## Features

- **Preserves NGC PyTorch** - Builds vLLM without overwriting NVIDIA's custom PyTorch
- **Model Presets** - Build + serve recipes for GLM-4.7, GLM-5.1, GLM-5.2, Kimi K2.6/K2.7, Laguna M.1, Gemma-4, Devstral, Llama, and Qwen
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

## Model Presets

| Preset | Default Model | GPUs | Container | Notes |
|--------|---------------|------|-----------|-------|
| `glm51_v19` (alias `glm51`) | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-1` | TP=4 + PP=2, vLLM v0.19.0. Multi-node PP decode wedge → serve behind `anthropic_proxy.py` serialization |
| `glm51_v20` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-2` | vLLM v0.20.0 + RayExecutorV2. **Quarantined** (same wedge) |
| `glm52` | `RedHatAI/GLM-5.2-FP8` | 12 (3 nodes × 4) | `vllm-glm52-1` | TP=4 + PP=3, block-FP8 (~755 GB). vLLM main pinned `091386a` + PR#45895 snapshot. Eager; fp8 KV + DeepGEMM |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | `vllm-glm47-1` | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | `vllm-kimi-4` | TP=4 + PP=2, native int4, MLA, multimodal, vLLM 0.21. Eager. reasoning_tokens on chat/completions |
| `kimi27` | `moonshotai/Kimi-K2.7-Code` | 8 (2 nodes × 4) | `vllm-kimi-4` (shared) | Same arch + container as K2.6 (no rebuild); thinking-only |
| `laguna` | `poolside/Laguna-M.1-FP8` | 4 | `vllm-laguna-1` | TP=4, single node. FP8 (~225 GB), dense attention (FLASH_ATTN), CUDAGraph on. vLLM v0.21.0, `poolside_v1` parsers |
| `gemma4` | Gemma 4 (31B, multimodal) | 1–2 | `vllm-gemma4-1` | vLLM v0.19.0, AWQ |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | `vllm-devstral-1` | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | — | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | — | TP=4 |
| `generic` | *(user specified)* | 4 | — | generic defaults |

> Build and serve presets are aligned by name (e.g. `./olivia.sh build glm52` then `./olivia.sh server start glm52`). The **Container** column is where each preset's server looks (`vllm-<name>-<index>-sandbox`); `kimi`/`kimi27` share index 4. See **[CLAUDE.md](CLAUDE.md)** for full per-model guides — memory layout, quant options, known issues, and multi-node architecture.

## Performance

Latest measured throughput / latency. **Update this section after every sweep** (with the date + config).

### GLM-5.2 (`glm52`) — 3 nodes × 4 GH200, eager, FP8 · 2026-06-18
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, thinking on):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 5.6 | 11.1 | 22.5 | 43.1 | 81.1 | 130.7 | 224.5 | 419.0 |
| Per-stream tok/s | 5.6 | 5.6 | 5.6 | 5.4 | 5.1 | 4.1 | 4.7 | 6.6 |

Stable 1→64 (0 failures, no decode wedge — RayExecutorV2). Single-stream is slow (~5.6 tok/s, eager) with high TTFT (~14 s, PP=3 prefill); strong batched throughput (~75× from 1→64). CUDAGraph capture IMAs on this NGC stack, so eager only.

### Kimi K2.7 / K2.6 — 2 nodes × 4 GH200, eager, native int4 · 2026-06-20
Concurrency sweep (256 output tokens, distinct prompts):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Aggregate tok/s | 17.2 | 37.3 | 77.8 | 133.6 | 264.3 | 590.9 |
| Per-stream tok/s | 17.2 | 18.6 | 19.5 | 16.8 | 16.5 | 18.5 |

Per-stream flat ~17–19 tok/s; TTFT ~1.0 s single-stream; 0 failures (re-confirmed 2026-06-20, unchanged vs 2026-06-15 within noise). Production K2.6 sustains ~830 tok/s at 48 concurrent. Eager (CUDAGraph capture unrecoverable on this stack).

> **Cold-start ≈ 40 min** (measured 2026-06-20): the ~640 GB int4 checkpoint loads at ~38 s/shard × 64 shards off Lustre (~270 MB/s), during which the server sits at "weights reserved, 0 % util, `/health` 000" — that is loading, **not** a hang. `./olivia.sh server watch` and any health-wait must allow ~40+ min before the server answers. Cross-node NCCL runs over TCP (`NET/Socket`, no CXI/RDMA plugin), but loading — not NCCL — dominates cold-start.

### GLM-5.1 / GLM-4.7
- **GLM-5.1** (2 nodes, AWQ): wedges under concurrent decode (`Running ≥ 2`) — serve behind `anthropic_proxy.py` request serialization; effectively single-stream for CLI use.
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
