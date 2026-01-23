# vLLM for NVIDIA GH200 (GraceHopper) on HPC Clusters

Build and run [vLLM](https://github.com/vllm-project/vllm) on NVIDIA GH200 ARM64 GPUs, specifically optimized for the NRIS Olivia HPC cluster. Includes full-featured CLI tooling, streaming chat client, and performance optimizations for high-latency SSH tunnel connections.

## Features

- **Preserves NGC PyTorch** - Builds vLLM without overwriting NVIDIA's custom PyTorch
- **Model Presets** - Pre-configured builds for GLM-4.7, Devstral, Llama, and Qwen
- **GH200 Optimizations** - NCCL/NVLink tuning, optimal GPU ordering, Flash Attention
- **Unified CLI** - Single command interface with SSH ControlMaster (one 2FA per session)
- **Streaming Proxy** - Batches SSE tokens for 3x faster streaming over SSH tunnels
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
export REMOTE_PORT=8000
export LOCAL_PORT=8000
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

# Build containers
./olivia.sh build glm47        # Build GLM-4.7 container
./olivia.sh build devstral     # Build Devstral container
./olivia.sh build llama        # Build Llama container

# Build options
./olivia.sh build glm47 --index 2    # Build second container (safe, won't touch existing)
./olivia.sh build glm47 --force      # Rebuild existing container
./olivia.sh build glm47 --sif        # Create SIF image after build
./olivia.sh build glm47 --vllm v0.6.6  # Override vLLM version
```

**Safety:** Builds fail by default if a container already exists. Use `--index N` to create a new container or `--force` to explicitly overwrite.

### Server Module

```bash
./olivia.sh server                   # Show server help
./olivia.sh server list              # List available containers
./olivia.sh server status            # Show running server status

# Start servers (preset with default model)
./olivia.sh server start glm47       # Start GLM-4.7 server
./olivia.sh server start devstral    # Start Devstral server
./olivia.sh server start llama       # Start Llama server

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

| Preset | Default Model | vLLM | Transformers |
|--------|---------------|------|--------------|
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | main | >=5.0.0rc0 |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | main | >=4.45.0 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | main | >=4.45.0 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | main | >=4.45.0 |
| `generic` | *(user specified)* | main | >=4.45.0 |

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
| `VLLM_VERSION` | `main` | vLLM version to build |
| `CREATE_SIF` | `0` | Create SIF image after build |
| `OVERWRITE` | `0` | Allow overwriting existing containers |
| `MAX_JOBS` | `8` | Parallel compilation jobs |
| `CONTAINER_DIR` | *(required)* | Output directory |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER` | *(required)* | Container name or path |
| `MODEL` | `mistralai/Devstral-2-123B-Instruct-2512` | HuggingFace model ID |
| `TP_SIZE` | `4` | Tensor parallel size |
| `GPU_MEM_UTIL` | `0.90` | GPU memory utilization |
| `MAX_MODEL_LEN` | `32768` | Maximum context length |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models |
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
# Basic usage
python chat_devstral.py localhost --port 8000 --stream

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
