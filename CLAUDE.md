# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for building and running vLLM on NVIDIA GH200 (GraceHopper) ARM64 GPUs on HPC clusters. The key challenge is preserving NGC's custom PyTorch build while installing vLLM and its dependencies.

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

# Build containers (deploys script and submits SLURM job)
./olivia.sh build glm47        # Build GLM-4.7 container
./olivia.sh build gemma4       # Build Gemma 4 31B container
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
./olivia.sh server start glm47       # Start GLM-4.7 server (4 GPUs)
./olivia.sh server start gemma4      # Start Gemma 4 31B server (2 GPUs)
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
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | TP=4, MTP speculative |
| `gemma4` | `QuantTrio/gemma-4-31B-it-AWQ` | 2 | TP=2, dense, multimodal |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | TP=4 |

**Per-preset GPU allocation:** `olivia.sh server start` passes `--gpus=N --cpus-per-task=N*8` to `sbatch`, overriding the `#SBATCH --gpus=4` directive baked into `run_vllm_server.sh`. Gemma 4 31B AWQ is small enough that TP=4 would waste GPUs and add needless all-reduces, so the preset runs on 2 GPUs.

#### Tunnel Module
```bash
./olivia.sh tunnel             # Show tunnel status
./olivia.sh tunnel up          # Open tunnel to vLLM server
./olivia.sh tunnel down        # Close tunnel
```

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
| `glm47` | GLM-4.7 (358B) flagship model | main | >=5.0.0rc0 |
| `gemma4` | Gemma 4 31B (dense, multimodal text+image) | v0.19.0 | >=5.5.0 |
| `devstral` | Devstral/Mistral models | main | >=4.45.0 |
| `llama` | Llama 3.x models | main | >=4.45.0 |
| `qwen` | Qwen 2.5 models | main | >=4.45.0 |
| `generic` | Generic build (default) | main | >=4.45.0 |

Custom MODEL_ID values are also supported - they will use generic defaults.

To add a new preset, edit the `apply_preset()` function in `build_vllm_gh200.sh`.

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
- `MAX_MODEL_LEN`: Maximum context length (default: 32768)
- `HF_TOKEN`: HuggingFace token for gated models
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: `FLASH_ATTN`)
- `VERBOSE`: Set to `1` for detailed logging including weight loading progress (default: `0`)
- `VLLM_LOGGING_LEVEL`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`, or `DEBUG` if VERBOSE=1)

**Speculative Decoding (ngram):**
- `ENABLE_SPECULATIVE`: Enable speculative decoding (default: `0`)
- `NUM_SPECULATIVE_TOKENS`: Number of tokens to speculate (default: 5)
- `PROMPT_LOOKUP_MAX`: Max n-gram window size (default: 4)

**GLM-4.7 Specific (auto-detected when MODEL contains "GLM-4.7"):**
- `GLM_TOOL_PARSER`: Tool call parser (default: `glm47`)
- `GLM_REASONING_PARSER`: Reasoning parser (default: `glm45`)
- `ENABLE_AUTO_TOOL_CHOICE`: Enable automatic tool selection (default: `0` for GLM, `1` for Gemma)
- `SERVED_MODEL_NAME`: Custom model name for API (default: empty, uses model ID)
- `MTP_SPECULATIVE_TOKENS`: MTP speculative tokens for GLM-4.7 (default: `1`)

**Gemma 4 Specific (auto-detected when MODEL contains "gemma-4"):**
- `GEMMA_TOOL_PARSER`: Tool call parser (default: `gemma4`)
- `GEMMA_REASONING_PARSER`: Reasoning parser (default: `gemma4`)
- When Gemma 4 is detected, `TP_SIZE` defaults to `2` (not `4`), `MAX_MODEL_LEN` defaults to `65536`, `ENABLE_AUTO_TOOL_CHOICE` defaults to `1`, `--trust-remote-code` is added, and `VLLM_USE_FLASHINFER_MOE_FP8` is disabled (dense architecture)

**MoE (Mixture of Experts) / AWQ Settings:**
- `ENABLE_EXPERT_PARALLEL`: Enable expert parallel sharding (default: `auto` - auto-enables for AWQ MoE models)

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

### Gemma 4 Quantization Options

| Model | Size | GH200 Compatible | Notes |
|-------|------|------------------|-------|
| `QuantTrio/gemma-4-31B-it-AWQ` | ~20 GiB | **Yes (Recommended)** | AWQ 4-bit, fits on 1×GH200, TP=2 gives long-context headroom |
| `google/gemma-4-31B-it` | ~62 GB | Yes | BF16, fits tightly on 1×GH200 |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | ~33 GB | **Avoid** | Known vLLM bug: produces gibberish (vllm-project/vllm#39049) |
| `RedHatAI/gemma-4-31B-it-FP8-block` | ~33 GB | **Avoid** | Known vLLM bug: logit softcap interaction (vllm-project/vllm#39407) |
| `nvidia/Gemma-4-31B-IT-NVFP4` | ~15 GB | **No** | Requires Blackwell FP4 tensor cores |

**Important:**
- **Avoid Gemma 4 FP8 variants for now.** Two active vLLM bugs make both the FP8-Dynamic and FP8-Block checkpoints unusable — the softcap/scale interaction produces garbage output. Stick with AWQ until these land.
- Gemma 4 31B is a **dense** model (not MoE), so expert-parallel flags and MoE flashinfer optimizations don't apply. The server script auto-disables them when `gemma-4` is detected in the model ID.
- Gemma 4 has its own `gemma4` tool/reasoning parser in vLLM — the server script sets `--tool-call-parser gemma4 --reasoning-parser gemma4 --enable-auto-tool-choice --trust-remote-code` automatically.
- Gemma 4 has a native **256K context window**. The preset sets `MAX_MODEL_LEN=65536` as a safe default; increase at your own risk (KV cache scales linearly).

### Gemma 4 Usage

```bash
# Build Gemma 4 container
MODEL_ID=gemma4 ./build_vllm_gh200.sh

# Run Gemma 4 with AWQ (RECOMMENDED for GH200)
# Uses TP=2 by default, fits easily with room for long context
CONTAINER=vllm-gemma4-1-sandbox MODEL=QuantTrio/gemma-4-31B-it-AWQ ./run_vllm_server.sh

# Via olivia.sh (automatically allocates 2 GPUs, TP=2)
./olivia.sh server start gemma4

# Run with longer context (up to 256K)
CONTAINER=vllm-gemma4-1-sandbox MODEL=QuantTrio/gemma-4-31B-it-AWQ \
    MAX_MODEL_LEN=131072 ./run_vllm_server.sh

# Override to single GPU (tighter fit, ~32K context)
CONTAINER=vllm-gemma4-1-sandbox MODEL=QuantTrio/gemma-4-31B-it-AWQ \
    TP_SIZE=1 MAX_MODEL_LEN=32768 ./run_vllm_server.sh
```

**Memory requirements for Gemma 4 31B (30.7B parameters, dense):**
| Quantization | Model Size | 1×GH200 (96GB) | 2×GH200 (192GB) | Notes |
|--------------|------------|----------------|------------------|-------|
| AWQ 4-bit | ~20 GiB | Yes | Recommended | TP=2, headroom for 256K context |
| BF16 | ~62 GB | Yes, tight | Yes | No quant bugs |
| FP8 | ~33 GB | N/A | N/A | **Broken in vLLM right now** |

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

### Container/Cache Structure

**Shared containers on the cluster** (`CONTAINER_DIR`):
- `vllm-{model}-{index}-sandbox/`: Singularity sandbox (writable container)
- `vllm-{model}-{index}.sif`: Compressed Singularity image (optional)

Naming examples:
- `vllm-glm47-1-sandbox` - GLM-4.7 build #1
- `vllm-gemma4-1-sandbox` - Gemma 4 31B build #1
- `vllm-devstral-1-sandbox` - Devstral build #1
- `vllm-generic-1-sandbox` - Generic build #1

**Local cache directories**:
- `cache/pip/`: Pip cache directory
- `cache/huggingface/`: HuggingFace model cache
- `cache/vllm/`: vLLM cache
- `logs/`: Build and server logs
