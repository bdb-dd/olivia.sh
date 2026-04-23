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
./olivia.sh build glm51        # Build GLM-5.1 container (vLLM v0.19.0)
./olivia.sh build glm47        # Build GLM-4.7 container
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
./olivia.sh server start glm51       # Start GLM-5.1 server (8 GPUs across 2 nodes, TP=4 + PP=2)
./olivia.sh server start glm47       # Start GLM-4.7 server
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
| `glm51` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | TP=4 + PP=2, MTP speculative, Ray cluster |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | TP=4, MTP speculative |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | TP=4 |

**Per-preset GPU allocation:** `olivia.sh server start` reads resources from `get_preset_resources()` in `olivia.sh` and passes corresponding `--nodes`, `--gpus-per-node`, `--cpus-per-task` overrides to `sbatch`. The `glm51` preset is the only one today that crosses a node boundary (2 nodes × 4 GPUs); everything else runs single-node 4 GPUs.

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
| `glm51` | GLM-5.1 (744B / 40B active) MoE+DSA flagship | v0.19.0 | >=5.3.0.dev0 |
| `glm47` | GLM-4.7 (358B) flagship model | main | >=5.0.0rc0 |
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
- `MAX_MODEL_LEN`: Maximum context length (default: `131072` for GLM-5.1 — native 205K context with ~260 GB KV cache headroom on 8×GH200 AWQ; `32768` elsewhere)
- `HF_TOKEN`: HuggingFace token for gated models
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: `FLASH_ATTN`)
- `VERBOSE`: Set to `1` for detailed logging including weight loading progress (default: `0`)
- `VLLM_LOGGING_LEVEL`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`, or `DEBUG` if VERBOSE=1)
- `CUDAGRAPH_MODE`: vLLM compilation / CUDAGraph mode (default: unset → vLLM auto-selects `PIECEWISE` or `FULL_AND_PIECEWISE`). Set `NONE` to disable graph capture (eager, slow — debugging escape hatch). Capture adds a multi-minute "Capturing cudagraph" phase at startup but typically yields 2-5× decode throughput.

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

**Multi-node Distributed Inference (GLM-5.1):**
- `NUM_NODES`: Number of nodes to use (default: `1`). When `>1`, `run_vllm_server.sh` bootstraps a Ray cluster via `srun`, starts head on node 0, workers on remaining nodes, waits for the cluster to register the expected GPU count, then launches vLLM with `--distributed-executor-backend ray`
- `PP_SIZE`: Pipeline parallel size (default: `1`). For 2-node glm51 jobs this is `2`
- `TP_SIZE`: Tensor parallel size (default: `4`). For multi-node jobs, this is the *intra-node* tensor parallel size (per node)
- `RAY_PORT`: Ray GCS port (default: `6379`)
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

### GLM-5.1 Usage

```bash
# Build GLM-5.1 container
./olivia.sh build glm51
# or direct:
MODEL_ID=glm51 sbatch build_vllm_gh200.sh

# Start GLM-5.1 server (preset auto-allocates 2 nodes × 4 GPUs)
./olivia.sh server start glm51

# Watch loading progress (handles multi-node GPU aggregation automatically)
./olivia.sh server watch

# Direct sbatch invocation for glm51 (multi-node overrides required)
CONTAINER=vllm-glm51-1-sandbox MODEL=cyankiwi/GLM-5.1-AWQ-4bit \
    NUM_NODES=2 TP_SIZE=4 PP_SIZE=2 \
    sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 \
    run_vllm_server.sh

# Override context length (GLM-5.1 has native 205K context window)
./olivia.sh server start glm51
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
# 1. Tunnel to the cluster (forwards localhost:8000 → vllm:8000 on the GPU node).
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
- `--upstream` — OpenAI-compatible URL (default `http://localhost:8000`, direct vLLM). Set to `http://localhost:8001` to chain through `vllm_proxy` (requires `ENABLE_PROXY=1` server-side).
- `--model` *(required)* — the model name forwarded to vLLM (e.g. `cyankiwi/GLM-5.1-AWQ-4bit`). All Anthropic model names in client requests are remapped to this single value.
- `-v` / `--verbose` — log request/response bodies for debugging.

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
- `vllm-devstral-1-sandbox` - Devstral build #1
- `vllm-generic-1-sandbox` - Generic build #1

**Local cache directories**:
- `cache/pip/`: Pip cache directory
- `cache/huggingface/`: HuggingFace model cache
- `cache/vllm/`: vLLM cache
- `logs/`: Build and server logs
