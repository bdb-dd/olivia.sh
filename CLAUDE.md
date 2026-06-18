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
| `glm51_v19` (alias `glm51`) | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | TP=4 + PP=2, vLLM v0.19.0, container index 1. Pair with `anthropic_proxy.py` serialization to work around multi-node PP decode wedge. |
| `glm51_v20` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | Same as glm51_v19 but on vLLM v0.20.0 + RayExecutorV2, container index 2. **Quarantined** — same wedge as v0.19.0; kept for diagnostic work only. |
| `glm52` | `RedHatAI/GLM-5.2-FP8` | 12 (3 nodes × 4) | TP=4 + PP=3. Block-FP8 (~755 GB) — does **not** fit 8 GPUs, hence 3 nodes. **Needs vLLM main + PR#45895** (new skip-topk DSA indexer); not in any release. Same multi-node PP wedge as glm51 → proxy serialization. fp8 KV cache + DeepGEMM (`VLLM_DEEP_GEMM_WARMUP=skip`). |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | TP=4 + PP=2, native int4, MLA, multimodal, vLLM v0.19.1 |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | TP=4 |

**Per-preset GPU allocation:** `olivia.sh server start` reads resources from `get_preset_resources()` in `olivia.sh` and passes corresponding `--nodes`, `--gpus-per-node`, `--cpus-per-task` overrides to `sbatch`. `glm51` (2 nodes × 4) and `glm52` (3 nodes × 4) cross node boundaries; everything else runs single-node 4 GPUs. The `cluster` snapshot reports start-time/feasibility for 1-, 2-, and 3-node × 4-GPU shapes.

#### Prefetch Module
```bash
./olivia.sh prefetch glm51_v19                  # preset -> its default repo
./olivia.sh prefetch cyankiwi/GLM-5.1-AWQ-4bit  # explicit HuggingFace repo id
./olivia.sh prefetch glm47 --revision <sha>     # pin a revision
./olivia.sh prefetch <model> --no-follow        # start, don't tail progress
```

Downloads model weights into the persistent HuggingFace cache (`HF_HOME`) **from a login
node** — never a GPU/SLURM job, so the multi-hundred-GB transfer doesn't burn allocation. The
download is **detached** (`setsid`, survives closing the CLI) and **resumable** (skips
already-complete blobs). Run it once before `server start` so the server job serves
immediately instead of downloading inside the allocation.

Mechanism notes:
- The Olivia login node is **amd64** while the GH200 compute nodes are **arm64**, so the
  containers can't exec on the login node. Prefetch instead builds a one-time throwaway
  `python3.12` venv (under `<HF_HOME parent>/.prefetch/venv`) with `huggingface_hub` + `hf_xet`.
- `HF_HOME` is pinned explicitly from your local environment (see "Persistent model cache"
  under Container/Cache Structure); `HF_TOKEN` is forwarded over stdin for gated repos.

#### Tunnel Module
```bash
./olivia.sh tunnel             # Show tunnel status
./olivia.sh tunnel up          # Open tunnel to vLLM server
./olivia.sh tunnel refresh     # Re-point tunnel after the job moved nodes
./olivia.sh tunnel down        # Close tunnel (also stops the login-node relay)
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

**Login-node follow-the-node relay (opt-in: `--login-proxy` / `LOGIN_PROXY=1`):**
Normally the local forward targets the GPU node directly, so a job moving to a
new node requires re-pointing the forward. With `--login-proxy`, the laptop
forwards to a *fixed* login-node port and a small user-space Python relay on the
login node (`~/.olivia/relay.py`, managed by olivia.sh) forwards to the current
GPU node. Node moves then only re-point the relay; the local forward is never
disturbed. `LOGIN_PROXY_PORT` defaults to `18000`.

> ⚠️ This runs a long-lived process on the *shared* login node — check the NRIS
> acceptable-use policy before relying on it. Validate on the cluster: it
> assumes the login node can reach `<gpu-node>:8000` directly (the same path the
> direct forward already uses) and that `python3` is on the login node.

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
| `glm52` | GLM-5.2 (744B / 40B active) MoE+DSA, successor to 5.1 — FP8. Builds vLLM main + auto-grafts PR#45895 (`VLLM_PATCHES`); not in any release | main + PR#45895 | >=5.4.0 |
| `glm47` | GLM-4.7 (358B) flagship model | main | >=5.0.0rc0 |
| `kimi` | Kimi K2.6 (1T / 32B active) MoE + MLA, multimodal | v0.19.1 | >=4.57.1,<5.0.0 |
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
- `HF_HOME`: Persistent HuggingFace cache directory for model weights (**required**). Must live on persistent project storage (e.g. `/cluster/projects/<proj>/huggingface`), **not** `/cluster/work` (auto-purged after 21–42 days — see "Persistent model cache" under Container/Cache Structure). `olivia.sh` forwards it from your local environment (`mise.local.toml`); `run_vllm_server.sh` errors if it is unset and warns if it resolves under `/cluster/work`. `HF_CACHE` is a back-compat alias.
- `HF_TOKEN`: HuggingFace token for gated models. `olivia.sh` forwards it over stdin (kept off argv and out of logs), so it lives only in local config — not the cluster shell.
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

**Upstream status (as of 2026-06-17): not in any vLLM release.** The skip-topk indexer path is implemented by upstream **PR#45895** ("Indexer init skip and MTP TopK share for iteration"), created 2026-06-17 and still unmerged. v0.23.0 was tagged two days before it. The build preset pins `VLLM_VERSION=main` and **grafts PR#45895 at build time** via the `VLLM_PATCHES` mechanism (`PRESET_VLLM_PATCHES="45895"`), so `./olivia.sh build glm52` is **not blocked on the merge** — see "Patch-during-build" below. A newer vLLM main may also require `NGC_PYTORCH_TAG=26.04-py3` or later. Our multi-node PP decode wedge workaround (proxy serialization) is **not** superseded — the upstream wedge issues (#26318, #30044, #24689) all went stale/closed-unfixed.

**Patch-during-build (`VLLM_PATCHES`):** `build_vllm_gh200.sh` takes a space-separated list of vLLM PR numbers (`VLLM_PATCHES`, defaulting to the preset's `PRESET_VLLM_PATCHES`). In Phase 3, after cloning vLLM and before the `pip install .` compile, it fetches each PR's cumulative diff from `github.com/vllm-project/vllm/pull/<N>.diff` and `git apply`s it to `/opt/vllm` (the build container has GitHub egress — it already clones vLLM there). The step is **idempotent** (a reverse-apply check skips PRs already present — e.g. once one merges into main) and **fails the build loudly** if a diff no longer applies cleanly (rather than compiling a half-patched tree). PR#45895 is pure Python (9 files), so no kernel recompile is triggered. Validated 2026-06-17: the diff applies cleanly to main `68ff30d`. A snapshot lives at `patches/vllm-pr45895-glm52-indexer.diff`; `olivia.sh build` deploys `patches/` alongside the build script and binds it read-only at `/opt/olivia-patches`, and the build **prefers the committed snapshot** (reproducible + offline), falling back to the **live** GitHub PR fetch only when no snapshot is bound in. **When PR#45895 merges:** drop `"45895"` from `PRESET_VLLM_PATCHES` (the idempotent apply tolerates it in the meantime), or build with `VLLM_PATCHES=""`.

**Why 3 nodes (not 2 like glm51):** the only quant available today is block-FP8 (`zai-org/GLM-5.2-FP8`, or the byte-identical re-host `RedHatAI/GLM-5.2-FP8`), ~755 GB. At ~94 GB/GPU that does **not** fit 8×GH200 (96 GB cards). Spreading across **3 nodes × 4 GPUs (TP=4 + PP=3 = 12 GPUs)** drops weights to ~63 GB/GPU, leaving ~18 GB/GPU for KV. When a `GLM-5.2-AWQ-4bit` (~430 GB) is eventually published, the 8-GPU / 2-node path returns and is preferred — repoint the preset model and set `nodes=2 pp=2`.

| Quantization | Model | Size | Olivia fit |
|--------------|-------|------|------------|
| block-FP8 (e4m3, [128,128]) | `zai-org/GLM-5.2-FP8` / `RedHatAI/GLM-5.2-FP8` | ~755 GB | **3 nodes × 4 GH200** (TP=4 + PP=3); 8 GPUs won't fit |
| AWQ-4bit | — | ~430 GB | Does **not exist yet** (would be the preferred 2-node / 8-GPU path) |
| NVFP4 | `Lorbus/GLM-5.2-NVFP4` etc. | — | **No** — needs Blackwell FP4 tensor cores |
| BF16 | `zai-org/GLM-5.2` | ~1.5 TB | No — 16+ GPUs |

**Runtime specifics auto-applied by `run_vllm_server.sh` when `MODEL` contains `GLM-5.2`** (`IS_GLM52`): `--trust-remote-code`, glm47 tool parser + glm45 reasoning parser, auto tool choice, sparse-MLA attention backend (auto-select), `MAX_MODEL_LEN=131072` default, **block-FP8 path** — `VLLM_USE_DEEP_GEMM=1`, `VLLM_DEEP_GEMM_WARMUP=skip` (avoids the multi-minute DeepGEMM JIT warmup; override with `VLLM_DEEP_GEMM_WARMUP=""`), and `--kv-cache-dtype fp8_e4m3` (roughly doubles fittable context/concurrency). MTP speculative decoding is auto-disabled because `PP_SIZE>1` (the existing PP+MTP guard). The cyankiwi chat-template override is **5.1-only** and is skipped for 5.2 (the FP8 repos ship a correct base template). PR#45895's validated flag set is the source of these defaults.

**`reasoning_tokens` on `/v1/chat/completions` usage (build patch):** upstream vLLM only surfaces `reasoning_tokens` on `/v1/responses` — chat/completions has no `completion_tokens_details` at all, and the glm45 reasoning parser (`DeepSeekV3ReasoningWithThinkingParser`) never counts reasoning tokens (the base `count_reasoning_tokens()` returns 0, the V3 wrapper doesn't forward it, and GLM emits `</think>` but omits the `<think>` start token so the inner R1 parser's start/end depth counter is 0 too). `build_vllm_gh200.sh` (`PYPATCH_REASONING_TOKENS`, applied in Phase 3 after the PR graft) fixes all three: it adds `count_reasoning_tokens` to **both** the glm45 (`DeepSeekV3ReasoningParser`) and `kimi_k2` parsers (count tokens before the first reasoning-end marker), adds `CompletionTokenUsageInfo` + `UsageInfo.completion_tokens_details` to `engine/protocol.py`, and populates it in the non-streaming and streaming usage paths of `chat_completion/serving.py` — tolerant of both vLLM 0.21 (`reasoning_parser` / `all_previous_token_ids`) and vLLM main (`parser`/`parsers` + its own `previous_token_ids` accumulator) serving layouts (0.21-specific anchors tried first under a guard, so the main-only `parser` var is never injected into 0.21). The patch is idempotent and defensive (a missing anchor logs + no-ops, so a future vLLM refactor never half-patches or fails the build) and the runtime guards leave non-reasoning models unaffected, so it runs on every build — **any reasoning model (glm47/glm51/glm52 and Kimi K2.x) gets it from this one block.** **Takes effect on the next container rebuild.**

**Weights storage:** prefetched to the at-risk-but-spacious work cache, not the quota-bound project area — the 703.7 GiB FP8 does not fit the ~462 GiB free on `/cluster/projects/nn10104k` (1 TiB quota). Override `HF_HOME` to `/cluster/work/projects/nn10104k/huggingface` for both prefetch and serving (see [[project_hf_storage_split]]):
```bash
HF_HOME=/cluster/work/projects/nn10104k/huggingface ./olivia.sh prefetch RedHatAI/GLM-5.2-FP8
```

### GLM-5.2 Usage

```bash
# 1. Prefetch FP8 weights to the work cache (HF_HOME override; detached, resumable)
HF_HOME=/cluster/work/projects/nn10104k/huggingface ./olivia.sh prefetch RedHatAI/GLM-5.2-FP8

# 2. Build the container. The glm52 preset builds vLLM main and auto-grafts
#    PR#45895 (VLLM_PATCHES=45895) for GLM-5.2's skip-topk indexer — no need to
#    wait for the upstream merge. Disable the graft with VLLM_PATCHES="" if the
#    PR has since merged into main.
./olivia.sh build glm52

# 3. Start the server (preset auto-allocates 3 nodes × 4 GPUs, TP=4 + PP=3)
HF_HOME=/cluster/work/projects/nn10104k/huggingface ./olivia.sh server start glm52

# Check whether a 3-node shape can schedule right now
./olivia.sh cluster
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

**Persistent model cache (`HF_HOME`)**: HuggingFace model weights (hundreds of GB) live in a persistent project area — e.g. `/cluster/projects/<proj>/huggingface`. They must **not** sit on `/cluster/work`, which NRIS auto-purges after 21–42 days (this silently deleted a ~430 GB GLM-5.1 AWQ cache once, leaving only metadata + dangling symlinks). `HF_HOME` is set in `mise.local.toml` and forwarded to jobs by `olivia.sh`; populate it with `./olivia.sh prefetch`. Note `/cluster/work` and `/cluster/projects` are different Lustre filesystems, so migrating weights between them is copy-then-delete (no cross-FS hardlinks).

**Ephemeral compile/JIT caches**: Triton, DeepGEMM, TorchInductor and vLLM compile caches are regenerable and stay under `$PWD/cache/` on `/cluster/work` (`run_vllm_server.sh` sets `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`). They're large and high-churn, so auto-purge is harmless — but they must avoid the small home quota (defaulting them to `~/.triton` crashes jobs mid-profile with `Disk quota exceeded`).
