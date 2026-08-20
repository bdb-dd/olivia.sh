#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=accel
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
# Walltime default is deliberately CONSERVATIVE (2 h), not the old 8 h.
# An interactive serve session bills every allocated hour whether or not a
# request ever arrives, and 8 h is long enough that an unattended job wastes
# most of it: qwen38 job 2022174 (2026-08-17) lost the SSH master mid-bench and
# then sat idle to the full 8 h TIMEOUT, burning ~8 GPU-hours to produce nothing.
# Pick the limit to fit the task and pass TIME_LIMIT for anything longer:
#   TIME_LIMIT=00:45:00 ./olivia.sh server start qwen38   # serve + one sweep
#   TIME_LIMIT=06:00:00 ./olivia.sh server start glm52    # long soak
# NB SLURM only lets a user LOWER a running job's limit, so raising it after
# submit is impossible — but `scontrol update jobid=N TimeLimit=..` to shorten
# an over-long job still works and is worth doing the moment you know.
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.log
# NOTE: Neither --ntasks nor --gpus/--gpus-per-node are set here as #SBATCH
# directives. olivia.sh always passes the right spec on the sbatch command line:
#   Single-node: --gpus=N
#   Multi-node:  --nodes=N --ntasks=N --ntasks-per-node=1 --gpus-per-node=M
# Setting #SBATCH --ntasks=1 would silently downgrade multi-node submissions to
# 1 node (SLURM reasons: "1 task needs only 1 node"). Setting #SBATCH --gpus=4
# conflicts with --gpus-per-node (SLURM: "4 GPUs fits on 1 node"). If you sbatch
# this script directly (not via olivia.sh), remember to pass --gpus=N yourself.
# Job name: olivia.sh overrides the static --job-name=vllm-server above with a
# per-branch --job-name=vllm-<DEPLOY_KEY>, so concurrent agents' servers don't
# collide in squeue-based resolution. Direct sbatch keeps the plain vllm-server.

# =============================================================================
# Run vLLM Server on GH200
# Optimized for NVIDIA GraceHopper with NVLink
# =============================================================================

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p "${WORKDIR:-$PWD}/logs"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Container directory (shared location on Olivia)
CONTAINER_DIR="${CONTAINER_DIR:-}"

# Container name (can be just the name or full path)
CONTAINER="${CONTAINER:-}"

# Model (default: Devstral 123B)
MODEL="${MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"

# Server settings
PORT="${PORT:-8000}"
# HOST: bind address for vLLM's HTTP server.
# Force 0.0.0.0 (all interfaces) so the SSH tunnel — which connects to
# ${gpu_node}:${port} via the login node — can reach us. We DON'T default to
# ${HOST:-...} because bash commonly auto-sets $HOST to the login-shell
# hostname (e.g., "uan02"), which leaks into the sbatch environment and causes
# vLLM to try binding a non-local address: OSError [Errno 99] EADDRNOTAVAIL.
# If you need to restrict to a specific interface, set VLLM_BIND_HOST instead.
HOST="${VLLM_BIND_HOST:-0.0.0.0}"

# Batching proxy settings (reduces streaming overhead over SSH tunnels)
# The proxy batches multiple tokens into single SSE events
ENABLE_PROXY="${ENABLE_PROXY:-0}"         # Set to 1 to enable batching proxy
PROXY_PORT="${PROXY_PORT:-8001}"          # Port for the batching proxy
PROXY_BATCH_TOKENS="${PROXY_BATCH_TOKENS:-15}"    # Flush after N tokens
PROXY_BATCH_CHARS="${PROXY_BATCH_CHARS:-100}"     # Flush after N characters
PROXY_BATCH_DELAY_MS="${PROXY_BATCH_DELAY_MS:-150}"  # Max delay before flush (ms)

# GPU settings
TP_SIZE="${TP_SIZE:-4}"                    # Tensor parallel size (per-node for multi-node)
PP_SIZE="${PP_SIZE:-1}"                    # Pipeline parallel size (1=single-node, 2=across nodes)
NUM_NODES="${NUM_NODES:-1}"                # Number of nodes to use (1 or 2)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"       # GPU memory utilization
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"       # Per-GPU weight offload to CPU/Grace LPDDR5X (GiB); empty=none
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-}"         # Native CPU KV-cache tier (GiB, total across TP ranks); empty=none. Auto-drops expandable_segments
# Ray compiled-DAG step timeout (seconds). Ray v2's default of 300s is too
# short for multi-node PP inference over Slingshot: a single engine step can
# run longer than that during big generations, and the raylet hits an
# assertion failure / tears the cluster down. 1800s (30 min) is generous.
# Only applied for multi-node (NUM_NODES > 1); single-node unaffected.
RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-1800}"
# Select RayExecutorV2 over the legacy RayDistributedExecutor. V2 inherits
# MultiprocExecutor's ZMQ/NCCL data plane and bypasses Ray's Compiled Graph
# entirely — structurally avoids the MutableObjectProvider deadlock on
# cross-node PP (ray#58426, vllm#26318, #26899). Required in vLLM v0.20.0+
# where V2 exists but is off by default (upstream TODO to flip).
# NOTE: on vLLM main, V2 breaks with our external multi-node Ray bootstrap —
# the subprocess EngineCore re-inits Ray as a new job, so the API server's
# worker ActorHandles are invalid across sessions (ActorHandleNotFoundError).
# GLM-5.2 therefore defaults to the legacy executor below. Capture whether the
# user pinned this so that default isn't silently overridden.
if [[ -n "${VLLM_USE_RAY_V2_EXECUTOR_BACKEND+x}" ]]; then _RAYV2_EXPLICIT=1; else _RAYV2_EXPLICIT=0; fi
VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
# (We tried VLLM_ENABLE_V1_MULTIPROCESSING=0 to run the EngineCore in-process and
# share one Ray session — but vLLM main ignores it; the V1 engine always forks a
# subprocess EngineCore. The legacy executor below is what actually avoids the
# cross-session ActorHandle failure.)
# MAX_MODEL_LEN default is model-dependent and gets resolved after GLM
# detection below: 131072 for GLM-5.1 (native 205K context, plenty of KV
# cache headroom on 8×GH200 AWQ), 32768 elsewhere.

# Speculative decoding settings
# - "auto": Enable MTP for GLM-4.7/GLM-5.1, disabled for other models
# - "1": Force enable (MTP for GLM-4.7/GLM-5.1, ngram for others)
# - "0": Force disable
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-auto}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-4}"  # Max n-gram window size

# GLM-4.7/GLM-5.1 specific settings (auto-detected from MODEL)
# GLM-5.1 reuses the GLM-4.7 tool/reasoning parsers per the official vLLM recipe
GLM_TOOL_PARSER="${GLM_TOOL_PARSER:-glm47}"
GLM_REASONING_PARSER="${GLM_REASONING_PARSER:-glm45}"
# Kimi K2.x (Moonshot) parsers — used when MODEL contains "Kimi-K2". Per the
# official deploy guide both are kimi_k2: --tool-call-parser enables tool calls,
# and --reasoning-parser is required because K2.6 runs thinking mode by default.
# (Set KIMI_REASONING_PARSER="" only if you deliberately want instant-mode raw
# output without reasoning extraction.)
KIMI_TOOL_PARSER="${KIMI_TOOL_PARSER:-kimi_k2}"
KIMI_REASONING_PARSER="${KIMI_REASONING_PARSER:-kimi_k2}"
# Laguna (Poolside) parsers — used when MODEL contains "Laguna". vLLM's PR#41129
# registers "poolside_v1" as both the tool-call and reasoning parser. Laguna has
# native reasoning; the reasoning parser extracts it into reasoning_content.
# Set either to "" to omit it. Thinking mode: the model card's recommended serve
# command passes enable_thinking via the default chat-template kwargs — keep it
# behind LAGUNA_ENABLE_THINKING (set to 0 to serve instant, no reasoning).
LAGUNA_TOOL_PARSER="${LAGUNA_TOOL_PARSER:-poolside_v1}"
LAGUNA_REASONING_PARSER="${LAGUNA_REASONING_PARSER:-poolside_v1}"
LAGUNA_ENABLE_THINKING="${LAGUNA_ENABLE_THINKING:-1}"
# Ornith 1.0 (Deep Reinforce) settings — used when MODEL contains "Ornith". The
# model card's vLLM serve command uses the qwen3_xml tool-call parser + qwen3
# reasoning parser (Ornith emits <think> reasoning). Set either to "" to omit.
ORNITH_TOOL_PARSER="${ORNITH_TOOL_PARSER:-qwen3_xml}"
ORNITH_REASONING_PARSER="${ORNITH_REASONING_PARSER:-qwen3}"
# Ornith ships a NATIVE MTP module (config mtp_num_hidden_layers=1), so vLLM runs
# Multi-Token-Prediction speculative decode from the model's own head — no
# external draft model. Method name and draft depth are overridable: the method
# is vLLM's generic "mtp" (the qwen3.5/3.6 recipe), draft depth 2 by default (the
# single MTP layer is applied autoregressively). Tune 1-3 and watch accept rate.
ORNITH_MTP_METHOD="${ORNITH_MTP_METHOD:-mtp}"
ORNITH_MTP_SPECULATIVE_TOKENS="${ORNITH_MTP_SPECULATIVE_TOKENS:-2}"
# GDN (Gated DeltaNet) linear-attention prefill backend. On Hopper vLLM defaults
# ("auto") to FlashInfer for the GDN prefill kernel, but flashinfer 0.6.x is
# version-skewed against the NGC container's cutlass-dsl (its Blackwell kernel
# imports `cutlass.cute.nvgpu.OperandMajorMode`, absent here) — importing it
# crashes engine init, and it's ALSO probed by the FP8-MoE oracle. Force the
# in-tree Triton/FLA GDN kernel so Ornith serves with NO flashinfer dependency.
# (Verified on-cluster 2026-07-16: triton path serves; flashinfer path does not.)
ORNITH_GDN_PREFILL_BACKEND="${ORNITH_GDN_PREFILL_BACKEND:-triton}"
# --- Qwen3.8 27B (Alibaba, open-weighted 2026-08-13/14, Apache 2.0) -----------
# Same Qwen 3.5 lineage as Ornith and therefore the same parser family, but a
# DENSE 27B rather than a MoE, so it gets its own knobs.
QWEN38_TOOL_PARSER="${QWEN38_TOOL_PARSER:-qwen3_xml}"
QWEN38_REASONING_PARSER="${QWEN38_REASONING_PARSER:-qwen3}"
# GDN prefill backend. Ornith is pinned to "triton" because its container had a
# BROKEN flashinfer that the build uninstalled. That is NOT the situation here:
# the glm53/v0.27.1 container carries a working flashinfer 0.6.17 ("imports
# cleanly — keeping" in the build log), so leave this EMPTY and let vLLM pick.
# Set QWEN38_GDN_PREFILL_BACKEND=triton to force the in-tree Triton/FLA kernel if
# the flashinfer GDN path misbehaves — that is the known-good Ornith fallback.
QWEN38_GDN_PREFILL_BACKEND="${QWEN38_GDN_PREFILL_BACKEND:-}"
# --- DeepSeek-V4-Flash (deepseek-ai, MIT) ---------------------------------
# vLLM v0.27.1 ships DEDICATED deepseek_v4 tool + reasoning parsers (verified in
# the installed container, not guessed).
DSV4_TOOL_PARSER="${DSV4_TOOL_PARSER:-deepseek_v4}"
DSV4_REASONING_PARSER="${DSV4_REASONING_PARSER:-deepseek_v4}"
# MTP draft depth. The checkpoint REALLY ships the head — 4705 `mtp.*` tensors of
# 72317 verified in the downloaded weight_map (e.g. mtp.0.hc_attn_base) — and
# config declares num_nextn_predict_layers=1, so unlike Ornith this is not a
# config-only claim. Tune 1-3 and watch the accept rate.
DSV4_MTP_SPECULATIVE_TOKENS="${DSV4_MTP_SPECULATIVE_TOKENS:-2}"
# ENABLE_AUTO_TOOL_CHOICE default is model-dependent and gets resolved after
# GLM detection below: 1 for GLM MoE models (tool parser is always set),
# 0 elsewhere. Users can still override explicitly.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-3}"  # MTP speculative tokens (GLM-4.7 and GLM-5.1)

# MoE (Mixture of Experts) settings
# Expert parallel is required for AWQ-quantized MoE models to shard experts across GPUs
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-auto}"  # auto, 0, or 1

# CUDAGraph / vLLM compilation knob. Note this conflates two distinct fields
# in vLLM's compilation-config:
#   - `mode`: compilation backend (NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE)
#   - `cudagraph_mode`: capture strategy (NONE | PIECEWISE | FULL_AND_PIECEWISE | ...)
# We expose a single env var because the practical choices are:
#   - unset (default) -> let vLLM auto-select; on glm51 this picks
#       mode=VLLM_COMPILE + cudagraph_mode=FULL_AND_PIECEWISE.
#   - NONE -> emit `{"mode": "NONE"}`, fully disable compilation+capture (the
#       legacy hardcoded behavior; eager dispatch, robust, ~5× slower decode).
#   - PIECEWISE / FULL_AND_PIECEWISE / FULL -> emit
#       `{"cudagraph_mode": "<value>"}`, keep compilation enabled and only
#       override the capture strategy.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-}"

# vLLM's custom all-reduce kernel is NOT CUDAGraph-safe under TP=4+PP=2 multi-
# node on Slingshot — it raises `cudaErrorInvalidValue` at custom_all_reduce.cuh:455
# during the cudagraph capture phase, killing all Ray workers with SYSTEM_ERROR.
# NCCL's all-reduce IS graph-safe; falling back to it costs a few % on small
# reductions but unlocks the ~5× decode speedup from CUDAGraph capture.
# auto = on whenever CUDAGRAPH_MODE != NONE (i.e. whenever capture runs).
# Set to 0 to keep custom kernel even with capture enabled (will likely crash).
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-auto}"

# Cache directories — split by lifetime, because model weights and compile
# artifacts have opposite storage needs:
#
#   HF_HOME -> model weights (HuggingFace hub cache). Hundreds of GB and slow to
#     re-download, so this MUST be persistent. It must NOT live on /cluster/work,
#     which NRIS auto-purges after 21-42 days: that silently deletes downloaded
#     weights and forces a multi-hour re-download on the next run (exactly what
#     wiped the GLM-5.1 AWQ cache — blobs gone, only dangling symlinks left).
#     Point HF_HOME at a persistent project area, e.g.
#     /cluster/projects/<proj>/huggingface.
#
#   Triton / DeepGEMM / TorchInductor / vLLM compile caches -> regenerable JIT
#     artifacts: large, high-churn, cheap to rebuild, so they stay under $PWD on
#     /cluster/work where auto-purge is harmless. They must still avoid the small
#     home quota — defaulting them to ~/.triton fills it and crashes jobs
#     mid-profile with `OSError: [Errno 122] Disk quota exceeded`, especially for
#     MoE models with many experts × shapes (GLM-5.1 hit this under ~/.triton).
#
# HF_HOME is HuggingFace's canonical location var; HF_CACHE is kept as a
# back-compat alias. One must be set — no hardcoded cluster path here (matches
# the CONTAINER_DIR contract).
HF_CACHE="${HF_HOME:-${HF_CACHE:-}}"
if [[ -z "${HF_CACHE}" ]]; then
    echo "Error: HF_HOME is not set." >&2
    echo "Set it to a PERSISTENT model cache on project storage, e.g.:" >&2
    echo "    export HF_HOME=/cluster/projects/<your-project>/huggingface" >&2
    echo "Avoid /cluster/work — NRIS auto-purges it after 21-42 days and weights are lost." >&2
    exit 1
fi
case "${HF_CACHE}" in
    /cluster/work/*)
        echo "WARNING: HF_HOME=${HF_CACHE}" >&2
        echo "         is on /cluster/work, which NRIS auto-purges after 21-42 days —" >&2
        echo "         downloaded weights WILL be deleted and re-downloaded. Move it to a" >&2
        echo "         persistent area, e.g. /cluster/projects/<proj>/huggingface." >&2
        ;;
esac
VLLM_CACHE="${VLLM_CACHE:-$PWD/cache/vllm}"
# VLLM_CACHE_ROOT: controls where vLLM writes torch_compile_cache, modelinfos,
# deep_gemm warmup caches, etc. Default inside vLLM is `~/.cache/vllm`; since
# Singularity preserves host UID, `~` resolves to the user's home dir (small
# quota), NOT /root/.cache/vllm (which is where we bind VLLM_CACHE below). Must
# be set explicitly for CUDAGraph capture to work — compile cache is many
# shapes × layers and blows past home quota on GLM-5.1.
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${VLLM_CACHE}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/cache/triton}"
DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$PWD/cache/deep_gemm}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/cache/torchinductor}"

if [[ -z "${CONTAINER_DIR}" ]]; then
    echo "Error: CONTAINER_DIR is not set."
    echo "Set CONTAINER_DIR to the directory containing your Singularity containers." 
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

mkdir -p "${HF_CACHE}" "${VLLM_CACHE}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${DG_JIT_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# GPU ordering: on a full 4-GPU node, put the slowest GPU (usually GPU 0) last —
# this improves tensor-parallel performance on GH200. Only do this when TP_SIZE=4
# (a whole-node allocation). For a PARTIAL-node allocation — e.g. the single-GH200
# `ornith_gh200` preset (TP=1) started with --gpus-per-node=1 — a hardcoded
# "1,2,3,0" would name devices that don't exist and crash init.
#   - If SLURM constrained the allocation it already exported CUDA_VISIBLE_DEVICES
#     to just the granted GPU(s) — honour that (the `:-` default won't fire).
#   - If SLURM handed over the whole node (var unset), default to a single card
#     (GPU 0) so we truly drive ONE GH200, matching the preset's intent.
# Either way CUDA_VISIBLE_DEVICES ends up defined and non-empty, so the later
# `set -u` reference and the container --env forward never trip / hide all GPUs.
if [[ "${TP_SIZE}" == "4" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,0}"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# Multi-node engine-as-actor (--data-parallel-backend=ray) device-assignment fix.
# On vLLM main >= ~251f7e4 the EngineCoreActor computes physical GPU ids for the
# WHOLE world (TP×PP, e.g. 8 for TP=4+PP=2) by INDEXING into CUDA_VISIBLE_DEVICES
# — but that env is only the node-local 4 GPUs ("0,1,2,3"), so device id 4 blows
# past it: `Exception: Error computing device indices ... local range: [0, 8)
# base value: "0,1,2,3"` (get_physical_gpu_ids_for_local_dp_rank). The fix is to
# leave CUDA_VISIBLE_DEVICES *unset IN THE CONTAINER*, so device_id_to_physical
# takes its no-env branch and returns the RAW id (interface.py) and Ray places the
# 8 workers across the nodes itself. NB unset != empty: empty ("") means ZERO
# visible GPUs, which breaks `ray start`'s GPU registration — so we OMIT the CVD
# --env from SING_CMD (below) rather than forwarding an empty value. NOSET stops
# Ray stamping a sticky per-actor CVD that would re-trigger the index. Single-node
# is untouched (keeps the 1,2,3,0 NVLink reorder above). CLEAR_CVD_MULTINODE=0 to opt out.
if [[ "${NUM_NODES}" -gt 1 && "${CLEAR_CVD_MULTINODE:-1}" == "1" ]]; then
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
    MULTINODE_UNSET_CVD=1
fi

# NCCL optimizations for GH200 NVLink
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"        # Use NVLink for P2P
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}" # GPU Direct RDMA level
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"         # Enable InfiniBand if available

# Memory optimizations. Override-respecting so a KV-offload run can drop it (below).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# KV-offload experiment (queue item (b), plans/proposed/glm52_kv_tiering.md). vLLM
# HARD-REJECTS any KV connector (--kv-offloading-size / --kv-transfer-config) when
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — its CUDA VMM allocator can remap
# KV pages, invalidating pinned/registered KV memory (ValueError at config validation,
# before load). Set KV_OFFLOAD_EXPERIMENT=1 to drop expandable_segments so the native
# CPU KV-offload path can start. TRADE-OFF: expandable_segments is our anti-
# fragmentation guard on the tight-HBM offload path — watch for fragmentation OOM; the
# alternative (keep it + enable the cumem allocator) is heavier. Forwarded by olivia.sh.
if [[ "${KV_OFFLOAD_EXPERIMENT:-0}" == "1" || -n "${KV_OFFLOAD_GB:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=""
    echo "[INFO] KV offload requested: dropped PYTORCH_CUDA_ALLOC_CONF=expandable_segments (required for KV connectors; watch for fragmentation OOM)"
fi

# Logging configuration
# Set VERBOSE=1 for detailed vLLM + Ray + NCCL logging. Useful when debugging
# startup hangs: `VERBOSE=1` flips vLLM to DEBUG, disables Ray log-dedup (so
# repeated errors are visible), raises Ray's C++ backend log level, and turns
# on NCCL debug output. Every variable can also be overridden individually.
VERBOSE="${VERBOSE:-0}"
if [[ "${VERBOSE}" == "1" ]]; then
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
    RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-debug}"
    RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
    NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
else
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
    RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-info}"
    RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
    NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi
# Enable vLLM's internal logging configuration
VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"

# Cache quantized weights (MARLIN, AWQ, etc.) to speed up subsequent loads
VLLM_CACHE_QUANTIZED_WEIGHTS="${VLLM_CACHE_QUANTIZED_WEIGHTS:-1}"

# vLLM attention backend (now set via CLI arg instead of env var).
# Left empty here so we can tell an explicit override from our default: GLM-5.1
# uses DeepSeek-style MLA (multi-head latent attention) and FLASH_ATTN rejects
# it at startup with `['head_size not supported', 'MLA not supported', ...]`.
# The final value is chosen below after model detection — TRITON_MLA for
# GLM-5.1, FLASH_ATTN for everyone else. User-set values pass through.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"

# AWQ-specific optimizations (recommended by QuantTrio for GLM-4.7-AWQ / GLM-5.1-AWQ)
# These help with AWQ-quantized MoE models on Hopper GPUs.
# Note: GLM-4.7-AWQ uses FLASHINFER_MOE_FP8=1; GLM-5.1-AWQ uses FLASHINFER_MOE_FP16=1
# instead. We adjust these below after detecting GLM-5.1.
# VLLM_USE_DEEP_GEMM's default is model-aware (GLM-5.2 block-FP8 wants it on),
# resolved after model detection below. Record whether the user pinned it first
# so an explicit VLLM_USE_DEEP_GEMM=0 is never silently overridden.
if [[ -n "${VLLM_USE_DEEP_GEMM+x}" ]]; then VLLM_USE_DEEP_GEMM_EXPLICIT=1; else VLLM_USE_DEEP_GEMM_EXPLICIT=0; fi
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-1}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Detect GLM-4.7 model
IS_GLM47=0
if [[ "${MODEL}" == *"GLM-4.7"* ]] || [[ "${MODEL}" == *"glm-4.7"* ]]; then
    IS_GLM47=1
fi

# Detect GLM-5.1 model
IS_GLM51=0
if [[ "${MODEL}" == *"GLM-5.1"* ]] || [[ "${MODEL}" == *"glm-5.1"* ]]; then
    IS_GLM51=1
fi

# Detect GLM-5.2 model (same GlmMoeDsaForCausalLM arch as 5.1, plus a new
# skip-topk DSA indexer — needs vLLM main + PR#45895). Default quant is block-FP8.
IS_GLM52=0
if [[ "${MODEL}" == *"GLM-5.2"* ]] || [[ "${MODEL}" == *"glm-5.2"* ]]; then
    IS_GLM52=1
fi

# Detect GLM-5.3. It is NOT a new architecture and NOT a new pretrain: Z.ai
# re-post-trained GLM-5.2's *same 744B base* ("every reported gain comes from
# scaled post-training"), so the checkpoint is byte-compatible in every way that
# matters to a serving stack — same GlmMoeDsaForCausalLM, same ~40B active, same
# skip-topk DSA indexer, same 1M native context, same block-FP8 quant.
#
# It therefore wants the ENTIRE GLM-5.2 runtime profile: eager CUDAGraph (capture
# IMAs on this NGC stack), the DeepGEMM block-FP8 path, RayExecutorV2, the PP=3
# layer partition, no bf16 KV override. Rather than duplicate ~5 blocks, we light
# IS_GLM52 as the "5.2-family profile" flag and keep IS_GLM53 for reporting and
# for any future 5.3-only divergence. If Z.ai ever ships a 5.3 that ISN'T the 5.2
# base, split them here first.
IS_GLM53=0
if [[ "${MODEL}" == *"GLM-5.3"* ]] || [[ "${MODEL}" == *"glm-5.3"* ]]; then
    IS_GLM53=1
    IS_GLM52=1
fi

# GLM-5.x family: everything 5.1, 5.2 and 5.3 share — sparse-MLA attention
# backend, --trust-remote-code, and the generous 128K default context. (5.1-only
# and 5.2-family-only specifics, e.g. the cyankiwi chat-template fix or the FP8
# DeepGEMM path, stay keyed on IS_GLM51 / IS_GLM52 individually.)
IS_GLM5=0
if [[ "${IS_GLM51}" == "1" || "${IS_GLM52}" == "1" ]]; then
    IS_GLM5=1
fi

# Detect Kimi K2.x model (Moonshot). 1T-param MoE (32B active), MLA attention,
# multimodal (MoonViT). Architecture KimiK25ForConditionalGeneration ships as
# custom_code, so it needs --trust-remote-code. Uses the kimi_k2 parser family,
# not the GLM parsers, so it's tracked separately from IS_GLM_MOE below.
IS_KIMI=0
if [[ "${MODEL}" == *"Kimi-K2"* ]] || [[ "${MODEL}" == *"kimi-k2"* ]]; then
    IS_KIMI=1
fi

# Detect Laguna (Poolside). 225B / 23B-active MoE coding model
# (LagunaForCausalLM), natively supported in vLLM >=0.21.0. Ordinary dense full
# attention + GQA (NOT MLA), so it keeps the FLASH_ATTN default and lets
# CUDAGraph capture run — no eager override like Kimi/glm52. Uses the poolside_v1
# tool + reasoning parser family.
IS_LAGUNA=0
if [[ "${MODEL}" == *"Laguna"* ]] || [[ "${MODEL}" == *"laguna"* ]]; then
    IS_LAGUNA=1
fi

# Detect Ornith 1.0 (Deep Reinforce). Agentic-coding MoE post-trained on Qwen 3.5
# (arch Qwen3_5MoeForConditionalGeneration / model_type qwen3_5_moe).
# GROUND TRUTH from the FP8 checkpoint (verified on-cluster 2026-07-16, NOT what
# the model card implies):
#   - HYBRID attention: each layer has BOTH a Gated-DeltaNet linear_attn mixer
#     (conv1d + in_proj_a/b/qkv/z) AND a full self_attn — the Qwen3-Next/3.5
#     lineage, NOT plain GQA. So we must let vLLM AUTO-SELECT the backend (the
#     linear-attn layers need a mamba/GDN kernel + recurrent state, not FLASH_ATTN).
#   - MULTIMODAL: ships a vision tower (model.visual.*, 333 tensors) + video
#     preprocessor. We serve it for TEXT (coding); the vision encoder just loads.
#   - MTP: config declares mtp_num_hidden_layers=1, but the FP8 export SHIPS NO
#     MTP WEIGHTS (layers stop at 39, zero mtp/nextn tensors). So MTP speculative
#     decode does NOT work on the published FP8 checkpoints — it is OFF by default
#     (ENABLE_SPECULATIVE=1 only if you point at a checkpoint that has the head).
# Uses the qwen3_xml/qwen3 parser family. Both `ornith` (2-node 397B) and
# `ornith_gh200` (1× GH200 35B) hit this block; it is shape-agnostic.
IS_ORNITH=0
if [[ "${MODEL}" == *"Ornith"* ]] || [[ "${MODEL}" == *"ornith"* ]]; then
    IS_ORNITH=1
fi

# Detect Qwen3.8 (Alibaba, open weights 2026-08-13/14, Apache 2.0). VERIFIED from
# the published Qwen/Qwen3.8-27B-FP8 config.json, NOT from the model card:
#   - arch `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` — the SAME
#     Qwen 3.5 family as Ornith, but DENSE (Ornith is ...MoeForConditionalGeneration).
#     Registered natively in vLLM v0.27.1 (registry.py:581), so no new build.
#   - 27B dense, 64 layers, hidden 5120, 24 q-heads / 4 KV-heads, 262144 ctx.
#   - HYBRID attention like Ornith: 48 linear-attn layers + 16 full, full every
#     4th. So do NOT force FLASH_ATTN — vLLM must auto-select (mamba/GDN kernels
#     + recurrent state for the linear layers).
#   - MULTIMODAL: 27-layer vision tower (hidden 1152, patch 16). We serve it for
#     TEXT; the encoder just loads.
#   - Quant is DeepSeek-style BLOCK-FP8 ([128,128] e4m3, dynamic) — note this
#     DIFFERS from Ornith, which is compressed-tensors channel/token W8A8. Block
#     FP8 means this one DOES ride the DeepGEMM path (glm52's lane, not Ornith's).
#   - MTP: the head is REALLY THERE. Verified against the downloaded checkpoint's
#     weight_map (2026-08-17): 22 `mtp.*` tensors — mtp.fc.weight,
#     mtp.layers.0.{input_layernorm,mlp.*,...} — out of 1606 total, alongside 333
#     vision tensors and language layers 0-63. This is the OPPOSITE of Ornith,
#     whose FP8 export declared mtp_num_hidden_layers=1 but shipped ZERO mtp
#     tensors, and vLLM v0.27.1 registers Qwen3_5MTP (registry.py:660). So MTP
#     speculative decode should genuinely work here — likely the single biggest
#     decode win available on a 1-GPU dense model.
#     Left OFF by default only because nothing has been SERVED yet; flipping an
#     untested spec-decode config on by default would turn a first-serve smoke
#     test into a confusing failure. `ENABLE_SPECULATIVE=1` is the first thing to
#     try once it serves clean.
IS_QWEN38=0
if [[ "${MODEL}" == *"Qwen3.8"* ]] || [[ "${MODEL}" == *"qwen3.8"* ]]; then
    IS_QWEN38=1
fi

# Detect Borealis (NbAiLab / National Library of Norway) — Norwegian-centric
# instruct family. VERIFIED from NbAiLab/borealis-27b config.json:
#   - arch `Gemma3ForConditionalGeneration`, model_type `gemma3`, 62 layers,
#     131072 max_position_embeddings, torch_dtype bfloat16, NO quantization_config.
#   - Multimodal: SigLIP vision tower (27 layers, 1152 hidden, 896x896 / patch 14).
#     We serve it for TEXT; the encoder loads dormant.
# No parser block: this is an instruct model that emits neither <think> blocks nor
# a tool-call grammar vLLM has a parser for — so no --reasoning-parser, no
# --tool-call-parser, no --enable-auto-tool-choice.
# It DOES need two things from us:
#   1. MAX_MODEL_LEN=131072 — the generic fallback is 32768, which would silently
#      throw away three quarters of the model's native window.
#   2. NO forced attention backend. Gemma 3's text path is ordinary
#      sliding-window+full attention, but the VISION tower makes the config
#      multimodal PrefixLM, and forcing FLASH_ATTN then fails engine init outright
#      (mm_prefix wants FlashAttention v4, which does not resolve for this
#      head_size on Hopper). See the attention-backend block below.
IS_BOREALIS=0
if [[ "${MODEL}" == *"borealis"* ]] || [[ "${MODEL}" == *"Borealis"* ]]; then
    IS_BOREALIS=1
fi

# Detect DeepSeek-V4-Flash. VERIFIED from the published config + the downloaded
# checkpoint (2026-08-18), not the model card:
#   - arch `DeepseekV4ForCausalLM`, model_type `deepseek_v4`, 43 layers, MoE with
#     256 routed + 1 shared expert, 6 active per token. Registered natively in
#     vLLM v0.27.1, so no build is needed.
#   - 1,048,576 context via YaRN (factor 16 over a 65,536 base).
#   - Quant is DeepSeek-style BLOCK-FP8 ([128,128] e4m3 dynamic) → DeepGEMM lane,
#     same as glm52, NOT Ornith's channel/token W8A8.
#   - MLA attention → must NOT force FLASH_ATTN; vLLM auto-selects.
#   - MTP head genuinely SHIPPED: 4705 `mtp.*` tensors of 72317 in the weight_map
#     (mtp.0.hc_attn_base, ...), and vLLM registers DeepSeekV4MTPModel. This is a
#     verified-present head, so MTP is left ON by default here — the opposite call
#     from Ornith, whose FP8 declared MTP and shipped nothing.
IS_DSV4=0
if [[ "${MODEL}" == *"DeepSeek-V4"* ]] || [[ "${MODEL}" == *"deepseek-v4"* ]]; then
    IS_DSV4=1
fi

# Kimi K2.6's fused MLA op (vllm.min_latency_fused_qkv_a_proj) has no fake/meta
# dispatch, so vLLM's torch.compile/CUDAGraph path fails during profile_run on
# this multi-node PP setup ("Multiple dispatch failed ... NotImplemented", from
# kimi_k25.py forward under cuda_graph.py). Default to eager (mode=NONE), like
# the multi-node glm51 path. Override CUDAGRAPH_MODE=PIECEWISE etc. to retry the
# capture once upstream ships a meta impl for the op (decode-speedup TODO).
if [[ "${IS_KIMI}" == "1" && -z "${CUDAGRAPH_MODE}" ]]; then
    CUDAGRAPH_MODE="NONE"
fi

# GLM-5.2: CUDAGraph capture hits an illegal-memory-access during capture_model
# (the same eager-only story as Kimi — verified 2026-06-18, job 1308936 died with
# "CUDA error: an illegal memory access" under <auto-select>). Default to eager
# (mode=NONE) when unset so `./olivia.sh server start glm52` works out of the box;
# override CUDAGRAPH_MODE=PIECEWISE etc. to retry capture. Scoped to the 5.2
# FAMILY (so GLM-5.3 inherits it — same base, same capture behaviour expected);
# glm51 keeps its own capture experiment (see its CUDAGraph TODO). Worth retrying
# on the glm53_v27 container: a newer torch/inductor may not miscompile.
if [[ "${IS_GLM52}" == "1" && -z "${CUDAGRAPH_MODE}" ]]; then
    CUDAGRAPH_MODE="NONE"
fi

# GLM-5.1: default to PIECEWISE CUDAGraph capture (not auto-select's
# FULL_AND_PIECEWISE). VALIDATED 2026-06-20 on the NGC-26.03 rebuild: PIECEWISE
# captures cleanly (51/51, no IMA); with the custom all-reduce auto-disabled
# (DISABLE_CUSTOM_ALL_REDUCE=auto fires since mode!=NONE → graph-safe NCCL) this
# config ELIMINATES the multi-node PP decode wedge (0 fail 1→64) AND gives
# ~22 tok/s/stream vs ~5 eager. (The de-wedge is the NCCL all-reduce; capture is
# the ~4.5× throughput.) See README "## Performance" + the wedge known-issue.
# Override CUDAGRAPH_MODE=NONE for eager (still de-wedged via NCCL, just slower).
if [[ "${IS_GLM51}" == "1" && -z "${CUDAGRAPH_MODE}" ]]; then
    CUDAGRAPH_MODE="PIECEWISE"
fi

# Ornith on a MULTI-NODE (PP>1) shape — the 397B 2-node preset — hits the same
# custom-all-reduce decode wedge as glm51 (that reduction is the intra-node TP=4
# NVLink one, model-independent). Default to PIECEWISE capture when unset: it
# trips DISABLE_CUSTOM_ALL_REDUCE=auto → graph-safe NCCL all-reduce, which
# de-wedges decode, and it amortizes per-step overhead. Ornith's ordinary GQA
# attention captures cleanly (like Laguna), so PIECEWISE should hold; if it IMAs
# on this NGC stack (like Kimi/glm52), fall back to CUDAGRAPH_MODE=NONE (eager)
# + anthropic_proxy.py request serialization. Single-GPU ornith_gh200 (PP=1) is
# untouched here and keeps auto-select capture.
if [[ "${IS_ORNITH}" == "1" && "${PP_SIZE}" -gt 1 && -z "${CUDAGRAPH_MODE}" ]]; then
    CUDAGRAPH_MODE="PIECEWISE"
fi

# Any GLM MoE model that uses the glm47/glm45 parser family
IS_GLM_MOE=0
if [[ "${IS_GLM47}" == "1" || "${IS_GLM5}" == "1" ]]; then
    IS_GLM_MOE=1
fi

# Resolve ENABLE_AUTO_TOOL_CHOICE. GLM MoE models ship with the glm47 tool
# parser wired up, so auto tool choice is safe to default-on and needed by
# any OpenAI tool-using client (Claude Code via anthropic_proxy.py, etc.).
# ``${VAR+x}`` distinguishes "user explicitly set (even to 0)" from "unset".
if [[ -z "${ENABLE_AUTO_TOOL_CHOICE+x}" ]]; then
    if [[ "${IS_GLM_MOE}" == "1" || "${IS_KIMI}" == "1" || "${IS_LAGUNA}" == "1" || "${IS_ORNITH}" == "1" || "${IS_QWEN38}" == "1" || "${IS_DSV4}" == "1" ]]; then
        ENABLE_AUTO_TOOL_CHOICE=1
    else
        ENABLE_AUTO_TOOL_CHOICE=0
    fi
fi

# Resolve MAX_MODEL_LEN. GLM-5.1 has a native 205K context window and ~260 GB
# of KV cache headroom on 8×GH200 AWQ, so a generous 128K default is safe and
# matches typical Claude Code usage (which requests max_tokens=32000).
# Other models keep the conservative 32K default.
if [[ -z "${MAX_MODEL_LEN+x}" ]]; then
    if [[ "${IS_DSV4}" == "1" ]]; then
        # Native window is 1,048,576 (YaRN x16). Default to 131072: the full 1M
        # would consume the entire KV budget for a single sequence. Raise it
        # explicitly for long-context runs (the context ladder does exactly that).
        MAX_MODEL_LEN=131072
    elif [[ "${IS_BOREALIS}" == "1" ]]; then
        # Gemma-3 27B BF16 is ~54 GB of a 96 GB card, and Gemma 3 interleaves
        # sliding-window layers with full-attention ones so the KV cache stays
        # modest — the full native 131072 window fits comfortably. Lower this if
        # you want more concurrent sequences instead of more context.
        MAX_MODEL_LEN=131072
    elif [[ "${IS_QWEN38}" == "1" ]]; then
        # Qwen3.8-27B is 262144 native (1M via YaRN, not enabled here). Dense 27B
        # block-FP8 is only ~30 GB on a 96 GB card, so the full window fits with
        # room to spare — no reason to clip it like the big multi-node models.
        MAX_MODEL_LEN=262144
    elif [[ "${IS_ORNITH}" == "1" ]]; then
        # Ornith serves the whole 256K native window on both shapes:
        #   ornith_gh200 (35B, 1 card)  — ~35 GB weights + ~10 GB KV @256K single-user
        #   ornith       (397B, 2 nodes) — ~400 GB weights across 8×GH200 leaves
        #                 ~280 GB for KV (~18 sequences at full 256K). For higher
        #                 concurrency, lower this or set KV_CACHE_DTYPE=fp8_e4m3.
        MAX_MODEL_LEN=262144
    elif [[ "${IS_GLM5}" == "1" || "${IS_KIMI}" == "1" || "${IS_LAGUNA}" == "1" ]]; then
        # GLM-5.1 ~205K, GLM-5.2/5.3 ~1M, Kimi K2.6 ~256K, Laguna M.1 ~256K native —
        # all ship far larger windows, but 128K is the safe default within budget.
        # On glm52's 3-node FP8 (~18 GB/GPU KV), 128K holds a few concurrent
        # sequences; raise with --kv-cache-dtype fp8 (see KV_CACHE_DTYPE) or
        # fewer seqs.
        MAX_MODEL_LEN=131072
    else
        MAX_MODEL_LEN=32768
    fi
fi

echo "=============================================="
echo "vLLM Server for GH200"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Container dir:    ${CONTAINER_DIR}"
echo "  Model:            ${MODEL}"
echo "  Tensor Parallel:  ${TP_SIZE}"
echo "  GPU Memory:       ${GPU_MEM_UTIL}"
echo "  Max Model Len:    ${MAX_MODEL_LEN}"
echo "  Port:             ${PORT}"
echo "  Log Level:        ${VLLM_LOGGING_LEVEL}"
echo "  CUDAGraph Mode:   ${CUDAGRAPH_MODE:-<auto-select>}"
if [[ "${VERBOSE}" == "1" ]]; then
    echo "  VERBOSE mode:     ON (Ray backend=${RAY_BACKEND_LOG_LEVEL}, dedup=${RAY_DEDUP_LOGS}, NCCL=${NCCL_DEBUG})"
fi
if [[ "${ENABLE_PROXY}" == "1" ]]; then
    echo ""
    echo "Batching Proxy:     ENABLED"
    echo "  Proxy Port:       ${PROXY_PORT}"
    echo "  Batch Tokens:     ${PROXY_BATCH_TOKENS}"
    echo "  Batch Chars:      ${PROXY_BATCH_CHARS}"
    echo "  Batch Delay:      ${PROXY_BATCH_DELAY_MS}ms"
fi
echo ""

# Detect AWQ quantized model
IS_AWQ=0
if [[ "${MODEL,,}" == *"-awq"* ]] || [[ "${MODEL,,}" == *"awq"* ]]; then
    IS_AWQ=1
fi

# GLM-5.1 AWQ: swap the MoE flashinfer kernel variant
# (QuantTrio recipe uses MOE_FP16 for GLM-5-AWQ, not MOE_FP8 like GLM-4.7-AWQ)
# Applies to the whole GLM-5.x AWQ lineage, not just 5.1: cyankiwi publishes both
# GLM-5.1-AWQ-4bit and GLM-5.2-AWQ-INT4, same GlmMoeDsaForCausalLM, same 4-bit
# weight-only MoE. Leaving the global FLASHINFER_MOE_FP8=1 default on for a 4-bit
# checkpoint points vLLM at an FP8 MoE kernel the weights cannot feed — and on
# containers where the build uninstalled a version-skewed flashinfer (the Ornith
# lesson, likely on the v0.27.1/glm53 container too) there is no flashinfer at all.
if [[ "${IS_GLM5}" == "1" && "${IS_AWQ}" == "1" ]]; then
    export VLLM_USE_FLASHINFER_MOE_FP8=0
    export VLLM_USE_FLASHINFER_MOE_FP16=1
fi

# Single-node GLM-5.2 AWQ CANNOT load without weight offload: the ~415 GiB AWQ
# checkpoint is ~104 GB/GPU across 4 GH200, over the 96 GB HBM before any KV. So
# default a conservative per-GPU offload when running single-node AWQ GLM-5.2 and
# the user hasn't pinned one — otherwise the job OOMs at load. The offload SWEEP
# (plans/proposed/glm52_awq_1n_offload.md) refines this value; 40 is a safe start
# (~64 GB/GPU resident weights, ~32 GB/GPU freed for bf16 KV).
if [[ "${IS_GLM52}" == "1" && "${IS_AWQ}" == "1" && "${NUM_NODES}" -le 1 && -z "${CPU_OFFLOAD_GB:-}" ]]; then
    CPU_OFFLOAD_GB=40
    echo "[INFO] single-node GLM-5.2 AWQ: defaulting CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB} (weights exceed HBM without offload)"
fi

# GLM-5.2 cold load is vLLM-pipeline-bound, not I/O-bound: the default loader
# takes ~56 min for the ~415 GiB AWQ checkpoint (and MTP double-loads to ~84 min).
# The Run:ai Model Streamer parallelizes the safetensors read ~6x — VALIDATED at
# ~9.5 min on glm52_awq_1n (loads, serves, decodes cleanly). It ships as a pip
# package we can't install offline into the container, so it's staged under the
# persistent (projects) HF_HOME and imported via PYTHONPATH (CONTAINER_PYTHONPATH).
# Default it on for GLM-5.2 when that staged package is present next to HF_HOME;
# no-op (fall back to the default loader) when it's absent — e.g. the FP8 preset
# resolves HF_HOME to the work tier where the pkg isn't staged. Fully overridable
# via LOAD_FORMAT / CONTAINER_PYTHONPATH.
if [[ "${IS_GLM52}" == "1" && -z "${LOAD_FORMAT:-}" && -d "${HF_HOME:-}/runai-pkg" ]]; then
    LOAD_FORMAT=runai_streamer
    CONTAINER_PYTHONPATH="${CONTAINER_PYTHONPATH:-${HF_HOME}/runai-pkg}"
    echo "[INFO] GLM-5.2: defaulting LOAD_FORMAT=runai_streamer (staged pkg at ${HF_HOME}/runai-pkg; ~6x faster cold load)"
fi

# GLM-5.2 FP8: block-wise FP8 ([128,128], e4m3) is the DeepSeek-style quant, so
# it runs through DeepGEMM on Hopper rather than the AWQ flashinfer-MoE path.
# Enable DeepGEMM and keep the FP8 (not FP16) MoE kernel. DeepGEMM JIT warmup
# adds a long startup phase; PR#45895's test plan skips it with
# VLLM_DEEP_GEMM_WARMUP=skip — we default to that (override to "" to warm up).
# fp8 KV cache roughly doubles the context/concurrency we can fit at FP8.
if [[ "${IS_GLM52}" == "1" && "${IS_AWQ}" == "0" ]]; then
    if [[ "${VLLM_USE_DEEP_GEMM_EXPLICIT}" == "0" ]]; then
        export VLLM_USE_DEEP_GEMM=1
    fi
    export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
    # KV cache dtype: leave at the model dtype (bf16). GLM-5.2's DSA needs a
    # SPARSE MLA backend; on Hopper (sm_90) the only one is FLASHMLA_SPARSE,
    # which does NOT support fp8 KV cache. (FlashInfer-MLA supports sparse+fp8
    # but only on Blackwell — "compute capability not supported" on GH200, which
    # is why PR#45895's fp8-KV test plan worked for them but not here.) Forcing
    # fp8_e4m3 makes the attention selector reject every backend:
    #   ValueError: No valid attention backend found ... use_sparse=True ...
    #   FLASHMLA_SPARSE: [kv_cache_dtype not supported]
    # Override KV_CACHE_DTYPE explicitly only if you know your backend supports it.
    : # (no fp8 KV default for glm52 on GH200)
fi

# Qwen3.8 block-FP8: [128,128] e4m3 dynamic is the DeepSeek-style blocked quant,
# the same GEMM lane GLM-5.2 uses — so it wants DeepGEMM on Hopper, NOT the
# flashinfer FP8-MoE kernel (and it is dense anyway, so there is no MoE kernel to
# pick). This is the one place Qwen3.8 diverges from Ornith, whose channel/token
# W8A8 never touches DeepGEMM. Skip the JIT warmup for the same reason glm52 does
# — it adds minutes to startup; override VLLM_DEEP_GEMM_WARMUP="" to warm up.
if [[ "${IS_QWEN38}" == "1" && "${IS_AWQ}" == "0" ]]; then
    if [[ "${VLLM_USE_DEEP_GEMM_EXPLICIT}" == "0" ]]; then
        export VLLM_USE_DEEP_GEMM=1
    fi
    export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
    export VLLM_USE_FLASHINFER_MOE_FP8=0
fi

# DeepSeek-V4-Flash block-FP8: same DeepGEMM lane as glm52/qwen38.
if [[ "${IS_DSV4}" == "1" && "${IS_AWQ}" == "0" ]]; then
    if [[ "${VLLM_USE_DEEP_GEMM_EXPLICIT}" == "0" ]]; then
        export VLLM_USE_DEEP_GEMM=1
    fi
    export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
    # fp8 KV cache is MANDATORY, not an optimisation. DeepSeek-V4's attention uses
    # the fp8_ds_mla layout, which refuses anything else — verified on-cluster
    # 2026-08-18 (job 2036910 died at worker start in 2m37s):
    #   AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
    #   got auto
    # Note this is the EXACT OPPOSITE of GLM-5.2 on this hardware, where forcing
    # fp8 KV makes the attention selector reject every backend (FLASHMLA_SPARSE
    # on Hopper does not support it). Same MLA family, inverted requirement — so
    # do not generalise a KV-dtype rule across the DSA/MLA models.
    # Useful side effect: fp8 KV roughly halves per-token cache cost, which is
    # what makes the long-context ladder reach further on this model.
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
fi

# --- GLM-5.2-family multi-node bits that are QUANT-INDEPENDENT --------------
# These two used to live inside the `IS_AWQ == 0` branch above, which was fine
# while block-FP8 was the only GLM-5.2 quant. It no longer is: a complete
# cyankiwi/GLM-5.2-AWQ-INT4 (compressed-tensors pack-quantized 4-bit, ~411 GB)
# now exists, and it is the SAME GlmMoeDsaForCausalLM with the SAME skip-topk
# indexer (index_topk_freq=4, index_skip_topk_offset=3 — verified on the
# on-cluster checkpoint). Both settings below are about DSA + cross-node PP, not
# about the GEMM path, so gating them on FP8 silently mis-served the AWQ.
if [[ "${IS_GLM52}" == "1" ]]; then
    # GLM-5.2 multi-node uses RayExecutorV2 (no Ray Compiled Graph). The legacy
    # executor's Compiled Graph wedges decode at 0 tok/s here; V2 decodes cleanly.
    # V2 with our external Ray bootstrap only works in engine-as-Ray-actor mode
    # (--data-parallel-backend=ray), wired in the multi-node launch below — a
    # plain subprocess EngineCore + V2 fails init (ActorHandleNotFoundError).
    # Default to V2 unless the user pinned VLLM_USE_RAY_V2_EXECUTOR_BACKEND.
    if [[ "${_RAYV2_EXPLICIT}" == "0" ]]; then
        VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
    fi
    # Custom PP layer partition. GLM-5.2's DSA skip-topk layer at a pipeline-stage
    # boundary trips `KeyError: model.layers.<N>.self_attn.attn` in
    # get_attn_backends_for_group. A boundary must land on a FULL-indexer layer;
    # empirically those satisfy max(L-2,0) % index_topk_freq == 0 (freq=4), i.e.
    # L in {0, 2, 6, 10, ... 4k+2}.
    #
    #   PP=3 (78 layers, FP8): default even split → boundary at layer 52, a
    #         skip-topk layer → KeyError. 26/24/28 puts boundaries at 0/26/50,
    #         all full-indexer. VALIDATED on-cluster (glm52 serves with this).
    #   PP=2 (78 layers, the ~411 GB AWQ shape): default even split → boundary at
    #         layer 39, and (39-2)%4 == 1, so it is NOT full-indexer and should
    #         trip the same KeyError. 38/40 puts the boundary at layer 38, which
    #         IS full-indexer ((38-2)%4 == 0), and stays near-balanced.
    #         DERIVED from the PP=3 rule, NOT yet validated on-cluster — if it
    #         still KeyErrors, try the next full-indexer boundary (42/36) or
    #         override VLLM_PP_LAYER_PARTITION directly.
    if [[ "${PP_SIZE}" == "3" ]]; then
        VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-26,24,28}"
    elif [[ "${PP_SIZE}" == "2" ]]; then
        VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-38,40}"
    fi
fi

# Resolve attention backend default. FLASH_ATTN is the right choice for
# everything except MLA models. GLM-5.1's GlmMoeDsaForCausalLM uses DeepSeek
# *sparse* MLA (DSA): FLASH_ATTN rejects it (`['MLA not supported']`) AND
# TRITON_MLA rejects it too (`['sparse not supported']`). A sparse-MLA
# backend like FLASHMLA_SPARSE/FLASHINFER_MLA is needed and typically requires
# extra packages. Leave the backend empty for GLM-5.1 so vLLM auto-selects
# from whatever is installed — if nothing works, the error will tell us what
# to install.
if [[ -z "${VLLM_ATTENTION_BACKEND}" ]]; then
    if [[ "${IS_GLM5}" == "1" || "${IS_KIMI}" == "1" || "${IS_ORNITH}" == "1" || "${IS_QWEN38}" == "1" || "${IS_BOREALIS}" == "1" || "${IS_DSV4}" == "1" ]]; then
        # Don't force FLASH_ATTN — these need vLLM to auto-select a special backend:
        #   Qwen3.8  HYBRID, same as Ornith (48 linear-attn + 16 full layers).
        #   Borealis (Gemma 3) MULTIMODAL PrefixLM. Forcing FLASH_ATTN here is a
        #     HARD FAILURE, verified on-cluster 2026-08-18 (job 2032118 died at
        #     engine init in 110 s):
        #       ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not
        #       valid for this configuration. Reason: ['"'"'mm_prefix (PrefixLM
        #       bidirectional attention) requires FlashAttention v4, which does not
        #       resolve for this head_size'"'"']
        #     Gemma 3 gives the image prefix BIDIRECTIONAL attention, and vLLM only
        #     serves that through FA4 — an SM100/Blackwell path that does not
        #     resolve for this head size on Hopper. Leaving the backend unset lets
        #     vLLM pick one that does support mm_prefix. (So "Gemma 3 is ordinary
        #     attention, keep FLASH_ATTN" was WRONG: it is ordinary for TEXT, but
        #     the vision tower changes the attention contract even when we only
        #     ever send text.)
        #   GLM-5.x  sparse-MLA (DSA); Kimi K2.6 standard MLA — FLASH_ATTN rejects MLA.
        #   Ornith   HYBRID (Gated-DeltaNet linear_attn + full self_attn) — the
        #            linear-attn layers use a mamba/GDN kernel + recurrent state, not
        #            FLASH_ATTN. vLLM's qwen3_5_moe wires the hybrid KV/state itself.
        VLLM_ATTENTION_BACKEND=""   # auto-select
    else
        VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    fi
fi
echo "  Attention Backend: ${VLLM_ATTENTION_BACKEND:-<auto-select>}"

# Determine if expert parallel should be enabled
# Auto-enable for AWQ-quantized MoE models (GLM-4.7-AWQ, GLM-5.1-AWQ)
USE_EXPERT_PARALLEL=0
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    USE_EXPERT_PARALLEL=1
elif [[ "${ENABLE_EXPERT_PARALLEL}" == "auto" ]]; then
    # Auto-enable for AWQ GLM MoE models, and for Kimi K2.x (large MoE — benefits
    # from expert parallel regardless of the quant checkpoint's naming).
    if [[ ( "${IS_AWQ}" == "1" && "${IS_GLM_MOE}" == "1" ) || "${IS_KIMI}" == "1" ]]; then
        USE_EXPERT_PARALLEL=1
        echo "[INFO] Auto-enabling expert parallel for MoE model"
    fi
fi

# Resolve DISABLE_CUSTOM_ALL_REDUCE: auto = on whenever capture is on.
# Capture is on whenever CUDAGRAPH_MODE is unset (auto-select picks a non-NONE
# mode) or set to anything other than NONE.
USE_DISABLE_CUSTOM_AR=0
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    USE_DISABLE_CUSTOM_AR=1
elif [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "auto" ]]; then
    if [[ "${CUDAGRAPH_MODE}" != "NONE" ]]; then
        USE_DISABLE_CUSTOM_AR=1
        echo "[INFO] Auto-disabling vLLM custom all-reduce: incompatible with CUDAGraph capture"
    fi
fi

# Determine if speculative decoding should be enabled
USE_SPECULATIVE=0
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    USE_SPECULATIVE=1
elif [[ "${ENABLE_SPECULATIVE}" == "auto" ]]; then
    # Auto-enable MTP for GLM-4.7/GLM-5.1 (improves throughput significantly).
    # EXCLUDE GLM-5.2: its MTP draft (DeepSeekMTPModel) is loaded as a SECOND full
    # 83-shard checkpoint pass — doubling an already ~53-min cold load — and MTP on
    # the new GLM-5.2 DSA arch is unproven here (2-node PP>1 disables it anyway; the
    # PP=1 single-node offload path auto-enabled it unintentionally, observed
    # 2026-07-01 job 1424649). Make it opt-in for GLM-5.2 via ENABLE_SPECULATIVE=1.
    if [[ "${IS_GLM_MOE}" == "1" && "${IS_GLM52}" != "1" ]]; then
        USE_SPECULATIVE=1
        echo "[INFO] Auto-enabling MTP speculative decoding for GLM MoE model"
    elif [[ "${IS_GLM52}" == "1" && "${MODEL}" == *MTP* ]]; then
        # GLM-5.2 MTP-GRAFTED checkpoint (dnhkng graft: cyankiwi AWQ body + the FP8
        # layer-78 MTP head from GLM-5.2-FP8). The plain cyankiwi AWQ checkpoint
        # DROPPED layer-78 during quantization → MTP init dies (deepseek_mtp.py:480
        # ValueError, observed job 1424649). Only enable MTP when the checkpoint
        # path signals the graft ("MTP"), and note: the container's vLLM must carry
        # the awq+fp8-mtp quant-config patch (patches/vllm-awq-fp8-mtp-quant-config.patch).
        USE_SPECULATIVE=1
        echo "[INFO] GLM-5.2 MTP-grafted checkpoint: auto-enabling MTP speculative decoding"
    elif [[ "${IS_GLM52}" == "1" ]]; then
        echo "[INFO] GLM-5.2 (non-MTP checkpoint): MTP not auto-enabled — cyankiwi AWQ lacks layer-78 weights. Use an MTP-grafted checkpoint (path contains 'MTP') to enable it."
    fi
    # DeepSeek-V4-Flash: MTP is OPT-IN, despite the head being present.
    #
    # The checkpoint genuinely ships it — 4705 `mtp.*` tensors of 72317 in the
    # weight_map (mtp.0.hc_attn_base, ...) — and vLLM registers DeepSeekV4MTPModel.
    # I auto-enabled it on that basis and it FAILED on-cluster 2026-08-18
    # (job 2036966, 3m20s):
    #     KeyError: 'model.layers.43.mtp_block.main_norm.weight'
    # v0.27.1's loader expects the head at model.layers.<N>.mtp_block.*, but this
    # 0731 checkpoint stores it as mtp.0.* — the weights are there, in a layout
    # this vLLM cannot map. So TENSOR PRESENCE IS NECESSARY BUT NOT SUFFICIENT:
    # Ornith taught that a config claim needs checking against the weight_map, and
    # this teaches that the weight_map needs checking against the loader's naming.
    # Re-test with ENABLE_SPECULATIVE=1 after a vLLM bump; if the KeyError persists
    # the layouts have genuinely diverged and it needs an upstream fix.
    if [[ "${IS_DSV4}" == "1" && "${ENABLE_SPECULATIVE}" == "auto" ]]; then
        echo "[INFO] DeepSeek-V4: MTP OFF (head present as mtp.0.* but v0.27.1 expects model.layers.N.mtp_block.*). ENABLE_SPECULATIVE=1 to retry."
    fi
    # Ornith is intentionally NOT auto-on: its config declares mtp_num_hidden_layers=1
    # but the published FP8 checkpoints ship NO MTP weights (verified on-cluster),
    # so --speculative-config mtp would fail to load a draft. Opt in with
    # ENABLE_SPECULATIVE=1 only against a checkpoint that actually has the head.
    if [[ "${IS_ORNITH}" == "1" && "${ENABLE_SPECULATIVE}" == "auto" ]]; then
        echo "[INFO] Ornith: MTP speculative decode OFF (FP8 checkpoint ships no MTP weights). ENABLE_SPECULATIVE=1 to force."
    fi
fi

# vLLM v0.19.x limitation: the MTP draft model (DeepSeekMTPModel) does not
# implement the SupportsPP interface. Combining --speculative-config with
# --pipeline-parallel-size > 1 raises:
#   NotImplementedError: Pipeline parallelism is not supported for this model.
# When PP > 1 we disable MTP. The throughput loss (~15-25%) is much smaller
# than the alternative (TP=N across the cross-node fabric instead of PP, which
# bottlenecks on all-reduce every layer). On vLLM main, MTP's SupportsPP may
# now be implemented (PR#45895 reworked GLM-5.2 MTP) — set ALLOW_MTP_PP=1 to
# try MTP+PP and measure (e.g. for glm52's slow single-stream decode).
if [[ "${USE_SPECULATIVE}" == "1" && "${PP_SIZE}" -gt 1 && "${ALLOW_MTP_PP:-0}" != "1" ]]; then
    echo "[WARN] Disabling MTP speculative decoding: incompatible with pipeline parallelism (PP_SIZE=${PP_SIZE}) on v0.19.x. Set ALLOW_MTP_PP=1 to try it on vLLM main."
    USE_SPECULATIVE=0
elif [[ "${USE_SPECULATIVE}" == "1" && "${PP_SIZE}" -gt 1 ]]; then
    echo "[INFO] ALLOW_MTP_PP=1: keeping MTP enabled with PP_SIZE=${PP_SIZE} (testing main's SupportsPP)"
fi

echo "Speculative Decoding:"
if [[ "${USE_SPECULATIVE}" == "1" ]]; then
    if [[ "${IS_GLM_MOE}" == "1" ]]; then
        echo "  Enabled:          yes"
        echo "  Method:           MTP (Multi-Token Prediction)"
        echo "  Spec Tokens:      ${MTP_SPECULATIVE_TOKENS}"
    elif [[ "${IS_ORNITH}" == "1" ]]; then
        echo "  Enabled:          yes"
        echo "  Method:           ${ORNITH_MTP_METHOD} (native qwen3_5_moe MTP)"
        echo "  Spec Tokens:      ${ORNITH_MTP_SPECULATIVE_TOKENS}"
    else
        echo "  Enabled:          yes"
        echo "  Method:           ngram"
        echo "  Spec Tokens:      ${NUM_SPECULATIVE_TOKENS}"
        echo "  Lookup Max:       ${PROMPT_LOOKUP_MAX}"
    fi
else
    echo "  Enabled:          no"
fi

if [[ "${IS_GLM_MOE}" == "1" ]]; then
    echo ""
    if [[ "${IS_GLM53}" == "1" ]]; then
        echo "GLM-5.3 Settings (GLM-5.2-family runtime profile):"
    elif [[ "${IS_GLM52}" == "1" ]]; then
        echo "GLM-5.2 Settings:"
    elif [[ "${IS_GLM51}" == "1" ]]; then
        echo "GLM-5.1 Settings:"
    else
        echo "GLM-4.7 Settings:"
    fi
    echo "  Tool Parser:      ${GLM_TOOL_PARSER}"
    echo "  Reasoning Parser: ${GLM_REASONING_PARSER}"
    echo "  Auto Tool Choice: ${ENABLE_AUTO_TOOL_CHOICE}"
    if [[ "${IS_AWQ}" == "1" ]]; then
        echo "  Quantization:     AWQ (Hopper-compatible)"
        echo "  Expert Parallel:  ${USE_EXPERT_PARALLEL}"
    elif [[ "${IS_GLM52}" == "1" ]]; then
        echo "  Quantization:     block-FP8 (DeepGEMM=${VLLM_USE_DEEP_GEMM}, warmup=${VLLM_DEEP_GEMM_WARMUP:-<default>})"
        echo "  KV cache dtype:   ${KV_CACHE_DTYPE:-<default>}"
    fi
fi

if [[ "${IS_KIMI}" == "1" ]]; then
    echo ""
    echo "Kimi K2.6 Settings:"
    echo "  Tool Parser:      ${KIMI_TOOL_PARSER}"
    echo "  Reasoning Parser: ${KIMI_REASONING_PARSER:-<none>}"
    echo "  Auto Tool Choice: ${ENABLE_AUTO_TOOL_CHOICE}"
    echo "  Multimodal:       MoonViT (mm-encoder-tp-mode=data)"
    echo "  Expert Parallel:  ${USE_EXPERT_PARALLEL}"
fi

if [[ "${IS_LAGUNA}" == "1" ]]; then
    echo ""
    echo "Laguna M.1 Settings:"
    echo "  Tool Parser:      ${LAGUNA_TOOL_PARSER:-<none>}"
    echo "  Reasoning Parser: ${LAGUNA_REASONING_PARSER:-<none>}"
    echo "  Auto Tool Choice: ${ENABLE_AUTO_TOOL_CHOICE}"
    echo "  Thinking Mode:    ${LAGUNA_ENABLE_THINKING}"
fi

if [[ "${IS_ORNITH}" == "1" ]]; then
    echo ""
    echo "Ornith 1.0 Settings:"
    echo "  Tool Parser:      ${ORNITH_TOOL_PARSER:-<none>}"
    echo "  Reasoning Parser: ${ORNITH_REASONING_PARSER:-<none>}"
    echo "  Auto Tool Choice: ${ENABLE_AUTO_TOOL_CHOICE}"
    echo "  Attention:        hybrid (Gated-DeltaNet linear + full self-attn), backend ${VLLM_ATTENTION_BACKEND:-<auto-select>}"
    echo "  Multimodal:       vision tower present (served for text)"
    if [[ "${USE_SPECULATIVE}" == "1" ]]; then
        echo "  MTP:              ${ORNITH_MTP_SPECULATIVE_TOKENS} tokens (forced via ENABLE_SPECULATIVE=1)"
    else
        echo "  MTP:              off (FP8 checkpoint ships no MTP weights)"
    fi
    if [[ "${TP_SIZE}" == "1" ]]; then
        echo "  Shape:            single GH200 card (TP=1) — max single-user throughput"
    elif [[ "${NUM_NODES}" -gt 1 ]]; then
        echo "  Shape:            ${NUM_NODES}-node 397B flagship (TP=${TP_SIZE} + PP=${PP_SIZE})"
    else
        echo "  Shape:            single node (TP=${TP_SIZE})"
    fi
fi

if [[ "${NUM_NODES}" -gt 1 ]]; then
    echo ""
    echo "Multi-Node Distributed Inference:"
    echo "  Num Nodes:        ${NUM_NODES}"
    echo "  Tensor Parallel:  ${TP_SIZE}  (intra-node, NVLink)"
    echo "  Pipeline Parallel: ${PP_SIZE}  (inter-node, Slingshot)"
    echo "  Total GPUs:       $((TP_SIZE * PP_SIZE))"
    echo "  Ray step timeout: ${RAY_CGRAPH_GET_TIMEOUT}s (RAY_CGRAPH_get_timeout)"
    echo "  Ray executor V2:  ${VLLM_USE_RAY_V2_EXECUTOR_BACKEND} (1=RayExecutorV2, 0=legacy RayDistributedExecutor)"
fi
echo ""
echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NCCL_P2P_LEVEL:       ${NCCL_P2P_LEVEL}"
echo ""
echo "Cache directories:"
echo "  HuggingFace: ${HF_CACHE}"
echo "  vLLM:        ${VLLM_CACHE}"
echo ""

# -----------------------------------------------------------------------------
# Container discovery and resolution
# -----------------------------------------------------------------------------

# List available containers if none specified
if [[ -z "${CONTAINER}" ]]; then
    echo "No container specified. Available containers in ${CONTAINER_DIR}:"
    echo ""
    FOUND_CONTAINERS=0
    if [[ -d "${CONTAINER_DIR}" ]]; then
        for container in "${CONTAINER_DIR}"/vllm-*-sandbox "${CONTAINER_DIR}"/vllm-*.sif; do
            if [[ -e "$container" ]]; then
                FOUND_CONTAINERS=1
                cname=$(basename "$container")
                if [[ -d "$container" ]]; then
                    echo "  ${cname} (sandbox)"
                else
                    echo "  ${cname} (sif)"
                fi
            fi
        done
    fi
    if [[ "${FOUND_CONTAINERS}" == "0" ]]; then
        echo "  (none found)"
    fi
    echo ""
    echo "Set CONTAINER=<name> to use a specific container."
    echo "Example: CONTAINER=vllm-glm47-1-sandbox ./run_vllm_server.sh"
    exit 1
fi

# Resolve container path
CONTAINER_PATH=""
if [[ -d "${CONTAINER}" ]]; then
    # Full path to sandbox directory
    CONTAINER_PATH="${CONTAINER}"
    CONTAINER_TYPE="sandbox"
elif [[ -f "${CONTAINER}" ]]; then
    # Full path to SIF file
    CONTAINER_PATH="${CONTAINER}"
    CONTAINER_TYPE="sif"
elif [[ -d "${CONTAINER_DIR}/${CONTAINER}" ]]; then
    # Name only - look in shared directory (sandbox)
    CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER}"
    CONTAINER_TYPE="sandbox"
elif [[ -f "${CONTAINER_DIR}/${CONTAINER}" ]]; then
    # Name only - look in shared directory (sif)
    CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER}"
    CONTAINER_TYPE="sif"
elif [[ -f "${CONTAINER_DIR}/${CONTAINER}.sif" ]]; then
    # Name only without .sif extension
    CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER}.sif"
    CONTAINER_TYPE="sif"
else
    echo "Error: Container not found: ${CONTAINER}"
    echo "Searched in:"
    echo "  - ${CONTAINER}"
    echo "  - ${CONTAINER_DIR}/${CONTAINER}"
    echo "  - ${CONTAINER_DIR}/${CONTAINER}.sif"
    echo ""
    echo "Run ./build_vllm_gh200.sh first, or set CONTAINER to a valid path."
    exit 1
fi

echo "Container:      ${CONTAINER}"
echo "Container path: ${CONTAINER_PATH}"
echo "Container type: ${CONTAINER_TYPE}"

# Check HuggingFace token
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo ""
    echo "Warning: No HuggingFace token found."
    echo "Set HF_TOKEN for gated models like Devstral."
    echo ""
fi

# -----------------------------------------------------------------------------
# NVFP4 Model Patch (for models like Salyut1/GLM-4.7-NVFP4)
# -----------------------------------------------------------------------------
# Some NVFP4 quantized models are missing k_scale/v_scale parameters.
# This patch adds a check to skip these if missing.
#
# WARNING: NVFP4 requires Blackwell GPUs (B100/B200) for native FP4 compute.
# GH200/Hopper GPUs do NOT support NVFP4 - use AWQ quantization instead.

if [[ "${MODEL,,}" == *"nvfp4"* ]] || [[ "${MODEL,,}" == *"nv-fp4"* ]]; then
    echo ""
    echo "=============================================="
    echo "WARNING: NVFP4 model detected!"
    echo "=============================================="
    echo ""
    echo "NVFP4 quantization requires Blackwell GPUs (B100/B200) for native"
    echo "FP4 compute. GH200/Hopper GPUs do NOT have FP4 tensor cores."
    echo ""
    echo "The model will FAIL with: 'No compiled nvfp4 quantization kernel'"
    echo ""
    echo "Recommended alternatives for GH200:"
    echo "  - QuantTrio/GLM-4.7-AWQ  (AWQ 4-bit, ~181GB, works on Hopper)"
    echo "  - zai-org/GLM-4.7-FP8    (FP8, ~358GB, needs reduced context)"
    echo ""
    echo "Continuing anyway in case you're on Blackwell hardware..."
    echo "=============================================="
    echo ""
    echo "[NVFP4] Checking and applying vLLM patch if needed..."

    if [[ "${CONTAINER_TYPE}" == "sif" ]]; then
        echo ""
        echo "ERROR: Cannot patch a read-only SIF container."
        echo "Please use a sandbox container or rebuild with the patch."
        exit 1
    fi

    # Apply/verify patch inside the container
    # This script checks if patch exists in the RIGHT place and applies if not
    singularity exec --fakeroot --writable "${CONTAINER_PATH}" python3 -c "
import os, re, sys

VLLM_GLM4_PATH = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py'

if not os.path.exists(VLLM_GLM4_PATH):
    print(f'ERROR: File not found: {VLLM_GLM4_PATH}')
    sys.exit(1)

with open(VLLM_GLM4_PATH, 'r') as f:
    content = f.read()
    lines = content.split('\n')

# Check if patch is correctly applied (the continue line should be RIGHT BEFORE param = params_dict[name])
patch_pattern = r\"if \('k_scale' in name or 'v_scale' in name\) and name not in params_dict: continue\"
target_pattern = r'param = params_dict\[name\]'

# Find if patch already exists in correct location
import re as re_module
patch_matches = list(re_module.finditer(patch_pattern, content))
target_matches = list(re_module.finditer(target_pattern, content))

patch_needed = True
if patch_matches and target_matches:
    # Check if any patch is right before a target
    for pm in patch_matches:
        patch_end = pm.end()
        # Find what comes after the patch (skip whitespace/newlines)
        rest = content[patch_end:].lstrip()
        if rest.startswith('param = params_dict[name]'):
            patch_needed = False
            break

if not patch_needed:
    print('[NVFP4] Patch already correctly applied.')
    sys.exit(0)

print('[NVFP4] Applying patch...')

# Read and patch
with open(VLLM_GLM4_PATH, 'r') as f:
    lines = f.readlines()

target = 'param = params_dict[name]'
new_lines = []
patched_count = 0

for i, line in enumerate(lines):
    # Check if this line has the target and previous line is NOT already the patch
    if target in line:
        # Check if previous line already has the patch
        prev_line = new_lines[-1] if new_lines else ''
        if 'k_scale' not in prev_line and 'v_scale' not in prev_line:
            ws = re_module.match(r'^(\s*)', line).group(1)
            patch_line = f\"{ws}if ('k_scale' in name or 'v_scale' in name) and name not in params_dict: continue\n\"
            new_lines.append(patch_line)
            patched_count += 1
    new_lines.append(line)

if patched_count > 0:
    with open(VLLM_GLM4_PATH, 'w') as f:
        f.writelines(new_lines)
    print(f'[NVFP4] Patch applied successfully ({patched_count} location(s))')
else:
    print('[NVFP4] WARNING: Could not find target line to patch')
    sys.exit(1)
"
    PATCH_RESULT=$?
    if [[ ${PATCH_RESULT} -ne 0 ]]; then
        echo ""
        echo "ERROR: Failed to apply NVFP4 patch."
        echo "You may need to apply it manually or use a different model."
        exit 1
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Build vLLM command
# -----------------------------------------------------------------------------

# Use python module invocation (more reliable than vllm CLI which may not be in PATH)
VLLM_ARGS=(
    "vllm" "serve" "${MODEL}"
    "--tensor-parallel-size" "${TP_SIZE}"
    "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--host" "${HOST}"
    "--port" "${PORT}"
)

# CPU/Grace weight offload (per GPU). On GH200 the offloaded weights live in the
# Hopper GPU's coherent Grace LPDDR5X and stream back over C2C (~450 GB/s) each
# forward. This is the lever that lets a model whose per-GPU shard exceeds the
# 96 GB HBM (e.g. single-node GLM-5.2-AWQ, ~104 GB/GPU) fit — trading some decode
# latency for HBM freed up for KV. Only emitted when set > 0.
if [[ -n "${CPU_OFFLOAD_GB:-}" && "${CPU_OFFLOAD_GB}" != "0" ]]; then
    VLLM_ARGS+=("--cpu-offload-gb" "${CPU_OFFLOAD_GB}")
fi

# Native CPU KV-cache offload (queue item (b), plans/proposed/glm52_kv_tiering.md):
# bulk KV blocks are tiered to coherent Grace LPDDR and reloaded on prefix reuse.
# KV_OFFLOAD_GB is the TOTAL CPU tier size in GiB summed across TP ranks (vLLM's
# --kv-offloading-size; backend defaults to 'native' → OffloadingConnector +
# CPUOffloadingSpec). Requires expandable_segments off — handled above (setting
# KV_OFFLOAD_GB drops it automatically). Distinct from CPU_OFFLOAD_GB (that offloads
# *weights*); both can be set together. Validated 2026-07-03 (job 1473586: store +
# reload proven on GLM-5.2 DSA). Note the connector reserves ~25K tok of HBM.
if [[ -n "${KV_OFFLOAD_GB:-}" && "${KV_OFFLOAD_GB}" != "0" ]]; then
    VLLM_ARGS+=("--kv-offloading-size" "${KV_OFFLOAD_GB}")
    echo "[INFO] native KV offload: --kv-offloading-size ${KV_OFFLOAD_GB} GiB (CPU tier, total across TP ranks)"
fi

# Optional load-format override, e.g. LOAD_FORMAT=runai_streamer for a parallel
# streaming loader (needs runai_model_streamer importable — see CONTAINER_PYTHONPATH).
if [[ -n "${LOAD_FORMAT:-}" ]]; then
    VLLM_ARGS+=("--load-format" "${LOAD_FORMAT}")
fi

# CUDAGraph knob. NONE → disable all compilation (mode=NONE). Anything else →
# keep compilation enabled and only override cudagraph_mode. Unset → no flag,
# vLLM auto-selects.
if [[ "${CAPTURE_EXPERIMENT:-0}" == "1" ]]; then
    # Capture experiment: assemble the full --compilation-config HERE from scalar
    # knobs (forwarding raw JSON through olivia.sh brace-expands the {} and ,).
    # Emits PIECEWISE capture with the NGC inductor combo-kernel autotuning
    # DISABLED — peels the `autotune_to_one_config` IMA. Pair DISABLE_CUSTOM_ALL_REDUCE=1.
    _cap_mode="${CAPTURE_CUDAGRAPH_MODE:-PIECEWISE}"
    _cap_size="${CAPTURE_MAX_SIZE:-64}"
    VLLM_ARGS+=("--compilation-config" "{\"cudagraph_mode\":\"${_cap_mode}\",\"max_cudagraph_capture_size\":${_cap_size},\"inductor_compile_config\":{\"enable_auto_functionalized_v2\":false,\"combo_kernels\":false,\"benchmark_combo_kernel\":false}}")
elif [[ "${CUDAGRAPH_MODE}" == "NONE" ]]; then
    VLLM_ARGS+=("--compilation-config" '{"mode": "NONE"}')
elif [[ -n "${CUDAGRAPH_MODE}" ]]; then
    VLLM_ARGS+=("--compilation-config" "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE}\"}")
fi

# Disable vLLM's custom all-reduce when CUDAGraph capture runs (custom kernel
# raises cudaErrorInvalidValue under graph capture; NCCL all-reduce works fine).
if [[ "${USE_DISABLE_CUSTOM_AR}" == "1" ]]; then
    VLLM_ARGS+=("--disable-custom-all-reduce")
fi

# Only pin the attention backend when we have an explicit choice. For GLM-5.1
# (sparse MLA) we leave this empty so vLLM auto-selects a compatible backend.
if [[ -n "${VLLM_ATTENTION_BACKEND}" ]]; then
    VLLM_ARGS+=("--attention-config.backend" "${VLLM_ATTENTION_BACKEND}")
fi

# Pipeline parallelism (multi-node)
if [[ "${PP_SIZE}" -gt 1 ]]; then
    VLLM_ARGS+=("--pipeline-parallel-size" "${PP_SIZE}")
fi

# Optional: Enable chunked prefill for long contexts
if [[ "${ENABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
fi

# Speculative decoding (method depends on model)
if [[ "${USE_SPECULATIVE}" == "1" ]]; then
    if [[ "${IS_GLM_MOE}" == "1" ]]; then
        # GLM-4.7/GLM-5.1 use MTP (Multi-Token Prediction) speculative decoding
        SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": '${MTP_SPECULATIVE_TOKENS}'}'
    elif [[ "${IS_DSV4}" == "1" ]]; then
        # DeepSeek-V4 native MTP head (verified present in the checkpoint).
        SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": '${DSV4_MTP_SPECULATIVE_TOKENS}'}'
    elif [[ "${IS_ORNITH}" == "1" ]]; then
        # Ornith (qwen3_5_moe) ships a native MTP module — vLLM's generic "mtp"
        # method loads it and drafts from the model's own head, no draft model.
        SPEC_CONFIG='{"method": "'${ORNITH_MTP_METHOD}'", "num_speculative_tokens": '${ORNITH_MTP_SPECULATIVE_TOKENS}'}'
    else
        # Default: ngram speculative decoding
        SPEC_CONFIG='{"method": "ngram", "num_speculative_tokens": '${NUM_SPECULATIVE_TOKENS}', "prompt_lookup_max": '${PROMPT_LOOKUP_MAX}'}'
    fi
    VLLM_ARGS+=("--speculative-config" "${SPEC_CONFIG}")
fi

# GLM-4.7 / GLM-5.1 shared arguments (same parser family)
if [[ "${IS_GLM_MOE}" == "1" ]]; then
    VLLM_ARGS+=("--tool-call-parser" "${GLM_TOOL_PARSER}")
    VLLM_ARGS+=("--reasoning-parser" "${GLM_REASONING_PARSER}")
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    fi
    # GLM-5.1 / GLM-5.2 (GlmMoeDsaForCausalLM) require trust-remote-code for
    # their custom attention/indexer config.
    if [[ "${IS_GLM5}" == "1" ]]; then
        VLLM_ARGS+=("--trust-remote-code")
    fi
    # Chat-template override is GLM-5.1-only: it fixes a bug in the *cyankiwi*
    # AWQ fork's template. The GLM-5.2 FP8 repos (zai-org / RedHatAI) ship the
    # correct base template, so no override there.
    if [[ "${IS_GLM51}" == "1" ]]; then
        # Override the shipped chat template. cyankiwi/GLM-5.1-AWQ-4bit's
        # chat_template.jinja has a bug in the role:tool handler — the
        # else-branch expects content items with a .name field (Anthropic's
        # tool_reference convention), but vLLM's OpenAI API auto-converts
        # string content to ``[{type:"text", text:"..."}]`` which has no
        # .name, so tool results render as empty ``<tools></tools>`` and the
        # model never sees any tool output. The base zai-org/GLM-5.1
        # template has a fallback branch using ``visible_text(m.content)``
        # that handles this case correctly.
        #
        # CHAT_TEMPLATE_FILE defaults to the base-template copy shipped in
        # this repo at templates/glm51_chat_template.jinja (deployed to
        # CONTAINER_DIR alongside this script).
        CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${CONTAINER_DIR}/glm51_chat_template.jinja}"
        if [[ -f "${CHAT_TEMPLATE_FILE}" ]]; then
            VLLM_ARGS+=("--chat-template" "${CHAT_TEMPLATE_FILE}")
            echo "[$(date '+%H:%M:%S')] Using GLM-5.1 base chat template: ${CHAT_TEMPLATE_FILE}"
        else
            echo "[$(date '+%H:%M:%S')] WARNING: GLM-5.1 chat template override not found at ${CHAT_TEMPLATE_FILE}"
            echo "                     Tool results will render as empty <tools></tools> — see CLAUDE.md."
        fi
    fi
fi

# Kimi K2.x (Moonshot) arguments
if [[ "${IS_KIMI}" == "1" ]]; then
    VLLM_ARGS+=("--tool-call-parser" "${KIMI_TOOL_PARSER}")
    if [[ -n "${KIMI_REASONING_PARSER}" ]]; then
        VLLM_ARGS+=("--reasoning-parser" "${KIMI_REASONING_PARSER}")
    fi
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    fi
    # KimiK25ForConditionalGeneration ships as custom_code.
    VLLM_ARGS+=("--trust-remote-code")
    # Multimodal (MoonViT vision encoder): replicate the encoder across the TP
    # group instead of sharding it (vLLM K2.6 recipe recommendation).
    VLLM_ARGS+=("--mm-encoder-tp-mode" "data")
fi

# Laguna (Poolside) arguments
if [[ "${IS_LAGUNA}" == "1" ]]; then
    if [[ -n "${LAGUNA_TOOL_PARSER}" ]]; then
        VLLM_ARGS+=("--tool-call-parser" "${LAGUNA_TOOL_PARSER}")
    fi
    if [[ -n "${LAGUNA_REASONING_PARSER}" ]]; then
        VLLM_ARGS+=("--reasoning-parser" "${LAGUNA_REASONING_PARSER}")
    fi
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    fi
    # LagunaForCausalLM is natively integrated in transformers/vLLM, but the repo
    # still ships custom pieces (tokenizer / chat template), so the model card
    # calls for --trust-remote-code.
    VLLM_ARGS+=("--trust-remote-code")
    # Thinking mode: the model card's recommended serve command enables it via
    # the default chat-template kwargs. LAGUNA_ENABLE_THINKING=0 serves instant.
    if [[ "${LAGUNA_ENABLE_THINKING}" == "1" ]]; then
        VLLM_ARGS+=("--default-chat-template-kwargs" '{"enable_thinking": true}')
    fi
fi

# Ornith 1.0 (Deep Reinforce) arguments. Mirrors the model card's vLLM serve
# command: qwen3_xml tool parser + qwen3 reasoning parser + auto-tool-choice +
# trust-remote-code + prefix caching. MTP is wired via --speculative-config above.
if [[ "${IS_ORNITH}" == "1" ]]; then
    if [[ -n "${ORNITH_TOOL_PARSER}" ]]; then
        VLLM_ARGS+=("--tool-call-parser" "${ORNITH_TOOL_PARSER}")
    fi
    if [[ -n "${ORNITH_REASONING_PARSER}" ]]; then
        VLLM_ARGS+=("--reasoning-parser" "${ORNITH_REASONING_PARSER}")
    fi
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    fi
    # qwen3_5_moe is a new arch that ships a custom config/code; the model card's
    # serve command passes --trust-remote-code.
    VLLM_ARGS+=("--trust-remote-code")
    # Prefix caching: the model card recommends it (repetitive agentic-coding
    # prompts share long prefixes). vLLM v1 defaults it on, but pass it explicitly
    # to match the card. ORNITH_ENABLE_PREFIX_CACHING=0 to disable.
    if [[ "${ORNITH_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
        VLLM_ARGS+=("--enable-prefix-caching")
    fi
    # Force the Triton/FLA GDN prefill kernel (avoids the broken flashinfer path).
    # Passed as one array element straight to `vllm serve`, so the JSON is safe.
    if [[ -n "${ORNITH_GDN_PREFILL_BACKEND}" ]]; then
        VLLM_ARGS+=("--additional-config" "{\"gdn_prefill_backend\": \"${ORNITH_GDN_PREFILL_BACKEND}\"}")
    fi
fi

# DeepSeek-V4-Flash serve arguments.
if [[ "${IS_DSV4}" == "1" ]]; then
    [[ -n "${DSV4_TOOL_PARSER}" ]] && VLLM_ARGS+=("--tool-call-parser" "${DSV4_TOOL_PARSER}")
    [[ -n "${DSV4_REASONING_PARSER}" ]] && VLLM_ARGS+=("--reasoning-parser" "${DSV4_REASONING_PARSER}")
    [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]] && VLLM_ARGS+=("--enable-auto-tool-choice")
    [[ -n "${SERVED_MODEL_NAME}" ]] && VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    VLLM_ARGS+=("--trust-remote-code")
fi

# Qwen3.8 27B dense — same Qwen3.5 parser family as Ornith, different shape.
if [[ "${IS_QWEN38}" == "1" ]]; then
    if [[ -n "${QWEN38_TOOL_PARSER}" ]]; then
        VLLM_ARGS+=("--tool-call-parser" "${QWEN38_TOOL_PARSER}")
    fi
    if [[ -n "${QWEN38_REASONING_PARSER}" ]]; then
        VLLM_ARGS+=("--reasoning-parser" "${QWEN38_REASONING_PARSER}")
    fi
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
    fi
    VLLM_ARGS+=("--trust-remote-code")
    if [[ "${QWEN38_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
        VLLM_ARGS+=("--enable-prefix-caching")
    fi
    # Empty by default here (unlike Ornith) — this container's flashinfer works.
    if [[ -n "${QWEN38_GDN_PREFILL_BACKEND}" ]]; then
        VLLM_ARGS+=("--additional-config" "{\"gdn_prefill_backend\": \"${QWEN38_GDN_PREFILL_BACKEND}\"}")
    fi
fi

# Expert parallel for MoE models (required for AWQ-quantized MoE like GLM-4.7-AWQ)
if [[ "${USE_EXPERT_PARALLEL}" == "1" ]]; then
    VLLM_ARGS+=("--enable-expert-parallel")
fi

# Optional: Quantization
if [[ -n "${QUANTIZATION:-}" ]]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION}")
fi

# Optional: KV cache dtype (defaults to fp8_e4m3 for GLM-5.2 FP8 to extend the
# context/concurrency we can fit; unset elsewhere → vLLM uses the model dtype).
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
    VLLM_ARGS+=("--kv-cache-dtype" "${KV_CACHE_DTYPE}")
fi

# Optional: LoRA
if [[ -n "${LORA_MODULES:-}" ]]; then
    VLLM_ARGS+=("--enable-lora" "--lora-modules" "${LORA_MODULES}")
fi

# Optional: extra raw vLLM args (space-separated), appended verbatim. An escape
# hatch for experiments without editing this script — e.g.
# EXTRA_VLLM_ARGS="--data-parallel-backend ray" to run the engine as a Ray actor.
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    # word-split intentionally so "--flag value" becomes two args
    VLLM_ARGS+=(${EXTRA_VLLM_ARGS})
fi

# IMPORTANT: Do NOT use --enforce-eager (this disables CUDA graphs!)
# If you get errors, check that NGC PyTorch is intact

echo "vLLM command:"
echo "  ${VLLM_ARGS[*]}"
echo ""

# -----------------------------------------------------------------------------
# Start server
# -----------------------------------------------------------------------------

echo "Starting vLLM server..."
echo "Access at: http://${HOST}:${PORT}"
if [[ "${ENABLE_PROXY}" == "1" ]]; then
    echo "Proxy at:  http://${HOST}:${PROXY_PORT} (use this for streaming over SSH)"
fi
echo ""
echo "API endpoints:"
echo "  POST /v1/completions      - Text completions"
echo "  POST /v1/chat/completions - Chat completions"
echo "  GET  /health              - Health check"
echo "  GET  /v1/models           - List models"
if [[ "${ENABLE_PROXY}" == "1" ]]; then
    echo ""
    echo "For streaming over SSH tunnels, connect to port ${PROXY_PORT} instead of ${PORT}"
    echo "The proxy batches tokens to reduce network overhead."
fi
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

# Estimate loading time based on model size
estimate_loading_time() {
    local model="$1"
    local estimate=""
    local size_hint=""

    # Detect model size from name patterns
    case "${model,,}" in  # lowercase for matching
        *glm-4.7*|*glm4.7*)
            if [[ "${model,,}" == *awq* ]]; then
                estimate="10-20 minutes"
                size_hint="358B params @ AWQ 4-bit (~181GB weights)"
            elif [[ "${model,,}" == *fp4* ]] || [[ "${model,,}" == *nvfp4* ]] || [[ "${model,,}" == *int4* ]]; then
                estimate="15-25 minutes"
                size_hint="358B params @ 4-bit (~179GB weights)"
            elif [[ "${model,,}" == *fp8* ]]; then
                estimate="25-40 minutes"
                size_hint="358B params @ FP8 (~358GB weights)"
            else
                estimate="30-45 minutes"
                size_hint="358B params @ BF16 (~716GB weights)"
            fi
            ;;
        *405b*|*400b*)
            estimate="20-35 minutes"
            size_hint="~400B params"
            ;;
        *123b*|*120b*|*devstral*)
            estimate="8-15 minutes"
            size_hint="~123B params"
            ;;
        *70b*|*72b*|*65b*)
            estimate="5-10 minutes"
            size_hint="~70B params"
            ;;
        *32b*|*34b*|*33b*)
            estimate="3-6 minutes"
            size_hint="~32B params"
            ;;
        *13b*|*14b*)
            estimate="2-4 minutes"
            size_hint="~13B params"
            ;;
        *7b*|*8b*|*9b*)
            estimate="1-3 minutes"
            size_hint="~7-9B params"
            ;;
        *1b*|*2b*|*3b*)
            estimate="30-60 seconds"
            size_hint="~1-3B params"
            ;;
        *)
            estimate="varies by model size"
            size_hint="unknown size"
            ;;
    esac

    echo "${estimate}|${size_hint}"
}

# Get loading estimate
LOAD_ESTIMATE=$(estimate_loading_time "${MODEL}")
ESTIMATE_TIME="${LOAD_ESTIMATE%%|*}"
ESTIMATE_SIZE="${LOAD_ESTIMATE##*|}"

# Run with singularity
echo ""
echo "=============================================="
echo "Starting vLLM Server"
echo "=============================================="
echo ""
echo "Model: ${MODEL}"
echo "Size:  ${ESTIMATE_SIZE}"
echo ""
echo "Estimated loading time: ${ESTIMATE_TIME}"
echo "(Actual time depends on storage speed and cache status)"
echo ""
echo "[$(date '+%H:%M:%S')] Starting model loading..."
echo ""

# -----------------------------------------------------------------------------
# Start batching proxy (if enabled)
# -----------------------------------------------------------------------------

PROXY_PID=""

cleanup_proxy() {
    if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo ""
        echo "[$(date '+%H:%M:%S')] Stopping batching proxy (PID ${PROXY_PID})..."
        kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true
    fi
}

if [[ "${ENABLE_PROXY}" == "1" ]]; then
    # Find the proxy script (same directory as this script)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROXY_SCRIPT="${SCRIPT_DIR}/vllm_proxy.py"

    if [[ ! -f "${PROXY_SCRIPT}" ]]; then
        echo "WARNING: Proxy script not found at ${PROXY_SCRIPT}"
        echo "         Proxy will not be started. Continuing with vLLM only."
        ENABLE_PROXY=0
    else
        echo "[$(date '+%H:%M:%S')] Starting batching proxy on port ${PROXY_PORT}..."

        # Start proxy in background
        python3 "${PROXY_SCRIPT}" \
            --vllm-host localhost \
            --vllm-port "${PORT}" \
            --proxy-port "${PROXY_PORT}" \
            --batch-tokens "${PROXY_BATCH_TOKENS}" \
            --batch-chars "${PROXY_BATCH_CHARS}" \
            --batch-delay-ms "${PROXY_BATCH_DELAY_MS}" \
            > "${WORKDIR:-$PWD}/logs/proxy_${SLURM_JOB_ID:-$$}.log" 2>&1 &

        PROXY_PID=$!
        echo "[$(date '+%H:%M:%S')] Proxy started (PID ${PROXY_PID})"
        echo "         Log: ${WORKDIR:-$PWD}/logs/proxy_${SLURM_JOB_ID:-$$}.log"
        echo ""

        # Set up cleanup trap
        trap cleanup_proxy EXIT INT TERM
    fi
fi

# CUDA_VISIBLE_DEVICES handling: single-node passes the (reordered) value so the
# container sees the intended GPUs. Multi-node must leave CVD *unset in-container*
# (see the device-assignment note above; unset != empty). Omitting our --env is NOT
# enough — singularity passes the HOST env by default, so SLURM's per-node
# CUDA_VISIBLE_DEVICES="0,1,2,3" leaks in and the EngineCoreActor still indexes [0,8)
# into it. So we STRIP it from singularity's own environment with `env -u` (the whole
# node's 4 GPUs stay visible via the cgroup; unset CVD -> device_id_to_physical
# returns raw ids). `env -u` prefixes the exec so it applies to every SING_CMD use
# (ray start AND the vLLM launch).
if [[ "${MULTINODE_UNSET_CVD:-0}" == "1" ]]; then
    _CVD_ENV=()
    _SING_PREFIX=(env -u CUDA_VISIBLE_DEVICES)
else
    _CVD_ENV=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
    _SING_PREFIX=()
fi

# =============================================================================
# Idle watchdog — cancel this job if the server is up but nobody is using it
# =============================================================================
# THE most expensive failure mode on this cluster is not a crash. A crash dies in
# 2-5 minutes and costs almost nothing. A server that comes up CORRECTLY and then
# receives no requests holds 100% of its allocation at 0% utilisation until
# walltime. Measured over 2026-08-17..19: six failed starts cost ~1.5 GPU-h in
# total, while TWO healthy-but-unattended servers cost 11 GPU-h — roughly 63% of
# all GPU time spent. Causes were mundane: an SSH master dropped mid-benchmark,
# and a benchmark driven by hand from a session that was not attached.
#
# run_proxy.sh has had ROUTER_EMPTY_TIMEOUT auto-spindown for a while — on the CPU
# router, which costs 2 CPUs. The GPU servers, at 4-12x the price, had nothing.
# This closes that gap.
#
# The watchdog only starts counting AFTER /health returns 200, so a long model load
# (Kimi and the 400 GB models take 10+ minutes) is never mistaken for idleness. It
# tracks vLLM's monotonic generation-token counter plus the running-request gauge;
# idle means both "no tokens produced since last check" and "nothing running".
#
#   IDLE_TIMEOUT=<seconds>   default 1200 (20 min). 0 DISABLES — use that for a
#                            server deliberately left up for interactive work.
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1200}"
if [[ "${IDLE_TIMEOUT}" =~ ^[0-9]+$ ]] && [[ "${IDLE_TIMEOUT}" -gt 0 ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    (
        _wd_curl() { curl -s --noproxy '*' -m 5 "$@" 2>/dev/null; }
        # Wait for readiness. Deliberately generous (90 min): a slow load must not
        # be cancelled, and if the server never comes up the JOB dies by itself.
        for _ in $(seq 1 540); do
            [ "$(_wd_curl -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health")" = "200" ] && break
            sleep 10
        done
        [ "$(_wd_curl -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health")" = "200" ] || exit 0
        echo "[watchdog] server ready; will cancel job ${SLURM_JOB_ID} after ${IDLE_TIMEOUT}s with no activity"
        _last_tokens=""; _idle=0
        while sleep 60; do
            _m=$(_wd_curl "http://localhost:${PORT}/metrics")
            [ -z "${_m}" ] && continue   # transient scrape failure is not idleness
            _tok=$(echo "${_m}" | awk '/^vllm:generation_tokens_total/{s+=$2} END{printf "%.0f", s+0}')
            _run=$(echo "${_m}" | awk '/^vllm:num_requests_running/{s+=$2} END{printf "%.0f", s+0}')
            if [ "${_tok}" != "${_last_tokens}" ] || [ "${_run}" != "0" ]; then
                _idle=0; _last_tokens="${_tok}"
            else
                _idle=$((_idle + 60))
                if [ "${_idle}" -ge "${IDLE_TIMEOUT}" ]; then
                    echo "[watchdog] no activity for ${_idle}s — cancelling job ${SLURM_JOB_ID} to stop burning allocation"
                    scancel "${SLURM_JOB_ID}"
                    exit 0
                fi
            fi
        done
    ) &
    echo "[$(date '+%H:%M:%S')] Idle watchdog armed (IDLE_TIMEOUT=${IDLE_TIMEOUT}s; set 0 to disable)"
else
    echo "[$(date '+%H:%M:%S')] Idle watchdog DISABLED (IDLE_TIMEOUT=${IDLE_TIMEOUT:-unset}) — job will hold its allocation until walltime"
fi

# Build shared singularity command array (used by both single-node and multi-node paths)
SING_CMD=(
    "${_SING_PREFIX[@]}"
    singularity exec --nv
    "${_CVD_ENV[@]}"
    --env "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-}"
    --env "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}"
    --env "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
    --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    --env "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
    --env "VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING}"
    --env "RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL}"
    --env "RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS}"
    --env "RAY_CGRAPH_get_timeout=${RAY_CGRAPH_GET_TIMEOUT}"
    --env "VLLM_USE_RAY_V2_EXECUTOR_BACKEND=${VLLM_USE_RAY_V2_EXECUTOR_BACKEND}"
    --env "NCCL_DEBUG=${NCCL_DEBUG}"
    --env "VLLM_CACHE_QUANTIZED_WEIGHTS=${VLLM_CACHE_QUANTIZED_WEIGHTS}"
    --env "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
    --env "VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8}"
    --env "VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16}"
    --env "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    --env "HF_HOME=${HF_CACHE}"
    --env "HF_TOKEN=${HF_TOKEN:-}"
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
    --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    --env "DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR}"
    --env "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
    --env "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
    --bind "${HF_CACHE}:${HF_CACHE}"
    --bind "${VLLM_CACHE_ROOT}:${VLLM_CACHE_ROOT}"
    --bind "${TRITON_CACHE_DIR}:${TRITON_CACHE_DIR}"
    --bind "${DG_JIT_CACHE_DIR}:${DG_JIT_CACHE_DIR}"
    --bind "${TORCHINDUCTOR_CACHE_DIR}:${TORCHINDUCTOR_CACHE_DIR}"
    # Make CONTAINER_DIR readable inside the container so vLLM can load
    # ${CONTAINER_DIR}/glm51_chat_template.jinja (and any other sibling
    # files we deploy alongside run_vllm_server.sh).
    --bind "${CONTAINER_DIR}:${CONTAINER_DIR}"
)

# If MODEL is a LOCAL directory (e.g. the grafted AWQ+FP8-MTP checkpoint at
# /cluster/projects/<proj>/models/...), its path must be visible INSIDE the
# container. The default binds cover HF_HOME + CONTAINER_DIR but not arbitrary
# local model dirs, so without this the path doesn't exist in the job, transformers
# treats it as a HF repo id, and config resolution dies with
# "Repo id must be in the form 'repo_name' or 'namespace/repo_name'". Bind the model
# dir (its internal symlinks resolve into HF_HOME, which is already bound).
if [[ "${MODEL}" == /* && -d "${MODEL}" ]]; then
    SING_CMD+=(--bind "${MODEL}:${MODEL}")
    echo "  Local model bind: ${MODEL}"
fi

# Optional: prepend a PYTHONPATH inside the container (e.g. a staged
# runai_model_streamer package under HF_HOME) so a pip package we can't install
# offline is still importable. The path must live under an already-bound dir.
if [[ -n "${CONTAINER_PYTHONPATH:-}" ]]; then
    SING_CMD+=(--env "PYTHONPATH=${CONTAINER_PYTHONPATH}")
    echo "  Container PYTHONPATH: ${CONTAINER_PYTHONPATH}"
fi

# GLM-5.2 block-FP8 sets VLLM_DEEP_GEMM_WARMUP=skip to avoid the multi-minute
# DeepGEMM JIT warmup at startup. Only forward it when set so other models keep
# vLLM's default warmup behaviour (empty value would read as "disabled").
if [[ -n "${VLLM_DEEP_GEMM_WARMUP:-}" ]]; then
    SING_CMD+=(--env "VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP}")
fi

# Diagnostic: CUDA_LAUNCH_BLOCKING=1 makes CUDA errors report synchronously at
# the faulting kernel launch — an IMA is otherwise async, so the Python
# traceback can't name the crashing kernel. Injected into EVERY container exec
# (incl. the Ray worker processes that host the vLLM worker actors) when set.
if [[ -n "${CUDA_LAUNCH_BLOCKING:-}" ]]; then
    SING_CMD+=(--env "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}")
fi

# Manual pipeline-parallel layer partition (comma-separated per-stage layer
# counts, must sum to num_hidden_layers). Forwarded only when set so vLLM's
# even split is the default. Used to land PP stage boundaries on full-indexer
# layers for GLM-5.2 (its skip-topk DSA layer at a stage boundary trips a
# KeyError in get_attn_backends_for_group on the default even split).
if [[ -n "${VLLM_PP_LAYER_PARTITION:-}" ]]; then
    SING_CMD+=(--env "VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION}")
fi

if [[ "${NUM_NODES}" -le 1 ]]; then
    # -------------------------------------------------------------------------
    # Single-node launch (existing behavior)
    # -------------------------------------------------------------------------
    "${SING_CMD[@]}" "${CONTAINER_PATH}" "${VLLM_ARGS[@]}"
else
    # -------------------------------------------------------------------------
    # Multi-node launch: bootstrap Ray cluster via srun, then launch vLLM on head
    # -------------------------------------------------------------------------
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: NUM_NODES > 1 requires running inside a SLURM allocation."
        echo "       Submit with: sbatch --nodes=${NUM_NODES} --ntasks-per-node=1 \\"
        echo "                           --gpus-per-node=${TP_SIZE} run_vllm_server.sh"
        exit 1
    fi

    # Discover allocated nodes from SLURM
    mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    NUM_ALLOCATED="${#NODES[@]}"

    if [[ "${NUM_ALLOCATED}" -ne "${NUM_NODES}" ]]; then
        echo "ERROR: SLURM allocated ${NUM_ALLOCATED} nodes but NUM_NODES=${NUM_NODES}."
        echo "       Ensure --nodes=${NUM_NODES} was passed to sbatch."
        exit 1
    fi

    HEAD_NODE="${NODES[0]}"
    WORKER_NODES=("${NODES[@]:1}")

    # Get head node IP on the high-speed fabric.
    # On Olivia (HPE Cray EX with Slingshot), the Slingshot interface is typically hsn0.
    # Fall back to the first non-loopback IP if hsn0 isn't present.
    HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
        bash -c "ip -4 -o addr show hsn0 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1 || hostname -I | awk '{print \$1}'")

    if [[ -z "${HEAD_NODE_IP}" ]]; then
        echo "ERROR: Failed to resolve head node IP for ${HEAD_NODE}"
        exit 1
    fi

    # Per-job Ray temp dir. Must be SHORT — Ray creates AF_UNIX sockets at
    # ${temp_dir}/session_TIMESTAMP_PID/sockets/plasma_store, and AF_UNIX paths
    # are capped at 107 bytes. Anything under /cluster/work/... immediately
    # blows past the limit. Use /tmp per-node; Ray doesn't need a shared temp
    # dir across nodes — each Ray process uses its local one.
    RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}"

    RAY_PORT="${RAY_PORT:-6379}"

    echo ""
    echo "=============================================="
    echo "Multi-Node Ray Bootstrap"
    echo "=============================================="
    echo "Head node:      ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})"
    echo "Worker nodes:   ${WORKER_NODES[*]}"
    echo "Ray temp dir:   ${RAY_TEMP_DIR}"
    echo "GPUs per node:  ${TP_SIZE}"
    echo "=============================================="
    echo ""

    # Signal handler: tear down Ray on all nodes at exit.
    # --overlap lets this srun share the allocation with already-running steps
    # (the Ray head/worker sruns still hold task slots even when backgrounded).
    cleanup_ray() {
        echo ""
        echo "[$(date '+%H:%M:%S')] Tearing down Ray cluster..."
        srun --overlap --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
            "${SING_CMD[@]}" "${CONTAINER_PATH}" ray stop 2>/dev/null || true
        cleanup_proxy 2>/dev/null || true
    }
    trap cleanup_ray EXIT INT TERM

    # Create the Ray temp dir on EVERY allocated node. Singularity --bind requires
    # the source path to exist on the host before the container launches.
    echo "[$(date '+%H:%M:%S')] Creating ${RAY_TEMP_DIR} on all ${NUM_NODES} nodes..."
    srun --overlap --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        mkdir -p "${RAY_TEMP_DIR}"

    # NOTE on --overlap: the Ray head/worker srun commands below use & to run in
    # the background but still HOLD their SLURM task allocations until killed.
    # Without --overlap, any subsequent srun (ray status poll, vLLM launch,
    # cleanup) blocks forever waiting for a task slot that's already consumed.
    # --overlap tells SLURM "run this step alongside existing ones, share the
    # resources". This is the standard pattern for SLURM multi-step scripts.

    # Match Ray's CPU view to the SLURM allocation. Without --num-cpus, Ray
    # auto-detects the host's logical CPU count (on GH200 that's 288), and its
    # worker_pool starts pre-warming ~288 idle python workers. Most of them
    # fail to register within the timeout and the raylet spews
    # "Some workers... have not registered within the timeout" at high rate,
    # which stalls legitimate actor placement. SLURM_CPUS_PER_TASK reflects
    # what we actually got from sbatch; fall back to TP_SIZE*8 (our sbatch
    # opts default) if the env var isn't set.
    RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-$((TP_SIZE * 8))}}"

    # Pin Ray + vLLM to the Slingshot interface on EVERY node. Two problems
    # get solved together here:
    #
    # 1. `ray start` on workers without --node-ip-address auto-detects via
    #    socket.gethostbyname(), which returns the ethernet IP (10.168.x.x).
    #    Ray then registers the worker under the ethernet IP even though the
    #    cluster was bootstrapped on Slingshot.
    #
    # 2. vLLM's RayExecutor validates "every node must have a unique IP" and
    #    also uses each actor's node IP to build its placement-group spec.
    #    If actors on the head see Slingshot and actors on workers see
    #    ethernet, vLLM gets 3+ unique IPs for 2 nodes and hard-fails with:
    #      RuntimeError: Every node should have a unique IP address.
    #
    # Fix: resolve each worker's Slingshot IP up front (same srun-from-login
    # approach used for HEAD_NODE_IP; doing it inside the container has been
    # flaky — hsn0 sometimes isn't visible from inside apptainer), then pass
    # the IP into the worker srun as --node-ip-address + VLLM_HOST_IP +
    # RAY_node_ip_address. Ray actors forked from that raylet inherit the
    # env vars and all report the Slingshot IP consistently.
    declare -A WORKER_IPS
    for worker in "${WORKER_NODES[@]}"; do
        WORKER_IPS[$worker]=$(srun --nodes=1 --ntasks=1 -w "$worker" \
            bash -c "ip -4 -o addr show hsn0 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1 || hostname -I | awk '{print \$1}'")
        if [[ -z "${WORKER_IPS[$worker]}" ]]; then
            echo "ERROR: Failed to resolve worker IP for ${worker}"
            exit 1
        fi
        echo "  worker: ${worker} -> ${WORKER_IPS[$worker]}"
    done

    # Start Ray head on the first node (backgrounded, --block keeps it alive)
    echo "[$(date '+%H:%M:%S')] Starting Ray head on ${HEAD_NODE} (num-cpus=${RAY_NUM_CPUS} num-gpus=${TP_SIZE})..."
    srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
        --output="${WORKDIR:-$PWD}/logs/ray_head_${SLURM_JOB_ID}.log" \
        "${SING_CMD[@]}" \
        --env "RAY_TMPDIR=${RAY_TEMP_DIR}" \
        --env "VLLM_HOST_IP=${HEAD_NODE_IP}" \
        --env "RAY_node_ip_address=${HEAD_NODE_IP}" \
        --bind "${RAY_TEMP_DIR}:${RAY_TEMP_DIR}" \
        "${CONTAINER_PATH}" \
        ray start --head \
            --node-ip-address="${HEAD_NODE_IP}" \
            --port="${RAY_PORT}" \
            --num-cpus="${RAY_NUM_CPUS}" \
            --num-gpus="${TP_SIZE}" \
            --temp-dir="${RAY_TEMP_DIR}" \
            --block &
    RAY_HEAD_SRUN_PID=$!

    # Give the head time to become reachable
    sleep 15

    # Start one Ray worker per remaining node. We launch each worker in its
    # own srun step so we can pass node-specific env (VLLM_HOST_IP,
    # RAY_node_ip_address) and a node-specific --node-ip-address. See the
    # WORKER_IPS preamble above for why we resolve each IP on the login node
    # rather than inside the container.
    RAY_WORKER_SRUN_PIDS=()
    for worker in "${WORKER_NODES[@]}"; do
        WIP="${WORKER_IPS[$worker]}"
        echo "[$(date '+%H:%M:%S')] Starting Ray worker on ${worker} (node-ip=${WIP})..."
        srun --overlap --nodes=1 --ntasks=1 -w "${worker}" \
            --output="${WORKDIR:-$PWD}/logs/ray_worker_${SLURM_JOB_ID}_${worker}.log" \
            "${SING_CMD[@]}" \
            --env "RAY_TMPDIR=${RAY_TEMP_DIR}" \
            --env "VLLM_HOST_IP=${WIP}" \
            --env "RAY_node_ip_address=${WIP}" \
            --bind "${RAY_TEMP_DIR}:${RAY_TEMP_DIR}" \
            "${CONTAINER_PATH}" \
            ray start \
                --address="${HEAD_NODE_IP}:${RAY_PORT}" \
                --node-ip-address="${WIP}" \
                --num-cpus="${RAY_NUM_CPUS}" \
                --num-gpus="${TP_SIZE}" \
                --temp-dir="${RAY_TEMP_DIR}" \
                --block &
        RAY_WORKER_SRUN_PIDS+=($!)
    done

    # Wait for the Ray cluster to show the expected GPU count.
    EXPECTED_GPUS=$((TP_SIZE * NUM_NODES))
    echo "[$(date '+%H:%M:%S')] Waiting for Ray cluster to register ${EXPECTED_GPUS} GPUs..."
    WAIT_MAX=60
    WAIT_COUNT=0
    while [[ "${WAIT_COUNT}" -lt "${WAIT_MAX}" ]]; do
        RAY_STATUS=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
            "${SING_CMD[@]}" \
            --env "RAY_TMPDIR=${RAY_TEMP_DIR}" \
            --bind "${RAY_TEMP_DIR}:${RAY_TEMP_DIR}" \
            "${CONTAINER_PATH}" \
            ray status --address="${HEAD_NODE_IP}:${RAY_PORT}" 2>/dev/null || echo "")
        ACTIVE_GPUS=$(echo "${RAY_STATUS}" | grep -oE '[0-9]+\.?[0-9]*/[0-9]+\.?[0-9]* GPU' | grep -oE '/[0-9]+' | tr -d '/' | head -1 || echo "0")
        if [[ "${ACTIVE_GPUS}" -ge "${EXPECTED_GPUS}" ]]; then
            echo "[$(date '+%H:%M:%S')] Ray cluster ready with ${ACTIVE_GPUS} GPUs."
            break
        fi
        WAIT_COUNT=$((WAIT_COUNT + 1))
        sleep 2
    done

    if [[ "${WAIT_COUNT}" -ge "${WAIT_MAX}" ]]; then
        echo "WARNING: Ray cluster did not reach expected ${EXPECTED_GPUS} GPUs within $((WAIT_MAX * 2))s."
        echo "         Continuing anyway; vLLM will fail fast if the cluster is incomplete."
    fi

    # Launch vLLM on the head node. It will discover and use the Ray cluster.
    echo ""
    echo "[$(date '+%H:%M:%S')] Launching vLLM on ${HEAD_NODE} with TP=${TP_SIZE} PP=${PP_SIZE}..."
    echo ""
    VLLM_ARGS+=("--distributed-executor-backend" "ray")

    # Engine-as-Ray-actor mode. CoreEngineActorManager runs the EngineCore as a
    # Ray actor in the cluster's single Ray job — avoiding the subprocess
    # EngineCore's second ray.init() (ActorHandleNotFoundError) AND using
    # RayExecutorV2 (no Ray Compiled Graph → no multi-node PP decode wedge). This
    # is what makes GLM-5.2 multi-node actually decode tokens (verified: it
    # answers, vs. the legacy executor wedging at 0 tok/s), so it is AUTO-ON for
    # GLM-5.2 (also triggerable via EXTRA_VLLM_ARGS="--data-parallel-backend ray").
    # HEAD_NODE_IP is known here.
    #
    # Ornith's 397B 2-node preset is ALSO built from vLLM main with this same
    # external Ray bootstrap, so it hits the identical subprocess-EngineCore
    # second-ray.init() failure — auto-on for it too. (Unverified on Olivia like
    # the rest of the Ornith preset; if it misbehaves, the glm52 path is the
    # reference. ornith_gh200 is single-node and never reaches here.)
    # ENGINE_AS_ACTOR=0 forces the PLAIN Ray distributed executor instead (no DP
    # actor). Useful on newer vLLM where the subprocess-EngineCore ray.init() issue
    # may be fixed — and where the engine-as-actor DP device-assignment miscomputes
    # physical GPU ids for a multi-node TP+PP (DP=1) engine (indexes the whole world
    # into a per-node 4-GPU CUDA_VISIBLE_DEVICES -> IndexError).
    if [[ "${ENGINE_AS_ACTOR:-auto}" != "0" && ( "${IS_GLM52}" == "1" || "${IS_ORNITH}" == "1" || "${EXTRA_VLLM_ARGS:-}" == *"data-parallel-backend"* ) ]]; then
        # Add the backend flag unless the user already supplied it via EXTRA_VLLM_ARGS.
        if [[ "${EXTRA_VLLM_ARGS:-}" != *"data-parallel-backend"* ]]; then
            VLLM_ARGS+=("--data-parallel-backend" "ray")
        fi
        VLLM_ARGS+=("--data-parallel-address" "${HEAD_NODE_IP}")
        # ParallelConfig.__post_init__ OVERRIDES --data-parallel-address with the
        # env VLLM_DP_MASTER_IP (default 127.0.0.1) in the DP=1 path we hit, so
        # create_dp_placement_groups asserts node:127.0.0.1 is missing. The env
        # is the actual lever — forward it (into the vLLM srun via SING_CMD).
        SING_CMD+=(--env "VLLM_DP_MASTER_IP=${HEAD_NODE_IP}")
        # Our single DP rank's workers (world_size = TP*PP = 12) span 3 nodes.
        # Default pack strategy "strict" computes dp_size_available =
        # gpus_per_node // world_size = 4 // 12 = 0 ("not enough resources").
        # "span" collects the rank's bundles ACROSS nodes (dp_size_available=1) —
        # exactly a multi-node TP*PP engine.
        SING_CMD+=(--env "VLLM_RAY_DP_PACK_STRATEGY=span")
        echo "[$(date '+%H:%M:%S')] engine-as-actor: DP master=${HEAD_NODE_IP}, pack=span"
    fi

    # Force vLLM's host-IP detection to use the Slingshot IP we bootstrapped
    # Ray with. Without this, vLLM's placement group spec pins bundle 0 to
    # `node:<ethernet_ip>` (e.g. 10.168.0.50), which Ray has never heard of
    # — Ray was started with --node-ip-address=<slingshot_ip> (10.63.x.x).
    # The symptom is an infinite
    #   "No available node types can fulfill resource request
    #    {'node:10.168.x.x': 0.001, 'GPU': 1.0}"
    # loop after "Connected to Ray cluster".
    srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
        --output="${WORKDIR:-$PWD}/logs/vllm_server_${SLURM_JOB_ID}_head.log" \
        "${SING_CMD[@]}" \
        --env "RAY_TMPDIR=${RAY_TEMP_DIR}" \
        --env "RAY_ADDRESS=${HEAD_NODE_IP}:${RAY_PORT}" \
        --env "VLLM_HOST_IP=${HEAD_NODE_IP}" \
        --env "RAY_node_ip_address=${HEAD_NODE_IP}" \
        --bind "${RAY_TEMP_DIR}:${RAY_TEMP_DIR}" \
        "${CONTAINER_PATH}" \
        "${VLLM_ARGS[@]}"
fi