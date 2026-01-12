#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=accel
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.log

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

# Container
CONTAINER="${CONTAINER:-vllm-gh200-sandbox}"

# Model (default: Devstral 123B)
MODEL="${MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"

# Server settings
PORT="${PORT:-8000}"
HOST="0.0.0.0"  # Always bind to 0.0.0.0, not hostname

# GPU settings
TP_SIZE="${TP_SIZE:-4}"                    # Tensor parallel size
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"       # GPU memory utilization
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"    # Max context length

# Speculative decoding settings (ngram method, disabled by default - works best for repetitive content)
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-4}"  # Max n-gram window size

# Cache directories
HF_CACHE="${HF_CACHE:-$PWD/cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$PWD/cache/vllm}"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

mkdir -p "${HF_CACHE}" "${VLLM_CACHE}"

# GPU ordering: Put slowest GPU (usually GPU 0) last
# This improves tensor parallel performance on GH200
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,0}"

# NCCL optimizations for GH200 NVLink
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"        # Use NVLink for P2P
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}" # GPU Direct RDMA level
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"         # Enable InfiniBand if available

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# vLLM attention backend (now set via CLI arg instead of env var)
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "=============================================="
echo "vLLM Server for GH200"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Container:        ${CONTAINER}"
echo "  Model:            ${MODEL}"
echo "  Tensor Parallel:  ${TP_SIZE}"
echo "  GPU Memory:       ${GPU_MEM_UTIL}"
echo "  Max Model Len:    ${MAX_MODEL_LEN}"
echo "  Port:             ${PORT}"
echo "  Attention Backend: ${VLLM_ATTENTION_BACKEND}"
echo ""
echo "Speculative Decoding (ngram):"
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    echo "  Enabled:          yes"
    echo "  Method:           ngram"
    echo "  Spec Tokens:      ${NUM_SPECULATIVE_TOKENS}"
    echo "  Lookup Max:       ${PROMPT_LOOKUP_MAX}"
else
    echo "  Enabled:          no"
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
# Pre-flight checks
# -----------------------------------------------------------------------------

# Check container exists
if [[ -d "${CONTAINER}" ]]; then
    CONTAINER_TYPE="sandbox"
elif [[ -f "${CONTAINER}" ]]; then
    CONTAINER_TYPE="sif"
else
    echo "Error: Container not found: ${CONTAINER}"
    echo "Run ./build_vllm_gh200.sh first"
    exit 1
fi
echo "Container type: ${CONTAINER_TYPE}"

# Check HuggingFace token
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo ""
    echo "Warning: No HuggingFace token found."
    echo "Set HF_TOKEN for gated models like Devstral."
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
    "--attention-config.backend" "${VLLM_ATTENTION_BACKEND}"
    "--compilation-config" '{"mode": "NONE"}'
)

# Optional: Enable chunked prefill for long contexts
if [[ "${ENABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
fi

# Speculative decoding with ngram (enabled by default)
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    # Build JSON config for ngram speculative decoding
    SPEC_CONFIG=$(cat <<EOF
{"method": "ngram", "num_speculative_tokens": ${NUM_SPECULATIVE_TOKENS}, "prompt_lookup_max": ${PROMPT_LOOKUP_MAX}}
EOF
)
    VLLM_ARGS+=("--speculative-config" "${SPEC_CONFIG}")
fi

# Optional: Quantization
if [[ -n "${QUANTIZATION:-}" ]]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION}")
fi

# Optional: LoRA
if [[ -n "${LORA_MODULES:-}" ]]; then
    VLLM_ARGS+=("--enable-lora" "--lora-modules" "${LORA_MODULES}")
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
echo ""
echo "API endpoints:"
echo "  POST /v1/completions      - Text completions"
echo "  POST /v1/chat/completions - Chat completions"
echo "  GET  /health              - Health check"
echo "  GET  /v1/models           - List models"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

# Run with singularity
singularity exec --nv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    --env "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}" \
    --env "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}" \
    --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    --bind "${VLLM_CACHE}:/root/.cache/vllm" \
    "${CONTAINER}" \
    "${VLLM_ARGS[@]}"