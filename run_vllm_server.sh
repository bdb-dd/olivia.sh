#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=accel
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0G
#SBATCH --time=08:00:00
#SBATCH --output=vllm_server_%j.log

# =============================================================================
# Run vLLM Server on GH200
# Optimized for NVIDIA GraceHopper with NVLink
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Container
CONTAINER="${CONTAINER:-vllm-gh200-sandbox}"

# Model (default: Devstral 123B)
MODEL="${MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"

# Server settings
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# GPU settings
TP_SIZE="${TP_SIZE:-4}"                    # Tensor parallel size
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"       # GPU memory utilization
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"    # Max context length

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

# vLLM specific
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

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

VLLM_ARGS=(
    "vllm" "serve" "${MODEL}"
    "--tensor-parallel-size" "${TP_SIZE}"
    "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--host" "${HOST}"
    "--port" "${PORT}"
)

# Optional: Enable chunked prefill for long contexts
if [[ "${ENABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
fi

# Optional: Speculative decoding
if [[ -n "${SPECULATIVE_MODEL:-}" ]]; then
    VLLM_ARGS+=(
        "--speculative-model" "${SPECULATIVE_MODEL}"
        "--num-speculative-tokens" "${NUM_SPECULATIVE_TOKENS:-5}"
    )
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
    --env "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    --bind "${VLLM_CACHE}:/root/.cache/vllm" \
    "${CONTAINER}" \
    "${VLLM_ARGS[@]}"
