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

# Container directory (shared location on Olivia)
CONTAINER_DIR="${CONTAINER_DIR:-}"

# Container name (can be just the name or full path)
CONTAINER="${CONTAINER:-}"

# Model (default: Devstral 123B)
MODEL="${MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"

# Server settings
PORT="${PORT:-8000}"
HOST="0.0.0.0"  # Always bind to 0.0.0.0, not hostname

# GPU settings
TP_SIZE="${TP_SIZE:-4}"                    # Tensor parallel size
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"       # GPU memory utilization
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"    # Max context length

# Speculative decoding settings
# - "auto": Enable MTP for GLM-4.7, disabled for other models
# - "1": Force enable (MTP for GLM-4.7, ngram for others)
# - "0": Force disable
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-auto}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-4}"  # Max n-gram window size

# GLM-4.7 specific settings (auto-detected when MODEL contains "GLM-4.7")
GLM_TOOL_PARSER="${GLM_TOOL_PARSER:-glm47}"
GLM_REASONING_PARSER="${GLM_REASONING_PARSER:-glm45}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-3}"  # MTP speculative tokens for GLM-4.7

# MoE (Mixture of Experts) settings
# Expert parallel is required for AWQ-quantized MoE models to shard experts across GPUs
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-auto}"  # auto, 0, or 1

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

# Logging configuration
# Set VERBOSE=1 for detailed vLLM logging (shows weight loading progress, etc.)
VERBOSE="${VERBOSE:-0}"
if [[ "${VERBOSE}" == "1" ]]; then
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
else
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
fi
# Enable vLLM's internal logging configuration
VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"

# Cache quantized weights (MARLIN, AWQ, etc.) to speed up subsequent loads
VLLM_CACHE_QUANTIZED_WEIGHTS="${VLLM_CACHE_QUANTIZED_WEIGHTS:-1}"

# vLLM attention backend (now set via CLI arg instead of env var)
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# AWQ-specific optimizations (recommended by QuantTrio/GLM-4.7-AWQ)
# These help with AWQ-quantized MoE models on Hopper GPUs
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

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
echo "  Attention Backend: ${VLLM_ATTENTION_BACKEND}"
echo "  Log Level:        ${VLLM_LOGGING_LEVEL}"
echo ""
# Detect GLM-4.7 model
IS_GLM47=0
if [[ "${MODEL}" == *"GLM-4.7"* ]] || [[ "${MODEL}" == *"glm-4.7"* ]]; then
    IS_GLM47=1
fi

# Detect AWQ quantized model
IS_AWQ=0
if [[ "${MODEL,,}" == *"-awq"* ]] || [[ "${MODEL,,}" == *"awq"* ]]; then
    IS_AWQ=1
fi

# Determine if expert parallel should be enabled
# Auto-enable for AWQ-quantized MoE models (GLM-4.7-AWQ)
USE_EXPERT_PARALLEL=0
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    USE_EXPERT_PARALLEL=1
elif [[ "${ENABLE_EXPERT_PARALLEL}" == "auto" ]]; then
    # Auto-enable for AWQ MoE models
    if [[ "${IS_AWQ}" == "1" ]] && [[ "${IS_GLM47}" == "1" ]]; then
        USE_EXPERT_PARALLEL=1
        echo "[INFO] Auto-enabling expert parallel for AWQ MoE model"
    fi
fi

# Determine if speculative decoding should be enabled
USE_SPECULATIVE=0
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    USE_SPECULATIVE=1
elif [[ "${ENABLE_SPECULATIVE}" == "auto" ]]; then
    # Auto-enable MTP for GLM-4.7 (improves throughput significantly)
    if [[ "${IS_GLM47}" == "1" ]]; then
        USE_SPECULATIVE=1
        echo "[INFO] Auto-enabling MTP speculative decoding for GLM-4.7"
    fi
fi

echo "Speculative Decoding:"
if [[ "${USE_SPECULATIVE}" == "1" ]]; then
    if [[ "${IS_GLM47}" == "1" ]]; then
        echo "  Enabled:          yes"
        echo "  Method:           MTP (Multi-Token Prediction)"
        echo "  Spec Tokens:      ${MTP_SPECULATIVE_TOKENS}"
    else
        echo "  Enabled:          yes"
        echo "  Method:           ngram"
        echo "  Spec Tokens:      ${NUM_SPECULATIVE_TOKENS}"
        echo "  Lookup Max:       ${PROMPT_LOOKUP_MAX}"
    fi
else
    echo "  Enabled:          no"
fi

if [[ "${IS_GLM47}" == "1" ]]; then
    echo ""
    echo "GLM-4.7 Settings:"
    echo "  Tool Parser:      ${GLM_TOOL_PARSER}"
    echo "  Reasoning Parser: ${GLM_REASONING_PARSER}"
    echo "  Auto Tool Choice: ${ENABLE_AUTO_TOOL_CHOICE}"
    if [[ "${IS_AWQ}" == "1" ]]; then
        echo "  Quantization:     AWQ (Hopper-compatible)"
        echo "  Expert Parallel:  ${USE_EXPERT_PARALLEL}"
    fi
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
    "--attention-config.backend" "${VLLM_ATTENTION_BACKEND}"
    "--compilation-config" '{"mode": "NONE"}'
)

# Optional: Enable chunked prefill for long contexts
if [[ "${ENABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
fi

# Speculative decoding (method depends on model)
if [[ "${USE_SPECULATIVE}" == "1" ]]; then
    if [[ "${IS_GLM47}" == "1" ]]; then
        # GLM-4.7 uses MTP (Multi-Token Prediction) speculative decoding
        SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": '${MTP_SPECULATIVE_TOKENS}'}'
    else
        # Default: ngram speculative decoding
        SPEC_CONFIG='{"method": "ngram", "num_speculative_tokens": '${NUM_SPECULATIVE_TOKENS}', "prompt_lookup_max": '${PROMPT_LOOKUP_MAX}'}'
    fi
    VLLM_ARGS+=("--speculative-config" "${SPEC_CONFIG}")
fi

# GLM-4.7 specific arguments
if [[ "${IS_GLM47}" == "1" ]]; then
    VLLM_ARGS+=("--tool-call-parser" "${GLM_TOOL_PARSER}")
    VLLM_ARGS+=("--reasoning-parser" "${GLM_REASONING_PARSER}")
    if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
        VLLM_ARGS+=("--enable-auto-tool-choice")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
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

singularity exec --nv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    --env "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}" \
    --env "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}" \
    --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
    --env "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}" \
    --env "VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING}" \
    --env "VLLM_CACHE_QUANTIZED_WEIGHTS=${VLLM_CACHE_QUANTIZED_WEIGHTS}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    --bind "${VLLM_CACHE}:/root/.cache/vllm" \
    "${CONTAINER_PATH}" \
    "${VLLM_ARGS[@]}"