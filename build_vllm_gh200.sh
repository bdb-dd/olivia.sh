#!/bin/bash
#SBATCH --job-name=build-vllm-gh200
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=build_vllm_%j.log

# =============================================================================
# Build vLLM for NVIDIA GH200 (GraceHopper) ARM64 GPUs
# Target: NRIS Olivia HPC Cluster
# Base: NGC PyTorch 25.12 (PyTorch 2.10.0a0+nv25.12, CUDA 13.1)
# 
# Key: Preserve NGC's custom PyTorch - don't let pip overwrite it!
# =============================================================================

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p "${WORKDIR:-$PWD}/logs"

# =============================================================================
# Model Presets
# =============================================================================
# Define model configurations here. Each preset specifies:
#   - Description: Human-readable description
#   - VLLM_VERSION: Recommended vLLM version
#   - TRANSFORMERS_MIN: Minimum transformers version
#   - NOTES: Any special build considerations
#
# To add a new preset, add entries to the case statement below.
# =============================================================================

show_presets() {
    echo "Available model presets:"
    echo ""
    echo "  glm47      - GLM-4.7 (358B) - Latest flagship model from THUDM"
    echo "               vLLM: main, transformers>=5.0.0rc0"
    echo "               Requires ~358GB VRAM (FP8) or ~716GB (BF16)"
    echo ""
    echo "  devstral   - Devstral/Mistral models (7B-123B)"
    echo "               vLLM: main, transformers>=4.45.0"
    echo "               Standard Mistral architecture"
    echo ""
    echo "  llama      - Llama 3.x models (8B-405B)"
    echo "               vLLM: main, transformers>=4.45.0"
    echo "               Meta's Llama architecture"
    echo ""
    echo "  qwen       - Qwen 2.5 models (7B-72B)"
    echo "               vLLM: main, transformers>=4.45.0"
    echo "               Alibaba's Qwen architecture"
    echo ""
    echo "  generic    - Generic build (default)"
    echo "               vLLM: main, transformers>=4.45.0"
    echo "               Use for unlisted models"
    echo ""
    echo "Usage: MODEL_ID=<preset> ./build_vllm_gh200.sh"
    echo "       MODEL_ID=glm47 ./build_vllm_gh200.sh"
    echo ""
    echo "Override defaults: MODEL_ID=glm47 VLLM_VERSION=v0.6.6 ./build_vllm_gh200.sh"
}

# Apply preset configuration
apply_preset() {
    local preset="$1"

    case "${preset}" in
        glm47|GLM47|glm-4.7|GLM-4.7)
            MODEL_ID="glm47"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=5.0.0rc0"
            PRESET_NOTES="GLM-4.7 requires MTP speculative decoding, tool/reasoning parsers"
            ;;
        devstral|mistral|Devstral|Mistral)
            MODEL_ID="devstral"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=4.45.0"
            PRESET_NOTES="Standard Mistral architecture, ngram speculative decoding supported"
            ;;
        llama|llama3|Llama|Llama3)
            MODEL_ID="llama"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=4.45.0"
            PRESET_NOTES="Meta Llama architecture"
            ;;
        qwen|qwen2|Qwen|Qwen2)
            MODEL_ID="qwen"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=4.45.0"
            PRESET_NOTES="Alibaba Qwen architecture"
            ;;
        generic|Generic)
            MODEL_ID="generic"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=4.45.0"
            PRESET_NOTES="Generic build for unlisted models"
            ;;
        help|--help|-h|list)
            show_presets
            exit 0
            ;;
        "")
            echo "No MODEL_ID specified."
            echo ""
            show_presets
            exit 1
            ;;
        *)
            # Unknown preset - use as custom MODEL_ID
            echo "Note: '${preset}' is not a known preset, using as custom MODEL_ID"
            MODEL_ID="${preset}"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=4.45.0"
            PRESET_NOTES="Custom model configuration"
            ;;
    esac
}

# =============================================================================
# Configuration
# =============================================================================

WORKDIR="${WORKDIR:-$PWD}"
NGC_IMAGE="docker://nvcr.io/nvidia/pytorch:25.12-py3"

# Container output directory (shared location on Olivia)
CONTAINER_DIR="${CONTAINER_DIR:-}"

# Model identifier - apply preset first
MODEL_ID="${MODEL_ID:-}"
apply_preset "${MODEL_ID}"

# Allow override of preset defaults
VLLM_VERSION="${VLLM_VERSION:-${PRESET_VLLM_VERSION}}"

# Build index (for multiple builds of same model type)
BUILD_INDEX="${BUILD_INDEX:-1}"

# Derived names
SANDBOX_NAME="vllm-${MODEL_ID}-${BUILD_INDEX}-sandbox"
SANDBOX_PATH="${CONTAINER_DIR}/${SANDBOX_NAME}"
FINAL_IMAGE="${CONTAINER_DIR}/vllm-${MODEL_ID}-${BUILD_INDEX}.sif"

# Cache directories (bind mount these to avoid filling container)
CACHE_DIR="${WORKDIR}/cache"
PIP_CACHE="${CACHE_DIR}/pip"
HF_CACHE="${CACHE_DIR}/huggingface"

echo "=============================================="
echo "Building vLLM for GH200"
echo "=============================================="
echo ""
echo "Preset:         ${MODEL_ID}"
echo "  Transformers: ${PRESET_TRANSFORMERS}"
echo "  Notes:        ${PRESET_NOTES}"
echo ""
echo "Build Configuration:"
echo "  Container dir:  ${CONTAINER_DIR}"
echo "  Build index:    ${BUILD_INDEX}"
echo "  Sandbox:        ${SANDBOX_NAME}"
echo "  Sandbox path:   ${SANDBOX_PATH}"
echo "  vLLM version:   ${VLLM_VERSION}"
echo "  NGC base:       ${NGC_IMAGE}"
echo ""

# Create container directory if it doesn't exist
mkdir -p "${CONTAINER_DIR}"

# Create cache directories
mkdir -p "${PIP_CACHE}" "${HF_CACHE}"

# Batch mode check (non-interactive when submitted via sbatch)
BATCH_MODE="${BATCH_MODE:-0}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    BATCH_MODE=1
fi

# Check if fakeroot is available
FAKEROOT_AVAILABLE=0
if singularity exec --fakeroot --help &>/dev/null; then
    FAKEROOT_AVAILABLE=1
    echo "Fakeroot: available"
else
    echo "Fakeroot: not available (will use writable-tmpfs or require root)"
fi

# -----------------------------------------------------------------------------
# Phase 1: Create sandbox from NGC base image
# -----------------------------------------------------------------------------
echo "[Phase 1] Creating sandbox from NGC base image..."

if [[ -d "${SANDBOX_PATH}" ]]; then
    if [[ "${BATCH_MODE}" == "1" ]]; then
        echo "Sandbox already exists. Using existing sandbox (batch mode)."
    else
        echo "Sandbox already exists. Remove it? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "${SANDBOX_PATH}"
        else
            echo "Using existing sandbox"
        fi
    fi
fi

if [[ ! -d "${SANDBOX_PATH}" ]]; then
    singularity build --sandbox "${SANDBOX_PATH}" "${NGC_IMAGE}"
fi

# -----------------------------------------------------------------------------
# Phase 2: Verify NGC PyTorch before modifications
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 2] Verifying NGC PyTorch installation..."

singularity exec --nv "${SANDBOX_PATH}" python3 -c "
import torch
import sys

print('=== NGC PyTorch Info ===')
print(f'Python: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch path: {torch.__path__[0]}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

# Verify this is the NGC build
assert 'nv25' in torch.__version__ or 'nv24' in torch.__version__, \
    f'Expected NGC PyTorch, got: {torch.__version__}'
print('\\n✓ NGC PyTorch verified')
"

# Save PyTorch info for later verification
singularity exec --nv "${SANDBOX_PATH}" python3 -c "import torch; print(torch.__version__)" > pytorch_version_before.txt
echo "NGC PyTorch version saved to pytorch_version_before.txt"

# -----------------------------------------------------------------------------
# Phase 3: Build vLLM with constraints to preserve NGC PyTorch
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 3] Building vLLM (preserving NGC PyTorch)..."

# Determine singularity exec options based on available features
if [[ "${FAKEROOT_AVAILABLE}" == "1" ]]; then
    echo "Using fakeroot mode (changes persist in sandbox)"
    SING_OPTS="--nv --fakeroot --writable"
else
    echo "Using writable mode (requires root or unprivileged user namespace)"
    SING_OPTS="--nv --writable"
fi

# Export preset values for use inside container
export PRESET_TRANSFORMERS="${PRESET_TRANSFORMERS}"
export VLLM_VERSION="${VLLM_VERSION}"

singularity exec ${SING_OPTS} \
    --env "PRESET_TRANSFORMERS=${PRESET_TRANSFORMERS}" \
    --env "VLLM_VERSION=${VLLM_VERSION}" \
    --bind "${PIP_CACHE}:/root/.cache/pip" \
    "${SANDBOX_PATH}" /bin/bash << 'BUILDSCRIPT'

set -euo pipefail

echo "Build environment:"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo "  PWD: $(pwd)"

echo "Installing build dependencies..."
pip install --no-cache-dir --root-user-action=ignore ninja cmake wheel packaging setuptools-scm

echo "Cloning vLLM repository..."
cd /opt
if [[ -d vllm ]]; then
    rm -rf vllm
fi

# Clone vLLM (use specific version or main)
VLLM_VERSION="${VLLM_VERSION:-main}"
if [[ "$VLLM_VERSION" == "main" ]]; then
    git clone --depth 1 https://github.com/vllm-project/vllm.git
else
    git clone --depth 1 --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git
fi

cd vllm
echo "vLLM source cloned: $(git describe --tags --always 2>/dev/null || echo 'unknown')"

# Get current NGC PyTorch version for constraints
NGC_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "NGC PyTorch version: ${NGC_TORCH_VERSION}"

# Create pip constraints file to prevent torch replacement
cat > /tmp/constraints.txt << CONSTRAINTS
# Pin torch to prevent pip from replacing NGC's custom build
# This version string must match exactly what NGC provides
torch==${NGC_TORCH_VERSION}
# These may need adjustment based on NGC container contents
# torchvision and torchaudio are often bundled
CONSTRAINTS

echo "Created constraints file:"
cat /tmp/constraints.txt

# Set build environment
export TORCH_CUDA_ARCH_LIST="9.0"  # Hopper architecture
export MAX_JOBS="${MAX_JOBS:-8}"
export PIP_CONSTRAINT=/tmp/constraints.txt
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"

# Check CUDA compiler
echo "NVCC version:"
nvcc --version

# Strategy: Install dependencies first, then vLLM with --no-deps
echo ""
# Use preset transformers version (passed via --env)
TRANSFORMERS_PKG="transformers${PRESET_TRANSFORMERS:->=4.45.0}"
echo "Installing vLLM dependencies (excluding torch)..."
echo "  Transformers package: ${TRANSFORMERS_PKG}"

# Parse requirements but skip torch-related packages
# This is the key step - we install everything EXCEPT torch
# Then vLLM will be installed with --no-deps so it can't overwrite torch
pip install --no-cache-dir \
    --root-user-action=ignore \
    --constraint /tmp/constraints.txt \
    numpy \
    "${TRANSFORMERS_PKG}" \
    tokenizers>=0.19.0 \
    sentencepiece \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.0 \
    prometheus-client \
    prometheus-fastapi-instrumentator>=7.0.0 \
    py-cpuinfo \
    tiktoken \
    lm-format-enforcer>=0.10.6 \
    outlines>=0.0.46 \
    typing_extensions>=4.10 \
    filelock \
    requests \
    tqdm \
    msgspec \
    gguf \
    importlib_metadata \
    huggingface_hub \
    mistral_common>=1.5.0 \
    pyyaml \
    pillow \
    blake3 \
    compressed-tensors>=0.8.0 \
    depyf \
    cloudpickle \
    partial-json-parser \
    openai>=1.0 \
    aiohttp \
    einops \
    protobuf \
    ray \
    psutil \
    cbor2 \
    cachetools \
    scipy \
    diskcache \
    xxhash \
    anthropic==0.71.0 \
    grpcio-reflection>=1.76.0 \
    ijson \
    "llguidance>=1.3.0,<1.4.0" \
    mcp \
    model-hosting-container-standards>=0.1.10 \
    openai-harmony>=0.0.3 \
    opencv-python-headless>=4.11.0 \
    pybase64 \
    setproctitle \
    lark==1.2.2 \
    2>&1 | tail -30 || true

# Try to install xgrammar (vLLM constraint grammar feature)
pip install --no-cache-dir --root-user-action=ignore xgrammar 2>&1 | tail -5 || echo "xgrammar not available, continuing..."

# Install flashinfer (use --no-deps to avoid pulling torch)
echo ""
echo "Installing flashinfer-python..."
pip install --no-cache-dir --root-user-action=ignore --no-deps flashinfer-python 2>&1 | tail -5 || echo "flashinfer not available for ARM64, continuing..."

# Install flash-attention for ARM64 (may need to build from source)
# CRITICAL: Use --no-deps to prevent flash-attn from pulling in PyPI torch!
echo ""
echo "Installing FlashAttention (with --no-deps to preserve NGC PyTorch)..."
pip install --no-cache-dir --no-deps --root-user-action=ignore flash-attn --no-build-isolation 2>&1 | tail -20 || {
    echo "FlashAttention pip install failed, trying from source..."
    pip install --no-cache-dir --no-deps --root-user-action=ignore git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation 2>&1 | tail -20 || {
        echo "Warning: FlashAttention installation failed (may affect performance)"
    }
}

# Build and install vLLM
echo ""
echo "========================================"
echo "Building vLLM CUDA kernels..."
echo "========================================"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"
echo ""
echo "This will take 10-30 minutes. Progress shown below:"
echo "----------------------------------------"

# CRITICAL: Use --no-deps to prevent pip from pulling in its own torch!
# We already installed all dependencies manually above.
# This is the key to preserving NGC PyTorch.
# Using -v for verbose output so we can see compilation progress
pip install -v --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    --root-user-action=ignore \
    . 2>&1 | tee /tmp/vllm_build.log | while IFS= read -r line; do
        # Show all lines but highlight important ones
        if [[ "$line" == *"Building"* ]] || \
           [[ "$line" == *"Compiling"* ]] || \
           [[ "$line" == *"nvcc"* ]] || \
           [[ "$line" == *".cpp"* ]] || \
           [[ "$line" == *".cu"* ]] || \
           [[ "$line" == *"error"* ]] || \
           [[ "$line" == *"Error"* ]] || \
           [[ "$line" == *"warning:"* ]] || \
           [[ "$line" == *"Successfully"* ]]; then
            echo "$line"
        fi
    done

BUILD_STATUS=${PIPESTATUS[0]}
echo "----------------------------------------"

# Copy build log to persistent location in sandbox
cp /tmp/vllm_build.log /opt/vllm_build.log 2>/dev/null || true
echo "Full build log saved to: /opt/vllm_build.log (inside container)"

# If --no-deps fails due to missing deps, fall back to constraints approach
if [[ ${BUILD_STATUS} -ne 0 ]]; then
    echo ""
    echo "WARNING: --no-deps build failed (exit code: ${BUILD_STATUS})"
    echo "Trying fallback with constraints..."
    echo "----------------------------------------"
    pip install -v --no-cache-dir \
        --no-build-isolation \
        --constraint /tmp/constraints.txt \
        --root-user-action=ignore \
        . 2>&1 | tee /tmp/vllm_build.log | tail -100
fi

# Verify NGC PyTorch is still intact
echo ""
echo "Verifying PyTorch after vLLM install..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
assert 'nv25' in torch.__version__ or 'nv24' in torch.__version__, \
    f'ERROR: NGC PyTorch was replaced! Got: {torch.__version__}'
print('✓ NGC PyTorch preserved!')
"

# Verify vLLM
echo ""
echo "Verifying vLLM installation..."
python3 -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
print('✓ vLLM imported successfully!')
"

echo ""
echo "Build complete!"

BUILDSCRIPT

# -----------------------------------------------------------------------------
# Phase 4: Verify final installation
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 4] Final verification..."

singularity exec --nv "${SANDBOX_PATH}" python3 << 'VERIFY'
import sys
print("=" * 50)
print("Final Installation Verification")
print("=" * 50)

# Check PyTorch
import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Quick GPU test
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print(f"GPU matmul test: ✓")

# Verify NGC build
if 'nv25' not in torch.__version__ and 'nv24' not in torch.__version__:
    print(f"\n⚠ WARNING: This may not be NGC PyTorch!")
    print(f"  Expected 'nv25' or 'nv24' in version string")
else:
    print(f"\n✓ NGC PyTorch confirmed")

# Check vLLM
import vllm
print(f"\nvLLM: {vllm.__version__}")

# Check if CUDA graphs work (this is the key test)
try:
    from vllm.worker.model_runner import CUDAGraphRunner
    print("CUDA Graphs module: ✓")
except ImportError as e:
    print(f"CUDA Graphs module: ⚠ {e}")

# Check torch.compile availability
try:
    @torch.compile
    def test_fn(x):
        return x * 2
    print("torch.compile: ✓")
except Exception as e:
    print(f"torch.compile: ⚠ {e}")

print("\n" + "=" * 50)
print("Verification complete!")
print("=" * 50)
VERIFY

# Compare PyTorch versions
echo ""
BEFORE=$(cat pytorch_version_before.txt)
AFTER=$(singularity exec --nv "${SANDBOX_PATH}" python3 -c "import torch; print(torch.__version__)")
echo "PyTorch version before: ${BEFORE}"
echo "PyTorch version after:  ${AFTER}"

if [[ "${BEFORE}" == "${AFTER}" ]]; then
    echo "✓ NGC PyTorch preserved successfully!"
else
    echo "⚠ WARNING: PyTorch version changed!"
    echo "  The build may have replaced NGC PyTorch."
fi

# -----------------------------------------------------------------------------
# Phase 5: Convert to SIF (optional)
# -----------------------------------------------------------------------------
echo ""
CREATE_SIF="${CREATE_SIF:-0}"
if [[ "${BATCH_MODE}" == "1" ]]; then
    if [[ "${CREATE_SIF}" == "1" ]]; then
        echo "[Phase 5] Converting sandbox to SIF (CREATE_SIF=1)..."
        singularity build "${FINAL_IMAGE}" "${SANDBOX_PATH}"
        echo "✓ Created ${FINAL_IMAGE}"
    else
        echo "[Phase 5] Skipping SIF conversion (set CREATE_SIF=1 to enable)"
    fi
else
    echo "[Phase 5] Convert sandbox to SIF image? (y/N)"
    read -r convert_response
    
    if [[ "$convert_response" =~ ^[Yy]$ ]]; then
        echo "Converting to ${FINAL_IMAGE}..."
        singularity build "${FINAL_IMAGE}" "${SANDBOX_PATH}"
        echo "✓ Created ${FINAL_IMAGE}"
        echo ""
        echo "You can now use: singularity exec --nv ${FINAL_IMAGE} ..."
    fi
fi

echo ""
echo "=============================================="
echo "Build Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Test with: ./test_vllm_gh200.sh"
echo "2. Run server: ./run_vllm_server.sh"
echo ""