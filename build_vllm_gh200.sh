#!/bin/bash
#SBATCH --job-name=build-vllm-gh200
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Bounded memory: NOT --mem=0, which requests the whole node's RAM (~808G) and
# blocks this 1-GPU build from backfilling into a partially-used node. 128G is
# ample for a vLLM compile; raise it if parallel nvcc/ninja ever pressures it.
#SBATCH --mem=128G
# Build needs ~30-90 min; a short walltime backfills into transient GPU holes
# far more easily than an 8h reservation (the scheduler can only place an 8h job
# in a hole that stays free for 8h before the next higher-priority reservation).
#SBATCH --time=02:00:00
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
    echo "  glm51_v19  - GLM-5.1 (744B, 40B active) MoE+DSA on vLLM v0.19.0 (recipe default)"
    echo "               vLLM: v0.19.0, transformers>=5.4.0"
    echo "               AWQ ~430GB, 8 GPUs (2 nodes × 4× GH200, TP=4 + PP=2)"
    echo "               KNOWN-WEDGE on multi-node PP concurrent decode; use with"
    echo "               anthropic_proxy.py request serialization as workaround."
    echo "               Alias: glm51 (for backwards compat)"
    echo ""
    echo "  glm51_v20  - GLM-5.1 on vLLM v0.20.0 + RayExecutorV2 (QUARANTINED)"
    echo "               vLLM: v0.20.0, transformers>=5.4.0, RayExecutorV2 data plane"
    echo "               Same multi-node PP wedge as glm51_v19 reproduced — kept for"
    echo "               diagnostic work only, not for routine use. Builds to index 2."
    echo ""
    echo "  glm47      - GLM-4.7 (358B) - Latest flagship model from THUDM"
    echo "               vLLM: main, transformers>=5.0.0rc0"
    echo "               Requires ~358GB VRAM (FP8) or ~716GB (BF16)"
    echo ""
    echo "  kimi       - Kimi K2.6 (1T MoE, 32B active) MLA + multimodal from Moonshot"
    echo "               vLLM: v0.19.1, transformers>=4.57.1,<5.0.0"
    echo "               native int4 ~640GB, 8 GPUs (2 nodes × 4× GH200, TP=4 + PP=2)"
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
        glm51_v19|GLM51_V19|glm51|GLM51|glm-5.1|GLM-5.1)
            # MODEL_ID stays "glm51" so the container name is vllm-glm51-<index>-sandbox
            # (both v19 and v20 variants share the glm51 container prefix, index
            # distinguishes them: index 1 = v0.19.0, index 2 = v0.20.0).
            MODEL_ID="glm51"
            # v0.19.0 is the version pinned by the official vLLM GLM-5 recipe
            # (https://github.com/vllm-project/recipes/blob/main/GLM/GLM5.md).
            # Recipe only covers single-node TP=8; multi-node PP is not an upstream-
            # validated config. On 2-node × 4×GH200 (TP=4 + PP=2) over Slingshot,
            # decode wedges reproducibly under concurrent requests — both Ray
            # Compiled Graph (v0.19.0 default) AND RayExecutorV2 (v0.20.0, see
            # glm51_v20) hit the same signature. Use anthropic_proxy.py's
            # request serialization to work around it until there's a proper fix.
            PRESET_VLLM_VERSION="v0.19.0"
            PRESET_TRANSFORMERS=">=5.4.0"
            PRESET_NOTES="GLM-5.1 (744B MoE+DSA), vLLM v0.19.0. Multi-node PP wedges on concurrent decode — pair with proxy serialization."
            ;;
        glm51_v20|GLM51_V20)
            # Quarantined. Builds to container index 2 (vllm-glm51-2-sandbox) so
            # it lives alongside the v0.19.0 build without overwriting.
            MODEL_ID="glm51"
            # v0.20.0 + RayExecutorV2 was attempted to escape the Ray Compiled
            # Graph deadlock (ray#58426). V2 bypasses Compiled Graph and uses
            # MultiprocExecutor's ZMQ/NCCL data plane. Confirmed active via
            # startup logs, BUT the same decode wedge still reproduces with
            # identical signature (Running >=1, 0 tok/s, KV cache frozen) —
            # falsifying the "Compiled Graph is the root cause" hypothesis.
            # Requires torch 2.11 (NGC 26.02+), CUDA 13.0+, transformers>=4.56.
            # Kept here for future diagnostic work, NOT routine use.
            PRESET_VLLM_VERSION="v0.20.0"
            PRESET_TRANSFORMERS=">=5.4.0"
            PRESET_NOTES="GLM-5.1 on vLLM v0.20.0 + RayExecutorV2 — quarantined, same multi-node PP wedge as v0.19.0"
            ;;
        glm47|GLM47|glm-4.7|GLM-4.7)
            MODEL_ID="glm47"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=5.0.0rc0"
            PRESET_NOTES="GLM-4.7 requires MTP speculative decoding, tool/reasoning parsers"
            ;;
        kimi|KIMI|kimi26|kimi-k2.6|kimi_k26|Kimi-K2.6)
            MODEL_ID="kimi"
            # Kimi K2.6 (Moonshot): 1T-param MoE (32B active), MLA attention,
            # multimodal (MoonViT). The base repo moonshotai/Kimi-K2.6 ships
            # native int4 (compressed-tensors, ~640GB) — there is no separate
            # -AWQ repo. Targets 8 GPUs across 2 nodes on Olivia (TP=4 + PP=2),
            # like glm51. Architecture KimiK25ForConditionalGeneration is
            # custom_code. vLLM 0.19.1 is the manually-verified stable release
            # per Moonshot's deploy guide; newer support is nightly-only.
            PRESET_VLLM_VERSION="v0.19.1"
            # Model card lists transformers >=4.57.1,<5.0.0 for STANDALONE HF use.
            # But vLLM serves Kimi via its NATIVE kimi_k25.py (only imports
            # BatchFeature), so transformers 5 works under vLLM — required for the
            # vLLM 0.21+ MQ Ray executor that fixes the multi-node PP compiled-DAG
            # wedge (ray#58426). Overridable via PRESET_TRANSFORMERS env for the
            # 0.21 build experiment (PRESET_TRANSFORMERS='>=5').
            PRESET_TRANSFORMERS="${PRESET_TRANSFORMERS:->=4.57.1,<5.0.0}"
            PRESET_NOTES="Kimi K2.6 (1T MoE, MLA, multimodal): native int4, multi-node TP=4 + PP=2, kimi_k2 parser"
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

# NGC PyTorch base image tag. We need one new enough to:
#
#   1. Ship the TORCH_BOX macro (landed upstream 2025-11-11, pytorch v2.10.0).
#      NGC 25.12 (Dec 2025) shipped the 2025-11-04 alpha which predates it by
#      a week, so `_C_stable_libtorch` fails to compile.
#
#   2. Ship the stable-ABI converter that accepts reference op signatures.
#      NGC 26.01 (Jan 2026) has TORCH_BOX but rejects vLLM v0.19.0's
#      `const torch::stable::Tensor&` parameters with
#      `static assertion failed` / `reference type ... in a union`.
#
# NGC 26.03 (Mar 2026) is the first tag late enough to satisfy both.
#
# vLLM v0.19.0 (Apr 2026) is the pinned version per the official recipe
# (https://github.com/vllm-project/recipes/blob/main/GLM/GLM5.md). If you pin
# a newer vLLM, you may need NGC_PYTORCH_TAG=26.04-py3 or later.
NGC_PYTORCH_TAG="${NGC_PYTORCH_TAG:-26.03-py3}"
NGC_IMAGE="${NGC_IMAGE:-docker://nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}}"

# Container output directory (shared location on Olivia)
CONTAINER_DIR="${CONTAINER_DIR:-}"

if [[ -z "${CONTAINER_DIR}" ]]; then
    echo "Error: CONTAINER_DIR is not set."
    echo "Set CONTAINER_DIR to the directory where containers should be created." 
    exit 1
fi

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

# Path the finished sandbox should end up at. `SANDBOX_PATH` may be rewritten
# to a temporary `.new.JOBID` path during the build so a failed run doesn't
# corrupt the working container (see Phase 1 below). At the end of the build
# we atomically rename the new sandbox into `FINAL_SANDBOX_PATH` and preserve
# the previous one as `.prev.TIMESTAMP` for rollback.
FINAL_SANDBOX_PATH="${SANDBOX_PATH}"

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

# Overwrite protection (requires OVERWRITE=1 to rebuild existing container in batch mode)
OVERWRITE="${OVERWRITE:-0}"

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
        if [[ "${OVERWRITE}" == "1" ]]; then
            # Build into a temporary sandbox path and swap atomically at the
            # end. Previously this branch only printed a warning — the
            # `if [[ ! -d "${SANDBOX_PATH}" ]]` guard below then short-circuited
            # the `singularity build` call, so every "rebuild" silently reused
            # the existing sandbox. Phase 3 still ran pip install on top, so
            # the artifact looked plausible, but the base image layers
            # (NGC PyTorch version, CUDA, etc.) never changed.
            #
            # Using a `.new.JOBID` path gives us two safety properties:
            #   1. If any phase fails, the working sandbox is untouched.
            #   2. On success, we preserve the previous sandbox as
            #      `.prev.TIMESTAMP` so the user can manually roll back if
            #      the new container has a latent runtime issue.
            SWAP_TOKEN="${SLURM_JOB_ID:-$$}"
            SANDBOX_PATH="${FINAL_SANDBOX_PATH}.new.${SWAP_TOKEN}"
            echo "OVERWRITE=1: building into temporary sandbox path."
            echo "  Existing (will be preserved until swap): ${FINAL_SANDBOX_PATH}"
            echo "  Build target (temp):                     ${SANDBOX_PATH}"
            # Clean up any orphan .new.* from a previous failed run — same
            # build index, but different job ids would just pile up.
            for stale in "${FINAL_SANDBOX_PATH}".new.*; do
                if [[ -d "$stale" && "$stale" != "${SANDBOX_PATH}" ]]; then
                    echo "  Removing stale build attempt: ${stale}"
                    rm -rf "${stale}"
                fi
            done
            echo ""
        else
            echo ""
            echo "=============================================="
            echo "ERROR: Container already exists!"
            echo "=============================================="
            echo ""
            echo "  Existing: ${SANDBOX_NAME}"
            echo "  Path:     ${SANDBOX_PATH}"
            echo ""
            echo "To avoid accidentally overwriting working containers,"
            echo "batch mode requires explicit confirmation."
            echo ""
            echo "Options:"
            echo "  1. Build with a different index:"
            echo "     MODEL_ID=${MODEL_ID} BUILD_INDEX=2 sbatch build_vllm_gh200.sh"
            echo ""
            echo "  2. Force overwrite existing container:"
            echo "     MODEL_ID=${MODEL_ID} OVERWRITE=1 sbatch build_vllm_gh200.sh"
            echo ""
            exit 1
        fi
    else
        echo "Sandbox already exists: ${SANDBOX_NAME}"
        echo "Remove it and rebuild? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Removing existing sandbox..."
            rm -rf "${SANDBOX_PATH}"
        else
            echo "Using existing sandbox (will rebuild vLLM inside it)"
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
assert any(m in torch.__version__ for m in ('nv24', 'nv25', 'nv26', 'nv27')), \
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

# Run the build in a PRIVATE IPC namespace (--ipc). On these GH200 nodes the
# shared host IPC namespace has an exhausted/clobbered SysV semaphore space, so
# parallel (MAX_JOBS>1) compiles die a few ninja steps into the vLLM kernel build
# with `semop(1): encountered an error: Invalid argument` (NOT memory — observed
# at ~58 GB). A private IPC namespace gives the build a fresh SysV semaphore space
# so the default MAX_JOBS=8 builds clean (verified). Serial MAX_JOBS=1 also avoids
# it but is far slower. Default ON; set BUILD_IPC=0 to disable (e.g. on a future
# node/Apptainer that lacks --ipc, or once the host IPC space is fixed).
if [[ "${BUILD_IPC:-1}" == "1" ]]; then
    SING_OPTS="${SING_OPTS} --ipc"
    echo "BUILD_IPC -> using a private IPC namespace (--ipc) for the build"
fi

# Export preset values for use inside the container.
# PRESET_TRANSFORMERS is forwarded via the SINGULARITYENV_/APPTAINERENV_ prefix
# rather than `singularity exec --env`: that flag splits a value on commas (and
# this Apptainer build does NOT honor backslash-escaping them), which mangles a
# pip range like ">=4.57.1,<5.0.0" (Kimi needs transformers <5.0.0) into a
# malformed second entry and aborts Phase 3. The prefix mechanism passes the
# value into the container env verbatim, with no comma parsing.
export SINGULARITYENV_PRESET_TRANSFORMERS="${PRESET_TRANSFORMERS}"
export APPTAINERENV_PRESET_TRANSFORMERS="${PRESET_TRANSFORMERS}"
export PRESET_TRANSFORMERS="${PRESET_TRANSFORMERS}"
export VLLM_VERSION="${VLLM_VERSION}"

singularity exec ${SING_OPTS} \
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

# Clone vLLM (use specific version or main).
#
# We avoid `--depth 1` for tagged versions: setuptools_scm reads tag history to
# derive the package version, and a shallow clone truncates history such that
# even an exact tag checkout gets labeled `<next>.dev0+g<sha>.d<date>` instead
# of the tag itself. That matters because vLLM's CMakeLists conditionally adds
# targets based on the detected version, and the wrong version label can
# silently activate code paths the pinned tag shouldn't include.
VLLM_VERSION="${VLLM_VERSION:-main}"
if [[ "$VLLM_VERSION" == "main" ]]; then
    git clone --depth 1 https://github.com/vllm-project/vllm.git
else
    git clone --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git
fi

cd vllm
echo "vLLM source cloned: $(git describe --tags --always 2>/dev/null || echo 'unknown')"

# vLLM's csrc/libtorch_stable holds the stable-ABI ops — including
# per_token_group_fp8_quant, which DeepSeek/GLM/Kimi-family models call at
# forward time. On NGC 26.01+ (we pin 26.03) this target compiles cleanly, so we
# no longer disable it (see the NOTE below for the history). If you override
# NGC_PYTORCH_TAG to an older tag lacking TORCH_BOX, _C_stable_libtorch fails to
# COMPILE — a loud build error, not a silent op drop — so bump to NGC 26.03+.

# NOTE: we previously patched out `_C_stable_libtorch` here because NGC 25.12's
# PyTorch alpha (2025-11-04) predated the upstream TORCH_BOX macro (2025-11-11,
# released in PyTorch 2.10.0). That workaround silently dropped ops like
# per_token_group_fp8_quant from the _C namespace, which GLM-5.1's DSA indexer
# calls unconditionally at forward-pass time. NGC 26.01+ ships a torch cut
# from after TORCH_BOX landed, so the stable-ABI target compiles cleanly and
# we want it built.

# Patch: drop the `hoist=True` kwarg from register_opaque_type calls.
# vLLM v0.19.0 calls `register_opaque_type(ModuleName, typ="value", hoist=True)`
# at import time in vllm/utils/torch_utils.py. The `hoist=` kwarg was added
# to PyTorch after NGC 26.03 (Feb 2026) was cut and NGC 26.04 isn't published
# yet (as of Apr 2026), so `import vllm` fails with:
#   TypeError: register_opaque_type() got an unexpected keyword argument 'hoist'
# `hoist=True` only controls a torch.compile dynamo-graph hoisting optimization
# for opaque-typed values — dropping it is safe for import, and CUDAGraph
# capture still runs. Note: a separate NGC-26.03-vs-vLLM-v0.19.0 skew around
# ModuleName.__fx_repr__ (set vs dict contract) is patched further below;
# that one only trips when dynamo codegen actually runs, i.e. under
# CUDAGRAPH_MODE != NONE. Revisit both patches when we bump to an NGC with
# newer torch.
echo "Patching vllm/utils/torch_utils.py: drop hoist= from register_opaque_type..."
python3 << 'PYPATCH_HOIST'
import re
from pathlib import Path
fp = Path('vllm/utils/torch_utils.py')
if fp.exists():
    src = fp.read_text()
    # Narrow match: only strip `, hoist=<value>` inside a register_opaque_type(...) call.
    new_src, n = re.subn(
        r'(register_opaque_type\([^)]*?),\s*hoist\s*=\s*[A-Za-z0-9_]+',
        r'\1',
        src,
    )
    if n:
        fp.write_text(new_src)
        print(f"Patched {n} register_opaque_type call(s): dropped hoist= kwarg")
    else:
        print("No register_opaque_type(..., hoist=...) call found (OK, may be a newer vLLM)")
else:
    print(f"{fp} not found (OK, may be a newer vLLM layout)")
PYPATCH_HOIST

# Patch: fix ModuleName.__fx_repr__ to return dict instead of set.
# vLLM v0.19.0 returns `(repr_str, {ModuleName})` (a set literal) as the second
# element of __fx_repr__, targeting a newer torch._library.opaque_object API.
# NGC 26.03's PyTorch enforces `(repr_str, dict[str, type])` and rejects set
# with:
#   TypeError: __fx_repr__ for ModuleName must return a dict as the second
#              element, got set
# The error only surfaces under CUDAGraph capture (dynamo fx codegen path).
# With mode=NONE, codegen never runs, so the patch wasn't needed until we
# flipped the default. The globals_dict is used by dynamo to resolve names in
# the generated FX code — since the repr string is `ModuleName(...)`, we map
# the string "ModuleName" to the class. Revisit when NGC ships a torch with
# the set-accepting variant.
echo "Patching vllm/utils/torch_utils.py: ModuleName.__fx_repr__ set -> dict..."
python3 << 'PYPATCH_FXREPR'
from pathlib import Path
fp = Path('vllm/utils/torch_utils.py')
if fp.exists():
    src = fp.read_text()
    old = '{ModuleName})'
    new = '{"ModuleName": ModuleName})'
    # Narrow: the set literal {ModuleName} only appears as __fx_repr__'s return.
    count = src.count(old)
    if count == 1:
        fp.write_text(src.replace(old, new, 1))
        print("Patched ModuleName.__fx_repr__: set {ModuleName} -> dict {\"ModuleName\": ModuleName}")
    elif count == 0:
        print("No {ModuleName} set literal found (OK, may be a newer vLLM)")
    else:
        print(f"WARNING: expected 1 match, found {count} — patch skipped, inspect manually")
else:
    print(f"{fp} not found (OK, may be a newer vLLM layout)")
PYPATCH_FXREPR

# Patch: disable the DeepSeek-V3 "min-latency" fused QKV-A GEMM so its custom op
# is never emitted into the graph.
# vllm/model_executor/models/deepseek_v2.py wraps the low-batch dsv3_fused_a_gemm
# kernel in a custom op `torch.ops.vllm.min_latency_fused_qkv_a_proj` (used by
# DeepSeek/Kimi-family MLA models, incl. Kimi K2.6, at batch <= 16, i.e. decode).
# It registers a fake/meta via direct_register_custom_op(..., fake_impl=...), but
# under NGC's torch alpha that fake never lands in the FakeTensor dispatch table,
# so torch.compile's profile_run dies with:
#   TypeError: Multiple dispatch failed for
#     'torch._ops.vllm.min_latency_fused_qkv_a_proj.default';
#     all __torch_dispatch__ handlers returned NotImplemented
# That crash is independent of cudagraph_mode (it happens before capture), so it
# blocks CUDAGraph entirely and forces eager (~5x slower decode). Forcing
# `_use_min_latency_gemm = False` makes the layer fall back to its parent
# MergedColumnParallelLinear.forward — the SAME merged matmul, identical output,
# only without the low-batch kernel micro-opt — which torch.compile traces fine.
# This is the fix that unblocks CUDAGraph for Kimi K2.6 on GH200. Revisit when
# NGC ships a torch where the custom-op fake registration works (then this whole
# block can go and the kernel micro-opt comes back). See the proposed plan
# plans/proposed/kimi_serving_perf.md for the full investigation.
echo "Patching vllm/model_executor/models/deepseek_v2.py: disable dsv3 min-latency gemm..."
python3 << 'PYPATCH_MINLATENCY'
import re
from pathlib import Path
fp = Path('vllm/model_executor/models/deepseek_v2.py')
if not fp.exists():
    print(f"{fp} not found (OK, may be a newer vLLM layout)")
else:
    src = fp.read_text()
    if 'NGC-torch patch' in src and '_use_min_latency_gemm = (False and' in src:
        print("Already patched (OK)")
    else:
        # Match only the original assignment: `(` immediately followed by newline.
        # The patched form is `(False and  # ...`, so re-runs find 0 matches.
        new_src, n = re.subn(
            r'self\._use_min_latency_gemm = \(\n',
            'self._use_min_latency_gemm = (False and  # NGC-torch patch: '
            'no working fake for the dsv3 min-latency custom op under NGC torch '
            'alpha; force off so torch.compile/CUDAGraph can trace MLA decode.\n',
            src,
        )
        if n == 1:
            fp.write_text(new_src)
            print("Patched _use_min_latency_gemm -> forced False (min-latency gemm disabled)")
        elif n == 0:
            print("No `_use_min_latency_gemm = (` assignment found "
                  "(OK, may be a newer vLLM that fixed this)")
        else:
            print(f"WARNING: expected 1 match, found {n} — patch skipped, inspect manually")
PYPATCH_MINLATENCY

# Patch: make the Kimi K2 reasoning parser actually count reasoning tokens.
# vLLM's base ReasoningParser.count_reasoning_tokens() returns 0 unless a parser
# opts in, and kimi_k2 never overrides it -> usage reasoning_tokens is always 0.
# Kimi K2/K2.7 also typically omit the <think> start token (reasoning begins at
# the start, ended by </think> or a tool-call section), so a naive start/end
# depth counter would also be 0. Mirror the parser's own extract_reasoning().
echo "Patching vllm/reasoning/kimi_k2_reasoning_parser.py: count reasoning tokens..."
python3 << 'PYPATCH_REASONING_COUNT'
from pathlib import Path
fp = Path('vllm/reasoning/kimi_k2_reasoning_parser.py')
if not fp.exists():
    print(f"{fp} not found (OK, may be a different vLLM layout)")
else:
    src = fp.read_text()
    if 'def count_reasoning_tokens' in src:
        print("Already patched (OK)")
    else:
        method = (
            "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
            "        # Olivia patch: base default returns 0; Kimi K2 omits the\n"
            "        # <think> start token, so count tokens before the first\n"
            "        # </think>/tool-call-section marker, dropping a leading\n"
            "        # <think>. Mirrors extract_reasoning.\n"
            "        if self._identity_parser is not None:\n"
            "            return 0\n"
            "        end_idx = None\n"
            "        for i, t in enumerate(token_ids):\n"
            "            if t == self._end_token_id or (\n"
            "                self._tool_section_start_token_id is not None\n"
            "                and t == self._tool_section_start_token_id\n"
            "            ):\n"
            "                end_idx = i\n"
            "                break\n"
            "        ids = list(token_ids)\n"
            "        region = ids[:end_idx] if end_idx is not None else ids\n"
            "        return sum(1 for t in region if t != self._start_token_id)\n\n"
        )
        marker = "    def extract_reasoning(\n"
        if marker in src:
            fp.write_text(src.replace(marker, method + marker, 1))
            print("Patched: added count_reasoning_tokens override")
        else:
            print("extract_reasoning marker not found (OK, different vLLM)")
PYPATCH_REASONING_COUNT

# Patch: surface reasoning_tokens on /v1/chat/completions usage. vLLM 0.21 only
# reports it on /v1/responses; chat/completions has no completion_tokens_details
# at all. Add the type+field to UsageInfo and populate it (via the reasoning
# parser's count_reasoning_tokens) in both the non-streaming and streaming paths.
echo "Patching chat/completions usage for reasoning_tokens (protocol + serving)..."
python3 << 'PYPATCH_CHAT_REASONING'
from pathlib import Path
proto = Path('vllm/entrypoints/openai/engine/protocol.py')
if not proto.exists():
    print(f"{proto} not found (OK, different vLLM layout)")
else:
    p = proto.read_text()
    if 'class CompletionTokenUsageInfo' in p:
        print("protocol already patched (OK)")
    else:
        old = (
            "class UsageInfo(OpenAIBaseModel):\n"
            "    prompt_tokens: int = 0\n"
            "    total_tokens: int = 0\n"
            "    completion_tokens: int | None = 0\n"
            "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
        )
        new = (
            "class CompletionTokenUsageInfo(OpenAIBaseModel):\n"
            "    reasoning_tokens: int | None = None\n\n\n"
            "class UsageInfo(OpenAIBaseModel):\n"
            "    prompt_tokens: int = 0\n"
            "    total_tokens: int = 0\n"
            "    completion_tokens: int | None = 0\n"
            "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
            "    completion_tokens_details: CompletionTokenUsageInfo | None = None\n"
        )
        if old in p:
            proto.write_text(p.replace(old, new, 1))
            print("protocol patched")
        else:
            print("protocol UsageInfo block not found (OK, different vLLM)")

serv = Path('vllm/entrypoints/openai/chat_completion/serving.py')
if not serv.exists():
    print(f"{serv} not found (OK, different vLLM layout)")
else:
    s = serv.read_text()
    orig = s
    if 'CompletionTokenUsageInfo' not in s and '    PromptTokenUsageInfo,\n' in s:
        s = s.replace('    PromptTokenUsageInfo,\n',
                      '    CompletionTokenUsageInfo,\n    PromptTokenUsageInfo,\n', 1)
    ns_old = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
        "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    )
    ns_new = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
        "        if reasoning_parser is not None:\n"
        "            _reasoning_toks = sum(\n"
        "                reasoning_parser.count_reasoning_tokens(output.token_ids)\n"
        "                for output in final_res.outputs\n"
        "            )\n"
        "            if _reasoning_toks:\n"
        "                usage.completion_tokens_details = CompletionTokenUsageInfo(\n"
        "                    reasoning_tokens=_reasoning_toks\n"
        "                )\n"
        "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    )
    if 'usage.completion_tokens_details' not in s and ns_old in s:
        s = s.replace(ns_old, ns_new, 1)
    st_old = (
        "                final_usage = UsageInfo(\n"
        "                    prompt_tokens=num_prompt_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                )\n"
        "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    )
    st_new = (
        "                final_usage = UsageInfo(\n"
        "                    prompt_tokens=num_prompt_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                )\n"
        "                if (\n"
        "                    reasoning_parser is not None\n"
        "                    and all_previous_token_ids is not None\n"
        "                ):\n"
        "                    _reasoning_toks = sum(\n"
        "                        reasoning_parser.count_reasoning_tokens(ids)\n"
        "                        for ids in all_previous_token_ids\n"
        "                    )\n"
        "                    if _reasoning_toks:\n"
        "                        final_usage.completion_tokens_details = (\n"
        "                            CompletionTokenUsageInfo(\n"
        "                                reasoning_tokens=_reasoning_toks\n"
        "                            )\n"
        "                        )\n"
        "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    )
    if 'final_usage.completion_tokens_details' not in s and st_old in s:
        s = s.replace(st_old, st_new, 1)
    if s != orig:
        serv.write_text(s)
        print("serving patched")
    else:
        print("serving already patched or markers not found (OK)")
PYPATCH_CHAT_REASONING

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
    depyf \
    cloudpickle \
    partial-json-parser \
    openai>=1.0 \
    aiohttp \
    einops \
    protobuf \
    "ray[default]" \
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
    lark==1.2.2
# Do NOT mask failures here (previously this ended with '2>&1 | tail -30 || true',
# which swallowed a fatal ResolutionImpossible and shipped a container missing
# ray/transformers/compressed-tensors — it still passed 'import vllm' but could
# not serve). set -euo pipefail is active, so a resolver failure now aborts the
# build loudly. See the NGC-26.05 incident (2026-06-14).

# compressed-tensors must be installed --no-deps: it requires torch>=2.10.0, and
# pip refuses to match that against NGC's *local prerelease* torch
# (e.g. 2.12.0a0+...nv26.05) under PEP 440 prerelease rules — so leaving it in
# the resolved batch above makes the whole batch ResolutionImpossible on newer
# NGC bases. torch is already in the container, so --no-deps installs it cleanly
# (vLLM/flashinfer/flash-attn are installed the same way for the same reason).
echo ""
echo "Installing compressed-tensors (--no-deps to preserve NGC PyTorch)..."
pip install --no-cache-dir --root-user-action=ignore --no-deps "compressed-tensors>=0.8.0"

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

# Install DeepGEMM - required by GLM-5.1's DeepSeek Sparse Attention (DSA)
# indexer and by the FP8 MoE kernel on Hopper. vLLM imports this at
# model-init time when the architecture is GlmMoeDsaForCausalLM; without it
# the engine refuses to start with:
#   RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.
#
# DeepGEMM JIT-compiles its kernels at first use, so install itself is fast
# (no CUDA compilation here). --no-deps keeps NGC PyTorch intact; failure is
# tolerated so non-DSA presets (glm47, devstral, llama, qwen) still build.
#
# Pinned commit: 59f2c07 (2025-09-29, "Add SM100 kernels"). Rationale:
# commit 38f8ef7 (2025-11-21) introduced an API break that now requires a
# 2D `context_lens` tensor in fp8_mqa_logits(), but vLLM v0.19.0's wrapper
# at `vllm/utils/deep_gemm.py` still passes a 1D `[B]` tensor. Newer
# DeepGEMM versions (including main at the time of writing) fail every
# request with:
#   RuntimeError: Assertion error (csrc/apis/attention.hpp:195): context_lens.dim() == 2
# 59f2c07 is the last commit that touches attention.hpp before that API
# change landed, so it matches vLLM v0.19.0's call convention.
echo ""
DEEPGEMM_REF="${DEEPGEMM_REF:-59f2c07}"
echo "Installing DeepGEMM @ ${DEEPGEMM_REF} (required for GLM-5.1 DSA indexer and FP8 MoE)..."
pip install --no-cache-dir --no-deps --root-user-action=ignore --no-build-isolation \
    "git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEPGEMM_REF}" 2>&1 | tail -30 || {
    echo "Warning: DeepGEMM install failed. GLM-5.1 (DSA) will not be able to load;"
    echo "         other presets are unaffected. Override DEEPGEMM_REF to pin a commit."
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
assert any(m in torch.__version__ for m in ('nv24', 'nv25', 'nv26', 'nv27')), \
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

# Verify DeepGEMM (GLM-5.1 DSA dependency). Non-fatal — if a preset doesn't
# need DSA, this being missing is fine.
echo ""
echo "Verifying DeepGEMM..."
python3 -c "
try:
    import deep_gemm
    print(f'DeepGEMM: OK ({getattr(deep_gemm, \"__version__\", \"unknown\")})')
except ImportError as e:
    print(f'DeepGEMM: NOT INSTALLED ({e})')
    print('  GLM-5.1 will not be able to load DSA indexer; other presets unaffected.')
" || true

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
if not any(m in torch.__version__ for m in ('nv24', 'nv25', 'nv26', 'nv27')):
    print(f"\n⚠ WARNING: This may not be NGC PyTorch!")
    print(f"  Expected 'nv24'..'nv27' marker in version string")
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
# Atomic sandbox swap (only when we built into a .new.JOBID temp path)
# -----------------------------------------------------------------------------
# If SANDBOX_PATH points somewhere other than FINAL_SANDBOX_PATH we built into
# a temp location (OVERWRITE=1 with an existing sandbox present). Move the
# previous sandbox aside as a rollback copy and promote the new one into
# place. The pair of `mv`s is as atomic as the filesystem allows: there is no
# window where FINAL_SANDBOX_PATH is missing. If the first mv succeeds but
# the second fails (e.g. ENOSPC), the .prev.TIMESTAMP copy is still a valid
# container — just under a different name.
if [[ "${SANDBOX_PATH}" != "${FINAL_SANDBOX_PATH}" ]]; then
    PREV_SUFFIX="prev.$(date +%Y%m%d-%H%M%S)"
    PREV_PATH="${FINAL_SANDBOX_PATH}.${PREV_SUFFIX}"
    echo ""
    echo "=============================================="
    echo "Swapping sandbox into place"
    echo "=============================================="
    echo "  Preserving old sandbox as: ${PREV_PATH}"
    mv "${FINAL_SANDBOX_PATH}" "${PREV_PATH}"
    echo "  Promoting new sandbox:     ${SANDBOX_PATH} -> ${FINAL_SANDBOX_PATH}"
    mv "${SANDBOX_PATH}" "${FINAL_SANDBOX_PATH}"
    SANDBOX_PATH="${FINAL_SANDBOX_PATH}"
    echo ""
    echo "Rollback: rm -rf '${FINAL_SANDBOX_PATH}' && mv '${PREV_PATH}' '${FINAL_SANDBOX_PATH}'"
    echo "Once confirmed working, delete the backup with:"
    echo "  rm -rf '${PREV_PATH}'"
    echo ""
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