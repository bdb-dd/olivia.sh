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
    echo "  glm52      - GLM-5.2 (744B, 40B active) MoE+DSA, successor to GLM-5.1"
    echo "               vLLM: main + PR#45895 (auto-grafted), transformers>=5.4.0"
    echo "               FP8 ~755GB, 12 GPUs (3 nodes × 4× GH200, TP=4 + PP=3)"
    echo "               NEW skip-topk DSA indexer (index_topk_freq/skip_topk_offset)"
    echo "               needs PR#45895 — NOT in any release (incl. v0.23.0); the build"
    echo "               git-applies it (VLLM_PATCHES=45895). Drop it once PR merges."
    echo "               Same multi-node PP decode wedge as glm51; use proxy serialization."
    echo ""
    echo "  gemma4     - Gemma 4 (31B dense, multimodal text+image)"
    echo "               vLLM: v0.19.0, transformers>=5.5.0"
    echo "               ~20GiB @ AWQ, fits 1-2x GH200 (FP8 has known bugs)"
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

    # Default: no upstream PRs to graft. A preset sets PRESET_VLLM_PATCHES to a
    # space-separated list of vLLM PR numbers it needs that aren't in a release
    # yet (applied during build — see the VLLM_PATCHES step in Phase 3).
    PRESET_VLLM_PATCHES=""

    # Default: no preset-specific NGC base. A preset can pin one (e.g. glm52 on
    # vLLM main needs a newer torch::stable ABI than 26.03 ships). Resolved into
    # NGC_PYTORCH_TAG after this function runs.
    PRESET_NGC_TAG=""

    # Default: no preset-specific DeepGEMM ref (build falls back to 59f2c07). A
    # preset can pin a newer commit (e.g. glm52 needs fp8_fp4_mqa_logits for
    # GLM-5.2's DSA sparse-attention indexer). Resolved into DEEPGEMM_REF below.
    PRESET_DEEPGEMM_REF=""

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
        glm52|GLM52|glm-5.2|GLM-5.2)
            # GLM-5.2 reuses GLM-5.1's GlmMoeDsaForCausalLM arch (shipped since
            # ~v0.19.0), BUT adds a new periodic/skip-topk DSA indexer
            # (index_topk_freq=4, index_skip_topk_offset=3) that GLM-5.1 lacks.
            # That path is fixed by upstream PR#45895 ("Indexer init skip and MTP
            # TopK share for iteration"), created 2026-06-17 and NOT yet merged —
            # so it is in NO tagged release (v0.23.0 was cut two days before it).
            #
            # We graft PR#45895 onto vLLM main at build time via PRESET_VLLM_PATCHES
            # below (the Phase 3 VLLM_PATCHES step git-applies it to the cloned
            # source before compiling; it's pure Python, 9 files). The apply is
            # idempotent — once PR#45895 merges into main, drop "45895" here (and
            # the snapshot in patches/). A newer vLLM main may need
            # NGC_PYTORCH_TAG=26.04-py3+.
            #
            # Quant is block-FP8 (zai-org / RedHatAI, [128,128], e4m3, dynamic) →
            # DeepGEMM path on Hopper. Default model = RedHatAI/GLM-5.2-FP8.
            MODEL_ID="glm52"
            PRESET_VLLM_VERSION="main"
            PRESET_TRANSFORMERS=">=5.4.0"
            PRESET_VLLM_PATCHES="45895"
            # GLM-5.2's DSA sparse-attention indexer calls fp8_fp4_mqa_logits at
            # decode (sparse_attn_indexer.py); the default pin 59f2c07 (Sep 2025)
            # predates it → "DeepGEMM backend not available or outdated" RuntimeError
            # that kills the engine on the first request. 88965b0781 (2026-06-01)
            # has it (FP4 Indexer landed 7f2a703e, 2026-04-17) and matches vLLM main.
            PRESET_DEEPGEMM_REF="88965b0781"
            # vLLM main's csrc/libtorch_stable (cuda_view.cu) uses torch::stable
            # APIs (Tensor::layout(), newer from_blob) absent from NGC 26.03's
            # torch 2.11.0a0 alpha — the build fails at the CUDA compile. 26.05
            # (newest as of 2026-06) ships a later 2.11.0 alpha with that ABI.
            PRESET_NGC_TAG="26.05-py3"
            PRESET_NOTES="GLM-5.2 (744B MoE+DSA) FP8. Builds vLLM main + PR#45895 (skip-topk indexer, grafted via VLLM_PATCHES); not in any release. Multi-node PP wedge as glm51 — pair with proxy serialization."
            ;;
        gemma4|Gemma4|gemma-4|Gemma-4)
            MODEL_ID="gemma4"
            # Pinned to v0.19.0: vLLM main after 2026-03-31 (commit 7c080dd3c,
            # PR #37503) uses torch::headeronly::CppTypeToScalarType, which
            # isn't in NGC PyTorch 25.12. v0.19.0 predates that migration and
            # is the version recommended by the QuantTrio gemma-4 AWQ model card.
            PRESET_VLLM_VERSION="v0.19.0"
            PRESET_TRANSFORMERS=">=5.5.0"
            PRESET_NOTES="Gemma 4 31B dense, multimodal text+image, AWQ recommended (FP8 broken in vLLM)"
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
# a newer vLLM, you may need NGC_PYTORCH_TAG=26.04-py3 or later. The default is
# resolved AFTER apply_preset (below), so a preset can pin its own NGC base
# (glm52 → 26.05). Explicit env > preset pin > 26.03 default.

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

# Resolve NGC base image now that the preset has run: explicit env wins, then
# the preset's pin (PRESET_NGC_TAG), then the 26.03 default.
NGC_PYTORCH_TAG="${NGC_PYTORCH_TAG:-${PRESET_NGC_TAG:-26.03-py3}}"
NGC_IMAGE="${NGC_IMAGE:-docker://nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}}"

# Resolve DeepGEMM ref: explicit env > preset pin > 59f2c07 default. Forwarded
# into the Phase 3 build container below.
DEEPGEMM_REF="${DEEPGEMM_REF:-${PRESET_DEEPGEMM_REF:-59f2c07}}"

# Upstream vLLM PRs to graft onto the cloned source during Phase 3 (space-
# separated PR numbers). Defaults to the preset's list; override with
# VLLM_PATCHES="..." or disable with VLLM_PATCHES="".
VLLM_PATCHES="${VLLM_PATCHES-${PRESET_VLLM_PATCHES}}"

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
echo "  vLLM patches:   ${VLLM_PATCHES:-<none>}"
echo "  DeepGEMM ref:   ${DEEPGEMM_REF}"
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
export VLLM_PATCHES="${VLLM_PATCHES}"
export DEEPGEMM_REF="${DEEPGEMM_REF}"

# Reproducible PR graft (#2): if a snapshot dir was deployed alongside this script
# (olivia.sh deploys ./patches/), bind it read-only so the VLLM_PATCHES step in
# Phase 3 prefers the committed snapshot over a live GitHub fetch. Absent dir =
# the build falls back to the live fetch (so this is purely additive).
PATCHES_BIND=""
PATCHES_DIR="${PATCHES_DIR:-${CONTAINER_DIR:-$(pwd)}/patches}"
if [[ -d "${PATCHES_DIR}" ]]; then
    PATCHES_BIND="--bind ${PATCHES_DIR}:/opt/olivia-patches:ro"
    echo "PR-graft snapshots: ${PATCHES_DIR} -> /opt/olivia-patches (ro)"
fi

singularity exec ${SING_OPTS} ${PATCHES_BIND} \
    --env "VLLM_VERSION=${VLLM_VERSION}" \
    --env "VLLM_PATCHES=${VLLM_PATCHES}" \
    --env "DEEPGEMM_REF=${DEEPGEMM_REF}" \
    --bind "${PIP_CACHE}:/root/.cache/pip" \
    "${SANDBOX_PATH}" /bin/bash << 'BUILDSCRIPT'

set -euo pipefail

echo "Build environment:"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo "  PWD: $(pwd)"

echo "Installing build dependencies..."
# setuptools-rust is required to even generate metadata for vLLM main's
# pyproject (it ships a Rust frontend). We build with --no-build-isolation, so
# every build-system.requires entry must be present in the env up front.
# Harmless for older pinned versions that don't use it.
pip install --no-cache-dir --root-user-action=ignore ninja cmake wheel packaging setuptools-scm setuptools-rust

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

# -----------------------------------------------------------------------------
# Graft requested upstream vLLM PRs (patch-during-build)
# -----------------------------------------------------------------------------
# VLLM_PATCHES (space-separated PR numbers, passed via --env) lists upstream vLLM
# PRs a preset needs that aren't in a release yet. Each PR's cumulative diff is
# git-applied to the cloned source HERE — before the `pip install .` compile
# below — so the fix is baked into the container. The glm52 preset uses this for
# PR#45895 (GLM-5.2's new skip-topk DSA indexer + MTP final-norm recycle; pure
# Python). cwd is /opt/vllm.
#
# Source of each diff (#2, reproducibility): a committed snapshot bound in at
# /opt/olivia-patches (patches/vllm-pr<N>*.diff) is PREFERRED — reproducible and
# offline. Only if no snapshot is present do we fetch the LIVE PR from GitHub
# (which can drift if the PR is force-updated/rebased). Drop the snapshot file
# (or the PR number) once the PR merges into the pinned ref.
#
# Idempotent: a reverse-apply check skips PRs already present (e.g. once the PR
# merges into the pinned ref, or on a --force rebuild). A diff that no longer
# applies fails the build loudly rather than compiling a half-patched tree.
if [[ -n "${VLLM_PATCHES:-}" ]]; then
    echo ""
    echo "Grafting upstream vLLM PR(s): ${VLLM_PATCHES}"
    for PR in ${VLLM_PATCHES}; do
        DIFF="/tmp/vllm-pr-${PR}.diff"
        # Prefer a committed local snapshot (reproducible + offline); fall back to
        # the live GitHub PR diff only if no snapshot was bound in.
        SNAP="$(ls /opt/olivia-patches/vllm-pr${PR}*.diff 2>/dev/null | head -1 || true)"
        if [[ -n "${SNAP}" ]]; then
            echo "  PR #${PR}: using committed snapshot $(basename "${SNAP}")"
            cp "${SNAP}" "${DIFF}"
        else
            echo "  PR #${PR}: no local snapshot bound — fetching live from GitHub..."
            if ! command -v curl >/dev/null 2>&1; then
                echo "  ERROR: curl not found in container; cannot fetch PR diffs."
                exit 1
            fi
            if ! curl -fsSL "https://github.com/vllm-project/vllm/pull/${PR}.diff" -o "${DIFF}"; then
                echo "  ERROR: failed to download PR #${PR} diff."
                exit 1
            fi
        fi
        if ! grep -q '^diff --git ' "${DIFF}"; then
            echo "  ERROR: PR #${PR} download is not a valid diff ($(wc -c < "${DIFF}") bytes)."
            exit 1
        fi
        N_FILES=$(grep -c '^diff --git ' "${DIFF}")
        if git apply --reverse --check "${DIFF}" 2>/dev/null; then
            echo "  PR #${PR}: already present (merged or re-run) — skipping (${N_FILES} files)"
        elif git apply --check "${DIFF}" 2>/dev/null; then
            git apply "${DIFF}"
            echo "  PR #${PR}: applied cleanly (${N_FILES} files)"
        else
            echo ""
            echo "  ERROR: PR #${PR} does not apply cleanly to vLLM ${VLLM_VERSION}."
            echo "         Upstream main likely drifted, or the PR was updated/merged"
            echo "         with changes. Writing .rej files under /opt/vllm for inspection..."
            git apply --reject "${DIFF}" || true
            echo "         Inspect: https://github.com/vllm-project/vllm/pull/${PR}"
            echo "         If the PR has merged into main, rebuild with VLLM_PATCHES=\"\"."
            exit 1
        fi
    done
    echo ""
fi

# -----------------------------------------------------------------------------
# Local patch: surface reasoning_tokens on /v1/chat/completions usage
# -----------------------------------------------------------------------------
# vLLM reports reasoning_tokens only on /v1/responses; chat/completions has no
# completion_tokens_details at all, and neither the kimi_k2 parser nor the glm45
# parser (DeepSeekV3ReasoningWithThinkingParser) counts reasoning tokens — the
# base ReasoningParser.count_reasoning_tokens() returns 0, and both families emit
# the reasoning-end token but omit the <think> START token (it lives in the
# prompt), so an inherited start/end depth counter is 0 too. This one block:
#   (1) adds count_reasoning_tokens to BOTH parsers (kimi_k2 + DeepSeekV3),
#   (2) adds CompletionTokenUsageInfo + UsageInfo.completion_tokens_details to the
#       protocol (once), and
#   (3) populates it in the non-streaming + streaming chat usage paths, tolerant
#       of BOTH vLLM 0.21 (var `reasoning_parser`, `all_previous_token_ids`) and
#       vLLM main (var `parser`/`parsers`, own `previous_token_ids` accumulator).
# Every edit is idempotent + defensive (a missing anchor logs and no-ops, so a
# future vLLM refactor never fails the build or half-patches the tree). For the
# serving paths the 0.21-specific anchors (which carry the trailing
# enable_prompt_tokens_details line) are tried FIRST and the bare main anchors
# SECOND under a `*.completion_tokens_details not in s` guard, so the main-only
# `parser` variable is never injected into 0.21's serving; the main streaming
# path is additionally gated on the 0.21 path not having applied so a 0.21 build
# gets no dead accumulator. cwd /opt/vllm.
echo ""
echo "Patching vLLM for reasoning_tokens on chat/completions usage (kimi + GLM)..."
python3 << 'PYPATCH_REASONING_TOKENS'
from pathlib import Path

root = Path(".")


def patch(rel, fn):
    fp = root / rel
    if not fp.exists():
        print(f"  {rel}: not found (OK, different vLLM layout) -- skipping")
        return
    src = fp.read_text()
    out = fn(src)
    if out is None or out == src:
        return
    fp.write_text(out)


# ---- parser A: kimi_k2 (Kimi K2.x) count_reasoning_tokens ----
def patch_kimi_parser(s):
    if "def count_reasoning_tokens" in s:
        print("  kimi parser: already patched (OK)")
        return s
    marker = "    def extract_reasoning(\n"
    if marker not in s:
        print("  kimi parser: anchor not found (OK, different vLLM) -- skipping")
        return s
    method = (
        "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
        "        # Olivia patch: base default returns 0; Kimi K2 omits the <think>\n"
        "        # start token, so count tokens before the first </think>/tool-call\n"
        "        # marker, dropping a leading <think>. Mirrors extract_reasoning.\n"
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
    print("  kimi parser: patched (count_reasoning_tokens added)")
    return s.replace(marker, method + marker, 1)


# ---- parser B: GLM DeepSeekV3 count_reasoning_tokens ----
def patch_glm_parser(s):
    if "def count_reasoning_tokens" in s:
        print("  glm parser: already patched (OK)")
        return s
    anchor = (
        "    def extract_content_ids(self, input_ids: list[int]) -> list[int]:\n"
        "        return self._parser.extract_content_ids(input_ids)\n"
    )
    if anchor not in s:
        print("  glm parser: anchor not found (OK, different vLLM) -- skipping")
        return s
    method = (
        "\n"
        "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
        "        # Olivia patch: base count_reasoning_tokens returns 0 and the V3\n"
        "        # wrapper does not forward it; GLM-5.x emits </think> but omits\n"
        "        # the <think> start (it is in the prompt), so the inner parser's\n"
        "        # start/end depth counter also yields 0. Count tokens before the\n"
        "        # first reasoning-end token, dropping a leading start token.\n"
        "        parser = self._parser\n"
        '        end_id = getattr(parser, "end_token_id", None)\n'
        "        if end_id is None:\n"
        "            return 0\n"
        '        start_id = getattr(parser, "start_token_id", None)\n'
        "        end_idx = None\n"
        "        for i, t in enumerate(token_ids):\n"
        "            if t == end_id:\n"
        "                end_idx = i\n"
        "                break\n"
        "        ids = list(token_ids)\n"
        "        region = ids[:end_idx] if end_idx is not None else ids\n"
        "        return sum(1 for t in region if t != start_id)\n"
    )
    print("  glm parser: patched (count_reasoning_tokens added)")
    return s.replace(anchor, anchor + method, 1)


# ---- protocol: CompletionTokenUsageInfo + UsageInfo field (shared) ----
def patch_protocol(s):
    if "class CompletionTokenUsageInfo" in s:
        print("  protocol: already patched (OK)")
        return s
    old = (
        "class UsageInfo(OpenAIBaseModel):\n"
        "    prompt_tokens: int = 0\n"
        "    total_tokens: int = 0\n"
        "    completion_tokens: int | None = 0\n"
        "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
    )
    if old not in s:
        print("  protocol: UsageInfo block not found (OK, different vLLM) -- skipping")
        return s
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
    print("  protocol: patched (CompletionTokenUsageInfo + field)")
    return s.replace(old, new, 1)


# ---- serving: import + non-streaming + streaming, tolerant of 0.21 AND main ----
def patch_serving(s):
    # import CompletionTokenUsageInfo (shared by both eras)
    if "CompletionTokenUsageInfo" not in s and "    PromptTokenUsageInfo,\n" in s:
        s = s.replace(
            "    PromptTokenUsageInfo,\n",
            "    CompletionTokenUsageInfo,\n    PromptTokenUsageInfo,\n", 1,
        )
        print("  serving: import added")

    # non-streaming: vLLM 0.21 (specific anchor + `reasoning_parser`) first ...
    ns021_old = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
        "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    )
    ns021_new = (
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
    if "usage.completion_tokens_details" not in s and ns021_old in s:
        s = s.replace(ns021_old, ns021_new, 1)
        print("  serving: non-streaming (vLLM 0.21) populated")
    # ... then vLLM main (bare anchor + `parser.reasoning_parser`), guarded.
    nsmain_old = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
    )
    nsmain_new = nsmain_old + (
        "        if parser is not None and parser.reasoning_parser is not None:\n"
        "            _reasoning_toks = sum(\n"
        "                parser.reasoning_parser.count_reasoning_tokens(output.token_ids)\n"
        "                for output in final_res.outputs\n"
        "            )\n"
        "            if _reasoning_toks:\n"
        "                usage.completion_tokens_details = CompletionTokenUsageInfo(\n"
        "                    reasoning_tokens=_reasoning_toks\n"
        "                )\n"
    )
    if "usage.completion_tokens_details" not in s and nsmain_old in s:
        s = s.replace(nsmain_old, nsmain_new, 1)
        print("  serving: non-streaming (vLLM main) populated")
    if "usage.completion_tokens_details" not in s:
        print("  serving: non-streaming anchor not found (OK, different vLLM)")

    # streaming: vLLM 0.21 first (uses pre-existing all_previous_token_ids) ...
    st021_old = (
        "                final_usage = UsageInfo(\n"
        "                    prompt_tokens=num_prompt_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                )\n"
        "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    )
    st021_new = (
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
    if "final_usage.completion_tokens_details" not in s and st021_old in s:
        s = s.replace(st021_old, st021_new, 1)
        print("  serving: streaming (vLLM 0.21) populated")
    # ... then vLLM main, ONLY if 0.21 didn't apply (no dead accumulator on 0.21).
    if "final_usage.completion_tokens_details" not in s:
        acc_old = '        previous_texts = [""] * num_choices\n'
        if "previous_token_ids" not in s and acc_old in s:
            s = s.replace(
                acc_old,
                acc_old
                + "        previous_token_ids: list[list[int]] = [[] for _ in range(num_choices)]\n",
                1,
            )
        app_old = "                    previous_num_tokens[i] += len(output.token_ids)\n"
        if "previous_token_ids[i].extend" not in s and app_old in s:
            s = s.replace(
                app_old,
                app_old
                + "                    previous_token_ids[i].extend(as_list(output.token_ids))\n",
                1,
            )
        stmain_old = (
            "                final_usage = UsageInfo(\n"
            "                    prompt_tokens=num_prompt_tokens,\n"
            "                    completion_tokens=completion_tokens,\n"
            "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
            "                )\n"
        )
        stmain_new = stmain_old + (
            "                _reasoning_toks = 0\n"
            "                for _i in range(num_choices):\n"
            "                    _p = parsers[_i] if _i < len(parsers) else None\n"
            "                    if _p is not None and _p.reasoning_parser is not None:\n"
            "                        _reasoning_toks += (\n"
            "                            _p.reasoning_parser.count_reasoning_tokens(\n"
            "                                previous_token_ids[_i]\n"
            "                            )\n"
            "                        )\n"
            "                if _reasoning_toks:\n"
            "                    final_usage.completion_tokens_details = (\n"
            "                        CompletionTokenUsageInfo(reasoning_tokens=_reasoning_toks)\n"
            "                    )\n"
        )
        if stmain_old in s:
            s = s.replace(stmain_old, stmain_new, 1)
            print("  serving: streaming (vLLM main) populated")
    if "final_usage.completion_tokens_details" not in s:
        print("  serving: streaming anchor not found (OK, different vLLM)")

    return s


patch("vllm/reasoning/kimi_k2_reasoning_parser.py", patch_kimi_parser)
patch("vllm/reasoning/deepseek_v3_reasoning_parser.py", patch_glm_parser)
patch("vllm/entrypoints/openai/engine/protocol.py", patch_protocol)
patch("vllm/entrypoints/openai/chat_completion/serving.py", patch_serving)
print("reasoning_tokens patch done.")
PYPATCH_REASONING_TOKENS

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
    uvloop \
    loguru \
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

# compressed-tensors and xgrammar are torch-dependent, so they live OUTSIDE the
# bulk install above. On newer NGC bases (e.g. 26.05's torch 2.12.0a0 pre-
# release) pip's resolver can't match their torch>=2.10 / torch<2.11 metadata
# against the pinned pre-release version, which fails the ENTIRE resolve — if
# they were in the bulk list that would silently take fastapi/uvloop/etc. down
# with them (exactly the glm52-on-26.05 failure). Install with deps first (works
# on 26.03, preserving glm51/kimi), then fall back to --no-deps. Their non-torch
# deps are supplied explicitly: loguru (compressed-tensors) is in the bulk list;
# apache-tvm-ffi (xgrammar's tvm_ffi backend) is installed just below.
echo ""
echo "Installing compressed-tensors..."
pip install --no-cache-dir --root-user-action=ignore --constraint /tmp/constraints.txt "compressed-tensors>=0.8.0" 2>&1 | tail -5 \
    || pip install --no-cache-dir --root-user-action=ignore --no-deps "compressed-tensors>=0.8.0" 2>&1 | tail -5 \
    || echo "compressed-tensors not available, continuing..."

echo ""
echo "Installing xgrammar (+ apache-tvm-ffi backend)..."
pip install --no-cache-dir --root-user-action=ignore --constraint /tmp/constraints.txt xgrammar 2>&1 | tail -5 \
    || pip install --no-cache-dir --root-user-action=ignore --no-deps xgrammar 2>&1 | tail -5 \
    || echo "xgrammar not available, continuing..."
# tvm_ffi backend for xgrammar (no torch dep → installs cleanly with deps).
# Needed when xgrammar went in via --no-deps; a harmless no-op otherwise.
pip install --no-cache-dir --root-user-action=ignore apache-tvm-ffi 2>&1 | tail -3 || echo "apache-tvm-ffi not available, continuing..."

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

# vLLM main (post-v0.20) ships a Rust frontend under vllm/vllm-rs (tokenizer,
# tool/reasoning parsers, incl. the deepseek_v32 renderer used by DSA models).
# Its pyproject build needs an actual Rust toolchain (cargo/rustc) in addition
# to setuptools-rust, and NGC ships neither. Install rustup ONLY when the cloned
# source actually has the Rust frontend, so older pinned versions (e.g. glm51's
# v0.19.0, which predates it) build exactly as before. cwd is /opt/vllm here.
if [[ -f rust-toolchain.toml || -d rust ]]; then
    echo ""
    echo "vLLM source has a Rust frontend — installing Rust toolchain (rustup)..."
    export RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo
    RUST_CHANNEL=$(grep -oE 'channel[[:space:]]*=[[:space:]]*"[^"]+"' rust-toolchain.toml 2>/dev/null | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
    RUST_CHANNEL="${RUST_CHANNEL:-stable}"
    echo "  Pinned Rust channel: ${RUST_CHANNEL}"
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_CHANNEL}"
    fi
    export PATH="${CARGO_HOME}/bin:${PATH}"
    if ! command -v cargo >/dev/null 2>&1; then
        echo "ERROR: cargo not on PATH after rustup install; cannot build vLLM Rust frontend."
        exit 1
    fi
    echo "  Rust toolchain ready: $(cargo --version)"
fi

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

# Check the OpenAI serving stack. `import vllm` succeeding is NOT enough — the
# api_server pulls runtime deps (uvloop, fastapi, compressed_tensors, xgrammar)
# that --no-deps / a poisoned resolver can silently drop, yielding a container
# that builds but can't `vllm serve`. Fail the build loudly instead of shipping
# that (this is exactly how the glm52-on-26.05 dep break slipped through once).
try:
    import vllm.entrypoints.openai.api_server  # noqa: F401  (pulls uvloop+fastapi)
    import compressed_tensors  # noqa: F401
    import xgrammar  # noqa: F401
    print("Serving stack (api_server + compressed_tensors + xgrammar): ✓")
except Exception as e:
    print(f"\n✗ FATAL: serving-stack import failed: {e!r}")
    print("  Container builds but cannot serve — check the dependency-install phase.")
    sys.exit(1)

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
