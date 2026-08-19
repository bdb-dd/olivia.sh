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
    echo "  glm53_v27  - GLM-5.3 (744B, 40B active) on a TAGGED vLLM release"
    echo "               vLLM: v0.27.1, transformers>=5.5.3, NGC 26.07 (torch 2.13)"
    echo "               FP8, 12 GPUs (3 nodes × 4× GH200, TP=4 + PP=3). Builds its"
    echo "               OWN container (vllm-glm53-1) — will not touch glm52's."
    echo "               GLM-5.3 == GLM-5.2's base re-post-trained, so no new model"
    echo "               support is needed: the plain 'glm53' SERVE preset reuses the"
    echo "               glm52 container and needs NO build. This preset exists only"
    echo "               to retire the pinned-main + PR#45895 graft (PR merged in"
    echo "               v0.24.0) and pick up the DSA/parser work through v0.27.1."
    echo "               UNVALIDATED — build against GLM-5.2-FP8 weights first."
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
    echo "  laguna     - Laguna M.1 (Poolside, 225B / 23B active) MoE coding model"
    echo "               vLLM: v0.21.0 (native Laguna), transformers>=5.7.0"
    echo "               FP8 ~225GB, single node (4× GH200, TP=4); poolside_v1 parsers"
    echo ""
    echo "  ornith     - Ornith 1.0 MoE (Deep Reinforce, Qwen3.5) — one container, two sizes"
    echo "               vLLM: main (qwen3_5_moe arch + MTP), transformers>=5.8.1, NGC 26.05"
    echo "               hybrid attn + multimodal, 256K ctx, FP8 (W8A8). Serve either:"
    echo "                 preset 'ornith'       → 397B flagship, 2 nodes × 4 (TP=4 + PP=2)"
    echo "                 preset 'ornith_gh200' → 35B (~3B active), single GH200 card (TP=1)"
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

    # Default: no preset-specific DeepGEMM *repo* (build falls back to the
    # upstream deepseek-ai one). vLLM moved its own pin to the vllm-project
    # DeepGEMM fork, so a preset tracking a modern vLLM release (glm53_v27) must
    # point here as well as at a ref. Resolved into DEEPGEMM_REPO below.
    PRESET_DEEPGEMM_REPO=""

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
            # We graft PR#45895 via PRESET_VLLM_PATCHES below (the Phase 3
            # VLLM_PATCHES step applies the committed patches/ snapshot to the
            # cloned source before compiling; pure Python, 9 files). For
            # reproducibility the base is PINNED to the exact main commit the
            # deployed glm52 container built from and the snapshot was validated
            # against (091386a, 2026-06-17), so clone(pinned) + apply(snapshot) is
            # byte-identical every build. Override VLLM_VERSION=main to track
            # latest (then re-pin + re-snapshot once validated). Once PR#45895
            # merges at/under the pin, drop "45895" + the snapshot. A newer base
            # may need NGC_PYTORCH_TAG=26.04-py3+.
            #
            # Quant is block-FP8 (zai-org / RedHatAI, [128,128], e4m3, dynamic) →
            # DeepGEMM path on Hopper. Default model = RedHatAI/GLM-5.2-FP8.
            MODEL_ID="glm52"
            PRESET_VLLM_VERSION="091386a99b9542691bb1e935ca44d0efbba6e111"
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
        glm53|GLM53|glm-5.3|GLM-5.3|glm53_v27|GLM53_V27)
            # GLM-5.3 (Z.ai, announced 2026-08-14) is NOT a new architecture and
            # NOT a new pretrain: it is GLM-5.2's *same 744B base* re-post-trained
            # (Z.ai's own framing — "every reported gain comes from scaled
            # post-training"). Same GlmMoeDsaForCausalLM, same ~40B active, same
            # skip-topk DSA indexer, same 1M context. So nothing about it needs
            # new *model* support in vLLM — the `glm53` runtime preset therefore
            # REUSES the already-validated glm52 container (presets.json points it
            # at container_prefix glm52, index 1) and needs no build at all.
            #
            # THIS build preset is the separate upgrade path (`glm53_v27`, its own
            # vllm-glm53-1-sandbox, so it cannot clobber the working glm52 one).
            # What it buys over glm52's build:
            #   - PR#45895 (the skip-topk DSA indexer) MERGED upstream 2026-06-19
            #     (ab666069) and is an ancestor of every release from v0.24.0 on,
            #     verified against v0.27.1. So the VLLM_PATCHES graft and the
            #     unreleased-main-commit pin both go away: we build a TAG.
            #   - v0.24.0 added the streaming parser engine for GLM-4.7/5.1/5.2
            #     (better tool-call streaming for Claude Code via anthropic_proxy).
            #   - v0.27.0 added "skip sparse indexer scoring for short dense
            #     prefills" (#48407) — straight on the GLM-5.x DSA hot path — plus
            #     Quark GLM-5.2 checkpoint inference fixes (#48886).
            # v0.27.0 is a BREAKING environment change: it moves to torch 2.13.0 +
            # Triton 3.7.1. NGC 26.05 (glm52's base) ships torch 2.12.0a0, so this
            # preset moves to NGC 26.07 (torch 2.13.0a0 + CUDA 13.3.1) — the newest
            # tag as of 2026-08-17 and the one that actually matches the pin, which
            # matters because our --no-deps strategy keeps NGC's torch, not pip's.
            #
            # DeepGEMM: vLLM no longer pins the deepseek-ai repo — v0.27.1 pins the
            # vllm-project/DeepGEMM fork at e21c821f (tools/install_deepgemm.sh,
            # kept in sync with cmake/external_projects/deepgemm.cmake). GLM-5.x's
            # DSA indexer calls into it (fp8_fp4_mqa_logits), so match vLLM's pin
            # exactly rather than guessing a deepseek-ai commit.
            #
            # UNVALIDATED as of 2026-08-17: not yet built or served on Olivia, and
            # GLM-5.3's weights are not public yet (Z.ai promised them ~2 weeks
            # after launch). Build it against GLM-5.2-FP8 — already in the work-tier
            # cache — to validate the whole toolchain BEFORE 5.3 weights land; that
            # is the cheapest way to de-risk this. If it regresses, `glm53` on the
            # glm52 container is the fallback and needs no rebuild.
            MODEL_ID="glm53"
            PRESET_VLLM_VERSION="v0.27.1"
            # v0.27.1's requirements/common.txt asks for transformers>=5.5.3, which
            # also satisfies GLM-5.x's own >=5.4.0 floor.
            PRESET_TRANSFORMERS=">=5.5.3"
            PRESET_VLLM_PATCHES=""
            PRESET_DEEPGEMM_REPO="https://github.com/vllm-project/DeepGEMM.git"
            PRESET_DEEPGEMM_REF="e21c821f39a2056d68067a466c64ddc942200106"
            PRESET_NGC_TAG="26.07-py3"
            PRESET_NOTES="GLM-5.3 (== GLM-5.2's base, re-post-trained) FP8 on a TAGGED vLLM: v0.27.1 + NGC 26.07 (torch 2.13) + vllm-project DeepGEMM, NO PR graft (PR#45895 merged upstream). Own container; the plain 'glm53' preset reuses the glm52 one instead. UNVALIDATED — build against GLM-5.2-FP8 first."
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
        laguna|Laguna|laguna-m1|laguna-m.1|Laguna-M.1)
            MODEL_ID="laguna"
            # Laguna M.1 (Poolside): 225B / 23B-active MoE coding model,
            # LagunaForCausalLM, natively supported in vLLM since v0.21.0
            # (PR#41129 added the Laguna arch + the poolside_v1 tool/reasoning
            # parsers). transformers >=5.7.0 is the first release to ship the
            # "laguna" model_type. Default quant is block-FP8
            # (poolside/Laguna-M.1-FP8) — auto-detected from quantization_config,
            # so no --quantization flag and no special DeepGEMM pin needed
            # (ordinary block-FP8, unlike glm52's DSA-indexer fp8_fp4_mqa_logits).
            PRESET_VLLM_VERSION="v0.21.0"
            PRESET_TRANSFORMERS=">=5.7.0"
            # NGC base: inherit the 26.03 default. The validated vLLM-0.21 Kimi
            # container builds AND serves on 26.03, so 0.21-era vLLM compiles
            # there. If v0.21.0's csrc/libtorch_stable hits the torch::stable ABI
            # wall (the glm52-on-main failure), bump NGC_PYTORCH_TAG=26.05-py3.
            PRESET_NOTES="Laguna M.1 (Poolside, 225B/23B MoE) FP8, single-node TP=4. vLLM v0.21.0 (native Laguna), transformers>=5.7.0, poolside_v1 parsers."
            ;;
        ornith*|Ornith*)
            # All ornith* aliases share ONE container (vllm-ornith-1-sandbox):
            # both sizes we serve are the SAME arch (qwen3_5_moe), so one build
            # runs both — the presets differ only in model repo / node shape at
            # serve time:
            #   ornith        → 397B MoE flagship, 2 nodes × 4 (TP=4 + PP=2)
            #   ornith_gh200  → 35B MoE, a single GH200 card (TP=1)
            MODEL_ID="ornith"
            # Ornith 1.0 (Deep Reinforce): agentic-coding model family
            # post-trained on Qwen 3.5. Both served sizes — the 35B MoE (~3B
            # active) and the 397B MoE flagship — are arch
            # Qwen3_5MoeForConditionalGeneration / model_type qwen3_5_moe, 256K
            # context. GROUND TRUTH from the 35B FP8 checkpoint (inspected
            # on-cluster 2026-07-16): it's the Qwen3-Next/3.5 lineage — HYBRID
            # attention (Gated-DeltaNet linear_attn + full self_attn), a shared
            # expert, and MULTIMODAL (vision tower). Quant is compressed-tensors
            # channel/token FP8 (W8A8, NOT DeepSeek block-FP8), auto-detected from
            # quantization_config → no --quantization flag, and DeepGEMM is NOT on
            # its GEMM path. The config declares mtp_num_hidden_layers=1 but the
            # FP8 export SHIPS NO MTP WEIGHTS, so MTP speculative decode is off by
            # default (see run_vllm_server.sh IS_ORNITH).
            #
            # vLLM version: qwen3_5_moe is a NEW architecture (hybrid + MoE +
            # multimodal). Older releases reject it ("architectures
            # ['Qwen3_5MoeForConditionalGeneration'] are not supported",
            # vllm#35344), and transformers 5.x renamed the config to
            # Qwen3_5MoeTextConfig which pre-5.x-aware vLLM can't load (vllm#36236).
            # Build from main (like glm47) to get the arch (incl. the mamba/GDN
            # kernels) + the transformers-5.x fix. The model card claims vLLM
            # >=0.19.1 but that predates the transformers-5.x rename fix, so main
            # is safer. main needs the newer torch::stable ABI → NGC 26.05 (the
            # glm52 lesson: 26.03 fails the csrc/libtorch_stable CUDA compile).
            # DeepGEMM is pinned to the main-matching ref for build compatibility
            # only (unused by Ornith's channel/token FP8). transformers >=5.8.1.
            #
            # PINNED to the exact main commit that BUILT + SERVED + benchmarked
            # ornith_gh200 on-cluster (251f7e4, 2026-07-16) — like glm52, so the
            # build is reproducible (main is a moving target: an earlier attempt
            # cloned 75bdad4 and main advanced ~1.7 h before this one succeeded).
            # Override VLLM_VERSION=main to track latest (then re-pin + re-validate).
            # If a tagged release ships qwen3_5_moe + the transformers-5.x fix,
            # switch to it and drop back to NGC 26.03 if its ABI allows.
            PRESET_VLLM_VERSION="251f7e478e8eb0c90a01eb8fff40056da2aa3ff7"
            PRESET_TRANSFORMERS=">=5.8.1"
            PRESET_NGC_TAG="26.05-py3"
            PRESET_DEEPGEMM_REF="88965b0781"
            PRESET_NOTES="Ornith 1.0 (Qwen3.5 hybrid-attn MoE, multimodal) FP8 W8A8: ornith=397B flagship 2-node TP=4+PP=2, ornith_gh200=35B single GH200 (TP=1). qwen3_xml/qwen3 parsers, MTP off (FP8 has no MTP weights). vLLM main + transformers>=5.8.1 + NGC 26.05."
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

# Resolve the DeepGEMM repo the same way. Default stays the upstream deepseek-ai
# repo (what every existing preset's pinned ref lives in); vLLM's own pin now
# lives in the vllm-project fork, so a preset on a modern vLLM release points
# there (see cmake/external_projects/deepgemm.cmake in the vLLM tree).
DEEPGEMM_REPO="${DEEPGEMM_REPO:-${PRESET_DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}}"

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
echo "  DeepGEMM ref:   ${DEEPGEMM_REF} (${DEEPGEMM_REPO})"
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
export DEEPGEMM_REPO="${DEEPGEMM_REPO}"

# Reproducible PR graft (#2): if a snapshot dir was deployed alongside this script
# (olivia.sh deploys ./patches/), bind it read-only so the VLLM_PATCHES step in
# Phase 3 prefers the committed snapshot over a live GitHub fetch. Absent dir =
# the build falls back to the live fetch (so this is purely additive).
PATCHES_BIND=""
PATCHES_DIR="${PATCHES_DIR:-${CONTAINER_DIR:-$(pwd)}/patches}"
if [[ -d "${PATCHES_DIR}" ]]; then
    # Apptainer does NOT auto-create bind destinations (unlike Docker); the bind
    # fails at container creation if the mount point is absent. The sandbox is a
    # writable dir tree on the host, so create the mount point in it first.
    mkdir -p "${SANDBOX_PATH}/opt/olivia-patches"
    PATCHES_BIND="--bind ${PATCHES_DIR}:/opt/olivia-patches:ro"
    echo "PR-graft snapshots: ${PATCHES_DIR} -> /opt/olivia-patches (ro)"
fi

singularity exec ${SING_OPTS} ${PATCHES_BIND} \
    --env "VLLM_VERSION=${VLLM_VERSION}" \
    --env "VLLM_PATCHES=${VLLM_PATCHES}" \
    --env "DEEPGEMM_REF=${DEEPGEMM_REF}" \
    --env "DEEPGEMM_REPO=${DEEPGEMM_REPO}" \
    --env "NGC_PYTORCH_TAG=${NGC_PYTORCH_TAG}" \
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
elif [[ "$VLLM_VERSION" =~ ^[0-9a-f]{7,40}$ ]]; then
    # Pinned commit SHA — full reproducibility for snapshot-grafted presets
    # (glm52): clone(pinned base) + apply(committed snapshot) is byte-identical
    # every build, unlike cloning a moving branch where the base drifts (and the
    # snapshot can stop applying). GitHub serves arbitrary commits
    # (allowAnySHA1InWant), so shallow-fetch the exact SHA. Like the "main" path
    # this is a dev build (version label <next>.dev0+g<sha>), so --depth 1 is fine
    # — the tag-history concern below only applies to tagged releases.
    echo "Pinning vLLM to commit ${VLLM_VERSION} (reproducible build)"
    mkdir vllm && cd vllm
    git init -q
    git remote add origin https://github.com/vllm-project/vllm.git
    git fetch --depth 1 origin "${VLLM_VERSION}"
    git checkout -q FETCH_HEAD
    cd /opt
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
# Stub stable-ABI get_cuda_view_from_cpu_tensor on older NGC torch (<26.05)
# -----------------------------------------------------------------------------
# vLLM-main's csrc/libtorch_stable/cuda_view.cu builds tensors with
# torch::stable::from_blob passing CAPTURING-LAMBDA deleters and reads
# Tensor::layout(). NGC 26.03/26.04 torch (2.11) only has the plain DeleterFnPtr
# from_blob overload and no Tensor::layout(), so _C_stable_libtorch fails to
# compile — the documented reason glm52 was pinned to 26.05 (see apply_preset).
# The op is NOT dispatched at runtime (vLLM calls the _C variant — see
# vllm/utils/torch_utils.py), so we stub the stable variant: the extension still
# compiles/imports, and the dead op throws a clear error if ever called. Gated
# to bases older than 26.05 (or CUDA_VIEW_STABLE_STUB=1). Idempotent; fails loud
# if the upstream anchor disappears.
if [[ "${NGC_PYTORCH_TAG}" == 26.03* || "${NGC_PYTORCH_TAG}" == 26.04* || "${CUDA_VIEW_STABLE_STUB:-0}" == "1" ]]; then
python3 << 'PYPATCH_CUDA_VIEW_STABLE'
import re, sys
f = "/opt/vllm/csrc/libtorch_stable/cuda_view.cu"
try:
    src = open(f).read()
except FileNotFoundError:
    print("PYPATCH_CUDA_VIEW_STABLE: file absent, skipping"); raise SystemExit(0)
if "stable-ABI variant unimplemented on NGC" in src:
    print("PYPATCH_CUDA_VIEW_STABLE: already applied"); raise SystemExit(0)
stub = (
    "torch::stable::Tensor get_cuda_view_from_cpu_tensor(\n"
    "    torch::stable::Tensor& cpu_tensor) {\n"
    "  // NGC<26.05 port: torch::stable::from_blob lacks the capturing-deleter\n"
    "  // overload (26.03/2.11 has only DeleterFnPtr) and Tensor::layout() is\n"
    "  // absent. Not dispatched at runtime (vLLM uses the _C op, torch_utils.py),\n"
    "  // so stub it so _C_stable_libtorch compiles/imports on this base.\n"
    "  STD_TORCH_CHECK(false, \"get_cuda_view_from_cpu_tensor: stable-ABI variant unimplemented on NGC 26.03 (use the _C op)\");\n"
    "  return cpu_tensor;\n"
    "}\n"
)
new, n = re.subn(
    r"torch::stable::Tensor get_cuda_view_from_cpu_tensor\(.*?\n\}\n?",
    stub, src, count=1, flags=re.DOTALL)
if n != 1:
    sys.stderr.write("PYPATCH_CUDA_VIEW_STABLE: ERROR anchor not found; vLLM source changed\n")
    raise SystemExit(1)
open(f, "w").write(new)
print("PYPATCH_CUDA_VIEW_STABLE: stubbed get_cuda_view_from_cpu_tensor")
PYPATCH_CUDA_VIEW_STABLE
fi

# ll_bf16 cute-dsl router-GEMM warmup (vLLM main, sm_90+): vLLM warms up a low-latency
# BF16 router GEMM (cute_dsl/ll_bf16.py), but its availability check only verifies
# `import cutlass.cute` (present on NGC) — NOT `quack` (the CuTe helper the kernel
# actually needs), which is absent here. So the check green-lights the kernel and
# kernel_warmup() then dies with "ModuleNotFoundError: No module named 'quack'",
# failing engine init for any MoE model on Hopper. Make the check honest: also require
# quack, so ll_bf16 disables cleanly when quack is missing and vLLM uses the standard
# router GEMM. Patches the /opt/vllm source before install. No-ops on vLLM without this
# file (older presets). Idempotent.
python3 << 'PYPATCH_LL_BF16_QUACK'
import os
f = "/opt/vllm/vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py"
if not os.path.exists(f):
    print("PYPATCH_LL_BF16_QUACK: file absent (older vLLM), skipping"); raise SystemExit(0)
s = open(f).read()
if "import quack.compile_utils" in s:
    print("PYPATCH_LL_BF16_QUACK: already applied"); raise SystemExit(0)
anchor = "        import cutlass.cute  # noqa: F401"
if anchor not in s:
    print("PYPATCH_LL_BF16_QUACK: anchor not found (vLLM changed), skipping"); raise SystemExit(0)
add = anchor + "\n        import quack.compile_utils  # noqa: F401  # NGC: ll_bf16 needs quack; skip if absent"
open(f, "w").write(s.replace(anchor, add, 1))
print("PYPATCH_LL_BF16_QUACK: is_available() now also requires quack")
PYPATCH_LL_BF16_QUACK

# compressed-tensors W8A8 FP8 cutlass linear double-sets `weight_loader`:
# process_weights_after_loading() sets it on the weight, then AGAIN inside the
# `pad_n > 0 and weight_scale.numel() > 1` branch (padding + channel-wise scale),
# which trips set_weight_attrs' `assert not hasattr(weight, "weight_loader")`.
# Only bites models whose linear dims need 16-alignment padding — e.g. the Ornith
# 397B multi-node FP8 (the 35B's dims are aligned, so it never hits it). The re-set
# is redundant (already set above), so drop it. No-ops on vLLM without this file.
# Idempotent.
python3 << 'PYPATCH_CUTLASS_WEIGHTLOADER'
import os
f = "/opt/vllm/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py"
if not os.path.exists(f):
    print("PYPATCH_CUTLASS_WEIGHTLOADER: file absent, skipping"); raise SystemExit(0)
s = open(f).read()
if "Ornith-397B fix" in s:
    print("PYPATCH_CUTLASS_WEIGHTLOADER: already applied"); raise SystemExit(0)
old = (
    '            replace_parameter(layer, weight_scale_name, padded_scale.data)\n'
    '            set_weight_attrs(\n'
    '                getattr(layer, weight_name),\n'
    '                {\n'
    '                    "weight_loader": self.padded_weight_loader,\n'
    '                },\n'
    '            )'
)
new = (
    '            replace_parameter(layer, weight_scale_name, padded_scale.data)\n'
    '            # NGC/Ornith-397B fix: weight_loader already set on this weight\n'
    '            # above; re-setting it here trips set_weight_attrs\' "Overwriting\n'
    '            # existing tensor attribute" assert when pad_n>0 + channel-wise FP8\n'
    '            # scale (unaligned linear dims). The re-set is redundant, so skip it.'
)
if old not in s:
    print("PYPATCH_CUTLASS_WEIGHTLOADER: anchor not found (vLLM changed), skipping"); raise SystemExit(0)
open(f, "w").write(s.replace(old, new, 1))
print("PYPATCH_CUTLASS_WEIGHTLOADER: dropped redundant weight_loader re-set")
PYPATCH_CUTLASS_WEIGHTLOADER

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
#
# fastapi is capped <0.137: fastapi 0.137 introduced an `_IncludedRouter` route
# type that prometheus-fastapi-instrumentator 8.0.0 (pulled below) doesn't handle
# — its routing._get_route_name does `route.path`, which `_IncludedRouter` lacks,
# so EVERY request (incl. /health) 500s with "AttributeError: '_IncludedRouter'
# object has no attribute 'path'" and the server never goes ready. Builds before
# ~2026-06-15 got fastapi 0.136.x and worked (e.g. the kimi 0.21 container); the
# laguna build on 2026-06-19 pulled 0.137.2 and broke. Cap until the instrumentator
# ships a fix. Affects every preset's HTTP layer, so the pin lives here, not per-preset.
pip install --no-cache-dir \
    --root-user-action=ignore \
    --constraint /tmp/constraints.txt \
    numpy \
    "${TRANSFORMERS_PKG}" \
    tokenizers>=0.19.0 \
    sentencepiece \
    "fastapi<0.137" \
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
# Repo defaults to upstream deepseek-ai (where every pre-existing preset's pinned
# ref lives). vLLM's own DeepGEMM pin moved to the vllm-project fork, so presets
# tracking a modern vLLM release (glm53_v27) set DEEPGEMM_REPO to that fork.
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
echo "Installing DeepGEMM @ ${DEEPGEMM_REF} from ${DEEPGEMM_REPO} (required for GLM-5.x DSA indexer and FP8 MoE)..."
pip install --no-cache-dir --no-deps --root-user-action=ignore --no-build-isolation \
    "git+${DEEPGEMM_REPO}@${DEEPGEMM_REF}" 2>&1 | tail -30 || {
    echo "Warning: DeepGEMM install failed. GLM-5.1 (DSA) will not be able to load;"
    echo "         other presets are unaffected. Override DEEPGEMM_REF/DEEPGEMM_REPO."
}

# tilelang — required by DeepSeek-V4's "mhc" (head-compression) attention path.
# Without it the model loads and then dies at worker start with:
#   ImportError: tilelang is required for mhc but is not installed.
# (verified on-cluster 2026-08-18, job 2037091). It is a pure wheel on aarch64
# (manylinux_2_34_aarch64 exists for 0.1.13), so this adds no compile time.
# Installed unconditionally and failure-tolerated, exactly like DeepGEMM above:
# it is additive for every other preset, and making it preset-conditional would
# mean a container that serves DeepSeek only if it happened to be built for it.
echo ""
echo "Installing tilelang (required by DeepSeek-V4 mhc attention)..."
# PIN to the version vLLM asks for. v0.27.1 requires tilelang==0.1.12 exactly; an
# unpinned install resolves 0.1.13 and pip then reports it as incompatible. Bump
# this in step with VLLM_VERSION. TILELANG_REF overrides.
TILELANG_REF="${TILELANG_REF:-0.1.12}"
# apache-tvm-ffi must be pinned ALONGSIDE tilelang or the two disagree and the
# workers abort at startup with a C++ terminate, not a Python traceback:
#   terminate called after throwing an instance of 'tvm::ffi::Error'
#     what(): TypeAttr `__ffi_repr__` is already registered for type index 130
# (verified on-cluster 2026-08-19, job 2043587: weights loaded fine, then all four
# workers died during model init). The container had drifted to 0.1.12, which
# satisfies NEITHER vLLM v0.27.1 (requires ==0.1.11) NOR tilelang 0.1.12
# (requires <=0.1.11). flashinfer accepts anything <0.2, so 0.1.11 is the single
# version all three agree on — pinning it brings the container back INTO
# compliance with vLLM's own requirement rather than away from it.
TVM_FFI_REF="${TVM_FFI_REF:-0.1.11}"
pip install --no-cache-dir --no-deps --root-user-action=ignore "apache-tvm-ffi==${TVM_FFI_REF}" 2>&1 | tail -3 || {
    echo "Warning: apache-tvm-ffi pin failed; tilelang/DeepSeek-V4 may abort at worker init."
}
pip install --no-cache-dir --no-deps --root-user-action=ignore "tilelang==${TILELANG_REF}" 2>&1 | tail -5 || {
    echo "Warning: tilelang install failed. DeepSeek-V4 will not load (mhc path);"
    echo "         all other presets are unaffected."
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

# flashinfer sanity: vLLM main pulls flashinfer, but on this NGC stack it can be
# version-skewed against the container's cutlass-dsl — its eagerly-imported Blackwell
# kernel references `cutlass.cute.nvgpu.OperandMajorMode` (absent), so `import
# flashinfer` throws. That is a HARD engine-init crash for any model whose init/forward
# probes flashinfer (Qwen3-Next/Ornith GDN prefill + the FP8-MoE backend oracle both
# do). GH200 is Hopper and needs none of flashinfer's Blackwell kernels, and vLLM
# falls back to Triton/CUTLASS — so if flashinfer is present but does NOT import
# cleanly, uninstall it. Self-guarding: a healthy flashinfer (older presets/stacks) is
# left untouched. Idempotent. (Ornith also forces the Triton GDN prefill kernel at
# serve time via --additional-config; see run_vllm_server.sh IS_ORNITH.)
if python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('flashinfer') else 1)" 2>/dev/null; then
    if python3 -c "import flashinfer" 2>/dev/null; then
        echo "[flashinfer] imports cleanly — keeping"
    else
        echo "[flashinfer] present but import FAILS on this stack (cute-dsl skew) — uninstalling so vLLM falls back to Triton/CUTLASS"
        pip uninstall -y flashinfer-python flashinfer 2>/dev/null || true
    fi
fi

# Leave the vLLM SOURCE tree before importing: `pip install .` ran from /opt/vllm,
# whose in-tree `vllm/` package has __init__.py but NONE of the compiled
# extensions (those go into site-packages). Python prepends CWD to sys.path, so an
# `import vllm` from /opt/vllm picks up the source shim and then dies on the first
# compiled-only import — on recent vLLM main that's `import vllm._C_stable_libtorch`
# (cuda.py), a FALSE "ModuleNotFoundError: vllm._C_stable_libtorch" that threw away
# an otherwise-good build. cd to a neutral dir so we verify the INSTALLED package.
cd /tmp

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

# Check that a CUDAGraph capture path is importable.
#
# `vllm.worker.model_runner.CUDAGraphRunner` is the V0 engine layout and no longer
# exists on modern vLLM: V1 moved workers under `vllm.v1.worker` and replaced the
# CUDAGraphRunner class with the CUDAGraphWrapper dispatcher in
# `vllm.compilation.cuda_graph`. So on anything recent the old probe reported
# "⚠ No module named 'vllm.worker'" on EVERY build — a stale check reading as a
# real defect (seen on the v0.27.1/glm53 build). Try the V1 path first, fall back
# to V0 so older pinned presets (glm51 v0.19.0, kimi v0.19.1) still report ✓.
_cudagraph_probe = None
for _mod, _sym in (
    ("vllm.compilation.cuda_graph", "CUDAGraphWrapper"),   # V1 (current)
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"), # V1 fallback
    ("vllm.worker.model_runner", "CUDAGraphRunner"),       # V0 (legacy pins)
):
    try:
        __import__(_mod, fromlist=[_sym])
        _cudagraph_probe = f"{_mod}.{_sym}"
        break
    except ImportError:
        continue
if _cudagraph_probe:
    print(f"CUDA Graphs module: ✓ ({_cudagraph_probe})")
else:
    print("CUDA Graphs module: ⚠ no known capture path importable")

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
