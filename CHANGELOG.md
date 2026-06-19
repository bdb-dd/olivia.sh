# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not (yet) follow semantic versioning; entries are pinned to
the date the work landed.

## [Unreleased]

### 2026-06-20 — Per-branch deploy isolation (fix concurrent-agent script clobber)

Multiple agents developing different presets on different branches were
clobbering each other's deployed cluster scripts. `olivia.sh` deployed
`run_vllm_server.sh` / `build_vllm_gh200.sh` (+ chat template, `patches/`) to a
**single shared path** under `CONTAINER_DIR` and submitted with `sbatch
run_vllm_server.sh`. SLURM snapshots the batch script at submit, so the
corruption window is deploy→submit: agent A deploys, agent B deploys a divergent
version (different `IS_<preset>` detection), A's `sbatch` then copies B's file →
A's job runs the wrong preset's config (seen: a `laguna` serve silently running
generic 32K-context with no reasoning parser, because an agent on `main`
re-deployed over the shared script).

#### Fixed

- **Per-branch deploy directory** (`olivia.sh`): deploy + submit now use
  `${CONTAINER_DIR}/deploys/<DEPLOY_KEY>/`, where `DEPLOY_KEY` defaults to the
  local git branch (override via env). `deploy_server_script` /
  `deploy_build_script` `mkdir -p` and upload there; serve/build submit
  `sbatch ${DEPLOY_DIR}/<script>` and pass `CHAT_TEMPLATE_FILE` / `PATCHES_DIR`
  from `DEPLOY_DIR`. Divergent branch scripts can no longer collide; an agent on
  this scheme neither clobbers nor is clobbered by the legacy shared path.
  Sandboxes, `logs/`, and the `$PWD/cache` compile cache stay shared (submit cwd
  is still `CONTAINER_DIR`).

> Verified on Olivia 2026-06-20: deploy lands in `deploys/<branch>/` (md5 ==
> local, shared path mtime unchanged); a simulated competing agent's divergent
> deploy left the dir untouched; a submitted job's `scontrol Command` is the
> per-branch script with `WorkDir=CONTAINER_DIR` (shared cache/logs preserved).

### 2026-06-19 — Laguna M.1 preset (Poolside, FP8, single-node 4×GH200)

Adds a `laguna` preset for Poolside's Laguna M.1 (225B total / 23B active MoE
coding model, `LagunaForCausalLM`, 256 experts top-k=16, dense GQA full
attention, 262K context). Default quant is block-FP8
(`poolside/Laguna-M.1-FP8`, ~225 GB), which fits a **single GH200 node** at
TP=4 — no cross-node pipeline parallel, so none of the multi-node PP decode
wedge that affects glm51/glm52/kimi. First preset that is natively supported by
upstream vLLM (≥ v0.21.0, PR#41129) with no graft or custom-code arch, and the
first to keep ordinary `FLASH_ATTN` + CUDAGraph capture rather than the
eager/MLA path the GLM-5.x and Kimi MoEs need.

#### Added

- **`laguna` preset** across `olivia.sh` (alias `normalize_preset` +
  `preset_field`: `poolside/Laguna-M.1-FP8`, single-node 1×4 GPUs, TP=4),
  `build_vllm_gh200.sh` (`apply_preset` + `show_presets`: pins `vLLM v0.21.0`,
  `transformers >= 5.7.0`, inherits the NGC 26.03 base), and
  `run_vllm_server.sh` (`IS_LAGUNA` detection → `--tool-call-parser
  poolside_v1`, `--reasoning-parser poolside_v1`, `--enable-auto-tool-choice`,
  `--trust-remote-code`, thinking-mode chat-template kwargs; 131072 default
  context; FLASH_ATTN; CUDAGraph capture left on).
- **Override knobs**: `LAGUNA_TOOL_PARSER`, `LAGUNA_REASONING_PARSER`, and
  `LAGUNA_ENABLE_THINKING` (set `0` for instant mode, no reasoning extraction).
- **Docs**: Laguna M.1 section + preset/build tables in `CLAUDE.md`, preset
  table + feature list in `README.md`.

> To verify on first build/serve (sourced from the model card + vLLM PR, not yet
> run on Olivia): the `poolside_v1` parser names, the
> `--default-chat-template-kwargs` flag, and that v0.21.0 compiles on NGC 26.03
> (fallback `NGC_PYTORCH_TAG=26.05-py3`, the glm52 base).

### 2026-04-21 — GLM-5.1-AWQ-4bit serving end-to-end on 2 × 4 GH200s

First working end-to-end run of `cyankiwi/GLM-5.1-AWQ-4bit` on Olivia. The
server now bootstraps a Ray cluster over SLURM, fits the ~430 GB model across
8 GH200s with `TP=4 + PP=2` over Slingshot, and streams tokens through
DeepSeek Sparse Attention's `DEEPSEEK_V32_INDEXER` backend.

#### Added

- **`glm51` preset** in `olivia.sh` and `build_vllm_gh200.sh`: defaults to
  `cyankiwi/GLM-5.1-AWQ-4bit`, requests 2 nodes × 4 GPUs, sets `TP=4 + PP=2`,
  pins `vLLM v0.19.0` and `transformers >= 5.4.0` per the official vLLM GLM-5
  recipe. Per-preset SLURM resource resolution lives in `get_preset_resources()`
  so non-multi-node presets continue to run on a single node.
- **Multi-node Ray bootstrap** in `run_vllm_server.sh`: when `NUM_NODES > 1`,
  starts a Ray head on node 0 and workers on the remaining nodes over `srun`,
  waits for the expected GPU count to register, then launches `vllm serve` on
  the head with `--distributed-executor-backend=ray`. Cleans up the cluster
  on exit via a trap.
- **DeepGEMM install** in `build_vllm_gh200.sh`, pinned to commit
  [`59f2c07`](https://github.com/deepseek-ai/DeepGEMM/commit/59f2c07) (2025-09-29).
  Required for GLM-5.1's DSA indexer and the FP8 MoE kernel on Hopper.
- **Per-worker Slingshot IP resolution**: each worker's `hsn0` IP is
  pre-resolved on the login node (inside-container detection is flaky) and
  passed to `ray start` via `--node-ip-address` plus `VLLM_HOST_IP` and
  `RAY_node_ip_address` env vars so Ray actors consistently report Slingshot
  addresses.
- **Sandbox swap semantics** in `build_vllm_gh200.sh`: with `OVERWRITE=1`,
  builds go into `<sandbox>.new.<jobid>` and atomically rename to the target
  path at the end, preserving the previous sandbox as `.prev.<timestamp>` for
  rollback. Orphan `.new.*` from prior failed builds are cleaned up on entry.
- **`build_logs`** subcommand reference in `CLAUDE.md` (implementation
  pending); prefetch-subcommand idea captured as a future improvement.

#### Changed

- **NGC PyTorch base image** default moved from `25.12-py3` to `26.03-py3`.
  25.12 predates the upstream `TORCH_BOX` macro (pytorch 2.10.0, landed
  2025-11-11), which means vLLM's `csrc/libtorch_stable` extension fails to
  compile and `torch.ops._C.per_token_group_fp8_quant` — which GLM-5.1's DSA
  indexer calls unconditionally — is never registered at runtime. 26.01 has
  `TORCH_BOX` but rejects v0.19.0's reference-type op signatures with a
  `static_assert`; 26.03 has both. Overridable via `NGC_PYTORCH_TAG`.
- **vLLM clone no longer `--depth 1`** for tagged versions: `setuptools_scm`
  needs tag history to report the real version. Previously we ended up with
  `v0.19.1.dev0+g<sha>.d<date>` even when pinning `v0.19.0`, which silently
  enables the wrong code paths.
- **`JOB_NAME_PATTERN`** in `olivia.sh` changed from `"vllm"` to
  `"vllm-server"`. Previous wildcard matched concurrently running
  `build-vllm-gh200` jobs and led `server logs` / `status` / `cancel` to act
  on the wrong job — including, potentially, cancelling an in-flight build.
- **`server logs` / `watch` / `start` / `restart`** now tail both the SLURM
  wrapper log and `vllm_server_<jobid>_head.log`. On multi-node jobs the real
  `vllm serve` output only appears in the head log; before this fix, the CLI
  stopped at the Ray-bootstrap handoff and appeared to hang forever.
- **JIT cache directories** (Triton, DeepGEMM, TorchInductor) now default to
  `${CONTAINER_DIR}/cache/*` on the project filesystem. Previously they landed
  in the small user home quota, causing `OSError: [Errno 122] Disk quota
  exceeded` partway through memory-profile forward passes.
- **Ray `--num-cpus`** explicitly set to `SLURM_CPUS_PER_TASK`. Auto-detection
  on GH200 sees 288 logical CPUs and triggers a worker-pre-warm storm that
  saturates the raylet before legitimate actor placement can proceed.
- **`find_vllm_node`** now expands compressed SLURM nodelists (e.g.
  `gpu-1-[90-91]` → `gpu-1-90`) so the chat tunnel target is a valid single
  hostname rather than a squeue-compressed range.
- **Chat client banner** (`chat_devstral.py`) queries `/v1/models` before
  rendering and shows the actual served model name instead of a hardcoded
  "Devstral Chat Client".
- **Attention-backend selection** for GLM-5.1 is left to vLLM. Forcing
  `FLASH_ATTN` was fine for non-MLA models but fails for sparse MLA with
  `['head_size not supported', 'MLA not supported', 'sparse not supported']`;
  leaving the choice open lets vLLM pick `DEEPSEEK_V32_INDEXER`.
- **`VERBOSE=1`** now flips `VLLM_LOGGING_LEVEL=DEBUG` *and*
  `RAY_BACKEND_LOG_LEVEL=debug`, `RAY_DEDUP_LOGS=0`, `NCCL_DEBUG=INFO`, all
  forwarded through `olivia.sh` → `sbatch` → `SING_CMD`. Critical for
  diagnosing startup hangs: the first silent hang of the day was only
  diagnosable after these variables were wired up correctly.
- **NGC-PyTorch version assertion** broadened from `nv24|nv25` to
  `nv24..nv27` so newer base images don't trip the "PyTorch was replaced"
  safety check.

#### Fixed

- **`OVERWRITE=1` silently reused stale sandboxes.** The Phase 1 guard
  printed `"will rebuild"` but never removed the existing directory, so the
  conditional `singularity build` short-circuited. Every rebuild in the
  session prior to the fix kept running on the original base image; Phase 3
  would reinstall vLLM on top so the artifact looked plausible, but the
  PyTorch layer, CUDA toolkit, and system libraries never changed. Now
  rebuild goes through a `.new.<jobid>` path and swaps atomically.
- **`register_opaque_type(..., hoist=True)`** in
  `vllm/utils/torch_utils.py:775` now has the `hoist=` kwarg stripped at
  build time. NGC 26.03's torch predates the kwarg's introduction; without
  the patch, `import vllm` raises `TypeError` during engine verify. Only
  affects torch.compile graph hoisting; we disable compilation anyway.
- **`Every node should have a unique IP address`** raised by vLLM's Ray
  executor: caused by Ray actors auto-detecting the ethernet IP
  (10.168.x.x) while the cluster itself was bootstrapped on Slingshot
  (10.63.x.x). Fixed by pinning both via env vars + `--node-ip-address`.
- **Ray placement group never resolving** (`No available node types can
  fulfill resource request {'node:10.168.0.50': 0.001, 'GPU': 1.0}`):
  vLLM was constructing the first bundle's node-affinity key from its own
  auto-detected host IP (ethernet) instead of Ray's registered IP
  (Slingshot). Fixed by exporting `VLLM_HOST_IP=${HEAD_NODE_IP}` on the
  driver.
- **DeepGEMM API mismatch** (`csrc/apis/attention.hpp:195:
  context_lens.dim() == 2`). DeepGEMM commit
  [`38f8ef7`](https://github.com/deepseek-ai/DeepGEMM/commit/38f8ef7)
  (2025-11-21) changed `fp8_mqa_logits`'s `context_lens` shape from 1D to
  2D, but vLLM v0.19.0's wrapper at `vllm/utils/deep_gemm.py` still passes
  a 1D `[B]` tensor. Pinned DeepGEMM to `59f2c07` — the last commit
  touching `attention.hpp` before the break.
- **DeepGEMM "Corrupted JIT cache directory"** on first inference. The env
  var is `DG_JIT_CACHE_DIR`, not `DG_CACHE_DIR`; when unset, DeepGEMM falls
  back to a path inside the container's read-only root, writes fail, and
  subsequent reads see a half-populated cache and bail.
- **Chat tunnel target `gpu-1-[90-91]:8000`** (a compressed SLURM nodelist,
  not a valid SSH target). `find_vllm_node` now expands the range and
  returns the head node.
- **Cosmetic noise** during server startup: `tail -F`'s "cannot open"
  chatter is suppressed until the log files actually appear.

#### Observability

- Added `VERBOSE=1` plumbing described above.
- Phase 4 of the build now prints `DeepGEMM: OK (<version>)` so the
  presence of the correct DeepGEMM commit is verifiable at-a-glance from
  the build log.
- Build banner echoes the final resolved `NGC_PYTORCH_TAG` so rebuilds on
  different bases are unambiguous.

#### Memory notes (for future sessions)

- `olivia.sh prefetch <preset|model>` proposed as a way to warm the HF
  cache from the login node before submitting a GPU job. First run of
  `glm51` spent ~20 minutes of a 2-node × 4-GPU allocation just
  downloading `cyankiwi/GLM-5.1-AWQ-4bit`; subsequent launches are fast
  because the cache is warm.
- `CUDAGRAPH_MODE` env var now controls vLLM compilation/CUDAGraph
  capture. Default (unset) lets vLLM auto-select, which on glm51 should
  land on `PIECEWISE` or `FULL_AND_PIECEWISE` and unlock 2-5× decode
  throughput vs. pure eager. Startup gains a multi-minute "Capturing
  cudagraph" phase. If capture hangs or crashes (known hazards:
  PP=2 cross-node collectives, DSA dynamic shapes, MoE routing), fall
  back with `CUDAGRAPH_MODE=NONE` — no code revert needed. Previously
  this was hardcoded to `NONE` in `VLLM_ARGS`.
