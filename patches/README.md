# vLLM build-time patches

Upstream vLLM PRs that aren't in any release yet but that a model preset needs.
`build_vllm_gh200.sh` applies them to the freshly cloned vLLM source (before the
`pip install .` compile) via the `VLLM_PATCHES` mechanism — a space-separated
list of GitHub PR numbers. Each is fetched from `github.com/vllm-project/vllm/pull/<N>.diff`
and `git apply`-ed; the step is idempotent (skips if already present, e.g. once
the PR merges into the pinned ref) and fails loudly if the diff no longer applies.

A preset can opt in by setting `PRESET_VLLM_PATCHES` in `apply_preset()`. Override
per-build with `VLLM_PATCHES="<n> <m>" MODEL_ID=... ./build_vllm_gh200.sh`, or
disable with `VLLM_PATCHES=""`.

The build fetches each PR **live** so it tracks the current state of the fix until
merge. The `.diff` files vendored here are **snapshots** for review / offline
reference / manual application — not necessarily byte-identical to the live PR if
it has been updated since.

| File | PR | Preset | What it fixes |
|------|----|--------|---------------|
| `vllm-pr45895-glm52-indexer.diff` | [#45895](https://github.com/vllm-project/vllm/pull/45895) | `glm52` | GLM-5.2's new skip-topk DSA indexer init + MTP final-norm recycle (pure Python, 9 files) |

**When a PR merges:** drop it from the preset's `PRESET_VLLM_PATCHES` (and delete
the snapshot) once it lands in the `VLLM_VERSION` the preset pins. Until then the
idempotent apply tolerates a merged ref by skipping.
