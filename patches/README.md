# vLLM build-time patches

Upstream vLLM PRs that aren't in any release yet but that a model preset needs.
`build_vllm_gh200.sh` applies them to the freshly cloned vLLM source (before the
`pip install .` compile) via the `VLLM_PATCHES` mechanism — a space-separated
list of GitHub PR numbers. Each is applied from the committed snapshot here (bound
into the build container at `/opt/olivia-patches`), or fetched live from
`github.com/vllm-project/vllm/pull/<N>.diff` if no snapshot is present, then
`git apply`-ed; the step is idempotent (skips if already present, e.g. once the PR
merges into the pinned ref) and fails loudly if the diff no longer applies.

A preset can opt in by setting `PRESET_VLLM_PATCHES` in `apply_preset()`. Override
per-build with `VLLM_PATCHES="<n> <m>" MODEL_ID=... ./build_vllm_gh200.sh`, or
disable with `VLLM_PATCHES=""`.

The build **prefers the committed snapshot** here, falling back to a live GitHub
fetch only if a preset's snapshot is absent — so a build is reproducible and
offline-capable. For **full** reproducibility the consuming preset should also
**pin `PRESET_VLLM_VERSION` to a commit** the snapshot was validated against
(glm52 pins main `091386a`), so clone(pinned base) + apply(snapshot) is
byte-identical every build. A snapshot vendored here must stay in sync with that
pinned base; re-snapshot if you bump the pin.

| File | PR | Preset | What it fixes |
|------|----|--------|---------------|
| `vllm-pr45895-glm52-indexer.diff` | [#45895](https://github.com/vllm-project/vllm/pull/45895) | `glm52` | GLM-5.2's new skip-topk DSA indexer init + MTP final-norm recycle (pure Python, 9 files) |

**When a PR merges:** drop it from the preset's `PRESET_VLLM_PATCHES` (and delete
the snapshot) once it lands in the `VLLM_VERSION` the preset pins. Until then the
idempotent apply tolerates a merged ref by skipping.
