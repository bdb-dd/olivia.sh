# Kimi K2.6 / K2.7 — outstanding follow-ups (for 2026-06-16)

End-of-day 2026-06-15. Everything is **torn down** (no GPU jobs running, 8020 tunnel
down, keepalive killed). The patches that make `reasoning_tokens` work are **live on
`vllm-kimi-4-sandbox` (hand-applied) and codified in `build_vllm_gh200.sh`**, but the
codification has **not yet been validated by a real rebuild**, and the working tree has
**uncommitted changes**. Details below.

## State at shutdown
- Canonical container: **`vllm-kimi-4-sandbox`** — vLLM `0.21.1.dev0+gad7125a4`, NGC 26.03,
  eager. Carries the live reasoning_tokens patches (parser `count_reasoning_tokens` +
  chat/completions `completion_tokens_details`). Serves both K2.6 and K2.7.
- Presets: `kimi` (K2.6) and `kimi27` (K2.7) both → index 4 = `vllm-kimi-4`.
- reasoning_tokens VERIFIED on K2.6 + K2.7, chat/completions (stream + non-stream) + responses.

## 1. Commit the staged work (do first)
The codification is **git-staged** (not committed). Run:
```
git commit -m "feat(kimi): report reasoning_tokens on chat/completions + responses" -m "..."
```
(full proposed message is in the chat; staged files: `build_vllm_gh200.sh`, `olivia.sh`,
`patch_kimi_count_reasoning.py`, `patch_chat_reasoning_tokens.py`).
Then decide on the **other uncommitted files**: `run_vllm_server.sh`, `CLAUDE.md`,
`plans/proposed/kimi_serving_perf.md`, `plans/proposed/kimi_k27_results.md`, `k27_sweep.py`.

## 2. Re-compile to validate the codified patches (the main "re-compile" task)
`vllm-kimi-4` is currently **hand-patched** — if it's ever rebuilt/lost, the patches must
come from the build script. Validate that the new `PYPATCH_REASONING_COUNT` +
`PYPATCH_CHAT_REASONING` steps actually apply during a real build:
- Rebuild a Kimi 0.21 container (NGC 26.03) from the codified `build_vllm_gh200.sh` to a
  **fresh index** (e.g. `BUILD_INDEX=8`) so the active `vllm-kimi-4` is untouched if it fails:
  `CONTAINER_DIR=... MODEL_ID=kimi BUILD_INDEX=8 VLLM_VERSION=v0.21.0 NGC_PYTORCH_TAG=26.03-py3 PRESET_TRANSFORMERS='>=5' MAX_JOBS=8 sbatch --time=4:00:00 build_vllm_gh200.sh`
- Watch the build log for the two new `Patching ...` lines reporting "Patched" (not "not found").
  Remember NGC 26.05 is a **dead end** (capture IMA + FastAPI break) — rebuild on **26.03**.
- After build: import-test (`UsageInfo(..., completion_tokens_details=...)` + parser has
  `count_reasoning_tokens`), then serve + verify reasoning_tokens. If good, repoint presets to
  index 8 (or rebuild over 4 with `OVERWRITE=1` for an atomic swap) and retire the hand-patched one.

## 3. Cleanup (scratch disk — no GPU cost)
Containers (`/cluster/work/projects/nn10104k/containers/`):
- **KEEP**: `vllm-kimi-4-sandbox` (active, patched, 0.21/NGC26.03).
- **REMOVE (confident)**: `vllm-kimi-6-sandbox` — the NGC-26.05 dead end (capture IMA +
  FastAPI 500; superseded). `rm -rf` it + its `.new.*/.prev.*` if any.
- **EVALUATE then remove**: `vllm-kimi-1` (0.19 legacy, old preset default), `vllm-kimi-2`,
  `vllm-kimi-3` (0.19), `vllm-kimi-5`, `vllm-kimi-7` (build-campaign artifacts). Confirm none
  are referenced before deleting.
- Stale **cache dirs**: `cache/*k2*` (all from the test servers; biggest is
  `cache/vllm_k27fap` ≈ 2.0 GB). Safe to remove — no server uses them now.
- Login-node leftovers in `~`: `k27_keepalive.sh`, `k27_sweep.py`, `diag_resolve*.sh`,
  `patch_*.py` (optional tidy).

## 4. Lower priority
- **Mistral Vibe bash tool calls**: earlier flagged ("bash tool calls need a fix"); last
  session reported "great results", so likely fine — confirm before closing. The fix on our
  side was `reasoning_field_name="reasoning"` for the `kimi27` provider (already applied).
- Reconcile any stale docs (`plans/proposed/kimi_serving_perf.md`, `CLAUDE.md`) that still
  describe the superseded 0.19/PIECEWISE story (memory already updated).
- Decide whether to keep K2.6 and K2.7 deployed long-term (presets + patched container ready).
