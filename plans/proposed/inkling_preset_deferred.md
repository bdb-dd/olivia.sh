# Inkling (Thinking Machines) preset — DEFERRED (no single-node checkpoint yet)

**Date:** 2026-07-16 · **Status:** DEFERRED — no preset added · **Branch:** `add-inkling`
**Model:** Inkling (Thinking Machines Lab), released 2026-07-15, Apache 2.0

**Verdict:** The two presets originally requested (an **FP8** preset, and a **single-node
4-bit** preset doing >350 tok/s/user on one GH200 node) **cannot be built today** — the
required checkpoints don't exist, and the only released 4-bit format (NVFP4) needs Blackwell
FP4 tensor cores for its fast path and ≥2 GH200 nodes for the Hopper fallback. Parked here so it
is trivial to resume the moment a fitting checkpoint ships. **No files under the preset system
were touched** (`presets.json`, `presets.py`, `build_vllm_gh200.sh`, `run_vllm_server.sh` are
untouched on this branch — it carries only this doc).

---

## 1. What Inkling is (confirmed facts)

From the model's own `config.json`, the HF `thinkingmachines` org page, TML's model card, and the
vLLM day-0 blog:

- **Architecture:** `architectures: ["InklingForConditionalGeneration"]`, `model_type:
  "inkling_mm_model"`. Multimodal (image/audio/text→text), decoder-only MoE.
- **Shape:** 66 layers, `n_routed_experts: 256`, `num_experts_per_tok: 6` (+2 shared),
  ~975B total / **41B active**, 1M context, controllable thinking effort
  (`none|minimal|low|medium|high|xhigh|max`).
- **Serving (vLLM PR #48768, day-0):** needs `--tokenizer-mode inkling --reasoning-parser
  inkling --tool-call-parser inkling --enable-auto-tool-choice --trust-remote-code`, MTP
  spec-decode, and env `VLLM_USE_V2_MODEL_RUNNER=1`, `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`.
  No pinned vLLM version published — treat like glm52 (build vLLM main + graft PR #48768).

## 2. Released checkpoints — the whole `thinkingmachines` org (verified 2026-07-16)

| Repo | Format | Notes |
|---|---|---|
| `thinkingmachines/Inkling` | BF16 | ~2 TB aggregated VRAM (16× H200 / 4× GH200 nodes) |
| `thinkingmachines/Inkling-NVFP4` | NVFP4 | ~553 GB / ≥600 GB aggregated |
| `thinkingmachines/meta-llama-3-tokenizer` | — | tokenizer only |
| `thinkingmachines/meta-llama-3-instruct-tokenizer` | — | tokenizer only |

**No FP8. No AWQ/GPTQ/INT4. No `Inkling-Small`. No community quants yet (day 2).**

## 3. Why each part of the original ask is blocked on GH200 (Hopper)

| Ask | Reality |
|---|---|
| **FP8 preset** | No FP8 checkpoint exists (official or community). vLLM lists FP8 as *future* work — "plan to explore FP8… by modifying the new FA4 kernel." Nothing to point a preset at. |
| **4-bit, single node, best on GH200** | Only 4-bit is **NVFP4**. Native **W4A4** needs Blackwell FP4 tensor cores → the `No compiled nvfp4 quantization kernel` wall we already document for GLM/Kimi/Laguna on Hopper. Hopper path is **W4A16**; TML's reference for it is **8× H200 ≈ 2 GH200 nodes**. At ~553 GB the weights don't fit one 4×GH200 node (384 GB) regardless. |
| **">350 tok/s/user, single node"** | Real figure, wrong hardware: **380 tok/s/user on 4× GB200 (Blackwell)** with NVFP4 **W4A4 + MTP** (140 without MTP). On Hopper you lose W4A4 *and* MTP-under-PP, and need ≥2 nodes. Not reproducible on GH200. The true single-node/350 candidate is **Inkling-Small (12B active)** — **not released** (preview only; "weights will follow after testing"). |

## 4. The one thing that *is* feasible today (if we ever want it before a better checkpoint)

A single **2-node** preset, shaped exactly like glm51/kimi, bleeding-edge like glm52:

- **`thinkingmachines/Inkling-NVFP4`, 2 nodes × 4 GH200, TP=4 + PP=2, W4A16.** ~553 GB across
  8×96 GB fits with KV headroom.
- **Build:** glm52 recipe — vLLM **main** + graft **PR #48768** (pin the commit), transformers
  `>=5.14.0`, `storage: work` (won't fit the 1 TiB project quota).
- **Inherited caveats** (all already charted in this repo):
  1. **Make-or-break unknown:** does vLLM's Inkling NVFP4 path even run **W4A16 on Hopper**?
     The blog only shows W4A4 on GB200. Same class of risk as glm52's DSA indexer — verify on
     first build before trusting it.
  2. **Multi-node PP decode wedge** → `anthropic_proxy.py` request-serialization workaround
     (like glm51/glm52/kimi).
  3. **MTP auto-disables under PP>1** (existing guard) → we forfeit the biggest chunk of the
     tok/s story, so even this preset would be modest on Olivia.

Decision (2026-07-16): **not worth building now** — a 2-node, MTP-less, W4A16-unverified preset
is a lot of bleeding-edge risk for a config that doesn't meet the actual goal (fast single-node
serving). Wait for a checkpoint that fits.

## 5. Resume checklist — pick this up when ANY of these lands

Ordered by how well each unblocks the *original* single-node/fast goal:

1. **`thinkingmachines/Inkling-Small` weights drop** (276B/12B active) — the real prize. 12B
   active → genuine single-node 4×GH200 candidate. FP8 of Inkling-Small ≈ 276 GB (fits one node);
   a 4-bit ≈ 138 GB (fits with huge KV). This is the one to watch.
2. **An FP8 Inkling checkpoint appears** (TML ships it, or RedHatAI/community re-hosts) → the
   requested FP8 preset becomes real. Full Inkling FP8 ≈ ~975 GB → still 2 nodes; Inkling-Small
   FP8 → single node.
3. **A Hopper-friendly 4-bit (AWQ/GPTQ/compressed-tensors int4) appears** (QuantTrio/cyankiwi
   pattern, like they did for GLM) → single-node full-Inkling 4-bit becomes plausible (~488 GB
   is still >384 GB for full Inkling, so still 2 nodes for the 975B; single-node only for Small).
4. **vLLM confirms NVFP4 W4A16 on Hopper** for Inkling → unblocks the §4 2-node preset if we
   want multi-node Inkling before any of the above.

**First moves on resume:**
```bash
# Re-check what checkpoints now exist (org page + config):
#   https://huggingface.co/thinkingmachines
#   https://huggingface.co/thinkingmachines/Inkling-NVFP4/raw/main/config.json  (quant_config?)
# Check vLLM Inkling integration status / version:
#   https://github.com/vllm-project/vllm/pull/48768
# Then, for a single-node FP8/4-bit Inkling-Small preset, follow the laguna pattern
# (single-node TP=4, native vLLM arch, storage tier per quota); for a 2-node NVFP4
# full-Inkling preset, follow the glm52 pattern (main + VLLM_PATCHES graft, work-tier).
# Add IS_INKLING detection to run_vllm_server.sh: inkling tokenizer-mode + inkling
# reasoning/tool parsers + --trust-remote-code + VLLM_USE_V2_MODEL_RUNNER=1.
```

## 6. Sources

- vLLM day-0 blog — <https://vllm.ai/blog/2026-07-15-inkling> (380/140 tok/s on 4× GB200; PR #48768)
- HF blog "Welcome Inkling" — <https://huggingface.co/blog/thinkingmachines-inkling>
- TML model card — <https://thinkingmachines.ai/model-card/inkling/>
- Model repo + config — <https://huggingface.co/thinkingmachines/Inkling>
- Inkling-Small still preview — TML news post + press (testingcatalog, marktechpost), 2026-07-15
