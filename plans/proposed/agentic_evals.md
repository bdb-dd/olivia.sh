# Agentic-behaviour evaluation strategy

**Date:** 2026-06-20 · **Status:** proposal · **Branch:** `agentic-evals`
**Scope:** how well do the self-hosted models behave *as the brain of an agent loop* —
tool calling, multi-step task completion, planning, knowing when to stop — when driven
through `anthropic_proxy.py` the way Claude Code drives them in production.
**Priority presets (locked 2026-06-20):** `glm52`, `kimi27`, `laguna` (M.1). `glm47`
as a fast smoke target, `glm51` only via the serialized path; everything else later.

This is **not** another throughput sweep. `bench_sweep.py` / `bench_serving.py` answer
"how fast does it decode tokens." This answers "are the tokens any good when the model
is wearing the agent hat." We have zero coverage of the latter today.

---

## 1. What "agentic behaviour" means for *this* stack

The product is "Claude Code, but the brain is a model we host." So the unit under test is
the full path:

```
Claude Code / harness ──/v1/messages──► anthropic_proxy.py ──/v1/chat/completions──► vLLM (preset)
```

Agentic competence on this path decomposes into four observable things, cheapest → dearest:

1. **Protocol conformance** — the model emits tool calls the proxy can actually translate:
   right tool name, JSON args that validate against the tool schema, no hallucinated tools,
   correct multi-tool / parallel-tool sequencing, clean `thinking → text → tool_use` block
   transitions, and *stops* when the task is done. This is where most self-hosted-agent
   failures live, and it's the layer where **the proxy itself is a confound** (a malformed
   tool call may be the model's fault or the translator's).
2. **Task completion** — given real tools (read/edit file, run bash, grep) and a real task,
   does it drive the loop to a verifiable end state.
3. **Loop hygiene** — turn efficiency, no thrashing/repeating actions, no runaway loops, no
   premature "I'm done" before the goal is met, recovery after a tool error.
4. **Reasoning value** — for the thinking models (`glm52`, `glm51`, `kimi*`), does extended
   thinking actually raise success, or just burn tokens at ~5 tok/s.

---

## 2. Design constraints (these dictate the architecture)

The serving reality is harsh and *non-negotiable* — the eval design bends around it, not
the reverse:

- **Single-stream is slow — except laguna.** glm52 ≈ 5.6 tok/s end-to-end, ~14 s TTFT (PP=3);
  kimi ≈ 17 tok/s. An agentic task is *N sequential turns*, each a full prefill+decode, so a
  15-turn SWE task on glm52 is realistically **20–60 min of wall-clock**. **`laguna` is the
  outlier**: single-node TP=4, CUDAGraph captured, **~63 tok/s single-stream, sub-second TTFT,
  no wedge** — ~10× cheaper per task, so it's the natural **L2 pilot** where we can afford
  full-fat slices and tune the harness before paying glm52/kimi wall-clock. Regardless, the
  harness must run **detached on the login node**, **checkpoint after every turn**, and be
  **fully resumable** — a dropped tunnel costs one turn, not the run. (Reuse `olivia.sh`'s SSH
  ControlMaster + detached-start machinery.)
- **Each preset is a distinct tool-call surface.** glm52/kimi/laguna ship *different* parsers
  (`poolside_v1` for laguna, GLM/Kimi reasoning parsers for the others) → different raw
  `tool_calls` JSON the proxy must translate. L0 fixtures run **per preset**, not once; a new
  preset = a new translation surface that can regress the proxy.
- **glm51 wedges under concurrency** (`Running ≥ 2`) → the proxy serializes it. So eval
  concurrency is a **per-preset knob**: 1 for glm51, higher for glm52/kimi (glm52 is clean
  to 64-way). Never assume you can parallelize task instances against a given endpoint.
- **Endpoints stall.** Borrow `bench_serving.py`'s `--stall` inter-token read-timeout so a
  wedged turn errors fast and the harness can retry/abandon the *instance* instead of hanging
  the whole run.
- **Cost is wall-clock + GPU-hours, not dollars.** So: small curated task sets, hard per-task
  **turn caps** and **wall-clock budgets**, token-frugal prompts. Favour many cheap signals
  over a few expensive ones.
- **The proxy can already record/replay.** `--dump-requests-dir` dumps every `/v1/messages`
  body (`req-{ts}-{id}.json`) for offline curl replay. This is the single most useful asset we
  have: it turns slow live runs into a **corpus** we can replay against the proxy for free,
  harvest L0 fixtures from, and diff proxy versions with — no GPU needed.

---

## 3. Layered architecture (gate cheap → expensive)

### L0 — Protocol conformance (fast, in-repo, runs at 5 tok/s without pain)
A fixture suite of single / few-turn cases that exercise the tool-call contract end-to-end
through the proxy. Each fixture = a `/v1/messages` request (tools + history) + an assertion on
the translated output:
- well-formed `tool_use` with schema-valid args; correct tool selected from a set;
- refuses to invent a tool not offered; threads a prior `tool_result` correctly;
- parallel tool calls in one turn; `thinking → tool_use → text` ordering;
- terminates (no tool call) when the answer is already known.

Oracle is deterministic (JSON-schema validate + structural asserts). **This layer doubles as
a proxy regression suite** and is the highest-leverage thing to build first: it's cheap,
attributes model-vs-proxy bugs, and directly protects `anthropic_proxy.py`. Seed it from real
captured payloads (`--dump-requests-dir`) so the fixtures are exactly what Claude Code sends.

### L1 — Micro-agent tasks (curated, verifiable, nightly-able)
10–30 hand-built tasks each solvable in a handful of turns inside a throwaway sandbox dir,
with a tiny tool set (read_file, write_file, run_bash, grep). **Deterministic success
oracle**: post-state file diff, command exit code, or string match. Examples: "make this
failing pytest pass," "find which function raises X and fix it," "rename symbol across 3
files," "parse this log and write the answer to out.txt." Cheap enough to run per preset on a
cadence; this is the day-to-day signal.

### L2 — Real agentic benchmark (gold standard, overnight, small N)
A **slice** (N≈20–50, not the full set) of an established harness, run two ways:
- **Faithful path (primary):** drive **Claude Code headless** — `claude -p --output-format
  json` with `ANTHROPIC_BASE_URL` pointed at the proxy — over **SWE-bench Verified** (or
  **terminal-bench**) instances. This measures the *actual production loop*, proxy included.
- **Isolating path (control):** the benchmark's native harness against the **raw OpenAI
  endpoint** (skip the proxy). Diffing the two paths attributes failures to model vs proxy.
- **τ-bench** as the pure multi-turn tool-use complement (less coding, more
  tool-orchestration + policy-following).

### Ceiling / reference — **deferred (not a v1 priority)**
A real-Anthropic-API reference run would make scores interpretable ("glm52 solves 12/20 vs
Claude 18/20"), but it's **deprioritized for now** (decision 2026-06-20): costs real $ and ships
eval tasks to the hosted API. Keep the adapter seam so it's a one-flag opt-in later; until then,
compare presets **relative to each other** and to published SWE-bench numbers for context.

---

## 4. Metrics (behavioural, not just pass/fail)

| Metric | Layer | Why |
|---|---|---|
| Task success rate (oracle-verified) | L1/L2 | the headline number |
| Tool-call validity rate (well-formed / total) | L0/all | protocol health; proxy regressions |
| Wrong-tool / hallucinated-tool rate | L0/all | planning + grounding |
| Turns-to-completion (median, p90) | L1/L2 | efficiency at 5 tok/s this *is* cost |
| Wasted-turn / thrash rate (repeated identical action) | L1/L2 | loop hygiene |
| Premature-stop vs runaway-loop rate | L1/L2 | knowing when it's done |
| Error-recovery rate (success after a tool error) | L1/L2 | robustness |
| Thinking on/off Δ (success and tokens) | all | is reasoning worth the tok/s |
| Wall-clock + tokens per solved task | all | ties back to the Performance culture |

---

## 5. Comparison matrix & reporting

Axes: **preset** × **thinking{on,off}** × **proxy{on,off / version}**, plus the **real-Claude
reference** row. Mirror the existing README discipline exactly: dated, tabular, "re-run after
any serving-config change." Add an **`## Agentic` subsection under `## Performance`** in the
README, e.g.:

```
### Agentic — L1 micro-agent suite (v1, 24 tasks) · 2026-0x-xx
| preset | think | success | tool-valid | med turns | thrash | wall/task |
|--------|-------|---------|-----------|-----------|--------|-----------|
| laguna | on    | 19/24   | 98%       | 5         | 3%     | 1 min     |
| kimi27 | on    | 20/24   | 99%       | 5         | 2%     | 4 min     |
| glm52  | on    | 18/24   | 97%       | 6         | 4%     | 9 min     |
```
(Real-Claude reference row deferred — see §3.)

Keep the per-run detail doc in `plans/proposed/` like `glm52_perf.md`; promote summary rows to
the README.

---

## 6. Build vs adopt

- **L0 + L1: build in-repo.** They're proxy-specific and tiny; reuse the request plumbing and
  `--stall` pattern from the bench scripts. No external dep worth the friction.
- **L2: adopt, don't reinvent.** Use SWE-bench / terminal-bench / τ-bench harnesses as-is. The
  only glue we write is the **Claude-Code-headless adapter** (env-point at the proxy, collect
  the JSON transcript) and an **OpenAI-direct adapter** (proxy-bypass control). Driving Claude
  Code itself is deliberate: it's the most ecologically valid measure *and* the least code,
  since the whole stack already targets that client.

---

## 7. Attribution: model vs proxy (the record/replay loop)

1. Live L2 run with `--dump-requests-dir` ON → every turn's exact payload is captured.
2. A failure becomes a **replayable fixture**: curl it at the proxy offline (no GPU) to see if
   the proxy mistranslated, then at the raw endpoint to see the model's raw output.
3. If it's a proxy bug → it becomes an L0 regression fixture. If it's a model bug → it's a row
   in the model's scorecard. Either way the slow GPU run is amortised into permanent cheap tests.

This is what makes the strategy tractable on a ~5 tok/s endpoint: we pay for each turn **once**.

---

## 8. Build order (each phase gated on the last)

1. **L0 harness + ~20 fixtures**, seeded from captured Claude Code payloads, run **per preset**
   (glm52, kimi27, laguna — three parser surfaces). Wire it as a proxy regression suite first —
   immediate value, GPU-light.
2. **L1 micro-agent harness + sandbox + ~24 tasks**, deterministic oracles. Run **laguna first**
   (fastest, cheapest to iterate), then kimi27 + glm52.
3. **Reporting**: scorecard generator → README `## Agentic` table; per-run doc in `plans/`.
4. **L2 adapter** (Claude Code headless → proxy) over a **20-instance SWE-bench Verified slice**,
   detached + resumable. **Pilot on laguna** (≈10× cheaper wall-clock), then kimi27/glm52. Add
   the OpenAI-direct control. Real-Claude ceiling deferred (§3).

## 9. Proposed layout

```
evals/
  protocol/     # L0 fixtures + runner (also imported as a proxy regression suite)
  micro/        # L1 sandbox harness, tasks/, oracles
  agentic/      # L2 adapters (claude-code-headless, openai-direct), slice configs
  report.py     # scorecard → README table + plans/ doc
  runner.py     # detached, checkpointed, resumable, per-preset concurrency + --stall
```

---

## 10. Decisions (locked 2026-06-20) + remaining questions

**Locked:**
- **Priority presets:** `glm52`, `kimi27`, `laguna` (M.1). laguna is the L1/L2 pilot (cheapest);
  glm47 = fast smoke target; glm51 = serialized path only.
- **Real-Anthropic ceiling:** **deferred** — not a v1 priority. Keep the seam, opt-in later.
- **First L2:** a 20-instance **SWE-bench Verified** slice (most comparable to published numbers).

**Still open (sensible defaults assumed; flag to change):**
- **Per-task budget** — default 25-turn cap, 30-min wall-clock cap (drop to ~5 min for laguna),
  `--stall 45s` per token.
- **Sandbox isolation for L1/L2** — local throwaway dir vs container; default to a per-task temp
  dir + git-clean reset between instances.

## Repro (once L0/L1 land)

```bash
# L0 — proxy/tool-call conformance (GPU-light; runs fine at 5 tok/s)
python evals/runner.py protocol --preset glm52 --proxy on

# L1 — micro-agent suite, detached + resumable, per-preset concurrency
nohup python evals/runner.py micro --preset kimi27 --think on \
  --turn-cap 25 --wall-cap 30m --stall 45 --resume &

# Report → README ## Agentic table + plans/ per-run doc
python evals/report.py --suite micro --emit readme,plan
```

---

## Next session — broaden the sweep to kimi27 + glm52 (handoff 2026-06-20)

**State:** L0–L2 + L2-real all built, validated, and swept on **Laguna** (PR #3, branch `agentic-evals`). Headline: bare loop **5/12 (42%)** → mini-swe-agent **7/12 (58%)** → Poolside prod agent **72.5%** on a 12-instance django 4.2/5.0 slice — gap is mostly *agent harness, not model*. Cluster harness (small-partition apptainer) and all gotchas documented.

**Goal tomorrow:** cross-model comparison on the *same validated django slice* (apples-to-apples), then optionally broaden repos.

### Phase 1 — cross-model (quick; harness is model-agnostic)
For each of `kimi27` (2-node) and `glm52` (3-node):
1. Serve it: `HF_HOME=/cluster/work/projects/nn10104k/huggingface TIME_LIMIT=6:00:00 ./olivia.sh server start <preset>` — **bump TIME_LIMIT** (the 8h default killed a run mid-sweep), note the **head node**, and run promptly. Confirm the served model name via `/v1/models`.
2. Real agent: `GPU_NODE=<head> MODEL=openai/<served-model> PRESET=<preset>-mini sbatch evals/swe_real/run_mini_on_cluster.sh`
3. Our bare loop (for the per-model loop-vs-agent delta): `GPU_NODE=<head> MODEL=<served-model> PRESET=<preset> sbatch evals/swe_real/run_on_cluster.sh` (note: this path uses `anthropic_proxy` in-job → set the right model).
4. Update README `## Performance > Agentic` with the cross-model table.

### Phase 2 — broaden repos (bigger; beyond django)
django was the only no-Docker-tractable repo. Proper route: **mini-swe-agent's native `--environment-class singularity`** pulls the official `sweb.eval.*` images via apptainer on the `small` partition (the Docker→apptainer path, automated) → full multi-repo Verified. Spike 1 instance first to confirm image-pull + (no) nested-apptainer issues. Alt: extend the clone+pip approach to more pure-python repos.

### Reusable assets already on the cluster (`/cluster/work/projects/nn10104k/swe/`)
`python311.sif`, `miniswe-venv` (mini-swe-agent installed), the django clone+venv, and the `agentic-evals` deploy. Just point at a new server. Gotchas (all live in the code): `HF_HOME=/cluster/work`; `no_proxy` must include the vLLM head node; `MSWEA_COST_TRACKING=ignore_errors`; mini-swe-agent talks to the OpenAI endpoint **directly** (no anthropic_proxy, since it's bash-in-markdown not tool-calls).
