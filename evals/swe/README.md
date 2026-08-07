# L2 — agentic coding (SWE-bench-style)

The gold-standard layer of the agentic-eval strategy
(`plans/proposed/agentic_evals.md`): can the model **localize a latent bug in a
small multi-file codebase and fix it**, judged by a **hidden** regression suite
it never sees?

Each task gives the agent a buggy mini-codebase (multiple files) and a precise
problem statement with examples — but **not** the grading test. The agent drives
the same multi-turn loop as L1 (read/write/list/grep/run_bash over a sandbox,
higher turn budget). Only **after** the loop ends does the harness materialize
the hidden `grade_test.py` (`oracle_files`) and run it. So the agent must solve
from the problem statement and its own reasoning — it can't read the answer.
This is the SWE-bench pattern (gold test patch applied at eval time).

### Why not the real SWE-bench Verified dataset?

The official harness runs each instance's test suite inside a per-instance
Linux/Docker image — not available on the dev workstation this was built on. So
this slice is **self-contained hard tasks** (pure-Python, no external deps,
deterministic) that reproduce SWE-bench's *shape*: real bug localization across
files + a hidden regression suite. The harness (`evals/micro/agent.py` loop +
`oracle_files`) is dataset-agnostic, so a real-SWE-bench loader (clone repo at
`base_commit` → agent → apply gold `test_patch` → run `FAIL_TO_PASS`) slots in as
the next step where a Linux/Docker runner is available.

## Run it

```bash
# proxy must be up (see ../protocol/README.md); then:
python evals/runner.py swe --preset laguna
python evals/runner.py swe --preset glm52 --max-turns 30
python evals/runner.py swe --preset laguna --only lru-cache-recency --keep-sandbox
```

Default `--max-turns 25`. Results JSON → `evals/results/swe-<preset>.json`.
Scorecard reports the same behavioural signals as L1 (oracle-verified success,
turns, thrash, premature-stop, runaway, invalid-tool-turns).

## Tasks (`tasks/*.json`)

| id | category | the bug |
|----|----------|---------|
| calc-operator-precedence | parser | recursive-descent calc ignores precedence + parens |
| interval-merge-adjacency | algorithm | touching intervals not merged (off-by-one) |
| lru-cache-recency | data-structure | get/update don't refresh LRU recency → wrong eviction |
| topo-sort-cycle-detection | algorithm | cyclic graph returns bogus order instead of raising |
| csv-quoted-field-parsing | parser | naive split breaks quoted fields / escaped quotes |

Same shape as an L1 task plus `oracle_files` (the hidden grader, written only at
oracle time). **Keep tasks calibrated:** the buggy state must fail the hidden
grader and a correct fix must pass it. `gen_tasks.py` defines every task with its
reference fix and asserts this for all of them — re-run it to regenerate/verify:

```bash
python3 evals/swe/gen_tasks.py     # from the worktree root; rewrites tasks/ + calibrates
```
