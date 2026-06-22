# L0 — protocol / tool-call conformance

The cheapest layer of the agentic-eval strategy
(`plans/proposed/agentic_evals.md`). It drives `anthropic_proxy.py` the way
Claude Code does — `POST /v1/messages` with tools — and checks two things:

* **invariants** — the response is a *well-formed* Anthropic message and any
  tool call is one the proxy can faithfully translate (valid id, an **offered**
  tool name — never hallucinated, an object `input`, `stop_reason` consistent
  with the blocks; in streaming mode the full SSE event grammar). These must
  hold for *any* non-broken model+proxy, so a failure is a **regression** and
  makes the run exit non-zero. **This doubles as a proxy regression suite.**
* **expectations** — the fixture's behavioural *intent* (picked the right tool,
  filled schema-valid args, stopped when it should). Model-dependent, reported
  as per-category **pass rates**, never gating.

Runs fine against a ~5 tok/s endpoint: fixtures are single- / few-turn and many
terminate early. Zero third-party deps (stdlib only), like `bench_sweep.py`.

## Run it

```bash
# 1. Bring up the model + proxy the usual way:
./olivia.sh server start laguna        # (on the cluster)
./olivia.sh tunnel up                  # forward localhost:8000 -> vllm
python anthropic_proxy.py --model poolside/Laguna-M.1-FP8   # listens on :8002

# 2. Run L0 against the proxy (default base-url http://localhost:8002):
python evals/runner.py protocol --preset laguna
python evals/runner.py protocol --preset glm52  --repeat 3   # stochastic checks
python evals/runner.py protocol --preset kimi27 --mode nonstream
python evals/runner.py protocol --preset glm51              # auto-serialized (no wedge)
```

Exit code is `0` only when the invariant gate is **clean**. Results JSON is
written to `evals/results/protocol-<preset>.json` (gitignored).

Key flags: `--base-url` (or `$OLIVIA_PROXY_URL`), `--mode nonstream,stream`,
`--repeat N`, `--think on` (also runs `requires_thinking` fixtures),
`--stall SEC` (streaming inter-token wedge guard), `--concurrency N` (forced to
1 for `glm51`).

## Verify the harness logic offline (no GPU)

```bash
python evals/protocol/selftest.py      # synthetic good/bad responses; must print 0 failed
```

## Add a fixture

Drop a JSON file in `fixtures/` (see `loader.py` for the shape and
`conformance.check_expectations` for the available `check` types). Invariants run
automatically on every fixture; `expect` only declares behavioural intent. The
highest-value new fixtures are **harvested from real Claude Code traffic**: run
the proxy with `--dump-requests <dir>`, capture a payload that misbehaves, and
turn it into a fixture so the bug becomes a permanent regression test.

## Files

| File | Role |
|------|------|
| `conformance.py` | pure checkers: invariants, expectations, minimal JSON-schema validator, SSE grammar validator + reconstruction |
| `client.py` | stdlib `/v1/messages` client (non-stream + streaming, inter-token stall timeout) |
| `loader.py` | load JSON fixtures |
| `runner.py` | orchestrate fixtures × modes × repeats, aggregate, print scorecard |
| `selftest.py` | offline self-test of the checker logic |
| `fixtures/*.json` | the conformance cases |
