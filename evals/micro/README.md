# L1 — micro-agent tasks

The day-to-day behavioural layer of the agentic-eval strategy
(`plans/proposed/agentic_evals.md`). Where L0 asks "is one tool call
well-formed?", L1 asks "can the model **drive a multi-turn loop to a verifiable
end state?**"

Each task runs a real agent loop against `anthropic_proxy.py`: the model gets a
prompt + a small tool set (`read_file`, `write_file`, `list_dir`, `grep`,
`run_bash`) over a throwaway sandbox dir; the harness executes each tool call and
feeds the result back until the model stops or hits the turn cap. Then a
**deterministic oracle** judges the final sandbox state — never the model's own
claim of success.

The loop reuses L0's `check_invariants` to score **tool-call validity per turn**,
so a protocol break mid-loop is caught here too, and it exercises the proxy's
history translation (assistant `tool_use` + user `tool_result`) on every turn.

## Run it

```bash
# proxy must be up (see ../protocol/README.md), then:
python evals/runner.py micro --preset laguna
python evals/runner.py micro --preset glm52  --max-turns 16
python evals/runner.py micro --preset kimi27 --only fix-failing-test,rename-symbol
python evals/runner.py micro --preset laguna --no-bash      # read/write tasks only
```

Detached + resumable-friendly for slow endpoints:
```bash
nohup python evals/runner.py micro --preset glm52 --max-turns 16 > micro-glm52.log 2>&1 &
```

Results JSON → `evals/results/micro-<preset>.json` (gitignored). Scorecard
reports oracle-verified **success rate** plus the behavioural signals from the
strategy: median turns-to-solve, thrash (repeated identical calls),
premature-stop, runaway (hit the cap), and per-turn invalid-tool-call rate.

## Verify offline (no GPU)

```bash
python evals/micro/selftest.py     # tools + oracle + agent loop via mock models
```

## Tasks

JSON in `tasks/`. A task is `{prompt, files, tools, max_turns, oracle}`; the
oracle is one or more deterministic checks (`file_equals`, `file_contains`,
`bash_exit_zero`, `stdout_contains`, …; see `oracle.py`). Test scripts are plain
`python3` (no pytest dependency). Keep tasks **calibrated**: the initial state
must fail the oracle and a correct solution must pass it (the calibration check
in the commit history shows how to assert this for new tasks).

> **Sandboxing caveat:** `run_bash` executes model-generated shell in a temp dir
> with a timeout; file tools are realpath-jailed but bash is not. Run untrusted
> models inside the cluster container or a disposable VM, or use `--no-bash`.

## Files

| File | Role |
|------|------|
| `agent.py` | the multi-turn loop + outcome metrics |
| `tools.py` | sandbox tool defs (Anthropic) + executors |
| `sandbox.py` | per-run temp dir seeded from a task's files |
| `oracle.py` | deterministic success checks |
| `runner.py` | orchestrate tasks × repeats, aggregate, scorecard |
| `selftest.py` | offline test of tools + oracle + loop |
| `tasks/*.json` | the task set |
