#!/usr/bin/env python3
"""Unified entry point for the Olivia agentic-eval layers.

v1 implements the ``protocol`` subcommand (L0 — tool-call / protocol conformance
through ``anthropic_proxy.py``). ``micro`` (L1) and ``agentic`` (L2) land later.

    python evals/runner.py protocol --preset laguna --base-url http://localhost:8001
    python evals/runner.py protocol --preset glm52 --mode nonstream --repeat 3

Runs equally as a script (``python evals/runner.py ...``) or a module
(``python -m evals.runner ...``); the bootstrap below puts the worktree root on
``sys.path`` for the script case.
"""
import argparse
import os
import sys

# Allow `python evals/runner.py ...` (script) as well as `-m evals.runner`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.protocol import loader  # noqa: E402
from evals.protocol import runner as protocol_runner  # noqa: E402

# Served-model label per preset (the proxy maps every model name to its single
# configured upstream, so this is for labelling/defaults only). glm51 must run
# serialized (concurrency 1) — it wedges at Running>=2.
PRESET_MODELS = {
    "glm52": "RedHatAI/GLM-5.2-FP8",
    "kimi27": "moonshotai/Kimi-K2.7",
    "laguna": "poolside/Laguna-M.1-FP8",
    "glm47": "QuantTrio/GLM-4.7-AWQ",
    "glm51": "cyankiwi/GLM-5.1-AWQ-4bit",
}
SERIALIZE_PRESETS = {"glm51"}

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURES = os.path.join(_HERE, "protocol", "fixtures")
DEFAULT_RESULTS_DIR = os.path.join(_HERE, "results")


def cmd_protocol(args: argparse.Namespace) -> int:
    if args.proxy == "off":
        print("error: --proxy off (raw OpenAI endpoint) is an L2 attribution "
              "control, not implemented in L0 v1. Use --proxy on.", file=sys.stderr)
        return 2

    fixtures = loader.load_fixtures(args.fixtures)
    if not fixtures:
        print(f"error: no fixtures found in {args.fixtures}", file=sys.stderr)
        return 2

    concurrency = args.concurrency
    if args.preset in SERIALIZE_PRESETS and concurrency != 1:
        print(f"note: {args.preset} wedges under concurrent decode; "
              f"forcing concurrency=1", file=sys.stderr)
        concurrency = 1

    out_path = args.out or os.path.join(DEFAULT_RESULTS_DIR,
                                        f"protocol-{args.preset}.json")
    cfg = protocol_runner.RunConfig(
        base_url=args.base_url,
        model=args.model or PRESET_MODELS.get(args.preset, "local-eval"),
        preset=args.preset,
        modes=tuple(m.strip() for m in args.mode.split(",") if m.strip()),
        repeat=args.repeat,
        stall=args.stall,
        timeout=args.timeout,
        concurrency=concurrency,
        think=args.think,
    )
    summary = protocol_runner.run(cfg, fixtures, out_path=out_path)
    # Exit non-zero if the invariant (proxy-regression) gate is dirty.
    return 0 if summary["invariants_clean"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evals/runner.py",
                                description="Olivia agentic-eval runner")
    sub = p.add_subparsers(dest="layer", required=True)

    pp = sub.add_parser("protocol", help="L0 tool-call / protocol conformance")
    pp.add_argument("--preset", default="generic",
                    help="model preset label (glm52, kimi27, laguna, glm47, glm51)")
    pp.add_argument("--base-url",
                    default=os.environ.get("OLIVIA_PROXY_URL", "http://localhost:8002"),
                    help="anthropic_proxy.py base URL (default $OLIVIA_PROXY_URL or :8002, "
                         "the proxy's default listen; :8001 is the vllm_proxy batcher)")
    pp.add_argument("--model", default=None,
                    help="override the model string sent in the request")
    pp.add_argument("--proxy", choices=["on", "off"], default="on",
                    help="on = Anthropic /v1/messages path (default); off reserved for L2")
    pp.add_argument("--mode", default="nonstream,stream",
                    help="comma list: nonstream,stream (default both)")
    pp.add_argument("--repeat", type=int, default=1,
                    help="runs per fixture/mode (behavioural checks are stochastic)")
    pp.add_argument("--think", choices=["on", "off", "default"], default="default",
                    help="server thinking state; 'on' also runs requires_thinking fixtures")
    pp.add_argument("--stall", type=float, default=45.0,
                    help="streaming inter-token stall timeout in seconds")
    pp.add_argument("--timeout", type=float, default=600.0,
                    help="non-stream overall request timeout in seconds")
    pp.add_argument("--concurrency", type=int, default=1,
                    help="parallel fixtures (forced to 1 for glm51)")
    pp.add_argument("--fixtures", default=DEFAULT_FIXTURES,
                    help="fixtures directory")
    pp.add_argument("--out", default=None,
                    help="results JSON path (default evals/results/protocol-<preset>.json)")
    pp.set_defaults(func=cmd_protocol)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
