#!/usr/bin/env bash
# Pre-submit checks for the shell infrastructure.
#
# `bash -n` is necessary and NOT sufficient. Both bugs below have cost real
# allocation, and neither is a syntax error:
#
#   * a nested Python heredoc -- the SHELL is valid, the PYTHON inside is not.
#     Latent on main for ~10 weeks; blocked every container build.
#     -> check_heredoc_py.py compiles every embedded python heredoc.
#
#   * a per-model knob referenced BEFORE it is assigned. Fatal under
#     `set -euo pipefail`, but only when the user has not exported it, i.e. the
#     common case. Killed job 2694557 one second after it finally got a node.
#     -> check_unbound_order.py compares first-use against first-assignment.
#
# Run this before every sbatch that exercises a change to these scripts.
set -uo pipefail
cd "$(dirname "$0")"
rc=0

echo "== bash -n =="
for f in run_vllm_server.sh build_vllm_gh200.sh olivia.sh; do
    if bash -n "$f" 2>&1; then echo "  OK    $f"; else echo "  FAIL  $f"; rc=1; fi
done

echo "== embedded python heredocs =="
python3 -I check_heredoc_py.py build_vllm_gh200.sh run_vllm_server.sh olivia.sh || rc=1

echo "== per-model knob use-before-assign =="
python3 -I check_unbound_order.py run_vllm_server.sh || rc=1

echo ""
[[ $rc -eq 0 ]] && echo "ALL CHECKS PASSED" || echo "CHECKS FAILED — do not submit"
exit $rc
