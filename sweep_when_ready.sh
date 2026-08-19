#!/bin/bash
# usage: sweep_when_ready.sh <jobid> <model> <outprefix> [levels] [prompt_tokens] [chars_per_token]
#
# Waits for a vLLM server to answer /health, runs a DISCARDED warmup pass and then
# a measured pass, and ALWAYS cancels the job on the way out.
#
# Two hard-won rules are encoded here:
#
#  1. `trap ... EXIT` rather than a scancel at the bottom. A benchmark that only
#     cancels on the happy path leaks the whole remaining allocation whenever it
#     crashes, is killed, or exits early. GPU jobs on this cluster cost 4-12
#     GPU-h/hour, so one leaked run erases a day of careful budgeting.
#
#  2. Liveness is checked separately from health. An earlier version polled only
#     /health, so when a job died 3 minutes in it kept polling a dead node for the
#     full hour and reported "SERVER NEVER READY" — which reads as "slow start"
#     rather than "job died 55 minutes ago".
JOB=$1; MODEL=$2; OUT=$3; LEVELS=${4:-1,2,4,8,16,32,64}; PTOK=${5:-0}; CPT=${6:-4.0}
D=/cluster/work/projects/nn10104k/containers
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy
cd "$D" || exit 1

cleanup() { rc=$?; echo "[sweep] exiting (rc=$rc) — cancelling job $JOB"; scancel "$JOB" 2>/dev/null; }
trap cleanup EXIT

job_alive() { squeue -j "$JOB" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|COMPLETING"; }
PT_ARG=""; [ "$PTOK" -gt 0 ] 2>/dev/null && PT_ARG="--prompt-tokens $PTOK --chars-per-token $CPT"

for i in $(seq 1 240); do
  job_alive || { echo "[sweep] job $JOB gone before it started"; exit 1; }
  N=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); [ -n "$N" ] && break; sleep 10
done
NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); echo "node=$NODE"

code=000
for i in $(seq 1 240); do
  job_alive || { echo "[sweep] job $JOB DIED while waiting for /health (last code=$code)"; exit 1; }
  code=$(curl -s --noproxy "*" -m 5 -o /dev/null -w "%{http_code}" "http://$NODE:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && break; sleep 15
done
echo "health=$code after ~$((i*15))s"
[ "$code" = "200" ] || { echo "[sweep] SERVER NEVER READY"; exit 1; }

echo "=== WARMUP PASS (discard) ==="
python3 bench_sweep.py --url "http://$NODE:8000" --model "$MODEL" --levels "$LEVELS" \
    --max-tokens 512 $PT_ARG > "${OUT}_cold.csv" 2>"${OUT}_cold.err"
echo "=== MEASURED PASS (warm) ==="
python3 bench_sweep.py --url "http://$NODE:8000" --model "$MODEL" --levels "$LEVELS" \
    --max-tokens 512 $PT_ARG > "${OUT}_warm.csv" 2>"${OUT}_warm.err"
echo "--- warm ---"; cat "${OUT}_warm.csv"
echo "=== SWEEP COMPLETE ==="
