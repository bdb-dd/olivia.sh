#!/bin/bash
# usage: ctx_ladder.sh <jobid> <model> <outprefix> <chars_per_token> <ctx1,ctx2,...> <levels>
#
# Self-driving context ladder: waits for /health, runs each context x concurrency
# cell, then CANCELS THE JOB when finished.
#
# The self-cancel is the whole point. Job 2038105 served correctly at a 512K
# window and then idled to its full 45-minute TIMEOUT because the ladder was being
# driven by hand from a session that was not attached -- roughly 3 GPU-hours spent
# for zero rows. A benchmark that needs a babysitter will eventually be left
# unattended, so this one ends itself.
JOB=$1; MODEL=$2; OUT=$3; CPT=${4:-5.8}; CTXS=${5:-16000,100000}; LEVELS=${6:-1,16}
D=/cluster/work/projects/nn10104k/containers
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy
cd "$D" || exit 1
job_alive() { squeue -j "$JOB" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|COMPLETING"; }
for i in $(seq 1 240); do
  job_alive || { echo "JOB $JOB gone before start"; exit 1; }
  N=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); [ -n "$N" ] && break; sleep 10
done
NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); echo "node=$NODE"
code=000
for i in $(seq 1 240); do
  job_alive || { echo "JOB $JOB DIED waiting for health"; exit 1; }
  code=$(curl -s --noproxy "*" -m 5 -o /dev/null -w "%{http_code}" "http://$NODE:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && break; sleep 15
done
echo "health=$code"
if [ "$code" != "200" ]; then echo "NEVER READY"; scancel "$JOB"; exit 1; fi
: > "${OUT}_ctx.csv"
for CTX in $(echo "$CTXS" | tr "," " "); do
  echo "### ctx_target=$CTX" | tee -a "${OUT}_ctx.csv"
  python3 bench_sweep.py --url "http://$NODE:8000" --model "$MODEL" \
      --levels "$LEVELS" --max-tokens 512 --prompt-tokens "$CTX" \
      --chars-per-token "$CPT" --timeout 1800 2>>"${OUT}_ctx.err" | tee -a "${OUT}_ctx.csv"
done
echo "=== LADDER COMPLETE - cancelling $JOB ==="
scancel "$JOB"
