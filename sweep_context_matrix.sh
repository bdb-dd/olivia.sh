#!/bin/bash
# usage: sweep_context_matrix.sh <jobid> <model> <outprefix>
#
# Context x concurrency matrix against ONE live server, then cancel the job.
#
# Why a matrix rather than sweep_when_ready.sh's single --prompt-tokens: that
# script takes one context size and cancels on exit, so covering N contexts
# would cost N jobs and N weight loads. This walks the matrix on one server.
#
# CONCURRENCY IS PAIRED TO CONTEXT, deliberately. The KV pool is the binding
# constraint on this shape -- dsv41 measured 229,815 tokens, so the reachable
# concurrency is pool/context: ~224 at 1K but only ~1.75 at 131K. Asking for 64
# streams at 128K does not measure throughput, it measures thrashing. Each row
# below stays inside what the pool can actually hold.
#
# Inherits sweep_when_ready.sh's two hard-won rules:
#   1. `trap ... EXIT` cancels the job on EVERY path. A benchmark that only
#      cancels on success leaks the remaining allocation whenever it crashes --
#      4-12 GPU-h/hour here.
#   2. Liveness is checked separately from health, so a job that died 3 minutes
#      in reports "job gone" rather than polling a corpse for an hour.
set -uo pipefail
JOB=$1; MODEL=$2; OUT=$3
D=/cluster/work/projects/nn10104k/containers
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy
cd "$D" || exit 1

# context:levels -- see the pairing rationale above
MATRIX=${MATRIX:-"1024:1,4,16,64 8192:1,4,16 32768:1,2,4 131072:1"}
MAXTOK=${MAXTOK:-200}

cleanup() { rc=$?; echo "[matrix] exiting (rc=$rc) — cancelling job $JOB"; scancel "$JOB" 2>/dev/null; }
trap cleanup EXIT

job_alive() { squeue -j "$JOB" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|COMPLETING"; }

# READY_WAIT is generous ON PURPOSE. The first version hardcoded 3600s, which
# means the driver CANCELS OUR OWN JOB (via the EXIT trap) when the cluster is
# merely busy rather than broken. Observed 2026-11-03: four nodes had >=4 free
# GPUs and none was usable -- two RESERVED for other accounts, two IDLE+PLANNED
# and held for higher-priority jobs -- so a devel job at priority 20019 simply
# waited. Walltime counts from job START, so waiting longer costs nothing;
# cancelling costs a resubmit and another queue round.
READY_WAIT=${READY_WAIT:-7200}
NODE=""
_iters=$((READY_WAIT / 10))
for i in $(seq 1 "$_iters"); do
  job_alive || { echo "[matrix] job $JOB gone before it was ready"; exit 1; }
  NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null | tr -d ' ')
  if [ -n "$NODE" ] && curl -sf --max-time 5 "http://${NODE}:8000/health" >/dev/null 2>&1; then
    echo "[matrix] server healthy on $NODE after ~$((i*10))s"; break
  fi
  # Heartbeat every 5 min so a long wait is visibly a WAIT, not a hang.
  if [ $((i % 30)) -eq 0 ]; then
    echo "[matrix] still waiting (~$((i*10))s/${READY_WAIT}s) state=$(squeue -j "$JOB" -h -o '%T' 2>/dev/null) node=${NODE:-none}"
  fi
  sleep 10
done
[ -n "$NODE" ] || { echo "[matrix] never resolved a node"; exit 1; }
curl -sf --max-time 5 "http://${NODE}:8000/health" >/dev/null 2>&1 \
  || { echo "[matrix] SERVER NEVER BECAME HEALTHY (job still alive: $(job_alive && echo yes || echo no))"; exit 1; }

URL="http://${NODE}:8000"
echo "[matrix] warmup (DISCARDED)"
python3 bench_sweep.py --url "$URL" --model "$MODEL" --levels 1 --max-tokens 32 >/dev/null 2>&1 || true

for cell in $MATRIX; do
  ptok=${cell%%:*}; levels=${cell##*:}
  echo ""
  echo "[matrix] ===== prompt_tokens=${ptok}  levels=${levels} ====="
  job_alive || { echo "[matrix] job died mid-matrix at ptok=${ptok}"; exit 1; }
  python3 bench_sweep.py --url "$URL" --model "$MODEL" \
      --levels "$levels" --max-tokens "$MAXTOK" \
      --prompt-tokens "$ptok" --chars-per-token 4.0 \
      2>&1 | tee -a "${OUT}_ptok${ptok}.txt"
done
echo ""
echo "[matrix] complete — results at ${OUT}_ptok*.txt"
