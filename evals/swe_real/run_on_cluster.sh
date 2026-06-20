#!/bin/bash
# L2-real SWE-bench runner — submit to Olivia's `small` CPU partition (x86 +
# internet + reaches the vLLM node), entirely within Sigma2 usage policy.
#
# Inside a python:3.11 apptainer it starts anthropic_proxy (-> the GPU node's
# vLLM) and runs evals/swe_real/runner.py against the cached django slice.
#
#   GPU_NODE=gpu-1-16 MODEL=poolside/Laguna-M.1-FP8 PRESET=laguna \
#     sbatch evals/swe_real/run_on_cluster.sh
#   # optional: ONLY=django__django-16950,django__django-16502  MAXTURNS=40
#SBATCH --partition=small
#SBATCH --account=nn10104k
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=swe-eval
#SBATCH --output=/cluster/work/projects/nn10104k/swe/swe-eval-%j.log
set -euo pipefail

WORK=/cluster/work/projects/nn10104k/swe
REPO="$WORK/agentic-evals"
SIF="$WORK/python311.sif"
GPU_NODE="${GPU_NODE:?set GPU_NODE to the vLLM node, e.g. gpu-1-16}"
MODEL="${MODEL:-poolside/Laguna-M.1-FP8}"
PRESET="${PRESET:-laguna}"
ONLY="${ONLY:-}"
MAXTURNS="${MAXTURNS:-40}"
GOLD="${GOLD:-}"   # set GOLD=1 to apply gold patches instead of the model (harness self-test)

echo "[$(date +%T)] swe-eval: node=$(hostname) gpu=$GPU_NODE model=$MODEL preset=$PRESET only=${ONLY:-all}"

apptainer exec --cleanenv --bind /cluster/work "$SIF" bash -c "
set -e
cd '$REPO'
# Compute nodes inherit http_proxy=http://uan03:3128 (squid for internet). urllib
# would route localhost:8002 and the GPU node through it (-> 503). Bypass the
# proxy for local + the vLLM node; keep it for git/pip (github/pypi).
export no_proxy='localhost,127.0.0.1,$GPU_NODE'
export NO_PROXY=\"\$no_proxy\"
# proxy venv (aiohttp); the runner itself is stdlib-only
[ -d '$WORK/proxyvenv' ] || python -m venv '$WORK/proxyvenv'
'$WORK/proxyvenv/bin/pip' install -q aiohttp
# start the Anthropic proxy pointed straight at the vLLM node (no tunnel needed)
'$WORK/proxyvenv/bin/python' anthropic_proxy.py --model '$MODEL' \
    --upstream 'http://$GPU_NODE:8000' --listen-port 8002 > '$WORK/proxy-$PRESET.log' 2>&1 &
PROXY=\$!
# wait for the proxy to accept connections
python - <<'PY'
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://localhost:8002/v1/models', timeout=3); print('proxy up'); break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('proxy did not come up')
PY
ONLY_ARG=''; [ -n '$ONLY' ] && ONLY_ARG='--only $ONLY'
GOLD_ARG=''; [ -n '$GOLD' ] && GOLD_ARG='--gold'
python evals/swe_real/runner.py --base-url http://localhost:8002 \
    --model '$MODEL' --preset '$PRESET' --work '$WORK' --max-turns '$MAXTURNS' \$ONLY_ARG \$GOLD_ARG
kill \$PROXY 2>/dev/null || true
"
echo "[$(date +%T)] swe-eval done"
