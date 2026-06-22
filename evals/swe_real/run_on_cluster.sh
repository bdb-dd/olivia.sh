#!/bin/bash
# L2-real SWE-bench runner — submit to Olivia's `small` CPU partition (x86 +
# internet + reaches the vLLM node), entirely within Sigma2 usage policy.
#
# Inside a python:3.11 apptainer it starts anthropic_proxy (-> the durable router
# on the `small` partition, or a single GPU node directly) and runs
# evals/swe_real/runner.py against the cached django slice.
#
#   # via the durable multi-model router (recommended — one stable host that
#   # routes by model name, so the same job can sweep presets):
#   ROUTER_NODE=c1-5 MODEL=poolside/Laguna-M.1-FP8 PRESET=laguna \
#     sbatch evals/swe_real/run_on_cluster.sh
#   # or straight to a single vLLM node (no router):
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
ROUTER_NODE="${ROUTER_NODE:-}"
GPU_NODE="${GPU_NODE:-}"
MODEL="${MODEL:-poolside/Laguna-M.1-FP8}"
PRESET="${PRESET:-laguna}"
ONLY="${ONLY:-}"
MAXTURNS="${MAXTURNS:-40}"
GOLD="${GOLD:-}"   # set GOLD=1 to apply gold patches instead of the model (harness self-test)
# Local listen port for the in-job anthropic_proxy. Override (e.g. 8012) when two
# eval jobs may share a `small` node — they share the node's localhost, so a fixed
# 8002 would collide. The router is the single shared endpoint; this is node-local.
PROXY_PORT="${PROXY_PORT:-8002}"

# Endpoint: prefer the durable router (port 8080; resolves $MODEL — preset name,
# alias, or served repo id — to the live backend), else a direct vLLM node (:8000).
if [ -n "$ROUTER_NODE" ]; then
    ENDPOINT_HOST="$ROUTER_NODE"; ENDPOINT_PORT=8080; ENDPOINT_KIND=router
elif [ -n "$GPU_NODE" ]; then
    ENDPOINT_HOST="$GPU_NODE"; ENDPOINT_PORT=8000; ENDPOINT_KIND="vLLM-direct"
else
    echo "set ROUTER_NODE (durable router on small, recommended) or GPU_NODE (direct vLLM node)" >&2
    exit 1
fi

echo "[$(date +%T)] swe-eval: node=$(hostname) endpoint=$ENDPOINT_HOST:$ENDPOINT_PORT ($ENDPOINT_KIND) model=$MODEL preset=$PRESET only=${ONLY:-all}"

apptainer exec --cleanenv --bind /cluster/work "$SIF" bash -c "
set -e
cd '$REPO'
# Compute nodes inherit http_proxy=http://uan03:3128 (squid for internet). urllib
# would route localhost:$PROXY_PORT and the endpoint host through it (-> 503). Bypass
# the proxy for local + the endpoint (router or vLLM node); keep it for git/pip.
export no_proxy='localhost,127.0.0.1,$ENDPOINT_HOST'
export NO_PROXY=\"\$no_proxy\"
# proxy venv (aiohttp); the runner itself is stdlib-only
[ -d '$WORK/proxyvenv' ] || python -m venv '$WORK/proxyvenv'
'$WORK/proxyvenv/bin/pip' install -q aiohttp
# start the Anthropic proxy pointed at the endpoint ($ENDPOINT_KIND; no tunnel needed)
'$WORK/proxyvenv/bin/python' anthropic_proxy.py --model '$MODEL' \
    --upstream 'http://$ENDPOINT_HOST:$ENDPOINT_PORT' --listen-port $PROXY_PORT > '$WORK/proxy-$PRESET.log' 2>&1 &
PROXY=\$!
# wait for the proxy to accept connections
python - <<PY
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://localhost:$PROXY_PORT/v1/models', timeout=3); print('proxy up'); break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('proxy did not come up')
PY
ONLY_ARG=''; [ -n '$ONLY' ] && ONLY_ARG='--only $ONLY'
GOLD_ARG=''; [ -n '$GOLD' ] && GOLD_ARG='--gold'
python evals/swe_real/runner.py --base-url http://localhost:$PROXY_PORT \
    --model '$MODEL' --preset '$PRESET' --work '$WORK' --max-turns '$MAXTURNS' \$ONLY_ARG \$GOLD_ARG
kill \$PROXY 2>/dev/null || true
"
echo "[$(date +%T)] swe-eval done"
