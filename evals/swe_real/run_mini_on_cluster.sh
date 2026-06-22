#!/bin/bash
# L2-real via mini-swe-agent — submit to Olivia's `small` CPU partition. Drives the
# model with the real mini-swe-agent harness (litellm -> the OpenAI endpoint
# directly; no anthropic_proxy, since mini-swe-agent uses bash-in-markdown not
# tool-calls) on the cached django slice, verified by our gold-validated harness.
#
# The OpenAI endpoint is either the durable router (routes by model name) or a
# vLLM node directly. With the router, MODEL's repo id (or a preset name) selects
# the backend, so one job can sweep presets.
#
#   # via the durable multi-model router (recommended):
#   ROUTER_NODE=c1-5 MODEL=openai/poolside/Laguna-M.1-FP8 PRESET=laguna-mini \
#     sbatch evals/swe_real/run_mini_on_cluster.sh
#   # or straight to a single vLLM node:
#   GPU_NODE=gpu-1-16 MODEL=openai/poolside/Laguna-M.1-FP8 PRESET=laguna-mini \
#     sbatch evals/swe_real/run_mini_on_cluster.sh
#   # optional: ONLY=django__django-15851,... STEPLIMIT=60
#SBATCH --partition=small
#SBATCH --account=nn10104k
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=swe-mini
#SBATCH --output=/cluster/work/projects/nn10104k/swe/swe-mini-%j.log
set -euo pipefail

WORK=/cluster/work/projects/nn10104k/swe
REPO="$WORK/agentic-evals"
SIF="$WORK/python311.sif"
ROUTER_NODE="${ROUTER_NODE:-}"
GPU_NODE="${GPU_NODE:-}"
MODEL="${MODEL:-openai/poolside/Laguna-M.1-FP8}"
PRESET="${PRESET:-laguna-mini}"
ONLY="${ONLY:-}"
STEPLIMIT="${STEPLIMIT:-60}"

# Endpoint: prefer the durable router (port 8080), else a direct vLLM node (:8000).
if [ -n "$ROUTER_NODE" ]; then
    ENDPOINT_HOST="$ROUTER_NODE"; ENDPOINT_PORT=8080; ENDPOINT_KIND=router
elif [ -n "$GPU_NODE" ]; then
    ENDPOINT_HOST="$GPU_NODE"; ENDPOINT_PORT=8000; ENDPOINT_KIND="vLLM-direct"
else
    echo "set ROUTER_NODE (durable router on small, recommended) or GPU_NODE (direct vLLM node)" >&2
    exit 1
fi

echo "[$(date +%T)] swe-mini: node=$(hostname) endpoint=$ENDPOINT_HOST:$ENDPOINT_PORT ($ENDPOINT_KIND) model=$MODEL preset=$PRESET only=${ONLY:-all}"

apptainer exec --cleanenv --bind /cluster/work "$SIF" bash -c "
set -e
cd '$REPO'
# compute-node squid proxy hijacks localhost/intra-cluster; bypass for the endpoint
export no_proxy='localhost,127.0.0.1,$ENDPOINT_HOST'
export NO_PROXY=\"\$no_proxy\"
# litellm -> OpenAI endpoint ($ENDPOINT_KIND) directly
export OPENAI_API_BASE='http://$ENDPOINT_HOST:$ENDPOINT_PORT/v1'
export OPENAI_API_KEY=dummy
export MSWEA_SILENT_STARTUP=1
export MSWEA_COST_TRACKING=ignore_errors   # self-hosted model isn't priced in litellm
ONLY_ARG=''; [ -n '$ONLY' ] && ONLY_ARG='--only $ONLY'
'$WORK/miniswe-venv/bin/python' evals/swe_real/mini_runner.py --model '$MODEL' \
    --preset '$PRESET' --work '$WORK' --step-limit '$STEPLIMIT' \$ONLY_ARG
"
echo "[$(date +%T)] swe-mini done"
