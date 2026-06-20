#!/bin/bash
# L2-real via mini-swe-agent — submit to Olivia's `small` CPU partition. Drives the
# model with the real mini-swe-agent harness (litellm -> vLLM's OpenAI endpoint
# directly; no proxy, since mini-swe-agent uses bash-in-markdown not tool-calls) on
# the cached django slice, verified by our gold-validated harness.
#
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
GPU_NODE="${GPU_NODE:?set GPU_NODE to the vLLM node, e.g. gpu-1-16}"
MODEL="${MODEL:-openai/poolside/Laguna-M.1-FP8}"
PRESET="${PRESET:-laguna-mini}"
ONLY="${ONLY:-}"
STEPLIMIT="${STEPLIMIT:-60}"

echo "[$(date +%T)] swe-mini: node=$(hostname) gpu=$GPU_NODE model=$MODEL preset=$PRESET only=${ONLY:-all}"

apptainer exec --cleanenv --bind /cluster/work "$SIF" bash -c "
set -e
cd '$REPO'
# compute-node squid proxy hijacks localhost/intra-cluster; bypass for the vLLM node
export no_proxy='localhost,127.0.0.1,$GPU_NODE'
export NO_PROXY=\"\$no_proxy\"
# litellm -> vLLM OpenAI endpoint directly
export OPENAI_API_BASE='http://$GPU_NODE:8000/v1'
export OPENAI_API_KEY=dummy
export MSWEA_SILENT_STARTUP=1
export MSWEA_COST_TRACKING=ignore_errors   # self-hosted model isn't priced in litellm
ONLY_ARG=''; [ -n '$ONLY' ] && ONLY_ARG='--only $ONLY'
'$WORK/miniswe-venv/bin/python' evals/swe_real/mini_runner.py --model '$MODEL' \
    --preset '$PRESET' --work '$WORK' --step-limit '$STEPLIMIT' \$ONLY_ARG
"
echo "[$(date +%T)] swe-mini done"
