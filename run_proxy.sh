#!/bin/bash
#SBATCH --job-name=olivia-router
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/olivia_router_%j.log
#SBATCH --error=logs/olivia_router_%j.log
# =============================================================================
# Durable multi-model vLLM router on Olivia's `small` (CPU) partition.
#
# Runs model_router.py — a CPU-only reverse proxy that fans one stable endpoint
# out to whichever GPU (`accel`) inference jobs are running, routing by the
# request's `model` field (preset name / alias / served repo id) to the live
# backend node found via squeue. See plans/proposed/small_partition_proxy.md.
#
# Why the small partition: it is the policy-correct home for a long-lived
# process (NRIS: "always use the queue system"), allows partial-node allocation
# (this asks for just 2 CPUs / 4 GiB), and permits up to a 7-day walltime — far
# more durable than the ~8 h GPU job, and a clean replacement for the login-node
# relay. Request minimal resources and stop it when idle (idle CPU jobs still
# bill their reservation).
#
# Submitted by `olivia.sh proxy start`, which deploys the router files and sets:
#   ROUTER_DIR   - dir holding model_router.py / presets.py / presets.json / vllm_proxy.py
#   ROUTER_PORT  - listen port (default 8080)
# Optional:
#   ROUTER_BACKEND_PORT   - port each vLLM server listens on (default 8000)
#   ROUTER_EMPTY_TIMEOUT  - auto-shutdown after Ns with no GPU servers (default 1800; 0 disables)
#   ROUTER_VENV           - venv dir for aiohttp (default: $HOME/.olivia/router-venv)
#   OLIVIA_PROXY_TOKEN    - require this bearer token / x-api-key
#   http_proxy/https_proxy - compute-node outbound proxy for pip (default NRIS proxy)
# =============================================================================
set -uo pipefail

ROUTER_DIR="${ROUTER_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROUTER_PORT="${ROUTER_PORT:-8080}"
ROUTER_BACKEND_PORT="${ROUTER_BACKEND_PORT:-8000}"   # port each vLLM server listens on
ROUTER_EMPTY_TIMEOUT="${ROUTER_EMPTY_TIMEOUT:-1800}" # auto-shutdown after Ns with no GPU servers (0 disables)
ROUTER_VENV="${ROUTER_VENV:-$HOME/.olivia/router-venv}"

# The router needs Python >= 3.9 (model_router.py uses `from __future__ import
# annotations` + modern asyncio). On Olivia's `small` nodes the system `python3`
# is 3.6 — too old — but `python3.11` / `python3.12` are present at /usr/bin.
# Autodetect a new-enough interpreter (override with PYTHON=...).
_py_ok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)' >/dev/null 2>&1; }
if [[ -z "${PYTHON:-}" ]]; then
    for _cand in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "${_cand}" >/dev/null 2>&1 && _py_ok "${_cand}"; then PYTHON="${_cand}"; break; fi
    done
fi
if [[ -z "${PYTHON:-}" ]] || ! _py_ok "${PYTHON}"; then
    echo "ERROR: no Python >= 3.9 found (system python3 is $(python3 --version 2>&1)). Set PYTHON=<path>." >&2
    exit 1
fi

# Compute nodes reach the internet only via the NRIS HTTP proxy; needed for the
# one-time `pip install aiohttp`. Pre-set values win (e.g. site changes the IP).
export http_proxy="${http_proxy:-http://10.63.2.48:3128/}"
export https_proxy="${https_proxy:-http://10.63.2.48:3128/}"

echo "================================================================"
echo " Olivia model router (small partition)"
echo "================================================================"
echo " Node:        $(hostname)"
echo " Router dir:  ${ROUTER_DIR}"
echo " Listen port: ${ROUTER_PORT}  (backends: vLLM :${ROUTER_BACKEND_PORT})"
echo " Spindown:    $([ "${ROUTER_EMPTY_TIMEOUT}" -gt 0 ] 2>/dev/null && echo "after ${ROUTER_EMPTY_TIMEOUT}s with no GPU servers" || echo 'disabled')"
echo " Venv:        ${ROUTER_VENV}"
echo " Python:      $(${PYTHON} --version 2>&1) (${PYTHON})"
echo " Auth:        $([ -n "${OLIVIA_PROXY_TOKEN:-}" ] && echo 'token required' || echo 'open (internal network)')"
echo "================================================================"

if [[ ! -f "${ROUTER_DIR}/model_router.py" ]]; then
    echo "ERROR: model_router.py not found in ROUTER_DIR=${ROUTER_DIR}" >&2
    echo "       Deploy with: ./olivia.sh proxy deploy" >&2
    exit 1
fi

# -- ensure an aiohttp venv (tiny; reused across restarts) --------------------
# Mirrors the prefetch venv pattern: a throwaway stdlib venv with one dep. The
# router code itself is stdlib + aiohttp only.
# Recreate the venv if it's missing or was built from a too-old python (e.g. a
# stale 3.6 venv from before this autodetect — its bin/python would still fail).
if [[ -x "${ROUTER_VENV}/bin/python" ]] && ! _py_ok "${ROUTER_VENV}/bin/python"; then
    echo "[setup] existing venv python is too old ($("${ROUTER_VENV}/bin/python" --version 2>&1)) — recreating"
    rm -rf "${ROUTER_VENV}"
fi
if [[ ! -x "${ROUTER_VENV}/bin/python" ]]; then
    echo "[setup] creating router venv at ${ROUTER_VENV} (with ${PYTHON})"
    if ! "${PYTHON}" -m venv "${ROUTER_VENV}"; then
        echo "ERROR: failed to create venv (is ${PYTHON} + venv available on the node?)" >&2
        exit 1
    fi
fi
VENV_PY="${ROUTER_VENV}/bin/python"
if ! "${VENV_PY}" -c "import aiohttp" 2>/dev/null; then
    echo "[setup] installing aiohttp into router venv (via ${https_proxy})"
    if ! "${VENV_PY}" -m pip install --quiet --upgrade pip aiohttp; then
        echo "ERROR: pip install aiohttp failed (check http_proxy reachability)" >&2
        exit 1
    fi
fi
"${VENV_PY}" -c "import aiohttp; print('[setup] aiohttp', aiohttp.__version__)"

# -- launch the router --------------------------------------------------------
# exec so SLURM signals (scancel, walltime) reach the python process directly.
cd "${ROUTER_DIR}"
echo "[run] starting router on $(hostname):${ROUTER_PORT}"
exec "${VENV_PY}" "${ROUTER_DIR}/model_router.py" \
    --listen-host 0.0.0.0 \
    --listen-port "${ROUTER_PORT}" \
    --backend-port "${ROUTER_BACKEND_PORT}" \
    --empty-timeout "${ROUTER_EMPTY_TIMEOUT}"
