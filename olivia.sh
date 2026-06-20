#!/bin/bash
# =============================================================================
# olivia.sh - Olivia HPC Cluster Management Tool
# =============================================================================
# Unified CLI for managing vLLM builds, servers, and chat sessions on Olivia.
# Uses SSH ControlMaster for single 2FA authentication per session.
#
# Usage:
#   ./olivia.sh                    # Show help and available commands
#   ./olivia.sh chat               # Connect to vLLM server and start chat
#   ./olivia.sh build              # List presets or build a container
#   ./olivia.sh server             # Manage vLLM server (restart, logs)
#   ./olivia.sh tunnel             # Manage SSH tunnel
#   ./olivia.sh status             # Show cluster and connection status
#
# Only ONE 2FA prompt per session - uses SSH ControlMaster.
# =============================================================================

set -euo pipefail

# Script version
OLIVIA_VERSION="1.0.0"

# =============================================================================
# Configuration
# =============================================================================

REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT=8000
# Local forward defaults to 8003, NOT 8000: another local dev service (an nginx)
# may bind :8000 and silently intercept the tunnel if the SSH forward ever drops
# (you'd talk to it, not vLLM, with no error). 8003 keeps the olivia 800x family
# together (8001 vllm_proxy batcher, 8002 anthropic_proxy, 8003 tunnel).
# Override with LOCAL_PORT=<n>. REMOTE_PORT (cluster-side vLLM) stays 8000.
LOCAL_PORT="${LOCAL_PORT:-8003}"
# Match the SLURM job name that run_vllm_server.sh's #SBATCH --job-name sets.
# Using the full `vllm-server` here (not just `vllm`) keeps us from picking up
# in-flight build jobs, whose name is `build-vllm-gh200` and also contains
# "vllm" — matching them would make `server logs` tail non-existent files and,
# worse, make `server cancel` kill a running build.
JOB_NAME_PATTERN="vllm-server"

# Remote paths
REMOTE_CONTAINER_DIR="${REMOTE_CONTAINER_DIR:-}"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SERVER_SCRIPT="${SCRIPT_DIR}/run_vllm_server.sh"
LOCAL_BUILD_SCRIPT="${SCRIPT_DIR}/build_vllm_gh200.sh"
LOCAL_PATCHES_DIR="${SCRIPT_DIR}/patches"
# Chat-template override for GLM-5.1 (see run_vllm_server.sh for why).
LOCAL_GLM51_CHAT_TEMPLATE="${SCRIPT_DIR}/templates/glm51_chat_template.jinja"
CHAT_SCRIPT="${SCRIPT_DIR}/chat_devstral.py"

# --- Per-branch deploy isolation -------------------------------------------
# The deployed cluster scripts (run_vllm_server.sh, build_vllm_gh200.sh) + aux
# files (GLM-5.1 chat template, patches/) used to live at ONE shared path under
# REMOTE_CONTAINER_DIR. Concurrent agents on different preset branches deploy
# DIVERGENT versions (each adds its own IS_<preset> detection), so they clobbered
# each other in the deploy→sbatch window — SLURM snapshots the batch script at
# submit, so whoever deployed last before your submit wins, and your job runs the
# wrong preset's config. Fix: each branch deploys to deploys/<DEPLOY_KEY>/ and
# submits from there, so divergent versions never collide. Sandboxes, logs, and
# the $PWD/cache compile cache stay SHARED (submit cwd is still
# REMOTE_CONTAINER_DIR); only the small scripts/aux become per-branch. DEPLOY_KEY
# defaults to the local git branch; override to share (e.g. DEPLOY_KEY=main) or
# split deploy dirs explicitly.
DEPLOY_KEY="${DEPLOY_KEY:-$(git -C "${SCRIPT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-' | tr -cd 'A-Za-z0-9._-')}"
DEPLOY_KEY="${DEPLOY_KEY:-default}"
DEPLOY_DIR="${REMOTE_CONTAINER_DIR}/deploys/${DEPLOY_KEY}"

# SSH control socket
SSH_CONTROL_DIR="${HOME}/.ssh/controls"
SSH_CONTROL_SOCKET="${SSH_CONTROL_DIR}/olivia-${REMOTE_USER}@${REMOTE_HOST}"

# SSH connection durability
# Olivia requires password + OTP on every *new* interactive SSH connection
# (keys do not bypass 2FA). ControlMaster lets us pay that 2FA once and then
# multiplex; keeping the master alive as long as possible is what avoids
# re-authenticating. A long ControlPersist + keepalives prevents idle/NAT drops.
# Unattended auto-reconnect (autossh) is intentionally NOT used: a fresh
# connection needs an interactive OTP, which cannot be supplied in the
# background. Use `./olivia.sh reconnect` to re-establish after a real drop.
SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-12h}"
SSH_ALIVE_INTERVAL="${SSH_ALIVE_INTERVAL:-30}"
SSH_ALIVE_COUNT="${SSH_ALIVE_COUNT:-3}"

# Optional login-node relay (opt-in via LOGIN_PROXY=1 or `--login-proxy`).
# The laptop forwards to a FIXED login-node port and a lightweight user-space
# relay on the login node follows the GPU node across job restarts, so node
# churn never tears down the local forward. NOTE: this runs a long-lived
# process on the shared login node - check the NRIS acceptable-use policy.
LOGIN_PROXY="${LOGIN_PROXY:-0}"
LOGIN_PROXY_PORT="${LOGIN_PROXY_PORT:-18000}"

# State files
TUNNEL_NODE_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.node"
TUNNEL_TARGET_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.target"

# =============================================================================
# Helper functions
# =============================================================================

info() {
    echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m" >&2
}

success() {
    echo -e "\033[1;32m==>\033[0m \033[1m$*\033[0m" >&2
}

warn() {
    echo -e "\033[1;33m==>\033[0m \033[1m$*\033[0m" >&2
}

error() {
    echo -e "\033[1;31m==>\033[0m \033[1m$*\033[0m" >&2
}

header() {
    echo ""
    echo -e "\033[1;35m$*\033[0m" >&2
    echo ""
}

require_remote_config() {
    if [[ -z "${REMOTE_HOST}" ]]; then
        error "REMOTE_HOST is not set"
        echo "    Set REMOTE_HOST to your cluster login hostname (e.g. export REMOTE_HOST=mycluster)" >&2
        return 1
    fi
    if [[ -z "${REMOTE_CONTAINER_DIR}" ]]; then
        error "REMOTE_CONTAINER_DIR is not set"
        echo "    Set REMOTE_CONTAINER_DIR to the container directory on the cluster" >&2
        echo "    (e.g. export REMOTE_CONTAINER_DIR=/path/to/containers)" >&2
        return 1
    fi
    return 0
}

# Desktop notification helper (macOS/Linux)
# On macOS, prefers terminal-notifier if installed (brew install terminal-notifier)
notify() {
    local title="$1"
    local message="$2"
    local sound="${3:-}"  # Optional sound

    if [[ "$(uname)" == "Darwin" ]]; then
        # Play sound directly with afplay (more reliable than notification sounds)
        if [[ -n "$sound" && -f "/System/Library/Sounds/${sound}.aiff" ]]; then
            afplay "/System/Library/Sounds/${sound}.aiff" &>/dev/null &
        fi

        if command -v terminal-notifier &>/dev/null; then
            # terminal-notifier: better macOS integration with custom app attribution
            terminal-notifier -title "$title" -message "$message" -group "olivia-vllm" 2>/dev/null || true
        else
            # Fallback: osascript (appears as "Script Editor")
            osascript -e "display notification \"${message}\" with title \"${title}\"" 2>/dev/null || true
        fi
    elif command -v notify-send &>/dev/null; then
        # Linux: use notify-send
        notify-send "$title" "$message" 2>/dev/null || true
    fi
}

# Re-point LOCAL_PORT and the per-port state files together (keep them in sync).
set_local_port() {
    LOCAL_PORT="$1"
    TUNNEL_NODE_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.node"
    TUNNEL_TARGET_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.target"
}

# SSH helpers
ssh_opts() {
    echo -o "ControlMaster=auto" \
         -o "ControlPath=${SSH_CONTROL_SOCKET}" \
         -o "ControlPersist=${SSH_CONTROL_PERSIST}" \
         -o "ServerAliveInterval=${SSH_ALIVE_INTERVAL}" \
         -o "ServerAliveCountMax=${SSH_ALIVE_COUNT}" \
         -o "TCPKeepAlive=yes"
}

ssh_run() {
    ssh $(ssh_opts) "$@"
}

scp_run() {
    scp -o "ControlPath=${SSH_CONTROL_SOCKET}" "$@"
}

# =============================================================================
# SSH ControlMaster management
# =============================================================================

is_master_alive() {
    ssh -o "ControlPath=${SSH_CONTROL_SOCKET}" -O check "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null
}

ensure_master_connection() {
    require_remote_config || return 1
    mkdir -p "${SSH_CONTROL_DIR}"
    chmod 700 "${SSH_CONTROL_DIR}"

    if is_master_alive; then
        return 0
    fi

    info "Establishing SSH connection to ${REMOTE_HOST}..."
    echo "    (You may be prompted for 2FA)" >&2
    echo "    Connection will persist for ${SSH_CONTROL_PERSIST} of idle time" >&2
    echo "" >&2

    # -f -N -M with ControlPersist leaves a detached master that survives this
    # terminal closing, so a single 2FA covers the whole persist window.
    ssh -f -N -M \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        -o "ControlPersist=${SSH_CONTROL_PERSIST}" \
        -o "ServerAliveInterval=${SSH_ALIVE_INTERVAL}" \
        -o "ServerAliveCountMax=${SSH_ALIVE_COUNT}" \
        -o "TCPKeepAlive=yes" \
        -o "ExitOnForwardFailure=yes" \
        "${REMOTE_USER}@${REMOTE_HOST}"

    sleep 1
    if is_master_alive; then
        success "SSH connection established"
        return 0
    else
        error "Failed to establish SSH connection"
        return 1
    fi
}

close_master_connection() {
    if is_master_alive; then
        ssh -o "ControlPath=${SSH_CONTROL_SOCKET}" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true
    fi
}

# =============================================================================
# Login-node relay (opt-in: LOGIN_PROXY=1)
# =============================================================================
# A tiny user-space TCP relay on the login node that listens on a fixed port
# and forwards to the CURRENT GPU node:REMOTE_PORT. The laptop forward then
# targets the stable login-node port, so when the SLURM job moves to a new node
# we only re-point the relay (cheap, over the existing master) instead of
# tearing down the local forward. The login node can already reach the compute
# node's port directly over the internal network (same path the direct forward
# uses), so no inner SSH hop is needed.

# Deploy the relay script to the login node (idempotent, tiny).
login_proxy_deploy() {
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        'mkdir -p "$HOME/.olivia" && cat > "$HOME/.olivia/relay.py"' <<'PYEOF'
import asyncio
import sys


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(local_reader, local_writer, host, port):
    try:
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
    except Exception:
        local_writer.close()
        return
    await asyncio.gather(
        _pipe(local_reader, remote_writer),
        _pipe(remote_reader, local_writer),
    )


async def _main():
    listen_port = int(sys.argv[1])
    upstream_host = sys.argv[2]
    upstream_port = int(sys.argv[3])
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, upstream_host, upstream_port),
        "127.0.0.1",
        listen_port,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
PYEOF
}

# Start or re-point the relay at the given GPU node.
login_proxy_ensure() {
    local gpu_node="$1"

    if ! login_proxy_deploy; then
        error "Failed to deploy login-node relay script"
        return 1
    fi

    info "Pointing login-node relay (:${LOGIN_PROXY_PORT}) at ${gpu_node}:${REMOTE_PORT}..."

    local out
    if ! out=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" bash -s -- \
        "${LOGIN_PROXY_PORT}" "${gpu_node}" "${REMOTE_PORT}" <<'REMOTE_EOF'
set -u
PORT="$1"; NODE="$2"; UPSTREAM="$3"
DIR="$HOME/.olivia"
PIDFILE="$DIR/relay-$PORT.pid"
LOGFILE="$DIR/relay-$PORT.log"

# Stop any existing relay on this port.
if [ -f "$PIDFILE" ]; then
    OLDPID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
        kill "$OLDPID" 2>/dev/null || true
        sleep 1
    fi
fi

nohup python3 "$DIR/relay.py" "$PORT" "$NODE" "$UPSTREAM" >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
sleep 1
NEWPID=$(cat "$PIDFILE" 2>/dev/null || true)
if [ -n "$NEWPID" ] && kill -0 "$NEWPID" 2>/dev/null; then
    echo "relay-up pid=$NEWPID"
else
    echo "relay-failed"
    tail -n 20 "$LOGFILE" 2>/dev/null || true
    exit 1
fi
REMOTE_EOF
    ); then
        error "Login-node relay failed to start"
        [[ -n "$out" ]] && echo "$out" >&2
        return 1
    fi

    success "Login-node relay running (${out#relay-up })"
    return 0
}

# Stop the relay on LOGIN_PROXY_PORT.
login_proxy_stop() {
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" bash -s -- "${LOGIN_PROXY_PORT}" <<'REMOTE_EOF' 2>/dev/null || true
PORT="$1"
DIR="$HOME/.olivia"
PIDFILE="$DIR/relay-$PORT.pid"
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null || true)
    [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
    rm -f "$PIDFILE"
fi
REMOTE_EOF
}

# =============================================================================
# Tunnel management
# =============================================================================

get_tunnel_node() {
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        cat "${TUNNEL_NODE_FILE}"
    fi
}

is_tunnel_alive() {
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        if lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

kill_tunnel() {
    if [[ ! -f "${TUNNEL_NODE_FILE}" && ! -f "${TUNNEL_TARGET_FILE}" ]]; then
        warn "No active tunnel found"
        return 1
    fi

    local node target
    node=$(cat "${TUNNEL_NODE_FILE}" 2>/dev/null || echo "")
    # The cancel spec must match the forward spec exactly; prefer the stored
    # target (handles login-proxy mode), fall back to the direct node target.
    target=$(cat "${TUNNEL_TARGET_FILE}" 2>/dev/null || echo "${node}:${REMOTE_PORT}")

    info "Canceling port forward (localhost:${LOCAL_PORT} -> ${target})..."
    ssh -O cancel \
        -L "${LOCAL_PORT}:${target}" \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true

    # A localhost:<port> target means we went via the login-node relay; stop it.
    # Only reach out if the master is alive - otherwise ssh_run would open a
    # fresh (2FA-prompting) connection just to tear down.
    if [[ "$target" == localhost:* ]]; then
        LOGIN_PROXY_PORT="${target#localhost:}"
        if is_master_alive; then
            login_proxy_stop
        else
            warn "Master connection down; login-node relay (:${LOGIN_PROXY_PORT}) may still be running"
            echo "    It will be reaped on the next --login-proxy use, or stop it manually:" >&2
            echo "    ssh ${REMOTE_HOST} 'kill \$(cat ~/.olivia/relay-${LOGIN_PROXY_PORT}.pid)'" >&2
        fi
    fi

    rm -f "${TUNNEL_NODE_FILE}" "${TUNNEL_TARGET_FILE}"
    success "Tunnel closed"
    return 0
}

find_vllm_node() {
    local squeue_output
    if ! squeue_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER -h -o '%j %N %T'" 2>/dev/null); then
        error "Failed to query job queue"
        return 1
    fi

    local job_line
    job_line=$(echo "$squeue_output" | grep -i "${JOB_NAME_PATTERN}" | grep "RUNNING" | head -n1) || true

    if [[ -z "$job_line" ]]; then
        return 1
    fi

    local raw_nodelist
    raw_nodelist=$(echo "$job_line" | awk '{print $2}')

    # Multi-node jobs report a SLURM-compressed nodelist (e.g. gpu-1-[90-91])
    # that's not a valid SSH tunnel target. Expand it and return just the
    # head node — that's where run_vllm_server.sh launches `vllm serve`,
    # so it's the only node listening on ${REMOTE_PORT}.
    if [[ "$raw_nodelist" == *"["* || "$raw_nodelist" == *","* ]]; then
        local expanded
        expanded=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
            "scontrol show hostnames '${raw_nodelist}' 2>/dev/null | head -n1" 2>/dev/null) || expanded=""
        if [[ -n "$expanded" ]]; then
            echo "$expanded"
            return 0
        fi
    fi

    echo "$raw_nodelist"
}

setup_tunnel() {
    local gpu_node="$1"

    # The remote end of the local -L forward. In login-proxy mode this is a
    # FIXED login-node port (so node moves don't disturb the local forward); in
    # direct mode it is the GPU node itself.
    local target
    if [[ "${LOGIN_PROXY}" == "1" ]]; then
        target="localhost:${LOGIN_PROXY_PORT}"
    else
        target="${gpu_node}:${REMOTE_PORT}"
    fi

    local stored_target=""
    [[ -f "${TUNNEL_TARGET_FILE}" ]] && stored_target=$(cat "${TUNNEL_TARGET_FILE}")

    if is_tunnel_alive && [[ "$stored_target" == "$target" ]]; then
        local existing_node
        existing_node=$(get_tunnel_node)
        if [[ "${LOGIN_PROXY}" == "1" ]]; then
            # Local forward already up to the fixed login port; just (re)point
            # the relay at the current node - no local teardown needed.
            login_proxy_ensure "$gpu_node" || return 1
            echo "$gpu_node" > "${TUNNEL_NODE_FILE}"
            if [[ "$existing_node" == "$gpu_node" ]]; then
                success "Tunnel active; relay already on ${gpu_node}"
            else
                success "Tunnel active; relay repointed ${existing_node} -> ${gpu_node}"
            fi
            return 0
        elif [[ "$existing_node" == "$gpu_node" ]]; then
            success "Tunnel already active to ${gpu_node} on port ${LOCAL_PORT}"
            return 0
        else
            warn "Tunnel to ${existing_node}, but job is now on ${gpu_node}"
            kill_tunnel
        fi
    elif is_tunnel_alive; then
        # Mode or target changed (e.g. switching to/from login-proxy); rebuild.
        warn "Tunnel target changed; rebuilding"
        kill_tunnel
    fi

    if [[ "${LOGIN_PROXY}" == "1" ]]; then
        login_proxy_ensure "$gpu_node" || return 1
    fi

    info "Setting up tunnel: localhost:${LOCAL_PORT} -> ${target}"

    if ! ssh -O forward \
        -L "${LOCAL_PORT}:${target}" \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        "${REMOTE_USER}@${REMOTE_HOST}" 2>&1; then
        error "Failed to set up tunnel"
        return 1
    fi

    sleep 1
    if ! lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        error "Tunnel failed - port ${LOCAL_PORT} not listening"
        return 1
    fi

    echo "$gpu_node" > "${TUNNEL_NODE_FILE}"
    echo "$target" > "${TUNNEL_TARGET_FILE}"
    success "Tunnel established"
    echo "    Local:  localhost:${LOCAL_PORT}" >&2
    if [[ "${LOGIN_PROXY}" == "1" ]]; then
        echo "    Path:   localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${LOGIN_PROXY_PORT} -> ${gpu_node}:${REMOTE_PORT}" >&2
    else
        echo "    Remote: ${gpu_node}:${REMOTE_PORT}" >&2
    fi
}

# =============================================================================
# Job management helpers
# =============================================================================

find_job_id() {
    local pattern="${1:-${JOB_NAME_PATTERN}}"
    local squeue_output
    if ! squeue_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER -h -o '%j %i %T'" 2>/dev/null); then
        return 1
    fi

    local job_line
    job_line=$(echo "$squeue_output" | grep -i "${pattern}" | grep "RUNNING" | head -n1) || true

    if [[ -z "$job_line" ]]; then
        return 1
    fi

    echo "$job_line" | awk '{print $2}'
}

cancel_job() {
    local pattern="${1:-${JOB_NAME_PATTERN}}"
    info "Looking for running ${pattern} job..."

    local job_id
    if ! job_id=$(find_job_id "$pattern"); then
        warn "No running ${pattern} job found"
        return 0
    fi

    info "Canceling job ${job_id}..."
    if ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "scancel ${job_id}"; then
        error "Failed to cancel job ${job_id}"
        return 1
    fi

    success "Job ${job_id} canceled"
    return 0
}

# Canonical log paths for a vLLM server job.
#
# run_vllm_server.sh writes SLURM wrapper output to
# `vllm_server_<jobid>.log` on *every* job. For multi-node jobs it *additionally*
# starts `vllm serve` under an `srun --output=...<jobid>_head.log`, so all
# useful post-bootstrap output lives in the head log. Single-node jobs never
# create the head log.
#
# Callers should tail *both* files with `tail -F` (which retries files that
# don't exist yet) so they capture Ray-bootstrap output from the wrapper log
# and vLLM output from the head log as each appears. For one-shot reads
# (grep, etc.) use `resolve_server_log` to pick the most informative single
# file.
server_log_wrapper() {
    echo "${REMOTE_CONTAINER_DIR}/logs/vllm_server_${1}.log"
}

server_log_head() {
    echo "${REMOTE_CONTAINER_DIR}/logs/vllm_server_${1}_head.log"
}

resolve_server_log() {
    local job_id="$1"
    local head_log wrap_log
    head_log="$(server_log_head "$job_id")"
    wrap_log="$(server_log_wrapper "$job_id")"
    if ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -s '${head_log}'" 2>/dev/null; then
        echo "$head_log"
    else
        echo "$wrap_log"
    fi
}

tail_job_logs() {
    local log_file="$1"
    local job_id="${2:-}"

    echo "" >&2
    if [[ -n "$job_id" ]]; then
        info "Waiting for job ${job_id} to start..."
    fi
    echo "    (Press Ctrl+C to stop watching - job will continue running)" >&2
    echo "" >&2

    local wait_count=0
    local max_wait=120
    local node_shown=false

    while ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -f '${log_file}'" 2>/dev/null; do
        if [[ -n "$job_id" ]]; then
            local job_info
            job_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -j ${job_id} -h -o '%T %N'" 2>/dev/null) || true

            if [[ -z "$job_info" ]]; then
                error "Job ${job_id} no longer in queue - may have failed"
                return 1
            fi

            local job_state node_name
            job_state=$(echo "$job_info" | awk '{print $1}')
            node_name=$(echo "$job_info" | awk '{print $2}')

            if [[ "$job_state" == "RUNNING" && -n "$node_name" && "$node_shown" == "false" ]]; then
                echo "" >&2
                success "Job running on node: ${node_name}"
                node_shown=true
            else
                printf "\r    Job state: %-10s (waiting for log file...)" "$job_state" >&2
            fi
        fi

        sleep 2
        ((wait_count+=2))

        if [[ $wait_count -ge $max_wait ]]; then
            echo "" >&2
            warn "Timeout waiting for log file"
            echo "    Check manually: ssh ${REMOTE_HOST} 'tail -f ${log_file}'" >&2
            return 0
        fi
    done

    # Show node info if not already shown
    if [[ -n "$job_id" && "$node_shown" == "false" ]]; then
        local node_name
        node_name=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
            "squeue -j ${job_id} -h -o '%N'" 2>/dev/null) || true
        if [[ -n "$node_name" ]]; then
            echo "" >&2
            success "Job running on node: ${node_name}"
        fi
    fi

    echo "" >&2
    success "Job started - tailing logs"
    echo "    Log files: ${log_file} + _head.log (multi-node jobs)" >&2
    echo "" >&2
    echo "==========================================="

    # Tail both wrapper and head log. `tail -F` retries files that don't exist
    # yet, so it naturally picks up the head log once vLLM starts writing to
    # it. Stderr is suppressed to silence "cannot open" / "has appeared"
    # transitions during the wait.
    local head_log
    head_log="$(server_log_head "$job_id")"
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -F '${log_file}' '${head_log}' 2>/dev/null" || true

    echo "" >&2
    info "Log tailing stopped. Job continues running."
}

# =============================================================================
# MODULE: Build
# =============================================================================

deploy_build_script() {
    if [[ ! -f "${LOCAL_BUILD_SCRIPT}" ]]; then
        error "Build script not found: ${LOCAL_BUILD_SCRIPT}"
        return 1
    fi

    info "Uploading build_vllm_gh200.sh to ${REMOTE_HOST}:${DEPLOY_DIR}/"

    if ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${DEPLOY_DIR}"; then
        error "Failed to create deploy dir ${DEPLOY_DIR}"
        return 1
    fi

    if ! scp_run "${LOCAL_BUILD_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/build_vllm_gh200.sh"; then
        error "Failed to upload build script"
        return 1
    fi

    # Deploy the PR-graft snapshots (patches/) alongside the build script so the
    # build applies the committed diff reproducibly instead of fetching the live
    # PR from GitHub (#2). Optional: if the dir is absent (or upload fails) the
    # build simply falls back to the live fetch.
    if [[ -d "${LOCAL_PATCHES_DIR}" ]]; then
        info "Uploading patches/ (PR-graft snapshots)"
        if ! scp_run -r "${LOCAL_PATCHES_DIR}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/"; then
            warn "Failed to upload patches/ — build will fetch PR diffs live"
        fi
    fi

    success "Build script deployed"
    return 0
}

list_containers() {
    info "Available containers in ${REMOTE_CONTAINER_DIR}:"
    echo ""
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "ls -1d ${REMOTE_CONTAINER_DIR}/vllm-*-sandbox 2>/dev/null | xargs -I{} basename {}" 2>/dev/null || true
    echo ""
}

cmd_build_logs() {
    ensure_master_connection || exit 1

    # Find the build job in ANY state (PENDING/CONFIGURING/RUNNING), not just
    # RUNNING. A queued build has no log file yet but is still the job the user
    # means — find_job_id is RUNNING-only by design (the server commands rely on
    # that), so build logs queries by name directly instead.
    local job_info
    job_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "squeue -u \$USER -h -o '%i %T %R' --name=build-vllm-gh200 2>/dev/null | head -n1") || true

    if [[ -z "$job_info" ]]; then
        error "No build job in the queue"
        local latest_log
        latest_log=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
            "ls -t ${REMOTE_CONTAINER_DIR}/build_vllm_*.log 2>/dev/null | head -n1") || true
        if [[ -n "$latest_log" ]]; then
            echo "    Most recent build log: ${latest_log}" >&2
            echo "    View with: ssh ${REMOTE_HOST} 'tail ${latest_log}'" >&2
        fi
        exit 1
    fi

    local job_id job_state job_reason
    job_id=$(echo "$job_info" | awk '{print $1}')
    job_state=$(echo "$job_info" | awk '{print $2}')
    # Reason is the remaining field(s), e.g. "(Resources)" / "(Priority)".
    job_reason=$(echo "$job_info" | awk '{$1=""; $2=""; sub(/^ +/, ""); print}')
    local log_file="${REMOTE_CONTAINER_DIR}/build_vllm_${job_id}.log"

    # If the job hasn't started, say so and wait for it to start (like the
    # server-watch PENDING phase). Ctrl+C stops watching; the build stays queued.
    if [[ "$job_state" != "RUNNING" ]]; then
        warn "Build job ${job_id} is ${job_state}${job_reason:+ ${job_reason}} — not started yet"
        info "Waiting for it to start (Ctrl+C to stop; the build stays queued)..."
        echo "" >&2
        while ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -f '${log_file}'" 2>/dev/null; do
            local st
            st=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -j ${job_id} -h -o '%T %R'" 2>/dev/null) || true
            if [[ -z "$st" ]]; then
                # Left the queue. Either it started and finished fast (log now
                # exists → tail it) or it was cancelled/failed before logging.
                if ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -f '${log_file}'" 2>/dev/null; then
                    break
                fi
                echo "" >&2
                error "Build job ${job_id} left the queue before producing a log (cancelled or failed early)"
                exit 1
            fi
            printf "\r    %-50s" "${st} (waiting for start...)" >&2
            sleep 10
        done
        echo "" >&2
        success "Build job ${job_id} started"
    fi

    info "Tailing logs for build job ${job_id}..."
    echo "    (Press Ctrl+C to stop watching - build will continue running)" >&2
    echo "" >&2
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -F '${log_file}' 2>/dev/null" || true
}

cmd_build_cancel() {
    ensure_master_connection || exit 1

    info "Looking for build-vllm-gh200 jobs..."

    # List all build jobs regardless of state (PENDING, RUNNING, CONFIGURING, ...).
    # cancel_job's find_job_id helper matches on RUNNING only and returns just
    # the first match, which is the wrong behavior here: a queued build is still
    # a build you probably want to kill, and a user who fat-fingered the CLI may
    # have stacked several.
    local squeue_output
    if ! squeue_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "squeue -u \$USER -h -o '%i %j %T %R' --name=build-vllm-gh200" 2>/dev/null); then
        error "Failed to query SLURM queue"
        return 1
    fi

    if [[ -z "$squeue_output" ]]; then
        warn "No build-vllm-gh200 jobs found"
        return 0
    fi

    local job_count
    job_count=$(echo "$squeue_output" | wc -l | tr -d ' ')

    if (( job_count > 1 )); then
        warn "Found ${job_count} build jobs:"
        echo "$squeue_output" | while read -r jid jname jstate jreason; do
            printf "    %s  %-20s  %-12s  %s\n" "$jid" "$jname" "$jstate" "$jreason" >&2
        done
        echo "" >&2
        local reply
        read -r -p "Cancel all ${job_count} jobs? [y/N] " reply
        if [[ ! "$reply" =~ ^[Yy]$ ]]; then
            info "Aborted"
            return 0
        fi
    fi

    local job_ids
    job_ids=$(echo "$squeue_output" | awk '{print $1}' | tr '\n' ' ')

    info "Canceling job(s): ${job_ids}"
    if ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "scancel ${job_ids}"; then
        error "Failed to cancel job(s): ${job_ids}"
        return 1
    fi

    success "Canceled ${job_count} build job(s)"
}

cmd_build() {
    # Subcommand: logs — tail the currently running build job's log
    if [[ "${1:-}" == "logs" ]]; then
        shift
        cmd_build_logs "$@"
        return $?
    fi

    # Subcommand: cancel — scancel running/pending build-vllm-gh200 jobs
    if [[ "${1:-}" == "cancel" ]]; then
        shift
        cmd_build_cancel "$@"
        return $?
    fi

    local model_id=""
    local build_index=""
    local do_deploy=true
    local tail_logs=true
    local vllm_version=""
    local create_sif=""
    local force_overwrite=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --list|-l)
                ensure_master_connection || exit 1
                list_containers
                info "To build a new container:"
                echo "    ./olivia.sh build <preset>" >&2
                echo "" >&2
                echo "Available presets: glm51_v19 (alias: glm51), glm51_v20, glm52, glm47, kimi, kimi27, laguna, devstral, llama, qwen, generic" >&2
                exit 0
                ;;
            --presets|-p)
                header "Available Model Presets"
                echo "  glm51_v19  GLM-5.1 (744B/40B) on vLLM v0.19.0 — recipe default"
                echo "             vLLM: v0.19.0, transformers>=5.4.0, container index 1"
                echo "             AWQ ~430GB, 8 GPUs (2×4-GPU nodes, TP=4 + PP=2)"
                echo "             Multi-node PP wedges on concurrent decode — use with"
                echo "             anthropic_proxy.py request serialization as workaround."
                echo "             Alias: glm51"
                echo ""
                echo "  glm51_v20  GLM-5.1 on vLLM v0.20.0 + RayExecutorV2 (QUARANTINED)"
                echo "             vLLM: v0.20.0, transformers>=5.4.0, container index 2"
                echo "             Same wedge as glm51_v19; kept for diagnostics only."
                echo ""
                echo "  glm52      GLM-5.2 (744B/40B) FP8 — bleeding edge"
                echo "             vLLM: main + PR#45895 (skip-topk indexer); NO release yet"
                echo "             FP8 ~755GB, 12 GPUs (3×4-GPU nodes, TP=4 + PP=3)"
                echo "             RedHatAI/GLM-5.2-FP8 (== zai-org). Same PP wedge as glm51."
                echo ""
                echo "  glm47      GLM-4.7 (358B) flagship model"
                echo "             vLLM: main, transformers>=5.0.0rc0"
                echo ""
                echo "  kimi       Kimi K2.6 (1T MoE, 32B active) MLA + multimodal"
                echo "             vLLM: v0.19.1, transformers>=4.57.1,<5.0.0"
                echo "             native int4 ~640GB, 8 GPUs (2×4-GPU nodes, TP=4 + PP=2)"
                echo ""
                echo "  kimi27     Kimi K2.7-Code (1T MoE, 32B active) coding-focused"
                echo "             Same arch as K2.6 — reuses kimi container (index 4,"
                echo "             eager vLLM 0.21), no rebuild. native int4 ~560GB,"
                echo "             8 GPUs (2×4-GPU nodes, TP=4 + PP=2). Thinking-mode only."
                echo "             Weights on /cluster/work scratch — serve with"
                echo "             HF_HOME=/cluster/work/projects/nn10104k/huggingface"
                echo ""
                echo "  laguna     Laguna M.1 (Poolside, 225B/23B active) MoE coding model"
                echo "             vLLM: v0.21.0 (native Laguna), transformers>=5.7.0"
                echo "             FP8 ~225GB, single node (4×GH200, TP=4); poolside_v1 parsers."
                echo "             Weights too big for project quota — prefetch+serve with"
                echo "             HF_HOME=/cluster/work/projects/nn10104k/huggingface"
                echo ""
                echo "  devstral   Devstral/Mistral models"
                echo "             vLLM: main, transformers>=4.45.0"
                echo ""
                echo "  llama      Llama 3.x models"
                echo "             vLLM: main, transformers>=4.45.0"
                echo ""
                echo "  qwen       Qwen 2.5 models"
                echo "             vLLM: main, transformers>=4.45.0"
                echo ""
                echo "  generic    Generic build (default)"
                echo "             vLLM: main, transformers>=4.45.0"
                echo ""
                info "Usage: ./olivia.sh build <preset> [options]"
                exit 0
                ;;
            --no-deploy)
                do_deploy=false
                shift
                ;;
            --no-tail)
                tail_logs=false
                shift
                ;;
            --index)
                build_index="$2"
                shift 2
                ;;
            --vllm)
                vllm_version="$2"
                shift 2
                ;;
            --sif)
                create_sif="1"
                shift
                ;;
            --force|-f)
                force_overwrite="1"
                shift
                ;;
            -h|--help)
                cmd_build_help
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                cmd_build_help
                exit 1
                ;;
            *)
                if [[ -z "$model_id" ]]; then
                    # Guard against subcommand-shaped tokens being silently
                    # accepted as MODEL_IDs. `build_vllm_gh200.sh` falls back
                    # to generic defaults for unknown MODEL_IDs, which means a
                    # typo like `./olivia.sh build cancel` would otherwise
                    # submit a real SLURM job building `vllm-cancel-1-sandbox`.
                    case "$1" in
                        cancel|logs|help|status|list|presets|stop)
                            error "'$1' looks like a subcommand, not a preset name"
                            echo "    Subcommands must come first: ./olivia.sh build $1" >&2
                            echo "    Run './olivia.sh build --presets' to see valid presets" >&2
                            exit 1
                            ;;
                    esac
                    model_id="$1"
                else
                    error "Unexpected argument: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Show help if no model specified
    if [[ -z "$model_id" ]]; then
        header "Build Module"
        echo "Build vLLM containers on Olivia HPC cluster."
        echo ""
        info "Usage: ./olivia.sh build <preset> [options]"
        echo ""
        echo "Quick commands:"
        echo "    ./olivia.sh build --presets     Show available presets"
        echo "    ./olivia.sh build --list        List existing containers"
        echo "    ./olivia.sh build logs          Tail logs of a running build"
        echo "    ./olivia.sh build cancel        Cancel running/pending build job(s)"
        echo ""
        echo "Build examples:"
        echo "    ./olivia.sh build glm47            Build GLM-4.7 container"
        echo "    ./olivia.sh build glm47 --index 2  Build second container (safe)"
        echo "    ./olivia.sh build glm47 --force    Rebuild existing (caution!)"
        echo ""
        echo "Options:"
        echo "    --index N      Build index (default: 1)"
        echo "    --force, -f    Overwrite existing container"
        echo "    --vllm VER     Override vLLM version"
        echo "    --sif          Create SIF image after build"
        echo "    --no-deploy    Don't upload build script (use existing)"
        echo "    --no-tail      Don't tail logs after submitting"
        exit 0
    fi

    ensure_master_connection || exit 1

    # Deploy build script
    if $do_deploy; then
        deploy_build_script || exit 1
        echo ""
    fi

    # Apply preset-specific default build index (glm51_v20 → 2). If the user
    # already passed --index, that wins.
    if [[ -z "$build_index" ]]; then
        build_index=$(get_preset_default_index "$model_id")
    fi

    # Build environment variables.
    # CONTAINER_DIR must be exported to the SLURM job — `cd` alone doesn't
    # propagate it as an env var (build_vllm_gh200.sh requires it explicitly).
    # CONTAINER_DIR stays the shared sandbox dir; PATCHES_DIR points at this
    # branch's deploy dir so concurrent builds use their own PR snapshots.
    local env_vars="CONTAINER_DIR=${REMOTE_CONTAINER_DIR} PATCHES_DIR=${DEPLOY_DIR}/patches MODEL_ID=${model_id}"
    [[ -n "$build_index" ]] && env_vars="${env_vars} BUILD_INDEX=${build_index}"
    [[ -n "$vllm_version" ]] && env_vars="${env_vars} VLLM_VERSION=${vllm_version}"
    [[ -n "$create_sif" ]] && env_vars="${env_vars} CREATE_SIF=${create_sif}"
    [[ -n "$force_overwrite" ]] && env_vars="${env_vars} OVERWRITE=${force_overwrite}"

    info "Submitting build job for '${model_id}'..."
    echo "    Environment: ${env_vars}" >&2

    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_CONTAINER_DIR} && ${env_vars} sbatch ${DEPLOY_DIR}/build_vllm_gh200.sh" 2>&1); then
        error "Failed to submit build job"
        echo "$submit_output" >&2
        return 1
    fi

    local job_id
    job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)
    success "Submitted build job ${job_id}"

    # Determine expected sandbox name. Version-tagged presets (glm51_v19,
    # glm51_v20) share a container prefix with their base preset, so use
    # resolve_container_prefix to get the on-disk directory name.
    local idx="${build_index:-1}"
    local prefix
    prefix=$(resolve_container_prefix "$model_id")
    local sandbox_name="vllm-${prefix}-${idx}-sandbox"
    echo "    Container: ${sandbox_name}" >&2

    if $tail_logs; then
        local log_file="${REMOTE_CONTAINER_DIR}/build_vllm_${job_id}.log"
        tail_job_logs "$log_file" "$job_id"
        echo ""
        info "Build complete. Check with: ./olivia.sh build --list"
    else
        echo ""
        info "Build running in background."
        echo "    Check status: squeue -j ${job_id}" >&2
        echo "    View logs:    ssh ${REMOTE_HOST} 'tail -f ${REMOTE_CONTAINER_DIR}/build_vllm_${job_id}.log'" >&2
    fi
}

cmd_build_help() {
    cat <<EOF
Usage: ./olivia.sh build [preset|logs|cancel] [options]

Build vLLM containers on Olivia HPC cluster.

Arguments:
    preset              Model preset (glm51, glm47, devstral, llama, qwen, generic)
    logs                Tail logs of the currently running build job
    cancel              Cancel running/pending build-vllm-gh200 job(s)

Options:
    --list, -l          List existing containers
    --presets, -p       Show available presets
    --index N           Build index for multiple builds (default: 1)
    --vllm VERSION      Override vLLM version (default: main)
    --sif               Create SIF image after build
    --force, -f         Overwrite existing container (use with caution!)
    --no-deploy         Use existing build script on cluster
    --no-tail           Don't tail logs after submitting
    -h, --help          Show this help

Safety:
    By default, the build will FAIL if a container with the same name already
    exists. This protects working containers from accidental overwrites.

    To build a new version without affecting existing containers:
        ./olivia.sh build glm47 --index 2

    To rebuild/overwrite an existing container:
        ./olivia.sh build glm47 --force

Examples:
    ./olivia.sh build glm47             Build GLM-4.7 container
    ./olivia.sh build glm47 --index 2   Build second GLM-4.7 container
    ./olivia.sh build glm47 --force     Rebuild existing container
    ./olivia.sh build devstral --sif    Build Devstral and create SIF
    ./olivia.sh build --list            List existing containers
    ./olivia.sh build logs              Tail running build job logs
    ./olivia.sh build cancel            Cancel running/pending build job(s)
EOF
}

# =============================================================================
# MODULE: Server
# =============================================================================

# Canonicalize a preset name: lowercase + map aliases to a canonical token.
# This is the ONLY place that knows about aliases, so every other preset lookup
# can match canonical tokens only. Unknown names pass through (lowercased) and
# are treated as a custom MODEL_ID downstream.
# (lowercase via `tr` rather than ${1,,} — macOS ships bash 3.2.)
normalize_preset() {
    local p
    p=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$p" in
        glm51|glm-5.1|glm5.1|glm51_v19) echo "glm51_v19" ;;
        glm51_v20)                      echo "glm51_v20" ;;
        glm52|glm-5.2|glm5.2)           echo "glm52" ;;
        glm47|glm-4.7)                  echo "glm47" ;;
        kimi|kimi26|kimi-k2.6|kimi_k26) echo "kimi" ;;
        kimi27|kimi-k2.7|kimi_k27|kimi-k2.7-code|kimi27code) echo "kimi27" ;;
        laguna|laguna-m1|laguna_m1|lagunam1|laguna-m.1|laguna_m.1) echo "laguna" ;;
        devstral|mistral)               echo "devstral" ;;
        llama|llama3)                   echo "llama" ;;
        qwen|qwen2)                     echo "qwen" ;;
        generic)                        echo "generic" ;;
        *)                              echo "$p" ;;
    esac
}

# Single source of truth for per-preset runtime config. Given a (raw) preset
# name and a field, emit that field's value. All per-preset data lives in the
# one case statement below — add or change a preset here and every accessor
# (get_default_model / get_preset_resources / resolve_container_* /
# get_preset_default_index) picks it up.
#
# Fields: model | nodes | gpus | pp | resources | prefix | index
preset_field() {
    local canon field
    canon=$(normalize_preset "$1")
    field="$2"
    # Defaults: single-node 4-GPU (matches run_vllm_server.sh #SBATCH directives),
    # container prefix == canonical name, index 1, no default model.
    local prefix="$canon" index="1" model="" nodes="1" gpus="4" pp="1"
    case "$canon" in
        glm51_v19)
            # glm51_v19/glm51_v20 share the "glm51" container prefix; the index
            # encodes the vLLM version (1 = v0.19.0, 2 = v0.20.0).
            prefix="glm51"; index="1"
            model="cyankiwi/GLM-5.1-AWQ-4bit"
            # 744B MoE AWQ (~430GB) needs 8 GPUs. TP=4 intra-node (NVLink) +
            # PP=2 cross-node (Slingshot) — PP tolerates the ~25 GB/s cross-node
            # fabric far better than naive TP=8 would.
            nodes="2"; gpus="4"; pp="2"
            ;;
        glm51_v20)
            prefix="glm51"; index="2"
            model="cyankiwi/GLM-5.1-AWQ-4bit"
            nodes="2"; gpus="4"; pp="2"
            ;;
        glm52)
            # GLM-5.2 (744B MoE+DSA, successor to 5.1). No AWQ-4bit quant exists
            # yet, so the runnable quant today is block-FP8 (~755 GB). At
            # ~94 GB/GPU it does NOT fit 8×GH200, so glm52 spans 3 nodes:
            # TP=4 intra-node (NVLink) + PP=3 cross-node (Slingshot) = 12 GPUs.
            # Default model is the RedHatAI re-host (byte-identical to the
            # official zai-org/GLM-5.2-FP8). Swap to an AWQ-4bit repo + 2-node
            # PP=2 once one is published (mirrors glm51).
            model="RedHatAI/GLM-5.2-FP8"
            nodes="3"; gpus="4"; pp="3"
            ;;
        glm47)
            model="QuantTrio/GLM-4.7-AWQ"
            ;;
        kimi)
            # Kimi K2.6 (1T MoE, 32B active). The base repo ships Moonshot's
            # native int4 (compressed-tensors, ~640GB) — there is no separate
            # -AWQ repo. ~640GB → 8 GPUs / 2 nodes, same TP=4 + PP=2 as glm51.
            # index 4 = the validated eager vLLM 0.21 container
            # (vllm-kimi-4-sandbox, shared with kimi27) — carries the
            # reasoning_tokens patches and is what the K2.6 deployment runs on.
            # (Was the default index 1 = legacy vLLM 0.19 vllm-kimi-1-sandbox.)
            prefix="kimi"; index="4"
            model="moonshotai/Kimi-K2.6"
            nodes="2"; gpus="4"; pp="2"
            ;;
        kimi27)
            # Kimi K2.7-Code (1T MoE, 32B active): coding-focused successor to
            # K2.6. SAME KimiK25ForConditionalGeneration arch + native int4
            # (compressed-tensors ~560GB), so it reuses the "kimi" container
            # (index 4 = the validated eager vLLM 0.21 build) with no rebuild.
            # 8 GPUs / 2 nodes, TP=4 + PP=2. Thinking-mode only (temp 1.0,
            # top_p 0.95). Weights live on /cluster/work scratch (the project
            # quota is full with K2.6) -> serve with HF_HOME pointed at the
            # scratch cache: HF_HOME=/cluster/work/projects/nn10104k/huggingface
            prefix="kimi"; index="4"
            model="moonshotai/Kimi-K2.7-Code"
            nodes="2"; gpus="4"; pp="2"
            ;;
        laguna)
            # Laguna M.1 (Poolside): 225B total / 23B active MoE coding model
            # (LagunaForCausalLM, 256 experts top-k=16, dense GQA full attention,
            # 256K context). Default quant is block-FP8 (poolside/Laguna-M.1-FP8,
            # ~225 GB) which fits a SINGLE GH200 node at TP=4 — no cross-node PP,
            # unlike glm52's 3-node FP8. Single-node defaults (nodes=1/gpus=4/pp=1)
            # apply. Native vLLM support (>=0.21.0); ordinary attention so it keeps
            # FLASH_ATTN and lets CUDAGraph capture run (no eager override).
            model="poolside/Laguna-M.1-FP8"
            ;;
        devstral)
            model="mistralai/Devstral-2-123B-Instruct-2512"
            ;;
        llama)
            model="meta-llama/Llama-3.3-70B-Instruct"
            ;;
        qwen)
            model="Qwen/Qwen2.5-72B-Instruct"
            ;;
        generic)
            ;;
        *)
            # Custom/unknown preset: keep the raw name for the container prefix
            # so it matches build_vllm_gh200.sh's MODEL_ID="${preset}" (which
            # preserves case); no default model, single-node defaults.
            prefix="$1"
            ;;
    esac
    case "$field" in
        model)     echo "$model" ;;
        nodes)     echo "$nodes" ;;
        gpus)      echo "$gpus" ;;
        pp)        echo "$pp" ;;
        resources) echo "$nodes $gpus $pp" ;;
        prefix)    echo "$prefix" ;;
        index)     echo "$index" ;;
        *)         echo "" ;;
    esac
}

# Thin accessors over preset_field — kept as named functions so call sites read
# clearly and don't all have to learn the field names.
get_default_model()        { preset_field "$1" model; }
get_preset_resources()     { preset_field "$1" resources; }  # "num_nodes gpus_per_node pp_size"
resolve_container_prefix() { preset_field "$1" prefix; }
get_preset_default_index() { preset_field "$1" index; }

# Resolve preset + index to a container sandbox name.
resolve_container_name() {
    local prefix
    prefix=$(preset_field "$1" prefix)
    echo "vllm-${prefix}-${2:-1}-sandbox"
}

deploy_server_script() {
    if [[ ! -f "${LOCAL_SERVER_SCRIPT}" ]]; then
        error "Server script not found: ${LOCAL_SERVER_SCRIPT}"
        return 1
    fi

    info "Uploading run_vllm_server.sh to ${REMOTE_HOST}:${DEPLOY_DIR}/"

    if ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${DEPLOY_DIR}"; then
        error "Failed to create deploy dir ${DEPLOY_DIR}"
        return 1
    fi

    if ! scp_run "${LOCAL_SERVER_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/run_vllm_server.sh"; then
        error "Failed to upload server script"
        return 1
    fi

    # Also upload auxiliary files that run_vllm_server.sh expects at runtime.
    # Currently just the GLM-5.1 chat template override; grows as needed.
    if [[ -f "${LOCAL_GLM51_CHAT_TEMPLATE}" ]]; then
        info "Uploading glm51_chat_template.jinja"
        if ! scp_run "${LOCAL_GLM51_CHAT_TEMPLATE}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/glm51_chat_template.jinja"; then
            error "Failed to upload GLM-5.1 chat template"
            return 1
        fi
    fi

    success "Server script deployed"
    return 0
}

list_server_containers() {
    info "Available containers in ${REMOTE_CONTAINER_DIR}:"
    echo ""
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "ls -1d ${REMOTE_CONTAINER_DIR}/vllm-*-sandbox ${REMOTE_CONTAINER_DIR}/vllm-*.sif 2>/dev/null | xargs -I{} basename {}" 2>/dev/null || true
    echo ""
}

start_server_job() {
    local container="$1"
    local model="$2"
    local num_nodes="${3:-1}"
    local gpus_per_node="${4:-4}"
    local pp_size="${5:-1}"

    info "Starting vLLM server..."
    echo "    Container: ${container}" >&2
    echo "    Model:     ${model}" >&2
    if [[ "${num_nodes}" -gt 1 ]]; then
        echo "    Nodes:     ${num_nodes}" >&2
        echo "    GPUs/node: ${gpus_per_node}" >&2
        echo "    Total GPUs: $((num_nodes * gpus_per_node))" >&2
        echo "    TP=${gpus_per_node} + PP=${pp_size} (multi-node over Slingshot)" >&2
    fi

    # Environment variables passed into the SLURM job.
    # CONTAINER_DIR must be exported — run_vllm_server.sh requires it explicitly;
    # `cd` alone doesn't propagate it as an env var into sbatch's job environment.
    local env_vars="CONTAINER_DIR=${REMOTE_CONTAINER_DIR}"
    env_vars+=" CONTAINER=${container} MODEL=${model}"
    env_vars+=" NUM_NODES=${num_nodes}"
    env_vars+=" TP_SIZE=${gpus_per_node}"
    env_vars+=" PP_SIZE=${pp_size}"

    # HF_HOME (persistent model cache) is forwarded from the local environment
    # (mise.local.toml) rather than read from the cluster shell, so there is a
    # single source of truth and nothing stale lives in the cluster ~/.bashrc.
    # It is a path, not a secret, so it rides in env_vars like the rest.
    # run_vllm_server.sh requires it.
    if [[ -z "${HF_HOME:-}" ]]; then
        error "HF_HOME is not set — cannot start server"
        echo "    HF_HOME should come from mise.local.toml (the persistent model cache)," >&2
        echo "    e.g. export HF_HOME=/cluster/projects/<proj>/huggingface" >&2
        return 1
    fi
    env_vars+=" HF_HOME=${HF_HOME}"

    # Point the GLM-5.1 chat-template override at this branch's deploy dir (the
    # default inside run_vllm_server.sh is ${CONTAINER_DIR}/glm51_chat_template.jinja,
    # which is the shared, clobber-prone path). Harmless for non-GLM-5.1 presets.
    env_vars+=" CHAT_TEMPLATE_FILE=${DEPLOY_DIR}/glm51_chat_template.jinja"

    # Forward selected debug/tuning env vars if the caller set them. Useful for
    # diagnosing startup hangs: `VERBOSE=1 ./olivia.sh server start glm51`
    # flips vLLM, Ray, and NCCL to verbose logging inside the SLURM job.
    # Tool/reasoning knobs are forwarded so Claude Code and other OpenAI
    # tool-using clients can enable the GLM tool-call parser per-session
    # without editing run_vllm_server.sh.
    # Cache-dir overrides (VLLM_CACHE*, *_CACHE_DIR) let several servers run
    # concurrently from the same workdir without racing on the shared
    # $PWD/cache/* compile/JIT caches — give each parallel experiment its own
    # cache root. DISABLE_CUSTOM_ALL_REDUCE is forwarded so CUDAGraph capture
    # experiments can pin the NCCL all-reduce fallback explicitly.
    for forward_var in VERBOSE VLLM_LOGGING_LEVEL RAY_BACKEND_LOG_LEVEL \
                       RAY_DEDUP_LOGS NCCL_DEBUG CUDAGRAPH_MODE \
                       ENABLE_AUTO_TOOL_CHOICE GLM_TOOL_PARSER \
                       GLM_REASONING_PARSER SERVED_MODEL_NAME \
                       MTP_SPECULATIVE_TOKENS ENABLE_SPECULATIVE ALLOW_MTP_PP \
                       MAX_MODEL_LEN GPU_MEM_UTIL RAY_CGRAPH_GET_TIMEOUT \
                       VLLM_USE_RAY_V2_EXECUTOR_BACKEND DISABLE_CUSTOM_ALL_REDUCE \
                       VLLM_CACHE VLLM_CACHE_ROOT TRITON_CACHE_DIR \
                       DG_JIT_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
                       VLLM_PP_LAYER_PARTITION EXTRA_VLLM_ARGS; do
        if [[ -n "${!forward_var:-}" ]]; then
            env_vars+=" ${forward_var}=${!forward_var}"
        fi
    done

    # sbatch directive overrides (override the #SBATCH lines in run_vllm_server.sh).
    # For multi-node jobs we MUST pass --ntasks=N explicitly. Without it, SLURM's
    # task-count resolution collapses the allocation to 1 node even when --nodes=N
    # is specified, because a single task needs only a single node.
    local sbatch_opts=""
    if [[ "${num_nodes}" -gt 1 ]]; then
        sbatch_opts="--nodes=${num_nodes} --ntasks=${num_nodes} --ntasks-per-node=1 --gpus-per-node=${gpus_per_node} --cpus-per-task=$((gpus_per_node * 8))"
    else
        # Single-node: pin the job to exactly ONE node with N GPUs. Passing
        # --gpus=N (a TOTAL count) alone lets SLURM scatter those N GPUs across
        # >1 node on a fragmented cluster — observed landing a laguna start on
        # 2 nodes — which breaks single-node TP=N (vLLM needs all N GPUs
        # co-located on one node). --nodes=1 --gpus-per-node=N makes it
        # unambiguous: the job waits for a full free node instead of accepting a
        # split allocation. --ntasks=1 keeps it a single task (run_vllm_server.sh
        # launches vllm directly, no srun/Ray for single-node).
        sbatch_opts="--nodes=1 --ntasks=1 --gpus-per-node=${gpus_per_node} --cpus-per-task=$((gpus_per_node * 8))"
    fi

    # Optional walltime override (TIME_LIMIT, e.g. "3:00:00" or "180"). Overrides
    # the #SBATCH --time in run_vllm_server.sh. Useful to cap a large multi-node
    # allocation so it doesn't sit idle for the full default if left unattended.
    # (A running job's limit can also be shortened live: scontrol update jobid=N
    # TimeLimit=HH:MM:SS.)
    if [[ -n "${TIME_LIMIT:-}" ]]; then
        sbatch_opts+=" --time=${TIME_LIMIT}"
    fi

    # Debug: echo the exact command being submitted so regressions like
    # "didn't request 2 nodes" are visible at submission time.
    echo "    sbatch opts: ${sbatch_opts}" >&2
    echo "    env vars:    ${env_vars}" >&2

    # HF_TOKEN is a secret, so it is NOT placed in env_vars (which is echoed
    # above) or on any command line (which `ps` could expose on the shared login
    # node). Instead it is piped over stdin; the remote shell reads it into its
    # environment and sbatch's default --export=ALL carries it into the job.
    local submit_output remote_submit
    remote_submit="cd ${REMOTE_CONTAINER_DIR} && ${env_vars} sbatch ${sbatch_opts} ${DEPLOY_DIR}/run_vllm_server.sh"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
            "IFS= read -r HF_TOKEN && export HF_TOKEN && ${remote_submit}" <<<"${HF_TOKEN}" 2>&1) \
            || { error "Failed to submit job"; echo "$submit_output" >&2; return 1; }
    else
        submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "${remote_submit}" 2>&1) \
            || { error "Failed to submit job"; echo "$submit_output" >&2; return 1; }
    fi

    local job_id
    job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)
    success "Submitted job ${job_id}"
    echo "$job_id"
}

check_container_exists() {
    local container="$1"
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "test -d '${REMOTE_CONTAINER_DIR}/${container}' || test -f '${REMOTE_CONTAINER_DIR}/${container}'" 2>/dev/null
}

# =============================================================================
# Server Watch - Intelligent model loading monitor
# =============================================================================

server_watch() {
    local base_interval="${1:-3}"  # Base update interval in seconds
    local max_interval=60          # Maximum interval (1 minute)
    local backoff_increment=3      # Seconds to add per iteration during back-off
    local current_interval="$base_interval"

    echo ""
    info "Starting intelligent server watch..."
    echo "    Press Ctrl+C to exit (monitoring continues indefinitely)"
    echo ""

    local phase="waiting"  # waiting, loading, initializing, serving
    local prev_mem=0
    local stable_count=0
    local health_checks=0
    local max_health_checks=60  # Give up after ~3 minutes of health checks
    local node_name=""         # Head node (for health check / SSH)
    local node_list=""         # Space-separated list of all allocated nodes (multi-node)
    local num_nodes_in_job=1
    local job_id=""
    local log_file=""
    local prev_state=""        # Track previous job state for notifications
    local serving_iterations=0 # Count iterations in serving phase for back-off

    # Cleanup on exit
    trap 'echo ""; info "Watch stopped"; return 0' INT TERM

    while true; do
        # Phase 1: Wait for job to start
        if [[ "$phase" == "waiting" ]]; then
            local job_info
            job_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -u \$USER -h -o '%j %i %N %T' | grep -i vllm" 2>/dev/null) || true

            if [[ -z "$job_info" ]]; then
                printf "\r\033[K[WAITING] No vLLM job found. Waiting for job submission..."
                prev_state=""
            else
                local state
                local raw_nodelist
                state=$(echo "$job_info" | awk '{print $4}' | head -1)
                job_id=$(echo "$job_info" | awk '{print $2}' | head -1)
                raw_nodelist=$(echo "$job_info" | awk '{print $3}' | head -1)

                # Expand SLURM compressed nodelist (e.g., "c1-[1-2]" → "c1-1 c1-2").
                # For single-node jobs this is a no-op pass-through.
                if [[ "$raw_nodelist" == *"["* || "$raw_nodelist" == *","* ]]; then
                    node_list=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                        "scontrol show hostnames '${raw_nodelist}' 2>/dev/null | paste -sd ' '" 2>/dev/null) || node_list="$raw_nodelist"
                else
                    node_list="$raw_nodelist"
                fi
                node_name=$(echo "$node_list" | awk '{print $1}')
                num_nodes_in_job=$(echo "$node_list" | wc -w | tr -d ' ')

                # Notify on state transitions
                if [[ "$state" == "RUNNING" && "$prev_state" == "PENDING" ]]; then
                    notify "Olivia vLLM" "Job ${job_id} is now RUNNING on ${node_name}" "Glass"
                fi

                if [[ "$state" == "RUNNING" && -n "$node_name" ]]; then
                    echo ""
                    if [[ "$num_nodes_in_job" -gt 1 ]]; then
                        success "Job ${job_id} running on ${num_nodes_in_job} nodes: ${node_list}"
                    else
                        success "Job ${job_id} running on ${node_name}"
                    fi
                    # Start with the wrapper log; promote to _head.log once it
                    # has content (multi-node jobs only — single-node leaves
                    # log_file on the wrapper for the entire run).
                    log_file="$(server_log_wrapper "$job_id")"
                    phase="loading"
                    current_interval="$base_interval"  # Reset interval
                    prev_state="$state"
                    sleep 2  # Give container time to start
                    continue
                elif [[ "$state" == "PENDING" ]]; then
                    printf "\r\033[K[PENDING] Job ${job_id} waiting for resources..."
                else
                    printf "\r\033[K[${state}] Job ${job_id}..."
                fi
                prev_state="$state"
            fi
            sleep "$current_interval"
            continue
        fi

        # Phase 2: Monitor GPU memory loading
        if [[ "$phase" == "loading" || "$phase" == "initializing" ]]; then
            # Promote log_file to the head log once vLLM starts writing there.
            # On multi-node jobs the wrapper log stops at the Ray-bootstrap
            # handoff; all post-bootstrap output (loading progress, KV cache,
            # throughput) lives in _head.log. Once we've promoted, stop
            # checking.
            if [[ "$log_file" != *"_head.log" ]]; then
                local _head_candidate
                _head_candidate="$(server_log_head "$job_id")"
                if ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                    "test -s '${_head_candidate}'" 2>/dev/null; then
                    log_file="$_head_candidate"
                fi
            fi

            # Check if job is still running
            local job_check
            job_check=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -j ${job_id} -h -o '%T' 2>/dev/null" 2>/dev/null) || true

            if [[ -z "$job_check" || "$job_check" != "RUNNING" ]]; then
                echo ""
                warn "Job ${job_id} failed during ${phase} phase (${job_check:-terminated})"
                notify "Olivia vLLM" "Server job ${job_id} failed during startup" "Basso"
                # Reset to waiting phase
                phase="waiting"
                prev_state=""
                job_id=""
                node_name=""
                current_interval="$base_interval"
                sleep "$current_interval"
                continue
            fi

            # Get GPU memory usage across all allocated nodes (single SSH, loop on remote)
            local gpu_info
            gpu_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "for n in ${node_list}; do ssh \$n 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null; done" 2>/dev/null) || true

            if [[ -n "$gpu_info" ]]; then
                # Parse GPU info (format: "used, total, util" per GPU)
                local total_used=0
                local total_mem=0
                local gpu_count=0
                local max_util=0

                while IFS=',' read -r used total util; do
                    used=$(echo "$used" | tr -d ' ')
                    total=$(echo "$total" | tr -d ' ')
                    util=$(echo "$util" | tr -d ' ')
                    total_used=$((total_used + used))
                    total_mem=$((total_mem + total))
                    gpu_count=$((gpu_count + 1))
                    if [[ "$util" -gt "$max_util" ]]; then
                        max_util="$util"
                    fi
                done <<< "$gpu_info"

                # Convert to GB
                local used_gb=$((total_used / 1024))
                local total_gb=$((total_mem / 1024))
                local pct=$((total_used * 100 / total_mem))

                # Check if memory is stable (loading complete)
                if [[ "$total_used" -eq "$prev_mem" ]]; then
                    stable_count=$((stable_count + 1))
                else
                    stable_count=0
                fi
                prev_mem=$total_used

                # Progress bar
                local bar_width=30
                local filled=$((pct * bar_width / 100))
                local empty=$((bar_width - filled))
                local bar=$(printf '%*s' "$filled" '' | tr ' ' '█')
                bar+=$(printf '%*s' "$empty" '' | tr ' ' '░')

                if [[ "$phase" == "loading" ]]; then
                    printf "\r\033[K[LOADING] GPU Memory: [%s] %d/%dGB (%d%%) | GPUs: %d" \
                        "$bar" "$used_gb" "$total_gb" "$pct" "$gpu_count"

                    # If memory stable for ~15 seconds, switch to initializing phase
                    if [[ "$stable_count" -ge 5 && "$used_gb" -gt 10 ]]; then
                        echo ""
                        info "Weights loaded (~${used_gb}GB). Initializing model..."
                        phase="initializing"
                        stable_count=0
                    fi
                else
                    # Initializing phase - check health endpoint
                    printf "\r\033[K[INIT] GPU: %d/%dGB | Checking server health..." \
                        "$used_gb" "$total_gb"

                    # Try health endpoint via SSH tunnel to node
                    local health_status
                    health_status=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                        "curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://${node_name}:${REMOTE_PORT}/health" 2>/dev/null) || health_status="000"

                    health_checks=$((health_checks + 1))

                    if [[ "$health_status" == "200" ]]; then
                        echo ""
                        echo ""
                        success "Server is READY! Monitoring throughput..."
                        notify "Olivia vLLM" "Server is READY and serving requests" "Glass"
                        echo ""
                        phase="serving"
                        serving_iterations=0
                        current_interval="$base_interval"
                    elif [[ "$health_checks" -ge "$max_health_checks" ]]; then
                        echo ""
                        warn "Server not responding after ${health_checks} checks"
                        echo "    Check logs: ./olivia.sh server logs"
                        return 1
                    fi
                fi
            else
                printf "\r\033[K[LOADING] Waiting for GPU info from ${node_name}..."
            fi
        fi

        # Phase 3: Monitor throughput when serving
        if [[ "$phase" == "serving" ]]; then
            # Check if job is still running
            local job_check
            job_check=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -j ${job_id} -h -o '%T' 2>/dev/null" 2>/dev/null) || true

            if [[ -z "$job_check" || "$job_check" != "RUNNING" ]]; then
                echo ""
                warn "Job ${job_id} is no longer running (${job_check:-terminated})"
                notify "Olivia vLLM" "Server job ${job_id} has stopped" "Basso"
                # Reset to waiting phase
                phase="waiting"
                prev_state=""
                job_id=""
                node_name=""
                current_interval="$base_interval"
                sleep "$current_interval"
                continue
            fi

            # Get GPU memory for display
            local gpu_info
            gpu_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "ssh ${node_name} 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null" 2>/dev/null) || true

            local total_used=0
            local total_mem=0
            if [[ -n "$gpu_info" ]]; then
                while IFS=',' read -r used total; do
                    used=$(echo "$used" | tr -d ' ')
                    total=$(echo "$total" | tr -d ' ')
                    total_used=$((total_used + used))
                    total_mem=$((total_mem + total))
                done <<< "$gpu_info"
            fi
            local used_gb=$((total_used / 1024))
            local total_gb=$((total_mem / 1024))

            # Get latest throughput from logs
            # Format: "Avg prompt throughput: 1.9 tokens/s, Avg generation throughput: 7.6 tokens/s, Running: 1 reqs"
            local log_line
            log_line=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "tail -20 '${log_file}' 2>/dev/null | grep -o 'Avg generation throughput: [0-9.]*.*Running: [0-9]* reqs' | tail -1" 2>/dev/null) || true

            local gen_throughput="--"
            local running_reqs="0"
            local kv_cache="--"

            if [[ -n "$log_line" ]]; then
                gen_throughput=$(echo "$log_line" | grep -oE 'generation throughput: [0-9.]+' | grep -oE '[0-9.]+') || gen_throughput="--"
                running_reqs=$(echo "$log_line" | grep -oE 'Running: [0-9]+' | grep -oE '[0-9]+') || running_reqs="0"
            fi

            # Get KV cache usage
            local kv_line
            kv_line=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "tail -5 '${log_file}' 2>/dev/null | grep -o 'KV cache usage: [0-9.]*%' | tail -1" 2>/dev/null) || true
            if [[ -n "$kv_line" ]]; then
                kv_cache=$(echo "$kv_line" | grep -oE '[0-9.]+%') || kv_cache="--"
            fi

            # Linear back-off: increase interval up to max when idle, reset when active
            serving_iterations=$((serving_iterations + 1))
            if [[ "$running_reqs" != "0" && "$running_reqs" != "--" ]]; then
                # Activity detected - reset to base interval
                current_interval="$base_interval"
                serving_iterations=0
            else
                # No activity - apply linear back-off
                local new_interval=$((base_interval + (serving_iterations * backoff_increment)))
                if [[ "$new_interval" -gt "$max_interval" ]]; then
                    new_interval="$max_interval"
                fi
                current_interval="$new_interval"
            fi

            # Display serving status with current poll interval
            printf "\r\033[K[SERVING] GPU: %d/%dGB | Throughput: %s tok/s | Reqs: %s | KV: %s | Poll: %ds" \
                "$used_gb" "$total_gb" "$gen_throughput" "$running_reqs" "$kv_cache" "$current_interval"
        fi

        sleep "$current_interval"
    done
}

cmd_server() {
    local action=""
    local do_deploy=false
    local container=""
    local model=""
    local preset=""
    # Empty sentinel: "user did not pass --index". After arg parsing, we fall
    # back to the preset's default index via get_preset_default_index (e.g.,
    # glm51_v20 defaults to 2 to match its vllm-glm51-2-sandbox container).
    local index=""
    local tail_logs=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            start)
                action="start"
                shift
                # Check if next arg is a preset (not a flag)
                if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
                    preset="$1"
                    shift
                fi
                ;;
            restart)
                action="restart"
                shift
                # Check if next arg is a preset (not a flag)
                if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
                    preset="$1"
                    shift
                fi
                ;;
            deploy)
                action="deploy"
                shift
                ;;
            logs)
                action="logs"
                shift
                ;;
            list|ls)
                action="list"
                shift
                ;;
            status)
                action="status"
                shift
                ;;
            cancel|stop)
                action="cancel"
                shift
                ;;
            ssh|shell)
                action="ssh"
                shift
                ;;
            watch)
                action="watch"
                shift
                ;;
            --container|-c)
                container="$2"
                shift 2
                ;;
            --model|-m)
                model="$2"
                shift 2
                ;;
            --index|-i)
                index="$2"
                shift 2
                ;;
            --deploy|-d)
                do_deploy=true
                shift
                ;;
            --no-tail)
                tail_logs=false
                shift
                ;;
            -h|--help)
                cmd_server_help
                exit 0
                ;;
            *)
                # Could be a preset for start/restart command
                if [[ -z "$action" ]]; then
                    error "Unknown argument: $1"
                    cmd_server_help
                    exit 1
                elif [[ ("$action" == "start" || "$action" == "restart") && -z "$preset" && ! "$1" =~ ^- ]]; then
                    preset="$1"
                    shift
                else
                    error "Unknown argument: $1"
                    exit 1
                fi
                ;;
        esac
    done

    # Show help if no action
    if [[ -z "$action" ]]; then
        cmd_server_help
        exit 0
    fi

    # If the user didn't pass --index, pick the preset's default (1 for most,
    # 2 for glm51_v20 which shares the glm51 container prefix). Presets that
    # don't have a special default still get "1".
    if [[ -z "$index" ]]; then
        index=$(get_preset_default_index "$preset")
    fi

    ensure_master_connection || exit 1

    case "$action" in
        list)
            list_server_containers
            ;;
        status)
            info "vLLM server status:"
            local job_info
            job_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -u \$USER -h -o '%j %i %N %T %M' | grep -i vllm" 2>/dev/null) || true
            if [[ -n "$job_info" ]]; then
                echo ""
                echo "$job_info" | while read -r name id node state time; do
                    if [[ "$state" == "RUNNING" ]]; then
                        success "Server is RUNNING"
                        echo "    Job ID:   ${id}"
                        echo "    Node:     ${node}"
                        echo "    Uptime:   ${time}"
                        echo ""
                        echo "    Connect:  ./olivia.sh chat"
                        echo "    Logs:     ./olivia.sh server logs"
                    else
                        warn "Server is ${state}"
                        echo "    Job ID:   ${id}"
                        echo "    Node:     ${node:-pending}"
                    fi
                done
                echo ""
            else
                warn "No vLLM server found"
                echo ""
                echo "    Start one with: ./olivia.sh server start <preset>" >&2
                echo "    List containers: ./olivia.sh server list" >&2
            fi
            ;;
        deploy)
            deploy_server_script
            ;;
        cancel)
            cancel_job
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            ;;
        start)
            # Resolve container name and resources
            local num_nodes=1
            local gpus_per_node=4
            local pp_size=1
            if [[ -n "$preset" ]]; then
                container=$(resolve_container_name "$preset" "$index")
                if [[ -z "$model" ]]; then
                    model=$(get_default_model "$preset")
                fi
                read -r num_nodes gpus_per_node pp_size < <(get_preset_resources "$preset")
            fi

            if [[ -z "$container" ]]; then
                error "No container specified"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server start <preset>              # e.g., glm47, devstral" >&2
                echo "    ./olivia.sh server start --container <name>    # explicit container" >&2
                echo "" >&2
                list_server_containers
                exit 1
            fi

            if [[ -z "$model" ]]; then
                error "No model specified"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server start glm47 --model QuantTrio/GLM-4.7-AWQ" >&2
                echo "" >&2
                exit 1
            fi

            # Check container exists
            if ! check_container_exists "$container"; then
                error "Container not found: ${container}"
                echo "" >&2
                list_server_containers
                exit 1
            fi

            # Deploy if requested
            if $do_deploy; then
                deploy_server_script || exit 1
                echo ""
            fi

            # Check if server already running
            if find_job_id >/dev/null 2>&1; then
                warn "A vLLM server is already running"
                echo "    Cancel it first with: ./olivia.sh server cancel" >&2
                echo "    Or use 'restart' to replace it" >&2
                exit 1
            fi

            # Close existing tunnel
            if is_tunnel_alive; then
                info "Closing existing tunnel..."
                kill_tunnel
            fi

            # Start server
            local job_id
            if ! job_id=$(start_server_job "$container" "$model" "$num_nodes" "$gpus_per_node" "$pp_size"); then
                exit 1
            fi

            if $tail_logs; then
                local log_file="${REMOTE_CONTAINER_DIR}/logs/vllm_server_${job_id}.log"
                tail_job_logs "$log_file" "$job_id"
                echo ""
                info "Connect with: ./olivia.sh chat"
            else
                echo ""
                info "Server starting in background"
                echo "    Check status: ./olivia.sh server status" >&2
                echo "    View logs:    ./olivia.sh server logs" >&2
                echo "    Connect:      ./olivia.sh chat" >&2
            fi
            ;;
        restart)
            # For restart, we need container and model
            local num_nodes=1
            local gpus_per_node=4
            local pp_size=1
            if [[ -n "$preset" ]]; then
                container=$(resolve_container_name "$preset" "$index")
                if [[ -z "$model" ]]; then
                    model=$(get_default_model "$preset")
                fi
                read -r num_nodes gpus_per_node pp_size < <(get_preset_resources "$preset")
            fi

            # If no container specified, try to get from running job or show error
            if [[ -z "$container" ]]; then
                error "No container specified for restart"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server restart <preset>              # e.g., glm47, devstral" >&2
                echo "    ./olivia.sh server restart --container <name> --model <model>" >&2
                exit 1
            fi

            if [[ -z "$model" ]]; then
                error "No model specified for restart"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server restart glm47 --model QuantTrio/GLM-4.7-AWQ" >&2
                exit 1
            fi

            # Check container exists
            if ! check_container_exists "$container"; then
                error "Container not found: ${container}"
                echo "" >&2
                list_server_containers
                exit 1
            fi

            # Deploy if requested
            if $do_deploy; then
                deploy_server_script || exit 1
                echo ""
            fi

            # Cancel existing job
            cancel_job
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            echo ""

            # Start server
            local job_id
            if ! job_id=$(start_server_job "$container" "$model" "$num_nodes" "$gpus_per_node" "$pp_size"); then
                exit 1
            fi

            if $tail_logs; then
                local log_file="${REMOTE_CONTAINER_DIR}/logs/vllm_server_${job_id}.log"
                tail_job_logs "$log_file" "$job_id"
                echo ""
                info "Connect with: ./olivia.sh chat"
            fi
            ;;
        logs)
            local job_id
            if job_id=$(find_job_id); then
                local wrap_log head_log
                wrap_log="$(server_log_wrapper "$job_id")"
                head_log="$(server_log_head "$job_id")"
                info "Tailing logs for job ${job_id}..."
                # Tail both wrapper and head log. `tail -F` retries files that
                # don't exist yet (single-node jobs never create the head log;
                # multi-node jobs write post-bootstrap output only to the head).
                # Stderr is suppressed to silence the `cannot open ... No such
                # file or directory` / `has appeared; following new file`
                # chatter that tail -F emits during the wait — legitimate
                # process errors (ssh connection loss, etc.) still surface via
                # ssh_run's exit status.
                ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -F '${wrap_log}' '${head_log}' 2>/dev/null" || true
            else
                error "No running vLLM server found"
                exit 1
            fi
            ;;
        ssh)
            # Find the node running the vLLM server
            local node_name
            if ! node_name=$(find_vllm_node); then
                error "No running vLLM server found"
                echo "" >&2
                echo "    Start one with: ./olivia.sh server start <preset>" >&2
                exit 1
            fi

            success "Opening shell on GPU node: ${node_name}"
            echo "    (Type 'exit' to return)" >&2
            echo "" >&2

            # SSH to Olivia, then SSH to the GPU node
            ssh_run -t "${REMOTE_USER}@${REMOTE_HOST}" "ssh ${node_name}"
            ;;
        watch)
            server_watch
            ;;
    esac
}

cmd_server_help() {
    cat <<EOF
Usage: ./olivia.sh server <action> [options]

Manage vLLM server on Olivia.

Actions:
    start <preset>      Start vLLM server with a container
    restart <preset>    Cancel running job and start new one
    stop, cancel        Cancel running vLLM job
    status              Show running server status
    watch               Smart monitor: GPU loading -> health check -> ready
    list, ls            List available containers
    logs                Tail logs of running server
    ssh, shell          Open a shell on the GPU node
    deploy              Upload run_vllm_server.sh to cluster

Presets (with default models):
    glm51               GLM-5.1-AWQ (cyankiwi/GLM-5.1-AWQ-4bit) — 2 nodes × 4 GPUs, TP=4 + PP=2
    glm52               GLM-5.2-FP8 (RedHatAI/GLM-5.2-FP8) — 3 nodes × 4 GPUs, TP=4 + PP=3 (needs vLLM main + PR#45895)
    glm47               GLM-4.7-AWQ (QuantTrio/GLM-4.7-AWQ)
    devstral            Devstral 123B (mistralai/Devstral-2-123B-Instruct-2512)
    llama               Llama 3.3 70B (meta-llama/Llama-3.3-70B-Instruct)
    qwen                Qwen 2.5 72B (Qwen/Qwen2.5-72B-Instruct)

Options:
    --container, -c     Explicit container name
    --model, -m         HuggingFace model ID
    --index, -i         Container index (default: 1)
    --deploy, -d        Deploy script before action
    --no-tail           Don't tail logs after starting
    -h, --help          Show this help

Examples:
    ./olivia.sh server list                          List containers
    ./olivia.sh server start glm47                   Start GLM-4.7 server
    ./olivia.sh server watch                         Monitor loading progress
    ./olivia.sh server start devstral                Start Devstral server
    ./olivia.sh server start glm47 --index 2         Use vllm-glm47-2-sandbox
    ./olivia.sh server start --container vllm-glm47-1-sandbox --model QuantTrio/GLM-4.7-AWQ
    ./olivia.sh server restart glm47 -d              Deploy and restart
    ./olivia.sh server status                        Check running server
    ./olivia.sh server logs                          Tail server logs
    ./olivia.sh server ssh                           Shell into GPU node
    ./olivia.sh server cancel                        Stop running server
EOF
}

# =============================================================================
# MODULE: Tunnel
# =============================================================================

tunnel_status() {
    if is_tunnel_alive; then
        local node target
        node=$(get_tunnel_node)
        target=$(cat "${TUNNEL_TARGET_FILE}" 2>/dev/null || echo "")
        success "Tunnel ACTIVE"
        echo "    Node:   $node"
        echo "    Local:  localhost:${LOCAL_PORT}"
        if [[ "$target" == localhost:* ]]; then
            echo "    Mode:   login-proxy (relay on ${REMOTE_HOST}:${target#localhost:})"
            echo "    Path:   localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${target#localhost:} -> ${node}:${REMOTE_PORT}"
        else
            echo "    Remote: ${node}:${REMOTE_PORT}"
        fi
    else
        warn "Tunnel NOT ACTIVE"
    fi
}

cmd_tunnel() {
    local action="${1:-status}"
    shift || true

    # Flags apply to up/refresh.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --login-proxy)      LOGIN_PROXY=1; shift ;;
            --login-proxy-port) LOGIN_PROXY_PORT="$2"; shift 2 ;;
            --port)             set_local_port "$2"; shift 2 ;;
            -h|--help)          cmd_tunnel_help; exit 0 ;;
            *) error "Unknown tunnel option: $1"; cmd_tunnel_help; exit 1 ;;
        esac
    done

    case "$action" in
        up|open|start|refresh)
            ensure_master_connection || exit 1
            local gpu_node
            if ! gpu_node=$(find_vllm_node); then
                error "No running vLLM job found"
                echo "" >&2
                info "Start a server with: ./olivia.sh server restart"
                exit 1
            fi
            setup_tunnel "$gpu_node"
            ;;
        down|close|kill)
            kill_tunnel
            ;;
        status)
            tunnel_status
            ;;
        -h|--help)
            cmd_tunnel_help
            exit 0
            ;;
        *)
            error "Unknown action: $action"
            cmd_tunnel_help
            exit 1
            ;;
    esac
}

cmd_tunnel_help() {
    cat <<EOF
Usage: ./olivia.sh tunnel <action> [options]

Manage SSH tunnel to vLLM server.

Actions:
    up, open, start     Open tunnel to running vLLM server
    refresh             Re-point tunnel at the current node (after a job move)
    down, close, kill   Close tunnel (and stop the login-node relay if used)
    status              Show tunnel status (default)

Options:
    --port PORT             Local port (default: ${LOCAL_PORT})
    --login-proxy           Route via a fixed login-node relay that follows the
                            GPU node across job restarts (opt-in; runs a small
                            long-lived process on the shared login node)
    --login-proxy-port PORT Login-node relay port (default: ${LOGIN_PROXY_PORT})

Examples:
    ./olivia.sh tunnel up                  Open a direct tunnel
    ./olivia.sh tunnel up --login-proxy    Open via the follow-the-node relay
    ./olivia.sh tunnel refresh             Re-point after the job moved nodes
    ./olivia.sh tunnel down                Close tunnel
    ./olivia.sh tunnel status              Check tunnel status
EOF
}

# =============================================================================
# MODULE: Chat
# =============================================================================

# Check if vLLM server is running and ready
check_server_running() {
    local job_info
    job_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "squeue -u \$USER -h -o '%j %i %N %T' | grep -i vllm" 2>/dev/null) || true

    if [[ -z "$job_info" ]]; then
        return 1
    fi

    # Check if job is RUNNING (not PENDING)
    if echo "$job_info" | grep -q "RUNNING"; then
        return 0
    fi

    return 1
}

# Get server job info for display
get_server_info() {
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "squeue -u \$USER -h -o '%j %i %N %T %M' | grep -i vllm" 2>/dev/null || true
}

cmd_chat() {
    local stream_flag="--stream"
    local tunnel_only=false
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-stream)
                stream_flag=""
                shift
                ;;
            --tunnel-only)
                tunnel_only=true
                shift
                ;;
            --port)
                set_local_port "$2"
                shift 2
                ;;
            --login-proxy)
                LOGIN_PROXY=1
                shift
                ;;
            --login-proxy-port)
                LOGIN_PROXY_PORT="$2"
                shift 2
                ;;
            --no-store)
                extra_args+=("--no-store")
                shift
                ;;
            --resume)
                extra_args+=("--resume")
                # --resume takes an optional numeric ID; consume only if next
                # arg looks like one, otherwise leave it as a bare flag.
                if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
                    extra_args+=("$2")
                    shift 2
                else
                    shift
                fi
                ;;
            -h|--help)
                cmd_chat_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    ensure_master_connection || exit 1

    # First, check if server is running at all
    info "Checking for running vLLM server..."

    local server_info
    server_info=$(get_server_info)

    if [[ -z "$server_info" ]]; then
        error "No vLLM server found"
        echo "" >&2
        echo "No vLLM job is currently running on Olivia." >&2
        echo "" >&2
        info "To start a server:"
        echo "    ./olivia.sh server start glm47      # Start GLM-4.7" >&2
        echo "    ./olivia.sh server start devstral   # Start Devstral" >&2
        echo "    ./olivia.sh server list             # List available containers" >&2
        exit 1
    fi

    # Check if job is RUNNING vs PENDING
    if ! echo "$server_info" | grep -q "RUNNING"; then
        warn "vLLM server job found but not yet running"
        echo "" >&2
        echo "Job status:" >&2
        echo "$server_info" | while read -r name id node state time; do
            echo "    $name (Job $id): $state" >&2
        done
        echo "" >&2
        info "Wait for the job to start, or check logs:"
        echo "    ./olivia.sh server logs" >&2
        echo "    ./olivia.sh server status" >&2
        exit 1
    fi

    # Server is running - extract node info for display
    local running_node running_job_id running_time
    running_node=$(echo "$server_info" | grep "RUNNING" | awk '{print $3}')
    running_job_id=$(echo "$server_info" | grep "RUNNING" | awk '{print $2}')
    running_time=$(echo "$server_info" | grep "RUNNING" | awk '{print $5}')

    success "vLLM server is running"
    echo "    Node:    ${running_node}" >&2
    echo "    Job ID:  ${running_job_id}" >&2
    echo "    Uptime:  ${running_time}" >&2

    # Check/setup tunnel
    local gpu_node
    local need_new_tunnel=true

    if is_tunnel_alive; then
        local existing_node
        existing_node=$(get_tunnel_node)
        info "Checking if server is still on ${existing_node}..."
        if gpu_node=$(find_vllm_node 2>/dev/null) && [[ "$gpu_node" == "$existing_node" ]]; then
            success "Tunnel valid for ${gpu_node}"
            need_new_tunnel=false
        elif [[ -n "${gpu_node:-}" ]]; then
            warn "Server moved from ${existing_node} to ${gpu_node}"
        fi
    fi

    if $need_new_tunnel; then
        if ! gpu_node=$(find_vllm_node); then
            error "Could not find server node"
            exit 1
        fi

        if ! setup_tunnel "$gpu_node"; then
            exit 1
        fi
    fi

    if $tunnel_only; then
        echo ""
        success "Tunnel ready"
        echo "    Connect: python chat_devstral.py localhost --port ${LOCAL_PORT} --stream"
        echo "    Close:   ./olivia.sh tunnel down"
        exit 0
    fi

    # Start chat
    echo ""
    info "Starting chat client..."
    echo ""

    if [[ ! -f "${CHAT_SCRIPT}" ]]; then
        error "Chat script not found: ${CHAT_SCRIPT}"
        exit 1
    fi

    python3 "${CHAT_SCRIPT}" localhost --port "${LOCAL_PORT}" ${stream_flag} \
        ${extra_args[@]+"${extra_args[@]}"}

    echo ""
    info "Chat ended. Tunnel remains open on port ${LOCAL_PORT}"
    echo "    Reconnect: ./olivia.sh chat"
    echo "    Close:     ./olivia.sh tunnel down"
}

cmd_chat_help() {
    cat <<EOF
Usage: ./olivia.sh chat [options]

Connect to vLLM server and start interactive chat.

Options:
    --port PORT             Local port (default: 8000)
    --tunnel-only           Only set up tunnel, don't start chat
    --no-stream             Disable streaming in chat
    --resume [ID]           Resume most recent stored conversation, or specific ID
    --no-store              Disable conversation persistence for this session
    --login-proxy           Route via the follow-the-node login-node relay
    --login-proxy-port PORT Login-node relay port (default: ${LOGIN_PROXY_PORT})
    -h, --help              Show this help

Examples:
    ./olivia.sh chat                   Connect and start chat
    ./olivia.sh chat --port 9000       Use different port
    ./olivia.sh chat --tunnel-only     Just set up tunnel
    ./olivia.sh chat --resume          Resume most recent conversation
    ./olivia.sh chat --resume 12       Resume conversation #12
    ./olivia.sh chat --login-proxy     Connect via the follow-the-node relay
EOF
}

# =============================================================================
# MODULE: Status
# =============================================================================

# Enriched job listing for `status`. Prints the base squeue line per job, then
# for each RUNNING vLLM-server job a derived phase line — LOADING <shards>,
# CAPTURING cudagraphs, SERVING (+ live throughput / KV usage / health), or
# ERROR — plus build-job progress. All derived in a single remote pass over the
# job logs so `status` stays one round trip.
print_jobs_enriched() {
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "CDIR='${REMOTE_CONTAINER_DIR}' VPORT='${REMOTE_PORT}' bash -s" <<'REMOTE_JOBS' 2>/dev/null || true
jobs=$(squeue -u "$USER" -h -o '%i|%j|%T|%M|%l|%R' 2>/dev/null)
if [ -z "$jobs" ]; then echo "  (no jobs in queue)"; exit 0; fi
printf '  %-9s %-18s %-10s %-13s %s\n' JOBID NAME STATE TIME/LIMIT NODES
echo "$jobs" | while IFS='|' read -r jid name state etime tlimit nodes; do
  printf '  %-9s %-18s %-10s %-13s %s\n' "$jid" "$name" "$state" "$etime/$tlimit" "$nodes"
  [ "$state" = RUNNING ] || continue
  case "$name" in
    *vllm-server*)
      head=$(scontrol show hostnames "$nodes" 2>/dev/null | head -1)
      hl="$CDIR/logs/vllm_server_${jid}_head.log"; wl="$CDIR/logs/vllm_server_${jid}.log"
      model=$(grep -oE 'Model: +[^[:space:]]+' "$wl" 2>/dev/null | head -1 | awk '{print $NF}')
      mode=$(grep -oE 'CUDAGraph Mode: +[^[:space:]]+' "$wl" 2>/dev/null | head -1 | awk '{print $NF}')
      tag="${model:-?} [cg=${mode:-?}]"
      if [ -f "$hl" ] && grep -qE 'Application startup complete|Uvicorn running on' "$hl" 2>/dev/null; then
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$head:${VPORT}/health" 2>/dev/null)
        tp=$(grep -oE 'Avg generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%' "$hl" 2>/dev/null | tail -1)
        echo "             -> SERVING  $tag  http://$head:${VPORT} (health $code)"
        [ -n "$tp" ] && echo "                ${tp#Avg }"
      elif [ -f "$hl" ] && grep -qE 'illegal memory access|Traceback \(most recent call last\)|EngineDeadError|Engine core initialization failed|Fatal Python error' "$hl" 2>/dev/null; then
        echo "             -> ERROR  $tag  (engine failed; tail logs/vllm_server_${jid}_head.log)"
      elif [ -f "$hl" ] && grep -qE 'Capturing cudagraph|Capturing CUDA graph' "$hl" 2>/dev/null; then
        echo "             -> CAPTURING cudagraphs  $tag  (weights loaded)"
      elif [ -f "$hl" ]; then
        sh=$(grep -oE 'Loading safetensors checkpoint shards: +[0-9]+% Completed \| [0-9]+/[0-9]+' "$hl" 2>/dev/null | tail -1)
        if [ -n "$sh" ]; then
          nn=$(echo "$sh" | grep -oE '[0-9]+/[0-9]+' | head -1); pct=$(echo "$sh" | grep -oE '[0-9]+%' | head -1)
          echo "             -> LOADING weights ${nn} (${pct})  $tag"
        else echo "             -> INIT / Ray bootstrap  $tag"; fi
      else echo "             -> starting (no head log yet)  $tag"; fi
      ;;
    *build-vllm*)
      bl="$CDIR/build_vllm_${jid}.log"
      if [ -f "$bl" ]; then
        if grep -qE 'Build Complete!' "$bl" 2>/dev/null; then echo "             -> build COMPLETE"
        else stp=$(grep -oE '\[[0-9]+/[0-9]+\]' "$bl" 2>/dev/null | tail -1); echo "             -> building ${stp:-(deps/clone phase)}"; fi
      fi
      ;;
  esac
done
REMOTE_JOBS
}

cmd_status() {
    header "Olivia Status"

    # SSH connection
    if is_master_alive; then
        success "SSH Connection: ACTIVE"
        echo "    Persist: ${SSH_CONTROL_PERSIST} idle (single 2FA covers this window)"
    else
        warn "SSH Connection: NOT ACTIVE"
        echo "    Run any command to establish connection (or: ./olivia.sh reconnect)"
    fi

    echo ""

    # Tunnel
    if is_tunnel_alive; then
        local node target
        node=$(get_tunnel_node)
        target=$(cat "${TUNNEL_TARGET_FILE}" 2>/dev/null || echo "")
        success "SSH Tunnel: ACTIVE"
        echo "    Node:   $node"
        echo "    Local:  localhost:${LOCAL_PORT}"
        if [[ "$target" == localhost:* ]]; then
            echo "    Mode:   login-proxy (relay on ${REMOTE_HOST}:${target#localhost:})"
        fi
    else
        warn "SSH Tunnel: NOT ACTIVE"
    fi

    echo ""

    # Remote jobs (if connected)
    if is_master_alive || ensure_master_connection 2>/dev/null; then
        info "Jobs on Olivia:"
        print_jobs_enriched
    fi
}

# =============================================================================
# MODULE: Cluster utilization
# =============================================================================

cmd_cluster() {
    local watch_mode=false
    local interval=30
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --watch|-w)
                watch_mode=true
                shift
                ;;
            --interval|-n)
                interval="$2"
                shift 2
                ;;
            -h|--help)
                cmd_cluster_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                cmd_cluster_help
                exit 1
                ;;
        esac
    done
    if [[ ! "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -lt 5 ]]; then
        error "--interval must be an integer >= 5 (got: $interval)"
        exit 1
    fi

    if $watch_mode; then
        # `watch` clears the screen between iterations, but we want to keep
        # color output and avoid re-launching ssh from scratch each tick (the
        # ControlMaster handles that already, but `watch` re-execs the whole
        # command). Use a simple loop with clear+sleep instead.
        info "Watching cluster (refresh every ${interval}s, Ctrl-C to stop)"
        ensure_master_connection || exit 1
        while true; do
            clear
            _cluster_snapshot
            printf "\n  ${BLD:-}refreshing in ${interval}s — Ctrl-C to stop${RST:-}\n"
            sleep "$interval"
        done
        return
    fi

    ensure_master_connection || exit 1
    _cluster_snapshot
}

# All data-gathering + formatting happens on the remote side in one SSH call
# (cheap thanks to ControlMaster). The remote script outputs colored text
# that we just stream back to the user's terminal.
_cluster_snapshot() {
    local remote_script
    remote_script=$(cat <<'REMOTE_SCRIPT'
set -u
PARTITION="accel"

# ANSI color helpers (mirror the local olivia.sh palette).
RED='\033[1;31m'; YEL='\033[1;33m'; GRN='\033[1;32m'; BLU='\033[1;34m'
MAG='\033[1;35m'; CYA='\033[1;36m'; BLD='\033[1m'; RST='\033[0m'

printf "\n${MAG}Olivia Cluster Utilization (partition=${PARTITION})${RST}\n\n"

# --- 1. Your jobs ------------------------------------------------------------
printf "${BLU}==>${RST} ${BLD}Your jobs${RST}\n"
running_count=$(squeue -h -u "$USER" -t RUNNING -o "%i" | wc -l)
pending_count=$(squeue -h -u "$USER" -t PENDING -o "%i" | wc -l)
if [ "$running_count" -eq 0 ] && [ "$pending_count" -eq 0 ]; then
    printf "    (none)\n"
else
    squeue -u "$USER" -h -o '%i|%j|%T|%M|%l|%R' | while IFS='|' read -r jid name state etime tlimit nodes; do
        printf "  ${BLD}%-8s${RST} %-16s %-9s %-13s %s\n" "$jid" "$name" "$state" "$etime/$tlimit" "$nodes"
        [ "$state" = RUNNING ] || continue
        case "$name" in
          (*vllm-server*)
            head=$(scontrol show hostnames "$nodes" 2>/dev/null | head -1)
            hl="$CDIR/logs/vllm_server_${jid}_head.log"; wl="$CDIR/logs/vllm_server_${jid}.log"
            model=$(grep -oE 'Model: +[^[:space:]]+' "$wl" 2>/dev/null | head -1 | awk '{print $NF}')
            mode=$(grep -oE 'CUDAGraph Mode: +[^[:space:]]+' "$wl" 2>/dev/null | head -1 | awk '{print $NF}')
            tag="${model:-?} [cg=${mode:-?}]"
            if [ -f "$hl" ] && grep -qE 'Application startup complete|Uvicorn running on' "$hl" 2>/dev/null; then
                code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$head:${VPORT}/health" 2>/dev/null)
                tp=$(grep -oE 'Avg generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%' "$hl" 2>/dev/null | tail -1)
                printf "           ${GRN}SERVING${RST}  %s  http://%s:%s (health %s)\n" "$tag" "$head" "$VPORT" "$code"
                [ -n "$tp" ] && printf "             %s\n" "${tp#Avg }"
            elif [ -f "$hl" ] && grep -qE 'illegal memory access|Traceback \(most recent call last\)|EngineDeadError|Engine core initialization failed|Fatal Python error' "$hl" 2>/dev/null; then
                printf "           ${RED}ERROR${RST}  %s  (engine failed; tail logs/vllm_server_%s_head.log)\n" "$tag" "$jid"
            elif [ -f "$hl" ] && grep -qE 'Capturing cudagraph|Capturing CUDA graph' "$hl" 2>/dev/null; then
                printf "           ${YEL}CAPTURING${RST} cudagraphs  %s  (weights loaded)\n" "$tag"
            elif [ -f "$hl" ]; then
                shd=$(grep -oE 'Loading safetensors checkpoint shards: +[0-9]+% Completed \| [0-9]+/[0-9]+' "$hl" 2>/dev/null | tail -1)
                if [ -n "$shd" ]; then
                    nn=$(echo "$shd" | grep -oE '[0-9]+/[0-9]+' | head -1); pct=$(echo "$shd" | grep -oE '[0-9]+%' | head -1)
                    printf "           ${YEL}LOADING${RST} weights %s (%s)  %s\n" "$nn" "$pct" "$tag"
                else printf "           ${YEL}INIT${RST} / Ray bootstrap  %s\n" "$tag"; fi
            else printf "           starting (no head log yet)  %s\n" "$tag"; fi
            ;;
          (*build-vllm*)
            bl="$CDIR/build_vllm_${jid}.log"
            if [ -f "$bl" ]; then
                if grep -qE 'Build Complete!' "$bl" 2>/dev/null; then printf "           ${GRN}build COMPLETE${RST}\n"
                else stp=$(grep -oE '\[[0-9]+/[0-9]+\]' "$bl" 2>/dev/null | tail -1); printf "           ${YEL}building${RST} %s\n" "${stp:-(deps/clone phase)}"; fi
            fi
            ;;
        esac
    done
    if [ "$pending_count" -gt 0 ]; then
        printf "\n  ${BLD}Estimated start (PENDING):${RST}\n"
        squeue --start -h -u "$USER" -o "  %.8i  %.20S  %R" 2>/dev/null || true
    fi
fi
echo

# --- 2. Partition node states ------------------------------------------------
# Pull "<state> <gres_used>" per node, count states, sum used vs total GPUs.
# State suffixes (@,*,#,~) are stripped to base state. -N forces one row per
# node (default sinfo aggregates by identical state+gres fields).
sinfo_data=$(sinfo -p "$PARTITION" -h -N --Format="StateLong:20,GresUsed:40")

total_nodes=$(echo "$sinfo_data" | grep -c .)
total_gpus=$(( total_nodes * 4 ))   # all accel nodes are h200:4

declare -A state_count free_slot_count
used_gpus=0; nonusable_gpus=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    state=$(echo "$line" | awk '{print $1}' | sed 's/[@*#~+]$//')
    gres_used=$(echo "$line" | awk '{print $2}')
    used_n=$(echo "$gres_used" | sed -n 's/^gpu:h200:\([0-9]*\).*/\1/p')
    used_n=${used_n:-0}
    state_count[$state]=$(( ${state_count[$state]:-0} + 1 ))
    used_gpus=$(( used_gpus + used_n ))
    case "$state" in
        (draining|drained|down|reserved|maint|fail*|unknown)
            nonusable_gpus=$(( nonusable_gpus + 4 - used_n ))
            ;;
        (idle|mixed|allocated)
            # Bucket schedulable nodes by how many GPUs are free on each.
            # The bucket count at key=4 is the number of nodes that can serve
            # a fresh single-node 4-GPU request right now.
            free_on_node=$(( 4 - used_n ))
            free_slot_count[$free_on_node]=$(( ${free_slot_count[$free_on_node]:-0} + 1 ))
            ;;
    esac
done <<< "$sinfo_data"

free_gpus=$(( total_gpus - used_gpus - nonusable_gpus ))
[ "$free_gpus" -lt 0 ] && free_gpus=0
full_nodes=${free_slot_count[4]:-0}

printf "${BLU}==>${RST} ${BLD}Partition state${RST}  (h200, $total_nodes nodes / $total_gpus GPUs total)\n"
printf "  ${GRN}free:${RST}          %3d GPUs  (%d full nodes)\n" "$free_gpus" "$full_nodes"
printf "  ${YEL}in use:${RST}        %3d GPUs\n" "$used_gpus"
printf "  ${RED}unavailable:${RST}   %3d GPUs  (drained/reserved/down)\n\n" "$nonusable_gpus"

printf "  ${BLD}Node-state breakdown:${RST}\n"
for state in "${!state_count[@]}"; do
    printf "    %-12s %3d nodes\n" "$state" "${state_count[$state]}"
done | sort
echo

# Fragmentation: explains why "free GPUs" can be high while pending jobs wait.
# A 4-GPU single-node job needs one full slot; a 2-node × 4-GPU job needs two.
# Scattered 1- or 2-GPU holes don't satisfy either, even if they sum high.
printf "  ${BLD}Free-slot fragmentation${RST} (schedulable nodes only):\n"
full4=${free_slot_count[4]:-0}
# Max across the 4 buckets for bar scaling.
frag_max=0
for k in 4 3 2 1; do
    c=${free_slot_count[$k]:-0}
    [ "$c" -gt "$frag_max" ] && frag_max=$c
done
[ "$frag_max" -eq 0 ] && frag_max=1
any_free=0
for k in 4 3 2 1; do
    c=${free_slot_count[$k]:-0}
    [ "$c" -eq 0 ] && continue
    any_free=1
    case "$k" in
        (4) label="full 4-GPU nodes"; color="$GRN";;
        (3) label="3-GPU holes      "; color="$YEL";;
        (2) label="2-GPU holes      "; color="$YEL";;
        (1) label="1-GPU holes      "; color="$YEL";;
    esac
    # Bar + trailing pad (manual — awk printf would work here too but keeping
    # pure bash for consistency with the rest of the block).
    w=$(( c * 20 / frag_max )); [ "$w" -lt 1 ] && w=1
    bar=""
    for ((i=0; i<w; i++)); do bar="${bar}█"; done
    pad=""
    for ((i=0; i<20-w; i++)); do pad="${pad} "; done
    # %b (not %s) on color args — bash printf only interprets \033 escapes
    # in the format string, or in %b arguments. $color is a literal-backslash
    # string here (from the remote-script's RED='\033[1;33m' definitions).
    printf "    %s  %b%s%b%s %3d\n" "$label" "$color" "$bar" "$RST" "$pad" "$c"
done
[ "$any_free" -eq 0 ] && printf "    (no free slots on schedulable nodes)\n"
# Multi-node feasibility callouts: glm51 (PP=2) needs 2 full nodes; glm52
# (PP=3) needs 3 full nodes.
if [ "$full4" -ge 2 ]; then
    printf "    ${GRN}→${RST} 2-node × 4-GPU shape (glm51) can start now (%d full nodes available)\n" "$full4"
else
    printf "    ${YEL}→${RST} 2-node × 4-GPU shape (glm51) cannot start now (%d/2 full nodes available)\n" "$full4"
fi
if [ "$full4" -ge 3 ]; then
    printf "    ${GRN}→${RST} 3-node × 4-GPU shape (glm52) can start now (%d full nodes available)\n" "$full4"
else
    printf "    ${YEL}→${RST} 3-node × 4-GPU shape (glm52) cannot start now (%d/3 full nodes available)\n" "$full4"
fi
echo

# --- 3. Queue summary --------------------------------------------------------
printf "${BLU}==>${RST} ${BLD}Queue (partition=${PARTITION})${RST}\n"
total_running=$(squeue -p "$PARTITION" -h -t RUNNING -o "%i" | wc -l)
total_pending=$(squeue -p "$PARTITION" -h -t PENDING -o "%i" | wc -l)
unique_users=$(squeue -p "$PARTITION" -h -o "%u" | sort -u | wc -l)
printf "  RUNNING: %3d jobs across %d users\n" "$total_running" "$unique_users"
printf "  PENDING: %3d jobs in queue\n" "$total_pending"

# Pending-reason histogram — SLURM's own verdict on why jobs aren't running.
# Useful for distinguishing normal queue pressure (Priority/Resources) from
# cluster-wide issues (ReqNodeNotAvail during maintenance, QOS ceilings).
# Capture the aggregated counts once and reuse for both display and metrics.
reasons_tally=""
if [ "$total_pending" -gt 0 ]; then
    reasons_tally=$(squeue -p "$PARTITION" -h -t PENDING -o '%R' \
        | sed 's/^(\(.*\))$/\1/' \
        | sort | uniq -c | sort -rn)
fi
if [ -n "$reasons_tally" ]; then
    printf "\n  ${BLD}Pending reasons${RST}:\n"
    # Max count for bar scaling (tally is sorted desc, so first line is max).
    max_cnt=$(echo "$reasons_tally" | head -1 | awk '{print $1}')
    [ -z "$max_cnt" ] && max_cnt=1
    # Bars color-coded by severity: red for cluster-wide blockers (maintenance,
    # QOS ceilings), yellow for resource pressure, green for normal queue states.
    echo "$reasons_tally" | head -8 \
        | awk -v max="$max_cnt" \
              -v RED="$RED" -v YEL="$YEL" -v GRN="$GRN" -v RST="$RST" '
        {
            n = $1; $1 = ""; sub(/^ /, ""); reason = $0
            w = int(n * 20 / max); if (w < 1) w = 1
            color = GRN
            if (reason ~ /ReqNodeNotAvail|Reserved|UnavailableNodes|QOS|AssocMax|Licenses/) color = RED
            else if (reason ~ /Resources/) color = YEL
            # Build bar + trailing space-padding manually: awk printf width
            # specifiers count bytes, and "█" is 3 bytes in UTF-8.
            bar = ""; for (i = 0; i < w; i++) bar = bar "█"
            pad = ""; for (i = 0; i < 20 - w; i++) pad = pad " "
            printf "    %-26s %s%s%s%s %d\n", reason, color, bar, RST, pad, n
        }'
fi

# Your position in the pending queue (rank by submit time, oldest first).
if [ "$pending_count" -gt 0 ]; then
    your_first=$(squeue -h -u "$USER" -t PENDING -o "%V" | sort | head -1)
    if [ -n "$your_first" ]; then
        ahead=$(squeue -p "$PARTITION" -h -t PENDING -o "%V" | awk -v t="$your_first" '$1<t' | wc -l)
        printf "  Your earliest pending job is behind ${BLD}%d${RST} other(s) by submit time\n" "$ahead"
    fi
fi

# Top blockers — longest-running 2+ node jobs (those most likely holding the
# resources a multi-node vLLM job needs). Show 5.
printf "\n  ${BLD}Top long-running multi-node jobs (potential blockers):${RST}\n"
squeue -p "$PARTITION" -h -t RUNNING -o "%.8i %.10u %.5D %.12M %.12l %R" | \
    awk '$3 >= 2' | sort -k 4 -r | head -5 | \
    awk 'BEGIN{printf "    %-9s %-11s %-5s %-12s %-12s %s\n","JOBID","USER","NODES","ELAPSED","TIME_LIMIT","NODELIST"}
         {printf "    %-9s %-11s %-5s %-12s %-12s %s\n",$1,$2,$3,$4,$5,$6}'
echo

# --- 4. Reservations ---------------------------------------------------------
# Use `scontrol -o` for one-line-per-reservation output so awk can parse all
# fields together (default multi-line output spreads State=/Nodes=/StartTime=
# across separate lines).
printf "${BLU}==>${RST} ${BLD}Active / upcoming reservations${RST} (next 7 days)\n"
now_epoch=$(date +%s)
horizon=$(( now_epoch + 7*86400 ))
shown=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    name=$(echo "$line" | sed -n 's/.*ReservationName=\([^ ]*\).*/\1/p')
    state=$(echo "$line" | sed -n 's/.*State=\([^ ]*\).*/\1/p')
    st=$(echo "$line" | sed -n 's/.*StartTime=\([^ ]*\).*/\1/p')
    nodes=$(echo "$line" | sed -n 's/.*Nodes=\([^ ]*\).*/\1/p')
    [ -z "$st" ] && continue
    st_e=$(date -d "$st" +%s 2>/dev/null)
    [ -z "$st_e" ] && continue
    [ "$st_e" -gt "$horizon" ] && continue
    [ "$st_e" -lt "$now_epoch" ] && [ "$state" != "ACTIVE" ] && continue
    printf "  %-26s state=%-8s starts=%s nodes=%s\n" "$name" "$state" "$st" "$nodes"
    shown=$(( shown + 1 ))
    [ "$shown" -ge 10 ] && break
done < <(scontrol -o show reservation 2>/dev/null)
[ "$shown" -eq 0 ] && printf "  (none in horizon)\n"
echo

# --- 5. Estimated start for a hypothetical new job ---------------------------
# `sbatch --test-only` runs SLURM's backfill scheduler against the given shape
# and prints a predicted start time without actually submitting. Reservation-
# aware (upcoming maintenance windows are factored in), but assumes every
# running job runs to its full TIME_LIMIT — so the result is a *pessimistic
# upper bound*, not a likely start time.
printf "${BLU}==>${RST} ${BLD}Estimated start for a new job${RST}  (SLURM backfill — pessimistic upper bound)\n"
now_epoch=$(date +%s)

estimate_start() {
    nodes=$1; gpus=$2; label=$3
    out=$(sbatch --test-only \
        --partition="$PARTITION" \
        --nodes="$nodes" --ntasks-per-node=1 \
        --gpus-per-node="$gpus" --cpus-per-task=32 \
        --mem=0 --time=04:00:00 \
        --wrap='true' 2>&1)
    # Expected: "sbatch: Job N to start at YYYY-MM-DDTHH:MM:SS ..."
    ts=$(echo "$out" | sed -n 's/.*to start at \([0-9T:\-]*\).*/\1/p' | head -1)
    if [ -z "$ts" ]; then
        printf "  %-22s ${RED}prediction failed${RST}\n" "$label"
        return
    fi
    ts_e=$(date -d "$ts" +%s 2>/dev/null)
    if [ -z "$ts_e" ] || [ "$ts_e" -le "$((now_epoch + 60))" ]; then
        printf "  %-22s ${GRN}starts immediately${RST}\n" "$label"
        return
    fi
    delta=$(( ts_e - now_epoch ))
    if [ "$delta" -lt 3600 ]; then
        human="$(( delta / 60 ))m"
    elif [ "$delta" -lt 86400 ]; then
        human="$(( delta / 3600 ))h$(( (delta % 3600) / 60 ))m"
    else
        human="$(( delta / 86400 ))d$(( (delta % 86400) / 3600 ))h"
    fi
    printf "  %-22s starts %s  (~%s from now)\n" "$label" "$ts" "$human"
}

estimate_start 1 4 "1 node  × 4 GPUs:"
estimate_start 2 4 "2 nodes × 4 GPUs:"
estimate_start 3 4 "3 nodes × 4 GPUs:"
echo

# --- 6. Hint -----------------------------------------------------------------
printf "${BLU}==>${RST} ${BLD}Notes${RST}\n"
draining_count=$(echo "${state_count[draining]:-0}")
if [ "$draining_count" -gt 10 ]; then
    printf "  ${YEL}!${RST} %d nodes are draining — likely an upcoming maintenance.\n" "$draining_count"
    printf "    Multi-node jobs may wait significantly longer than the queue depth suggests.\n"
fi
if [ "$pending_count" -gt 1 ]; then
    printf "  ${YEL}!${RST} You have ${BLD}%d${RST} pending jobs of your own. Consider scancel'ing duplicates.\n" "$pending_count"
fi
echo

# --- Machine-readable metrics line for local persistence ---------------------
# Format: __METRICS__<TAB>free_total<TAB>f4<TAB>f3<TAB>f2<TAB>f1<TAB>pending<TAB>draining<TAB>reasons
# reasons: pipe-separated key=count pairs (e.g. "Priority=5|Resources=3"), or "-" if none.
# Local side captures this line, prepends its own timestamp, and appends to
# cache/cluster-samples.tsv for the "Last hour" trend panel.
reasons_str=$(echo "$reasons_tally" | awk '
    BEGIN{s=""}
    NF {n=$1; $1=""; sub(/^ /,""); s = s (s?"|":"") $0 "=" n}
    END{print (s?s:"-")}')
printf "__METRICS__\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n" \
    "$free_gpus" \
    "${free_slot_count[4]:-0}" \
    "${free_slot_count[3]:-0}" \
    "${free_slot_count[2]:-0}" \
    "${free_slot_count[1]:-0}" \
    "$total_pending" \
    "${state_count[draining]:-0}" \
    "$reasons_str"
REMOTE_SCRIPT
)
    local sample_file="${SCRIPT_DIR}/cache/cluster-samples.tsv"
    mkdir -p "$(dirname "$sample_file")"
    local local_ts
    local_ts=$(date +%s)

    # Pipe ssh output through awk: display lines pass through to the terminal
    # (with line-buffered flushes so --watch feels instant); the __METRICS__
    # sentinel line is diverted into the sample file, prefixed with the local
    # timestamp. Single SSH round-trip, no stderr juggling.
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "CDIR='${REMOTE_CONTAINER_DIR}' VPORT='${REMOTE_PORT}'; $remote_script" \
        | awk -v f="$sample_file" -v ts="$local_ts" '
            /^__METRICS__\t/ {
                sub(/^__METRICS__\t/, "")
                printf "%s\t%s\n", ts, $0 >> f
                close(f)
                next
            }
            { print; fflush() }
          '

    _prune_samples "$sample_file"
    _render_history_panel "$sample_file"
}

# Keep only samples from the last 24h. Rewrites the file in place.
_prune_samples() {
    local f="$1"
    [ ! -s "$f" ] && return
    local cutoff
    cutoff=$(( $(date +%s) - 86400 ))
    awk -F'\t' -v c="$cutoff" '$1 + 0 >= c' "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
}

# Render a "Last hour" panel from the accumulated samples: min/avg/max of
# free GPUs, free 4-GPU slots, and pending jobs — plus top pending reasons
# by average count across the window. Meaningful only with ≥2 samples in
# the last hour (use `cluster --watch` to build history quickly).
_render_history_panel() {
    local f="$1"
    local cutoff
    cutoff=$(( $(date +%s) - 3600 ))
    # Local ANSI escapes — top-level color vars live only inside the remote heredoc.
    local red=$'\033[1;31m' yel=$'\033[1;33m' grn=$'\033[1;32m'
    local blu=$'\033[1;34m' bld=$'\033[1m' dim=$'\033[2m' rst=$'\033[0m'

    printf "%s==>%s %sLast hour%s" "$blu" "$rst" "$bld" "$rst"

    if [ ! -s "$f" ]; then
        printf "  (no samples yet — run %scluster --watch%s to build history)\n\n" "$bld" "$rst"
        return
    fi

    local in_window
    in_window=$(awk -F'\t' -v c="$cutoff" '$1 + 0 >= c' "$f" | wc -l | tr -d ' ')

    if [ "$in_window" -eq 0 ]; then
        printf "  (no samples in last hour — run %scluster --watch%s to build history)\n\n" "$bld" "$rst"
        return
    fi
    if [ "$in_window" -eq 1 ]; then
        printf "  (1 sample — run again or use %s--watch%s for a trend)\n\n" "$bld" "$rst"
        return
    fi
    printf "\n"

    # Sparklines: Unicode block-height ramp (▁▂▃▄▅▆▇█) shows the shape of each
    # time series. When samples exceed the max width, we bucket-average down.
    awk -F'\t' -v c="$cutoff" \
        -v RED="$red" -v YEL="$yel" -v GRN="$grn" -v DIM="$dim" -v RST="$rst" '
    BEGIN {
        split("▁ ▂ ▃ ▄ ▅ ▆ ▇ █", spark, " ")
        maxw = 40
    }
    # Render one sparkline from a series. Returns a string of block-height
    # chars. When n > maxw, each output char represents an avg over a bucket.
    function render_spark(n, minv, maxv, vals,    i, j, lo, hi, sum, cnt, v, idx, out) {
        if (n == 0) return ""
        if (maxv == minv) {
            out = ""
            lim = (n < maxw) ? n : maxw
            for (i = 1; i <= lim; i++) out = out spark[4]
            return out
        }
        out = ""
        if (n <= maxw) {
            for (i = 1; i <= n; i++) {
                idx = int((vals[i] - minv) / (maxv - minv) * 7) + 1
                if (idx < 1) idx = 1; if (idx > 8) idx = 8
                out = out spark[idx]
            }
        } else {
            for (i = 1; i <= maxw; i++) {
                lo = int((i - 1) * n / maxw) + 1
                hi = int(i * n / maxw)
                if (hi < lo) hi = lo
                sum = 0; cnt = 0
                for (j = lo; j <= hi; j++) { sum += vals[j]; cnt++ }
                v = sum / cnt
                idx = int((v - minv) / (maxv - minv) * 7) + 1
                if (idx < 1) idx = 1; if (idx > 8) idx = 8
                out = out spark[idx]
            }
        }
        return out
    }
    $1 + 0 >= c {
        n++
        values_ft[n] = $2; values_f4[n] = $3; values_pd[n] = $7
        if (n == 1) {
            first_ts = $1
            min_ft = max_ft = $2
            min_f4 = max_f4 = $3
            min_pd = max_pd = $7
        } else {
            if ($2 < min_ft) min_ft = $2; if ($2 > max_ft) max_ft = $2
            if ($3 < min_f4) min_f4 = $3; if ($3 > max_f4) max_f4 = $3
            if ($7 < min_pd) min_pd = $7; if ($7 > max_pd) max_pd = $7
        }
        last_ts = $1
        sum_ft += $2; sum_f4 += $3; sum_pd += $7
        if ($9 != "" && $9 != "-") {
            ncnt = split($9, parts, "|")
            for (i = 1; i <= ncnt; i++) {
                eq = index(parts[i], "=")
                if (eq == 0) continue
                rk = substr(parts[i], 1, eq - 1)
                rv = substr(parts[i], eq + 1) + 0
                rsum[rk] += rv
            }
        }
    }
    END {
        span = last_ts - first_ts
        if (span < 90) {
            span_str = sprintf("%ds", span)
        } else if (span < 3600) {
            span_str = sprintf("%dm", int((span + 30) / 60))
        } else {
            span_str = sprintf("%dh%dm", int(span / 3600), int((span % 3600 + 30) / 60))
        }
        printf "  %s(%d samples over %s)%s\n", DIM, n, span_str, RST

        s_ft = render_spark(n, min_ft, max_ft, values_ft)
        s_f4 = render_spark(n, min_f4, max_f4, values_f4)
        s_pd = render_spark(n, min_pd, max_pd, values_pd)

        # Sparkline width (char count, not bytes) is identical across rows
        # because all three series share the same n, so column alignment of
        # the scalar summary on the right works without explicit padding.
        printf "  %-20s %s%s%s  min=%-3d avg=%-5.1f max=%-3d\n", "Free GPUs:",        GRN, s_ft, RST, min_ft, sum_ft/n, max_ft
        printf "  %-20s %s%s%s  min=%-3d avg=%-5.1f max=%-3d\n", "Free 4-GPU slots:", GRN, s_f4, RST, min_f4, sum_f4/n, max_f4
        printf "  %-20s %s%s%s  min=%-3d avg=%-5.1f max=%-3d\n", "Pending jobs:",     YEL, s_pd, RST, min_pd, sum_pd/n, max_pd

        # Top reasons — insertion sort on keys[] by rsum desc, then render
        # with colored bars scaled to the highest-average reason.
        ni = 0
        for (k in rsum) { keys[++ni] = k }
        for (j = 2; j <= ni; j++) {
            x = keys[j]; xv = rsum[x]; i = j - 1
            while (i >= 1 && rsum[keys[i]] < xv) { keys[i+1] = keys[i]; i-- }
            keys[i+1] = x
        }
        if (ni > 0) {
            max_avg = 0
            for (k in rsum) { a = rsum[k] / n; if (a > max_avg) max_avg = a }
            if (max_avg <= 0) max_avg = 1
            printf "  Top pending reasons (avg per sample):\n"
            shown = 0
            for (j = 1; j <= ni && shown < 5; j++) {
                avg = rsum[keys[j]] / n
                if (avg < 0.05) continue
                color = GRN
                if (keys[j] ~ /ReqNodeNotAvail|Reserved|UnavailableNodes|QOS|AssocMax|Licenses/) color = RED
                else if (keys[j] ~ /Resources/) color = YEL
                w = int(avg * 15 / max_avg); if (w < 1) w = 1
                bar = ""; for (i = 0; i < w; i++) bar = bar "█"
                pad = ""; for (i = 0; i < 15 - w; i++) pad = pad " "
                printf "    %-26s %s%s%s%s %.1f\n", keys[j], color, bar, RST, pad, avg
                shown++
            }
            if (shown == 0) printf "    %s(no pending jobs recorded in window)%s\n", DIM, RST
        }
        printf "\n"
    }
    ' "$f"
}

cmd_cluster_help() {
    cat <<EOF
Usage: ./olivia.sh cluster [--watch] [--interval SECS]

Show a snapshot of Olivia's accel partition: your jobs, GPU availability,
queue depth, top blockers, and active reservations.

Options:
    -w, --watch              Refresh continuously (Ctrl-C to stop)
    -n, --interval SECS      Refresh interval for --watch (default: 30, min: 5)
    -h, --help               Show this help

Useful when:
  - Your job is PENDING and you want to know why
  - You're deciding whether to submit a 2-node job vs 1-node
  - You suspect maintenance is causing scheduler slowness

Examples:
    ./olivia.sh cluster                       # one-shot snapshot
    ./olivia.sh cluster --watch               # refresh every 30s
    ./olivia.sh cluster --watch --interval 10 # refresh every 10s
EOF
}

# =============================================================================
# MODULE: Prefetch
# =============================================================================
# Download model weights into the persistent HF cache (HF_HOME) from a LOGIN
# node — never from a GPU/SLURM job. Why login-node + venv (not a container):
#   * Login nodes have internet egress and a modern system python3.12. The
#     compute nodes are GH200 (arm64) and their containers cannot exec on the
#     amd64 login node at all, so we build a tiny throwaway venv instead.
#   * HF_HOME points at persistent project storage (/cluster/projects/...), so
#     weights survive the /cluster/work auto-purge (21-42 days) and are fetched
#     once rather than re-downloaded per job.
# The transfer runs detached (setsid) and is resumable (hf download skips
# already-complete blobs), so closing the CLI never aborts a multi-100GB pull.

cmd_prefetch() {
    local target="" revision="" follow=true
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -r|--revision) revision="$2"; shift 2 ;;
            --no-follow)   follow=false; shift ;;
            -h|--help)
                echo "Usage: $(basename "$0") prefetch <preset|repo_id> [--revision REV] [--no-follow]" >&2
                echo "" >&2
                echo "Downloads model weights into HF_HOME on a login node (detached, resumable)." >&2
                echo "Examples:" >&2
                echo "    ./olivia.sh prefetch glm51_v19                 # preset -> default repo" >&2
                echo "    ./olivia.sh prefetch cyankiwi/GLM-5.1-AWQ-4bit # explicit repo id" >&2
                return 0 ;;
            -*) error "Unknown option: $1"; return 1 ;;
            *)  target="$1"; shift ;;
        esac
    done

    if [[ -z "$target" ]]; then
        error "No model specified"
        echo "    Usage: ./olivia.sh prefetch <preset|repo_id>" >&2
        return 1
    fi

    require_remote_config || return 1
    if [[ -z "${HF_HOME:-}" ]]; then
        error "HF_HOME is not set"
        echo "    Set HF_HOME to the persistent model cache on the cluster, e.g.:" >&2
        echo "    export HF_HOME=/cluster/projects/<proj>/huggingface" >&2
        return 1
    fi
    ensure_master_connection || return 1

    # Resolve a preset alias to its default repo id; otherwise treat the
    # argument as a literal HuggingFace repo id.
    local repo; repo=$(get_default_model "$target")
    [[ -z "$repo" ]] && repo="$target"

    local state venv slug log
    state="$(dirname "${HF_HOME}")/.prefetch"
    venv="${state}/venv"
    slug=$(printf '%s' "$repo" | tr '/:' '__')
    log="${state}/${slug}.log"

    info "Prefetching ${repo}"
    echo "    -> ${HF_HOME}  (login node, detached, resumable)" >&2

    # HF_TOKEN (secret, needed for gated repos) is forwarded over stdin so it
    # stays off argv and out of logs; the exported value is inherited by the
    # detached `hf download`. Public repos work with it unset.
    local token_read=""
    [[ -n "${HF_TOKEN:-}" ]] && token_read="IFS= read -r HF_TOKEN && export HF_TOKEN; "

    # All ${...} below are host-interpolated; \$ / \" are evaluated remotely.
    local remote_cmd="${token_read}
        set -e
        HF='${HF_HOME}'; STATE='${state}'; VENV='${venv}'; LOG='${log}'; REPO='${repo}'
        mkdir -p \"\$HF\" \"\$STATE\"
        if [ ! -x \"\$VENV/bin/hf\" ]; then
            echo 'Creating prefetch venv (python3.12)…'
            python3.12 -m venv \"\$VENV\"
            \"\$VENV/bin/pip\" -q install -U pip huggingface_hub hf_xet
        fi
        PIDF=\"\$STATE/${slug}.pid\"
        if [ -f \"\$PIDF\" ] && kill -0 \"\$(cat \"\$PIDF\")\" 2>/dev/null; then
            echo \"Already downloading (pid \$(cat \"\$PIDF\"))\"
        else
            HF_HOME=\"\$HF\" HF_XET_HIGH_PERFORMANCE=1 setsid nohup \\
                \"\$VENV/bin/hf\" download \"\$REPO\" ${revision:+--revision '${revision}'} \\
                >\"\$LOG\" 2>&1 </dev/null &
            echo \$! > \"\$PIDF\"
            echo \"Started pid \$(cat \"\$PIDF\")\"
        fi
    "
    if [[ -n "${HF_TOKEN:-}" ]]; then
        ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "${remote_cmd}" <<<"${HF_TOKEN}" \
            || { error "Failed to start prefetch"; return 1; }
    else
        ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "${remote_cmd}" \
            || { error "Failed to start prefetch"; return 1; }
    fi

    if [[ "$follow" != true ]]; then
        echo "    Watch: ssh ${REMOTE_HOST} \"tail -f ${log}\"" >&2
        return 0
    fi
    echo "    (Ctrl-C stops watching; the download continues in the background)" >&2
    echo "" >&2
    # hf writes \r-style progress bars; translate to lines so they stream.
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -F '${log}' 2>/dev/null | tr '\r' '\n'" || true
}

# =============================================================================
# MODULE: Reconnect
# =============================================================================

# Re-establish the 2FA'd master after a real drop (laptop sleep, network change,
# persist window expiry). Costs one OTP, then restores any recorded tunnel.
cmd_reconnect() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)             set_local_port "$2"; shift 2 ;;
            --login-proxy-port) LOGIN_PROXY_PORT="$2"; shift 2 ;;
            -h|--help)
                echo "Usage: ./olivia.sh reconnect [--port PORT]"
                echo "Re-establish the SSH master connection and restore the recorded tunnel."
                exit 0
                ;;
            *) error "Unknown option: $1"; exit 1 ;;
        esac
    done

    require_remote_config || exit 1

    info "Re-establishing SSH master connection..."
    close_master_connection
    ensure_master_connection || exit 1

    # Restore a previously-recorded tunnel, inferring the mode from its target.
    if [[ -f "${TUNNEL_TARGET_FILE}" || -f "${TUNNEL_NODE_FILE}" ]]; then
        local target
        target=$(cat "${TUNNEL_TARGET_FILE}" 2>/dev/null || echo "")
        if [[ "$target" == localhost:* ]]; then
            LOGIN_PROXY=1
            LOGIN_PROXY_PORT="${target#localhost:}"
        fi
        local gpu_node
        if gpu_node=$(find_vllm_node); then
            info "Restoring tunnel..."
            setup_tunnel "$gpu_node"
        else
            warn "No running vLLM job found; tunnel not restored"
        fi
    fi
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat <<EOF
Olivia HPC Cluster Management Tool v${OLIVIA_VERSION}

Usage: $(basename "$0") <command> [options]

Commands:
    chat        Connect to vLLM server and start chat (default)
    build       Build vLLM containers
    server      Manage vLLM server (deploy, restart, logs)
    prefetch    Download model weights into persistent HF_HOME (login node)
    tunnel      Manage SSH tunnel
    reconnect   Re-establish the SSH master after a drop and restore the tunnel
    status      Show cluster and connection status
    cluster     Show Olivia partition utilization (queue, GPUs, reservations)

Global Options:
    --kill-all          Close tunnel and SSH connection
    -h, --help          Show this help
    -v, --version       Show version

Connection durability (env overrides):
    SSH_CONTROL_PERSIST  How long the 2FA'd master persists when idle (default: ${SSH_CONTROL_PERSIST})
    LOGIN_PROXY=1        Route the tunnel via a fixed login-node relay that
                         follows the GPU node across job restarts (opt-in)

Examples:
    $(basename "$0") chat               Connect to vLLM and chat
    $(basename "$0") build glm47        Build GLM-4.7 container
    $(basename "$0") server restart     Restart vLLM server
    $(basename "$0") reconnect          Re-auth after a drop, restore tunnel
    $(basename "$0") status             Check status

Run '$(basename "$0") <command> --help' for command-specific help.
EOF
}

main() {
    # Handle global options
    case "${1:-}" in
        --kill-all)
            kill_tunnel 2>/dev/null || true
            close_master_connection
            success "All connections closed"
            exit 0
            ;;
        -v|--version)
            echo "olivia.sh v${OLIVIA_VERSION}"
            exit 0
            ;;
        -h|--help|"")
            usage
            exit 0
            ;;
    esac

    # Route to command
    local cmd="${1:-chat}"
    shift || true

    case "$cmd" in
        chat)
            cmd_chat "$@"
            ;;
        build)
            cmd_build "$@"
            ;;
        server)
            cmd_server "$@"
            ;;
        prefetch)
            cmd_prefetch "$@"
            ;;
        tunnel)
            cmd_tunnel "$@"
            ;;
        reconnect)
            cmd_reconnect "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        cluster)
            cmd_cluster "$@"
            ;;
        *)
            error "Unknown command: $cmd"
            usage
            exit 1
            ;;
    esac
}

main "$@"
