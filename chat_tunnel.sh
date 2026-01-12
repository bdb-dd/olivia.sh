#!/bin/bash
# =============================================================================
# chat_tunnel.sh - Dynamic SSH Tunnel for vLLM Chat
# =============================================================================
# Automatically detects the GPU node running vLLM and sets up an SSH tunnel.
#
# Usage:
#   ./chat_tunnel.sh              # Auto-detect node, use port 8000
#   ./chat_tunnel.sh --port 9000  # Use different local port
#   ./chat_tunnel.sh --tunnel-only  # Just set up tunnel, don't start chat
#   ./chat_tunnel.sh --kill       # Kill existing tunnel
#   ./chat_tunnel.sh --status     # Check tunnel and job status
#   ./chat_tunnel.sh --deploy     # Upload latest run_vllm_server.sh to cluster
#   ./chat_tunnel.sh --restart    # Cancel and restart the vLLM job
#   ./chat_tunnel.sh --deploy-restart  # Deploy and restart in one step
#
# The tunnel stays open after the chat exits. Use --kill to close it.
# Only ONE 2FA prompt per session - uses SSH ControlMaster.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT=8000
LOCAL_PORT="${LOCAL_PORT:-8000}"
JOB_NAME_PATTERN="vllm"  # Matches job names containing "vllm"

# Remote working directory for vLLM
REMOTE_WORKDIR="${REMOTE_WORKDIR:-}"

# Local script to deploy
LOCAL_SERVER_SCRIPT="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/run_vllm_server.sh"

# SSH control socket for connection reuse (single socket for all operations)
SSH_CONTROL_DIR="${HOME}/.ssh/controls"
SSH_CONTROL_SOCKET="${SSH_CONTROL_DIR}/vllm-${REMOTE_USER}@${REMOTE_HOST}"

# File to track the tunnel target node
TUNNEL_NODE_FILE="/tmp/vllm-tunnel-${LOCAL_PORT}.node"

# Chat script location (same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_SCRIPT="${SCRIPT_DIR}/chat_devstral.py"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# All logging functions print to stderr so stdout is reserved for return values
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

# Common SSH options for ControlMaster
ssh_opts() {
    echo -o "ControlMaster=auto" -o "ControlPath=${SSH_CONTROL_SOCKET}" -o "ControlPersist=600"
}

# Run SSH command using the control socket
ssh_run() {
    ssh $(ssh_opts) "$@"
}

# -----------------------------------------------------------------------------
# SSH ControlMaster management
# -----------------------------------------------------------------------------

# Check if master connection is already established
is_master_alive() {
    ssh -o "ControlPath=${SSH_CONTROL_SOCKET}" -O check "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null
}

# Establish master connection (this is where 2FA happens - only once!)
ensure_master_connection() {
    mkdir -p "${SSH_CONTROL_DIR}"
    chmod 700 "${SSH_CONTROL_DIR}"

    if is_master_alive; then
        return 0
    fi

    info "Establishing SSH connection to ${REMOTE_HOST}..."
    echo "    (You may be prompted for 2FA)"
    echo ""

    # Start master connection in background with a dummy command
    # -f: background after auth, -N: no command, -M: master mode
    ssh -f -N -M \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        -o "ControlPersist=600" \
        -o "ServerAliveInterval=60" \
        -o "ServerAliveCountMax=3" \
        "${REMOTE_USER}@${REMOTE_HOST}"

    # Verify it's up
    sleep 1
    if is_master_alive; then
        success "SSH connection established (ControlMaster active)"
        return 0
    else
        error "Failed to establish SSH master connection"
        return 1
    fi
}

# Close master connection
close_master_connection() {
    if is_master_alive; then
        ssh -o "ControlPath=${SSH_CONTROL_SOCKET}" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true
    fi
}

# -----------------------------------------------------------------------------
# Tunnel management functions
# -----------------------------------------------------------------------------

get_tunnel_node() {
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        cat "${TUNNEL_NODE_FILE}"
    fi
}

is_tunnel_alive() {
    # Check if the port is listening AND we have a record of the tunnel
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        if lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

kill_tunnel() {
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        local node
        node=$(cat "${TUNNEL_NODE_FILE}")
        info "Canceling port forward to ${node}..."

        # Cancel the port forward through the ControlMaster
        ssh -O cancel \
            -L "${LOCAL_PORT}:${node}:${REMOTE_PORT}" \
            -o "ControlPath=${SSH_CONTROL_SOCKET}" \
            "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true

        rm -f "${TUNNEL_NODE_FILE}"
        success "Tunnel closed"
        return 0
    else
        warn "No active tunnel found"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Deploy and restart functions
# -----------------------------------------------------------------------------

deploy_server_script() {
    if [[ ! -f "${LOCAL_SERVER_SCRIPT}" ]]; then
        error "Server script not found: ${LOCAL_SERVER_SCRIPT}"
        return 1
    fi

    info "Uploading run_vllm_server.sh to ${REMOTE_HOST}:${REMOTE_WORKDIR}/"

    # Use scp with ControlMaster for efficient transfer
    if ! scp -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        "${LOCAL_SERVER_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORKDIR}/run_vllm_server.sh"; then
        error "Failed to upload server script"
        return 1
    fi

    success "Server script deployed successfully"
    return 0
}

find_vllm_job_id() {
    # Find the job ID of the running vLLM job
    local squeue_output
    if ! squeue_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER -h -o '%j %i %T'" 2>/dev/null); then
        error "Failed to run squeue on ${REMOTE_HOST}"
        return 1
    fi

    # Find running vLLM job and get its ID
    local job_line
    job_line=$(echo "$squeue_output" | grep -i "${JOB_NAME_PATTERN}" | grep "RUNNING" | head -n1) || true

    if [[ -z "$job_line" ]]; then
        return 1  # No running job
    fi

    # Extract job ID (second field)
    echo "$job_line" | awk '{print $2}'
}

cancel_vllm_job() {
    info "Looking for running vLLM job..."

    local job_id
    if ! job_id=$(find_vllm_job_id); then
        warn "No running vLLM job found"
        return 0  # Not an error - job might not exist
    fi

    info "Canceling job ${job_id}..."
    if ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "scancel ${job_id}"; then
        error "Failed to cancel job ${job_id}"
        return 1
    fi

    success "Job ${job_id} canceled"

    # Kill the tunnel since the node will change
    if is_tunnel_alive; then
        info "Closing tunnel (node will change after restart)..."
        kill_tunnel
    fi

    return 0
}

start_vllm_job() {
    info "Starting new vLLM server job..."

    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_WORKDIR} && sbatch run_vllm_server.sh" 2>&1); then
        error "Failed to submit job"
        echo "$submit_output" >&2
        return 1
    fi

    # Extract job ID from "Submitted batch job 12345"
    local new_job_id
    new_job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)

    success "Submitted job ${new_job_id}"

    # Store job ID for log tailing
    echo "$new_job_id"
}

tail_job_logs() {
    local job_id="$1"
    local log_file="${REMOTE_WORKDIR}/logs/vllm_server_${job_id}.log"

    echo ""
    info "Waiting for job ${job_id} to start..."
    echo "    (Press Ctrl+C to stop watching - job will continue running)"
    echo ""

    # Wait for log file to appear (job to start), with timeout
    local wait_count=0
    local max_wait=60  # 60 seconds max wait

    while ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -f '${log_file}'" 2>/dev/null; do
        # Check job status
        local job_state
        job_state=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
            "squeue -j ${job_id} -h -o '%T'" 2>/dev/null) || true

        if [[ -z "$job_state" ]]; then
            error "Job ${job_id} no longer in queue - may have failed"
            return 1
        fi

        printf "\r    Job state: %-10s (waiting for log file...)" "$job_state"

        sleep 2
        ((wait_count+=2))

        if [[ $wait_count -ge $max_wait ]]; then
            echo ""
            warn "Timeout waiting for log file"
            echo "    Check manually: ssh ${REMOTE_HOST} 'tail -f ${log_file}'"
            return 0
        fi
    done

    echo ""
    success "Job started - tailing logs"
    echo "    Log file: ${log_file}"
    echo ""
    echo "==========================================="

    # Tail the log file (user can Ctrl+C to exit)
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${log_file}'" || true

    echo ""
    info "Log tailing stopped. Job continues running."
    echo "    Reconnect with: ./chat_tunnel.sh"
}

restart_server() {
    cancel_vllm_job
    echo ""

    local job_id
    if ! job_id=$(start_vllm_job); then
        return 1
    fi

    tail_job_logs "$job_id"
}

# -----------------------------------------------------------------------------
# Remote job detection
# -----------------------------------------------------------------------------

find_vllm_node() {
    info "Checking for vLLM job on ${REMOTE_HOST}..."

    # Run squeue on remote host (uses ControlMaster - no 2FA needed)
    local squeue_output
    if ! squeue_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER -h -o '%j %N %T'" 2>/dev/null); then
        error "Failed to run squeue on ${REMOTE_HOST}"
        return 1
    fi

    # Find running vLLM job
    local job_line
    job_line=$(echo "$squeue_output" | grep -i "${JOB_NAME_PATTERN}" | grep "RUNNING" | head -n1) || true

    if [[ -z "$job_line" ]]; then
        error "No running vLLM job found"
        echo "" >&2
        echo "Current jobs:" >&2
        ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER" >&2 2>&1 || true
        echo "" >&2
        echo "Start a vLLM server with: sbatch run_vllm_server.sh" >&2
        return 1
    fi

    # Extract node name (second field)
    local node_name
    node_name=$(echo "$job_line" | awk '{print $2}')

    if [[ -z "$node_name" ]]; then
        error "Could not parse node name from: $job_line"
        return 1
    fi

    echo "$node_name"
}

# -----------------------------------------------------------------------------
# Tunnel setup
# -----------------------------------------------------------------------------

setup_tunnel() {
    local gpu_node="$1"

    # Check if tunnel already exists to this node
    if is_tunnel_alive; then
        local existing_node
        existing_node=$(get_tunnel_node)
        if [[ "$existing_node" == "$gpu_node" ]]; then
            success "Tunnel already active to ${gpu_node} on port ${LOCAL_PORT}"
            return 0
        else
            warn "Existing tunnel to ${existing_node}, but job is now on ${gpu_node}"
            info "Closing old tunnel..."
            kill_tunnel
        fi
    fi

    info "Setting up SSH tunnel: localhost:${LOCAL_PORT} -> ${gpu_node}:${REMOTE_PORT}"

    # Request port forwarding through the existing ControlMaster (no 2FA needed)
    # -O forward: request the master to add a port forward
    if ! ssh -O forward \
        -L "${LOCAL_PORT}:${gpu_node}:${REMOTE_PORT}" \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        "${REMOTE_USER}@${REMOTE_HOST}" 2>&1; then
        error "Failed to set up port forwarding"
        return 1
    fi

    # Verify the port is listening
    sleep 1
    if ! lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        error "Tunnel setup failed - port ${LOCAL_PORT} not listening"
        return 1
    fi

    # Save tunnel info (just the node, master handles the connection)
    echo "$gpu_node" > "${TUNNEL_NODE_FILE}"

    success "Tunnel established"
    echo "    Local:  localhost:${LOCAL_PORT}"
    echo "    Remote: ${gpu_node}:${REMOTE_PORT}"
}

# -----------------------------------------------------------------------------
# Status display
# -----------------------------------------------------------------------------

show_status() {
    echo "=== vLLM Tunnel Status ==="
    echo ""

    # Check SSH master connection
    if is_master_alive; then
        success "SSH ControlMaster: ACTIVE"
    else
        warn "SSH ControlMaster: NOT ACTIVE"
    fi

    # Check local tunnel
    if is_tunnel_alive; then
        local node
        node=$(get_tunnel_node)
        success "Tunnel: ACTIVE"
        echo "    Node:   $node"
        echo "    Local:  localhost:${LOCAL_PORT}"
        echo "    Remote: ${node}:${REMOTE_PORT}"
    else
        warn "Tunnel: NOT ACTIVE"
    fi

    echo ""

    # Check remote job (establish master if needed)
    if ! is_master_alive; then
        ensure_master_connection || return 1
    fi
    info "Remote job status:"
    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Automatically set up an SSH tunnel to the vLLM server and start the chat client.
Only ONE 2FA prompt per session - uses SSH ControlMaster for connection reuse.

Options:
    --port PORT      Local port to use (default: 8000)
    --tunnel-only    Only set up the tunnel, don't start chat
    --kill           Kill the existing tunnel (keeps SSH master alive)
    --kill-all       Kill tunnel AND close SSH master connection
    --status         Show tunnel and job status
    --no-stream      Disable streaming in chat client
    --deploy         Upload latest run_vllm_server.sh to cluster
    --restart        Cancel running vLLM job and start a new one
    --deploy-restart Deploy script and restart server (combines --deploy --restart)
    -h, --help       Show this help message

Environment variables:
    LOCAL_PORT       Same as --port

Examples:
    $(basename "$0")              # Connect to vLLM and start chat
    $(basename "$0") --port 9000  # Use local port 9000
    $(basename "$0") --kill       # Close the tunnel
    $(basename "$0") --kill-all   # Close tunnel + SSH connection
    $(basename "$0") --deploy     # Upload latest server script
    $(basename "$0") --restart    # Cancel and restart vLLM job
    $(basename "$0") --deploy-restart  # Deploy and restart in one step
EOF
}

main() {
    local tunnel_only=false
    local stream_flag="--stream"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)
                LOCAL_PORT="$2"
                TUNNEL_NODE_FILE="/tmp/vllm-tunnel-${LOCAL_PORT}.node"
                shift 2
                ;;
            --tunnel-only)
                tunnel_only=true
                shift
                ;;
            --kill)
                kill_tunnel
                exit $?
                ;;
            --kill-all)
                kill_tunnel
                close_master_connection
                success "All connections closed"
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            --deploy)
                ensure_master_connection || exit 1
                deploy_server_script
                exit $?
                ;;
            --restart)
                ensure_master_connection || exit 1
                restart_server
                exit $?
                ;;
            --deploy-restart)
                ensure_master_connection || exit 1
                deploy_server_script || exit 1
                echo ""
                restart_server
                exit $?
                ;;
            --no-stream)
                stream_flag=""
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Clean up stale PID file from old script version
    rm -f "/tmp/vllm-tunnel-${LOCAL_PORT}.pid" 2>/dev/null || true

    # Establish SSH master connection FIRST (only 2FA prompt happens here)
    if ! ensure_master_connection; then
        exit 1
    fi

    # Check if tunnel is already up and connected to correct node
    local gpu_node
    local need_new_tunnel=true

    if is_tunnel_alive; then
        local existing_node
        existing_node=$(get_tunnel_node)

        # Verify the job is still on the same node (uses master connection - no 2FA)
        info "Checking if vLLM job is still on ${existing_node}..."
        if gpu_node=$(find_vllm_node 2>/dev/null) && [[ "$gpu_node" == "$existing_node" ]]; then
            success "Tunnel still valid for ${gpu_node}"
            need_new_tunnel=false
        else
            if [[ -n "${gpu_node:-}" ]]; then
                warn "Job moved from ${existing_node} to ${gpu_node}"
            fi
        fi
    fi

    # Find node and setup tunnel if needed (all use master connection - no 2FA)
    if $need_new_tunnel; then
        if ! gpu_node=$(find_vllm_node); then
            exit 1
        fi

        if ! setup_tunnel "$gpu_node"; then
            exit 1
        fi
    fi

    # Exit here if tunnel-only mode
    if $tunnel_only; then
        echo ""
        success "Tunnel ready. Connect with:"
        echo "    python chat_devstral.py localhost --port ${LOCAL_PORT} --stream"
        echo ""
        echo "To close the tunnel later:"
        echo "    ./chat_tunnel.sh --kill"
        exit 0
    fi

    # Start chat client
    echo ""
    info "Starting chat client..."
    echo ""

    if [[ ! -f "${CHAT_SCRIPT}" ]]; then
        error "Chat script not found: ${CHAT_SCRIPT}"
        exit 1
    fi

    # Run the chat script
    python3 "${CHAT_SCRIPT}" localhost --port "${LOCAL_PORT}" ${stream_flag}

    echo ""
    info "Chat session ended. Tunnel remains open on port ${LOCAL_PORT}"
    echo "    Reconnect: ./chat_tunnel.sh"
    echo "    Close:     ./chat_tunnel.sh --kill"
}

main "$@"
