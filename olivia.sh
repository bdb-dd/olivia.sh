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
LOCAL_PORT="${LOCAL_PORT:-8000}"
JOB_NAME_PATTERN="vllm"

# Remote paths
REMOTE_WORKDIR="${REMOTE_WORKDIR:-}"
REMOTE_CONTAINER_DIR="${REMOTE_CONTAINER_DIR:-}"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SERVER_SCRIPT="${SCRIPT_DIR}/run_vllm_server.sh"
LOCAL_BUILD_SCRIPT="${SCRIPT_DIR}/build_vllm_gh200.sh"
CHAT_SCRIPT="${SCRIPT_DIR}/chat_devstral.py"

# SSH control socket
SSH_CONTROL_DIR="${HOME}/.ssh/controls"
SSH_CONTROL_SOCKET="${SSH_CONTROL_DIR}/olivia-${REMOTE_USER}@${REMOTE_HOST}"

# State files
TUNNEL_NODE_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.node"

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

# SSH helpers
ssh_opts() {
    echo -o "ControlMaster=auto" -o "ControlPath=${SSH_CONTROL_SOCKET}" -o "ControlPersist=600"
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
    mkdir -p "${SSH_CONTROL_DIR}"
    chmod 700 "${SSH_CONTROL_DIR}"

    if is_master_alive; then
        return 0
    fi

    info "Establishing SSH connection to ${REMOTE_HOST}..."
    echo "    (You may be prompted for 2FA)" >&2
    echo "" >&2

    ssh -f -N -M \
        -o "ControlPath=${SSH_CONTROL_SOCKET}" \
        -o "ControlPersist=600" \
        -o "ServerAliveInterval=60" \
        -o "ServerAliveCountMax=3" \
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
    if [[ -f "${TUNNEL_NODE_FILE}" ]]; then
        local node
        node=$(cat "${TUNNEL_NODE_FILE}")
        info "Canceling port forward to ${node}..."

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

    echo "$job_line" | awk '{print $2}'
}

setup_tunnel() {
    local gpu_node="$1"

    if is_tunnel_alive; then
        local existing_node
        existing_node=$(get_tunnel_node)
        if [[ "$existing_node" == "$gpu_node" ]]; then
            success "Tunnel already active to ${gpu_node} on port ${LOCAL_PORT}"
            return 0
        else
            warn "Tunnel to ${existing_node}, but job is now on ${gpu_node}"
            kill_tunnel
        fi
    fi

    info "Setting up tunnel: localhost:${LOCAL_PORT} -> ${gpu_node}:${REMOTE_PORT}"

    if ! ssh -O forward \
        -L "${LOCAL_PORT}:${gpu_node}:${REMOTE_PORT}" \
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
    success "Tunnel established"
    echo "    Local:  localhost:${LOCAL_PORT}" >&2
    echo "    Remote: ${gpu_node}:${REMOTE_PORT}" >&2
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

    while ! ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "test -f '${log_file}'" 2>/dev/null; do
        if [[ -n "$job_id" ]]; then
            local job_state
            job_state=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "squeue -j ${job_id} -h -o '%T'" 2>/dev/null) || true

            if [[ -z "$job_state" ]]; then
                error "Job ${job_id} no longer in queue - may have failed"
                return 1
            fi

            printf "\r    Job state: %-10s (waiting for log file...)" "$job_state" >&2
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

    echo "" >&2
    success "Job started - tailing logs"
    echo "    Log file: ${log_file}" >&2
    echo "" >&2
    echo "==========================================="

    ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${log_file}'" || true

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

    info "Uploading build_vllm_gh200.sh to ${REMOTE_HOST}:${REMOTE_CONTAINER_DIR}/"

    if ! scp_run "${LOCAL_BUILD_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CONTAINER_DIR}/build_vllm_gh200.sh"; then
        error "Failed to upload build script"
        return 1
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

cmd_build() {
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
                echo "Available presets: glm47, devstral, llama, qwen, generic" >&2
                exit 0
                ;;
            --presets|-p)
                header "Available Model Presets"
                echo "  glm47      GLM-4.7 (358B) flagship model"
                echo "             vLLM: main, transformers>=5.0.0rc0"
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

    # Build environment variables
    local env_vars="MODEL_ID=${model_id}"
    [[ -n "$build_index" ]] && env_vars="${env_vars} BUILD_INDEX=${build_index}"
    [[ -n "$vllm_version" ]] && env_vars="${env_vars} VLLM_VERSION=${vllm_version}"
    [[ -n "$create_sif" ]] && env_vars="${env_vars} CREATE_SIF=${create_sif}"
    [[ -n "$force_overwrite" ]] && env_vars="${env_vars} OVERWRITE=${force_overwrite}"

    info "Submitting build job for '${model_id}'..."
    echo "    Environment: ${env_vars}" >&2

    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_CONTAINER_DIR} && ${env_vars} sbatch build_vllm_gh200.sh" 2>&1); then
        error "Failed to submit build job"
        echo "$submit_output" >&2
        return 1
    fi

    local job_id
    job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)
    success "Submitted build job ${job_id}"

    # Determine expected sandbox name
    local idx="${build_index:-1}"
    local sandbox_name="vllm-${model_id}-${idx}-sandbox"
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
Usage: ./olivia.sh build [preset] [options]

Build vLLM containers on Olivia HPC cluster.

Arguments:
    preset              Model preset (glm47, devstral, llama, qwen, generic)

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
EOF
}

# =============================================================================
# MODULE: Server
# =============================================================================

deploy_server_script() {
    if [[ ! -f "${LOCAL_SERVER_SCRIPT}" ]]; then
        error "Server script not found: ${LOCAL_SERVER_SCRIPT}"
        return 1
    fi

    info "Uploading run_vllm_server.sh to ${REMOTE_HOST}:${REMOTE_WORKDIR}/"

    if ! scp_run "${LOCAL_SERVER_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORKDIR}/run_vllm_server.sh"; then
        error "Failed to upload server script"
        return 1
    fi

    success "Server script deployed"
    return 0
}

start_server_job() {
    info "Starting vLLM server job..."

    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_WORKDIR} && sbatch run_vllm_server.sh" 2>&1); then
        error "Failed to submit job"
        echo "$submit_output" >&2
        return 1
    fi

    local job_id
    job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)
    success "Submitted job ${job_id}"
    echo "$job_id"
}

cmd_server() {
    local action=""
    local do_deploy=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            restart)
                action="restart"
                shift
                ;;
            deploy)
                action="deploy"
                shift
                ;;
            logs)
                action="logs"
                shift
                ;;
            cancel|stop)
                action="cancel"
                shift
                ;;
            --deploy|-d)
                do_deploy=true
                shift
                ;;
            -h|--help)
                cmd_server_help
                exit 0
                ;;
            *)
                error "Unknown argument: $1"
                cmd_server_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$action" && ! $do_deploy ]]; then
        cmd_server_help
        exit 0
    fi

    ensure_master_connection || exit 1

    case "$action" in
        deploy)
            deploy_server_script
            ;;
        cancel)
            cancel_job "vllm"
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            ;;
        restart)
            if $do_deploy; then
                deploy_server_script || exit 1
                echo ""
            fi
            cancel_job "vllm"
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            echo ""
            local job_id
            if ! job_id=$(start_server_job); then
                exit 1
            fi
            local log_file="${REMOTE_WORKDIR}/logs/vllm_server_${job_id}.log"
            tail_job_logs "$log_file" "$job_id"
            echo ""
            info "Reconnect with: ./olivia.sh chat"
            ;;
        logs)
            local job_id
            if job_id=$(find_job_id "vllm"); then
                local log_file="${REMOTE_WORKDIR}/logs/vllm_server_${job_id}.log"
                info "Tailing logs for job ${job_id}..."
                ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${log_file}'" || true
            else
                error "No running vLLM server found"
                exit 1
            fi
            ;;
    esac
}

cmd_server_help() {
    cat <<EOF
Usage: ./olivia.sh server <action> [options]

Manage vLLM server on Olivia.

Actions:
    deploy              Upload run_vllm_server.sh to cluster
    restart             Cancel running job and start new one
    cancel, stop        Cancel running vLLM job
    logs                Tail logs of running server

Options:
    --deploy, -d        Deploy script before action
    -h, --help          Show this help

Examples:
    ./olivia.sh server deploy           Upload server script
    ./olivia.sh server restart          Restart server
    ./olivia.sh server restart -d       Deploy and restart
    ./olivia.sh server logs             Tail server logs
    ./olivia.sh server cancel           Stop running server
EOF
}

# =============================================================================
# MODULE: Tunnel
# =============================================================================

cmd_tunnel() {
    local action="${1:-status}"

    case "$action" in
        up|open|start)
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
            if is_tunnel_alive; then
                local node
                node=$(get_tunnel_node)
                success "Tunnel ACTIVE"
                echo "    Node:   $node"
                echo "    Local:  localhost:${LOCAL_PORT}"
                echo "    Remote: ${node}:${REMOTE_PORT}"
            else
                warn "Tunnel NOT ACTIVE"
            fi
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
Usage: ./olivia.sh tunnel <action>

Manage SSH tunnel to vLLM server.

Actions:
    up, open, start     Open tunnel to running vLLM server
    down, close, kill   Close tunnel
    status              Show tunnel status (default)

Examples:
    ./olivia.sh tunnel up       Open tunnel
    ./olivia.sh tunnel down     Close tunnel
    ./olivia.sh tunnel status   Check tunnel status
EOF
}

# =============================================================================
# MODULE: Chat
# =============================================================================

cmd_chat() {
    local stream_flag="--stream"
    local tunnel_only=false

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
                LOCAL_PORT="$2"
                TUNNEL_NODE_FILE="/tmp/olivia-tunnel-${LOCAL_PORT}.node"
                shift 2
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

    # Check/setup tunnel
    local gpu_node
    local need_new_tunnel=true

    if is_tunnel_alive; then
        local existing_node
        existing_node=$(get_tunnel_node)
        info "Checking if vLLM job is still on ${existing_node}..."
        if gpu_node=$(find_vllm_node 2>/dev/null) && [[ "$gpu_node" == "$existing_node" ]]; then
            success "Tunnel valid for ${gpu_node}"
            need_new_tunnel=false
        elif [[ -n "${gpu_node:-}" ]]; then
            warn "Job moved from ${existing_node} to ${gpu_node}"
        fi
    fi

    if $need_new_tunnel; then
        if ! gpu_node=$(find_vllm_node); then
            error "No running vLLM job found"
            echo "" >&2
            info "Current jobs:"
            ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER" 2>&1 || true
            echo "" >&2
            info "Start a server with: ./olivia.sh server restart"
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

    python3 "${CHAT_SCRIPT}" localhost --port "${LOCAL_PORT}" ${stream_flag}

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
    --port PORT         Local port (default: 8000)
    --tunnel-only       Only set up tunnel, don't start chat
    --no-stream         Disable streaming in chat
    -h, --help          Show this help

Examples:
    ./olivia.sh chat                Connect and start chat
    ./olivia.sh chat --port 9000    Use different port
    ./olivia.sh chat --tunnel-only  Just set up tunnel
EOF
}

# =============================================================================
# MODULE: Status
# =============================================================================

cmd_status() {
    header "Olivia Status"

    # SSH connection
    if is_master_alive; then
        success "SSH Connection: ACTIVE"
    else
        warn "SSH Connection: NOT ACTIVE"
        echo "    Run any command to establish connection"
    fi

    echo ""

    # Tunnel
    if is_tunnel_alive; then
        local node
        node=$(get_tunnel_node)
        success "SSH Tunnel: ACTIVE"
        echo "    Node:   $node"
        echo "    Local:  localhost:${LOCAL_PORT}"
    else
        warn "SSH Tunnel: NOT ACTIVE"
    fi

    echo ""

    # Remote jobs (if connected)
    if is_master_alive || ensure_master_connection 2>/dev/null; then
        info "Jobs on Olivia:"
        ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u \$USER" 2>/dev/null || true
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
    tunnel      Manage SSH tunnel
    status      Show cluster and connection status

Global Options:
    --kill-all          Close tunnel and SSH connection
    -h, --help          Show this help
    -v, --version       Show version

Examples:
    $(basename "$0") chat               Connect to vLLM and chat
    $(basename "$0") build glm47        Build GLM-4.7 container
    $(basename "$0") server restart     Restart vLLM server
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
        tunnel)
            cmd_tunnel "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        *)
            error "Unknown command: $cmd"
            usage
            exit 1
            ;;
    esac
}

main "$@"
