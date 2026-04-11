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

# Capture git provenance for the current worktree.
# Output is a single line, e.g.:
#   branch=add-glm-5.1 commit=62cd320 dirty=3
#   branch=main commit=abc1234 clean
#   no-git  (if SCRIPT_DIR isn't a git checkout)
#
# This is used at deploy time (for visibility) and at sbatch submission time
# (via --comment=, so the provenance lives in SLURM metadata). Uncommitted
# testing is explicitly supported — this is audit, not a gate.
get_git_context() {
    local dir="${1:-${SCRIPT_DIR}}"
    (
        cd "$dir" 2>/dev/null || { echo "no-git"; exit 0; }
        if ! git rev-parse --git-dir >/dev/null 2>&1; then
            echo "no-git"
            exit 0
        fi
        local branch commit dirty
        branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
        commit=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
        dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$dirty" == "0" ]]; then
            echo "branch=${branch} commit=${commit} clean"
        else
            echo "branch=${branch} commit=${commit} dirty=${dirty}"
        fi
    )
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
    require_remote_config || return 1
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

    # Anchor to the start of the job-name field ($1) so "vllm" doesn't
    # accidentally match "build-vllm-gh200" (substring match was the bug).
    local job_line
    job_line=$(echo "$squeue_output" | awk -v pat="${JOB_NAME_PATTERN}" \
        'tolower($1) ~ ("^" tolower(pat)) && $NF == "RUNNING" {print; exit}') || true

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

    # Anchor to the start of the job-name field ($1) so pattern "vllm"
    # matches "vllm-server" but not "build-vllm-gh200".
    local job_line
    job_line=$(echo "$squeue_output" | awk -v pat="$pattern" \
        'tolower($1) ~ ("^" tolower(pat)) && $NF == "RUNNING" {print; exit}') || true

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
    echo "    Source:  ${SCRIPT_DIR}" >&2
    echo "    Git:     $(get_git_context)" >&2

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
                echo "Available presets: glm47, gemma4, devstral, llama, qwen, generic" >&2
                exit 0
                ;;
            --presets|-p)
                header "Available Model Presets"
                echo "  glm47      GLM-4.7 (358B) flagship model"
                echo "             vLLM: main, transformers>=5.0.0rc0"
                echo ""
                echo "  gemma4     Gemma 4 31B (dense, multimodal text+image)"
                echo "             vLLM: v0.19.0, transformers>=5.5.0"
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
    # CONTAINER_DIR must be forwarded explicitly — sbatch does not inherit it
    # from our `cd` alone (see commit 62cd320 which made it a hard requirement).
    local env_vars="CONTAINER_DIR=${REMOTE_CONTAINER_DIR} MODEL_ID=${model_id}"
    [[ -n "$build_index" ]] && env_vars="${env_vars} BUILD_INDEX=${build_index}"
    [[ -n "$vllm_version" ]] && env_vars="${env_vars} VLLM_VERSION=${vllm_version}"
    [[ -n "$create_sif" ]] && env_vars="${env_vars} CREATE_SIF=${create_sif}"
    [[ -n "$force_overwrite" ]] && env_vars="${env_vars} OVERWRITE=${force_overwrite}"

    # Distinct job name so concurrent builds are distinguishable in squeue
    # and so `cancel_job` / `find_job_id` can target a specific build.
    # Format: build-vllm-<preset>-<index>  (e.g. build-vllm-gemma4-1)
    # The "build-" prefix keeps these from matching JOB_NAME_PATTERN="vllm"
    # (which is anchored to ^vllm), so server operations never touch builds.
    local idx="${build_index:-1}"
    local job_name="build-vllm-${model_id}-${idx}"

    # Git provenance → sbatch --comment, so `scontrol show job` / `sacct -o Comment`
    # can tell you which worktree / commit a given job came from.
    local git_ctx
    git_ctx=$(get_git_context)

    info "Submitting build job '${job_name}' for '${model_id}'..."
    echo "    Environment: ${env_vars}" >&2
    echo "    Git:         ${git_ctx}" >&2

    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_CONTAINER_DIR} && ${env_vars} sbatch --job-name=${job_name} --comment='${git_ctx}' build_vllm_gh200.sh" 2>&1); then
        error "Failed to submit build job"
        echo "$submit_output" >&2
        return 1
    fi

    local job_id
    job_id=$(echo "$submit_output" | grep -oE '[0-9]+' | tail -1)
    success "Submitted build job ${job_id}"

    # Expected sandbox name (idx was set above when computing job_name)
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
    preset              Model preset (glm47, gemma4, devstral, llama, qwen, generic)

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

# Default model mappings for presets
get_default_model() {
    local preset="$1"
    case "${preset}" in
        glm47|glm-4.7)
            echo "QuantTrio/GLM-4.7-AWQ"
            ;;
        gemma4|gemma-4)
            echo "QuantTrio/gemma-4-31B-it-AWQ"
            ;;
        devstral|mistral)
            echo "mistralai/Devstral-2-123B-Instruct-2512"
            ;;
        llama|llama3)
            echo "meta-llama/Llama-3.3-70B-Instruct"
            ;;
        qwen|qwen2)
            echo "Qwen/Qwen2.5-72B-Instruct"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Default GPU count per preset (matches TP_SIZE defaults in run_vllm_server.sh).
# sbatch CLI flags override the #SBATCH --gpus=4 directive baked into the script.
get_default_gpus() {
    local preset="$1"
    case "${preset}" in
        gemma4|gemma-4)
            echo "2"
            ;;
        *)
            echo "4"
            ;;
    esac
}

# Resolve preset to container name
resolve_container_name() {
    local preset="$1"
    local index="${2:-1}"
    echo "vllm-${preset}-${index}-sandbox"
}

deploy_server_script() {
    if [[ ! -f "${LOCAL_SERVER_SCRIPT}" ]]; then
        error "Server script not found: ${LOCAL_SERVER_SCRIPT}"
        return 1
    fi

    info "Uploading run_vllm_server.sh to ${REMOTE_HOST}:${REMOTE_CONTAINER_DIR}/"
    echo "    Source:  ${SCRIPT_DIR}" >&2
    echo "    Git:     $(get_git_context)" >&2

    if ! scp_run "${LOCAL_SERVER_SCRIPT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CONTAINER_DIR}/run_vllm_server.sh"; then
        error "Failed to upload server script"
        return 1
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
    local gpus="${3:-4}"
    local cpus=$((gpus * 8))

    # Derive a distinct, identifiable job name from the container.
    #   vllm-gemma4-1-sandbox → vllm-gemma4-1
    #   vllm-glm47-2.sif      → vllm-glm47-2
    # Falls back to prefixing with "vllm-" if the container name doesn't already
    # start with it (custom containers via `-c`). Must start with "vllm" to match
    # the anchored JOB_NAME_PATTERN used by find_job_id / cancel_job.
    local job_name="$(basename "$container")"
    job_name="${job_name%-sandbox}"
    job_name="${job_name%.sif}"
    [[ "$job_name" == vllm-* ]] || job_name="vllm-${job_name}"

    # Git provenance → sbatch --comment for auditability across worktrees
    local git_ctx
    git_ctx=$(get_git_context)

    info "Starting vLLM server..."
    echo "    Container: ${container}" >&2
    echo "    Model:     ${model}" >&2
    echo "    GPUs:      ${gpus} (CPUs: ${cpus})" >&2
    echo "    Job name:  ${job_name}" >&2
    echo "    Git:       ${git_ctx}" >&2

    # CONTAINER_DIR must be forwarded explicitly (see commit 62cd320)
    local env_vars="CONTAINER_DIR=${REMOTE_CONTAINER_DIR} CONTAINER=${container} MODEL=${model}"

    # sbatch CLI flags override the #SBATCH directives baked into run_vllm_server.sh
    local submit_output
    if ! submit_output=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd ${REMOTE_CONTAINER_DIR} && ${env_vars} sbatch --job-name=${job_name} --comment='${git_ctx}' --gpus=${gpus} --cpus-per-task=${cpus} run_vllm_server.sh" 2>&1); then
        error "Failed to submit job"
        echo "$submit_output" >&2
        return 1
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
    local node_name=""
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
                state=$(echo "$job_info" | awk '{print $4}' | head -1)
                job_id=$(echo "$job_info" | awk '{print $2}' | head -1)
                node_name=$(echo "$job_info" | awk '{print $3}' | head -1)

                # Notify on state transitions
                if [[ "$state" == "RUNNING" && "$prev_state" == "PENDING" ]]; then
                    notify "Olivia vLLM" "Job ${job_id} is now RUNNING on ${node_name}" "Glass"
                fi

                if [[ "$state" == "RUNNING" && -n "$node_name" ]]; then
                    echo ""
                    success "Job ${job_id} running on ${node_name}"
                    log_file="${REMOTE_CONTAINER_DIR}/logs/vllm_server_${job_id}.log"
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

            # Get GPU memory usage from node
            local gpu_info
            gpu_info=$(ssh_run "${REMOTE_USER}@${REMOTE_HOST}" \
                "ssh ${node_name} 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null" 2>/dev/null) || true

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
    local index="1"
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
            cancel_job "vllm"
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            ;;
        start)
            # Resolve container name
            if [[ -n "$preset" ]]; then
                container=$(resolve_container_name "$preset" "$index")
                if [[ -z "$model" ]]; then
                    model=$(get_default_model "$preset")
                fi
            fi

            if [[ -z "$container" ]]; then
                error "No container specified"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server start <preset>              # e.g., glm47, gemma4, devstral" >&2
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
            if find_job_id "vllm" >/dev/null 2>&1; then
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

            # Determine GPU count for this preset (Gemma 4 uses 2, others use 4)
            local gpus
            gpus=$(get_default_gpus "$preset")

            # Start server
            local job_id
            if ! job_id=$(start_server_job "$container" "$model" "$gpus"); then
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
            if [[ -n "$preset" ]]; then
                container=$(resolve_container_name "$preset" "$index")
                if [[ -z "$model" ]]; then
                    model=$(get_default_model "$preset")
                fi
            fi

            # If no container specified, try to get from running job or show error
            if [[ -z "$container" ]]; then
                error "No container specified for restart"
                echo "" >&2
                echo "Usage:" >&2
                echo "    ./olivia.sh server restart <preset>              # e.g., glm47, gemma4, devstral" >&2
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
            cancel_job "vllm"
            if is_tunnel_alive; then
                info "Closing tunnel (node will change)..."
                kill_tunnel
            fi
            echo ""

            # Determine GPU count for this preset (Gemma 4 uses 2, others use 4)
            local gpus
            gpus=$(get_default_gpus "$preset")

            # Start server
            local job_id
            if ! job_id=$(start_server_job "$container" "$model" "$gpus"); then
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
            if job_id=$(find_job_id "vllm"); then
                local log_file="${REMOTE_CONTAINER_DIR}/logs/vllm_server_${job_id}.log"
                info "Tailing logs for job ${job_id}..."
                ssh_run "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${log_file}'" || true
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
    glm47               GLM-4.7-AWQ (QuantTrio/GLM-4.7-AWQ)             [4 GPUs]
    gemma4              Gemma 4 31B AWQ (QuantTrio/gemma-4-31B-it-AWQ)  [2 GPUs]
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
        echo "    ./olivia.sh server start gemma4     # Start Gemma 4 31B (2 GPUs)" >&2
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
    $(basename "$0") build gemma4       Build Gemma 4 container
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
