# vLLM for NVIDIA GH200 (GraceHopper) on HPC Clusters

Build and run [vLLM](https://github.com/vllm-project/vllm) on NVIDIA GH200 ARM64 GPUs, specifically optimized for the NRIS Olivia HPC cluster — from single-node 4-GPU models up to 1T-parameter MoE models sharded across 3 nodes / 12 GPUs. Includes full-featured CLI tooling, a streaming chat client, an Anthropic/Claude-Code bridge, and performance optimizations for high-latency SSH tunnel connections.

## Features

- **Preserves NGC PyTorch** - Builds vLLM without overwriting NVIDIA's custom PyTorch
- **Model Presets** - Build + serve recipes for GLM-4.7, GLM-5.1, GLM-5.2, Kimi K2.6/K2.7, Laguna M.1, Gemma-4, Devstral, Llama, and Qwen
- **Multi-node serving** - TP=4 intra-node + pipeline parallel across nodes over Slingshot, with an auto-bootstrapped Ray cluster (GLM-5.1/5.2 and Kimi span 2–3 nodes)
- **Reproducible builds** - Pin a vLLM commit and graft not-yet-released upstream PRs from committed snapshots (`VLLM_PATCHES`), so a container rebuilds byte-identically
- **GH200 Optimizations** - NCCL/NVLink tuning, optimal GPU ordering, Flash Attention, DeepGEMM/FP8 paths
- **Unified CLI** - Single command interface with SSH ControlMaster (one 2FA per session) + durable reconnect
- **Claude Code bridge** - `anthropic_proxy.py` serves the Anthropic Messages API (thinking/reasoning + tool calls) on top of the OpenAI endpoint
- **reasoning_tokens** - Reported on `/v1/chat/completions` usage for the reasoning models (Kimi, GLM-5.x)
- **Streaming Proxy** - Batches SSE tokens for ~3x faster streaming over SSH tunnels
- **Smart Monitoring** - Multi-phase server watch with GPU loading progress and live throughput

## Quick Start

```bash
# Show help and available commands
./olivia.sh

# Check cluster status
./olivia.sh status

# Build a GLM-4.7 container
./olivia.sh build glm47

# Start the server
./olivia.sh server start glm47

# Monitor server startup (GPU loading → health → serving)
./olivia.sh server watch

# Connect and chat
./olivia.sh chat
```

## Prerequisites

- SSH access to your HPC cluster with 2FA configured
- Python 3.8+ with `requests` and `rich` libraries (for chat client)
- HuggingFace token for gated models (Llama, etc.)

## Configuration

This repo ships with **safe generic defaults**. You must configure your cluster settings via environment variables.

### CLI (`olivia.sh`) configuration

Set these before using `./olivia.sh`:

```bash
export REMOTE_HOST=<your-cluster-login-host>
export REMOTE_CONTAINER_DIR=<path-on-cluster-for-containers>

# Optional (defaults shown)
export REMOTE_USER=$USER
export REMOTE_PORT=8000   # cluster-side vLLM port
export LOCAL_PORT=8003    # local forward; avoids :8000 (another local dev service may bind it)
```

### Direct scripts configuration

The direct scripts require `CONTAINER_DIR` (the directory containing your Singularity sandboxes/SIFs on the cluster):

```bash
export CONTAINER_DIR=<path-on-cluster-for-containers>
```

## CLI Reference

### `./olivia.sh`

Unified CLI for all operations. Uses SSH ControlMaster for single 2FA authentication per session.

| Command | Description |
|---------|-------------|
| `chat` | Connect to vLLM and start interactive chat |
| `build` | Build vLLM containers |
| `server` | Manage vLLM server (start, stop, logs) |
| `proxy` | Durable multi-model router on the small partition (start, tunnel, status) |
| `tunnel` | Manage SSH tunnel to GPU node |
| `status` | Show cluster and connection status |
| `--kill-all` | Close tunnel and SSH connection |

### Chat Module

```bash
./olivia.sh chat               # Connect and start interactive chat
./olivia.sh chat --port 9000   # Use different local port
./olivia.sh chat --tunnel-only # Just set up tunnel, don't start chat
./olivia.sh chat --no-stream   # Disable streaming responses
```

### Build Module

```bash
./olivia.sh build              # Show build help
./olivia.sh build --presets    # List available model presets
./olivia.sh build --list       # List existing containers on cluster

# Build containers (build + serve presets share the same name)
./olivia.sh build glm52        # GLM-5.2 (FP8, pinned vLLM commit + PR#45895 snapshot)
./olivia.sh build kimi         # Kimi K2.6/K2.7 (shared container)
./olivia.sh build glm51        # GLM-5.1
./olivia.sh build glm47        # GLM-4.7 (single node)
./olivia.sh build devstral     # Devstral

# Build options
./olivia.sh build glm47 --index 2    # Build second container (safe, won't touch existing)
./olivia.sh build glm47 --force      # Rebuild existing container
./olivia.sh build glm47 --sif        # Create SIF image after build
./olivia.sh build glm47 --vllm v0.6.6  # Override vLLM version (branch, tag, or commit SHA)
```

**Safety:** Builds fail by default if a container already exists. Use `--index N` to create a new container or `--force` to explicitly overwrite.

### Server Module

```bash
./olivia.sh server                   # Show server help
./olivia.sh server list              # List available containers
./olivia.sh server status            # Show running server status

# Start servers (preset with default model; multi-node presets auto-allocate nodes)
./olivia.sh server start glm52       # GLM-5.2 (3 nodes × 4, eager) — see HF_HOME note below
./olivia.sh server start kimi        # Kimi K2.6 (2 nodes × 4, eager)
./olivia.sh server start glm51       # GLM-5.1 (2 nodes × 4)
./olivia.sh server start glm47       # GLM-4.7 (single node, 4 GPUs)

# Start with options
./olivia.sh server start glm47 --index 2              # Use vllm-glm47-2-sandbox
./olivia.sh server start glm47 --model custom/model   # Override default model
./olivia.sh server start -c vllm-custom-1-sandbox -m my/model  # Explicit container

# Monitoring and management
./olivia.sh server watch             # Smart monitor with progress bars
./olivia.sh server logs              # Tail logs of running server
./olivia.sh server ssh               # Open shell on GPU node
./olivia.sh server restart glm47     # Cancel running job and restart
./olivia.sh server cancel            # Cancel running vLLM job
./olivia.sh server deploy            # Upload run_vllm_server.sh to cluster
```

**Watch command phases:**
1. **WAITING** - Waits for SLURM job to be submitted
2. **PENDING** - Job queued, waiting for resources
3. **LOADING** - GPU memory increasing as weights load (progress bar)
4. **INIT** - Weights loaded, checking /health endpoint
5. **SERVING** - Live throughput monitoring (tok/s, active requests, KV cache)

### Tunnel Module

```bash
./olivia.sh tunnel             # Show tunnel status
./olivia.sh tunnel up          # Open tunnel to vLLM server
./olivia.sh tunnel down        # Close tunnel
```

### Proxy Module (durable multi-model router)

A CPU-only reverse proxy on Olivia's **`small`** partition (up to a 7-day
walltime) that gives clients **one stable endpoint** routing to whichever GPU
server is live. Select a model by the request's `model` field — a preset name
(`glm51`, `kimi27`, ...), alias, or served repo id — and the router finds the
backend by listing running `vllm-*` jobs and **probing each `/v1/models`**, so it
works regardless of job naming; you never need to know the node or container
index. It replaces the (now-removed) login-node relay with a queue-system job
(the NRIS-policy-correct place for a long-lived process), and **auto-stops after
30 min with no GPU servers up** so an idle CPU job doesn't bill its reservation.
Full design + Sigma2 policy analysis:
[`plans/proposed/small_partition_proxy.md`](plans/proposed/small_partition_proxy.md).

```bash
./olivia.sh proxy start        # Deploy + submit the router (small partition)
./olivia.sh server start glm51 # Start GPU servers as usual; router picks them up (~15s)
./olivia.sh proxy tunnel       # Forward localhost:8003 -> router node
curl localhost:8003/v1/models  # See which presets are currently live
./olivia.sh proxy status       # Router job + live models
./olivia.sh proxy stop         # Cancel the router (it bills its small reservation while up)
```

> Compute nodes aren't internet-facing, so the laptop still tunnels in through
> the login node — but the tunnel target (the `small` node) is now stable for the
> job's lifetime instead of moving on every GPU job restart. On-cluster
> validation is pending (see the plan doc's checklist).

## Model Presets

| Preset | Default Model | GPUs | Container | Notes |
|--------|---------------|------|-----------|-------|
| `glm51_v19` (alias `glm51`) | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-1` | TP=4 + PP=2, vLLM v0.19.0. Multi-node PP decode wedge → serve behind `anthropic_proxy.py` serialization |
| `glm51_v20` | `cyankiwi/GLM-5.1-AWQ-4bit` | 8 (2 nodes × 4) | `vllm-glm51-2` | vLLM v0.20.0 + RayExecutorV2. **Quarantined** (same wedge) |
| `glm52` | `RedHatAI/GLM-5.2-FP8` | 12 (3 nodes × 4) | `vllm-glm52-1` | TP=4 + PP=3, block-FP8 (~755 GB). vLLM main pinned `091386a` + PR#45895 snapshot. Eager; fp8 KV + DeepGEMM |
| `glm47` | `QuantTrio/GLM-4.7-AWQ` | 4 | `vllm-glm47-1` | TP=4, MTP speculative |
| `kimi` | `moonshotai/Kimi-K2.6` | 8 (2 nodes × 4) | `vllm-kimi-4` | TP=4 + PP=2, native int4, MLA, multimodal, vLLM 0.21. Eager. reasoning_tokens on chat/completions |
| `kimi27` | `moonshotai/Kimi-K2.7-Code` | 8 (2 nodes × 4) | `vllm-kimi-4` (shared) | Same arch + container as K2.6 (no rebuild); thinking-only |
| `laguna` | `poolside/Laguna-M.1-FP8` | 4 | `vllm-laguna-1` | TP=4, single node. FP8 (~225 GB), dense attention (FLASH_ATTN), CUDAGraph on. vLLM v0.21.0, `poolside_v1` parsers |
| `gemma4` | Gemma 4 (31B, multimodal) | 1–2 | `vllm-gemma4-1` | vLLM v0.19.0, AWQ |
| `devstral` | `mistralai/Devstral-2-123B-Instruct-2512` | 4 | `vllm-devstral-1` | TP=4 |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | — | TP=4 |
| `qwen` | `Qwen/Qwen2.5-72B-Instruct` | 4 | — | TP=4 |
| `generic` | *(user specified)* | 4 | — | generic defaults |

> Build and serve presets are aligned by name (e.g. `./olivia.sh build glm52` then `./olivia.sh server start glm52`). The **Container** column is where each preset's server looks (`vllm-<name>-<index>-sandbox`); `kimi`/`kimi27` share index 4. See **[CLAUDE.md](CLAUDE.md)** for full per-model guides — memory layout, quant options, known issues, and multi-node architecture.

## Performance

Latest measured throughput / latency. **Update this section after every sweep** (with the date + config).

### Laguna M.1 (`laguna`) — 1 node × 4 GH200, FP8, CUDAGraph · 2026-06-20
Concurrency sweep (`bench_sweep.py`, `max_tokens=512`), reasoning on (`enable_thinking=true`) vs off:

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Agg tok/s — reasoning on  | 62.7 | 118.6 | 229.7 | 346.6 | 616.3 | 1182.3 | 2055.7 |
| Per-stream tok/s — on     | 62.7 | 59.4 | 57.5 | 43.3 | 38.5 | 37.0 | 32.2 |
| Agg tok/s — reasoning off | 63.3 | 118.9 | 229.9 | 411.8 | 717.6 | 1147.6 | 2002.3 |
| Per-stream tok/s — off    | 63.3 | 59.5 | 57.6 | 51.5 | 44.9 | 35.9 | 31.5 |

Single node, TP=4 — no multi-node PP, so none of the glm51/glm52 decode wedge. **First preset to run with CUDAGraph capture** (Laguna's ordinary dense attention captures cleanly, unlike the eager Kimi/glm52): single-stream **~63 tok/s** (~3.7× the eager Kimi's ~17), aggregate near-linear to **~2050 tok/s at 64-way**, 0 failures 1→64, sub-second TTFT (one transient ~2 s blip at 8–16 reasoning-on). Reasoning on vs off is the same decode rate — thinking just emits more tokens per request (~300 reasoning tokens on a 400-token answer), so it's longer per request, not slower per token. vLLM v0.21.0, transformers 5.12, fastapi 0.136.3.

### GLM-5.2 (`glm52`) — 3 nodes × 4 GH200, eager, FP8 · 2026-06-18
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, thinking on):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 5.6 | 11.1 | 22.5 | 43.1 | 81.1 | 130.7 | 224.5 | 419.0 |
| Per-stream tok/s | 5.6 | 5.6 | 5.6 | 5.4 | 5.1 | 4.1 | 4.7 | 6.6 |

Stable 1→64 (0 failures, no decode wedge — RayExecutorV2). Single-stream is slow (~5.6 tok/s, eager) with high TTFT (~14 s, PP=3 prefill); strong batched throughput (~75× from 1→64). CUDAGraph capture IMAs on this NGC stack, so eager only.

### Kimi K2.7 / K2.6 — 2 nodes × 4 GH200, eager, native int4 · 2026-06-20
Concurrency sweep (256 output tokens, distinct prompts):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Aggregate tok/s | 17.2 | 37.3 | 77.8 | 133.6 | 264.3 | 590.9 |
| Per-stream tok/s | 17.2 | 18.6 | 19.5 | 16.8 | 16.5 | 18.5 |

Per-stream flat ~17–19 tok/s; TTFT ~1.0 s single-stream; 0 failures (re-confirmed 2026-06-20, unchanged vs 2026-06-15 within noise). Production K2.6 sustains ~830 tok/s at 48 concurrent. Eager (CUDAGraph capture unrecoverable on this stack).

> **Cold-start ≈ 40 min** (measured 2026-06-20): the ~640 GB int4 checkpoint loads at ~38 s/shard × 64 shards off Lustre (~270 MB/s), during which the server sits at "weights reserved, 0 % util, `/health` 000" — that is loading, **not** a hang. `./olivia.sh server watch` and any health-wait must allow ~40+ min before the server answers. Cross-node NCCL runs over TCP (`NET/Socket`, no CXI/RDMA plugin), but loading — not NCCL — dominates cold-start.

### GLM-5.1 (`glm51`) — 2 nodes × 4 GH200, AWQ, **PIECEWISE CUDAGraph capture** · NGC 26.03 · 2026-06-20
Concurrency sweep (`bench_sweep.py`, `max_tokens=256`, streaming):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Aggregate tok/s  | 22.0 | 44.6 | 87.6 | 170.6 | 246.2 | 626.9 | 894.6 |
| Per-stream tok/s | 22.0 | 22.3 | 21.9 | 21.3 | 15.4 | 19.6 | 14.0 |
| p95 TTFT (s)     | 0.09 | 0.08 | 0.14 | 0.19 | 2.54 | 0.41 | 0.68 |

This config runs **0 failures across 1→64** (previously hung at `Running ≥ 2`, forcing `anthropic_proxy.py` serialization). An isolating experiment separates the two effects — on the freshly rebuilt NGC-**26.03** `vllm-glm51-1-sandbox`:
- **De-wedge = NCCL all-reduce, not capture.** `CUDAGRAPH_MODE=PIECEWISE` auto-disables vLLM's custom all-reduce → graph-safe NCCL. The isolating test — **eager + `DISABLE_CUSTOM_ALL_REDUCE=1`, no capture — also runs 0 failures at concurrency 1–16**, so the *custom all-reduce kernel* was the wedge cause; NCCL fixes it. Capture isn't needed to de-wedge (it just forces the custom kernel off, since it isn't graph-safe).
- **Throughput = capture.** PIECEWISE capture (51/51 graphs, no IMA — 26.03's inductor handles GLM-DSA capture where **26.05 IMAs** on glm52/Kimi) lifts single-stream from **~5 tok/s** (eager+NCCL) to **~22 tok/s** (~4.5×), holding ~22/stream through 8-way, ~895 tok/s @64.

Recommended config: **capture + NCCL all-reduce** (de-wedged *and* fast). The earlier "wedge → serialize" workaround is **superseded** for this container.

### GLM-4.7
- **GLM-4.7** (single node, 4 GPUs, AWQ): fast single-node, no multi-node wedge.

> Sweep tools: `bench_sweep.py` (concurrency, streaming SSE) and `bench_serving.py` (TTFT + decode). Re-run after any serving-config change and refresh the tables above.

### Agentic behaviour — Laguna M.1 (`laguna`) · 2026-06-20

First live run of the agentic-eval harness (`evals/`, see `plans/proposed/agentic_evals.md`) — measures *behaviour*, not throughput. Driven through `anthropic_proxy.py` (`:8002`) exactly as Claude Code would, against laguna single-node, thinking on, CUDAGraph capture. Serving note: laguna needs `HF_HOME=/cluster/work/...` (project quota is full; the FP8 weights live on `/cluster/work`).

**L0 — protocol / tool-call conformance** (`evals/runner.py protocol`, 8 fixtures × {nonstream, stream}):
- **Invariant gate (proxy regression): 1562/1562 clean** — the `poolside_v1` tool/reasoning parser translates faithfully in both modes, and the streaming SSE block grammar holds. This is the headline proxy-health result.
- **Behavioural: 100%** across tool-selection, arg-schema (incl. enum), multi-turn tool_result threading, parallel tools, no-tool-when-unneeded, hallucination guard, and thinking→tool_use ordering.

**L1 — micro-agent tasks** (`evals/runner.py micro`, 8 sandboxed tasks, oracle-verified, `max_turns=14`):

| metric | value |
|---|---|
| success (oracle-verified) | **8/8 (100%)** |
| median turns to solve | 4.5 |
| premature-stop / runaway / errors | 0 / 0 / 0 |
| invalid-tool-turns (protocol break mid-loop) | 0 |
| thrash (repeated identical calls) | 3 (across 2 tasks) |

Per task (turns): write-file 3 · fix-failing-test 7 · implement-fn 6 · find/report 5 · rename-symbol (multi-file) 10 · count-files 4 · fix-syntax 4 · sum-csv 3 — all solved. Laguna drives the multi-turn loop reliably with clean tool-call validity throughout.

**L2 — agentic coding** (`evals/runner.py swe`, 5 harder multi-file tasks, `max_turns=25`): a buggy mini-codebase + problem statement, graded by a **hidden** regression suite written only *after* the loop ends — the agent never sees the test (SWE-bench-style).

| metric | value |
|---|---|
| success (oracle-verified) | **5/5 (100%)** |
| median turns to solve | 5 |
| premature-stop / runaway / errors / invalid-tool-turns | 0 / 0 / 0 / 0 |

Per task (turns): calc-operator-precedence (wrote a recursive-descent precedence parser) 16 · interval-merge-adjacency 4 · lru-cache-recency 5 · topo-sort-cycle-detection 6 · csv-quoted-field-parsing 4 — all solved against the hidden grader. This slice did **not** find Laguna's ceiling (it's a strong coding agent on well-defined bugs); the real SWE-bench Verified dataset is L2-real below.

**L2-real — SWE-bench Verified** (`evals/swe_real/`): *real* django instances from SWE-bench Verified. Runs as a SLURM job on Olivia's **`small` CPU partition** inside a `python:3.11` apptainer (x86 + internet + reaches the vLLM node over the cluster network → within Sigma2 usage policy; no login-node compute, no GPU waste, no tunnel). Per instance: reset django to the base commit, drive the model through the agent loop to fix the real bug, then apply the gold `test_patch` (the hidden FAIL_TO_PASS/PASS_TO_PASS suite) and run the affected modules. Harness validated by a gold-patch self-test (`--gold`): **12/12** resolve, confirming setup+verify independent of the model.

12-instance django 4.2/5.0 slice · 2026-06-20 (Laguna, `max_turns=40`, ~23 min for all 12):

| metric | value |
|---|---|
| **resolved** | **5/12 (42%)** |
| precision when it edits | **5/5** — every non-empty patch passed the hidden suite |
| dominant failure mode | **7/7 unresolved made no edit** (patch=0) |

Resolved: 15851, 16255, 16485, 16527, 16801. The signal L1/L2 (both 100%) couldn't give: Laguna's **fixes are reliable** (5/5 correct), but **7/7 failures made no edit at all**.

**Turn-budget experiment (single variable, `max_turns` 40→80 on those 7):** came back **0/7, still `patch=0`** — 6 ran the full 80 turns without editing, one *stopped* at 30 turns declaring done without writing a fix. So it is **not a turn-budget problem**: doubling the budget changed nothing. The failure mode is failing to translate exploration into an *edit* (reads/runs but never commits a `write_file`, or stops prematurely). The lever is **agent scaffolding that forces a patch** (e.g. require a diff before stopping), not more turns or fix quality. A legitimate real-benchmark number (SWE-bench Verified is hard). Slice is django-only (the tractable subset on this cluster — no per-instance Docker); broadening repos/instances and adding glm52/kimi27 is next.

> Eval tools: `evals/runner.py {protocol,micro,swe}` (L0/L1/L2, in-repo); `evals/swe_real/run_on_cluster.sh` (L2-real, SLURM on `small`). Offline self-tests: `evals/{protocol,micro}/selftest.py`; harness self-test: `--gold`. Re-run per preset after any proxy/serving change; results in `evals/results/` (L0–L2) and `/cluster/work/.../swe/` (L2-real).

## Direct Script Usage

The underlying scripts can be used directly on the cluster without the CLI:

### Build Container

```bash
# List presets
MODEL_ID=help ./build_vllm_gh200.sh

# Build using a preset
MODEL_ID=glm47 ./build_vllm_gh200.sh

# Build second container of same type
MODEL_ID=glm47 BUILD_INDEX=2 ./build_vllm_gh200.sh

# Submit as SLURM job
MODEL_ID=glm47 sbatch build_vllm_gh200.sh

# Override preset defaults
MODEL_ID=glm47 VLLM_VERSION=v0.6.6 ./build_vllm_gh200.sh

# Create SIF image after build
MODEL_ID=glm47 CREATE_SIF=1 ./build_vllm_gh200.sh
```

### Run Server

```bash
# List available containers
./run_vllm_server.sh

# Run specific container
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Submit as SLURM job
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ sbatch run_vllm_server.sh

# Enable batching proxy for faster streaming over SSH
ENABLE_PROXY=1 CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh
```

## GLM-4.7 Guide

GLM-4.7 is a 358B parameter Mixture-of-Experts model requiring careful memory management.

### Quantization Options

| Model | Size | GH200 Compatible | Notes |
|-------|------|------------------|-------|
| `QuantTrio/GLM-4.7-AWQ` | ~181GB | **Yes (Recommended)** | AWQ 4-bit, leaves ~200GB for KV cache |
| `QuantTrio/GLM-4.7-AWQ` | ~358GB | Yes | FP8, tight fit - reduce MAX_MODEL_LEN |
| `Salyut1/GLM-4.7-NVFP4` | ~179GB | **No** | Requires Blackwell GPUs (B100/B200) |

### Running GLM-4.7

```bash
# Build GLM-4.7 container
./olivia.sh build glm47

# Start with AWQ quantization (recommended)
./olivia.sh server start glm47

# Or with direct scripts:
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# FP8 with reduced context (tight memory fit)
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ MAX_MODEL_LEN=8192 ./run_vllm_server.sh

# Enable MTP speculative decoding
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_SPECULATIVE=1 ./run_vllm_server.sh

# Enable tool calling
CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ENABLE_AUTO_TOOL_CHOICE=1 ./run_vllm_server.sh
```

### Memory Requirements

| Quantization | Model Size | 4×GH200 (384GB) | Notes |
|--------------|------------|-----------------|-------|
| AWQ 4-bit | ~181GB | ~200GB free | Recommended |
| FP8 | ~358GB | ~26GB free | Reduce MAX_MODEL_LEN |
| BF16 | ~716GB | Won't fit | Needs 8+ GPUs |

## Batching Proxy

When accessing vLLM over SSH tunnels, streaming responses can be slow due to per-token network overhead. The batching proxy aggregates tokens into single SSE events, improving streaming throughput by ~3x.

```
Client <--[batched SSE]--> Proxy:8001 <--[per-token SSE]--> vLLM:8000
         (SSH tunnel)                    (localhost, fast)
```

### Performance

| Mode | Without Proxy | With Proxy |
|------|---------------|------------|
| Non-streaming | 17 tok/s | 17 tok/s |
| Streaming | ~5 tok/s | ~15 tok/s |

### Usage

```bash
# Enable proxy when starting server
ENABLE_PROXY=1 CONTAINER=vllm-glm47-1-sandbox MODEL=QuantTrio/GLM-4.7-AWQ ./run_vllm_server.sh

# Tunnel to proxy port
ssh -L 8001:localhost:8001 user@<cluster-login-host>...

# Or run proxy standalone
python vllm_proxy.py --vllm-port 8000 --proxy-port 8001 --batch-tokens 15 --batch-delay-ms 150
```

## Environment Variables

### Build Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | *(required)* | Model preset or custom identifier |
| `BUILD_INDEX` | `1` | Build index for multiple containers |
| `VLLM_VERSION` | `main` | vLLM ref to build — branch, tag, **or commit SHA** (presets may pin a SHA for reproducible builds) |
| `VLLM_PATCHES` | *(preset)* | Space-separated vLLM PR numbers to graft at build time (committed `patches/` snapshot preferred, else live GitHub fetch) |
| `NGC_PYTORCH_TAG` | `26.03-py3` | NGC PyTorch base-image tag (a preset may pin, e.g. glm52 → `26.05-py3`) |
| `DEEPGEMM_REF` | `59f2c07` | DeepGEMM commit (a preset may pin, e.g. glm52) |
| `CREATE_SIF` | `0` | Create SIF image after build |
| `OVERWRITE` | `0` | Allow overwriting existing containers |
| `MAX_JOBS` | `8` | Parallel compilation jobs |
| `CONTAINER_DIR` | *(required)* | Output directory |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER` | *(required)* | Container name or path |
| `MODEL` | `mistralai/Devstral-2-123B-Instruct-2512` | HuggingFace model ID |
| `HF_HOME` | *(required)* | Persistent HF weights cache. Must be on project storage (auto-purge-safe), **not** `/cluster/work` — except glm52, whose ~700 GB FP8 exceeds the project quota, so override to the work cache. Normally forwarded from `mise.local.toml` |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models (forwarded over stdin) |
| `TP_SIZE` | `4` | Tensor parallel size (intra-node) |
| `NUM_NODES` | `1` | Nodes for multi-node serving (glm51/kimi = 2, glm52 = 3); auto-bootstraps Ray |
| `PP_SIZE` | `1` | Pipeline-parallel size across nodes (2 for glm51/kimi, 3 for glm52) |
| `CUDAGRAPH_MODE` | *(auto)* | `NONE` = eager. Kimi and glm52 default to eager (CUDAGraph capture IMAs on this NGC stack) |
| `GPU_MEM_UTIL` | `0.90` | GPU memory utilization |
| `MAX_MODEL_LEN` | `32768` | Max context length (131072 for GLM-5.x / Kimi) |
| `VERBOSE` | `0` | Enable detailed logging |

### Speculative Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SPECULATIVE` | `auto` | Enable speculative decoding (`auto`, `0`, `1`) |
| `NUM_SPECULATIVE_TOKENS` | `5` | Tokens to speculate (ngram) |
| `PROMPT_LOOKUP_MAX` | `4` | Max n-gram window size |
| `MTP_SPECULATIVE_TOKENS` | `3` | MTP tokens for GLM-4.7 |

### GLM-4.7 Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_TOOL_PARSER` | `glm47` | Tool call parser |
| `GLM_REASONING_PARSER` | `glm45` | Reasoning parser |
| `ENABLE_AUTO_TOOL_CHOICE` | `0` | Enable automatic tool selection |
| `SERVED_MODEL_NAME` | *(model ID)* | Custom model name for API |
| `ENABLE_EXPERT_PARALLEL` | `auto` | Expert parallel for AWQ MoE models |

### Batching Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PROXY` | `0` | Enable batching proxy |
| `PROXY_PORT` | `8001` | Proxy server port |
| `PROXY_BATCH_TOKENS` | `15` | Flush after N tokens |
| `PROXY_BATCH_CHARS` | `100` | Flush after N characters |
| `PROXY_BATCH_DELAY_MS` | `150` | Max delay before flush (ms) |

## Architecture

### Build Process (`build_vllm_gh200.sh`)

Five-phase build using Singularity:

1. **Create sandbox** from NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`)
2. **Verify NGC PyTorch** installation is intact
3. **Build vLLM** with pip constraints to preserve NGC PyTorch (`--no-deps` strategy)
4. **Verify final installation** (PyTorch version unchanged, vLLM functional)
5. **Optionally convert** sandbox to SIF image

**Key constraint:** NGC PyTorch must not be replaced by pip. The build uses a constraints file and `--no-deps` installation to prevent this.

### Server Optimizations (`run_vllm_server.sh`)

GH200-specific optimizations:

- **GPU Ordering:** `CUDA_VISIBLE_DEVICES=1,2,3,0` puts slowest GPU last
- **NCCL for NVLink:** `NCCL_P2P_LEVEL=NVL` enables NVLink peer-to-peer
- **GPU Direct RDMA:** `NCCL_NET_GDR_LEVEL=PHB`
- **Flash Attention:** Backend set to `FLASH_ATTN`
- **Memory:** Expandable PyTorch memory segments

### Directory Structure

```
vllm-ngc/
├── olivia.sh              # Unified CLI
├── build_vllm_gh200.sh    # Container build script
├── run_vllm_server.sh     # Server run script
├── chat_devstral.py       # Interactive chat client
├── vllm_proxy.py          # SSE batching proxy
├── patch_glm47_nvfp4.py   # NVFP4 compatibility patch
├── cache/                 # Local cache directories
│   ├── pip/
│   ├── huggingface/
│   └── vllm/
└── logs/                  # Build and server logs
```

**Shared containers on the cluster** (`CONTAINER_DIR`):

```
vllm-glm47-1-sandbox/      # GLM-4.7 build #1
vllm-devstral-1-sandbox/   # Devstral build #1
vllm-generic-1-sandbox/    # Generic build #1
vllm-glm47-1.sif           # Compressed SIF image (optional)
```

## Chat Client

Interactive chat client with rich terminal UI:

```bash
# Basic usage (port matches the tunnel's local port; default LOCAL_PORT=8003)
python chat_devstral.py localhost --port 8003 --stream

# Features:
# - Multi-turn conversation history
# - Token usage and generation speed metrics
# - Markdown rendering
# - Streaming with live display
```

Dependencies: `pip install requests rich`

## Troubleshooting

### Build fails with "container already exists"

Use `--index N` to create a new container or `--force` to overwrite:

```bash
./olivia.sh build glm47 --index 2   # Create vllm-glm47-2-sandbox
./olivia.sh build glm47 --force     # Overwrite vllm-glm47-1-sandbox
```

### Server won't start - out of memory

Reduce context length or use quantized models:

```bash
# Reduce context length
MAX_MODEL_LEN=8192 ./olivia.sh server start glm47

# Use AWQ quantization (recommended for GLM-4.7)
./olivia.sh server start glm47 --model QuantTrio/GLM-4.7-AWQ
```

### Slow streaming over SSH tunnel

Enable the batching proxy:

```bash
ENABLE_PROXY=1 ./olivia.sh server start glm47
# Then tunnel to port 8001 instead of 8000
```

### NVFP4 model fails to load

NVFP4 quantization requires Blackwell GPUs (B100/B200). Use AWQ instead:

```bash
# Don't use NVFP4 on GH200
# MODEL=Salyut1/GLM-4.7-NVFP4  # Won't work!

# Use AWQ instead
MODEL=QuantTrio/GLM-4.7-AWQ ./olivia.sh server start glm47
```

### 2FA prompt on every command

Ensure SSH ControlMaster is working:

```bash
./olivia.sh status  # Check connection status
./olivia.sh --kill-all && ./olivia.sh status  # Reset and reconnect
```

## License

MIT
