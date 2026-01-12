# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for building and running vLLM on NVIDIA GH200 (GraceHopper) ARM64 GPUs, specifically targeting the NRIS Olivia HPC cluster. The key challenge is preserving NGC's custom PyTorch build while installing vLLM and its dependencies.

## Commands

### Build vLLM Container
```bash
# Interactive (prompts for decisions)
./build_vllm_gh200.sh

# Submit as SLURM job
sbatch build_vllm_gh200.sh

# With options
VLLM_VERSION=v0.6.6 CREATE_SIF=1 ./build_vllm_gh200.sh
```

### Run vLLM Server
```bash
# Interactive
./run_vllm_server.sh

# Submit as SLURM job
sbatch run_vllm_server.sh

# With custom configuration
MODEL=mistralai/Devstral-2-123B-Instruct-2512 TP_SIZE=4 PORT=8000 ./run_vllm_server.sh
```

## Architecture

### Build Script (`build_vllm_gh200.sh`)
Five-phase build process using Singularity:
1. Create sandbox from NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`)
2. Verify NGC PyTorch installation
3. Build vLLM with pip constraints to preserve NGC PyTorch (uses `--no-deps` strategy)
4. Verify final installation (PyTorch version unchanged, vLLM functional)
5. Optionally convert sandbox to SIF image

**Critical constraint**: NGC PyTorch must not be replaced by pip. The build uses a constraints file and `--no-deps` installation to prevent this.

### Server Script (`run_vllm_server.sh`)
Runs vLLM server with GH200-optimized settings:
- Tensor parallelism across 4 GPUs by default
- NCCL optimizations for NVLink (`NCCL_P2P_LEVEL=NVL`)
- GPU reordering for better performance (`CUDA_VISIBLE_DEVICES=1,2,3,0`)
- Flash Attention backend

### Key Environment Variables

**Build:**
- `VLLM_VERSION`: vLLM version to build (default: `main`)
- `CREATE_SIF`: Set to `1` to create SIF image after build
- `MAX_JOBS`: Parallel compilation jobs (default: 8)

**Server:**
- `MODEL`: HuggingFace model ID
- `TP_SIZE`: Tensor parallel size (default: 4)
- `GPU_MEM_UTIL`: GPU memory utilization (default: 0.90)
- `MAX_MODEL_LEN`: Maximum context length (default: 32768)
- `HF_TOKEN`: HuggingFace token for gated models
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: `FLASH_ATTN`)

### Container/Cache Structure
- `vllm-gh200-sandbox/`: Singularity sandbox (writable container)
- `vllm-gh200.sif`: Compressed Singularity image (optional)
- `cache/pip/`: Pip cache directory
- `cache/huggingface/`: HuggingFace model cache
- `cache/vllm/`: vLLM cache
- `logs/`: Build and server logs
