# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for building and running vLLM on NVIDIA GH200 (GraceHopper) ARM64 GPUs, specifically targeting the NRIS Olivia HPC cluster. The key challenge is preserving NGC's custom PyTorch build while installing vLLM and its dependencies.

## Commands

### Build vLLM Container
```bash
# List available model presets
./build_vllm_gh200.sh
# or
MODEL_ID=help ./build_vllm_gh200.sh

# Build using a preset (applies recommended vLLM and transformers versions)
MODEL_ID=glm47 ./build_vllm_gh200.sh
MODEL_ID=devstral ./build_vllm_gh200.sh
MODEL_ID=llama ./build_vllm_gh200.sh

# Build second container of same type (different index)
MODEL_ID=glm47 BUILD_INDEX=2 ./build_vllm_gh200.sh

# Override preset defaults
MODEL_ID=glm47 VLLM_VERSION=v0.6.6 ./build_vllm_gh200.sh

# Submit as SLURM job
MODEL_ID=glm47 sbatch build_vllm_gh200.sh

# Create SIF image after build
MODEL_ID=glm47 CREATE_SIF=1 ./build_vllm_gh200.sh

# Custom model ID (uses generic defaults)
MODEL_ID=my-custom-model ./build_vllm_gh200.sh
```

### Run vLLM Server
```bash
# List available containers (run without CONTAINER set)
./run_vllm_server.sh

# Run specific container by name
CONTAINER=vllm-glm47-1-sandbox MODEL=zai-org/GLM-4.7-FP8 ./run_vllm_server.sh

# Run Devstral container
CONTAINER=vllm-devstral-1-sandbox MODEL=mistralai/Devstral-2-123B-Instruct-2512 ./run_vllm_server.sh

# Submit as SLURM job
CONTAINER=vllm-glm47-1-sandbox MODEL=zai-org/GLM-4.7-FP8 sbatch run_vllm_server.sh
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

### Model Presets

The build script includes predefined configurations for common models. Run without `MODEL_ID` or with `MODEL_ID=help` to see available presets:

| Preset | Description | vLLM | Transformers |
|--------|-------------|------|--------------|
| `glm47` | GLM-4.7 (358B) flagship model | main | >=5.0.0rc0 |
| `devstral` | Devstral/Mistral models | main | >=4.45.0 |
| `llama` | Llama 3.x models | main | >=4.45.0 |
| `qwen` | Qwen 2.5 models | main | >=4.45.0 |
| `generic` | Generic build (default) | main | >=4.45.0 |

Custom MODEL_ID values are also supported - they will use generic defaults.

To add a new preset, edit the `apply_preset()` function in `build_vllm_gh200.sh`.

### Server Script (`run_vllm_server.sh`)
Runs vLLM server with GH200-optimized settings:
- Tensor parallelism across 4 GPUs by default
- NCCL optimizations for NVLink (`NCCL_P2P_LEVEL=NVL`)
- GPU reordering for better performance (`CUDA_VISIBLE_DEVICES=1,2,3,0`)
- Flash Attention backend
- Speculative decoding with ngram method (disabled by default, enable for repetitive workloads)

### Key Environment Variables

**Build:**
- `CONTAINER_DIR`: Output directory for containers (default: `/path/to/containers`)
- `MODEL_ID`: Model identifier for naming (default: `generic`, e.g., `glm47`, `devstral`)
- `BUILD_INDEX`: Build index for multiple builds of same model (default: `1`)
- `VLLM_VERSION`: vLLM version to build (default: `main`)
- `CREATE_SIF`: Set to `1` to create SIF image after build
- `MAX_JOBS`: Parallel compilation jobs (default: 8)

**Server:**
- `CONTAINER_DIR`: Directory to search for containers (default: `/path/to/containers`)
- `CONTAINER`: Container name or path (if not set, lists available containers)
- `MODEL`: HuggingFace model ID (default: `mistralai/Devstral-2-123B-Instruct-2512`)
- `TP_SIZE`: Tensor parallel size (default: 4)
- `GPU_MEM_UTIL`: GPU memory utilization (default: 0.90)
- `MAX_MODEL_LEN`: Maximum context length (default: 32768)
- `HF_TOKEN`: HuggingFace token for gated models
- `VLLM_ATTENTION_BACKEND`: Attention backend (default: `FLASH_ATTN`)

**Speculative Decoding (ngram):**
- `ENABLE_SPECULATIVE`: Enable speculative decoding (default: `0`)
- `NUM_SPECULATIVE_TOKENS`: Number of tokens to speculate (default: 5)
- `PROMPT_LOOKUP_MAX`: Max n-gram window size (default: 4)

**GLM-4.7 Specific (auto-detected when MODEL contains "GLM-4.7"):**
- `GLM_TOOL_PARSER`: Tool call parser (default: `glm47`)
- `GLM_REASONING_PARSER`: Reasoning parser (default: `glm45`)
- `ENABLE_AUTO_TOOL_CHOICE`: Enable automatic tool selection (default: `0`)
- `SERVED_MODEL_NAME`: Custom model name for API (default: empty, uses model ID)
- `MTP_SPECULATIVE_TOKENS`: MTP speculative tokens for GLM-4.7 (default: `1`)

### GLM-4.7 Usage

```bash
# Build GLM-4.7 container
MODEL_ID=glm47 ./build_vllm_gh200.sh

# Run GLM-4.7 with FP8 quantization (recommended for 4x GH200)
CONTAINER=vllm-glm47-1-sandbox MODEL=zai-org/GLM-4.7-FP8 ./run_vllm_server.sh

# Run GLM-4.7 with MTP speculative decoding
CONTAINER=vllm-glm47-1-sandbox MODEL=zai-org/GLM-4.7-FP8 ENABLE_SPECULATIVE=1 ./run_vllm_server.sh

# Run GLM-4.7 with tool calling enabled
CONTAINER=vllm-glm47-1-sandbox MODEL=zai-org/GLM-4.7-FP8 ENABLE_AUTO_TOOL_CHOICE=1 ./run_vllm_server.sh
```

**Note:** GLM-4.7 (358B parameters) requires ~358GB VRAM with FP8 quantization or ~716GB with BF16. Minimum 4 GPUs with tensor parallelism.

### Container/Cache Structure

**Shared containers on the cluster** (`/path/to/containers/`):
- `vllm-{model}-{index}-sandbox/`: Singularity sandbox (writable container)
- `vllm-{model}-{index}.sif`: Compressed Singularity image (optional)

Naming examples:
- `vllm-glm47-1-sandbox` - GLM-4.7 build #1
- `vllm-devstral-1-sandbox` - Devstral build #1
- `vllm-generic-1-sandbox` - Generic build #1

**Local cache directories**:
- `cache/pip/`: Pip cache directory
- `cache/huggingface/`: HuggingFace model cache
- `cache/vllm/`: vLLM cache
- `logs/`: Build and server logs
