#!/usr/bin/env python3
"""
Patch vLLM to support Salyut1/GLM-4.7-NVFP4 model.

This model has missing k_scale and v_scale parameters that cause vLLM to crash.
This patch adds a check to skip these parameters if they're missing.

Usage:
    python patch_glm47_nvfp4.py

Run inside the container before starting vLLM with GLM-4.7-NVFP4.
"""

import sys
import os
import re

# Path to the vLLM model file
VLLM_GLM4_MOE_PATH = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py'

def patch_glm4_moe():
    path = VLLM_GLM4_MOE_PATH

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        print("Make sure you're running this inside the vLLM container.")
        sys.exit(1)

    with open(path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "k_scale" in content and "continue" in content and "params_dict" in content:
        # More specific check
        if "if ('k_scale' in name or 'v_scale' in name) and name not in params_dict:" in content:
            print(f"File already patched: {path}")
            return True

    # Find and patch the target line
    target_str = 'param = params_dict[name]'

    if target_str not in content:
        print(f"ERROR: Target string not found in {path}")
        print("vLLM version may be incompatible with this patch.")
        sys.exit(1)

    # Read line by line to preserve indentation
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    patched = False

    for line in lines:
        if target_str in line and not patched:
            # Get the indentation
            whitespace = re.match(r'^(\s*)', line).group(1)

            # Add the skip logic before the param assignment
            patch_line = f"{whitespace}if ('k_scale' in name or 'v_scale' in name) and name not in params_dict: continue\n"

            new_lines.append(patch_line)
            new_lines.append(line)
            patched = True
        else:
            new_lines.append(line)

    if patched:
        with open(path, 'w') as f:
            f.writelines(new_lines)
        print(f"Successfully patched: {path}")
        print("vLLM is now compatible with Salyut1/GLM-4.7-NVFP4")
        return True
    else:
        print("ERROR: Failed to apply patch")
        sys.exit(1)

if __name__ == '__main__':
    patch_glm4_moe()
