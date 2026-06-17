#!/usr/bin/env python3
"""Wire reasoning_tokens into /v1/chat/completions usage (vLLM 0.21).

vLLM 0.21 only reports reasoning_tokens on /v1/responses; chat/completions has
no completion_tokens_details at all. This:
  1. engine/protocol.py: add CompletionTokenUsageInfo + UsageInfo.completion_tokens_details
  2. chat_completion/serving.py: import it and populate it in BOTH the
     non-streaming and streaming final-usage paths, via the reasoning parser's
     count_reasoning_tokens(). (Olivia local patch 2026-06-15)
Idempotent. Usage: patch_chat_reasoning_tokens.py <engine/protocol.py> <chat_completion/serving.py>
"""
import sys

proto, serving = sys.argv[1], sys.argv[2]

# ---- 1. protocol.py: type + field ----
p = open(proto).read()
if "class CompletionTokenUsageInfo" in p:
    print("protocol.py already patched")
else:
    old = (
        "class UsageInfo(OpenAIBaseModel):\n"
        "    prompt_tokens: int = 0\n"
        "    total_tokens: int = 0\n"
        "    completion_tokens: int | None = 0\n"
        "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
    )
    new = (
        "class CompletionTokenUsageInfo(OpenAIBaseModel):\n"
        "    reasoning_tokens: int | None = None\n\n\n"
        "class UsageInfo(OpenAIBaseModel):\n"
        "    prompt_tokens: int = 0\n"
        "    total_tokens: int = 0\n"
        "    completion_tokens: int | None = 0\n"
        "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
        "    completion_tokens_details: CompletionTokenUsageInfo | None = None\n"
    )
    assert old in p, "protocol UsageInfo block not found"
    open(proto, "w").write(p.replace(old, new, 1))
    print("protocol.py patched")

# ---- 2. serving.py: import + two usage sites ----
s = open(serving).read()
orig = s

if "CompletionTokenUsageInfo" not in s:
    imp = "    PromptTokenUsageInfo,\n"
    assert imp in s, "serving import marker not found"
    s = s.replace(imp, "    CompletionTokenUsageInfo,\n    PromptTokenUsageInfo,\n", 1)

# non-streaming (chat_completion_full_generator)
if "usage.completion_tokens_details" not in s:
    ns_old = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
        "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    )
    ns_new = (
        "        usage = UsageInfo(\n"
        "            prompt_tokens=num_prompt_tokens,\n"
        "            completion_tokens=num_generated_tokens,\n"
        "            total_tokens=num_prompt_tokens + num_generated_tokens,\n"
        "        )\n"
        "        if reasoning_parser is not None:\n"
        "            _reasoning_toks = sum(\n"
        "                reasoning_parser.count_reasoning_tokens(output.token_ids)\n"
        "                for output in final_res.outputs\n"
        "            )\n"
        "            if _reasoning_toks:\n"
        "                usage.completion_tokens_details = CompletionTokenUsageInfo(\n"
        "                    reasoning_tokens=_reasoning_toks\n"
        "                )\n"
        "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    )
    assert ns_old in s, "non-streaming usage marker not found"
    s = s.replace(ns_old, ns_new, 1)

# streaming (chat_completion_stream_generator)
if "final_usage.completion_tokens_details" not in s:
    st_old = (
        "                final_usage = UsageInfo(\n"
        "                    prompt_tokens=num_prompt_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                )\n"
        "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    )
    st_new = (
        "                final_usage = UsageInfo(\n"
        "                    prompt_tokens=num_prompt_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                )\n"
        "                if (\n"
        "                    reasoning_parser is not None\n"
        "                    and all_previous_token_ids is not None\n"
        "                ):\n"
        "                    _reasoning_toks = sum(\n"
        "                        reasoning_parser.count_reasoning_tokens(ids)\n"
        "                        for ids in all_previous_token_ids\n"
        "                    )\n"
        "                    if _reasoning_toks:\n"
        "                        final_usage.completion_tokens_details = (\n"
        "                            CompletionTokenUsageInfo(\n"
        "                                reasoning_tokens=_reasoning_toks\n"
        "                            )\n"
        "                        )\n"
        "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    )
    assert st_old in s, "streaming usage marker not found"
    s = s.replace(st_old, st_new, 1)

if s != orig:
    open(serving, "w").write(s)
    print("serving.py patched")
else:
    print("serving.py already patched")
