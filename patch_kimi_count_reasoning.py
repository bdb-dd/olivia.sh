#!/usr/bin/env python3
"""Patch vLLM's KimiK2ReasoningParser to actually count reasoning tokens.

The base ReasoningParser.count_reasoning_tokens() returns 0 unless a parser
opts in, and kimi_k2 never overrides it -> usage reasoning_tokens is always 0
even though the reasoning IS extracted. Kimi K2/K2.7 also typically omit the
<think> start token (reasoning begins at the start, ended by </think> or a
tool-call section), so a naive start/end depth counter would also count 0.

This inserts a correct override that counts tokens before the first end marker,
dropping any leading <think> -- mirroring the parser's own extract_reasoning().
Idempotent. Usage: python3 patch_kimi_count_reasoning.py <parser.py>
"""
import sys

f = sys.argv[1]
s = open(f).read()
if "def count_reasoning_tokens" in s:
    print("already patched"); sys.exit(0)

method = '''    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning tokens (Olivia local patch 2026-06-15).

        Base default returns 0; Kimi K2 omits the <think> start token, so count
        all generated tokens before the first </think> / tool-call-section
        marker, dropping a leading <think> if present. Mirrors extract_reasoning.
        """
        if self._identity_parser is not None:
            return 0
        end_idx = None
        for i, t in enumerate(token_ids):
            if t == self._end_token_id or (
                self._tool_section_start_token_id is not None
                and t == self._tool_section_start_token_id
            ):
                end_idx = i
                break
        ids = list(token_ids)
        region = ids[:end_idx] if end_idx is not None else ids
        return sum(1 for t in region if t != self._start_token_id)

'''

marker = "    def extract_reasoning(\n"
if marker not in s:
    print("ERROR: insertion marker not found"); sys.exit(1)
s = s.replace(marker, method + marker, 1)
open(f, "w").write(s)
print("patched OK")
