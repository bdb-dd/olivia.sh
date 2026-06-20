#!/usr/bin/env python3
"""Offline self-test for the L0 conformance checkers.

Proves the checker *logic* (invariants, expectations, JSON-schema validator, SSE
grammar validator + reconstruction) classifies synthetic good/bad responses
correctly — with NO GPU, NO proxy, NO network. Run this before trusting a live
scorecard, and after any change to :mod:`conformance`.

    python evals/protocol/selftest.py      # or: python -m evals.protocol.selftest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.protocol import conformance as C  # noqa: E402

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}
TOOLS = [WEATHER_TOOL]

_passed = 0
_failed = 0


def check(cond, label):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  SELFTEST FAIL: {label}")


def fails(checks):
    return [c.name for c in checks if not c.passed]


def msg(content, stop_reason, usage=None):
    return {"type": "message", "role": "assistant", "content": content,
            "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5}}


# --- non-stream invariants + expectations ---------------------------------- #

def test_good_tool_call():
    r = msg([{"type": "tool_use", "id": "t1", "name": "get_weather",
              "input": {"location": "Paris"}}], "tool_use")
    check(fails(C.check_invariants(r, TOOLS)) == [], "good tool call: invariants clean")
    exp = [{"check": "calls_tool", "tool": "get_weather"},
           {"check": "tool_args_valid", "tool": "get_weather"},
           {"check": "tool_args_include", "tool": "get_weather", "keys": ["location"]}]
    check(fails(C.check_expectations(r, TOOLS, exp)) == [], "good tool call: expectations pass")


def test_hallucinated_tool():
    r = msg([{"type": "tool_use", "id": "t1", "name": "search_web", "input": {}}], "tool_use")
    check("tool_use[0]_name_offered" in fails(C.check_invariants(r, TOOLS)),
          "hallucinated tool name flagged by invariant")


def test_stop_reason_mismatch():
    r = msg([{"type": "tool_use", "id": "t1", "name": "get_weather",
              "input": {"location": "Paris"}}], "end_turn")
    check("stop_reason_matches_tool_use" in fails(C.check_invariants(r, TOOLS)),
          "tool_use with stop_reason!=tool_use flagged")


def test_bad_args_but_structurally_ok():
    r = msg([{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}], "tool_use")
    check(fails(C.check_invariants(r, TOOLS)) == [],
          "missing-required is not an invariant failure (structure is fine)")
    exp = [{"check": "tool_args_valid", "tool": "get_weather"},
           {"check": "tool_args_include", "tool": "get_weather", "keys": ["location"]}]
    f = fails(C.check_expectations(r, TOOLS, exp))
    check("tool_args_valid:get_weather" in f and "tool_args_include:get_weather" in f,
          "missing-required flagged by expectations")


def test_no_tool_good():
    r = msg([{"type": "text", "text": "The answer is 68."}], "end_turn")
    check(fails(C.check_invariants(r, TOOLS)) == [], "text answer: invariants clean")
    exp = [{"check": "no_tool_call"}, {"check": "final_text_nonempty"},
           {"check": "text_contains", "text": "68"}]
    check(fails(C.check_expectations(r, TOOLS, exp)) == [], "text answer: expectations pass")


def test_unknown_block_type():
    r = msg([{"type": "image", "source": {}}], "end_turn")
    check("block[0]_known_type" in fails(C.check_invariants(r, TOOLS)),
          "unknown block type flagged")


def test_distractor_does_not_call():
    r = msg([{"type": "tool_use", "id": "t1", "name": "get_weather",
              "input": {"location": "Tokyo"}}], "tool_use")
    exp = [{"check": "does_not_call", "tool": "send_email"}]
    check(fails(C.check_expectations(r, TOOLS, exp)) == [], "does_not_call passes when not called")


# --- JSON schema validator -------------------------------------------------- #

def test_schema_validator():
    s = WEATHER_TOOL["input_schema"]
    check(C.validate_schema({"location": "Paris"}, s) == [], "schema: valid minimal")
    check(C.validate_schema({}, s) != [], "schema: missing required rejected")
    check(C.validate_schema({"location": 5}, s) != [], "schema: wrong type rejected")
    check(C.validate_schema({"location": "P", "unit": "kelvin"}, s) != [],
          "schema: bad enum rejected")
    check(C.validate_schema({"location": "P", "unit": "celsius"}, s) == [],
          "schema: good enum accepted")
    nested = {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "integer"}}}}
    check(C.validate_schema({"items": [1, 2, 3]}, nested) == [], "schema: array of int ok")
    check(C.validate_schema({"items": [1, "x"]}, nested) != [], "schema: array bad element rejected")


# --- streaming SSE grammar + reconstruction --------------------------------- #

def good_tool_stream():
    return [
        ("message_start", {"type": "message_start",
                           "message": {"role": "assistant", "usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block":
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}}),
        ("content_block_delta", {"index": 0, "delta":
            {"type": "input_json_delta", "partial_json": '{"location":'}}),
        ("content_block_delta", {"index": 0, "delta":
            {"type": "input_json_delta", "partial_json": ' "Paris"}'}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def test_stream_good():
    sr = C.validate_sse(good_tool_stream())
    check(fails(sr.checks) == [], "good tool stream: grammar clean")
    check(sr.response["content"][0]["input"] == {"location": "Paris"},
          "good tool stream: reconstructed args")
    check(sr.response["stop_reason"] == "tool_use", "good tool stream: stop_reason")
    check(fails(C.check_invariants(sr.response, TOOLS)) == [],
          "reconstructed response: invariants clean")
    exp = [{"check": "calls_tool", "tool": "get_weather"},
           {"check": "tool_args_valid", "tool": "get_weather"}]
    check(fails(C.check_expectations(sr.response, TOOLS, exp)) == [],
          "reconstructed response: expectations pass")


def test_stream_thinking_then_text_order():
    events = [
        ("message_start", {"type": "message_start", "message": {"role": "assistant", "usage": {"input_tokens": 3}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "Answer: 68"}}),
        ("content_block_stop", {"index": 1}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    sr = C.validate_sse(events)
    check(fails(sr.checks) == [], "thinking->text stream: grammar clean")
    exp = [{"check": "block_order", "order": ["thinking", "text"]},
           {"check": "text_contains", "text": "68"}]
    check(fails(C.check_expectations(sr.response, TOOLS, exp)) == [],
          "thinking->text stream: block_order + text pass")


def test_stream_bad_index():
    ev = good_tool_stream()
    ev[1] = ("content_block_start", {"index": 1, "content_block":
             {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}})
    check("cb_start_index_contiguous" in fails(C.validate_sse(ev).checks),
          "non-contiguous block index flagged")


def test_stream_delta_type_mismatch():
    ev = [
        ("message_start", {"message": {"usage": {"input_tokens": 1}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    check("delta_type_matches_block" in fails(C.validate_sse(ev).checks),
          "delta type not matching block flagged")


def test_stream_bad_tool_json():
    ev = good_tool_stream()
    ev[2] = ("content_block_delta", {"index": 0, "delta":
             {"type": "input_json_delta", "partial_json": '{"location":'}})
    ev[3] = ("content_block_delta", {"index": 0, "delta":
             {"type": "input_json_delta", "partial_json": ' Paris'}})  # invalid JSON
    check("tool_use_partial_json_parses" in fails(C.validate_sse(ev).checks),
          "unparseable accumulated tool JSON flagged")


def test_stream_overlap():
    ev = [
        ("message_start", {"message": {"usage": {"input_tokens": 1}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_stop", {"index": 1}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    check("cb_start_no_overlap" in fails(C.validate_sse(ev).checks),
          "overlapping (unclosed) blocks flagged")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"L0 conformance selftest — {len(tests)} test groups")
    for t in tests:
        t()
    print(f"\nselftest: {_passed} checks passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
