"""L0 protocol-conformance checkers — pure, zero-dependency.

Given a *normalized Anthropic Messages response* (the dict ``anthropic_proxy.py``
returns from ``POST /v1/messages``) plus the fixture's tool set and declared
behavioural expectations, produce a flat list of :class:`Check` results.

Two classes of check:

* **invariant** — must hold for ANY non-broken model+proxy. A failure is a
  *regression* (proxy bug or hard protocol violation), and breaks the run's
  "invariants clean" gate. These are what make this suite double as a proxy
  regression test.
* **expect** — the fixture's behavioural *intent* (did the model pick the right
  tool, stop when it should, fill valid args, ...). Model-dependent, so these
  are reported as a pass *rate*, never a hard gate.

The streaming path (:func:`validate_sse`) checks the proxy's SSE event grammar
— the ``thinking → text → tool_use`` block state machine that is the single
most fragile surface in the bridge — and reconstructs a response dict so the
same content-level invariants/expectations run against streamed output too.

No third-party deps on purpose: this runs on a bare cluster login node, same as
``bench_sweep.py``.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import json

# Anthropic stop_reason values the proxy is allowed to emit (see
# anthropic_proxy.py:_finish_to_stop_reason).
STOP_REASONS = {"end_turn", "max_tokens", "tool_use", "stop_sequence"}
# The only content-block types the proxy emits.
BLOCK_TYPES = {"thinking", "text", "tool_use"}
# delta.type expected inside content_block_delta, per block type.
DELTA_FOR_BLOCK = {
    "text": "text_delta",
    "thinking": "thinking_delta",
    "tool_use": "input_json_delta",
}


@dataclass
class Check:
    name: str
    kind: str  # "invariant" | "expect"
    passed: bool
    detail: str = ""


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (subset used by tool input_schemas)           #
# --------------------------------------------------------------------------- #

def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    return True  # unknown type keyword → don't fail on it


def validate_schema(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate ``instance`` against a tool ``input_schema``. Returns a list of
    human-readable error strings (empty == valid). Deliberately covers only the
    keywords that show up in real tool schemas: type, properties, required,
    enum, array items, minItems. Unknown keywords are ignored, not errors."""
    errs: list[str] = []
    if not isinstance(schema, dict):
        return errs

    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(instance, tt) for tt in types):
            got = type(instance).__name__
            errs.append(f"{path}: expected type {t}, got {got}")
            return errs  # further checks are meaningless once the type is wrong

    enum = schema.get("enum")
    if enum is not None and instance not in enum:
        errs.append(f"{path}: {instance!r} not in enum {enum}")

    if isinstance(instance, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in instance:
                errs.append(f"{path}: missing required '{req}'")
        for key, subschema in props.items():
            if key in instance:
                errs.extend(validate_schema(instance[key], subschema, f"{path}.{key}"))

    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(instance):
                errs.extend(validate_schema(el, items, f"{path}[{i}]"))
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            errs.append(f"{path}: array len {len(instance)} < minItems {min_items}")

    return errs


# --------------------------------------------------------------------------- #
# Response accessors                                                           #
# --------------------------------------------------------------------------- #

def _content(resp: dict) -> list:
    c = resp.get("content")
    return c if isinstance(c, list) else []


def tool_uses(resp: dict) -> list[dict]:
    return [b for b in _content(resp)
            if isinstance(b, dict) and b.get("type") == "tool_use"]


def text_of(resp: dict) -> str:
    return "".join(b.get("text", "") for b in _content(resp)
                   if isinstance(b, dict) and b.get("type") == "text")


def block_types(resp: dict) -> list[str]:
    return [b.get("type") for b in _content(resp) if isinstance(b, dict)]


def tool_names(tools: Optional[list]) -> set:
    return {t.get("name") for t in (tools or []) if isinstance(t, dict)}


def _schema_for(tools: Optional[list], name: str) -> dict:
    for t in tools or []:
        if isinstance(t, dict) and t.get("name") == name:
            return t.get("input_schema") or {}
    return {}


# --------------------------------------------------------------------------- #
# Invariants — must hold for any non-broken model + proxy                      #
# --------------------------------------------------------------------------- #

def check_invariants(resp: Any, tools: Optional[list]) -> list[Check]:
    out: list[Check] = []

    def inv(name, passed, detail=""):
        out.append(Check(name, "invariant", bool(passed), detail))

    if not isinstance(resp, dict):
        inv("response_is_object", False, f"got {type(resp).__name__}")
        return out
    inv("response_is_object", True)
    inv("type_message", resp.get("type") == "message", repr(resp.get("type")))
    inv("role_assistant", resp.get("role") == "assistant", repr(resp.get("role")))

    content = resp.get("content")
    inv("content_is_list", isinstance(content, list),
        f"got {type(content).__name__}")

    for i, b in enumerate(content or []):
        bt = b.get("type") if isinstance(b, dict) else None
        inv(f"block[{i}]_known_type", bt in BLOCK_TYPES, repr(bt))

    sr = resp.get("stop_reason")
    inv("stop_reason_valid", sr in STOP_REASONS, repr(sr))

    usage = resp.get("usage") or {}
    inv("usage_tokens_int",
        isinstance(usage.get("input_tokens"), int)
        and isinstance(usage.get("output_tokens"), int),
        f"usage={usage}")

    names = tool_names(tools)
    tus = tool_uses(resp)
    seen_ids: set = set()
    for j, tu in enumerate(tus):
        tid = tu.get("id")
        inv(f"tool_use[{j}]_id_nonempty", isinstance(tid, str) and bool(tid), repr(tid))
        inv(f"tool_use[{j}]_id_unique", tid not in seen_ids, repr(tid))
        seen_ids.add(tid)
        nm = tu.get("name")
        inv(f"tool_use[{j}]_name_offered", isinstance(nm, str) and nm in names,
            f"{nm!r} not in {sorted(n for n in names if n)}")
        inv(f"tool_use[{j}]_input_object", isinstance(tu.get("input"), dict),
            f"input is {type(tu.get('input')).__name__}")

    # Anthropic contract: tool_use blocks ⇔ stop_reason == "tool_use".
    if tus:
        inv("stop_reason_matches_tool_use", sr == "tool_use",
            f"have {len(tus)} tool_use but stop_reason={sr!r}")
    if sr == "tool_use":
        inv("tool_use_present_for_stop", len(tus) >= 1,
            "stop_reason=tool_use but no tool_use block")

    return out


# --------------------------------------------------------------------------- #
# Expectations — the fixture's behavioural intent (model-dependent)           #
# --------------------------------------------------------------------------- #

def check_expectations(resp: dict, tools: Optional[list],
                       expects: Optional[list]) -> list[Check]:
    out: list[Check] = []

    def exp(name, passed, detail=""):
        out.append(Check(name, "expect", bool(passed), detail))

    tus = tool_uses(resp)

    for e in expects or []:
        c = e.get("check")

        if c == "calls_tool":
            name = e["tool"]
            exp(f"calls_tool:{name}", any(tu.get("name") == name for tu in tus))

        elif c == "calls_one_of":
            allowed = set(e["tools"])
            hit = [tu.get("name") for tu in tus if tu.get("name") in allowed]
            exp(f"calls_one_of:{sorted(allowed)}", bool(hit), f"called={hit}")

        elif c == "no_tool_call":
            exp("no_tool_call", not tus,
                f"called {[tu.get('name') for tu in tus]}")

        elif c == "does_not_call":
            name = e["tool"]
            exp(f"does_not_call:{name}", all(tu.get("name") != name for tu in tus))

        elif c == "tool_args_valid":
            name = e["tool"]
            schema = _schema_for(tools, name)
            calls = [tu for tu in tus if tu.get("name") == name]
            if not calls:
                exp(f"tool_args_valid:{name}", False, "tool not called")
            else:
                errs: list[str] = []
                for tu in calls:
                    errs += validate_schema(tu.get("input"), schema)
                exp(f"tool_args_valid:{name}", not errs, "; ".join(errs))

        elif c == "tool_args_include":
            name, keys = e["tool"], e["keys"]
            calls = [tu for tu in tus if tu.get("name") == name]
            missing = [k for tu in calls for k in keys
                       if k not in (tu.get("input") or {})]
            exp(f"tool_args_include:{name}", bool(calls) and not missing,
                f"missing={missing}" if calls else "tool not called")

        elif c == "tool_args_equal":
            name, want = e["tool"], e["args"]
            calls = [tu for tu in tus if tu.get("name") == name]
            ok = bool(calls) and all(
                (tu.get("input") or {}).get(k) == v
                for tu in calls for k, v in want.items())
            exp(f"tool_args_equal:{name}", ok,
                f"got={[tu.get('input') for tu in calls]}")

        elif c == "block_order":
            order = e["order"]
            bts = block_types(resp)
            idx = {t: bts.index(t) for t in order if t in bts}
            present = [t for t in order if t in idx]
            ordered = all(idx[present[i]] < idx[present[i + 1]]
                          for i in range(len(present) - 1))
            if e.get("require_all"):
                ordered = ordered and all(t in idx for t in order)
            exp(f"block_order:{'<'.join(order)}", ordered, f"blocks={bts}")

        elif c == "text_contains":
            subs = e.get("substrings") or ([e["text"]] if "text" in e else [])
            ci = e.get("case_insensitive", True)
            mode = e.get("mode", "any")
            hay = text_of(resp)
            hay_cmp = hay.lower() if ci else hay
            probe = [(s.lower() if ci else s) for s in subs]
            fn = all if mode == "all" else any
            exp("text_contains", bool(probe) and fn(s in hay_cmp for s in probe),
                f"text[:80]={hay[:80]!r}")

        elif c == "final_text_nonempty":
            exp("final_text_nonempty", bool(text_of(resp).strip()))

        else:
            exp(f"unknown_check:{c}", False, "unknown check type")

    return out


# --------------------------------------------------------------------------- #
# Streaming — validate the SSE event grammar and reconstruct the response      #
# --------------------------------------------------------------------------- #

@dataclass
class StreamResult:
    checks: list[Check] = field(default_factory=list)
    response: dict = field(default_factory=dict)  # reconstructed, for content checks


def validate_sse(events: list[tuple]) -> StreamResult:
    """Validate the Anthropic SSE event grammar emitted by the proxy and
    reconstruct an equivalent response dict.

    ``events`` is a list of ``(event_name, data_dict)`` in arrival order.
    Grammar enforced (each a proxy regression invariant):

    * first event ``message_start``, last ``message_stop``;
    * blocks open/stop in contiguous, non-overlapping index order;
    * a block's deltas carry the delta type matching its block type;
    * a ``tool_use`` block's concatenated ``partial_json`` parses to an object;
    * a ``message_delta`` carrying ``stop_reason`` precedes ``message_stop``.
    """
    out: list[Check] = []

    def inv(name, passed, detail=""):
        out.append(Check(name, "invariant", bool(passed), detail))

    names = [n for n, _ in events]
    inv("sse_starts_message_start", bool(events) and names[0] == "message_start",
        names[:1])
    inv("sse_ends_message_stop", bool(events) and names[-1] == "message_stop",
        names[-1:])

    reconstructed: list[dict] = []
    open_index: Optional[int] = None
    open_type: Optional[str] = None
    buf = ""                       # text/thinking/json accumulator for open block
    cur_meta: dict = {}            # id/name for an open tool_use block
    expected_index = 0
    saw_message_delta = False
    stop_reason = None
    in_tokens = 0
    out_tokens = 0

    for nm, data in events:
        if nm == "ping":
            continue

        if nm == "message_start":
            msg = (data.get("message") or {})
            in_tokens = (msg.get("usage") or {}).get("input_tokens", 0) or 0

        elif nm == "content_block_start":
            inv("cb_start_no_overlap", open_index is None,
                f"index {data.get('index')} opened while {open_index} still open")
            inv("cb_start_index_contiguous", data.get("index") == expected_index,
                f"got {data.get('index')}, expected {expected_index}")
            open_index = data.get("index")
            expected_index += 1
            block = data.get("content_block") or {}
            open_type = block.get("type")
            inv("cb_start_type_known", open_type in BLOCK_TYPES, repr(open_type))
            buf = ""
            cur_meta = {"id": block.get("id"), "name": block.get("name")} \
                if open_type == "tool_use" else {}

        elif nm == "content_block_delta":
            inv("delta_index_matches_open", data.get("index") == open_index,
                f"delta idx {data.get('index')} vs open {open_index}")
            delta = data.get("delta") or {}
            dt = delta.get("type")
            want_dt = DELTA_FOR_BLOCK.get(open_type or "")
            inv("delta_type_matches_block", dt == want_dt,
                f"{dt!r} on {open_type!r} block (want {want_dt!r})")
            if dt == "text_delta":
                buf += delta.get("text", "")
            elif dt == "thinking_delta":
                buf += delta.get("thinking", "")
            elif dt == "input_json_delta":
                buf += delta.get("partial_json", "")

        elif nm == "content_block_stop":
            inv("cb_stop_matches_open", data.get("index") == open_index,
                f"stop idx {data.get('index')} vs open {open_index}")
            if open_type == "text":
                reconstructed.append({"type": "text", "text": buf})
            elif open_type == "thinking":
                reconstructed.append({"type": "thinking", "thinking": buf})
            elif open_type == "tool_use":
                try:
                    parsed = json.loads(buf or "{}")
                    parse_ok = isinstance(parsed, dict)
                except json.JSONDecodeError:
                    parsed, parse_ok = {}, False
                inv("tool_use_partial_json_parses", parse_ok,
                    f"accumulated={buf[:120]!r}")
                reconstructed.append({
                    "type": "tool_use",
                    "id": cur_meta.get("id"),
                    "name": cur_meta.get("name"),
                    "input": parsed if isinstance(parsed, dict) else {},
                })
            open_index, open_type, buf, cur_meta = None, None, "", {}

        elif nm == "message_delta":
            saw_message_delta = True
            stop_reason = (data.get("delta") or {}).get("stop_reason", stop_reason)
            out_tokens = (data.get("usage") or {}).get("output_tokens", out_tokens) or out_tokens

        elif nm == "message_stop":
            inv("no_open_block_at_message_stop", open_index is None,
                f"block {open_index} still open at message_stop")

    inv("saw_message_delta", saw_message_delta, "no message_delta event")

    response = {
        "type": "message",
        "role": "assistant",
        "content": reconstructed,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": int(in_tokens), "output_tokens": int(out_tokens)},
    }
    return StreamResult(checks=out, response=response)
