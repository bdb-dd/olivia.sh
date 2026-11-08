#!/usr/bin/env python3
"""Needle-in-a-haystack probe: does a long context actually WORK, or just run?

Throughput sweeps only prove a server accepts N tokens and emits tokens back.
They say nothing about whether positions past the trained window are encoded
correctly — a broken RoPE/YaRN setup decodes at full speed and answers
nonsense. This plants a random passcode near the START of a long filler prompt
and asks for it at the END, so a correct answer requires attending across the
whole window.

Stdlib only (runs on the login node). One line of CSV per target length:
    target_tokens,prompt_tokens,found,seconds,answer
"""
import argparse
import json
import random
import string
import time
import urllib.request

FILLER = [
    "The archive kept its ledgers in long rows, each entry dated and initialled.",
    "Rain moved across the fjord in slow grey curtains through the afternoon.",
    "A clerk copied the shipping manifests twice so that neither copy could drift.",
    "The lighthouse keeper logged wind speed, visibility and passing vessels hourly.",
    "Engineers measured the bridge cables every spring for creep and corrosion.",
    "In the library the catalogue cards were sorted by author and then by year.",
    "The ferry schedule changed with the seasons but the first sailing never did.",
    "Surveyors walked the boundary line and drove a marker at every bend.",
]


def build(target_tokens, cpt, code):
    needle = f"IMPORTANT: the secret passcode is {code}. Remember it exactly."
    rng = random.Random(target_tokens)
    parts, chars = [], 0
    budget = int(target_tokens * cpt)
    while chars < budget:
        s = rng.choice(FILLER)
        parts.append(s)
        chars += len(s) + 1
    # needle at ~2% depth: deep enough to be past any attention sink, far
    # enough from the question that the whole window has to be spanned
    parts.insert(max(1, len(parts) // 50), needle)
    return " ".join(parts) + (
        "\n\nQuestion: what is the secret passcode stated near the beginning of "
        "this text? Reply with the passcode only."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="32768,200000")
    ap.add_argument("--chars-per-token", type=float, default=5.48)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    print("target_tokens,prompt_tokens,found,seconds,answer")
    for t in [int(x) for x in a.targets.split(",")]:
        code = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": build(t, a.chars_per_token, code)}],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            a.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                resp = json.load(r)
            ans = (resp["choices"][0]["message"].get("content") or "").strip()
            ptok = resp.get("usage", {}).get("prompt_tokens", -1)
            found = code in ans
        except Exception as e:  # report, don't die: the next length may still work
            ans, ptok, found = f"ERROR {type(e).__name__}: {e}"[:200], -1, False
        ans = ans.replace("\n", " ").replace(",", ";")[:120]
        print(f"{t},{ptok},{found},{time.time()-t0:.1f},{ans}", flush=True)


if __name__ == "__main__":
    main()
