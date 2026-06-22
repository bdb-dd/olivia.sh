"""Minimal stdlib client for the Anthropic ``/v1/messages`` endpoint exposed by
``anthropic_proxy.py``. No third-party deps (urllib + json), so it runs on a
bare cluster login node just like ``bench_sweep.py``.

Two call modes:

* non-streaming — POST and parse the JSON response body;
* streaming — POST with ``stream:true`` and parse the SSE frames into a list of
  ``(event_name, data_dict)`` for :func:`conformance.validate_sse`.

The streaming read uses a per-recv socket timeout as an *inter-token stall*
guard (same idea as ``bench_serving.py --stall``): a wedged stream errors fast
instead of hanging the whole L0 run.
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import socket
import time
import urllib.error
import urllib.request


@dataclass
class CallResult:
    ok: bool
    mode: str                       # "nonstream" | "stream"
    status: Optional[int] = None
    response: Optional[dict] = None     # parsed Anthropic message (nonstream)
    events: Optional[list] = None       # list[(name, data)] (stream)
    error: Optional[str] = None
    latency_s: float = 0.0
    ttft_s: Optional[float] = None      # stream only: time to first content event


def _build_request(base_url: str, body: dict, version: str, api_key: str):
    data = json.dumps(body).encode("utf-8")
    return urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=data,
        method="POST",
        headers={
            "content-type": "application/json",
            "anthropic-version": version,
            "x-api-key": api_key,
        },
    )


def call_nonstream(base_url: str, body: dict, *, timeout: float = 600.0,
                   version: str = "2023-06-01", api_key: str = "sk-local-eval") -> CallResult:
    b = dict(body, stream=False)
    req = _build_request(base_url, b, version, api_key)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
        resp = json.loads(raw)
        return CallResult(True, "nonstream", status=status, response=resp,
                          latency_s=time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        return CallResult(False, "nonstream", status=e.code,
                          error=f"HTTP {e.code}: {detail}",
                          latency_s=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 — surface any transport failure as a result
        return CallResult(False, "nonstream", error=f"{type(e).__name__}: {e}",
                          latency_s=time.perf_counter() - t0)


def _parse_sse(line_iter):
    """Yield (event_name, data_dict) from an SSE byte-line iterator. The proxy
    emits ``event: <name>\\ndata: <json>\\n\\n`` frames."""
    event_name = None
    for raw in line_iter:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            event_name = None
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = line[len("data:"):].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"_unparsed": payload}
            yield (event_name or data.get("type"), data)


def call_stream(base_url: str, body: dict, *, stall: float = 45.0,
                connect_timeout: float = 30.0, version: str = "2023-06-01",
                api_key: str = "sk-local-eval") -> CallResult:
    b = dict(body, stream=True)
    req = _build_request(base_url, b, version, api_key)
    t0 = time.perf_counter()
    events: list = []
    ttft = None
    try:
        resp = urllib.request.urlopen(req, timeout=connect_timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        return CallResult(False, "stream", status=e.code,
                          error=f"HTTP {e.code}: {detail}",
                          latency_s=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        return CallResult(False, "stream", error=f"{type(e).__name__}: {e}",
                          latency_s=time.perf_counter() - t0)

    # Per-recv socket timeout == inter-token stall budget: a stalled stream
    # raises socket.timeout after `stall` seconds of silence.
    try:
        fp = resp.fp  # underlying socket file object
        sock = fp.raw._sock if hasattr(fp, "raw") else None
        if sock is not None:
            sock.settimeout(stall)
    except Exception:  # noqa: BLE001 — best-effort; fall back to no per-recv timeout
        pass

    try:
        with resp:
            for name, data in _parse_sse(resp):
                if ttft is None and name in ("content_block_start", "content_block_delta"):
                    ttft = time.perf_counter() - t0
                events.append((name, data))
                if name == "message_stop":
                    break
        return CallResult(True, "stream", status=200, events=events,
                          latency_s=time.perf_counter() - t0, ttft_s=ttft)
    except socket.timeout:
        return CallResult(False, "stream", status=200, events=events,
                          error=f"stall: no token for {stall}s "
                                f"(got {len(events)} events)",
                          latency_s=time.perf_counter() - t0, ttft_s=ttft)
    except Exception as e:  # noqa: BLE001
        return CallResult(False, "stream", status=200, events=events,
                          error=f"{type(e).__name__}: {e}",
                          latency_s=time.perf_counter() - t0, ttft_s=ttft)
