"""Load L0 protocol fixtures from JSON.

A fixture is a single ``/v1/messages`` request plus the behavioural intent we
expect of a competent agent. Shape::

    {
      "id": "weather-single-tool",
      "category": "tool-selection",
      "description": "one obviously-needed tool → must call it with valid args",
      "request": {
        "system": "...",                         # optional
        "messages": [ {role, content}, ... ],    # Anthropic message blocks
        "tools": [ {name, description, input_schema}, ... ],
        "max_tokens": 512,                        # optional (default 512)
        "temperature": 0                          # optional (default 0)
      },
      "expect": [ {check spec}, ... ],            # see conformance.check_expectations
      "modes": ["nonstream", "stream"],           # optional override
      "requires_thinking": false                  # optional: skip unless --think on
    }

Invariant checks run on every fixture automatically; ``expect`` only declares
the *behavioural* intent.
"""
from dataclasses import dataclass, field
from typing import Optional
import glob
import json
import os


@dataclass
class Fixture:
    id: str
    category: str
    description: str
    request: dict
    expect: list = field(default_factory=list)
    modes: Optional[list] = None
    requires_thinking: bool = False
    path: str = ""

    @property
    def tools(self) -> list:
        return self.request.get("tools") or []


_REQUIRED = ("id", "request")


def _validate(raw: dict, path: str) -> None:
    for k in _REQUIRED:
        if k not in raw:
            raise ValueError(f"{path}: fixture missing required key '{k}'")
    if not isinstance(raw["request"].get("messages"), list):
        raise ValueError(f"{path}: request.messages must be a list")


def load_fixtures(fixtures_dir: str) -> list[Fixture]:
    out: list[Fixture] = []
    for path in sorted(glob.glob(os.path.join(fixtures_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _validate(raw, path)
        out.append(Fixture(
            id=raw["id"],
            category=raw.get("category", "uncategorized"),
            description=raw.get("description", ""),
            request=raw["request"],
            expect=raw.get("expect", []),
            modes=raw.get("modes"),
            requires_thinking=bool(raw.get("requires_thinking", False)),
            path=path,
        ))
    ids = [fx.id for fx in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate fixture ids: {sorted(dupes)}")
    return out
