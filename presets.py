#!/usr/bin/env python3
"""
Preset registry — single source of truth shared by bash and Python
==================================================================

Loads ``presets.json`` and exposes the model-preset table to:

- ``olivia.sh`` (via the small CLI below — ``normalize`` / ``field`` /
  ``shellvars``), so the bash ``preset_field`` / ``normalize_preset`` accessors
  delegate here instead of duplicating a giant ``case`` statement; and
- ``model_router.py`` (via ``import presets``), so the router can resolve an
  incoming request's ``model`` field to a preset and thus to a backend.

Stdlib only (no third-party deps) so it runs unchanged on a laptop, the login
node, and a CPU ``small``-partition node.

Semantics intentionally mirror the old olivia.sh accessors:

- ``normalize(name)`` lower-cases the input and maps any alias (or canonical
  name) to its canonical name; an unrecognised name is returned lower-cased
  (a "custom" preset), exactly like the old ``normalize_preset`` ``*)`` branch.
- ``resolve(name)`` returns a fully-populated :class:`Preset`. For an unknown
  name it synthesises one from ``defaults`` with ``container_prefix`` set to the
  *original* (case-preserved) argument — matching the old ``preset_field`` ``*)``
  branch, which kept ``MODEL_ID``'s case for the container name.

CLI::

    presets.py normalize <name>            # canonical name (or lower-cased input)
    presets.py field <name> <field>        # model|served_name|nodes|gpus|pp|resources|prefix|index|port|description
    presets.py shellvars <name>            # PRESET_*=... lines for `eval` in bash
    presets.py container <name> [index]    # vllm-<prefix>-<index>-sandbox
    presets.py list [--aliases]            # canonical preset names
    presets.py match <model-string>        # canonical preset a request `model` routes to (empty if none)
    presets.py json                        # dump the resolved table
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PRESETS_JSON = os.environ.get(
    "OLIVIA_PRESETS_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json"),
)


@dataclass
class Preset:
    canonical: str
    aliases: list = field(default_factory=list)
    model: str = ""
    served_model_name: Optional[str] = None
    nodes: int = 1
    gpus: int = 4
    pp: int = 1
    container_prefix: str = ""
    index: int = 1
    port: int = 8000
    description: str = ""
    known: bool = True  # False for synthesised custom presets

    @property
    def served_name(self) -> str:
        """The name vLLM serves this model under (what clients send as `model`)."""
        return self.served_model_name or self.model

    def container_name(self, index: Optional[int] = None) -> str:
        idx = self.index if index is None else index
        return f"vllm-{self.container_prefix}-{idx}-sandbox"

    def route_keys(self) -> list:
        """All strings that should route a request to this preset (lower-cased)."""
        keys = [self.canonical, *self.aliases]
        if self.model:
            keys.append(self.model)
        if self.served_model_name:
            keys.append(self.served_model_name)
        return [k.lower() for k in keys if k]


class Registry:
    def __init__(self, path: str = PRESETS_JSON):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.defaults = raw.get("defaults", {})
        self._presets: dict = {}
        for canonical, spec in raw.get("presets", {}).items():
            self._presets[canonical] = Preset(
                canonical=canonical,
                aliases=list(spec.get("aliases", [])),
                model=spec.get("model", ""),
                served_model_name=spec.get("served_model_name"),
                nodes=int(spec.get("nodes", self.defaults.get("nodes", 1))),
                gpus=int(spec.get("gpus", self.defaults.get("gpus", 4))),
                pp=int(spec.get("pp", self.defaults.get("pp", 1))),
                container_prefix=spec.get("container_prefix", canonical),
                index=int(spec.get("index", self.defaults.get("index", 1))),
                port=int(spec.get("port", self.defaults.get("port", 8000))),
                description=spec.get("description", ""),
            )
        # Lower-cased alias/canonical -> canonical, for normalize().
        self._alias_index: dict = {}
        for p in self._presets.values():
            self._alias_index[p.canonical.lower()] = p.canonical
            for a in p.aliases:
                self._alias_index[a.lower()] = p.canonical

    def normalize(self, name: str) -> str:
        return self._alias_index.get(name.lower(), name.lower())

    def resolve(self, name: str) -> Preset:
        canonical = self.normalize(name)
        if canonical in self._presets:
            return self._presets[canonical]
        # Unknown/custom preset: mirror preset_field's `*)` branch — keep the
        # original (case-preserved) name as the container prefix so it matches
        # build_vllm_gh200.sh's MODEL_ID=<name>.
        d = self.defaults
        return Preset(
            canonical=canonical,
            container_prefix=name,
            nodes=int(d.get("nodes", 1)),
            gpus=int(d.get("gpus", 4)),
            pp=int(d.get("pp", 1)),
            index=int(d.get("index", 1)),
            port=int(d.get("port", 8000)),
            known=False,
        )

    def all(self) -> list:
        return list(self._presets.values())

    def match_model(self, model_string: str) -> Optional[Preset]:
        """Resolve a request's `model` field to a preset.

        Matches (case-insensitively) the canonical name, any alias, the model
        repo id, or an explicit served_model_name. Returns None if nothing
        matches — the router turns that into a clear 404 rather than guessing.
        """
        if not model_string:
            return None
        key = model_string.strip().lower()
        for p in self._presets.values():
            if key in p.route_keys():
                return p
        return None


# Module-level convenience (lazy singleton).
_REGISTRY: Optional[Registry] = None


def registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


def resolve(name: str) -> Preset:
    return registry().resolve(name)


def normalize(name: str) -> str:
    return registry().normalize(name)


def match_model(model_string: str) -> Optional[Preset]:
    return registry().match_model(model_string)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _field_value(p: Preset, fname: str) -> str:
    mapping = {
        "model": p.model,
        "served_name": p.served_name,
        "nodes": p.nodes,
        "gpus": p.gpus,
        "pp": p.pp,
        "resources": f"{p.nodes} {p.gpus} {p.pp}",
        "prefix": p.container_prefix,
        "index": p.index,
        "port": p.port,
        "description": p.description,
        "canonical": p.canonical,
        "container": p.container_name(),
    }
    return "" if fname not in mapping else str(mapping[fname])


def _cmd_shellvars(p: Preset) -> str:
    # Emit values safe for `eval` in bash (single-quote, escape embedded quotes).
    def q(v):
        return "'" + str(v).replace("'", "'\\''") + "'"
    lines = [
        f"PRESET_CANONICAL={q(p.canonical)}",
        f"PRESET_MODEL={q(p.model)}",
        f"PRESET_SERVED_NAME={q(p.served_name)}",
        f"PRESET_NODES={q(p.nodes)}",
        f"PRESET_GPUS={q(p.gpus)}",
        f"PRESET_PP={q(p.pp)}",
        f"PRESET_PREFIX={q(p.container_prefix)}",
        f"PRESET_INDEX={q(p.index)}",
        f"PRESET_PORT={q(p.port)}",
        f"PRESET_CONTAINER={q(p.container_name())}",
        f"PRESET_KNOWN={q('1' if p.known else '0')}",
    ]
    return "\n".join(lines)


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    reg = registry()

    if cmd == "normalize":
        print(reg.normalize(rest[0]) if rest else "")
        return 0
    if cmd == "field":
        if len(rest) < 2:
            print("usage: presets.py field <name> <field>", file=sys.stderr)
            return 2
        print(_field_value(reg.resolve(rest[0]), rest[1]))
        return 0
    if cmd == "shellvars":
        if not rest:
            print("usage: presets.py shellvars <name>", file=sys.stderr)
            return 2
        print(_cmd_shellvars(reg.resolve(rest[0])))
        return 0
    if cmd == "container":
        if not rest:
            print("usage: presets.py container <name> [index]", file=sys.stderr)
            return 2
        idx = int(rest[1]) if len(rest) > 1 else None
        print(reg.resolve(rest[0]).container_name(idx))
        return 0
    if cmd == "list":
        show_aliases = "--aliases" in rest
        for p in reg.all():
            if show_aliases and p.aliases:
                print(f"{p.canonical}\t{', '.join(p.aliases)}")
            else:
                print(p.canonical)
        return 0
    if cmd == "match":
        if not rest:
            print("", end="")
            return 0
        p = reg.match_model(rest[0])
        print(p.canonical if p else "")
        return 0 if p else 1
    if cmd == "json":
        out = {
            p.canonical: {
                "aliases": p.aliases,
                "model": p.model,
                "served_name": p.served_name,
                "nodes": p.nodes, "gpus": p.gpus, "pp": p.pp,
                "container": p.container_name(),
                "port": p.port,
                "description": p.description,
            }
            for p in reg.all()
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
