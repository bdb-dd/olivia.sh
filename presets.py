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
    presets.py field <name> <field>        # model|served_name|nodes|gpus|pp|resources|prefix|index|port|storage|description
    presets.py shellvars <name>            # PRESET_*=... lines for `eval` in bash
    presets.py env <name>                  # KEY=VALUE server env defaults of a preset variant
    presets.py card <name>                 # model card: everything presets.json knows
    presets.py table                       # one line per preset: name, previous name, status
    presets.py lint                        # names match <family>-<version>-<params>-<ctx>[-Nn][-variant]
    presets.py container <name> [index]    # vllm-<prefix>-<index>-sandbox
    presets.py list [--aliases]            # canonical preset names
    presets.py match <model-string>        # canonical preset a request `model` routes to (empty if none)
    presets.py json                        # dump the resolved table
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

PRESETS_JSON = os.environ.get(
    "OLIVIA_PRESETS_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json"),
)


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ENV_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|_KEY$")


def _validated_env(canonical: str, env: dict) -> dict:
    """Reject anything that could not be a plain server knob.

    These values are echoed by olivia.sh and ride in the sbatch command line, so
    a credential must never be stored here — and a malformed name would only
    fail later, on the cluster, after a queue wait.
    """
    if not isinstance(env, dict):
        raise ValueError(f"preset {canonical!r}: 'env' must be an object")
    out = {}
    for k, v in env.items():
        if not _ENV_NAME.match(k):
            raise ValueError(f"preset {canonical!r}: bad env var name {k!r}")
        if _ENV_SECRET.search(k):
            raise ValueError(f"preset {canonical!r}: {k!r} looks like a credential")
        if not isinstance(v, (str, int, float)) or isinstance(v, bool):
            raise ValueError(f"preset {canonical!r}: {k!r} must be a string or number")
        out[k] = str(v)
    return out


# Model-card fields carried by a preset (all optional except for `lint`).
CARD_FIELDS = ("family", "version", "params", "active", "quant", "ctx",
               "variant", "node_suffix", "status", "perf")


def derived_name(card: dict, nodes: int) -> str:
    """<family>-<version>-<params>-<ctx>[-<N>n][-<variant>], lowercase.

    The node suffix appears on every multi-node preset, and on a single-node
    shape (`node_suffix: true`) of a model that also has a multi-node one — e.g.
    ornith-1.5-397b-256k-2n vs ...-1n. A behaviour variant adds a word
    (`-lowlat`, `-mtp`, `-prebuilt`); a context variant changes `ctx` instead.
    """
    parts = [card["family"]]
    if card.get("version"):
        parts.append(card["version"])
    parts += [card["params"], card["ctx"]]
    if nodes > 1 or card.get("node_suffix"):
        parts.append(f"{nodes}n")
    if card.get("variant"):
        parts.append(card["variant"])
    return "-".join(str(x) for x in parts).lower()


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
    storage: str = "projects"  # HF-cache tier: "projects" (persistent) or "work" (at-risk)
    description: str = ""
    # Server env defaults baked into a preset VARIANT (e.g. a 1M-context or a
    # low-latency shape of the same weights). olivia.sh exports each one only if
    # the caller has not set it, so an explicit `FOO=x ./olivia.sh ...` wins.
    env: dict = field(default_factory=dict)
    # Model-card fields (see CARD_FIELDS). The canonical name is DERIVED from
    # family/version/params/ctx/node-count/variant and `presets.py lint` enforces
    # it, so a name always says what the preset is.
    card: dict = field(default_factory=dict)
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
                storage=spec.get("storage", self.defaults.get("storage", "projects")),
                description=spec.get("description", ""),
                env=_validated_env(canonical, spec.get("env", {})),
                card={k: spec[k] for k in CARD_FIELDS if k in spec},
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
            storage=d.get("storage", "projects"),
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
        "storage": p.storage,
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


def _cmd_card(p: Preset) -> str:
    """Everything presets.json knows about a preset, as a short model card."""
    c = p.card
    shape = (f"{p.nodes} node(s) x {p.gpus} GPU" + ("s" if p.gpus != 1 else "")
             + (f", TP={p.gpus} + PP={p.pp}" if p.pp > 1 else f", TP={p.gpus}"))
    rows = [
        ("Preset", p.canonical + (f"   (aliases: {', '.join(p.aliases[:4])}{', ...' if len(p.aliases) > 4 else ''})" if p.aliases else "")),
        ("Model", p.model or "<none>"),
        ("Family", " ".join(x for x in (c.get("family", ""), c.get("version", "")) if x) or "—"),
        ("Parameters", c.get("params", "—") + (f" total, {c['active']} active" if c.get("active") else "")),
        ("Quantization", c.get("quant", "—")),
        ("Context", c.get("ctx", "—")),
        ("Shape", shape),
        ("Container", p.container_name()),
        ("Weights tier", p.storage),
        ("Status", c.get("status", "—")),
        ("Measured", c.get("perf", "—")),
    ]
    if p.env:
        rows.append(("Preset env", " ".join(f"{k}={v}" for k, v in p.env.items())))
    w = max(len(k) for k, _ in rows)
    out = ["┌─ model card " + "─" * 50]
    out += [f"│ {k:<{w}}  {v}" for k, v in rows]
    if p.description:
        out.append("│")
        import textwrap
        for line in textwrap.wrap(p.description, 96):
            out.append(f"│ {line}")
    out.append("└" + "─" * 63)
    return "\n".join(out)


def _lint(reg: "Registry") -> list:
    """Naming-convention and card checks; returns human-readable problems."""
    problems = []
    for p in reg.all():
        if p.canonical == "generic":
            continue
        missing = [k for k in ("family", "params", "ctx", "status") if k not in p.card]
        if missing:
            problems.append(f"{p.canonical}: missing card field(s) {', '.join(missing)}")
            continue
        want = derived_name(p.card, p.nodes)
        if p.canonical != want:
            problems.append(f"{p.canonical}: name does not match its card — expected {want!r}")
    return problems


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
    if cmd == "card":
        if not rest:
            print("usage: presets.py card <name>", file=sys.stderr)
            return 2
        print(_cmd_card(reg.resolve(rest[0])))
        return 0
    if cmd == "table":
        for p in reg.all():
            if p.canonical == "generic":
                continue
            old = p.aliases[0] if p.aliases else ""
            print(f"    {p.canonical:38s} {('(' + old + ')') if old else '':22s} {p.card.get('status', '')}")
        return 0
    if cmd == "lint":
        problems = _lint(reg)
        for line in problems:
            print(line)
        return 1 if problems else 0
    if cmd == "env":
        if not rest:
            print("usage: presets.py env <name>", file=sys.stderr)
            return 2
        for k, v in reg.resolve(rest[0]).env.items():
            print(f"{k}={v}")
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
                "storage": p.storage,
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
