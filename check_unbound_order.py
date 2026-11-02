"""Flag per-model knobs REFERENCED before they are ASSIGNED.

`set -euo pipefail` makes such a reference fatal, but only when the user has not
exported the variable -- i.e. in the common case. `bash -n` cannot see it: the
syntax is valid, the ordering is wrong. Job 2694557 died in 1 second this way.

Safe:   ${NAME:-default}  (carries its own default)
Unsafe: ${NAME} / $NAME    before the first `NAME=` line
"""
import io, re, sys

# Internal per-model knobs: never supplied by olivia.sh's env block, always
# given their default inside the script, so ordering is entirely on us.
PREFIX = re.compile(r"^(DSV4|DSV41|GLM5|GLM52|GLM53|GLM53_FLASH|GLM47|KIMI|ORNITH|"
                    r"QWEN38|BOREALIS|LAGUNA|INKLING|MINIMAX)_[A-Z0-9_]+$")
ASSIGN = re.compile(r"^\s*(?:local\s+|export\s+)?([A-Z][A-Z0-9_]*)=")
SAFE_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::[-=?+]|##|%%|#|%)")
ANY_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}|\$([A-Z][A-Z0-9_]*)\b")

def check(path):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    first_assign, first_unsafe = {}, {}
    for i, line in enumerate(lines, 1):
        code = re.sub(r"(?<!\\)#.*$", "", line) if not line.lstrip().startswith("#") else ""
        if not code.strip():
            continue
        m = ASSIGN.match(code)
        if m and m.group(1) not in first_assign:
            first_assign[m.group(1)] = i
        safe = {g for g in SAFE_REF.findall(code)}
        for a, b in ANY_REF.findall(code):
            name = a or b
            if name in safe or name in first_unsafe:
                continue
            # a self-defaulting assignment is the assignment, not a use
            if m and m.group(1) == name:
                continue
            first_unsafe[name] = i

    bad = 0
    for name, aline in sorted(first_assign.items(), key=lambda kv: kv[1]):
        if not PREFIX.match(name):
            continue
        uline = first_unsafe.get(name)
        if uline is not None and uline < aline:
            print(f"  USE-BEFORE-ASSIGN  {name}")
            print(f"      referenced line {uline}: {lines[uline-1].strip()[:95]}")
            print(f"      assigned   line {aline}: {lines[aline-1].strip()[:95]}")
            bad += 1
    print(f"{path}: {bad} use-before-assign issue(s) among "
          f"{sum(1 for n in first_assign if PREFIX.match(n))} per-model knobs")
    return bad

sys.exit(1 if sum(check(p) for p in sys.argv[1:]) else 0)
