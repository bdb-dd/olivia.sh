"""Compile every embedded `python3 ... <<'TAG'` heredoc in a shell script.

`bash -n` cannot catch a broken one: the SHELL is valid, the PYTHON inside is
not. That is exactly how a nested heredoc sat latent on main for ten weeks.
"""
import io, re, sys

PAT = re.compile(r"""<<-?\s*'([A-Za-z_][A-Za-z0-9_]*)'""")

def check(path):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    ok = bad = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = PAT.search(line)
        # only heredocs fed to a python interpreter
        if m and re.search(r"\bpython3?\b", line):
            tag = m.group(1)
            body, j = [], i + 1
            while j < len(lines) and lines[j].strip() != tag:
                body.append(lines[j]); j += 1
            if j >= len(lines):
                print(f"  UNTERMINATED {tag} opened at line {i+1}"); bad += 1
            else:
                try:
                    compile("\n".join(body) + "\n", f"{path}:{tag}@{i+1}", "exec")
                    ok += 1
                except SyntaxError as e:
                    print(f"  BROKEN {tag} (opened line {i+1}): "
                          f"{e.msg} at body line {e.lineno}")
                    if e.text: print(f"         {e.text.rstrip()}")
                    bad += 1
            i = j + 1
            continue
        i += 1
    print(f"{path}: {ok} OK, {bad} broken")
    return bad

sys.exit(1 if sum(check(p) for p in sys.argv[1:]) else 0)
