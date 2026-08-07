#!/usr/bin/env python3
"""Generate + calibrate the L2 (swe) task set. Writes evals/swe/tasks/*.json and
verifies for each task that the buggy code FAILS the hidden grader and a known
reference fix PASSES it. Run from the agentic-evals worktree root."""
import json, os, subprocess, sys, tempfile, textwrap

OUT = "evals/swe/tasks"
os.makedirs(OUT, exist_ok=True)

def dedent(s):
    return textwrap.dedent(s).lstrip("\n")

TASKS = []

# --- T1: operator precedence in a recursive-descent calculator (multi-file) ---
TOKENIZER = dedent(r'''
    import re


    def tokenize(s):
        return re.findall(r"\d+|[+\-*/()]", s.replace(" ", ""))
''')
CALC_BUGGY = dedent('''
    from tokenizer import tokenize


    def evaluate(expr):
        tokens = tokenize(expr)
        result = int(tokens[0])
        i = 1
        while i < len(tokens):
            op = tokens[i]
            rhs = int(tokens[i + 1])
            if op == "+":
                result += rhs
            elif op == "-":
                result -= rhs
            elif op == "*":
                result *= rhs
            elif op == "/":
                result //= rhs
            i += 2
        return result
''')
CALC_FIX = dedent('''
    from tokenizer import tokenize


    def evaluate(expr):
        tokens = tokenize(expr)
        pos = 0

        def peek():
            return tokens[pos] if pos < len(tokens) else None

        def advance():
            nonlocal pos
            t = tokens[pos]
            pos += 1
            return t

        def parse_expr():
            val = parse_term()
            while peek() in ("+", "-"):
                op = advance()
                rhs = parse_term()
                val = val + rhs if op == "+" else val - rhs
            return val

        def parse_term():
            val = parse_factor()
            while peek() in ("*", "/"):
                op = advance()
                rhs = parse_factor()
                val = val * rhs if op == "*" else val // rhs
            return val

        def parse_factor():
            t = advance()
            if t == "(":
                val = parse_expr()
                advance()  # consume ")"
                return val
            return int(t)

        return parse_expr()
''')
CALC_GRADE = dedent('''
    from calc import evaluate

    assert evaluate("2+3*4") == 14
    assert evaluate("2*3+4") == 10
    assert evaluate("(2+3)*4") == 20
    assert evaluate("10-2-3") == 5
    assert evaluate("2+3*4-1") == 13
    assert evaluate("100/5/2") == 10
    assert evaluate("2*(3+4)*2") == 28
    assert evaluate("7") == 7
    assert evaluate("1+2*3-4/2") == 5
    print("PASS")
''')
TASKS.append(dict(
    id="calc-operator-precedence", category="parser",
    description="Recursive-descent calculator ignores operator precedence and parentheses.",
    prompt=("evaluate(expr) in calc.py computes integer arithmetic over strings like '2+3*4'. "
            "It currently evaluates strictly left-to-right, ignoring operator precedence and "
            "parentheses: evaluate('2+3*4') returns 20 but should return 14, and parentheses are "
            "ignored entirely. Fix calc.py (and tokenizer.py if needed) so evaluate() honors "
            "standard precedence (* and / bind tighter than + and -) and parentheses. Use integer "
            "division (//) for '/'. Keep the public function signature evaluate(expr: str) -> int; "
            "do not rename it. tokenizer.tokenize already returns the correct token list."),
    files={"tokenizer.py": TOKENIZER, "calc.py": CALC_BUGGY},
    tools=["read_file", "write_file", "list_dir", "grep", "run_bash"],
    oracle_files={"grade_test.py": CALC_GRADE},
    oracle=[{"check": "bash_exit_zero", "command": "python3 grade_test.py"}],
    _fix={"calc.py": CALC_FIX},
))

# --- T2: interval merge off-by-one on touching intervals ---
INTERVALS_BUGGY = dedent('''
    def merge(intervals):
        if not intervals:
            return []
        s = sorted(intervals)
        result = [list(s[0])]
        for start, end in s[1:]:
            if start < result[-1][1]:
                result[-1][1] = max(result[-1][1], end)
            else:
                result.append([start, end])
        return [tuple(x) for x in result]
''')
INTERVALS_FIX = INTERVALS_BUGGY.replace("if start < result[-1][1]:",
                                        "if start <= result[-1][1]:")
INTERVALS_GRADE = dedent('''
    from intervals import merge

    assert merge([]) == []
    assert merge([(1, 3)]) == [(1, 3)]
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]
    assert merge([(1, 5), (2, 3)]) == [(1, 5)]
    assert merge([(1, 4), (2, 5), (7, 9)]) == [(1, 5), (7, 9)]
    assert merge([(5, 6), (1, 2), (3, 4)]) == [(1, 2), (3, 4), (5, 6)]
    assert merge([(1, 10), (2, 3), (4, 5), (11, 12)]) == [(1, 10), (11, 12)]
    assert merge([(1, 2), (3, 4), (2, 3)]) == [(1, 4)]
    print("PASS")
''')
TASKS.append(dict(
    id="interval-merge-adjacency", category="algorithm",
    description="Interval merge treats touching intervals as disjoint.",
    prompt=("merge(intervals) in intervals.py merges a list of (start, end) integer intervals into "
            "the minimal sorted set of non-overlapping intervals. It currently treats touching "
            "intervals as separate: merge([(1,2),(2,3)]) returns [(1,2),(2,3)] but should return "
            "[(1,3)] because they touch at 2. Overlapping intervals already merge; the input may be "
            "unsorted. Fix merge() so touching AND overlapping intervals combine. Keep the signature "
            "merge(intervals) -> list[(start, end)] sorted by start."),
    files={"intervals.py": INTERVALS_BUGGY},
    tools=["read_file", "write_file", "list_dir", "grep", "run_bash"],
    oracle_files={"grade_test.py": INTERVALS_GRADE},
    oracle=[{"check": "bash_exit_zero", "command": "python3 grade_test.py"}],
    _fix={"intervals.py": INTERVALS_FIX},
))

# --- T3: LRU cache doesn't refresh recency on get/update ---
LRU_BUGGY = dedent('''
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.store = {}
            self.order = []  # least-recently-used at front

        def get(self, key):
            if key not in self.store:
                return -1
            return self.store[key]

        def put(self, key, value):
            if key in self.store:
                self.store[key] = value
                return
            if len(self.store) >= self.capacity:
                oldest = self.order.pop(0)
                del self.store[oldest]
            self.store[key] = value
            self.order.append(key)
''')
LRU_FIX = dedent('''
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.store = {}
            self.order = []  # least-recently-used at front

        def _touch(self, key):
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)

        def get(self, key):
            if key not in self.store:
                return -1
            self._touch(key)
            return self.store[key]

        def put(self, key, value):
            if key in self.store:
                self.store[key] = value
                self._touch(key)
                return
            if len(self.store) >= self.capacity:
                oldest = self.order.pop(0)
                del self.store[oldest]
            self.store[key] = value
            self._touch(key)
''')
LRU_GRADE = dedent('''
    from lru import LRUCache

    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)            # should evict key 2 (least recently used)
    assert c.get(2) == -1
    c.put(4, 4)            # should evict key 1
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4

    d = LRUCache(2)
    d.put(1, 1)
    d.put(2, 2)
    d.put(1, 10)          # updating key 1 makes it most-recently-used
    d.put(3, 3)           # should evict key 2
    assert d.get(2) == -1
    assert d.get(1) == 10
    assert d.get(3) == 3
    print("PASS")
''')
TASKS.append(dict(
    id="lru-cache-recency", category="data-structure",
    description="LRU cache fails to refresh recency on get and on updating put.",
    prompt=("LRUCache(capacity) in lru.py implements a least-recently-used cache: get(key) returns "
            "the value or -1 if absent; put(key, value) inserts/updates; when full, put evicts the "
            "least-recently-used key. Bug: recency is only tracked on initial insertion, so a get() "
            "that hits and a put() that updates an existing key do NOT mark the key most-recently-"
            "used, causing the wrong key to be evicted. Fix lru.py so every successful get and every "
            "put marks the key most-recently-used. Keep the class name LRUCache and methods get/put."),
    files={"lru.py": LRU_BUGGY},
    tools=["read_file", "write_file", "list_dir", "grep", "run_bash"],
    oracle_files={"grade_test.py": LRU_GRADE},
    oracle=[{"check": "bash_exit_zero", "command": "python3 grade_test.py"}],
    _fix={"lru.py": LRU_FIX},
))

# --- T4: topological sort missing cycle detection ---
GRAPH_BUGGY = dedent('''
    def topo_sort(graph):
        """graph: dict node -> list of nodes it points to. Return an ordering where
        every node precedes the nodes it points to. Raise ValueError on a cycle."""
        visited = set()
        result = []

        def visit(n):
            if n in visited:
                return
            visited.add(n)
            for m in graph.get(n, []):
                visit(m)
            result.append(n)

        for n in list(graph):
            visit(n)
        return result[::-1]
''')
GRAPH_FIX = dedent('''
    def topo_sort(graph):
        """graph: dict node -> list of nodes it points to. Return an ordering where
        every node precedes the nodes it points to. Raise ValueError on a cycle."""
        visited = set()
        visiting = set()
        result = []

        def visit(n):
            if n in visited:
                return
            if n in visiting:
                raise ValueError("cycle detected at %r" % (n,))
            visiting.add(n)
            for m in graph.get(n, []):
                visit(m)
            visiting.discard(n)
            visited.add(n)
            result.append(n)

        for n in list(graph):
            visit(n)
        return result[::-1]
''')
GRAPH_GRADE = dedent('''
    from graph import topo_sort

    def nodes_of(g):
        ns = set(g)
        for u in g:
            ns.update(g[u])
        return ns

    def check_order(g):
        order = topo_sort(g)
        assert sorted(order) == sorted(nodes_of(g)), (order, sorted(nodes_of(g)))
        pos = {n: i for i, n in enumerate(order)}
        for u in g:
            for v in g[u]:
                assert pos[u] < pos[v], "%r must precede %r" % (u, v)

    check_order({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []})
    check_order({"x": ["y"], "y": ["z"], "z": []})
    check_order({"a": [], "b": ["a"], "c": ["a", "b"]})

    for cyc in [{"a": ["b"], "b": ["a"]}, {"a": ["a"]},
                {"a": ["b"], "b": ["c"], "c": ["a"]}]:
        try:
            topo_sort(cyc)
            raise AssertionError("expected ValueError for cycle: %r" % (cyc,))
        except ValueError:
            pass
    print("PASS")
''')
TASKS.append(dict(
    id="topo-sort-cycle-detection", category="algorithm",
    description="Topological sort returns a bogus order on cyclic input instead of raising.",
    prompt=("topo_sort(graph) in graph.py orders the nodes of a dependency graph (dict mapping a "
            "node to the list of nodes it points to) so each node precedes the nodes it points to. "
            "It works for acyclic graphs but, on a graph that contains a cycle (including a self-loop "
            "like {'a': ['a']}), it silently returns a bogus order instead of raising. Fix topo_sort "
            "so it raises ValueError when the graph contains a cycle, while still returning a valid "
            "order for any acyclic graph. Keep the function name and the dict input format."),
    files={"graph.py": GRAPH_BUGGY},
    tools=["read_file", "write_file", "list_dir", "grep", "run_bash"],
    oracle_files={"grade_test.py": GRAPH_GRADE},
    oracle=[{"check": "bash_exit_zero", "command": "python3 grade_test.py"}],
    _fix={"graph.py": GRAPH_FIX},
))

# --- T5: naive CSV line parser breaks on quotes ---
CSV_BUGGY = dedent('''
    def parse_line(line):
        """Split one CSV record into fields. Must support double-quoted fields that
        contain commas, and doubled quotes ("") as an escaped quote inside a quoted
        field. Surrounding quotes are stripped from quoted fields."""
        return line.split(",")
''')
CSV_FIX = dedent('''
    def parse_line(line):
        """Split one CSV record into fields. Supports double-quoted fields containing
        commas and doubled quotes as an escaped quote within a quoted field."""
        fields = []
        buf = []
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if c == '"':
                i += 1
                while i < n:
                    if line[i] == '"':
                        if i + 1 < n and line[i + 1] == '"':
                            buf.append('"')
                            i += 2
                        else:
                            i += 1
                            break
                    else:
                        buf.append(line[i])
                        i += 1
            elif c == ",":
                fields.append("".join(buf))
                buf = []
                i += 1
            else:
                buf.append(c)
                i += 1
        fields.append("".join(buf))
        return fields
''')
CSV_GRADE = dedent('''
    from csvparse import parse_line

    assert parse_line("a,b,c") == ["a", "b", "c"]
    assert parse_line('a,"b,c",d') == ["a", "b,c", "d"]
    assert parse_line('"hello, world",x') == ["hello, world", "x"]
    assert parse_line('a,"she said ""hi""",b') == ["a", 'she said "hi"', "b"]
    assert parse_line('"",x') == ["", "x"]
    assert parse_line("one") == ["one"]
    assert parse_line("a,,b") == ["a", "", "b"]
    assert parse_line('"a,b","c,d"') == ["a,b", "c,d"]
    print("PASS")
''')
TASKS.append(dict(
    id="csv-quoted-field-parsing", category="parser",
    description="CSV line parser splits naively and breaks quoted fields.",
    prompt=("parse_line(line) in csvparse.py splits a single CSV record into a list of fields. It "
            "must support double-quoted fields that contain commas (\"a,b\" is the single field a,b) "
            "and doubled quotes inside a quoted field as an escaped quote (\"\" -> \"). The current "
            "version just calls line.split(','), which breaks any quoted field. Implement correct "
            "parsing. Surrounding quotes are stripped from quoted fields; unquoted fields are taken "
            "literally. Keep the signature parse_line(line: str) -> list[str]."),
    files={"csvparse.py": CSV_BUGGY},
    tools=["read_file", "write_file", "list_dir", "grep", "run_bash"],
    oracle_files={"grade_test.py": CSV_GRADE},
    oracle=[{"check": "bash_exit_zero", "command": "python3 grade_test.py"}],
    _fix={"csvparse.py": CSV_FIX},
))


def run_grade(d):
    p = subprocess.run("python3 grade_test.py", shell=True, cwd=d,
                       capture_output=True, text=True, timeout=30)
    return p.returncode


def calibrate(task):
    files, fix, ofiles = task["files"], task["_fix"], task["oracle_files"]
    with tempfile.TemporaryDirectory() as d:
        for k, v in {**files, **ofiles}.items():
            with open(os.path.join(d, k), "w") as f:
                f.write(v)
        buggy_rc = run_grade(d)
    with tempfile.TemporaryDirectory() as d:
        for k, v in {**files, **fix, **ofiles}.items():
            with open(os.path.join(d, k), "w") as f:
                f.write(v)
        fixed_rc = run_grade(d)
    return buggy_rc != 0, fixed_rc == 0


allok = True
for i, t in enumerate(TASKS, 1):
    buggy_fails, fixed_passes = calibrate(t)
    good = buggy_fails and fixed_passes
    allok = allok and good
    out = {k: v for k, v in t.items() if not k.startswith("_")}
    out["max_turns"] = 25
    path = os.path.join(OUT, f"{i:02d}_{t['id']}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"  {'OK ' if good else 'BAD'} {t['id']:<30} buggy_fails={buggy_fails} fixed_passes={fixed_passes}  -> {path}")

print("\nALL L2 TASKS WELL-CALIBRATED" if allok else "\n*** MISCALIBRATED TASK(S) ABOVE ***")
sys.exit(0 if allok else 1)
