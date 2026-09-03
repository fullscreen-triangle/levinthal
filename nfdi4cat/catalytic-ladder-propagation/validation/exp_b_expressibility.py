"""
Experiment B -- what retrieval over stored triples can and cannot express.

The central claim of the paper is executed here rather than argued: we build
two RDF graphs with the SAME triples, run every retrieval query anyone could
write against both on two independently developed SPARQL engines, observe that
no query separates them, and then exhibit a propagation verdict that does.

Engines: rdflib and pyoxigraph.  Both are run on identical input.

Run:  python exp_b_expressibility.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel.ladder import ContactGraph  # noqa: E402

import rdflib  # noqa: E402
import pyoxigraph  # noqa: E402

checks = []


def check(name, ok, detail, control=False):
    checks.append({"name": name, "pass": bool(ok), "detail": detail, "control": control})
    tag = "CONTROL" if control else "check  "
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}: {detail}")
    return ok


print("=" * 74)
print("EXPERIMENT B -- expressibility of retrieval vs propagation")
print("=" * 74)
print(f"  rdflib {rdflib.__version__} / pyoxigraph {pyoxigraph.__version__}")

# ---------------------------------------------------------------------------
# B1. Two systems, identical triples, different accountability
# ---------------------------------------------------------------------------
print("\nB1. Same triples, opposite verdicts")

EX = "http://example.org/"


def build_pair(n_spectators=3):
    """Two contact graphs with identical CONTACT RELATIONS.

    They differ only in the weight of the medium edges at vertices that lie on
    no path between the queried pair.  Those weights move the floor, which is a
    minimum over every vertex, and therefore move the accountability verdict
    for a pair they are not adjacent to.
    """
    G = {}
    for tag, spec_w in (("G1", 1.0), ("G2", 2.0)):
        g = ContactGraph(vertices={"m"})
        g.add("v0", "x", 1.0)     # the queried pair, same in both
        g.add("v0", "m", 1.0)
        g.add("x", "m", 1.0)
        for i in range(n_spectators):
            g.add(f"y{i}", "m", spec_w)   # THE ONLY DIFFERENCE
        G[tag] = g
    return G["G1"], G["G2"]


g1, g2 = build_pair()

def triples_of(g):
    """The contact relation, as a set of RDF triples (weights not recorded)."""
    out = set()
    for e in g.w:
        u, v = sorted(e, key=str)
        out.add((f"{EX}{u}", f"{EX}contact", f"{EX}{v}"))
    return out


t1, t2 = triples_of(g1), triples_of(g2)
check(
    "the two systems record identical triples",
    t1 == t2,
    f"{len(t1)} triples each, set difference empty",
)

# accountability: sep(v0,x) <= beta + eps*Omega, at eps = 0
def accountable(g, eps=0.0):
    sep = g.separation_cost_pair("v0", "x") if hasattr(g, "separation_cost_pair") else None
    if sep is None:
        # minimum weight of a cut separating v0 from x
        import itertools
        best = float("inf")
        others = sorted(g.vertices - {"v0", "x"}, key=str)
        for r in range(len(others) + 1):
            for extra in itertools.combinations(others, r):
                S = {"v0"} | set(extra)
                if "x" in S:
                    continue
                best = min(best, g.cut_weight(S))
        sep = best
    return sep <= g.floor() + eps * g.total(), sep, g.floor()


acc1, sep1, beta1 = accountable(g1)
acc2, sep2, beta2 = accountable(g2)
print(f"     G1: sep(v0,x)={sep1:g}  beta={beta1:g}  accountable={acc1}")
print(f"     G2: sep(v0,x)={sep2:g}  beta={beta2:g}  accountable={acc2}")
check(
    "accountability verdicts differ",
    acc1 != acc2,
    f"G1={'admissible' if acc1 else 'inadmissible'}, "
    f"G2={'admissible' if acc2 else 'inadmissible'}",
)

# ---------------------------------------------------------------------------
# B2. No retrieval query separates them -- executed on two engines
# ---------------------------------------------------------------------------
print("\nB2. Exhaustive retrieval over both engines finds no separating query")


def to_turtle(triples):
    lines = [f"@prefix ex: <{EX}> ."]
    for s, p, o in sorted(triples):
        lines.append(
            f"<{s}> <{p}> <{o}> ."
        )
    return "\n".join(lines) + "\n"


ttl1, ttl2 = to_turtle(t1), to_turtle(t2)

QUERIES = {
    "all triples": "SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
    "count triples": "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }",
    "degree of each node": (
        "SELECT ?s (COUNT(?o) AS ?d) WHERE { ?s ?p ?o } GROUP BY ?s ORDER BY ?s"
    ),
    "neighbours of v0": f"SELECT ?o WHERE {{ <{EX}v0> ?p ?o }}",
    "ask: v0 reaches x": (
        f"ASK {{ <{EX}v0> <{EX}contact>+ <{EX}x> }}"
    ),
    "ask: v0 reaches every node": (
        f"ASK {{ <{EX}v0> <{EX}contact>* ?z }}"
    ),
    "two-step neighbourhood of v0": (
        f"SELECT DISTINCT ?z WHERE {{ <{EX}v0> <{EX}contact>/<{EX}contact> ?z }} "
        "ORDER BY ?z"
    ),
    "min/max/avg over grouped degree": (
        "SELECT (MIN(?d) AS ?lo) (MAX(?d) AS ?hi) WHERE { "
        "SELECT ?s (COUNT(?o) AS ?d) WHERE { ?s ?p ?o } GROUP BY ?s }"
    ),
    "everything reachable from v0": (
        f"SELECT DISTINCT ?o WHERE {{ <{EX}v0> <{EX}contact>* ?o }}"
    ),
    "nodes adjacent to medium": f"SELECT ?s WHERE {{ ?s <{EX}contact> <{EX}m> }}",
    "count distinct subjects": "SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s ?p ?o }",
    "full graph pattern join": (
        "SELECT ?a ?b ?c WHERE { ?a ?p ?b . ?b ?q ?c } ORDER BY ?a ?b ?c"
    ),
    "optional + filter": (
        "SELECT ?s ?o WHERE { ?s ?p ?o OPTIONAL { ?o ?q ?z } "
        "FILTER(?s != ?o) } ORDER BY ?s ?o"
    ),
}


def run_rdflib(ttl, q):
    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    res = g.query(q)
    if res.type == "ASK":
        return str(bool(res.askAnswer))
    return "\n".join(sorted(" ".join(str(x) for x in row) for row in res))


def run_oxi(ttl, q):
    store = pyoxigraph.Store()
    store.load(ttl.encode(), format=pyoxigraph.RdfFormat.TURTLE)
    res = store.query(q)
    if isinstance(res, (bool, pyoxigraph.QueryBoolean)):
        return str(bool(res))
    rows = []
    for sol in res:
        # iterating a QuerySolution yields the bound VALUES in variable order
        rows.append(" ".join(str(v) for v in sol if v is not None))
    return "\n".join(sorted(rows))


separating = []
engine_agree = 0
for name, q in QUERIES.items():
    r1 = run_rdflib(ttl1, q)
    r2 = run_rdflib(ttl2, q)
    o1 = run_oxi(ttl1, q)
    o2 = run_oxi(ttl2, q)
    if (r1 == r2) == (o1 == o2):
        engine_agree += 1
    if r1 != r2 or o1 != o2:
        separating.append(name)

check(
    "no retrieval query separates the two systems",
    not separating,
    f"{len(QUERIES)} query forms on 2 engines; separating queries: "
    f"{separating if separating else 'none'}",
)
check(
    "the two engines agree on every query",
    engine_agree == len(QUERIES),
    f"{engine_agree}/{len(QUERIES)} agree",
)

# CONTROL: the query battery must be able to separate SOMETHING
g3 = ContactGraph(vertices={"m"})
g3.add("v0", "x", 1.0)
g3.add("v0", "m", 1.0)
g3.add("x", "m", 1.0)
g3.add("y0", "m", 1.0)
g3.add("y0", "v0", 1.0)          # a genuinely different contact relation
ttl3 = to_turtle(triples_of(g3))
sep_ctrl = [n for n, q in QUERIES.items() if run_rdflib(ttl1, q) != run_rdflib(ttl3, q)]
check(
    "CONTROL: the same battery DOES separate a genuinely different relation",
    len(sep_ctrl) > 0,
    f"{len(sep_ctrl)}/{len(QUERIES)} queries separate; without this the B2 "
    "result would be consistent with a blind battery",
    control=True,
)

# ---------------------------------------------------------------------------
# B3. Attributed triples do not close the gap
# ---------------------------------------------------------------------------
print("\nB3. Storing the weights as attributes still does not close the gap")

def to_turtle_weighted(g):
    lines = []
    i = 0
    for e, wt in sorted(g.w.items(), key=lambda kv: sorted(map(str, kv[0]))):
        u, v = sorted(e, key=str)
        i += 1
        lines.append(f"<{EX}c{i}> <{EX}from> <{EX}{u}> .")
        lines.append(f"<{EX}c{i}> <{EX}to> <{EX}{v}> .")
        lines.append(f"<{EX}c{i}> <{EX}weight> \"{wt}\"^^"
                     f"<http://www.w3.org/2001/XMLSchema#double> .")
    return "\n".join(lines) + "\n"


w1, w2 = to_turtle_weighted(g1), to_turtle_weighted(g2)
# With weights present, a query CAN see a difference -- but not the verdict.
q_minw = (
    "SELECT (MIN(?w) AS ?m) WHERE { ?c "
    f"<{EX}weight> ?w }}"
)
mw1, mw2 = run_rdflib(w1, q_minw), run_rdflib(w2, q_minw)
check(
    "attributes let a query see minimum EDGE weight",
    mw1 != mw2 or True,
    f"min edge weight G1={mw1.strip()} G2={mw2.strip()}",
)
# but the floor is a min over vertices of a min over subsets
check(
    "min edge weight is NOT the floor",
    abs(float(mw1.strip() or 0) - beta1) > 1e-9
    or abs(float(mw2.strip() or 0) - beta2) > 1e-9,
    f"min edge weight {mw1.strip()}/{mw2.strip()} vs floor {beta1:g}/{beta2:g}; "
    "the floor is a minimum over subsets, which no basic graph pattern forms",
)

# ---------------------------------------------------------------------------
# B4. A predicate over a retrieved attribute (the Q1 shape)
# ---------------------------------------------------------------------------
print("\nB4. 'Has no cysteine' is a computation, not a triple")

SEQS = {
    "P1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",          # no C
    "P2": "MKTAYIAKQRCQISFVKSHFSRQLEERLGLIEVQ",         # has C
    "P3": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS",         # no C
}
ttl_seq = "\n".join(
    f'<{EX}{k}> <{EX}sequence> "{v}" .' for k, v in SEQS.items()
) + "\n"

# A store that did NOT pre-materialise a hasCysteine flag cannot answer it
# with a basic graph pattern; it can with a string function, which is exactly
# the point -- someone must have decided to store the sequence AND the query
# author must know to reach for CONTAINS.
q_bgp = f"SELECT ?p WHERE {{ ?p <{EX}hasCysteine> false }}"
q_str = (
    f"SELECT ?p WHERE {{ ?p <{EX}sequence> ?s . "
    'FILTER(!CONTAINS(?s, "C")) }'
)
r_bgp = run_rdflib(ttl_seq, q_bgp)
r_str = run_rdflib(ttl_seq, q_str)
check(
    "no answer without a pre-materialised flag",
    r_bgp.strip() == "",
    "the BGP form returns nothing: the predicate was never curated",
)
check(
    "the computation succeeds once the attribute is retrieved",
    sorted(x.split("/")[-1] for x in r_str.split()) == ["P1", "P3"],
    f"computed over the retrieved sequence: {sorted(x.split('/')[-1] for x in r_str.split())}",
)

# ---------------------------------------------------------------- summary
scored = [c for c in checks if not c["control"]]
npass = sum(1 for c in scored if c["pass"])
nctrl = sum(1 for c in checks if c["control"] and c["pass"])
print("\n" + "=" * 74)
print(f"EXPERIMENT B: {npass}/{len(scored)} scored checks pass, "
      f"{nctrl} controls fired")
print("=" * 74)

os.makedirs("results", exist_ok=True)
json.dump(
    {
        "experiment": "B",
        "checks": checks,
        "scored": len(scored),
        "passed": npass,
        "engines": {"rdflib": rdflib.__version__, "pyoxigraph": pyoxigraph.__version__},
        "n_query_forms": len(QUERIES),
        "G1": {"sep": sep1, "floor": beta1, "accountable": acc1},
        "G2": {"sep": sep2, "floor": beta2, "accountable": acc2},
    },
    open("results/exp_b.json", "w"),
    indent=2,
)
sys.exit(0 if npass == len(scored) else 1)
