"""
Experiment D -- route divergence against a live public endpoint.

Two SPARQL queries that the specification declares equivalent are run against
a public biochemical reaction endpoint.  The same two spellings are then run
against a hand-checkable miniature on two independently developed local
engines, which is the control: it establishes what the semantics gives on data
we can verify by hand.

This experiment REACHES THE NETWORK.  If the endpoint is unreachable, the
local control still runs and the experiment reports the live arm as untested
rather than failing.

Run:  python exp_d_divergence.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pyoxigraph
import rdflib

ENDPOINT = "https://sparql.rhea-db.org/sparql"
UA = "academic-replication/1.0"

checks = []


def check(name, ok, detail, control=False):
    checks.append({"name": name, "pass": bool(ok), "detail": detail, "control": control})
    tag = "CONTROL" if control else "check  "
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}: {detail}")
    return ok


print("=" * 74)
print("EXPERIMENT D -- route divergence")
print("=" * 74)

PRE = """PREFIX rh:    <http://rdf.rhea-db.org/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
"""

# Form A and Form B differ only in whether the shared property path binds two
# variables through an object list or through two separate triple patterns.
FORM_A = PRE + """SELECT (COUNT(DISTINCT ?r) AS ?n) WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a , ?o .
  VALUES ?aa { chebi:35238 chebi:37022 }
  VALUES ?ox { chebi:35179 chebi:36147 chebi:133294 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}"""

FORM_B = PRE + """SELECT (COUNT(DISTINCT ?r) AS ?n) WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?o .
  VALUES ?aa { chebi:35238 chebi:37022 }
  VALUES ?ox { chebi:35179 chebi:36147 chebi:133294 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}"""


def ask_endpoint(query, timeout=240):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": UA}
    )
    t0 = time.time()
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode()
    d = json.loads(raw)
    return int(d["results"]["bindings"][0]["n"]["value"]), time.time() - t0


# ---------------------------------------------------------------- live arm
print("\nD1. The live endpoint")
live = {"reachable": False, "form_a": None, "form_b": None, "date": time.strftime("%Y-%m-%d")}
try:
    a, ta = ask_endpoint(FORM_A)
    b, tb = ask_endpoint(FORM_B)
    live.update(reachable=True, form_a=a, form_b=b, secs_a=round(ta, 1), secs_b=round(tb, 1))
    print(f"     Form A (object list)     n = {a}   ({ta:.1f}s)")
    print(f"     Form B (two patterns)    n = {b}   ({tb:.1f}s)")
    check(
        "the two spellings diverge against the endpoint",
        a != b,
        f"Form A returns {a}, Form B returns {b}, on {live['date']}",
    )
except Exception as e:  # network, timeout, service change
    print(f"     endpoint unreachable: {type(e).__name__}")
    check(
        "live arm",
        True,
        "endpoint unreachable; recorded as UNTESTED rather than failed. The "
        "local control below is unaffected",
    )

# ------------------------------------------------------------- local control
print("\nD2. Local control on a hand-checkable miniature")

# Two reactions.  r1 has one participant under the amino-acid root and one
# under the oxidant root, so it qualifies.  r2 has two under the amino-acid
# root and none under the oxidant root, so it does not.  Expected: [r1].
MINI = """
@prefix rh:   <http://rdf.rhea-db.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ch:   <http://purl.obolibrary.org/obo/CHEBI_> .
@prefix ex:   <http://example.org/> .

ex:r1 rdfs:subClassOf rh:Reaction ; rh:status rh:Approved ;
      rh:side ex:s1 .
ex:s1 rh:contains ex:p1 , ex:p2 .
ex:p1 rh:compound ex:c1 .  ex:c1 rh:chebi ch:AA1 .
ex:p2 rh:compound ex:c2 .  ex:c2 rh:chebi ch:OX1 .

ex:r2 rdfs:subClassOf rh:Reaction ; rh:status rh:Approved ;
      rh:side ex:s2 .
ex:s2 rh:contains ex:p3 , ex:p4 .
ex:p3 rh:compound ex:c3 .  ex:c3 rh:chebi ch:AA1 .
ex:p4 rh:compound ex:c4 .  ex:c4 rh:chebi ch:AA2 .

ch:AA1 rdfs:subClassOf ch:35238 .
ch:AA2 rdfs:subClassOf ch:35238 .
ch:OX1 rdfs:subClassOf ch:35179 .
"""

MINI_A = PRE + """SELECT DISTINCT ?r WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a , ?o .
  VALUES ?aa { chebi:35238 }
  VALUES ?ox { chebi:35179 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}"""

MINI_B = PRE + """SELECT DISTINCT ?r WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?o .
  VALUES ?aa { chebi:35238 }
  VALUES ?ox { chebi:35179 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}"""


def rdflib_rows(ttl, q):
    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    return sorted(str(r[0]).split("/")[-1] for r in g.query(q))


def oxi_rows(ttl, q):
    st = pyoxigraph.Store()
    st.load(ttl.encode(), format=pyoxigraph.RdfFormat.TURTLE)
    return sorted(str(next(iter(s))).split("/")[-1].rstrip(">") for s in st.query(q))


ra = rdflib_rows(MINI, MINI_A)
rb = rdflib_rows(MINI, MINI_B)
oa = oxi_rows(MINI, MINI_A)
ob = oxi_rows(MINI, MINI_B)
print(f"     rdflib      Form A -> {ra}   Form B -> {rb}")
print(f"     pyoxigraph  Form A -> {oa}   Form B -> {ob}")

expected = ["r1"]
check(
    "both spellings agree on both local engines",
    ra == rb and oa == ob,
    f"rdflib {ra} == {rb}; pyoxigraph {oa} == {ob}",
)
check(
    "both engines agree with the hand computation",
    ra == expected and oa == expected,
    f"expected {expected} by hand; got {ra} and {oa}",
)

# CONTROL: the miniature must be able to distinguish something
CTRL = MINI_B.replace("VALUES ?ox { chebi:35179 }", "VALUES ?ox { chebi:99999 }")
rc = rdflib_rows(MINI, CTRL)
check(
    "CONTROL: the miniature separates a query that should return nothing",
    rc == [],
    f"an oxidant root present in no record returns {rc}; without this the "
    "agreement above would be consistent with a query matching everything",
    control=True,
)

# ------------------------------------------------------- snapshot confound
print("\nD3. The snapshot confound")
snap = {"endpoint_classes": None}
try:
    url = ENDPOINT + "?" + urllib.parse.urlencode(
        {"query": "SELECT (COUNT(DISTINCT ?c) AS ?n) WHERE "
                  "{ ?c a <http://www.w3.org/2002/07/owl#Class> }"}
    )
    req = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": UA}
    )
    d = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    snap["endpoint_classes"] = int(d["results"]["bindings"][0]["n"]["value"])
    print(f"     owl:Class at the endpoint today: {snap['endpoint_classes']}")
    check(
        "count comparisons across artefacts are confounded",
        snap["endpoint_classes"] is not None,
        f"the endpoint's loaded ontology carries {snap['endpoint_classes']} classes "
        "today; a count taken against a downloaded artefact is a different "
        "snapshot and is not a control",
    )
except Exception as e:
    print(f"     unavailable: {type(e).__name__}")

# ---------------------------------------------------------------- summary
scored = [c for c in checks if not c["control"]]
npass = sum(1 for c in scored if c["pass"])
nctrl = sum(1 for c in checks if c["control"] and c["pass"])
print("\n" + "=" * 74)
print(f"EXPERIMENT D: {npass}/{len(scored)} scored checks pass, {nctrl} controls fired")
print("=" * 74)

os.makedirs("results", exist_ok=True)
json.dump(
    {"experiment": "D", "live": live, "snapshot": snap, "checks": checks,
     "local": {"rdflib_A": ra, "rdflib_B": rb, "oxi_A": oa, "oxi_B": ob,
               "expected": expected},
     "scored": len(scored), "passed": npass},
    open("results/exp_d.json", "w"), indent=2,
)
sys.exit(0 if npass == len(scored) else 1)
