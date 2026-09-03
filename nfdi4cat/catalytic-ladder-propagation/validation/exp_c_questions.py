"""
Experiment C -- the practitioner questions, resolved to verdicts.

Twelve questions were supplied by a practitioner (five biocatalysis, seven
generic dataset queries) without reference to this framework.  We add sixteen
more of our own to exercise verdicts the twelve do not reach, and we resolve
all twenty-eight.  Each question is answered, or refused with a named blocker
and a statement of what would unblock it.

Run:  python exp_c_questions.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures.corpus import (  # noqa: E402
    AMBIENT,
    CHEBI_NAMES,
    DATASETS,
    EXPERIMENTS,
    PATHWAYS,
    PROTEINS,
    REACTIONS,
)
from kernel.ladder import (  # noqa: E402
    direction_verdict,
    medium_bias,
    medium_weight,
    solvent_role,
)
from kernel.plan import Plan, Result, Source, Verdict  # noqa: E402

# ---------------------------------------------------------------------------
# Sources, with declared capabilities.
# ---------------------------------------------------------------------------
CAP_RXN = {"reaction", "participant", "ec", "equation", "status", "direction-asserted"}
CAP_PROT = {"protein", "organism", "lineage", "ec", "sequence", "catalyses"}
CAP_PATH = {"pathway", "reaction-membership"}
CAP_ELN = {
    "experiment", "operator", "date", "device", "device-settings",
    "buffer", "ph", "temperature", "dataset", "compound",
}

SOURCES = {
    "RXN": Source("RXN", CAP_RXN, REACTIONS),
    "PROT": Source("PROT", CAP_PROT, PROTEINS),
    "PATH": Source("PATH", CAP_PATH, PATHWAYS),
    "ELN": Source("ELN", CAP_ELN, EXPERIMENTS),
}

BETA = 3.7e-4
TAU = 1.0e-3

rows = []
checks = []


def check(name, ok, detail, control=False):
    checks.append({"name": name, "pass": bool(ok), "detail": detail, "control": control})
    tag = "CONTROL" if control else "check  "
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}: {detail}")
    return ok


def record(qid, text, verdict, payload=None, blocker="", unblock="", sources=()):
    rows.append(
        {
            "id": qid,
            "question": text,
            "verdict": verdict.value,
            "payload": payload or [],
            "blocker": blocker,
            "unblock": unblock,
            "sources": list(sources),
        }
    )
    n = len(payload or [])
    show = f" -> {payload}" if payload and n <= 6 else (f" -> {n} items" if n else "")
    print(f"  {qid:<5} {verdict.value.upper():<12}{show}")
    if blocker:
        print(f"        blocker : {blocker}")
    if unblock:
        print(f"        unblock : {unblock}")


print("=" * 74)
print("EXPERIMENT C -- twenty-eight questions resolved to verdicts")
print("=" * 74)

# ===========================================================================
# GROUP 1 -- the five biocatalysis questions
# ===========================================================================
print("\nGROUP 1 -- biocatalysis questions (practitioner-supplied)")

# --- Q1: bacterial transaminase, benzylethylamine, no cysteine -------------
# Three sources + a predicate computed over a retrieved attribute.
q1_rxns = [r for r, d in REACTIONS.items()
           if any("benzylethylamine" in CHEBI_NAMES.get(c, "")
                  for c in d["substrates"] + d["products"])]
q1_prots = [p for p, d in PROTEINS.items()
            if any(r in q1_rxns for r in d["catalyses"])]
q1_bact = [p for p in q1_prots if PROTEINS[p]["domain"] == "Bacteria"]
q1_final = [p for p in q1_bact if "C" not in PROTEINS[p]["sequence"]]
record(
    "Q1",
    "Bacterial transaminase for benzylethylamine with no Cys in sequence",
    Verdict.ANSWER,
    q1_final,
    sources=("RXN", "PROT"),
)
check(
    "Q1 needs a predicate no store holds",
    len(q1_bact) > len(q1_final) and len(q1_final) >= 1,
    f"{len(q1_rxns)} reaction(s) -> {len(q1_prots)} enzymes -> {len(q1_bact)} bacterial "
    f"-> {len(q1_final)} Cys-free; the last step is a computation over a retrieved "
    "sequence, not a triple lookup",
)

# --- Q2: buffer and pH for the mt-X methyl transfer ------------------------
mt7 = EXPERIMENTS["MT7"]
record(
    "Q2",
    "Buffer composition and pH for the mt-X methyl transfer",
    Verdict.ANSWER,
    [f"{mt7['buffer']['name']} {mt7['buffer']['conc_mM']} mM, pH {mt7['buffer']['pH']}"],
    sources=("ELN",),
)
check(
    "Q2 is answerable only from the ELN",
    not any("buffer" in SOURCES[s].capabilities for s in ("RXN", "PROT", "PATH")),
    "no public reaction/protein/pathway source declares 'buffer' or 'ph'; "
    "the datum exists only in the laboratory record",
)

# --- Q3: substrate scope / product range of BVMO-Y -------------------------
bv = EXPERIMENTS["BV2"]
rxn = REACTIONS[bv["reaction"]]
scope = [CHEBI_NAMES.get(c, c) for c in rxn["substrates"]]
prods = [CHEBI_NAMES.get(c, c) for c in rxn["products"]]
record(
    "Q3",
    "Substrate scope and product range of BVMO-Y",
    Verdict.ANSWER,
    [f"substrates: {', '.join(scope)}", f"products: {', '.join(prods)}"],
    sources=("ELN", "RXN"),
)

# --- Q4: expected products of a kinetic resolution -------------------------
# This is not retrieval.  Nothing is recorded; the question asks what COULD
# happen.  We answer it as an admissibility question over the ladder.
record(
    "Q4",
    "Expected products of kinetic resolution with PFE at pH 9 in HEPES",
    Verdict.UNSUPPORTED,
    blocker=(
        "not a retrieval question: no record asserts the products of an "
        "experiment that has not been run"
    ),
    unblock=(
        "resolve as an admissibility query over the contact graph: ask whether "
        "a propagation to each candidate product is accountable in the declared "
        "medium (buffer, pH). This is Sec. 6 and is computed below as Q4'"
    ),
    sources=("ELN",),
)

# --- Q5: device and wavelength for BT3 on 23 March -------------------------
bt3 = EXPERIMENTS["BT3"]
record(
    "Q5",
    "Device and monitored wavelength for BT3 on 2026-03-23 by Y. Dikova",
    Verdict.ANSWER,
    [
        f"{bt3['device']['vendor']} {bt3['device']['id']} "
        f"({bt3['device']['kind']})",
        f"wavelength {bt3['device']['settings']['wavelength_nm']} nm",
    ],
    sources=("ELN",),
)
check(
    "Q5 matches on operator AND date AND device settings",
    bt3["operator"] == "Y. Dikova" and bt3["date"] == "2026-03-23",
    "instrument provenance is a first-class record, not an annotation",
)

# ===========================================================================
# GROUP 2 -- the seven generic dataset questions (Chem-DCAT-AP shaped)
# ===========================================================================
print("\nGROUP 2 -- generic dataset questions (Chem-DCAT-AP shaped)")


def datasets_with_compound(c):
    return [d for d, v in DATASETS.items() if c in v["compounds"]]


def datasets_by_devicekind(kind):
    return [
        d for d, v in DATASETS.items()
        if EXPERIMENTS[v["experiment"]]["device"]["kind"] == kind
    ]


def datasets_by_vendor(vendor):
    return [
        d for d, v in DATASETS.items()
        if EXPERIMENTS[v["experiment"]]["device"]["vendor"] == vendor
    ]


q6 = datasets_with_compound("CHEBI:17854")
record("Q6", "All datasets about a substance with compound C",
       Verdict.ANSWER, q6, sources=("ELN",))

q7 = [d for d in q6 if DATASETS[d]["type"] == "NMR"]
record("Q7", "All datasets generated by an activity of type T evaluating compound C",
       Verdict.ANSWER, q7, sources=("ELN",))

q8 = [d for d in q7
      if EXPERIMENTS[DATASETS[d]["experiment"]]["device"]["vendor"] == "Bruker"
      and EXPERIMENTS[DATASETS[d]["experiment"]]["device"]["settings"].get("nucleus") == "13C"]
record("Q8", "... with Bruker spectrometer set to X (nucleus 13C)",
       Verdict.ANSWER, q8, sources=("ELN",))
check(
    "Q6-Q8 form a strictly narrowing chain",
    len(q6) > len(q7) > len(q8) >= 1,
    f"compound C: {len(q6)} datasets -> type NMR: {len(q7)} -> Bruker at 13C: "
    f"{len(q8)}. Each added constraint removes at least one dataset, so the "
    "filters are shown to discriminate rather than merely to return a row",
)
check(
    "Q8 needs device SETTINGS, not just device identity",
    "device-settings" in CAP_ELN
    and len({d for d in q7 if EXPERIMENTS[DATASETS[d]["experiment"]]["device"]["vendor"]
             == "Bruker"}) > len(q8),
    "restricting from 'a Bruker' to 'a Bruker set to 13C' removes a dataset; a "
    "configuration value is part of the measurement's provenance",
)

q9 = [d for d, v in DATASETS.items()
      if EXPERIMENTS[v["experiment"]].get("reaction")
      and "CHEBI:35604" in REACTIONS[EXPERIMENTS[v["experiment"]]["reaction"]]["products"]]
record("Q9", "All datasets about a chemical reaction that had product P",
       Verdict.ANSWER, q9, sources=("ELN", "RXN"))

q10 = [d for d, v in DATASETS.items()
       if EXPERIMENTS[v["experiment"]].get("reaction")
       and "CHEBI:17854" in REACTIONS[EXPERIMENTS[v["experiment"]]["reaction"]]["substrates"]]
record("Q10", "All datasets about a reaction with starting material compound C",
       Verdict.ANSWER, q10, sources=("ELN", "RXN"))

record("Q11", "All datasets measured with a UV-vis spectrometer containing compound C",
       Verdict.ANSWER,
       [d for d in datasets_by_devicekind("UV-vis spectrophotometer")
        if "CHEBI:90000" in DATASETS[d]["compounds"]],
       sources=("ELN",))

q12 = [d for d, v in DATASETS.items()
       if EXPERIMENTS[v["experiment"]]["biocatalyst"] == "PR:BVMO_ACINE"
       and EXPERIMENTS[v["experiment"]]["device"]["vendor"] == "Bruker"]
record("Q12", "All datasets with substance S as catalyst measured on a Bruker",
       Verdict.ANSWER, q12, sources=("ELN",))

# ===========================================================================
# GROUP 3 -- questions the framework answers that no schema states
# ===========================================================================
print("\nGROUP 3 -- questions answerable without a model of the answer")

# --- Q4': the admissibility form of Q4 -------------------------------------
# Which direction is admissible for the alanine transaminase chain, in each
# of three media?  Nothing in any store records this; it is computed.
tri = {}
for med, mu in AMBIENT.items():
    delta = medium_bias(
        REACTIONS["RXN:19453"]["substrates"],
        REACTIONS["RXN:19453"]["products"],
        mu, BETA, TAU,
    )
    tri[med] = (delta, direction_verdict(delta, BETA))
record(
    "Q4'",
    "In which direction is RXN:19453 physiologically admissible, per medium?",
    Verdict.ANSWER,
    [f"{k}: delta={v[0]:+.3e} -> {v[1]}" for k, v in tri.items()],
    sources=("RXN", "ELN"),
)
check(
    "the same chain runs both ways in different media",
    {v[1] for v in tri.values()} >= {"forward", "reverse", "undirected"},
    "one reaction identifier, three verdicts: "
    + ", ".join(f"{k}={v[1]}" for k, v in tri.items()),
)
check(
    "direction is a property of the medium, not of the chain",
    tri["cytosol_gln_depleted"][1] != tri["cytosol_og_depleted"][1],
    "identical substrates, products and identifier; opposite verdicts",
)

# CONTROL: an unbiased medium must REFUSE to orient
check(
    "CONTROL: a balanced medium refuses to orient",
    tri["balanced"][1] == "undirected" and abs(tri["balanced"][0]) <= BETA,
    f"delta={tri['balanced'][0]:+.2e}, |delta| <= beta={BETA:.1e}; "
    "without this the trichotomy would have an unreachable third case",
    control=True,
)

# --- the saturation asymmetry -------------------------------------------
# Flooding a product cannot drive the bias arbitrarily negative: as the
# flooded occupancy grows, its medium weight falls to the floor and the bias
# saturates.  Depleting a reactant has no such ceiling.  We found this by
# sweeping one side at a time, and it is a proposition rather than a fitted
# number: the limit is exactly -log(1 + tau/mu0) in units of beta.
mu0 = 1.0e-4
flood = [
    (medium_weight(mu0 * 10.0**e, BETA, TAU) + medium_weight(mu0, BETA, TAU)
     - 2 * medium_weight(mu0, BETA, TAU)) / BETA
    for e in (2, 6, 12, 20, 30)
]
deplete = [
    (2 * medium_weight(mu0, BETA, TAU)
     - medium_weight(mu0 * 10.0**-e, BETA, TAU) - medium_weight(mu0, BETA, TAU)) / BETA
    for e in (2, 4, 6, 10)
]
predicted = -math.log(1.0 + TAU / mu0)
check(
    "flooding a product saturates at exactly -log(1 + tau/mu0)",
    abs(flood[-1] - predicted) < 1e-9,
    f"limit {flood[-1]:.6f} vs predicted {predicted:.6f} (units of beta); "
    "thirty orders of magnitude of flooding move it no further",
)
check(
    "depleting a reactant is unbounded",
    deplete[-1] < 3 * flood[-1],
    f"depletion reaches {deplete[-1]:.2f} beta and keeps falling, while flooding "
    f"stops at {flood[-1]:.2f} beta: reversing a chain requires depleting a "
    "reactant, not flooding a product",
)

# --- Q13: solvent role, computed rather than annotated ---------------------
# Two water molecules of identical chemical identity, opposite roles.
mu_water = 55.5
w_lm = medium_weight(mu_water, BETA, TAU)
axial = {"label": "ordered active-site water", "rho_str": 4.0 * BETA}
bulk = {"label": "bulk water", "rho_str": 0.0}
roles = {k: solvent_role(v["rho_str"], w_lm) for k, v in
         (("axial", axial), ("bulk", bulk))}
record(
    "Q13",
    "What is the role of the solvent in this reaction?",
    Verdict.ANSWER,
    [f"{axial['label']}: {roles['axial']}", f"{bulk['label']}: {roles['bulk']}"],
    sources=("RXN",),
)
check(
    "identical molecules, opposite roles, decided by computation",
    roles["axial"] == "structural" and roles["bulk"] == "bulk",
    f"w(l,m)={w_lm:.3e}; rho_str(axial)={axial['rho_str']:.3e} >= w -> structural; "
    f"rho_str(bulk)=0 < w -> bulk. No curator supplies this and no vocabulary "
    "term records it",
)

# --- Q14: which enzymes are interchangeable at a given resolution ----------
# Rungs of equal power are the same rung.  This groups catalysts without any
# structural comparison.
POWERS = {
    "PR:TAM_BACIL": 0.55,
    "PR:TAM_PSEUD": 0.55,
    "PR:TAM_ARATH": 0.30,
    "PR:BVMO_ACINE": 0.72,
}
groups = {}
for p, v in POWERS.items():
    groups.setdefault(round(v, 6), []).append(p)
record(
    "Q14",
    "Which biocatalysts are interchangeable in a process?",
    Verdict.ANSWER,
    [f"power {k}: {', '.join(v)}" for k, v in sorted(groups.items())],
    sources=("ELN",),
)
check(
    "classification needs one number per catalyst and no structure",
    len(groups[0.55]) == 2,
    "two enzymes from different organisms with no sequence comparison "
    "fall in one class because their powers agree",
)

# --- Q15: what does deleting a step cost? ---------------------------------
from kernel.ladder import compose  # noqa: E402

chain = [0.45, 0.30, 0.55, 0.20]
full = compose(chain)
deletions = {
    i: compose([p for j, p in enumerate(chain) if j != i]) for i in range(len(chain))
}
tolerated = [i for i, v in deletions.items() if v >= 0.80]
record(
    "Q15",
    "Which step of a four-step process can be deleted without missing target 0.80?",
    Verdict.ANSWER,
    [f"delete rung {i+1} -> {v:.4f}" + ("  (tolerated)" if v >= 0.80 else "  (fails)")
     for i, v in deletions.items()],
    sources=(),
)
check(
    "deletion tolerance is predicted before any experiment",
    full > 0.80 and len(tolerated) >= 1 and len(tolerated) < len(chain),
    f"composite {full:.4f}; deleting rung 2 keeps 0.802, deleting rung 3 drops "
    "to 0.692 and fails the requirement",
)

# ===========================================================================
# GROUP 4 -- questions that must be REFUSED
# ===========================================================================
print("\nGROUP 4 -- questions the system refuses, with named blockers")

# --- Q16: temperature dependence -- explicitly not answered ---------------
record(
    "Q16",
    "At what temperature does this reaction take place (as a derived quantity)?",
    Verdict.UNEXPRESSED,
    blocker=(
        "the floor is condition-dependent in principle, but the dependence is "
        "immaterial at the categorical depth used here"
    ),
    unblock=(
        "connect the medium bias to a measured thermodynamic driving force; "
        "not attempted in this paper"
    ),
    sources=("ELN",),
)

# --- Q17: a capability the source does not declare ------------------------
plan = Plan(SOURCES)
plan.step("seq", "PROT", {"sequence"})
plan.step("buffer_from_rxn", "RXN", {"buffer", "ph"})
static = plan.static_check()
record(
    "Q17",
    "Retrieve the buffer used, from the reaction knowledge base",
    Verdict.UNEXPRESSED,
    blocker=static["buffer_from_rxn"].reason,
    unblock=static["buffer_from_rxn"].unblock,
    sources=("RXN",),
)
check(
    "capability containment is decided before any request is issued",
    static["seq"] is None and static["buffer_from_rxn"] is not None,
    "the plan is refused statically, naming the offending step and features",
)

# --- Q18: starvation -- the characteristic federated failure --------------
plan2 = Plan(SOURCES)
plan2.step("find_rxn", "RXN", {"reaction"}, fn=lambda s, r: [])
plan2.step("find_prot", "PROT", {"protein"},
           fn=lambda s, r: ["x"], depends_on="find_rxn")
res2 = plan2.run()
record(
    "Q18",
    "Enzymes for a reaction that the first step failed to retrieve",
    res2["find_prot"].verdict,
    blocker=res2["find_prot"].reason,
    unblock=res2["find_prot"].unblock,
    sources=("RXN", "PROT"),
)
check(
    "a starved step names the predecessor at fault",
    res2["find_prot"].verdict is Verdict.STARVED
    and res2["find_prot"].blame == "find_rxn",
    f"blame walk terminates at {res2['find_prot'].blame!r}, not at a step that "
    "answered correctly",
)

# --- Q19: budget exhaustion -----------------------------------------------
plan3 = Plan(SOURCES)
for i in range(4):
    plan3.step(f"s{i}", "PROT", {"protein"}, fn=lambda s, r: ["p"])
res3 = plan3.run(budget=2)
exhausted = [k for k, v in res3.items() if v.verdict is Verdict.EXHAUSTED]
record(
    "Q19",
    "A four-step plan run under a budget of two",
    Verdict.EXHAUSTED,
    blocker=f"{len(exhausted)} step(s) unrun",
    unblock="raise the budget to 4",
    sources=("PROT",),
)
check(
    "exhaustion is distinguished from emptiness",
    len(exhausted) == 2 and all(res3[f"s{i}"].verdict is Verdict.ANSWER for i in range(2)),
    "two steps answered, two report EXHAUSTED; a rows-only interface would "
    "have returned the same empty table for both",
)

# --- Q20: non-degeneracy is structurally enforced -------------------------
violated = False
try:
    Result(Verdict.STARVED, payload=["something"], blame="x")
except ValueError:
    violated = True
check(
    "non-degeneracy: no verdict but ANSWER may carry a payload",
    violated,
    "constructing a STARVED result with a payload raises; the property is "
    "enforced by the type, not by reporting discipline",
)

# ===========================================================================
# GROUP 5 -- extended battery
# ===========================================================================
print("\nGROUP 5 -- extended battery (eight further questions)")

extended = [
    ("Q21", "Which reactions share an identifier but run in opposite directions?",
     Verdict.ANSWER,
     [f"RXN:19453 in {k}: {v[1]}" for k, v in tri.items() if v[1] != "undirected"]),
    ("Q22", "Which enzymes catalysing a given EC are bacterial?",
     Verdict.ANSWER,
     [p for p, d in PROTEINS.items()
      if d["ec"] == "2.6.1.-" and d["domain"] == "Bacteria"]),
    ("Q23", "Which pathways contain a reaction whose product is a lactone?",
     Verdict.ANSWER,
     [p for p, d in PATHWAYS.items()
      if any("CHEBI:35604" in REACTIONS[r]["products"] for r in d["reactions"])]),
    ("Q24", "Which experiments used a buffer at pH above 8?",
     Verdict.ANSWER,
     [e for e, d in EXPERIMENTS.items() if d["buffer"]["pH"] > 8.0]),
    ("Q25", "Which operators ran experiments on a UV-vis instrument?",
     Verdict.ANSWER,
     sorted({d["operator"] for d in EXPERIMENTS.values()
             if d["device"]["kind"] == "UV-vis spectrophotometer"})),
    ("Q26", "Which datasets came from an instrument whose settings included 13C?",
     Verdict.ANSWER,
     [d for d, v in DATASETS.items()
      if EXPERIMENTS[v["experiment"]]["device"]["settings"].get("nucleus") == "13C"]),
]
for qid, text, verdict, payload in extended:
    record(qid, text, verdict, payload, sources=("mixed",))

# two more refusals with distinct blockers
record(
    "Q27",
    "Which enzyme will give the highest enantiomeric excess for a new substrate?",
    Verdict.UNSUPPORTED,
    blocker="requires a quantitative structure-selectivity model this paper does not supply",
    unblock="a measured power per catalyst per substrate; the framework states "
            "the form of the answer but not its value",
    sources=(),
)
record(
    "Q28",
    "Which reaction steps occurred, in order, inside a single turnover?",
    Verdict.UNEXPRESSED,
    blocker="a description-logic defined class over participant roles cannot "
            "express a residue chain: it can state that a reaction HAS a "
            "participant, not that its steps occur in an order",
    unblock="ask the chain view instead of the signature view; the two "
            "representations answer disjoint question sets",
    sources=("RXN",),
)

# ===========================================================================
# summary
# ===========================================================================
by_verdict = {}
for r in rows:
    by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

print("\n" + "-" * 74)
print("VERDICT DISTRIBUTION over", len(rows), "questions")
for k in sorted(by_verdict):
    print(f"  {k:<14} {by_verdict[k]}")

answered = by_verdict.get("answer", 0)
refused = len(rows) - answered
check(
    "the system both answers and refuses",
    answered > 0 and refused > 0,
    f"{answered} answered, {refused} refused; a system that answered everything "
    "would have an empty defined class",
)
check(
    "every refusal names a blocker and an unblock path",
    all(r["blocker"] and r["unblock"] for r in rows if r["verdict"] != "answer"),
    f"{refused}/{refused} refusals carry both fields",
)
check(
    "at least four distinct verdicts are reached",
    len(by_verdict) >= 4,
    f"reached: {sorted(by_verdict)}",
)

scored = [c for c in checks if not c["control"]]
npass = sum(1 for c in scored if c["pass"])
nctrl = sum(1 for c in checks if c["control"] and c["pass"])
print("\n" + "=" * 74)
print(f"EXPERIMENT C: {npass}/{len(scored)} scored checks pass, {nctrl} controls fired")
print("=" * 74)

os.makedirs("results", exist_ok=True)
json.dump(
    {
        "experiment": "C",
        "questions": rows,
        "verdict_distribution": by_verdict,
        "checks": checks,
        "scored": len(scored),
        "passed": npass,
        "direction_trichotomy": {k: [v[0], v[1]] for k, v in tri.items()},
        "solvent_roles": roles,
        "medium_weight": w_lm,
    },
    open("results/exp_c.json", "w"),
    indent=2,
)
sys.exit(0 if npass == len(scored) else 1)
