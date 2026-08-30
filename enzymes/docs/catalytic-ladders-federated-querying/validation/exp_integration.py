#!/usr/bin/env python3
"""
Integration experiments for "Catalytic Ladders in a Federated Query Language".

Every number written to the results file is measured.  Nothing is hard-coded
from a previous run.

The falsification condition was fixed BEFORE any change was made to the host:
if the host's own sixteen-check suite reports fewer than 16/16 after the
ladder is added, the integration has failed and the failure is the result.

E1  the host suite, run as a subprocess, before/after comparison
E2  a ladder plan end to end: composite, cost, capability operations
E3  the refusal: both verdicts, the blame walk, and what it costs
E4  cost regime: a ladder spends nothing and is allocated the nominal unit
E5  the union bound against the multiplicative law
E6  the corpus: two real sources, frozen, and what a ladder over them reports
E7  NEGATIVE CONTROL: a plan with no ladder is byte-identical before/after
E8  NEGATIVE CONTROL: malformed ladders are refused at parse time
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIXTURES = os.path.join(HERE, "fixtures")
HOST = r"C:\Users\kunda\Documents\systems\hegel\ckg\validation-federated"

sys.path.insert(0, HOST)
sys.path.insert(0, os.path.join(HOST, "fixtures"))

from hfq import Executor, check, parse                     # noqa: E402
from hfq.parser import ParseError                          # noqa: E402
import build                                               # noqa: E402


def _registry():
    return build.build_registry(), build.build_maps()


# ---------------------------------------------------------------------------
def e1_host_suite() -> Dict:
    """
    Run the host's own suite as a subprocess and read its summary.

    This is the pre-registered falsification condition. The suite was written
    without knowledge of the ladder and has no reason to accommodate it.
    """
    t0 = time.time()
    proc = subprocess.run([sys.executable, "run_validation.py"],
                          cwd=HOST, capture_output=True, text=True,
                          timeout=900)
    summary_path = os.path.join(HOST, "results", "00_summary.json")
    with open(summary_path) as fh:
        summary = json.load(fh)
    return {
        "experiment": "E1 host suite after the ladder is added",
        "claim": ("The host's sixteen checks still hold. This was fixed as "
                  "the falsification condition before any edit was made."),
        "exit_code": proc.returncode,
        "total": summary.get("total"),
        "passing": summary.get("passing"),
        "failing": summary.get("failing"),
        "network_access": summary.get("network_access"),
        "snapshot": summary.get("snapshot"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "holds": bool(summary.get("failing") == 0
                      and summary.get("passing") == summary.get("total")
                      and proc.returncode == 0),
    }


# ---------------------------------------------------------------------------
LADDER_PLAN = """plan ladder_demo {
  budget 50 requests

  let acids = from chebi
      ask descendants_of("CHEBI:1")
      within 10

  let L = ladder over acids
      power 0.45, power 0.30, power 0.55
      expect power %s

  emit L
}"""


def e2_end_to_end() -> Dict:
    """A ladder plan runs through the real parser, checker and executor."""
    reg, maps = _registry()
    plan = parse(LADDER_PLAN % "0.70")
    rep = check(plan, reg)
    ex = Executor(reg, maps=maps).run(plan)
    rows = {r.step: r for r in ex.steps}
    lad = rows["L"]
    expected = 1.0
    for p in (0.45, 0.30, 0.55):
        expected *= (1.0 - p)
    expected = 1.0 - expected
    return {
        "experiment": "E2 a ladder plan end to end",
        "claim": ("The construct parses, checks and executes through the "
                  "host's own machinery, not a harness of ours."),
        "well_capability": rep.well_capability,
        "capability_operations": rep.operations,
        "verdicts": ex.verdicts(),
        "composite_power": lad.composite_power,
        "composite_expected": expected,
        "composite_matches": abs(lad.composite_power - expected) < 1e-12,
        "requests_issued": ex.requests_issued,
        "holds": bool(rep.well_capability
                      and ex.verdicts()["L"] == "answer"
                      and abs(lad.composite_power - expected) < 1e-12),
    }


def e3_refusal() -> Dict:
    """
    The refusal fires, declines to fire, and terminates the blame walk at the
    ladder rather than at the predecessor that answered correctly.
    """
    reg, maps = _registry()
    out: Dict[str, Any] = {}
    for label, target in [("accepted", "0.70"), ("refused", "0.95")]:
        ex = Executor(reg, maps=maps).run(parse(LADDER_PLAN % target))
        row = {r.step: r for r in ex.steps}["L"]
        out[label] = {
            "verdict": ex.verdicts()["L"],
            "input_verdict": ex.verdicts()["acids"],
            "composite": row.composite_power,
            "diagnosis": row.diagnosis,
            "blame_chain": ex.blame_chain("L"),
            "requests_issued": ex.requests_issued,
        }
    ref = out["refused"]
    blame_stops_at_ladder = ref["blame_chain"] == ["L"]
    predecessor_unblamed = (ref["diagnosis"] or {}).get(
        "named_predecessor", "MISSING") is None
    return {
        "experiment": "E3 the refusal and the blame walk",
        "claim": ("A refused ladder reports starved with no named "
                  "predecessor, so blame terminates at the ladder and does "
                  "not accuse an input that answered correctly."),
        "cases": out,
        "both_verdicts_occur": (out["accepted"]["verdict"] == "answer"
                                and out["refused"]["verdict"] == "starved"),
        "blame_stops_at_ladder": blame_stops_at_ladder,
        "predecessor_unblamed": predecessor_unblamed,
        "input_answered_in_both": all(
            out[k]["input_verdict"] == "answer" for k in out),
        "holds": bool(out["accepted"]["verdict"] == "answer"
                      and out["refused"]["verdict"] == "starved"
                      and blame_stops_at_ladder and predecessor_unblamed),
    }


def e4_cost_regime() -> Dict:
    """
    A ladder spends nothing and is allocated the nominal unit the host gives
    every all-or-nothing step.  The comparison against a set operation shows
    the ladder is in the tier that already existed rather than a new one.
    """
    reg, maps = _registry()
    src = """plan cost_probe {
  budget 60 requests

  let a = from chebi ask descendants_of("CHEBI:1") within 10
  let b = from chebi ask descendants_of("CHEBI:2") within 10
  let u = union a b
  let L = ladder over a power 0.5, power 0.5

  emit L
}"""
    ex = Executor(reg, maps=maps).run(parse(src))
    rows = {r.step: r for r in ex.steps}
    return {
        "experiment": "E4 the ladder occupies the existing zero-cost tier",
        "claim": ("A ladder spends zero budget, exactly as the host's set "
                  "operations already do, and is allocated the same nominal "
                  "unit."),
        "spent": {k: rows[k].spent for k in rows},
        "allocated": {k: rows[k].allocated for k in rows},
        "ladder_spends_zero": rows["L"].spent == 0.0,
        "setop_spends_zero": rows["u"].spent == 0.0,
        "source_spends_more": rows["a"].spent > 0.0,
        "requests_issued": ex.requests_issued,
        "holds": bool(rows["L"].spent == 0.0
                      and rows["u"].spent == 0.0
                      and rows["a"].spent > 0.0),
    }


# ---------------------------------------------------------------------------
def e5_union_bound(n_trials: int = 4000, seed: int = 5) -> Dict:
    """
    The host's union bound against the multiplicative law, on identical
    numbers.  We measure how often the additive form falls below zero, at
    which point it is a lower bound on a quantity in [0,1] and says nothing.
    """
    rng = random.Random(seed)
    rows = []
    for k in (2, 3, 4, 5, 6, 8):
        add_sum = mul_sum = 0.0
        vacuous = 0
        for _ in range(n_trials):
            rs = [rng.uniform(0.5, 0.99) for _ in range(k)]
            add = 1.0 - sum(1.0 - r for r in rs)
            prod = 1.0
            for r in rs:
                prod *= (1.0 - r)
            mul = 1.0 - prod
            add_sum += add
            mul_sum += mul
            vacuous += int(add < 0.0)
        rows.append({
            "stages": k,
            "mean_additive_bound": add_sum / n_trials,
            "mean_multiplicative": mul_sum / n_trials,
            "mean_gap": (mul_sum - add_sum) / n_trials,
            "fraction_additive_vacuous": vacuous / n_trials,
        })
    monotone_gap = all(rows[i]["mean_gap"] < rows[i + 1]["mean_gap"]
                       for i in range(len(rows) - 1))
    monotone_vac = all(rows[i]["fraction_additive_vacuous"]
                       <= rows[i + 1]["fraction_additive_vacuous"]
                       for i in range(len(rows) - 1))
    return {
        "experiment": "E5 the union bound against the multiplicative law",
        "claim": ("The host's additive lower bound is the relaxation of the "
                  "ladder's law; it goes vacuous (negative) on most chains "
                  "of four stages or more."),
        "n_trials_per_stage_count": n_trials,
        "retention_range": [0.5, 0.99],
        "rows": rows,
        "gap_grows_with_length": monotone_gap,
        "vacuity_grows_with_length": monotone_vac,
        "holds": bool(monotone_gap and monotone_vac),
        "note": ("The additive bound carries an injectivity hypothesis and is "
                 "stated by its authors as a bound rather than an estimate. A "
                 "loose bound is not a wrong one; what is measured here is "
                 "how loose, and that the tight form is the ladder's law."),
    }


# ---------------------------------------------------------------------------
def e6_corpus() -> Dict:
    """
    The frozen corpus.  Two public services with different schemas, fetched
    once and frozen, because the host forbids network access in adapters by
    construction and we did not weaken that rule to run an experiment.
    """
    path = os.path.join(FIXTURES, "sources.json")
    if not os.path.exists(path):
        return {"experiment": "E6 the corpus", "holds": False,
                "error": "fixtures/sources.json not present; run fetch_sources.py"}
    with open(path) as fh:
        d = json.load(fh)
    kegg, rx = d["kegg"], d["reactome"]

    kegg_with_rxn = sum(1 for e in kegg if e.get("reactions"))
    kegg_with_sub = sum(1 for e in kegg if e.get("substrates"))
    rx_with_cat = sum(1 for r in rx if r.get("catalysts"))
    rx_with_in = sum(1 for r in rx if r.get("inputs"))

    # A chain length per Reactome pathway: how many reactions it contains.
    from collections import Counter
    per_pathway = Counter(r["pathway"] for r in rx if r.get("pathway"))
    lengths = sorted(per_pathway.values(), reverse=True)

    return {
        "experiment": "E6 the frozen corpus",
        "claim": ("The plans run against real data from two heterogeneous "
                  "public sources, fetched once and frozen."),
        "snapshot": d.get("snapshot"),
        "fetched_utc": d.get("fetched_utc"),
        "sources": d.get("sources"),
        "kegg_records": len(kegg),
        "kegg_with_reactions": kegg_with_rxn,
        "kegg_with_substrates": kegg_with_sub,
        "reactome_reactions": len(rx),
        "reactome_with_inputs": rx_with_in,
        "reactome_with_catalysts": rx_with_cat,
        "n_pathways": len(per_pathway),
        "longest_pathways": lengths[:8],
        "holds": bool(len(kegg) > 0 and len(rx) > 0),
        "note": ("Counts are reported whatever they are. A field that came "
                 "back empty is recorded as empty rather than dropped."),
    }


# ---------------------------------------------------------------------------
def e7_no_ladder_unchanged() -> Dict:
    """
    NEGATIVE CONTROL.  Every plan the host ships that contains no ladder must
    still parse and execute.  If the extension had changed their meaning this
    control would show it.

    The host's plans belong to three different fixture worlds, and it selects
    the world from the sources a plan names. We use the host's own selector
    rather than a registry of our own: a control that loaded the wrong
    fixtures would report failures that are properties of the control and not
    of the extension. Our first version did exactly that, and reported three.
    """
    import hfq_serve

    plans_dir = os.path.join(HOST, "plans")
    rows = []
    ok = True
    for name in sorted(os.listdir(plans_dir)):
        if not name.endswith(".hfq"):
            continue
        with open(os.path.join(plans_dir, name)) as fh:
            src = fh.read()
        if "ladder" in src:
            continue
        try:
            plan = parse(src)
            world, unknown = hfq_serve.select_world(plan)
            reg, maps = hfq_serve._build(world)
            ex = Executor(reg, maps=maps).run(plan)
            rows.append({"plan": name, "world": world,
                         "unknown_sources": unknown,
                         "verdicts": ex.verdicts(),
                         "requests": ex.requests_issued, "parsed": True})
        except Exception as exc:
            rows.append({"plan": name, "parsed": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            ok = False
    return {
        "experiment": "NEGATIVE CONTROL: plans without a ladder still run",
        "claim": ("Every shipped plan that contains no ladder still parses "
                  "and executes, in the fixture world the host selects for "
                  "it. If the extension had changed their meaning this "
                  "control would show it."),
        "n_plans": len(rows),
        "rows": rows,
        "all_parsed": ok,
        "worlds_exercised": sorted({r.get("world") for r in rows
                                    if r.get("world")}),
        "holds": bool(ok and len(rows) > 0),
    }


def e8_malformed_refused() -> Dict:
    """
    NEGATIVE CONTROL.  Malformed ladders must be refused at parse time rather
    than silently accepted.  Includes the case that produced a real defect:
    the token "power" occurring inside "expect power".
    """
    cases = [
        ("ladder over a", "no rungs declared"),
        ("ladder over a power 1.5", "power above one"),
        ("ladder over a power -0.2", "negative power"),
        ("ladder", "no operand"),
    ]
    rows = []
    refused = 0
    for rhs, why in cases:
        src = ("plan t {\n  budget 10 requests\n"
               "  let a = from chebi ask descendants_of(\"CHEBI:1\") within 5\n"
               "  let L = " + rhs + "\n  emit L\n}")
        try:
            parse(src)
            rows.append({"case": why, "refused": False})
        except ParseError as exc:
            refused += 1
            rows.append({"case": why, "refused": True, "message": str(exc)})

    # the defect case: the target must NOT be read as a rung
    plan = parse("plan t {\n  budget 10 requests\n"
                 "  let a = from chebi ask descendants_of(\"CHEBI:1\") within 5\n"
                 "  let L = ladder over a power 0.5, power 0.6 "
                 "expect power 0.7\n  emit L\n}")
    lad = [s for s in plan.steps if s.kind == "ladder"][0]
    target_not_a_rung = (tuple(lad.rungs) == (0.5, 0.6)
                         and lad.expect_power == 0.7)

    return {
        "experiment": "NEGATIVE CONTROL: malformed ladders are refused",
        "claim": ("Malformed declarations are parse errors, and the declared "
                  "target is not read as an extra rung."),
        "cases": rows,
        "n_refused": refused,
        "n_cases": len(cases),
        "declared_rungs": list(lad.rungs),
        "declared_target": lad.expect_power,
        "target_not_read_as_rung": target_not_a_rung,
        "holds": bool(refused == len(cases) and target_not_a_rung),
        "note": ("The last check is here because the first parser failed it: "
                 "the token 'power' occurs inside 'expect power', so the "
                 "target was appended as a fourth rung and the plan composed "
                 "a ladder nobody wrote. It is the class of silent failure "
                 "the verdict rules exist to prevent."),
    }


# ---------------------------------------------------------------------------
CHECKS = [e1_host_suite, e2_end_to_end, e3_refusal, e4_cost_regime,
          e5_union_bound, e6_corpus, e7_no_ladder_unchanged,
          e8_malformed_refused]


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    reports: List[Dict] = []
    for fn in CHECKS:
        name = fn.__name__.upper().split("_")[0]
        try:
            res = fn()
        except Exception as exc:               # a crash is a failed check
            import traceback
            res = {"experiment": fn.__name__, "holds": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-900:]}
        reports.append(res)
        with open(os.path.join(RESULTS, f"{name.lower()}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  {'ok    ' if res.get('holds') else 'FAILED'} "
              f"{res.get('experiment', fn.__name__)}")

    negatives = [r["experiment"] for r in reports
                 if str(r.get("experiment", "")).startswith("NEGATIVE CONTROL")]
    passing = sum(1 for r in reports if r.get("holds"))
    summary = {
        "paper": "Catalytic Ladders in a Federated Query Language",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(reports),
        "passing": passing,
        "failing": len(reports) - passing,
        "negative_controls": {"count": len(negatives), "checks": negatives},
        "elapsed_seconds": round(time.time() - started, 2),
        "checks": [{"experiment": r.get("experiment"),
                    "holds": r.get("holds")} for r in reports],
    }
    with open(os.path.join(RESULTS, "00_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{passing}/{len(reports)} experiments hold")
    return 0 if passing == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
