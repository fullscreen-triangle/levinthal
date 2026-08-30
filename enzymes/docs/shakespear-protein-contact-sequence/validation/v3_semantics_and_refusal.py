#!/usr/bin/env python3
"""
V3 --- The language: composition, sensitivity, freeness, clock monotonicity,
       conservativity, and the refusal.

The freeness checks are the operational content of the paper's criterion: a
rule may leave M unchanged exactly when it commits no cut.  The refusal checks
discharge the host's requirement that a construct which never refuses
classifies nothing, and they check BOTH directions -- a predicate that refused
everything would be as empty as one that refused nothing.

Negative controls are named with the prefix "NEGATIVE CONTROL" so a harvester
can find them by name, following the convention of the host suite.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

from shk_core import (Config, ContactGraph, Ladder, Machine, Rung, Verdict,
                      chain_graph, complete_graph, compose_additive,
                      compose_max, compose_mean, compose_multiplicative,
                      min_rungs_for, random_ladder, saturation_diagnostic,
                      static_reachable)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def v3_1_composition_law(n_ladders: int = 4000, seed: int = 31) -> Dict:
    """
    Four candidate laws against simulation.  Scoring only the favoured law
    would not distinguish it from the others, so all four are scored.
    """
    rng = random.Random(seed)
    laws = {"multiplicative": compose_multiplicative,
            "additive": compose_additive,
            "max": compose_max,
            "mean": compose_mean}
    err = {k: 0.0 for k in laws}
    for _ in range(n_ladders):
        n = rng.randint(2, 8)
        ps = [rng.uniform(0.05, 0.85) for _ in range(n)]
        gap = 1.0
        for p in ps:
            gap -= p * gap                     # simulate rung by rung
        truth = 1.0 - gap
        for k, f in laws.items():
            err[k] += abs(f(ps) - truth)
    mae = {k: v / n_ladders for k, v in err.items()}
    best = min(mae, key=mae.get)
    return {
        "test": "V3.1 composition is multiplicative, against three alternatives",
        "n_ladders": n_ladders,
        "MAE_by_law": mae,
        "best_law": best,
        "multiplicative_exact": mae["multiplicative"] < 1e-12,
        "alternatives_all_worse": all(mae[k] > 1e-6
                                      for k in laws if k != "multiplicative"),
        "passed": bool(best == "multiplicative"
                       and mae["multiplicative"] < 1e-12
                       and all(mae[k] > 1e-6 for k in laws
                               if k != "multiplicative")),
    }


def v3_2_sensitivity_at_strongest(n_ladders: int = 3000,
                                  seed: int = 32) -> Dict:
    """
    The counter-intuitive prediction: control lies at the HIGHEST-power rung.
    Checked two ways -- the closed form, and a finite-difference experiment
    that does not use it.
    """
    rng = random.Random(seed)
    form_ok = 0
    argmax_agrees = 0
    fd_agrees = 0
    for _ in range(n_ladders):
        n = rng.randint(2, 7)
        L = random_ladder(n, rng, 0.05, 0.9)
        s = L.sensitivity()
        P = L.residual_fraction()
        if all(abs(s[j] - P / (1 - L.powers[j])) < 1e-12 for j in range(n)):
            form_ok += 1
        if int(max(range(n), key=lambda j: s[j])) == \
           int(max(range(n), key=lambda j: L.powers[j])):
            argmax_agrees += 1
        # finite difference: which rung returns most for a fixed improvement?
        base = L.composite_power()
        gains = []
        for j in range(n):
            ps = L.powers[:]
            ps[j] = min(0.99, ps[j] + 0.01)
            gains.append(compose_multiplicative(ps) - base)
        if int(max(range(n), key=lambda j: gains[j])) == \
           int(max(range(n), key=lambda j: L.powers[j])):
            fd_agrees += 1
    return {
        "test": "V3.2 sensitivity is P/(1-pi_j), maximised at the strongest rung",
        "n_ladders": n_ladders,
        "closed_form_matches": form_ok,
        "argmax_sensitivity_is_argmax_power": argmax_agrees,
        "finite_difference_agrees": fd_agrees,
        "passed": bool(form_ok == n_ladders
                       and argmax_agrees == n_ladders
                       and fd_agrees == n_ladders),
        "interpretation": (
            "The finite-difference check does not use the closed form, so it "
            "is an independent test of the direction rather than a "
            "restatement of the derivative."
        ),
    }


# ---------------------------------------------------------------------------
# Freeness and the clock
# ---------------------------------------------------------------------------
def v3_3_free_rules_do_not_advance_M(n_ops: int = 5000,
                                     seed: int = 33) -> Dict:
    """
    E-Power, E-Derive and E-Observe-Power must leave M untouched; E-Climb must
    advance it by exactly one per rung.
    """
    rng = random.Random(seed)
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    cfg = Config(g, 0, 1.0, "probe")
    for _ in range(n_ops):
        m.probe(cfg)                       # stands for the free rules
    M_after_free = cfg.M

    L = Ladder([Rung(0.3), Rung(0.4), Rung(0.2)])
    m2 = Machine(g)
    c2 = m2.run(L, gap0=1.0)
    return {
        "test": "V3.3 free rules leave M untouched; climb advances once per rung",
        "n_free_operations": n_ops,
        "M_after_free_operations": M_after_free,
        "M_after_climb_of_3_rungs": c2.M,
        "residues_recorded": len(m2.residues),
        "every_residue_at_least_floor": all(r >= g.floor - 1e-15
                                            for r in m2.residues),
        "passed": bool(M_after_free == 0 and c2.M == 3
                       and len(m2.residues) == 3
                       and all(r >= g.floor - 1e-15 for r in m2.residues)),
    }


def v3_4_no_caching_of_climbs(n_repeats: int = 500, seed: int = 34) -> Dict:
    """Clock monotonicity: re-climbing is a new commitment, never a cache."""
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    cfg = Config(g, 0, 1.0, "commit")
    seq = []
    for _ in range(n_repeats):
        m.commit(cfg, Rung(0.4))
        seq.append(cfg.M)
    strict = all(seq[i + 1] == seq[i] + 1 for i in range(len(seq) - 1))
    return {
        "test": "V3.4 re-evaluating a commitment is never cached",
        "n_identical_commits": n_repeats,
        "M_final": cfg.M,
        "strictly_increments_by_one": strict,
        "passed": bool(cfg.M == n_repeats and strict),
    }


def v3_5_negative_control_probes_vs_commits(seed: int = 35) -> Dict:
    """
    NEGATIVE CONTROL for V3.3/V3.4.  If free operations also advanced M, those
    tests would be measuring "operations performed" rather than "cuts
    committed" and the freeness criterion would be untested.
    """
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    cfg = Config(g, 0, 1.0, "probe")
    for _ in range(1000):
        m.probe(cfg)
    after_free = cfg.M
    m.commit(cfg, Rung(0.5))
    after_one = cfg.M
    return {
        "test": "NEGATIVE CONTROL: free operations inert, commitment registers",
        "M_after_1000_free": after_free,
        "M_after_one_commit": after_one,
        "free_are_inert": after_free == 0,
        "commit_registers": after_one == 1,
        "passed": bool(after_free == 0 and after_one == 1),
    }


# ---------------------------------------------------------------------------
# Static analysis and the refusal
# ---------------------------------------------------------------------------
def v3_6_static_agrees_with_execution(n_programs: int = 5000,
                                      seed: int = 36) -> Dict:
    """
    Tightness, measured.  Decide statically, then execute, then compare.
    Programs are generated so that both verdicts occur in quantity: a suite in
    which everything was accepted would not have tested rejection.
    """
    rng = random.Random(seed)
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    agree = disagree = n_acc = n_rej = 0
    examples: List[Dict] = []
    for _ in range(n_programs):
        L = random_ladder(rng.randint(1, 8), rng, 0.02, 0.85)
        target = rng.uniform(0.2, 0.999)
        v_static = static_reachable(L.powers, target)
        cfg = Machine(g).run(L, gap0=1.0)
        v_exec = (1.0 - cfg.gap) >= target - 1e-12
        if v_static == v_exec:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append({"powers": L.powers, "target": target,
                                 "static": v_static, "executed": v_exec})
        n_acc += int(v_static)
        n_rej += int(not v_static)
    return {
        "test": "V3.6 the refusal is tight: static verdict agrees with execution",
        "n_programs": n_programs,
        "n_accepted": n_acc,
        "n_rejected": n_rej,
        "n_agree": agree,
        "n_disagree": disagree,
        "disagreement_examples": examples,
        "both_verdicts_occur": n_acc > 0 and n_rej > 0,
        "passed": bool(disagree == 0 and n_acc > 0 and n_rej > 0),
    }


def v3_7_refusal_non_vacuous() -> Dict:
    """
    NEGATIVE CONTROL: the refusal fires on presentable ladders AND declines to
    fire on others.  A predicate that refused everything would classify as
    little as one that refused nothing, so both directions are required.
    """
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    refused = m.run_verdict(Ladder([Rung(0.1), Rung(0.1)]), 0.9)
    empty = m.run_verdict(Ladder([]), 0.5)
    accepted = m.run_verdict(Ladder([Rung(0.5), Rung(0.6)]), 0.7)
    return {
        "test": "NEGATIVE CONTROL: refusal fires and also declines to fire",
        "refused_verdict": refused.label,
        "refused_shortfall": refused.payload.get("target", 0)
                             - refused.payload.get("best_possible", 0),
        "empty_verdict": empty.label,
        "accepted_verdict": accepted.label,
        "separates_both_ways": (refused.label == "subfloor"
                                and empty.label == "empty"
                                and accepted.label == "reached"),
        "passed": bool(refused.label == "subfloor"
                       and empty.label == "empty"
                       and accepted.label == "reached"),
    }


def v3_8_refusal_commits_no_cut() -> Dict:
    """
    NEGATIVE CONTROL: a refused ladder must commit nothing.  The clock must not
    advance for a program the language declined to run.
    """
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    before_edges = len(g.weights)
    v = m.run_verdict(Ladder([Rung(0.1), Rung(0.1)]), 0.95)
    return {
        "test": "NEGATIVE CONTROL: a refusal commits no cut",
        "verdict": v.label,
        "residues_recorded": len(m.residues),
        "edges_unchanged": len(g.weights) == before_edges,
        "passed": bool(v.label == "subfloor" and len(m.residues) == 0
                       and len(g.weights) == before_edges),
    }


def v3_9_verdict_payload_discipline() -> Dict:
    """
    NEGATIVE CONTROL: only gap-carrying verdicts may carry a gap.  A verdict
    type that let a refusal carry a result would reintroduce the ambiguity the
    verdict set exists to remove.
    """
    caught = 0
    trials = 0
    for label, payload in [("subfloor", {"gap": 1.0}),
                           ("empty", {"gap": 0.0}),
                           ("reached", {}),
                           ("short", {})]:
        trials += 1
        try:
            Verdict(label, payload)
        except ValueError:
            caught += 1
    ok_cases = 0
    for label, payload in [("reached", {"gap": 0.2}),
                           ("subfloor", {"reason": "x"})]:
        try:
            Verdict(label, payload)
            ok_cases += 1
        except ValueError:
            pass
    return {
        "test": "NEGATIVE CONTROL: verdict payload discipline is enforced",
        "malformed_rejected": caught,
        "malformed_attempted": trials,
        "wellformed_accepted": ok_cases,
        "passed": bool(caught == trials and ok_cases == 2),
    }


def v3_10_saturation_diagnostic(n_cases: int = 3000, seed: int = 37) -> Dict:
    """The bound-based refusal is sound and both outcomes occur."""
    rng = random.Random(seed)
    correct = flagged = 0
    for _ in range(n_cases):
        n = rng.randint(1, 10)
        p_max = rng.uniform(0.05, 0.6)
        target = rng.uniform(0.3, 0.995)
        diag = saturation_diagnostic(n, p_max, target)
        truth = (1.0 - (1.0 - p_max) ** n) < target
        correct += int(diag == truth)
        flagged += int(diag)
    return {
        "test": "V3.10 saturation diagnostic is sound",
        "n_cases": n_cases,
        "n_correct": correct,
        "n_flagged": flagged,
        "both_outcomes_occur": 0 < flagged < n_cases,
        "passed": bool(correct == n_cases and 0 < flagged < n_cases),
    }


def v3_11_min_rungs_exact() -> Dict:
    """Cost of a target: the bound is tight, not merely sufficient."""
    rows, ok = [], True
    for target in (0.5, 0.8, 0.9, 0.99):
        for p in (0.2, 0.35, 0.55):
            n = min_rungs_for(target, p)
            at_n = 1.0 - (1.0 - p) ** n
            at_n1 = (1.0 - (1.0 - p) ** (n - 1)) if n > 0 else 0.0
            tight = at_n >= target - 1e-12 and at_n1 < target
            ok = ok and tight
            rows.append({"target": target, "power": p, "n": n,
                         "composite_at_n": at_n,
                         "composite_at_n_minus_1": at_n1,
                         "tight": tight})
    return {"test": "V3.11 minimum rung count is exact", "rows": rows,
            "passed": ok}


def v3_12_conservativity(n_programs: int = 2000, seed: int = 38) -> Dict:
    """
    Conservativity, checked operationally: a ladder-free program reaches the
    same final state whether or not the ladder rules are present.  Here a
    ladder-free program is a sequence of plain commitments.
    """
    rng = random.Random(seed)
    mismatches = 0
    for _ in range(n_programs):
        g = complete_graph(4, [rng.uniform(0.2, 3.0) for _ in range(6)])
        ops = [rng.uniform(0.05, 0.9) for _ in range(rng.randint(1, 6))]
        m1 = Machine(g)
        c1 = Config(g, 0, 1.0, "commit")
        for p in ops:
            m1.commit(c1, Rung(p))
        # the same program expressed as a climb: must agree exactly
        m2 = Machine(g)
        c2 = m2.run(Ladder([Rung(p) for p in ops]), gap0=1.0)
        if c1.M != c2.M or abs(c1.gap - c2.gap) > 1e-12:
            mismatches += 1
    return {
        "test": "V3.12 climbing agrees with the equivalent commitment sequence",
        "n_programs": n_programs,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }


def v3_13_label_independence(n_ladders: int = 2000, seed: int = 39) -> Dict:
    """
    Label independence, with its control.  Relabelling is performed on
    annotations external to the formalism, because a rung has no identity
    field; the control perturbs a POWER and must move the observable.
    """
    rng = random.Random(seed)
    same = 0
    moved = 0
    for _ in range(n_ladders):
        L = random_ladder(rng.randint(2, 7), rng, 0.05, 0.9)
        relabelled = Ladder([Rung(p) for p in L.powers])   # names discarded
        if abs(L.composite_power() - relabelled.composite_power()) < 1e-15:
            same += 1
        j = rng.randrange(len(L.rungs))
        ps = L.powers[:]
        ps[j] = min(0.99, ps[j] + 0.05)
        if compose_multiplicative(ps) - L.composite_power() > 1e-9:
            moved += 1
    return {
        "test": "V3.13 labels move nothing; powers move everything (with control)",
        "n_ladders": n_ladders,
        "unchanged_under_relabelling": same,
        "control_moved_under_power_perturbation": moved,
        "passed": bool(same == n_ladders and moved == n_ladders),
        "interpretation": (
            "Without the control, the first count would be consistent with a "
            "constant observable and would carry no information."
        ),
    }


def v3_14_not_tested_note() -> Dict:
    """
    Recorded as NOT TESTED.  Whether the derived power corresponds to any
    physical rate is a question about chemistry, not about this formalism, and
    no check in this suite bears on it.
    """
    return {
        "test": "Correspondence of derived power to measured rates",
        "verdict": "NOT TESTED",
        "scored": False,
        "passed": None,
        "detail": (
            "The powers here are derived from a contact graph, which is "
            "itself a model. Whether they match rate constants measured in a "
            "laboratory is an empirical question requiring data this suite "
            "does not have. No check here should be read as supporting it."
        ),
    }


def main() -> Dict:
    os.makedirs(RESULTS, exist_ok=True)
    tests = [
        v3_1_composition_law(),
        v3_2_sensitivity_at_strongest(),
        v3_3_free_rules_do_not_advance_M(),
        v3_4_no_caching_of_climbs(),
        v3_5_negative_control_probes_vs_commits(),
        v3_6_static_agrees_with_execution(),
        v3_7_refusal_non_vacuous(),
        v3_8_refusal_commits_no_cut(),
        v3_9_verdict_payload_discipline(),
        v3_10_saturation_diagnostic(),
        v3_11_min_rungs_exact(),
        v3_12_conservativity(),
        v3_13_label_independence(),
        v3_14_not_tested_note(),
    ]
    scored = [t for t in tests if t.get("scored", True) is not False]
    n_pass = sum(1 for t in scored if t.get("passed"))
    negs = [t["test"] for t in tests
            if t["test"].startswith("NEGATIVE CONTROL")]
    res = {"script": "v3_semantics_and_refusal.py",
           "tests": tests,
           "negative_controls": negs,
           "summary": {"n": len(tests), "scored": len(scored),
                       "passed": n_pass,
                       "all_scored_passed": n_pass == len(scored)}}
    out = os.path.join(RESULTS, "v3_semantics_and_refusal.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[V3] {n_pass}/{len(scored)} scored passed -> {out}")
    for t in tests:
        if t.get("scored", True) is False:
            print(f"  ----      {t['test']}")
        else:
            print(f"  {'PASS' if t.get('passed') else 'FAIL'}  {t['test']}")
    return res


if __name__ == "__main__":
    main()
