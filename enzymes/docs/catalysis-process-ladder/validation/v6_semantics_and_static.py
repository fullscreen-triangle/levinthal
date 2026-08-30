#!/usr/bin/env python3
"""
V6 --- Operational semantics: progress, preservation, monotone commitment (P6).
V7 --- Static analysis: reachability decided before execution (P7).
V8 --- The worked demonstration, recomputed from the specification.

V6 executes generated programs exhaustively and checks the three theorems as
invariants of every step, not as assertions about a chosen example.

V7 is the sharpest test in the suite: the static verdict is computed WITHOUT
executing, the program is then executed, and the two are compared.  Programs
designed to be rejected are included, so the test can fail in both directions.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

import numpy as np

from ladder_core import (Config, ContactGraph, Ladder, Machine, Rung,
                         complete_graph, compose_multiplicative,
                         min_rungs_for, random_ladder, saturation_diagnostic,
                         static_reachable)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# ---------------------------------------------------------------------------
def v6_1_progress(n_programs: int = 2000, seed: int = 61) -> Dict:
    """
    Progress: a well-typed non-value expression always admits a reduction.
    Operationally: no generated program gets stuck.
    """
    rng = random.Random(seed)
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    stuck = 0
    terminated = 0
    for _ in range(n_programs):
        lad = random_ladder(rng.randint(1, 12), rng)
        m = Machine(g)
        cfg = m.run(lad, gap0=1.0)
        if cfg.expr in ("Reached", "Short"):
            terminated += 1
        else:
            stuck += 1
    return {
        "test": "V6.1 progress: no program gets stuck (P6)",
        "n_programs": n_programs,
        "n_terminated": terminated,
        "n_stuck": stuck,
        "passed": stuck == 0,
    }


def v6_2_preservation(n_programs: int = 2000, seed: int = 62) -> Dict:
    """
    Preservation: the floor is unchanged by reduction and M never decreases
    at any step.  Checked step-by-step, not just at the end.
    """
    rng = random.Random(seed)
    violations_floor = 0
    violations_M = 0
    for _ in range(n_programs):
        g = complete_graph(4, [rng.uniform(0.2, 3.0) for _ in range(6)])
        floor_before = g.floor
        m = Machine(g)
        cfg = Config(g, 0, 1.0, "climb")
        lad = random_ladder(rng.randint(1, 10), rng)
        M_prev = cfg.M
        for rung in lad.rungs:
            m.commit(cfg, rung)
            if cfg.M < M_prev:
                violations_M += 1
            M_prev = cfg.M
        if abs(g.floor - floor_before) > 0:
            violations_floor += 1
    return {
        "test": "V6.2 preservation: floor invariant, M non-decreasing (P6)",
        "n_programs": n_programs,
        "floor_violations": violations_floor,
        "M_decrease_violations": violations_M,
        "passed": violations_floor == 0 and violations_M == 0,
    }


def v6_3_monotone_commitment(n_repeats: int = 500, seed: int = 63) -> Dict:
    """
    Monotone commitment: re-evaluating an identical expression is a NEW
    commitment at strictly higher M, never a cached retrieval.
    """
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    cfg = Config(g, 0, 1.0, "commit")
    same_rung = Rung(0.4)
    seq = []
    for _ in range(n_repeats):
        m.commit(cfg, same_rung)
        seq.append(cfg.M)
    strictly_increasing = all(seq[i + 1] == seq[i] + 1
                              for i in range(len(seq) - 1))
    return {
        "test": "V6.3 monotone commitment: no caching of commits (P6)",
        "n_identical_commits": n_repeats,
        "M_final": cfg.M,
        "M_equals_commit_count": cfg.M == n_repeats,
        "strictly_increments_by_one": strictly_increasing,
        "residues_recorded": len(m.residues),
        "passed": bool(cfg.M == n_repeats and strictly_increasing
                       and len(m.residues) == n_repeats),
    }


def v6_4_probes_do_not_increment(n_probes: int = 5000, seed: int = 64) -> Dict:
    """
    CONTROL for V6.3.  Commits increment M; probes must not.  If probes also
    incremented, V6.3 would be measuring "operations performed" rather than
    "commitments performed" and the asymmetry would be untested here.
    """
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    m = Machine(g)
    cfg = Config(g, 0, 1.0, "probe")
    for _ in range(n_probes):
        m.probe(cfg)
    M_after_probes = cfg.M
    m.commit(cfg, Rung(0.5))
    M_after_one_commit = cfg.M
    return {
        "test": "V6.4 CONTROL: probes leave M untouched, commits move it",
        "n_probes": n_probes,
        "M_after_probes": M_after_probes,
        "M_after_one_commit": M_after_one_commit,
        "probes_inert": M_after_probes == 0,
        "commit_registers": M_after_one_commit == 1,
        "passed": bool(M_after_probes == 0 and M_after_one_commit == 1),
    }


# ---------------------------------------------------------------------------
def v7_1_static_agrees_with_execution(n_programs: int = 5000,
                                      seed: int = 71) -> Dict:
    """
    P7.  Decide reachability statically, then execute, then compare.
    Programs are generated so that roughly half are expected to FAIL, and the
    counts are reported: a test in which nothing is ever rejected would be
    uninformative.
    """
    rng = random.Random(seed)
    g = complete_graph(4, [1.0, 2.0, 0.5, 3.0])
    agree = 0
    disagree = 0
    n_accept = 0
    n_reject = 0
    examples: List[Dict] = []
    for _ in range(n_programs):
        n = rng.randint(1, 8)
        lad = random_ladder(n, rng, 0.02, 0.85)
        target = rng.uniform(0.2, 0.999)

        verdict_static = static_reachable(lad.powers, target)

        m = Machine(g)
        cfg = m.run(lad, gap0=1.0)
        achieved = 1.0 - cfg.gap
        verdict_exec = achieved >= target - 1e-12

        if verdict_static == verdict_exec:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append({"powers": lad.powers, "target": target,
                                 "static": verdict_static,
                                 "executed": verdict_exec,
                                 "achieved": achieved})
        n_accept += int(verdict_static)
        n_reject += int(not verdict_static)

    return {
        "test": "V7.1 static verdict agrees with execution (P7)",
        "n_programs": n_programs,
        "n_accepted_statically": n_accept,
        "n_rejected_statically": n_reject,
        "n_agree": agree,
        "n_disagree": disagree,
        "disagreement_examples": examples,
        "both_verdicts_occur": n_accept > 0 and n_reject > 0,
        "passed": bool(disagree == 0 and n_accept > 0 and n_reject > 0),
        "interpretation": (
            "Both verdicts occur in quantity, so the test could have failed "
            "in either direction.  A suite in which every program was "
            "accepted would not have tested rejection."
        ),
    }


def v7_2_saturation_diagnostic(n_cases: int = 3000, seed: int = 72) -> Dict:
    """
    The compiler refuses a ladder of near-duplicate rungs that cannot reach
    its declared target.  Check the diagnostic against the true composite.
    """
    rng = random.Random(seed)
    correct = 0
    flagged = 0
    for _ in range(n_cases):
        n = rng.randint(1, 10)
        p_max = rng.uniform(0.05, 0.6)
        target = rng.uniform(0.3, 0.995)
        diag = saturation_diagnostic(n, p_max, target)
        best_possible = 1.0 - (1.0 - p_max) ** n
        truth = best_possible < target
        if diag == truth:
            correct += 1
        flagged += int(diag)
    return {
        "test": "V7.2 saturation diagnostic is sound",
        "n_cases": n_cases,
        "n_correct": correct,
        "n_flagged_unreachable": flagged,
        "both_outcomes_occur": 0 < flagged < n_cases,
        "passed": bool(correct == n_cases and 0 < flagged < n_cases),
    }


def v7_3_min_rung_count(seed: int = 73) -> Dict:
    """Cost of a target: ceil(log(1-pi*)/log(1-pi)) rungs."""
    rows, ok = [], True
    for target in [0.5, 0.8, 0.9, 0.99]:
        for p in [0.2, 0.35, 0.55]:
            n = min_rungs_for(target, p)
            reached = 1.0 - (1.0 - p) ** n
            reached_one_less = (1.0 - (1.0 - p) ** (n - 1)) if n > 0 else 0.0
            tight = reached >= target - 1e-12 and reached_one_less < target
            ok = ok and tight
            rows.append({"target": target, "rung_power": p, "n_required": n,
                         "composite_at_n": reached,
                         "composite_at_n_minus_1": reached_one_less,
                         "bound_is_tight": tight})
    return {
        "test": "V7.3 minimum rung count is exact",
        "rows": rows,
        "passed": ok,
    }


# ---------------------------------------------------------------------------
def v8_1_worked_demonstration() -> Dict:
    """
    Recompute every number reported in the paper's demonstration section,
    from the specification alone.  Nothing here is entered by hand except the
    four declared powers and the declared target.
    """
    powers = [0.45, 0.30, 0.55, 0.20]
    target_power = 0.80
    lad = Ladder([Rung(p) for p in powers])

    composite = lad.composite_power()
    residual = lad.residual_fraction()
    sens = lad.sensitivity()

    gains = {}
    for j, d in [(3, 0.10), (2, 0.10)]:      # rung 4 and rung 3, 0-indexed
        ps = powers[:]
        ps[j] = ps[j] + d
        gains[f"rung{j+1}_plus_{d}"] = compose_multiplicative(ps) - composite

    deletions = {}
    for j in range(4):
        ps = [powers[i] for i in range(4) if i != j]
        c = compose_multiplicative(ps)
        deletions[f"drop_rung{j+1}"] = {
            "composite": c, "still_compiles": c >= target_power}

    n_min = min_rungs_for(target_power, max(powers))

    two_rung = compose_multiplicative([0.60, 0.60])

    return {
        "test": "V8.1 worked demonstration recomputed",
        "declared_powers": powers,
        "declared_target": target_power,
        "composite_power": composite,
        "residual_fraction": residual,
        "compiles": composite >= target_power,
        "sensitivities": sens,
        "control_at_rung": int(np.argmax(sens)) + 1,
        "control_rung_is_highest_power": (int(np.argmax(sens))
                                          == int(np.argmax(powers))),
        "marginal_gains": gains,
        "deletion_analysis": deletions,
        "min_rungs_for_target": n_min,
        "rungs_used": len(powers),
        "two_rung_alternative_composite": two_rung,
        "passed": bool(abs(composite - 0.8614) < 5e-5
                       and int(np.argmax(sens)) == 2
                       and deletions["drop_rung2"]["still_compiles"]
                       and not deletions["drop_rung3"]["still_compiles"]
                       and n_min == 3),
    }


# ---------------------------------------------------------------------------
def main() -> Dict:
    os.makedirs(RESULTS, exist_ok=True)
    tests = [
        v6_1_progress(),
        v6_2_preservation(),
        v6_3_monotone_commitment(),
        v6_4_probes_do_not_increment(),
        v7_1_static_agrees_with_execution(),
        v7_2_saturation_diagnostic(),
        v7_3_min_rung_count(),
        v8_1_worked_demonstration(),
    ]
    n_pass = sum(1 for t in tests if t["passed"])
    res = {"script": "v6_semantics_and_static.py",
           "predictions": ["P6", "P7"],
           "tests": tests,
           "summary": {"n": len(tests), "passed": n_pass,
                       "all_passed": n_pass == len(tests)}}
    out = os.path.join(RESULTS, "v6_semantics_and_static.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[V6/V7/V8] {n_pass}/{len(tests)} passed -> {out}")
    for t in tests:
        print(f"  {'PASS' if t['passed'] else 'FAIL'}  {t['test']}")
    return res


if __name__ == "__main__":
    main()
