#!/usr/bin/env python3
"""
V3 --- Composition (P1), diminishing returns (P2), saturation (P3).
V4 --- Sensitivity and the direction of control (P4).
V5 --- Inertness (P5).

V3 is a MODEL COMPARISON, not a goodness-of-fit.  The multiplicative law is
scored against additive, max and mean alternatives on the same simulated
ladders.  A test that only confirmed the favoured law would be uninformative.

V4 tests the counter-intuitive direction: sensitivity prod_{i!=j}(1-pi_i) is
maximised at the HIGHEST-power rung.  The naive expectation (weakest rung) is
included as the competing hypothesis and must lose.

V5 tests inertness by relabelling.  Since the Rung type has no identity field,
the "relabelling" is performed on an external annotation that the formalism
never reads -- which is the point.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

import numpy as np

from ladder_core import (Ladder, Rung, compose_additive, compose_max,
                         compose_mean, compose_multiplicative,
                         min_rungs_for, random_ladder)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def simulate_gap(powers, gap0: float = 1.0) -> float:
    """Independent simulation: apply each rung to the running gap."""
    g = gap0
    for p in powers:
        g = g - p * g          # rung closes fraction p of the REMAINING gap
    return g


# ---------------------------------------------------------------------------
def v3_1_composition_model_comparison(n_trials: int = 4000,
                                      seed: int = 31) -> Dict:
    """
    P1.  Simulate ladders; compare four candidate composition laws against the
    simulated composite.  The multiplicative law should reproduce it exactly.
    """
    rng = random.Random(seed)
    errs = {"multiplicative": [], "additive": [], "max": [], "mean": []}
    for _ in range(n_trials):
        n = rng.randint(2, 9)
        powers = [rng.uniform(0.02, 0.9) for _ in range(n)]
        truth = 1.0 - simulate_gap(powers)
        errs["multiplicative"].append(abs(compose_multiplicative(powers) - truth))
        errs["additive"].append(abs(compose_additive(powers) - truth))
        errs["max"].append(abs(compose_max(powers) - truth))
        errs["mean"].append(abs(compose_mean(powers) - truth))

    stats = {k: {"MAE": float(np.mean(v)), "max_err": float(np.max(v))}
             for k, v in errs.items()}
    best = min(stats, key=lambda k: stats[k]["MAE"])
    exact = stats["multiplicative"]["max_err"] < 1e-12
    return {
        "test": "V3.1 composition: model comparison (P1)",
        "n_trials": n_trials,
        "candidate_laws": stats,
        "best_model": best,
        "multiplicative_exact": exact,
        "passed": bool(best == "multiplicative" and exact),
        "interpretation": (
            "The multiplicative law reproduces the simulated composite to "
            "machine precision; the alternatives do not.  Scoring only the "
            "favoured law would not have distinguished them."
        ),
    }


def v3_2_diminishing_returns() -> Dict:
    """P2.  Marginal contribution of the n-th repetition is pi(1-pi)^(n-1)."""
    rows, ok = [], True
    for p in [0.1, 0.3, 0.5, 0.7]:
        marg, comps = [], []
        prev = 0.0
        for n in range(1, 26):
            c = 1.0 - (1.0 - p) ** n
            comps.append(c)
            marg.append(c - prev)
            prev = c
        predicted = [p * (1.0 - p) ** (k) for k in range(25)]
        err = max(abs(a - b) for a, b in zip(marg, predicted))
        strictly_dec = all(marg[i + 1] < marg[i] for i in range(len(marg) - 1))
        below_one = all(c < 1.0 for c in comps)
        good = err < 1e-12 and strictly_dec and below_one
        ok = ok and good
        rows.append({"power": p, "max_marginal_error": err,
                     "marginal_strictly_decreasing": strictly_dec,
                     "never_reaches_one": below_one,
                     "composite_at_n25": comps[-1]})
    return {
        "test": "V3.2 diminishing returns (P2)",
        "rows": rows,
        "passed": ok,
    }


def v3_3_saturation_dichotomy(n_rungs: int = 400) -> Dict:
    """
    P3.  Divergent sum -> gap to zero; convergent sum -> plateau.
    Indexed from i=2 so that no rung has power 1 and the dichotomy is
    exhibited by TAIL behaviour, per the paper's Remark.
    """
    series = {
        "harmonic_divergent_1_over_i": [1.0 / i for i in range(2, n_rungs + 2)],
        "convergent_1_over_i_squared": [1.0 / i ** 2
                                        for i in range(2, n_rungs + 2)],
        "convergent_geometric_2^-i": [2.0 ** (-i)
                                      for i in range(2, n_rungs + 2)],
    }
    rows, ok = [], True
    for name, ps in series.items():
        assert all(0.0 <= p < 1.0 for p in ps)
        gap = 1.0
        for p in ps:
            gap *= (1.0 - p)
        s = sum(ps)
        diverges = name.startswith("harmonic")
        drove_to_zero = gap < 1e-2
        consistent = (drove_to_zero == diverges)
        ok = ok and consistent
        rows.append({"series": name, "sum_of_powers": s,
                     "residual_gap_after_n": gap,
                     "sum_diverges": diverges,
                     "gap_driven_to_zero": drove_to_zero,
                     "dichotomy_holds": consistent})
    return {
        "test": "V3.3 saturation dichotomy (P3)",
        "n_rungs": n_rungs,
        "indexed_from": 2,
        "rows": rows,
        "passed": ok,
    }


# ---------------------------------------------------------------------------
def v4_1_sensitivity_direction(n_trials: int = 5000, seed: int = 41) -> Dict:
    """
    P4.  Sensitivity is prod_{i!=j}(1-pi_i) = P/(1-pi_j), maximised at the
    HIGHEST power.  The naive hypothesis (weakest rung) is the competitor.
    """
    rng = random.Random(seed)
    analytic_ok = 0
    argmax_is_highest = 0
    argmax_is_lowest = 0
    for _ in range(n_trials):
        n = rng.randint(2, 8)
        lad = random_ladder(n, rng, 0.02, 0.9)
        sens = lad.sensitivity()
        # numerical derivative as an independent check
        h = 1e-7
        num = []
        for j in range(n):
            ps = lad.powers[:]
            ps[j] += h
            num.append((compose_multiplicative(ps)
                        - lad.composite_power()) / h)
        if max(abs(a - b) for a, b in zip(sens, num)) < 1e-5:
            analytic_ok += 1
        j_star = int(np.argmax(sens))
        if j_star == int(np.argmax(lad.powers)):
            argmax_is_highest += 1
        if j_star == int(np.argmin(lad.powers)):
            argmax_is_lowest += 1
    return {
        "test": "V4.1 sensitivity direction (P4)",
        "n_trials": n_trials,
        "analytic_matches_numerical": analytic_ok / n_trials,
        "argmax_sensitivity_is_highest_power": argmax_is_highest / n_trials,
        "argmax_sensitivity_is_lowest_power": argmax_is_lowest / n_trials,
        "naive_hypothesis_rejected": (argmax_is_lowest / n_trials) < 0.05,
        "passed": bool(analytic_ok == n_trials
                       and argmax_is_highest == n_trials),
        "interpretation": (
            "Sensitivity is maximised at the highest-power rung in every "
            "trial, and at the lowest-power rung in none.  The intuitive "
            "expectation that the bottleneck carries the control is wrong "
            "for this quantity."
        ),
    }


def v4_2_marginal_value(seed: int = 42) -> Dict:
    """Improving rung j by delta buys delta * prod_{i!=j}(1-pi_i)."""
    powers = [0.45, 0.30, 0.55, 0.20]
    lad = Ladder([Rung(p) for p in powers])
    base = lad.composite_power()
    sens = lad.sensitivity()
    rows = []
    ok = True
    for j in range(len(powers)):
        for delta in [0.01, 0.05, 0.10]:
            ps = powers[:]
            ps[j] = min(1.0, ps[j] + delta)
            actual = compose_multiplicative(ps) - base
            predicted = delta * sens[j]
            err = abs(actual - predicted)
            good = err < 1e-12
            ok = ok and good
            if delta == 0.10:
                rows.append({"rung": j + 1, "power": powers[j],
                             "sensitivity": sens[j],
                             "delta": delta,
                             "predicted_gain": predicted,
                             "actual_gain": actual,
                             "exact": good})
    return {
        "test": "V4.2 marginal value of improvement",
        "powers": powers,
        "composite": base,
        "sensitivities": sens,
        "rows_at_delta_0.10": rows,
        "linear_prediction_exact": ok,
        "passed": ok,
        "note": ("The gain is exactly linear in delta because composite power "
                 "is affine in each pi_j individually."),
    }


# ---------------------------------------------------------------------------
def v5_1_inertness_under_relabelling(n_trials: int = 3000,
                                     seed: int = 51) -> Dict:
    """
    P5.  Attach arbitrary external labels to rungs, permute them, and check
    that every admissible observable is unchanged.

    The Rung type has no identity field, so the labels live outside the
    formalism entirely -- which is exactly the claim being tested.
    """
    rng = random.Random(seed)
    identical = 0
    for _ in range(n_trials):
        n = rng.randint(2, 8)
        lad = random_ladder(n, rng)
        labels = [f"L{rng.randint(0, 10**6)}" for _ in range(n)]
        rng.shuffle(labels)             # relabel: formalism never reads these

        a0, floor_norm = 0.9, 1e-3
        before = {
            "composite": lad.composite_power(),
            "residual": lad.residual_fraction(),
            "gaps": lad.gap_trajectory(),
            "alignments": lad.alignment_trajectory(a0, floor_norm),
            "sensitivity": lad.sensitivity(),
            "commitments": len(lad.rungs),
        }
        # rebuild the ladder from powers alone; labels discarded
        lad2 = Ladder([Rung(p) for p in lad.powers])
        after = {
            "composite": lad2.composite_power(),
            "residual": lad2.residual_fraction(),
            "gaps": lad2.gap_trajectory(),
            "alignments": lad2.alignment_trajectory(a0, floor_norm),
            "sensitivity": lad2.sensitivity(),
            "commitments": len(lad2.rungs),
        }
        if before == after:
            identical += 1
    return {
        "test": "V5.1 inertness under relabelling (P5)",
        "n_trials": n_trials,
        "n_identical_on_all_observables": identical,
        "fraction": identical / n_trials,
        "passed": identical == n_trials,
    }


def v5_2_control_powers_do_separate(n_trials: int = 3000,
                                    seed: int = 52) -> Dict:
    """
    CONTROL for V5.1.  If NO perturbation changed the observables, V5.1 would
    be vacuous.  Perturbing a POWER must change them.  This establishes that
    the observable set is not blind.
    """
    rng = random.Random(seed)
    changed = 0
    for _ in range(n_trials):
        n = rng.randint(2, 8)
        lad = random_ladder(n, rng, 0.05, 0.9)
        j = rng.randrange(n)
        ps = lad.powers[:]
        ps[j] = min(0.99, ps[j] + 0.05)
        lad2 = Ladder([Rung(p) for p in ps])
        if abs(lad2.composite_power() - lad.composite_power()) > 1e-12:
            changed += 1
    return {
        "test": "V5.2 CONTROL: perturbing a power does change observables",
        "n_trials": n_trials,
        "n_changed": changed,
        "fraction": changed / n_trials,
        "passed": changed == n_trials,
        "interpretation": (
            "V5.1 shows labels do not move the observables.  This control "
            "shows powers do.  Without it, V5.1 would be consistent with a "
            "constant observable set and would carry no information."
        ),
    }


def v5_3_converse_fails() -> Dict:
    """
    The paper's Proposition: equal composite power does NOT imply equal power
    sequence.  Exhibit the counterexample and confirm intermediates separate.
    """
    A = Ladder([Rung(0.5), Rung(0.5)])
    B = Ladder([Rung(0.75), Rung(0.0)])
    same_composite = abs(A.composite_power() - B.composite_power()) < 1e-12
    gaps_differ = A.gap_trajectory() != B.gap_trajectory()
    return {
        "test": "V5.3 converse fails: composite alone underdetermines",
        "ladder_A_powers": A.powers,
        "ladder_B_powers": B.powers,
        "composite_A": A.composite_power(),
        "composite_B": B.composite_power(),
        "same_composite": same_composite,
        "gap_trajectory_A": A.gap_trajectory(),
        "gap_trajectory_B": B.gap_trajectory(),
        "intermediates_separate_them": gaps_differ,
        "passed": bool(same_composite and gaps_differ),
        "interpretation": (
            "An end-to-end assay cannot distinguish these ladders; an assay "
            "resolving intermediates can.  The framework states the limit of "
            "what composite power determines rather than concealing it."
        ),
    }


# ---------------------------------------------------------------------------
def main() -> Dict:
    os.makedirs(RESULTS, exist_ok=True)
    tests = [
        v3_1_composition_model_comparison(),
        v3_2_diminishing_returns(),
        v3_3_saturation_dichotomy(),
        v4_1_sensitivity_direction(),
        v4_2_marginal_value(),
        v5_1_inertness_under_relabelling(),
        v5_2_control_powers_do_separate(),
        v5_3_converse_fails(),
    ]
    n_pass = sum(1 for t in tests if t["passed"])
    res = {"script": "v3_composition_and_inertness.py",
           "predictions": ["P1", "P2", "P3", "P4", "P5"],
           "tests": tests,
           "summary": {"n": len(tests), "passed": n_pass,
                       "all_passed": n_pass == len(tests)}}
    out = os.path.join(RESULTS, "v3_composition_and_inertness.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[V3/V4/V5] {n_pass}/{len(tests)} passed -> {out}")
    for t in tests:
        print(f"  {'PASS' if t['passed'] else 'FAIL'}  {t['test']}")
    return res


if __name__ == "__main__":
    main()
