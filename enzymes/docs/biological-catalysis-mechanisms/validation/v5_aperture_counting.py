#!/usr/bin/env python3
"""
V5 --- Aperture counting and the efficiency law
       (Predictions P3 and P4; Theorem 'Rate law' and its Corollary).

CLAIM UNDER TEST
    log10(kcat/KM) = log10(nu_floor) - dC
    with ONE framework-wide nu_floor, dC counted from mechanism by the
    simultaneity rule, and NO per-enzyme adjustable parameter.

WHAT WOULD FALSIFY IT
    P3: requiring a per-enzyme nu_floor -- i.e. the law only works as a fit.
    P4: rate enhancement bounded by a ratio of step counts rather than
        exponential in dC (the reciprocal alternative).

METHODOLOGICAL COMMITMENT
    dC is assigned from documented elementary-step counts by a stated rule
    applied uniformly, WITHOUT looking at the measured rate.  The assignment
    is frozen in APERTURE_TABLE below with the rule recorded per entry.
    We then evaluate the law with a single global constant and report the
    error.  We also fit a free per-enzyme constant and show how much better
    that does -- if the free fit is much better, the parameter-free claim
    fails and we say so.
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------
# Aperture assignment.
#
# RULE (stated once, applied uniformly):
#   dC = number of elementary categorical transitions in the accepted
#        mechanism, where coordinate changes that occur TOGETHER as a single
#        concerted event count as ONE aperture (simultaneity rule), and
#        sequential steps count separately.
#
# Each entry records the mechanism decomposition used, so the count can be
# audited independently of the rate.
# ---------------------------------------------------------------------------
APERTURE_TABLE: List[Dict] = [
    {"enzyme": "Superoxide dismutase", "ec": "1.15.1.1", "dC": 1,
     "mechanism": "single electron transfer Cu(II)->Cu(I) (one aperture)",
     "kcat_over_KM": 2.8e9},
    {"enzyme": "Catalase", "ec": "1.11.1.6", "dC": 1,
     "mechanism": "two-electron transfer to Cpd I, concerted",
     "kcat_over_KM": 3.6e7},
    {"enzyme": "Carbonic anhydrase II", "ec": "4.2.1.1", "dC": 1,
     "mechanism": "Zn-OH nucleophilic attack, concerted",
     "kcat_over_KM": 8.3e7},
    {"enzyme": "Acetylcholinesterase", "ec": "3.1.1.7", "dC": 2,
     "mechanism": "acylation + deacylation",
     "kcat_over_KM": 1.6e8},
    {"enzyme": "Triosephosphate isomerase", "ec": "5.3.1.1", "dC": 2,
     "mechanism": "proton abstraction + reprotonation via enediolate",
     "kcat_over_KM": 9.1e6},
    {"enzyme": "Fumarase", "ec": "4.2.1.2", "dC": 2,
     "mechanism": "water addition + proton transfer",
     "kcat_over_KM": 1.6e8},
    {"enzyme": "Crotonase", "ec": "4.2.1.17", "dC": 2,
     "mechanism": "water addition + enolate protonation",
     "kcat_over_KM": 2.8e8},
    {"enzyme": "Beta-lactamase", "ec": "3.5.2.6", "dC": 2,
     "mechanism": "acyl-enzyme formation + hydrolysis",
     "kcat_over_KM": 1.0e8},
    {"enzyme": "Adenylate kinase", "ec": "2.7.4.3", "dC": 3,
     "mechanism": "domain closure + phosphoryl transfer + opening",
     "kcat_over_KM": 3.0e6},
    {"enzyme": "Lactate dehydrogenase", "ec": "1.1.1.27", "dC": 3,
     "mechanism": "loop closure + hydride transfer + proton transfer",
     "kcat_over_KM": 3.1e6},
    {"enzyme": "Alcohol dehydrogenase", "ec": "1.1.1.1", "dC": 3,
     "mechanism": "Zn coordination + hydride transfer + proton relay",
     "kcat_over_KM": 6.7e3},
    {"enzyme": "Hexokinase", "ec": "2.7.1.1", "dC": 3,
     "mechanism": "induced fit closure + phosphoryl transfer + release",
     "kcat_over_KM": 6.7e6},
    {"enzyme": "Chymotrypsin", "ec": "3.4.21.1", "dC": 4,
     "mechanism": "tetrahedral int. 1 + acyl-enzyme + tetrahedral int. 2 + release",
     "kcat_over_KM": 1.5e5},
    {"enzyme": "Trypsin", "ec": "3.4.21.4", "dC": 4,
     "mechanism": "same serine-protease four-step mechanism",
     "kcat_over_KM": 2.5e5},
    {"enzyme": "Thrombin", "ec": "3.4.21.5", "dC": 4,
     "mechanism": "serine-protease four-step mechanism",
     "kcat_over_KM": 6.0e6},
    {"enzyme": "Aldolase", "ec": "4.1.2.13", "dC": 4,
     "mechanism": "Schiff base + C-C cleavage + enamine + hydrolysis",
     "kcat_over_KM": 1.7e6},
    {"enzyme": "Aspartate aminotransferase", "ec": "2.6.1.1", "dC": 4,
     "mechanism": "transaldimination + quinonoid + reprotonation + hydrolysis",
     "kcat_over_KM": 1.7e5},
    {"enzyme": "Lysozyme", "ec": "3.2.1.17", "dC": 4,
     "mechanism": "substrate distortion + oxocarbenium + covalent int. + hydrolysis",
     "kcat_over_KM": 8.3e4},
    {"enzyme": "Rubisco", "ec": "4.1.1.39", "dC": 5,
     "mechanism": "enolisation + carboxylation + hydration + C-C cleavage + protonation",
     "kcat_over_KM": 3.0e5},
    {"enzyme": "Cytochrome P450 3A4", "ec": "1.14.14.1", "dC": 6,
     "mechanism": "substrate binding + e- + O2 + e- + Cpd I + HAT/rebound",
     "kcat_over_KM": 1.0e6},
]


def v5_1_parameter_free_law(nu_floor_log10: float = 10.0,
                            delta_decades: float = 1.0) -> Dict:
    """
    Evaluate log10(kcat/KM) = nu_floor_log10 - delta*dC with the SINGLE global
    constant.  No per-enzyme freedom.
    """
    rows = []
    errs = []
    for r in APERTURE_TABLE:
        obs = math.log10(r["kcat_over_KM"])
        pred = nu_floor_log10 - delta_decades * r["dC"]
        err = pred - obs
        errs.append(err)
        rows.append({**{k: r[k] for k in ("enzyme", "ec", "dC", "mechanism")},
                     "log10_obs": round(obs, 3),
                     "log10_pred": round(pred, 3),
                     "error_decades": round(err, 3)})

    errs_arr = np.array(errs)
    mae = float(np.abs(errs_arr).mean())
    rmse = float(np.sqrt((errs_arr ** 2).mean()))
    within_1 = int((np.abs(errs_arr) <= 1.0).sum())
    within_2 = int((np.abs(errs_arr) <= 2.0).sum())

    obs = np.array([math.log10(r["kcat_over_KM"]) for r in APERTURE_TABLE])
    dc = np.array([r["dC"] for r in APERTURE_TABLE], dtype=float)
    corr = float(np.corrcoef(dc, obs)[0, 1])

    return {
        "test": "V5.1 parameter-free efficiency law",
        "law": "log10(kcat/KM) = log10(nu_floor) - delta * dC",
        "nu_floor_log10_GLOBAL": nu_floor_log10,
        "delta_decades_GLOBAL": delta_decades,
        "n_free_parameters": 0,
        "n_enzymes": len(rows),
        "rows": rows,
        "MAE_decades": mae,
        "RMSE_decades": rmse,
        "n_within_1_decade": within_1,
        "n_within_2_decades": within_2,
        "corr_dC_vs_log10_obs": corr,
        "passed": bool(corr < -0.5 and mae < 2.0),
        "interpretation": (
            "The scored claim is the NEGATIVE correlation between dC and "
            "efficiency with a single global constant. A strong negative "
            "correlation with modest MAE supports the law; MAE alone would "
            "not, since a constant predictor can have small MAE on a narrow "
            "range."
        ),
    }


def v5_2_free_fit_comparison() -> Dict:
    """
    HONESTY TEST.  Fit slope AND intercept freely.  If the free fit is
    dramatically better than the parameter-free law, the parameter-free claim
    is weak and must be reported as such.
    """
    obs = np.array([math.log10(r["kcat_over_KM"]) for r in APERTURE_TABLE])
    dc = np.array([r["dC"] for r in APERTURE_TABLE], dtype=float)

    slope, intercept = np.polyfit(dc, obs, 1)
    pred_free = slope * dc + intercept
    mae_free = float(np.abs(pred_free - obs).mean())

    pred_fixed = 10.0 - 1.0 * dc
    mae_fixed = float(np.abs(pred_fixed - obs).mean())

    improvement = mae_fixed - mae_free
    ratio = mae_fixed / mae_free if mae_free > 0 else float("inf")

    return {
        "test": "V5.2 free fit vs parameter-free",
        "free_fit_slope": float(slope),
        "free_fit_intercept": float(intercept),
        "framework_slope": -1.0,
        "framework_intercept": 10.0,
        "MAE_free_fit": mae_free,
        "MAE_parameter_free": mae_fixed,
        "improvement_decades": improvement,
        "MAE_ratio_fixed_over_free": ratio,
        "slope_within_50pct_of_framework": bool(abs(slope + 1.0) < 0.5),
        "passed": bool(abs(slope + 1.0) < 0.5 and ratio < 2.5),
        "interpretation": (
            "If the freely fitted slope lands near -1 and the free fit is not "
            "dramatically better, the parameter-free law is carrying real "
            "structure rather than being rescued by tuning."
        ),
    }


def v5_3_exponential_vs_reciprocal() -> Dict:
    """
    P4.  Compare the exponential law against the reciprocal alternative
    k ~ 1/(dC * tau).  The reciprocal law bounds the dynamic range by the
    ratio of dC values; the data span far more than that.
    """
    obs = np.array([math.log10(r["kcat_over_KM"]) for r in APERTURE_TABLE])
    dc = np.array([r["dC"] for r in APERTURE_TABLE], dtype=float)

    observed_range_decades = float(obs.max() - obs.min())
    dc_ratio = float(dc.max() / dc.min())
    reciprocal_max_range_decades = math.log10(dc_ratio)

    # best-case reciprocal fit
    recip_pred = -np.log10(dc)
    recip_pred = recip_pred - recip_pred.mean() + obs.mean()
    mae_recip = float(np.abs(recip_pred - obs).mean())

    exp_pred = 10.0 - dc
    mae_exp = float(np.abs(exp_pred - obs).mean())

    return {
        "test": "V5.3 exponential vs reciprocal (P4)",
        "observed_dynamic_range_decades": observed_range_decades,
        "dC_ratio_max_over_min": dc_ratio,
        "max_range_reciprocal_law_can_produce_decades":
            reciprocal_max_range_decades,
        "reciprocal_law_can_span_observed_range":
            bool(reciprocal_max_range_decades >= observed_range_decades),
        "MAE_reciprocal_best_case": mae_recip,
        "MAE_exponential": mae_exp,
        "exponential_better": bool(mae_exp < mae_recip),
        "passed": bool(reciprocal_max_range_decades < observed_range_decades),
        "interpretation": (
            "The reciprocal law is bounded: with dC from 1 to 6 it can span "
            "at most log10(6) ~ 0.78 decades. The data span far more. The "
            "reciprocal law is therefore excluded by the dynamic range alone, "
            "independently of any fit quality."
        ),
    }


def v5_4_shuffle_control(n_perm: int = 20000, seed: int = 54) -> Dict:
    """
    NEGATIVE CONTROL.  Shuffle dC across enzymes.  If the dC-efficiency
    correlation survives shuffling, dC carries no information and V5.1 is
    non-discriminating.
    """
    rng = np.random.default_rng(seed)
    obs = np.array([math.log10(r["kcat_over_KM"]) for r in APERTURE_TABLE])
    dc = np.array([r["dC"] for r in APERTURE_TABLE], dtype=float)

    observed_corr = float(np.corrcoef(dc, obs)[0, 1])
    null = []
    for _ in range(n_perm):
        d = rng.permutation(dc)
        null.append(float(np.corrcoef(d, obs)[0, 1]))
    null_arr = np.array(null)
    # one-sided: we predict NEGATIVE correlation
    p = float(((null_arr <= observed_corr).sum() + 1) / (n_perm + 1))

    return {
        "test": "V5.4 CONTROL: shuffled dC",
        "observed_corr": observed_corr,
        "n_permutations": n_perm,
        "null_mean_corr": float(null_arr.mean()),
        "null_5th_percentile": float(np.percentile(null_arr, 5)),
        "empirical_p_value_one_sided": p,
        "dC_carries_information": bool(p < 0.05),
        "passed": bool(p < 0.05),
        "interpretation": (
            "If p were large, the observed dC-efficiency relation would be "
            "reproducible by random assignment and the aperture count would "
            "be doing no work."
        ),
    }


# ---------------------------------------------------------------------------
def main() -> Dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tests = [v5_1_parameter_free_law(), v5_2_free_fit_comparison(),
             v5_3_exponential_vs_reciprocal(), v5_4_shuffle_control()]
    n_pass = sum(1 for t in tests if t["passed"])

    results = {
        "script": "v5_aperture_counting.py",
        "predictions": ["P3 aperture counting", "P4 exponential not reciprocal"],
        "aperture_rule": (
            "dC = number of elementary categorical transitions; concerted "
            "coordinate changes count as ONE aperture; sequential steps count "
            "separately. Assigned from mechanism WITHOUT reference to rate."
        ),
        "tests": tests,
        "summary": {"n_tests": len(tests), "n_passed": n_pass,
                    "all_passed": n_pass == len(tests)},
    }

    out = os.path.join(RESULTS_DIR, "v5_aperture_counting.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[V5] {n_pass}/{len(tests)} passed -> {out}")
    for t in tests:
        print(f"  {'PASS' if t['passed'] else 'FAIL'}  {t['test']}")
    return results


if __name__ == "__main__":
    main()
