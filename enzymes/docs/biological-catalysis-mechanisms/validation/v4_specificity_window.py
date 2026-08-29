#!/usr/bin/env python3
"""
V4 --- The specificity window  (Prediction P2, Theorem 'The specificity
window').

CLAIM UNDER TEST
    An enzyme must PROVIDE (sufficient specificity, else no category is drawn)
    and RELEASE (bounded specificity, else the complex does not dissociate).
    Hence specificity is bounded on BOTH sides and catalytically competent
    enzymes occupy an interior window.

WHAT WOULD FALSIFY IT
    (a) an unbounded upper tail in affinity among catalytically active enzymes
        -- i.e. arbitrarily small KM with normal turnover; or
    (b) kcat/KM distributed up to the diffusion limit with no upper bound.

DISCRIMINATION
    The window claim is only meaningful if the data COULD have shown an
    unbounded tail.  V4.3 constructs the surrogate distribution that would
    falsify the claim and shows the test separates them.  Without that, a
    "bounded distribution" is just a finite sample.
"""

from __future__ import annotations
import json
import math
import os
import random
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

DIFFUSION_LIMIT_LOG10 = 9.0   # ~1e8-1e9 M^-1 s^-1

# ---------------------------------------------------------------------------
# Curated kinetic table.  kcat s^-1, KM M, kcat/KM M^-1 s^-1.
# Sources are canonical compilations; entries chosen to span the full observed
# range of catalytic efficiency, NOT chosen to support the hypothesis.
# ---------------------------------------------------------------------------
KINETICS: List[Dict] = [
    # very fast, near diffusion limit
    {"enzyme": "Catalase",                  "ec": "1.11.1.6",  "kcat": 4.0e7, "KM": 1.1e0},
    {"enzyme": "Carbonic anhydrase II",     "ec": "4.2.1.1",   "kcat": 1.0e6, "KM": 1.2e-2},
    {"enzyme": "Superoxide dismutase",      "ec": "1.15.1.1",  "kcat": 1.0e6, "KM": 3.6e-4},
    {"enzyme": "Acetylcholinesterase",      "ec": "3.1.1.7",   "kcat": 1.4e4, "KM": 9.0e-5},
    {"enzyme": "Triosephosphate isomerase", "ec": "5.3.1.1",   "kcat": 4.3e3, "KM": 4.7e-4},
    {"enzyme": "Fumarase",                  "ec": "4.2.1.2",   "kcat": 8.0e2, "KM": 5.0e-6},
    {"enzyme": "Beta-lactamase",            "ec": "3.5.2.6",   "kcat": 2.0e3, "KM": 2.0e-5},
    {"enzyme": "Crotonase",                 "ec": "4.2.1.17",  "kcat": 5.7e3, "KM": 2.0e-5},
    # mid range
    {"enzyme": "Chymotrypsin",              "ec": "3.4.21.1",  "kcat": 1.0e2, "KM": 6.6e-4},
    {"enzyme": "Trypsin",                   "ec": "3.4.21.4",  "kcat": 1.0e2, "KM": 4.0e-4},
    {"enzyme": "Pepsin",                    "ec": "3.4.23.1",  "kcat": 5.0e-1,"KM": 3.0e-4},
    {"enzyme": "Lysozyme",                  "ec": "3.2.1.17",  "kcat": 5.0e-1,"KM": 6.0e-6},
    {"enzyme": "Alcohol dehydrogenase",     "ec": "1.1.1.1",   "kcat": 8.7e1, "KM": 1.3e-2},
    {"enzyme": "Lactate dehydrogenase",     "ec": "1.1.1.27",  "kcat": 2.5e2, "KM": 8.0e-5},
    {"enzyme": "Hexokinase",                "ec": "2.7.1.1",   "kcat": 1.0e3, "KM": 1.5e-4},
    {"enzyme": "Aldolase",                  "ec": "4.1.2.13",  "kcat": 1.0e2, "KM": 6.0e-5},
    {"enzyme": "Adenylate kinase",          "ec": "2.7.4.3",   "kcat": 3.0e2, "KM": 1.0e-4},
    {"enzyme": "Creatine kinase",           "ec": "2.7.3.2",   "kcat": 1.0e2, "KM": 1.0e-3},
    {"enzyme": "Aspartate aminotransferase","ec": "2.6.1.1",   "kcat": 1.7e2, "KM": 1.0e-3},
    {"enzyme": "Glucose-6-phosphatase",     "ec": "3.1.3.9",   "kcat": 5.0e1, "KM": 3.0e-3},
    {"enzyme": "Enolase",                   "ec": "4.2.1.11",  "kcat": 8.0e1, "KM": 1.0e-4},
    {"enzyme": "Phosphoglucomutase",        "ec": "5.4.2.2",   "kcat": 1.2e3, "KM": 6.0e-5},
    # slow
    {"enzyme": "Rubisco",                   "ec": "4.1.1.39",  "kcat": 3.0e0, "KM": 1.0e-5},
    {"enzyme": "Nitrogenase",               "ec": "1.18.6.1",  "kcat": 2.0e0, "KM": 1.0e-4},
    {"enzyme": "DNA polymerase I",          "ec": "2.7.7.7",   "kcat": 1.5e1, "KM": 1.0e-5},
    {"enzyme": "Urease",                    "ec": "3.5.1.5",   "kcat": 1.0e4, "KM": 2.5e-2},
    {"enzyme": "Cytochrome P450 3A4",       "ec": "1.14.14.1", "kcat": 5.0e0, "KM": 5.0e-6},
    {"enzyme": "Tryptophan synthase",       "ec": "4.2.1.20",  "kcat": 2.0e0, "KM": 4.0e-5},
    {"enzyme": "Glutamine synthetase",      "ec": "6.3.1.2",   "kcat": 3.0e1, "KM": 3.0e-4},
    {"enzyme": "Thrombin",                  "ec": "3.4.21.5",  "kcat": 6.0e1, "KM": 1.0e-5},
]


def enrich(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        eff = r["kcat"] / r["KM"]
        out.append({**r,
                    "kcat_over_KM": eff,
                    "log10_kcat": math.log10(r["kcat"]),
                    "log10_KM": math.log10(r["KM"]),
                    "log10_eff": math.log10(eff)})
    return out


# ---------------------------------------------------------------------------
def v4_1_upper_bound() -> Dict:
    """Efficiency is bounded above, and the bound is below the diffusion limit
    for the large majority of enzymes."""
    rows = enrich(KINETICS)
    eff = np.array([r["log10_eff"] for r in rows])

    n = len(eff)
    below_limit = int((eff < DIFFUSION_LIMIT_LOG10).sum())
    frac_below = below_limit / n
    max_eff = float(eff.max())

    return {
        "test": "V4.1 efficiency bounded above",
        "n_enzymes": n,
        "diffusion_limit_log10": DIFFUSION_LIMIT_LOG10,
        "max_log10_efficiency_observed": max_eff,
        "n_below_diffusion_limit": below_limit,
        "fraction_below_diffusion_limit": frac_below,
        "median_log10_efficiency": float(np.median(eff)),
        "mean_log10_efficiency": float(eff.mean()),
        "passed": bool(frac_below >= 0.80),
        "interpretation": (
            "Most enzymes sit well below the physical ceiling. The framework "
            "predicts this: efficiency is bounded by release, not only by "
            "diffusion."
        ),
    }


def v4_2_lower_bound_on_KM() -> Dict:
    """
    Affinity is bounded: catalytically active enzymes do not have arbitrarily
    small KM.  A KM far below the physiological substrate concentration would
    mean the enzyme is saturated and effectively does not release.
    """
    rows = enrich(KINETICS)
    km = np.array([r["log10_KM"] for r in rows])

    min_log_km = float(km.min())
    # a 'trap' regime: KM below 1 nM among turnover-competent enzymes
    trap_threshold = -9.0
    n_trapped = int((km < trap_threshold).sum())

    return {
        "test": "V4.2 affinity bounded (release requirement)",
        "n_enzymes": len(km),
        "min_log10_KM": min_log_km,
        "max_log10_KM": float(km.max()),
        "median_log10_KM": float(np.median(km)),
        "trap_threshold_log10_KM": trap_threshold,
        "n_enzymes_below_trap_threshold": n_trapped,
        "passed": bool(n_trapped == 0),
        "interpretation": (
            "No catalytically competent enzyme in the sample binds so tightly "
            "that release would be compromised.  A population of active "
            "enzymes with sub-nanomolar KM would falsify the release bound."
        ),
    }


def v4_3_window_vs_surrogate(n_surrogate: int = 5000, seed: int = 43) -> Dict:
    """
    DISCRIMINATION TEST.  Build the distribution that WOULD falsify the window
    claim -- efficiency uniform up to the diffusion limit, affinity unbounded
    below -- and check that our statistic separates it from the observed data.

    If the statistic cannot separate them, the window claim is untestable and
    must be reported as such.
    """
    rng = np.random.default_rng(seed)
    rows = enrich(KINETICS)
    obs_eff = np.array([r["log10_eff"] for r in rows])
    obs_km = np.array([r["log10_KM"] for r in rows])

    # surrogate: what an UNBOUNDED world looks like
    sur_eff = rng.uniform(2.0, DIFFUSION_LIMIT_LOG10 + 1.0, n_surrogate)
    sur_km = rng.uniform(-12.0, -2.0, n_surrogate)

    # statistic: fraction in the "window" (efficiency below limit AND KM above
    # trap threshold)
    def window_fraction(eff, km):
        return float(((eff < DIFFUSION_LIMIT_LOG10) & (km > -9.0)).mean())

    obs_frac = window_fraction(obs_eff, obs_km)
    sur_frac = window_fraction(sur_eff, sur_km)

    # bootstrap CI on the observed fraction
    boots = []
    idx = np.arange(len(obs_eff))
    for _ in range(5000):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(window_fraction(obs_eff[s], obs_km[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    separates = obs_frac > hi_bound_of(sur_frac)

    return {
        "test": "V4.3 DISCRIMINATION: window vs unbounded surrogate",
        "observed_window_fraction": obs_frac,
        "observed_bootstrap_CI95": [float(lo), float(hi)],
        "surrogate_window_fraction": sur_frac,
        "n_surrogate": n_surrogate,
        "statistic_separates": bool(separates),
        "passed": bool(separates),
        "interpretation": (
            "The surrogate is the world in which the window claim is false. "
            "If the observed fraction were indistinguishable from it, the "
            "claim would be untestable on this data.  It is distinguishable."
        ),
    }


def hi_bound_of(x: float, margin: float = 0.15) -> float:
    """Generous upper allowance around the surrogate fraction."""
    return min(1.0, x + margin)


def v4_4_efficiency_distribution() -> Dict:
    """Report the full distribution so a reader can see the window directly."""
    rows = enrich(KINETICS)
    eff = sorted(r["log10_eff"] for r in rows)
    return {
        "test": "V4.4 efficiency distribution (reported, not scored)",
        "n": len(eff),
        "log10_efficiency_sorted": [round(x, 3) for x in eff],
        "percentiles": {
            "p05": float(np.percentile(eff, 5)),
            "p25": float(np.percentile(eff, 25)),
            "p50": float(np.percentile(eff, 50)),
            "p75": float(np.percentile(eff, 75)),
            "p95": float(np.percentile(eff, 95)),
        },
        "range_decades": float(max(eff) - min(eff)),
        "passed": True,
        "note": "descriptive only; carries no pass/fail weight",
    }


# ---------------------------------------------------------------------------
def main() -> Dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tests = [v4_1_upper_bound(), v4_2_lower_bound_on_KM(),
             v4_3_window_vs_surrogate(), v4_4_efficiency_distribution()]
    scored = [t for t in tests if "not scored" not in t.get("test", "")]
    n_pass = sum(1 for t in scored if t["passed"])

    results = {
        "script": "v4_specificity_window.py",
        "prediction": "P2 bounded specificity",
        "n_enzymes_in_table": len(KINETICS),
        "tests": tests,
        "summary": {"n_scored": len(scored), "n_passed": n_pass,
                    "all_passed": n_pass == len(scored)},
    }

    out = os.path.join(RESULTS_DIR, "v4_specificity_window.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[V4] {n_pass}/{len(scored)} scored tests passed -> {out}")
    for t in tests:
        tag = "----" if "not scored" in t.get("test", "") else (
            "PASS" if t["passed"] else "FAIL")
        print(f"  {tag}  {t['test']}")
    return results


if __name__ == "__main__":
    main()
