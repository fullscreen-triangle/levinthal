"""
Validation 01: Floor Theorem for the receiver R_bio.

Verifies Theorem 7.1 (Paper 1, Eq. 11 floor decomposition):

    Floor(R_bio) = Floor_disc + Floor_Q + Floor_conv > 0

with explicit numerical evaluation under the receiver's reference parameters.

Outputs: results/01_floor_theorem.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Receiver reference parameters (Paper 1, Sec. 7, after Eq. 11)
RECURSION_DEPTH = 9
INTEGRATION_TIME_S = 1.0e-3

# Four hardware oscillators (CPU clock, memory bus, GPU/LED, refresh)
# (frequency_Hz, Q_factor) per Paper 1 Sec. 4.2 / Strobes paper Sec. 5
OSCILLATORS = [
    {"name": "CPU clock",   "freq_hz": 3.0e9,  "Q": 1.0e6},
    {"name": "memory bus",  "freq_hz": 8.0e8,  "Q": 1.0e5},
    {"name": "GPU/LED",     "freq_hz": 4.6e14, "Q": 1.0e8},
    {"name": "refresh/RTC", "freq_hz": 6.4e4,  "Q": 1.0e4},
]


def floor_disc(d: int) -> float:
    """Discretisation floor: 1/(2 * 3^d)."""
    return 1.0 / (2.0 * 3 ** d)


def floor_Q(oscillators: list[dict], T_int: float) -> dict:
    """Allan-deviation-style oscillator floor, summed in quadrature."""
    sigmas = []
    for osc in oscillators:
        sigma = 1.0 / (osc["Q"] * math.sqrt(T_int * osc["freq_hz"]))
        sigmas.append({"oscillator": osc["name"], "sigma": sigma})
    quadrature = math.sqrt(sum(s["sigma"] ** 2 for s in sigmas))
    return {"per_oscillator": sigmas, "quadrature_sum": quadrature}


def floor_conv(d: int, n_functors: int = 6) -> float:
    """Conversion functor discretisation floor: n_functors / 3^d."""
    return n_functors / 3.0 ** d


def main() -> dict:
    f_disc = floor_disc(RECURSION_DEPTH)
    f_Q = floor_Q(OSCILLATORS, INTEGRATION_TIME_S)
    f_conv = floor_conv(RECURSION_DEPTH)
    f_total = f_disc + f_Q["quadrature_sum"] + f_conv

    # Sanity checks
    checks = {
        "floor_disc_positive": f_disc > 0.0,
        "floor_Q_positive": f_Q["quadrature_sum"] > 0.0,
        "floor_conv_positive": f_conv > 0.0,
        "floor_total_positive": f_total > 0.0,
        "floor_total_finite": math.isfinite(f_total),
        "matches_paper_estimate_3p7e-4": math.isclose(
            f_total, 3.7e-4, rel_tol=0.5
        ),
    }

    # Sensitivity analysis: how does floor scale with recursion depth?
    depth_sweep = []
    for d in range(6, 13):
        depth_sweep.append({
            "depth": d,
            "floor_disc": floor_disc(d),
            "floor_conv": floor_conv(d),
            "floor_total": (
                floor_disc(d)
                + f_Q["quadrature_sum"]
                + floor_conv(d)
            ),
        })

    result = {
        "validation_id": "01_floor_theorem",
        "paper_reference": "Paper 1, Theorem 7.1, Eq. 11",
        "parameters": {
            "recursion_depth": RECURSION_DEPTH,
            "integration_time_s": INTEGRATION_TIME_S,
            "oscillators": OSCILLATORS,
        },
        "floor_components": {
            "floor_disc": f_disc,
            "floor_Q": f_Q,
            "floor_conv": f_conv,
        },
        "floor_total": f_total,
        "paper_quoted_estimate": 3.7e-4,
        "checks": checks,
        "depth_sensitivity": depth_sweep,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "01_floor_theorem.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] Floor(R_bio) = {out['floor_total']:.4e}")
    print(f"  paper estimate: {out['paper_quoted_estimate']:.4e}")
    print(f"  -> wrote {out_path}")
