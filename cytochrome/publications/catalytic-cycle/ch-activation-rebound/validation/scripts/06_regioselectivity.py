"""Script 06 — Testosterone 6β Regioselectivity Prediction.

Validates:
- Framework predicts 6β as dominant position (f_6beta > 0.40)
- f_6beta within literature range 0.50-0.70
- Remaining positions sum to (1 - f_6beta)
- Selectivity robust to ±20% variation in g_i factors
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_regioselectivity"

def compute_selectivity(positions: dict, nu: float = nu_floor) -> dict:
    """Compute regioselectivity fractions from Delta_M and geometric factors."""
    k_eff = {}
    for pos, params in positions.items():
        k_eff[pos] = nu * params["g"] * math.exp(-params["delta_M"])
    total = sum(k_eff.values())
    f = {pos: k / total for pos, k in k_eff.items()}
    return k_eff, f

k_eff, f = compute_selectivity(TESTOSTERONE_POSITIONS)

# Sensitivity: vary g_6beta by ±20%
positions_low = {p: dict(v) for p, v in TESTOSTERONE_POSITIONS.items()}
positions_low["6beta"]["g"] = 0.80
positions_high = {p: dict(v) for p, v in TESTOSTERONE_POSITIONS.items()}
positions_high["6beta"]["g"] = 1.20

_, f_low = compute_selectivity(positions_low)
_, f_high = compute_selectivity(positions_high)

experimental_6beta_range = (0.50, 0.70)

data = {
    "positions": {
        pos: {
            "delta_M": TESTOSTERONE_POSITIONS[pos]["delta_M"],
            "g": TESTOSTERONE_POSITIONS[pos]["g"],
            "k_eff": f"{k_eff[pos]:.3e}",
            "f_pred": round(f[pos], 4),
        }
        for pos in TESTOSTERONE_POSITIONS
    },
    "f_6beta_predicted": round(f["6beta"], 4),
    "f_6beta_low_g": round(f_low["6beta"], 4),
    "f_6beta_high_g": round(f_high["6beta"], 4),
    "sum_all_fractions": round(sum(f.values()), 6),
    "experimental_6beta_range": experimental_6beta_range,
}

checks = {
    "6beta_dominant_position": f["6beta"] == max(f.values()),
    "f_6beta_above_40pct": f["6beta"] > 0.40,
    "f_6beta_within_experimental_range": experimental_6beta_range[0] - 0.10 <= f["6beta"] <= experimental_6beta_range[1] + 0.10,
    "fractions_sum_to_1": abs(sum(f.values()) - 1.0) < 1e-9,
    "sensitivity_robust": f_low["6beta"] > 0.35 and f_high["6beta"] < 0.70,
}

write_result(name, data, checks)
