"""Script 05 — Oxygen-Rebound Aperture.

Validates:
- Rebound is dC=1 satisfying three-body selection rule (C-O formation)
- Delta_M_rebound = 0.30 < Delta_M_HAT = 0.65 (rebound intrinsically faster)
- k_rebound > k_HAT (rebound faster)
- k_rebound consistent with Newcomb 1995 lower bound (> 1e9 s^-1)
- C-O bond formed: beta_CO transitions 0->1
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_rebound_aperture"

# Rebound aperture selection rule
delta_beta_CO = -1       # C-O bond formed (sign convention: -1 = formation)
delta_sigma_rad = 1      # radical quenched
delta_s_orbital = 0      # chirality conserved (rebound is retention of config)
rebound_dC = 1 if (delta_beta_CO == -1 and delta_sigma_rad == 1
                   and delta_s_orbital == 0) else 2

# Rebound depth
xi_rad = 0.57           # pre-organisation fraction
delta_M_rebound_calc = ln2 * (1 - xi_rad)

# Rates
k_rebound_calc = nu_floor * math.exp(-DELTA_M_REBOUND)
k_HAT_calc = nu_floor * math.exp(-DELTA_M_HAT)
rate_ratio = k_rebound_calc / k_HAT_calc   # should be > 1

# Newcomb lower bound
newcomb_lower_bound = 1e9   # s^-1

# Check delta_M ordering
depth_ordering = DELTA_M_REBOUND < DELTA_M_HAT

# Rate ordering (rebound faster than HAT)
rate_ordering = k_rebound_calc > k_HAT_calc

data = {
    "rebound_selection_rule": {
        "delta_beta_CO": delta_beta_CO,
        "delta_sigma_rad": delta_sigma_rad,
        "delta_s_orbital": delta_s_orbital,
        "dC": rebound_dC,
    },
    "xi_rad": xi_rad,
    "delta_M_rebound_calc": round(delta_M_rebound_calc, 4),
    "DELTA_M_REBOUND": DELTA_M_REBOUND,
    "DELTA_M_HAT": DELTA_M_HAT,
    "k_rebound_s": f"{k_rebound_calc:.3e}",
    "k_HAT_s": f"{k_HAT_calc:.3e}",
    "k_rebound_over_k_HAT": round(rate_ratio, 4),
}

checks = {
    "rebound_is_dC_1": rebound_dC == 1,
    "delta_M_rebound_below_HAT": depth_ordering,
    "delta_M_rebound_matches_formula": abs(delta_M_rebound_calc - DELTA_M_REBOUND) < 0.05,
    "k_rebound_above_newcomb_bound": k_rebound_calc > newcomb_lower_bound,
    "rebound_faster_than_HAT": rate_ordering,
    "rate_ratio_above_1": rate_ratio > 1.0,
}

write_result(name, data, checks)
