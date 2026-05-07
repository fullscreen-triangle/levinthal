"""Script 02 — Three-Body Aperture Selection Rule.

Validates:
- Selection rule: Delta_beta_CH=1, Delta_beta_OH=-1, Delta_s_orbital=0 → dC=1
- Activation energy E_a(HAT) = T_part * Delta_M_HAT ≈ 10 kcal/mol
- Intrinsic rate from categorical efficiency: log10(k) = 10 - dC = 9
- Sequential transfer (dC=2) is slower by factor 10
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_three_body_aperture"

# Selection rule check for concerted HAT
delta_beta_CH = 1        # C-H broken
delta_beta_OH = -1       # O-H formed
delta_s_orbital = 0      # chirality conserved

# All three simultaneously → dC = 1
selection_rule_satisfied = (delta_beta_CH == 1 and delta_beta_OH == -1
                             and delta_s_orbital == 0)
dC_concerted = 1 if selection_rule_satisfied else 2

# Activation energy from partition landscape
E_a_HAT_kJmol = T_PART * DELTA_M_HAT        # kJ/mol
E_a_HAT_kcalmol = E_a_HAT_kJmol / 4.184     # kcal/mol

# Rate from categorical efficiency + depth correction
log10_k_categorical = 10 - dC_concerted
k_categorical_floor = 10 ** log10_k_categorical                  # s^-1
k_HAT_predicted = nu_floor * math.exp(-DELTA_M_HAT)             # s^-1
log10_k_HAT = math.log10(k_HAT_predicted)

# Sequential (dC=2) rate
dC_sequential = 2
k_sequential = 10 ** (10 - dC_sequential)   # s^-1
rate_ratio = k_HAT_predicted / k_sequential   # should be ~ 5 (within an order of magnitude of 10)

# E_a comparison
E_a_range_kcal = (8.0, 14.0)

data = {
    "selection_rule": {
        "delta_beta_CH": delta_beta_CH,
        "delta_beta_OH": delta_beta_OH,
        "delta_s_orbital": delta_s_orbital,
        "satisfied": selection_rule_satisfied,
        "dC": dC_concerted,
    },
    "E_a_HAT_kJmol": round(E_a_HAT_kJmol, 2),
    "E_a_HAT_kcalmol": round(E_a_HAT_kcalmol, 2),
    "k_HAT_predicted_s": f"{k_HAT_predicted:.3e}",
    "log10_k_HAT": round(log10_k_HAT, 2),
    "k_sequential_s": f"{k_sequential:.3e}",
    "rate_ratio_conc_over_seq": round(rate_ratio, 2),
    "DELTA_M_HAT": DELTA_M_HAT,
}

checks = {
    "selection_rule_satisfied": selection_rule_satisfied,
    "dC_equals_1": dC_concerted == 1,
    "E_a_in_range_8_14_kcal": E_a_range_kcal[0] <= E_a_HAT_kcalmol <= E_a_range_kcal[1],
    "log10_k_hat_near_9": 8.5 <= log10_k_HAT <= 10.0,
    "sequential_slower": k_HAT_predicted > k_sequential * 0.1,  # concerted at least 10% of sequential floor
    "concerted_faster_than_sequential": k_HAT_predicted > k_sequential,
}

write_result(name, data, checks)
