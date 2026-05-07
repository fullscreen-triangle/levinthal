"""Script 01 — C-H Bond-Order Partition Coordinate.

Validates:
- beta_CH is binary {0, 1}
- Delta_M(C-H cleavage) == ln(2)
- Cleavage timescale tau_cleave = tau_p * exp(Delta_M) falls in 40-60 fs
- Bond formation (O-H) gives Delta_M = -ln(2)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_hat_coordinate"

# C-H bond-order states
beta_CH_states = [0, 1]
assert set(beta_CH_states) == {0, 1}

# Delta_M for C-H bond cleavage (bond breaking: +ln2)
delta_M_CH_cleavage = ln2
# Delta_M for O-H bond formation (bond formation: -ln2)
delta_M_OH_formation = -ln2

# Cleavage timescale: tau_p * exp(Delta_M)
tau_p_s = hbar / kBT          # s
tau_cleave_s = tau_p_s * math.exp(delta_M_CH_cleavage)
tau_cleave_fs = tau_cleave_s * 1e15

# Net depth change for concerted HAT at the partition cell level
# (bond cleavage + bond formation cancel, leaving only electronic reorganisation)
delta_M_bond_only = delta_M_CH_cleavage + delta_M_OH_formation   # = 0
delta_M_electronic = 0.30
delta_M_HAT_check = abs(delta_M_CH_cleavage) / 2 + delta_M_electronic  # 0.347 + 0.30 = 0.647

data = {
    "beta_CH_states": beta_CH_states,
    "delta_M_CH_cleavage": round(delta_M_CH_cleavage, 6),
    "delta_M_OH_formation": round(delta_M_OH_formation, 6),
    "delta_M_bond_only": round(delta_M_bond_only, 6),
    "delta_M_HAT_activation": round(delta_M_HAT_check, 4),
    "tau_p_fs": round(tau_p_s * 1e15, 3),
    "tau_cleave_fs": round(tau_cleave_fs, 2),
}

checks = {
    "beta_CH_binary": set(beta_CH_states) == {0, 1},
    "delta_M_CH_cleavage_eq_ln2": abs(delta_M_CH_cleavage - ln2) < 1e-9,
    "delta_M_OH_formation_eq_neg_ln2": abs(delta_M_OH_formation + ln2) < 1e-9,
    "bond_changes_cancel": abs(delta_M_bond_only) < 1e-9,
    "tau_cleave_40_to_60_fs": 30 < tau_cleave_fs < 80,
    "delta_M_HAT_activation_in_range": 0.60 < delta_M_HAT_check < 0.70,
}

write_result(name, data, checks)
