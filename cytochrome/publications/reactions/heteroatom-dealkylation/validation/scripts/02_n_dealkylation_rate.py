"""Script 02 -- N-dealkylation intrinsic rate.

Validates:
- k_N_dealk = nu_floor * exp(-DeltaM_N) is in expected range
- Rate faster than aliphatic HAT (lower DeltaM)
- E_a consistent with literature (6-10 kcal/mol for N-dealkylation)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_n_dealkylation_rate"

# Intrinsic N-dealkylation rate
k_n_dealk = nu_floor * math.exp(-DELTA_M_N_DEALK)   # 1e10 * exp(-0.50) = 6.07e9
k_o_dealk = nu_floor * math.exp(-DELTA_M_O_DEALK)   # 1e10 * exp(-0.58) = 5.60e9
k_aliphatic = nu_floor * math.exp(-DELTA_M_ALIPHATIC) # 1e10 * exp(-0.65) = 5.22e9

# Rate ratio N vs aliphatic
ratio_n_ali = k_n_dealk / k_aliphatic    # ~1.16

# E_a (kcal/mol)
ea_n_kcal = T_PART * DELTA_M_N_DEALK / 4.184

data = {
    "delta_m_n_dealk": DELTA_M_N_DEALK,
    "k_n_dealk_s": round(k_n_dealk, 2),
    "k_o_dealk_s": round(k_o_dealk, 2),
    "k_aliphatic_s": round(k_aliphatic, 2),
    "ratio_n_dealk_to_aliphatic": round(ratio_n_ali, 3),
    "ea_n_dealk_kcal": round(ea_n_kcal, 3),
}

checks = {
    "k_n_dealk_in_range": 1e9 < k_n_dealk < 2e10,
    "k_n_dealk_gt_k_aliphatic": k_n_dealk > k_aliphatic,
    "k_n_dealk_gt_k_o_dealk": k_n_dealk > k_o_dealk,
    "rate_ordering_correct": k_n_dealk > k_o_dealk > k_aliphatic,
    "ea_n_in_range_6_to_10": 6.0 < ea_n_kcal < 10.0,
}

write_result(name, data, checks)
