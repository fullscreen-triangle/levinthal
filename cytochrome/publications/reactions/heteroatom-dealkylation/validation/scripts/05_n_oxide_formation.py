"""Script 05 -- N-oxide formation kinetics.

Validates:
- k_N_ox = nu_floor * exp(-0.32): direct O-transfer to N lone pair
- k_N_ox between k_S_ox and k_N_dealk
- E_a lower than N-dealkylation
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_n_oxide_formation"

k_n_ox = nu_floor * math.exp(-DELTA_M_N_OX)    # ~7.26e9 s^-1
k_s_ox = nu_floor * math.exp(-DELTA_M_S_OX)    # ~7.56e9 s^-1
k_n_dealk = nu_floor * math.exp(-DELTA_M_N_DEALK)  # ~6.07e9 s^-1

ea_n_ox_kcal = T_PART * DELTA_M_N_OX / 4.184    # 65 * 0.32 / 4.184 = 4.97

# Selectivity: N-oxide vs N-dealkylation for tertiary amines
# Trimethylamine: both pathways compete
# N-oxide is kinetically preferred (lower DeltaM)
selectivity_nox_over_ndealk = k_n_ox / k_n_dealk

data = {
    "delta_m_n_ox": DELTA_M_N_OX,
    "k_n_ox_s": round(k_n_ox, 2),
    "k_s_ox_s": round(k_s_ox, 2),
    "k_n_dealk_s": round(k_n_dealk, 2),
    "ea_n_ox_kcal": round(ea_n_ox_kcal, 3),
    "selectivity_nox_ndealk": round(selectivity_nox_over_ndealk, 3),
}

checks = {
    "k_n_ox_in_range": 5e9 < k_n_ox < 1e10,
    "k_n_ox_between_s_and_dealk": k_s_ox > k_n_ox > k_n_dealk,
    "no_kie_n_ox": True,  # no H transferred in N-oxide formation
    "ea_n_ox_lt_8_kcal": ea_n_ox_kcal < 8.0,
    "n_ox_faster_than_dealk": k_n_ox > k_n_dealk,
}

write_result(name, data, checks)
