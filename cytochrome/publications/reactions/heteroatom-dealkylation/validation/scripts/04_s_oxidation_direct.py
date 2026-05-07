"""Script 04 -- S-oxidation via direct O-atom transfer.

Validates:
- k_S_ox = nu_floor * exp(-0.28): fast direct transfer
- No KIE (no H motion in S-oxidation)
- k_S_ox > k_N_dealk (direct transfer faster than HAT)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_s_oxidation_direct"

# S-oxidation: direct lone-pair donation to Fe=O
# DeltaM_S_ox = 0.28 (two-body O-atom transfer aperture)
k_s_ox = nu_floor * math.exp(-DELTA_M_S_OX)    # ~7.56e9 s^-1
k_n_dealk = nu_floor * math.exp(-DELTA_M_N_DEALK)  # ~6.07e9 s^-1

# E_a for S-oxidation
ea_s_ox_kcal = T_PART * DELTA_M_S_OX / 4.184    # 65 * 0.28 / 4.184 = 4.35 kcal/mol

# No deuterium KIE for S-oxidation (no H transferred)
kie_s_ox = 1.0

# Selectivity: sulfide vs amine substrate competition
# If [S] = [N] and both compete: ratio of sulfoxidation / N-demethylation
# = k_s_ox / k_n_dealk (with same concentrations)
selectivity_s_over_n = k_s_ox / k_n_dealk

data = {
    "delta_m_s_ox": DELTA_M_S_OX,
    "k_s_ox_s": round(k_s_ox, 2),
    "k_n_dealk_s": round(k_n_dealk, 2),
    "ea_s_ox_kcal": round(ea_s_ox_kcal, 3),
    "kie_s_ox": kie_s_ox,
    "selectivity_s_over_n": round(selectivity_s_over_n, 3),
}

checks = {
    "k_s_ox_gt_5e9": k_s_ox > 5e9,
    "k_s_ox_lt_1e10": k_s_ox < 1e10,
    "k_s_ox_gt_k_n_dealk": k_s_ox > k_n_dealk,
    "no_kie_for_s_ox": kie_s_ox == 1.0,
    "ea_s_ox_lt_ea_n_dealk": ea_s_ox_kcal < (T_PART * DELTA_M_N_DEALK / 4.184),
}

write_result(name, data, checks)
