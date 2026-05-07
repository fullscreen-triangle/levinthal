"""Script 01 -- Alpha-carbon BDE ordering and DeltaM scaling.

Validates:
- BDE ordering: N-CH3 < O-CH3 < aliphatic
- DeltaM ordering mirrors BDE (lower BDE -> smaller DeltaM -> faster)
- Activation energy from T_PART * DeltaM in expected range
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_alpha_carbon_bde"

# BDE ordering
bde_n = BDE_N_CH3      # 87 kcal/mol
bde_o = BDE_O_CH3      # 92 kcal/mol
bde_ali = BDE_aliphatic # 100 kcal/mol

# DeltaM from BDE: scale relative to aliphatic reference
# DeltaM_x = DeltaM_ref * (BDE_x / BDE_ref)
delta_m_n_scaled = DELTA_M_ALIPHATIC * (bde_n / bde_ali)    # 0.65 * 87/100 = 0.5655
delta_m_o_scaled = DELTA_M_ALIPHATIC * (bde_o / bde_ali)    # 0.65 * 92/100 = 0.598

# Activation energies (kcal/mol) = T_PART * DeltaM / 4.184 (T_PART in kJ/mol)
ea_n_kcal = T_PART * DELTA_M_N_DEALK / 4.184    # 65 * 0.50 / 4.184 = 7.77 kcal/mol
ea_o_kcal = T_PART * DELTA_M_O_DEALK / 4.184    # 65 * 0.58 / 4.184 = 9.01 kcal/mol
ea_ali_kcal = T_PART * DELTA_M_ALIPHATIC / 4.184 # 65 * 0.65 / 4.184 = 10.09 kcal/mol

data = {
    "bde_n_ch3_kcal": bde_n,
    "bde_o_ch3_kcal": bde_o,
    "bde_aliphatic_kcal": bde_ali,
    "delta_m_n_dealk": DELTA_M_N_DEALK,
    "delta_m_o_dealk": DELTA_M_O_DEALK,
    "delta_m_aliphatic": DELTA_M_ALIPHATIC,
    "ea_n_kcal": round(ea_n_kcal, 3),
    "ea_o_kcal": round(ea_o_kcal, 3),
    "ea_aliphatic_kcal": round(ea_ali_kcal, 3),
}

checks = {
    "bde_ordering_n_lt_o_lt_ali": bde_n < bde_o < bde_ali,
    "delta_m_ordering_n_lt_o_lt_ali": DELTA_M_N_DEALK < DELTA_M_O_DEALK < DELTA_M_ALIPHATIC,
    "ea_n_in_range_6_to_10": 6.0 < ea_n_kcal < 10.0,
    "ea_ali_in_range_9_to_13": 9.0 < ea_ali_kcal < 13.0,
    "delta_m_scaled_consistent": abs(delta_m_n_scaled - DELTA_M_N_DEALK) < 0.10,
}

write_result(name, data, checks)
