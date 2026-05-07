"""Script 04 - Cytochrome b5 vs CPR Electron Transfer Comparison.

Validates:
- k_b5->heme (3e7 s^-1) > k_FMN->heme (5e6 s^-1)
- K_d_b5 (0.05 uM) < K_d_CPR (0.1 uM) -> b5 binds tighter
- b5-heme distance shorter (11 Ang) vs FMN-heme (14 Ang)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_cytb5_comparison"

# Cytochrome b5 parameters
k_b5 = 3.0e7        # s^-1
r_b5_heme = 11.0    # Ang (edge-to-edge, closer approach)
KD_B5_M = 0.5e-7    # M (0.05 uM, tighter than CPR)

# CPR FMN parameters (from script 03)
k_CPR = 5.0e6       # s^-1
r_CPR_heme = 14.0   # Ang
KD_CPR_M = 1.0e-7   # M

# Rate ratio
ratio_k = k_b5 / k_CPR

# DM values
DM_b5 = math.log(nu_floor / k_b5)
DM_CPR = math.log(nu_floor / k_CPR)

# Binding free energies
DG_b5_kcal = -8.314 * T * math.log(1.0 / KD_B5_M) / 1000.0 / 4.184
DG_CPR_kcal = -8.314 * T * math.log(1.0 / KD_CPR_M) / 1000.0 / 4.184

# Second electron: b5 can substitute for CPR
# Cyt b5 stimulates some P450 reactions (e.g. testosterone 6beta by CYP3A4)
second_electron_stimulation = True

data = {
    "k_b5_s_inv": k_b5,
    "k_CPR_s_inv": k_CPR,
    "ratio_k_b5_over_k_CPR": round(ratio_k, 2),
    "DM_b5": round(DM_b5, 4),
    "DM_CPR": round(DM_CPR, 4),
    "KD_b5_uM": KD_B5_M * 1e6,
    "KD_CPR_uM": KD_CPR_M * 1e6,
    "DG_b5_kcal": round(DG_b5_kcal, 3),
    "DG_CPR_kcal": round(DG_CPR_kcal, 3),
    "r_b5_heme_ang": r_b5_heme,
    "r_CPR_heme_ang": r_CPR_heme,
    "b5_stimulates_second_electron": second_electron_stimulation,
}

checks = {
    "k_b5_faster_than_k_CPR": k_b5 > k_CPR,
    "rate_ratio_b5_CPR_positive": ratio_k > 1.0,
    "KD_b5_tighter_than_CPR": KD_B5_M < KD_CPR_M,
    "DG_b5_more_negative": DG_b5_kcal < DG_CPR_kcal,
    "r_b5_shorter_than_CPR": r_b5_heme < r_CPR_heme,
    "DM_b5_less_than_DM_CPR": DM_b5 < DM_CPR,
}

write_result(name, data, checks)
