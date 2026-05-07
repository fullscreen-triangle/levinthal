"""Script 03 -- Aromatic epoxidation (arene oxide formation) kinetics."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_arene_oxide"

k_epox = K_EPOX    # 1e10 * exp(-0.35)
kie_epox = 1.0     # no H-atom transfer; no primary KIE

ea_epox_kcal = T_PART * DELTA_M_EPOXIDATION / 4.184

data = {
    "delta_m_epoxidation": DELTA_M_EPOXIDATION,
    "k_epox_s": round(k_epox, 2),
    "kie_epox": kie_epox,
    "ea_epox_kcal": round(ea_epox_kcal, 3),
}

checks = {
    "k_epox_in_range_4e9_1e10": 4e9 < k_epox < 1e10,
    "kie_epox_is_1": kie_epox == 1.0,
    "ea_epox_lt_6": ea_epox_kcal < 6.0,
    "k_epox_gt_k_aliphatic": k_epox > nu_floor * math.exp(-0.65),
    "delta_m_epox_lt_delta_m_aliphatic": DELTA_M_EPOXIDATION < 0.65,
}

write_result(name, data, checks)
