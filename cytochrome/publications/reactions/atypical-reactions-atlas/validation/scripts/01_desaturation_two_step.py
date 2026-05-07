"""Script 01 -- Desaturation as two-step HAT with rebound competition."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_desaturation_two_step"

# k_eff for desaturation: k1 * k2/(k2 + k_rebound)
k_desat_eff = K_DESAT_EFF
k_single_hat = K_DESAT_1    # aliphatic single HAT

# Branching fraction: what fraction of radical goes to desaturation vs hydroxylation
f_desat = K_DESAT_2 / (K_DESAT_2 + K_REBOUND)
f_hydroxylation = K_REBOUND / (K_DESAT_2 + K_REBOUND)

data = {
    "delta_m_desat_step1": DELTA_M_DESATURATION_1,
    "delta_m_desat_step2": DELTA_M_DESATURATION_2,
    "k_desat_step1_s": round(K_DESAT_1, 2),
    "k_desat_step2_s": round(K_DESAT_2, 2),
    "k_rebound_s": K_REBOUND,
    "k_desat_eff_s": round(k_desat_eff, 2),
    "k_single_hat_s": round(k_single_hat, 2),
    "fraction_desaturation": round(f_desat, 4),
    "fraction_hydroxylation": round(f_hydroxylation, 4),
}

checks = {
    "k_desat_eff_lt_k_single_hat": k_desat_eff < k_single_hat,
    "k_desat_eff_gt_1e8": k_desat_eff > 1e8,
    "k_desat_eff_gt_0": k_desat_eff > 0,
    "fraction_desat_between_0_and_1": 0 < f_desat < 1,
    "fraction_hydrox_gt_0": f_hydroxylation > 0,
    "fractions_sum_to_1": abs(f_desat + f_hydroxylation - 1.0) < 1e-9,
}

write_result(name, data, checks)
