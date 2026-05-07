"""Script 07 - Chemical vs ET Rate Hierarchy.

Validates:
- Chemical steps (Cpd0->CpdI, HAT) much faster than FMN->heme tunneling
- k_chem / k_ET_paper11 ratio >= 100 (chemistry not rate-limiting)
- k_CpdI = nu_floor * exp(-0.693) ~ 5e9 s^-1
- k_FMN_heme_paper11 = 5e6 s^-1 (from Paper 11)
- ratio ~ 1000 (k_chem >> k_ET)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_rate_hierarchy"

# Chemical step rates (within the cycle, from DM_STEPS)
k_heterolysis = K_STEPS["Cpd0_to_CpdI_heterolysis"]   # ~5e9 s^-1
k_HAT         = K_STEPS["CpdI_HAT_activation"]         # ~5.2e9 s^-1
k_protonation = K_STEPS["5_to_Cpd0_protonation"]       # ~6.4e9 s^-1
k_O2          = K_STEPS["3_to_4_O2_binding"]           # ~5.8e9 s^-1

# Comparison: FMN->heme tunneling from Paper 11 (the slow physical ET step)
k_ET_paper11 = K_FMN_HEME_PAPER11  # 5e6 s^-1

# Rate ratios
ratio_heterolysis = k_heterolysis / k_ET_paper11
ratio_HAT         = k_HAT / k_ET_paper11
ratio_protonation = k_protonation / k_ET_paper11
ratio_O2          = k_O2 / k_ET_paper11

min_chem_ratio = min(ratio_heterolysis, ratio_HAT, ratio_protonation, ratio_O2)

data = {
    "k_heterolysis_s_inv": round(k_heterolysis, 0),
    "k_HAT_s_inv":         round(k_HAT, 0),
    "k_protonation_s_inv": round(k_protonation, 0),
    "k_O2_binding_s_inv":  round(k_O2, 0),
    "k_FMN_heme_paper11":  k_ET_paper11,
    "ratio_heterolysis_ET": round(ratio_heterolysis, 0),
    "ratio_HAT_ET":         round(ratio_HAT, 0),
    "ratio_protonation_ET": round(ratio_protonation, 0),
    "ratio_O2_ET":          round(ratio_O2, 0),
    "min_chem_over_ET_ratio": round(min_chem_ratio, 0),
    "DM_FMN_heme_paper11":  round(DM_FMN_HEME, 4),
}

checks = {
    "k_heterolysis_faster_than_ET": k_heterolysis > k_ET_paper11,
    "k_HAT_faster_than_ET":         k_HAT > k_ET_paper11,
    "min_chem_ratio_ge_100":        min_chem_ratio >= 100.0,
    "ratio_HAT_ET_ge_100":          ratio_HAT >= 100.0,
    "ET_is_rate_determining_over_chemistry": k_ET_paper11 < k_HAT and k_ET_paper11 < k_heterolysis,
}

write_result(name, data, checks)
