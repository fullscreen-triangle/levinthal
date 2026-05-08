"""Script 04 -- Codeine (O-demethylation by CYP2D6) toxicity in UM patients."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_codeine_toxicity_model"

# Codeine -> morphine via CYP2D6 O-demethylation
# Standard dose D_std; morphine plasma level ~ D_std * k_DM / k_elim
# For UM: k_DM_UM >> k_DM_EM -> excess morphine -> respiratory depression risk

# Morphine relative exposure = k_DM / k_elim, normalized to EM baseline
k_elim = 2e-4  # morphine elimination rate constant (s^-1, approximate)
morphine_em = K_EM * 1e-6 / k_elim    # units cancel; relative proxy
morphine_um = K_UM * 1e-6 / k_elim
morphine_pm = K_PM * 1e-6 / k_elim

ratio_um_em = morphine_um / morphine_em
ratio_pm_em = morphine_pm / morphine_em

# UM patients produce > 1.3x morphine (clinical threshold for excess exposure)
# PM patients produce < 0.7x (inadequate analgesia)
data = {
    "k_codeine_EM_s":   round(K_EM, 2),
    "k_codeine_UM_s":   round(K_UM, 2),
    "k_codeine_PM_s":   round(K_PM, 2),
    "morphine_ratio_UM_EM": round(ratio_um_em, 4),
    "morphine_ratio_PM_EM": round(ratio_pm_em, 4),
}

checks = {
    "UM_excess_morphine":       ratio_um_em > 1.3,
    "PM_reduced_morphine":      ratio_pm_em < 0.7,
    "UM_PM_ratio_gt_2":         ratio_um_em / ratio_pm_em > 2.0,
    "k_UM_gt_k_EM":             K_UM > K_EM,
    "k_EM_gt_k_PM":             K_EM > K_PM,
    "ratio_um_em_between_1_3":  1.0 < ratio_um_em < 3.0,
}

write_result(name, data, checks)
