"""Script 06 -- Competitive inhibition model: Ki shifts apparent ΔM."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_inhibition_competitive"

# Competitive inhibition of CYP2D6 by fluoxetine
# Ki(fluoxetine for CYP2D6) ≈ 0.24 uM (literature)
# At clinical [I] ≈ 0.5 uM: alpha = 1 + [I]/Ki
Ki_uM   = 0.24
I_uM    = 0.50
alpha   = 1.0 + I_uM / Ki_uM

# Apparent Km increases by alpha; Vmax unchanged
# Apparent ΔM_apparent = ΔM_EM + ln(alpha)
dm_apparent = DELTA_M_EM + math.log(alpha)
k_apparent  = nu_floor * math.exp(-dm_apparent)

# Inhibition ratio: R = 1 / alpha (fraction of uninhibited rate)
R_inhibition = 1.0 / alpha

# DDI prediction: AUC_inhibited / AUC_baseline = alpha for a low-ER victim
auc_ratio_ddi = alpha   # for low hepatic extraction drugs

data = {
    "Ki_uM":          Ki_uM,
    "I_uM":           I_uM,
    "alpha":          round(alpha, 4),
    "dm_apparent":    round(dm_apparent, 4),
    "k_apparent_s":   round(k_apparent, 2),
    "R_inhibition":   round(R_inhibition, 4),
    "auc_ratio_ddi":  round(auc_ratio_ddi, 4),
}

checks = {
    "alpha_gt_1":                 alpha > 1.0,
    "dm_apparent_gt_dm_em":       dm_apparent > DELTA_M_EM,
    "k_apparent_lt_k_em":         k_apparent < K_EM,
    "R_inhibition_lt_0.50":       R_inhibition < 0.50,
    "auc_ddi_gt_1.5":             auc_ratio_ddi > 1.5,
    "alpha_between_2_and_4":      2.0 < alpha < 4.0,
}

write_result(name, data, checks)
