"""Script 01 -- CYP2D6 allele rate constants: PM/IM/EM/UM hierarchy."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_cyp2d6_allele_rates"

rate_ratio_um_em = K_UM / K_EM
rate_ratio_em_im = K_EM / K_IM
rate_ratio_im_pm = K_IM / K_PM
rate_ratio_um_pm = K_UM / K_PM

data = {
    "k_UM_s":    round(K_UM, 2),
    "k_EM_s":    round(K_EM, 2),
    "k_IM_s":    round(K_IM, 2),
    "k_PM_s":    round(K_PM, 2),
    "ratio_UM_EM": round(rate_ratio_um_em, 3),
    "ratio_EM_IM": round(rate_ratio_em_im, 3),
    "ratio_IM_PM": round(rate_ratio_im_pm, 3),
    "ratio_UM_PM": round(rate_ratio_um_pm, 1),
}

checks = {
    "rate_order_UM_gt_EM":  K_UM > K_EM,
    "rate_order_EM_gt_IM":  K_EM > K_IM,
    "rate_order_IM_gt_PM":  K_IM > K_PM,
    "um_pm_ratio_gt_5":     rate_ratio_um_pm > 5.0,
    "pm_rate_lt_1e9":       K_PM < 1e9,
    "um_rate_gt_em_rate":   K_UM > K_EM,
}

write_result(name, data, checks)
