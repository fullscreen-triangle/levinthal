"""Script 02 -- CYP2C9 allele effect on warfarin dose (S-warfarin hydroxylation)."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_cyp2c9_warfarin_dosing"

# Warfarin dose ∝ 1/k_met (maintenance dose inversely proportional to metabolic rate)
# Normalized to EM (*1/*1) baseline
dose_em = 1.0
dose_2  = K_2C9_EM / K_2C9_2   # *1/*2: ~1.3x dose needed
dose_3  = K_2C9_EM / K_2C9_3   # *1/*3: ~4-10x dose reduction needed (>4x)

# *3/*3 homozygous: rate ~ K_2C9_3^2 / K_2C9_EM (simplified homozygous model)
# Approximate: k_homozyg_3 ≈ K_2C9_3 (already represents the isoform's reduced activity)
dose_33 = K_2C9_EM / K_2C9_3   # same as *1/*3 for single allele model

data = {
    "k_2c9_EM_s":    round(K_2C9_EM, 2),
    "k_2c9_star2_s": round(K_2C9_2, 2),
    "k_2c9_star3_s": round(K_2C9_3, 2),
    "relative_dose_star2": round(dose_2, 3),
    "relative_dose_star3": round(dose_3, 3),
}

checks = {
    "k_star2_lt_k_em":          K_2C9_2 < K_2C9_EM,
    "k_star3_lt_k_star2":       K_2C9_3 < K_2C9_2,
    "dose_star3_gt_3x":         dose_3 > 3.0,
    "dose_star2_between_1_2":   1.0 < dose_2 < 2.0,
    "em_rate_gt_star2_rate":    K_2C9_EM > K_2C9_2,
    "star3_rate_lt_5pct_em":    K_2C9_3 < 0.05 * K_2C9_EM,
}

write_result(name, data, checks)
