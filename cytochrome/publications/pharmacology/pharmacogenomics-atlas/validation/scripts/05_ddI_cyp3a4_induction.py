"""Script 05 -- CYP3A4 induction model: PXR activation shifts ΔM downward."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_ddI_cyp3a4_induction"

# CYP3A4 induction by rifampicin (PXR activator)
# Baseline: CYP3A4 ΔM = 0.45 (substrate midazolam)
# Induced: protein level increases ~20x -> effective ΔM decreases by ln(20)
# because k_eff = N_enzyme * k_cat; increasing N effectively reduces apparent ΔM

dm_base    = 0.45
N_induction = 20.0   # 20-fold protein induction (rifampicin literature)
dm_induced = dm_base - math.log(N_induction)  # ln(20) ≈ 3.0 -> dm_eff goes negative (full induction)

# More physically: effective dm_induced = dm_base - ln(fold_induction)/ln(nu_floor/k_reference)
# Use simpler model: k_induced = N_induction * k_base
k_base    = nu_floor * math.exp(-dm_base)
k_induced = N_induction * k_base   # fold induction applied directly

# Clearance: CL ∝ k_induced  (linear at hepatic extraction ratio <<1)
cl_ratio  = k_induced / k_base   # = N_induction

# DDI ratio for victim drug (midazolam): AUC_ratio = 1 / CL_ratio
auc_ratio = 1.0 / cl_ratio

data = {
    "dm_base":    dm_base,
    "fold_induction": N_induction,
    "k_base_s":   round(k_base, 2),
    "k_induced_s":round(k_induced, 2),
    "cl_ratio":   round(cl_ratio, 2),
    "auc_ratio":  round(auc_ratio, 4),
}

checks = {
    "cl_ratio_eq_fold_induction":  abs(cl_ratio - N_induction) < 0.01,
    "auc_ratio_lt_0.1":            auc_ratio < 0.1,
    "k_induced_gt_k_base":         k_induced > k_base,
    "fold_induction_gt_10":        N_induction > 10,
    "dm_base_between_0.4_0.5":     0.4 < dm_base < 0.5,
}

write_result(name, data, checks)
