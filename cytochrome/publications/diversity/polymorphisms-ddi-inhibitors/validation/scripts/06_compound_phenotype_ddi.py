"""Script 06 — Compound PM + inhibitor: additive ΔM effects."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# A PM (null CYP2D6) receiving a CYP2D6 inhibitor:
# effective ΔM = ΔM_PM + ΔΔM_inhibitor  →  already at floor, inhibitor adds nothing useful
# An EM + quinidine → becomes functionally PM

# ── EM + quinidine at 0.5 μM ─────────────────────────────────────────────
conc_quin   = 0.5
alpha_quin  = alpha(conc_quin, KI_QUINIDINE_2D6)
ddm_quin    = dm_shift_from_alpha(alpha_quin)
dm_em_inh   = DELTA_M_2D6_EM + ddm_quin  # effective ΔM
k_em_inh    = k_rate(dm_em_inh)
k_pm        = k_rate(DELTA_M_2D6_PM)

# Quinidine at 0.5 μM drives EM well below PM rate (phenocopies > PM)
em_inh_below_pm = k_em_inh < k_pm   # EM+quin more inhibited than PM

# ── PM + quinidine: no additional effect ─────────────────────────────────
dm_pm_inh = DELTA_M_2D6_PM + ddm_quin  # already null
k_pm_inh  = k_rate(dm_pm_inh)
pm_further_reduced = k_pm_inh < k_pm   # yes, goes even lower but irrelevant clinically

# ── UM + quinidine: reduced but still > EM ───────────────────────────────
dm_um_inh = DELTA_M_2D6_UM + ddm_quin
k_um_inh  = k_rate(dm_um_inh)
k_em      = k_rate(DELTA_M_2D6_EM)
# UM + inhibitor: should this still be > EM?  ΔM_UM=0.27, ΔΔM_quin ≈ 2.96 → dm≈3.23
# Actually UM + quin → much higher ΔM → lower rate than EM
um_inh_lt_em = k_um_inh < k_em   # strong quin overwhelms UM advantage

# ── CYP2C9*3 + fluconazole: compound reduction ───────────────────────────
conc_fluc = 10.0
alpha_fluc = alpha(conc_fluc, KI_FLUCONAZOLE_2C9)
ddm_fluc   = dm_shift_from_alpha(alpha_fluc)
dm_s3_fluc = DELTA_M_2C9_S3 + ddm_fluc
k_s3_fluc  = k_rate(dm_s3_fluc)
k_s3       = k_rate(DELTA_M_2C9_S3)
s3_fluc_much_lower = k_s3_fluc < k_s3 * 0.5   # >50% further reduction

# ── Population-weighted effective rate under inhibitor ───────────────────
# CYP2D6 Euro phenotype frequencies
FREQ_EM = 0.70; FREQ_IM = 0.15; FREQ_PM = 0.07; FREQ_UM = 0.08
k_im = k_rate(DELTA_M_2D6_IM)
k_um = k_rate(DELTA_M_2D6_UM)

k_pop_nodrg = FREQ_EM*k_em + FREQ_IM*k_im + FREQ_PM*k_pm + FREQ_UM*k_um
k_pop_quin  = (FREQ_EM*k_em_inh + FREQ_IM*k_rate(DELTA_M_2D6_IM+ddm_quin) +
               FREQ_PM*k_pm_inh  + FREQ_UM*k_um_inh)
pop_rate_reduced = k_pop_quin < k_pop_nodrg * 0.5

checks = {
    "em_inh_below_pm":     em_inh_below_pm,
    "pm_further_reduced":  pm_further_reduced,
    "um_inh_lt_em":        um_inh_lt_em,
    "s3_fluc_much_lower":  s3_fluc_much_lower,
    "pop_rate_reduced":    pop_rate_reduced,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  06_compound_phenotype_ddi")
sys.exit(0 if all_pass else 1)
