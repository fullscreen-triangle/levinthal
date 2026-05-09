"""Script 01 — Polymorphism ΔM shifts and allele rate ratios."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

results = {}

# ── CYP2D6 allele rate ratios relative to EM ────────────────────────────
k_em = k_rate(DELTA_M_2D6_EM)
k_im = k_rate(DELTA_M_2D6_IM)
k_pm = k_rate(DELTA_M_2D6_PM)
k_um = k_rate(DELTA_M_2D6_UM)

ratio_um_em = k_um / k_em          # should be > 1
ratio_em_im = k_em / k_im          # should be > 1 (IM slower than EM)
ratio_em_pm = k_em / k_pm          # should be >> 1 (PM much slower)

results["ratio_um_em"] = ratio_um_em
results["ratio_em_im"] = ratio_em_im
results["ratio_em_pm"] = ratio_em_pm

# Checks
um_em_gt1    = ratio_um_em > 1.0          # UM faster than EM
em_im_gt1    = ratio_em_im > 1.0          # EM faster than IM
em_pm_gt5    = ratio_em_pm > 5.0           # PM at least 5x slower than EM
pm_rate_lt1e9 = k_pm < 1e9               # PM rate < 1e9 s⁻¹

# ── CYP2C9 allele rate ratios ───────────────────────────────────────────
k_2c9_wt = k_rate(DELTA_M_2C9_WT)
k_2c9_s2 = k_rate(DELTA_M_2C9_S2)
k_2c9_s3 = k_rate(DELTA_M_2C9_S3)

ratio_wt_s3 = k_2c9_wt / k_2c9_s3
ratio_wt_s2 = k_2c9_wt / k_2c9_s2
frac_s3     = k_2c9_s3 / k_2c9_wt        # should be < 0.05

results["ratio_wt_s3"]  = ratio_wt_s3
results["ratio_wt_s2"]  = ratio_wt_s2
results["frac_s3_of_wt"] = frac_s3

s3_lt5pct   = frac_s3 < 0.05
s2_lt50pct  = k_2c9_s2 / k_2c9_wt < 0.50

# ── CYP3A4*22 expression reduction ──────────────────────────────────────
k_3a4_wt = k_rate(DELTA_M_3A4_WT)
k_3a4_22 = k_rate(DELTA_M_3A4_22)
frac_22  = k_3a4_22 / k_3a4_wt   # should be ~0.50

results["frac_3a4_22"] = frac_22
star22_near_half = 0.40 < frac_22 < 0.60

# ── Report ───────────────────────────────────────────────────────────────
checks = {
    "um_em_gt1":     um_em_gt1,
    "em_im_gt1":     em_im_gt1,
    "em_pm_gt5":     em_pm_gt5,
    "pm_rate_lt1e9": pm_rate_lt1e9,
    "s3_lt5pct":     s3_lt5pct,
    "s2_lt50pct":    s2_lt50pct,
    "star22_near_half": star22_near_half,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  01_polymorphism_dm_shift")
sys.exit(0 if all_pass else 1)
