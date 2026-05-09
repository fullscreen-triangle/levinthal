"""Script 08 — Full DDI summary table: all checks in one pass."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# Recreate key values from scripts 01-07
checks = {}

# ── From script 01 ───────────────────────────────────────────────────────
k_em = k_rate(DELTA_M_2D6_EM);  k_pm = k_rate(DELTA_M_2D6_PM)
k_um = k_rate(DELTA_M_2D6_UM)
checks["ratio_em_pm_gt5"]   = (k_em / k_pm) > 5.0
checks["s3_lt5pct_wt"]      = k_rate(DELTA_M_2C9_S3) / k_rate(DELTA_M_2C9_WT) < 0.05
checks["star22_near_half"]   = 0.40 < k_rate(DELTA_M_3A4_22) / k_rate(DELTA_M_3A4_WT) < 0.60

# ── From script 02 ───────────────────────────────────────────────────────
rauc_keto = auc_ratio_inhibition(0.2, KI_KETOCONAZOLE_3A4)
checks["keto_strong_ddi"]   = rauc_keto > AUC_STRONG_DDI
ddm_keto  = dm_shift_from_alpha(rauc_keto)
checks["keto_ddm_gt1"]      = ddm_keto > 1.0

# ── From script 03 ───────────────────────────────────────────────────────
kobs_c = kobs_mbi(4.0, KI_CLARITHROMYCIN, KINACT_CLARITHROMYCIN)
checks["clari_kobs_gt_kdeg"] = kobs_c > KDEG_3A4
frac60 = math.exp(-(kobs_c + KDEG_3A4) * 60)
checks["clari_frac_lt50pct"] = frac60 < 0.5

# ── From script 04 ───────────────────────────────────────────────────────
checks["rif_auc_lt10pct"]    = auc_ratio_induction(RIFAMPICIN_FOLD) < 0.10
checks["pb_auc_lt50pct"]     = auc_ratio_induction(PHENOBARBITAL_FOLD) < 0.50

# ── From script 05 ───────────────────────────────────────────────────────
checks["itra_strongest_ki"]  = KI_ITRACONAZOLE_3A4 < KI_KETOCONAZOLE_3A4
rauc_quin = auc_ratio_inhibition(0.5, KI_QUINIDINE_2D6)
checks["quin_strong_ddi"]    = rauc_quin > AUC_STRONG_DDI

# ── From script 06 ───────────────────────────────────────────────────────
ddm_quin = dm_shift_from_alpha(rauc_quin)
dm_em_inh = DELTA_M_2D6_EM + ddm_quin
checks["em_quin_below_pm"]   = k_rate(dm_em_inh) < k_pm

# ── From script 07 ───────────────────────────────────────────────────────
pot_clari = KINACT_CLARITHROMYCIN / KI_CLARITHROMYCIN
pot_dilti  = KINACT_DILTIAZEM     / KI_DILTIAZEM
pot_ery    = KINACT_ERYTHROMYCIN  / KI_ERYTHROMYCIN
checks["mbi_potency_rank"]   = pot_clari > pot_dilti > pot_ery

for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

n_pass = sum(checks.values())
n_total = len(checks)
all_pass = n_pass == n_total
print(f"\n{n_pass}/{n_total} checks passed")
print(f"\n{'PASS' if all_pass else 'FAIL'}  08_full_ddi_table")
sys.exit(0 if all_pass else 1)
