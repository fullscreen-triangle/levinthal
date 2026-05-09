"""Script 04 — CYP induction: PXR pathway and AUC fold-change."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# Induction: CYP3A4 protein increases E-fold → k_eff increases E-fold
# Victim drug AUC decreases: R_AUC = 1/E_fold (for CYP3A4-metabolised drug)
# Apparent ΔM shift: ΔΔM_ind = -ln(E_fold) (lower effective barrier from more enzyme)

# ── Rifampicin (strong CYP3A4 inducer) ──────────────────────────────────
r_auc_rif   = auc_ratio_induction(RIFAMPICIN_FOLD)
ddm_rif     = -math.log(RIFAMPICIN_FOLD)
# Midazolam AUC drops to 1/20 = 5 % of baseline
midazolam_auc_frac = r_auc_rif
rif_auc_lt10pct = midazolam_auc_frac < 0.10

# ── Phenobarbital (moderate inducer, CAR/PXR) ───────────────────────────
r_auc_pb    = auc_ratio_induction(PHENOBARBITAL_FOLD)
ddm_pb      = -math.log(PHENOBARBITAL_FOLD)
pb_strong   = r_auc_pb < 1.0 / AUC_MODERATE_DDI   # AUC ratio < 0.5

# ── Omeprazole (weak CYP1A2 inducer, AhR) ───────────────────────────────
r_auc_om    = auc_ratio_induction(OMEPRAZOLE_FOLD)
om_weak     = r_auc_om >= 0.15   # AUC decrease < 85 % (weak inducer)

# ── ΔΔM ordering: stronger induction = more negative ΔΔM ────────────────
ddm_om      = -math.log(OMEPRAZOLE_FOLD)
ddm_order_correct = ddm_rif < ddm_pb < ddm_om

# ── Simvastatin: 90 % AUC reduction by rifampicin ───────────────────────
simva_frac = 1.0 / RIFAMPICIN_FOLD
simva_auc_lt10 = simva_frac < 0.10

# ── PXR EC50-based E_max model ───────────────────────────────────────────
# E_fold = 1 + E_max × [inducer] / (EC50 + [inducer])
E_max_rif  = 30.0  # maximum fold induction
EC50_rif   = 0.50  # μM
conc_rif   = 2.0   # μM (clinical)
E_fold_pxr = 1.0 + E_max_rif * conc_rif / (EC50_rif + conc_rif)
pxr_model_gt20 = E_fold_pxr > 20.0   # should exceed 20×

checks = {
    "rif_auc_lt10pct":      rif_auc_lt10pct,
    "pb_strong":            pb_strong,
    "om_weak":              om_weak,
    "ddm_order_correct":    ddm_order_correct,
    "simva_auc_lt10":       simva_auc_lt10,
    "pxr_model_gt20":       pxr_model_gt20,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  04_induction_fold")
sys.exit(0 if all_pass else 1)
