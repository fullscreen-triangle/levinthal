"""Script 02 — Competitive inhibition: α modulus and apparent ΔM shift."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

results = {}

# ── Ketoconazole on CYP3A4 at clinical [I] = 0.2 μM ────────────────────
conc_keto = 0.2   # μM unbound plasma
alpha_keto = alpha(conc_keto, KI_KETOCONAZOLE_3A4)
ddm_keto   = dm_shift_from_alpha(alpha_keto)
rauc_keto  = auc_ratio_inhibition(conc_keto, KI_KETOCONAZOLE_3A4)

results["alpha_ketoconazole"] = alpha_keto
results["ddm_ketoconazole"]   = ddm_keto
results["rauc_ketoconazole"]  = rauc_keto

keto_strong   = rauc_keto > AUC_STRONG_DDI
keto_ddm_gt1  = ddm_keto > 1.0

# ── Quinidine on CYP2D6 at [I] = 0.5 μM ────────────────────────────────
conc_quin  = 0.5
alpha_quin = alpha(conc_quin, KI_QUINIDINE_2D6)
ddm_quin   = dm_shift_from_alpha(alpha_quin)
rauc_quin  = auc_ratio_inhibition(conc_quin, KI_QUINIDINE_2D6)

results["alpha_quinidine"] = alpha_quin
results["ddm_quinidine"]   = ddm_quin
quin_strong = rauc_quin > AUC_STRONG_DDI

# ── Fluoxetine on CYP2D6 at [I] = 0.5 μM ───────────────────────────────
conc_fluox = 0.5
alpha_fluox = alpha(conc_fluox, KI_FLUOXETINE_2D6)
ddm_fluox   = dm_shift_from_alpha(alpha_fluox)
rauc_fluox  = auc_ratio_inhibition(conc_fluox, KI_FLUOXETINE_2D6)

results["alpha_fluoxetine"] = alpha_fluox
fluox_moderate = AUC_MODERATE_DDI <= rauc_fluox < AUC_STRONG_DDI

# ── Fluconazole on CYP2C9 at [I] = 10 μM ───────────────────────────────
conc_fluc  = 10.0
alpha_fluc = alpha(conc_fluc, KI_FLUCONAZOLE_2C9)
rauc_fluc  = auc_ratio_inhibition(conc_fluc, KI_FLUCONAZOLE_2C9)
ddm_fluc   = dm_shift_from_alpha(alpha_fluc)

results["alpha_fluconazole_2c9"] = alpha_fluc
fluc_moderate = rauc_fluc >= AUC_MODERATE_DDI

# ── Apparent rate under inhibition: keto substantially reduces 3A4 ──────
k_3a4_wt  = k_rate(DELTA_M_3A4_WT)
k_3a4_inh = k_3a4_wt / alpha_keto    # apparent rate with ketoconazole
# Inhibited rate should be < 20% of uninhibited
inh_reduces_rate = k_3a4_inh < k_3a4_wt * 0.20

# ── DDI classification consistency ──────────────────────────────────────
# Within CYP3A4: itraconazole tighter than ketoconazole
# Within CYP2D6: quinidine tighter than fluoxetine
ki_rank_correct = (KI_ITRACONAZOLE_3A4 < KI_KETOCONAZOLE_3A4 and
                   KI_QUINIDINE_2D6 < KI_FLUOXETINE_2D6)

checks = {
    "keto_strong":          keto_strong,
    "keto_ddm_gt1":         keto_ddm_gt1,
    "quin_strong":          quin_strong,
    "fluox_moderate":       fluox_moderate,
    "fluc_moderate":        fluc_moderate,
    "inh_reduces_rate":     inh_reduces_rate,
    "ki_rank_correct":      ki_rank_correct,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  02_competitive_inhibition_alpha")
sys.exit(0 if all_pass else 1)
