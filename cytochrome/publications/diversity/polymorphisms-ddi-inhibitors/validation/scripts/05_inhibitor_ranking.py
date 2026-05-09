"""Script 05 — Inhibitor DDI risk ranking by AUC ratio."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# Clinical unbound concentrations [I]_u (μM) from literature
CLINICAL_CONC = {
    "ketoconazole_3A4":  0.2,
    "itraconazole_3A4":  0.1,
    "quinidine_2D6":     0.5,
    "paroxetine_2D6":    0.2,
    "fluoxetine_2D6":    0.5,
    "fluconazole_2C9":  10.0,
}

KI_MAP = {
    "ketoconazole_3A4":  KI_KETOCONAZOLE_3A4,
    "itraconazole_3A4":  KI_ITRACONAZOLE_3A4,
    "quinidine_2D6":     KI_QUINIDINE_2D6,
    "paroxetine_2D6":    KI_PAROXETINE_2D6,
    "fluoxetine_2D6":    KI_FLUOXETINE_2D6,
    "fluconazole_2C9":   KI_FLUCONAZOLE_2C9,
}

rauc = {k: auc_ratio_inhibition(CLINICAL_CONC[k], KI_MAP[k]) for k in KI_MAP}
ddm  = {k: dm_shift_from_alpha(rauc[k]) for k in rauc}

# ── Classification ───────────────────────────────────────────────────────
strong   = {k: v > AUC_STRONG_DDI   for k, v in rauc.items()}
moderate = {k: AUC_MODERATE_DDI <= v <= AUC_STRONG_DDI for k, v in rauc.items()}

n_strong   = sum(strong.values())
n_moderate = sum(moderate.values())

# Ketoconazole and itraconazole should be strong
keto_strong  = strong["ketoconazole_3A4"]
itra_strong  = strong["itraconazole_3A4"]
quin_strong  = strong["quinidine_2D6"]

# Fluconazole is moderate on 2C9 at 10 μM
fluc_moderate_or_strong = rauc["fluconazole_2C9"] >= AUC_MODERATE_DDI

# ΔΔM ordering: itraconazole has lowest Ki → highest ΔΔM
ddm_itra_gt_keto = ddm["itraconazole_3A4"] > ddm["ketoconazole_3A4"]

# At least 3 strong inhibitors
ge3_strong = n_strong >= 3

# ΔΔM for ketoconazole > 1.0 (strong shift)
keto_ddm_gt1 = ddm["ketoconazole_3A4"] > 1.0

checks = {
    "keto_strong":           keto_strong,
    "itra_strong":           itra_strong,
    "quin_strong":           quin_strong,
    "fluc_moderate_or_strong": fluc_moderate_or_strong,
    "ddm_itra_gt_keto":      ddm_itra_gt_keto,
    "ge3_strong":            ge3_strong,
    "keto_ddm_gt1":          keto_ddm_gt1,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  05_inhibitor_ranking")
sys.exit(0 if all_pass else 1)
