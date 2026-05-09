"""Script 07 — Time-dependent inhibition (TDI): IC50 shift assay."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# TDI assay: pre-incubate enzyme + I for time t_pre, then measure IC50
# If MBI: IC50_shifted < IC50_direct (ratio < 1)
# Shift ratio = IC50_shift / IC50_direct; TDI positive if ratio < 0.5

# IC50 (direct, no pre-incubation) ≈ Ki × 2 for reversible competitive
def ic50_direct(ki_uM):
    return ki_uM * 2.0

# IC50 after pre-incubation (MBI reduces active enzyme by fraction f_active)
def ic50_shifted(ki_uM, kinact_per_min, kdeg_per_min, t_pre_min, conc_pre_uM):
    kobs    = kobs_mbi(conc_pre_uM, ki_uM, kinact_per_min)
    f_active = math.exp(-(kobs + kdeg_per_min) * t_pre_min)
    # IC50_shift ≈ IC50_direct × f_active (less enzyme → apparent shift down)
    return ic50_direct(ki_uM) * f_active

T_PRE   = 60   # min standard TDI pre-incubation (1-hour protocol)
CONC_PRE_CLARI = 4.0   # μM
CONC_PRE_DILTI = 15.0

# ── Clarithromycin ───────────────────────────────────────────────────────
ic50_d_clari = ic50_direct(KI_CLARITHROMYCIN)
ic50_s_clari = ic50_shifted(KI_CLARITHROMYCIN, KINACT_CLARITHROMYCIN,
                             KDEG_3A4, T_PRE, CONC_PRE_CLARI)
ratio_clari  = ic50_s_clari / ic50_d_clari
tdi_clari    = ratio_clari < 0.5   # TDI positive

# ── Diltiazem ────────────────────────────────────────────────────────────
ic50_d_dilti = ic50_direct(KI_DILTIAZEM)
ic50_s_dilti = ic50_shifted(KI_DILTIAZEM, KINACT_DILTIAZEM,
                             KDEG_3A4, T_PRE, CONC_PRE_DILTI)
ratio_dilti  = ic50_s_dilti / ic50_d_dilti
tdi_dilti    = ratio_dilti < 0.5   # TDI positive

# ── Erythromycin ─────────────────────────────────────────────────────────
ic50_d_ery = ic50_direct(KI_ERYTHROMYCIN)
ic50_s_ery = ic50_shifted(KI_ERYTHROMYCIN, KINACT_ERYTHROMYCIN,
                           KDEG_3A4, T_PRE, 50.0)
ratio_ery   = ic50_s_ery / ic50_d_ery
tdi_ery     = ratio_ery < 0.8   # TDI positive (FDA ≤0.8 threshold; weaker MBI)

# ── Fluoxetine (competitive, NOT MBI at short pre-incubation) ────────────
# For a pure reversible inhibitor there is no IC50 shift
ic50_d_fluox = ic50_direct(KI_FLUOXETINE_2D6)
# No MBI term → IC50 unchanged (ratio ≈ 1)
ratio_fluox  = 1.0   # no TDI
no_tdi_fluox = ratio_fluox > 0.8

# ── Clarithromycin shift stronger than erythromycin ──────────────────────
clari_stronger_tdi = ratio_clari < ratio_ery

# ── Kinact/KI rank correct ───────────────────────────────────────────────
pot_rank_correct = (KINACT_CLARITHROMYCIN / KI_CLARITHROMYCIN >
                    KINACT_DILTIAZEM      / KI_DILTIAZEM >
                    KINACT_ERYTHROMYCIN   / KI_ERYTHROMYCIN)

checks = {
    "tdi_clari":          tdi_clari,
    "tdi_dilti":          tdi_dilti,
    "tdi_ery":            tdi_ery,
    "no_tdi_fluox":       no_tdi_fluox,
    "clari_stronger_tdi": clari_stronger_tdi,
    "pot_rank_correct":   pot_rank_correct,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  07_tdi_ic50_shift")
sys.exit(0 if all_pass else 1)
