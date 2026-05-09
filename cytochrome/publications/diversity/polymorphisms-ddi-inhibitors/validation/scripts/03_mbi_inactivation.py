"""Script 03 — Mechanism-based inactivation (MBI): Kitz-Wilson kinetics."""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

# MBI: E + I ⇌ E·I → E* (irreversible)
# kobs = kinact × [I] / (KI + [I])
# Loss: dE/dt = -(kobs + kdeg) × E → E(t) = E0 × exp(-(kobs+kdeg)×t)
# Time to 50% activity: t_half = ln2 / (kobs + kdeg)

# ── Clarithromycin on CYP3A4 ────────────────────────────────────────────
conc_clari    = 4.0   # μM (typical clinical Cmax unbound)
kobs_clari    = kobs_mbi(conc_clari, KI_CLARITHROMYCIN, KINACT_CLARITHROMYCIN)
t_half_clari  = time_to_half_mbi(KI_CLARITHROMYCIN, KINACT_CLARITHROMYCIN, KDEG_3A4, conc_clari)

# ΔΔM from MBI (time-averaged after 1h ≈ 60 min)
t_exposure = 60  # min
frac_active_60 = math.exp(-(kobs_clari + KDEG_3A4) * t_exposure)
ddm_clari_60 = -math.log(frac_active_60)  # = (kobs+kdeg)×t

clari_kobs_gt_kdeg    = kobs_clari > KDEG_3A4
clari_thalf_lt60min   = t_half_clari < 60   # meaningful inactivation
clari_frac_lt_50pct   = frac_active_60 < 0.5

# ── Diltiazem on CYP3A4 ─────────────────────────────────────────────────
conc_dilti   = 0.5   # μM unbound clinical Cmax
kobs_dilti   = kobs_mbi(conc_dilti, KI_DILTIAZEM, KINACT_DILTIAZEM)
t_half_dilti = time_to_half_mbi(KI_DILTIAZEM, KINACT_DILTIAZEM, KDEG_3A4, conc_dilti)
frac_dilti   = math.exp(-(kobs_dilti + KDEG_3A4) * 60)

dilti_thalf_gt_clari = t_half_dilti > t_half_clari   # diltiazem slower

# ── Erythromycin on CYP3A4 ──────────────────────────────────────────────
conc_ery    = 50.0   # μM clinical
kobs_ery    = kobs_mbi(conc_ery, KI_ERYTHROMYCIN, KINACT_ERYTHROMYCIN)
ery_weaker  = kobs_ery < kobs_clari   # erythromycin weaker MBI

# ── Kitz-Wilson: kinact/KI ratio as intrinsic MBI potency ───────────────
pot_clari = KINACT_CLARITHROMYCIN / KI_CLARITHROMYCIN
pot_dilti = KINACT_DILTIAZEM      / KI_DILTIAZEM
pot_ery   = KINACT_ERYTHROMYCIN   / KI_ERYTHROMYCIN
clari_gt_dilti_potency = pot_clari > pot_dilti

# ── Report ───────────────────────────────────────────────────────────────
checks = {
    "clari_kobs_gt_kdeg":    clari_kobs_gt_kdeg,
    "clari_thalf_lt60min":   clari_thalf_lt60min,
    "clari_frac_lt_50pct":   clari_frac_lt_50pct,
    "dilti_thalf_gt_clari":  dilti_thalf_gt_clari,
    "ery_weaker":            ery_weaker,
    "clari_gt_dilti_potency": clari_gt_dilti_potency,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'XX'}  {name}")

all_pass = all(checks.values())
print(f"\n{'PASS' if all_pass else 'FAIL'}  03_mbi_inactivation")
sys.exit(0 if all_pass else 1)
