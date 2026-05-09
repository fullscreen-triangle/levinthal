"""Shared constants for Paper 15 — Polymorphisms, DDI, and Inhibitors."""
import math

NU_FLOOR = 1e10   # s⁻¹  (categorical floor rate)
T_PART   = 65.0   # kJ/mol per ΔM unit
RT       = 8.314e-3 * 298  # kJ/mol at 298 K

# ── Allele ΔM values ────────────────────────────────────────────────────
DELTA_M_2D6_EM   = 0.55   # CYP2D6*1  wild-type EM
DELTA_M_2D6_IM   = 0.75   # CYP2D6*10/*17  intermediate metaboliser
DELTA_M_2D6_PM   = 2.50   # CYP2D6*4/*5   poor metaboliser (null)
DELTA_M_2D6_UM   = 0.27   # gene duplication  ultra-rapid

DELTA_M_2C9_WT   = 0.48   # CYP2C9*1  wild-type
DELTA_M_2C9_S3   = 3.60   # CYP2C9*3  I359L  (<5 % activity)
DELTA_M_2C9_S2   = 1.20   # CYP2C9*2  R144C  (~30 % activity)

DELTA_M_3A4_WT   = 0.55   # CYP3A4*1  wild-type
# CYP3A4*22 reduces expression ~50 %; ΔM_eff = ΔM_WT + ln(2)
DELTA_M_3A4_22   = DELTA_M_3A4_WT + math.log(2)

# ── Competitive inhibitor Ki (μM) ────────────────────────────────────────
KI_KETOCONAZOLE_3A4  = 0.037   # potent CYP3A4 inhibitor
KI_ITRACONAZOLE_3A4  = 0.013   # very potent
KI_QUINIDINE_2D6     = 0.027   # potent CYP2D6
KI_FLUOXETINE_2D6    = 0.24    # moderate CYP2D6
KI_PAROXETINE_2D6    = 0.15    # CYP2D6 (also MBI)
KI_FLUCONAZOLE_2C9   = 7.0     # moderate CYP2C9
KI_FLUCONAZOLE_2C19  = 0.40    # stronger on 2C19

# ── Mechanism-based inactivation (MBI) ──────────────────────────────────
KI_CLARITHROMYCIN    = 3.7     # μM  CYP3A4
KINACT_CLARITHROMYCIN = 0.04   # min⁻¹
KI_DILTIAZEM         = 14.0    # μM  CYP3A4
KINACT_DILTIAZEM     = 0.06    # min⁻¹
KI_ERYTHROMYCIN      = 72.0    # μM  CYP3A4
KINACT_ERYTHROMYCIN  = 0.025   # min⁻¹

# CYP3A4 constitutive degradation rate (kdeg)
KDEG_3A4 = 0.00032  # min⁻¹  (half-life ≈ 36 h)

# ── Induction fold-changes ───────────────────────────────────────────────
RIFAMPICIN_FOLD     = 20.0   # CYP3A4  strong inducer
PHENOBARBITAL_FOLD  = 10.0   # CYP3A4 / 2B6
OMEPRAZOLE_FOLD     =  5.0   # CYP1A2

# ── DDI significance thresholds (AUC ratio) ─────────────────────────────
AUC_STRONG_DDI   = 5.0    # R_AUC > 5
AUC_MODERATE_DDI = 2.0    # R_AUC 2–5
AUC_WEAK_DDI     = 1.25   # R_AUC 1.25–2


# ── Helper functions ─────────────────────────────────────────────────────
def k_rate(delta_m):
    return NU_FLOOR * math.exp(-delta_m)

def alpha(conc_uM, ki_uM):
    return 1.0 + conc_uM / ki_uM

def dm_shift_from_alpha(alpha_val):
    """Competitive inhibitor shifts apparent ΔM by ln(α)."""
    return math.log(alpha_val)

def kobs_mbi(conc_uM, ki_uM, kinact_per_min):
    return kinact_per_min * conc_uM / (ki_uM + conc_uM)

def auc_ratio_inhibition(conc_uM, ki_uM):
    return alpha(conc_uM, ki_uM)

def auc_ratio_induction(fold):
    return 1.0 / fold

def time_to_half_mbi(ki_uM, kinact_per_min, kdeg_per_min, conc_uM):
    """Time for enzyme to reach 50 % activity under MBI."""
    kobs = kobs_mbi(conc_uM, ki_uM, kinact_per_min)
    k_loss = kobs + kdeg_per_min
    return math.log(2) / k_loss  # min
