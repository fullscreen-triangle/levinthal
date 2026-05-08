"""Script 03 -- Resonance Raman Fe=O stretch and 18O isotope shift."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_raman_feo_stretch"

# 16O vs 18O isotope shift for Fe=O: classical diatomic harmonic model
# nu_18O = nu_16O * sqrt(mu_16 / mu_18) where mu = reduced mass
m_Fe = 55.845   # amu
m_O16 = 15.999
m_O18 = 17.999

mu_16 = (m_Fe * m_O16) / (m_Fe + m_O16)
mu_18 = (m_Fe * m_O18) / (m_Fe + m_O18)

nu_16 = RAMAN_FEO_CM1
nu_18_pred = nu_16 * math.sqrt(mu_16 / mu_18)
shift_pred = nu_16 - nu_18_pred

# Literature shift: ~795 → ~757 cm^-1 (shift ~38 cm^-1; Schulz 1992, JACS)
nu_18_lit  = 757.0
shift_lit  = nu_16 - nu_18_lit

data = {
    "nu_16O_cm1":         round(nu_16, 1),
    "nu_18O_pred_cm1":    round(nu_18_pred, 1),
    "nu_18O_lit_cm1":     nu_18_lit,
    "shift_pred_cm1":     round(shift_pred, 1),
    "shift_lit_cm1":      shift_lit,
    "mu_16_amu":          round(mu_16, 4),
    "mu_18_amu":          round(mu_18, 4),
}

checks = {
    "nu_16O_near_795":     790 <= nu_16 <= 800,
    "shift_gt_30cm1":      shift_pred > 30,
    "shift_lt_50cm1":      shift_pred < 50,
    "nu_18O_lt_nu_16O":    nu_18_pred < nu_16,
    "pred_shift_within_10_of_lit": abs(shift_pred - shift_lit) < 10,
}

write_result(name, data, checks)
