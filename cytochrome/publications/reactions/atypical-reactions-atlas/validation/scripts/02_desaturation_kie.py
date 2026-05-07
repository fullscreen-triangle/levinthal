"""Script 02 -- Apparent KIE for desaturation vs hydroxylation."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_desaturation_kie"

nu_ch = 3000.0    # cm^-1 aliphatic
nu_cd = nu_ch / math.sqrt(2)
zpe_h = h_J * c_cms * nu_ch / 2.0
zpe_d = h_J * c_cms * nu_cd / 2.0
KIE_single_HAT = math.exp((zpe_h - zpe_d) / kBT) * 1.16  # Paper 6 value

# Desaturation competes with rebound at the radical intermediate.
# Apparent KIE for desaturation product:
# f_desat_H / f_desat_D where f_desat = k2/(k2+k_r)
# For deuterated substrate at C1: k1_D = k1/KIE, k2 unchanged (H at C2)
# k_desat_D = (k1/KIE) * k2/(k2+k_r) -- only the first HAT isotope effect
# KIE_desat = k_desat_H / k_desat_D = KIE_single_HAT (same rate-limiting first HAT)
# BUT: intrinsic KIE is attenuated by the commitment factor
# Commitment: C = k2/(k2 + k_r) -- fraction continuing to desat
# Observed KIE_desat = (KIE_intrinsic - 1) * C + 1... Northrop equation simplified
f_desat_H = K_DESAT_2 / (K_DESAT_2 + K_REBOUND)
KIE_intrinsic = KIE_single_HAT
# Northrop simplified: KIE_obs = (KIE_int - 1)/(1 + 1/C) + 1
C_forward = K_DESAT_2 / K_DESAT_1    # forward commitment ~1.1
KIE_desat_apparent = (KIE_intrinsic - 1.0) / (1.0 + 1.0/C_forward) + 1.0

data = {
    "KIE_single_HAT": round(KIE_single_HAT, 3),
    "KIE_intrinsic": round(KIE_intrinsic, 3),
    "commitment_factor": round(C_forward, 4),
    "KIE_desat_apparent": round(KIE_desat_apparent, 3),
    "f_desat": round(f_desat_H, 4),
}

checks = {
    "KIE_desat_gt_3": KIE_desat_apparent > 3.0,
    "KIE_desat_lt_KIE_single_HAT": KIE_desat_apparent < KIE_single_HAT,
    "KIE_desat_in_range_3_to_9": 3.0 <= KIE_desat_apparent <= 9.0,
    "KIE_single_HAT_gt_7": KIE_single_HAT > 7.0,
    "commitment_positive": C_forward > 0,
}

write_result(name, data, checks)
