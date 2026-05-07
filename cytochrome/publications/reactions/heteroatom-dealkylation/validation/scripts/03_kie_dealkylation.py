"""Script 03 -- KIE for N-dealkylation vs aliphatic HAT.

Validates:
- KIE for N-dealkylation predicted from alpha-C-H stretch frequency
- KIE_N < KIE_aliphatic (softer alpha-C-H -> smaller ZPE difference)
- KIE_N in range 3.5-8.0 (literature: ~3-6 for N-demethylation)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_kie_dealkylation"

# Alpha-C-H near nitrogen: softer frequency due to N lone pair donation
nu_n_ch = NU_N_CH_CM1       # 2800 cm^-1
nu_n_cd = nu_n_ch / math.sqrt(2)   # 1980 cm^-1

nu_ali_ch = NU_ALIPHATIC_CM1  # 3000 cm^-1
nu_ali_cd = nu_ali_ch / math.sqrt(2)  # 2121 cm^-1

def zpe_j(nu_cm1):
    """Zero point energy in J for stretching mode."""
    return h_J * c_cms * nu_cm1 / 2.0

# ZPE differences
delta_zpe_n = zpe_j(nu_n_ch) - zpe_j(nu_n_cd)
delta_zpe_ali = zpe_j(nu_ali_ch) - zpe_j(nu_ali_cd)

# Classical KIE (ZPE contribution only, no tunneling for N-dealk)
kie_n_zpe = math.exp(delta_zpe_n / kBT)
kie_ali_zpe = math.exp(delta_zpe_ali / kBT)

# N-dealkylation: no tunneling correction (SET contribution reduces apparent tunneling)
kie_n = kie_n_zpe

# Aliphatic: with tunneling correction from Paper 6
kie_ali = kie_ali_zpe * 1.16    # ~7.7 * 1.16... but Paper 6 gave 7.2
# Use Paper 6 value directly
kie_aliphatic_p6 = 7.2

data = {
    "nu_n_ch_cm1": nu_n_ch,
    "nu_n_cd_cm1": round(nu_n_cd, 1),
    "delta_zpe_n_J": round(delta_zpe_n, 4),
    "delta_zpe_n_over_kBT": round(delta_zpe_n / kBT, 4),
    "kie_n_zpe": round(kie_n_zpe, 3),
    "kie_n_dealk": round(kie_n, 3),
    "kie_aliphatic_ref": kie_aliphatic_p6,
    "kie_n_lt_kie_ali": kie_n < kie_aliphatic_p6,
}

checks = {
    "kie_n_in_range_3p5_to_8": 3.5 <= kie_n <= 8.0,
    "kie_n_lt_kie_aliphatic": kie_n < kie_aliphatic_p6,
    "kie_n_gt_1": kie_n > 1.0,
    "softer_frequency_reduces_zpe": delta_zpe_n < delta_zpe_ali,
    "kie_n_zpe_correct_sign": kie_n_zpe > 1.0,
}

write_result(name, data, checks)
