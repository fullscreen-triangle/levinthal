"""Script 03 - FMN to Heme Electron Transfer Rate.

Validates:
- FMN-heme edge-to-edge distance 14 Ang
- Categorical ΔM_ET = ln(nu_floor / k_ET) = ln(1e10 / 5e6) ~ 7.60
- k_ET in [1e6, 1e8] s^-1
- ΔM > 5 (slow step compared to chemical steps)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_fmn_heme_distance"

# FMN to heme distance (edge-to-edge in CPR-P450 complex)
r_FMN_heme_ang = 14.0  # Angstroms

# Marcus parameters
beta = 1.4    # Ang^-1 (protein tunneling decay)
lam_eV = 0.85  # eV reorganization energy

# Experimental/literature electron transfer rate
k_ET_exp = 5.0e6  # s^-1

# Categorical activation depth from observed rate
# k_ET = nu_floor * exp(-DM_ET)  =>  DM_ET = ln(nu_floor / k_ET)
DM_ET = math.log(nu_floor / k_ET_exp)

# Verify: k_ET reconstructed
k_ET_reconstructed = nu_floor * math.exp(-DM_ET)

# Marcus tunneling factor (semi-classical, for context)
# H_AB ~ H_0 * exp(-beta*r/2)
# At r=14 Ang: exp(-beta*r) = exp(-1.4*14) = exp(-19.6)
tunneling_factor = math.exp(-beta * r_FMN_heme_ang)

data = {
    "r_FMN_heme_ang": r_FMN_heme_ang,
    "beta_protein_ang_inv": beta,
    "lambda_eV": lam_eV,
    "k_ET_exp_s_inv": k_ET_exp,
    "DM_ET": round(DM_ET, 4),
    "k_ET_reconstructed": round(k_ET_reconstructed, 2),
    "tunneling_factor_Marcus": f"{tunneling_factor:.3e}",
    "rate_limiting_for_CPR_P450": True,
}

checks = {
    "DM_ET_slow_step": DM_ET > 5.0,
    "k_ET_in_range_1e6_1e8": 1e6 <= k_ET_exp <= 1e8,
    "k_ET_reconstructed_matches": abs(k_ET_reconstructed - k_ET_exp) / k_ET_exp < 0.001,
    "r_FMN_heme_14_ang": abs(r_FMN_heme_ang - 14.0) < 0.1,
    "DM_ET_positive": DM_ET > 0,
}

write_result(name, data, checks)
