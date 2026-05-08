"""Script 04 -- Spin-state equilibrium: HS/LS ratio from ΔG_spin."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_spin_state_equilibrium"

# Spin-state equilibrium: FeIII LS <-> FeIII HS
# ΔG_spin (LS → HS) varies with substrate binding
# Substrate-free: ΔG_spin ≈ +2.0 kJ/mol (LS favored at 310K)
# Substrate-bound: ΔG_spin ≈ -4.0 kJ/mol (HS favored, drives reduction)

kB = 1.380649e-23  # J/K
NA = 6.022141e23
T  = 310.0

delta_g_sf = +2.0   # kJ/mol, substrate-free (LS favoured)
delta_g_sb = -4.0   # kJ/mol, substrate-bound (HS favoured)

K_sf = math.exp(-delta_g_sf * 1000 / (kB * NA * T))   # HS/LS ratio substrate-free
K_sb = math.exp(-delta_g_sb * 1000 / (kB * NA * T))   # HS/LS ratio substrate-bound

frac_hs_sf = K_sf / (1 + K_sf)
frac_hs_sb = K_sb / (1 + K_sb)

# ΔM proxy for spin conversion (from Paper 1 / categorical mechanics)
dm_spin_sf = abs(delta_g_sf) / T_PART
dm_spin_sb = abs(delta_g_sb) / T_PART

data = {
    "delta_g_sf_kJ":   delta_g_sf,
    "delta_g_sb_kJ":   delta_g_sb,
    "K_hs_ls_sf":      round(K_sf, 4),
    "K_hs_ls_sb":      round(K_sb, 4),
    "frac_hs_sf":      round(frac_hs_sf, 4),
    "frac_hs_sb":      round(frac_hs_sb, 4),
    "dm_spin_sf":      round(dm_spin_sf, 4),
    "dm_spin_sb":      round(dm_spin_sb, 4),
}

checks = {
    "ls_favored_substrate_free":    frac_hs_sf < 0.5,
    "hs_favored_substrate_bound":   frac_hs_sb > 0.5,
    "frac_hs_sb_gt_0.7":            frac_hs_sb > 0.7,
    "frac_hs_sf_lt_0.4":            frac_hs_sf < 0.4,
    "dm_spin_positive":             dm_spin_sf > 0 and dm_spin_sb > 0,
}

write_result(name, data, checks)
