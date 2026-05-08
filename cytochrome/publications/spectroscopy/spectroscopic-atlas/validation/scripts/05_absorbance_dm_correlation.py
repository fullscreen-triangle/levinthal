"""Script 05 -- Soret peak energy correlates with ΔM across the 7 states."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_absorbance_dm_correlation"

# Compute Pearson r between Soret photon energy (cm^-1) and ΔM_spec
soret_nm_vals = list(SORET_NM.values())
soret_cm1 = [1e7 / lam for lam in soret_nm_vals]   # wavenumber in cm^-1
dm_vals   = list(SORET_DM.values())

n = len(soret_cm1)
mean_e = sum(soret_cm1) / n
mean_d = sum(dm_vals) / n
cov  = sum((soret_cm1[i]-mean_e)*(dm_vals[i]-mean_d) for i in range(n)) / n
std_e = math.sqrt(sum((e-mean_e)**2 for e in soret_cm1) / n)
std_d = math.sqrt(sum((d-mean_d)**2 for d in dm_vals) / n)
r = cov / (std_e * std_d)

# All Soret DM values should be in range 4-5 (high energy photons)
dm_min = min(dm_vals)
dm_max = max(dm_vals)

data = {
    "soret_cm1":   {k: round(1e7/v, 1) for k, v in SORET_NM.items()},
    "soret_dm":    {k: round(v, 4) for k, v in SORET_DM.items()},
    "pearson_r":   round(r, 4),
    "dm_range":    [round(dm_min, 4), round(dm_max, 4)],
}

checks = {
    "r_positive":               r > 0,
    "abs_r_gt_0.9":             abs(r) > 0.9,
    "all_dm_between_3_and_6":   all(3.0 < d < 6.0 for d in dm_vals),
    "hs_dm_gt_ls_dm":           SORET_DM["substrate_bound_HS"] > SORET_DM["resting_FeIII_LS"],
    "seven_states_in_correlation": len(soret_cm1) == 7,
}

write_result(name, data, checks)
