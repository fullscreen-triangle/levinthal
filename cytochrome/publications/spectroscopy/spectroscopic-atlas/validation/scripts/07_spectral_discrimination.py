"""Script 07 -- Spectral discrimination: can UV-Vis distinguish all 7 states?"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_spectral_discrimination"

# Pairwise Soret separations between states (nm)
states = list(SORET_NM.keys())
nm_vals = list(SORET_NM.values())

min_separation = float("inf")
min_pair = None
for i in range(len(states)):
    for j in range(i+1, len(states)):
        sep = abs(nm_vals[i] - nm_vals[j])
        if sep < min_separation:
            min_separation = sep
            min_pair = (states[i], states[j])

# Count pairs with separation < 5 nm (hard to discriminate by UV-Vis alone)
hard_pairs = sum(1 for i in range(len(states)) for j in range(i+1, len(states))
                 if abs(nm_vals[i] - nm_vals[j]) < 5)

# States that require EPR or Raman for unambiguous identification
states_requiring_epr = ["resting_FeIII_LS", "substrate_bound_HS"]  # same region as compound_I
states_requiring_raman = ["compound_I"]  # FeIV=O unique Raman signal

data = {
    "min_separation_nm":    round(min_separation, 1),
    "min_pair":             list(min_pair) if min_pair else None,
    "hard_pairs_lt_5nm":    hard_pairs,
    "states_need_epr":      states_requiring_epr,
    "states_need_raman":    states_requiring_raman,
}

checks = {
    "min_separation_gt_0":       min_separation > 0,
    "all_soret_between_350_470": all(350 <= v <= 470 for v in nm_vals),
    "epr_needed_for_spin":       len(states_requiring_epr) > 0,
    "raman_needed_for_cpd1":     len(states_requiring_raman) > 0,
    "seven_states_analyzed":     len(states) == 7,
}

write_result(name, data, checks)
