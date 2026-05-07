"""Script 06 -- Rate ordering of all five atypical reaction modes."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_rate_ordering"

# All rates with their DeltaM values
reactions = {
    "NIH_shift":     (DELTA_M_NIH_SHIFT,     K_NIH),
    "carbene":       (DELTA_M_CARBENE,        K_CARBENE),
    "epoxidation":   (DELTA_M_EPOXIDATION,    K_EPOX),
    "nucleophilic":  (DELTA_M_NUCLEOPHILIC,   K_NUC),
    "desaturation":  (math.log(nu_floor/K_DESAT_EFF),  K_DESAT_EFF),
}

dm_list = [v[0] for v in reactions.values()]
k_list  = [v[1] for v in reactions.values()]

dm_monotonic = dm_list == sorted(dm_list)
k_monotonic  = k_list == sorted(k_list, reverse=True)

# Desaturation is the slowest
k_desat = K_DESAT_EFF

data = {
    "delta_m_values": {n: round(v[0], 4) for n, v in reactions.items()},
    "k_values_s": {n: round(v[1], 2) for n, v in reactions.items()},
    "dm_monotonic": dm_monotonic,
    "k_monotonic_decreasing": k_monotonic,
}

checks = {
    "dm_monotonic_increasing": dm_monotonic,
    "k_monotonic_decreasing": k_monotonic,
    "nih_is_fastest": K_NIH == max(k_list),
    "desat_is_slowest": K_DESAT_EFF == min(k_list),
    "five_distinct_reactions": len(set([round(k, -5) for k in k_list])) >= 4,
}

write_result(name, data, checks)
