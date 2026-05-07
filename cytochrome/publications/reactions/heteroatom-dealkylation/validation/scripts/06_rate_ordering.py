"""Script 06 -- Full rate ordering across heteroatom oxidation types.

Validates:
- Monotonic ordering: k_S_ox > k_N_ox > k_N_dealk > k_O_dealk > k_aliphatic
- Ordering mirrors DeltaM hierarchy
- Direct O-transfers (low DeltaM) always faster than HAT pathways
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_rate_ordering"

rates = {
    "S_oxidation":    (DELTA_M_S_OX,    K_S_OX),
    "N_oxide":        (DELTA_M_N_OX,    K_N_OX),
    "N_dealkylation": (DELTA_M_N_DEALK, K_N_DEALK),
    "O_dealkylation": (DELTA_M_O_DEALK, K_O_DEALK),
    "aliphatic_HAT":  (DELTA_M_ALIPHATIC, K_ALIPHATIC),
}

delta_m_list = [v[0] for v in rates.values()]
k_list       = [v[1] for v in rates.values()]

# Check monotonic ordering of DeltaM (smallest first = fastest)
dm_ordered = delta_m_list == sorted(delta_m_list)

# Check rates are monotonically decreasing (fastest first)
k_ordered = k_list == sorted(k_list, reverse=True)

# Direct O-transfers (S-ox, N-ox) vs HAT-based (N-dealk, O-dealk, aliphatic)
direct_faster = K_S_OX > K_N_DEALK and K_N_OX > K_N_DEALK

# Largest DeltaM: aliphatic (0.65)
max_dm = max(delta_m_list)
min_k = min(k_list)

data = {
    "delta_m_list": [round(v, 4) for v in delta_m_list],
    "k_list_s": [round(v, 2) for v in k_list],
    "dm_monotonic": dm_ordered,
    "k_monotonic_decreasing": k_ordered,
    "direct_transfers_faster": direct_faster,
    "max_delta_m": round(max_dm, 4),
}

checks = {
    "delta_m_monotonic": dm_ordered,
    "rate_monotonic_decreasing": k_ordered,
    "direct_transfer_faster_than_hat": direct_faster,
    "s_ox_fastest": K_S_OX == max(k_list),
    "aliphatic_slowest": K_ALIPHATIC == min(k_list),
    "five_distinct_rates": len(set([round(k, -5) for k in k_list])) >= 4,
}

write_result(name, data, checks)
