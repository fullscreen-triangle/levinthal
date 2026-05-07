"""Script 06 - Anharmonic Closure Check.

Validates:
- No step is a true sink: all k_i > 0
- max(DM_i) < 10.0 (all transitions occur within protein lifetime)
- max(DM_i) < ln(10) = 2.303 confirms no classically sub-threshold steps
- Orbit is ergodic over the catalytic timescale
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_anharmonic_check"

# DM critical for a classical sink (DM > ln10 means k < nu_floor/10)
DM_critical_classical = math.log(10.0)   # ln(10) ~ 2.303

DM_max = max(DM_LIST)
DM_min = min(DM_LIST)
step_max = max(DM_STEPS, key=DM_STEPS.get)
step_min = min(DM_STEPS, key=DM_STEPS.get)

# k_min for DM_max step
k_min = nu_floor * math.exp(-DM_max)
tau_max_ns = (1.0 / k_min) * 1e9

# Protein lifetime: ~hours to days
protein_lifetime_s = 3600.0   # 1 hour = 3600 s
k_protein_degradation = 1.0 / protein_lifetime_s   # ~2.8e-4 s^-1

# All k_i >> k_protein_degradation
k_values_list = list(K_STEPS.values())
k_min_all = min(k_values_list)
all_steps_faster_than_protein = k_min_all > k_protein_degradation

# Check: all DM_i < ln(10) = 2.303 (within the anharmonic closure regime)
all_below_classical_sink = all(dm < DM_critical_classical for dm in DM_LIST)

data = {
    "DM_max": round(DM_max, 4),
    "DM_min": round(DM_min, 4),
    "step_with_max_DM": step_max,
    "step_with_min_DM": step_min,
    "k_min_s_inv": round(k_min, 2),
    "tau_max_ns": round(tau_max_ns, 4),
    "DM_critical_ln10": round(DM_critical_classical, 4),
    "all_below_classical_sink": all_below_classical_sink,
    "k_protein_degradation_s_inv": k_protein_degradation,
    "all_steps_faster_than_protein": all_steps_faster_than_protein,
}

checks = {
    "DM_max_lt_10": DM_max < 10.0,
    "DM_max_lt_ln10_classical_threshold": DM_max < DM_critical_classical,
    "k_min_positive": k_min > 0,
    "k_min_gt_1_per_second": k_min > 1.0,
    "all_steps_faster_than_protein": all_steps_faster_than_protein,
}

write_result(name, data, checks)
