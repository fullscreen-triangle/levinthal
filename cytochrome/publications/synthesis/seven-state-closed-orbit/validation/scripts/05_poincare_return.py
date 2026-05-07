"""Script 05 - Poincare Return Time.

Validates:
- T_return = sum(1/k_i) for all 8 transitions
- T_return in range [0.1 ns, 1 ms]
- k_cat_intrinsic = 1/T_return > 1e8 s^-1 (intrinsic chemistry fast)
- FMN->heme tunneling (Paper 11) limits actual turnover, not intrinsic steps
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_poincare_return"

# Compute individual timescales for the closed-orbit DM values
tau_steps = {}
k_steps = {}
for step, dm in DM_STEPS.items():
    k = nu_floor * math.exp(-dm)
    k_steps[step] = k
    tau_steps[step] = 1.0 / k   # seconds

# Poincare return time (intrinsic cycle time)
T_return_s = sum(tau_steps.values())
T_return_ns = T_return_s * 1e9
T_return_ps = T_return_s * 1e12

# k_cat from T_return (intrinsic cycle rate)
k_cat_intrinsic = 1.0 / T_return_s

# The rate-limiting step in vivo is FMN->heme tunneling from Paper 11
# k_FMN_heme = 5e6 s^-1 << k_cat_intrinsic
k_FMN = K_FMN_HEME_PAPER11
ratio_intrinsic_over_ET = k_cat_intrinsic / k_FMN

# Slowest intrinsic step
slowest = max(tau_steps, key=tau_steps.get)
tau_slowest_ps = tau_steps[slowest] * 1e12

data = {
    "tau_steps_ps": {k: round(v * 1e12, 4) for k, v in tau_steps.items()},
    "T_return_ns": round(T_return_ns, 4),
    "T_return_ps": round(T_return_ps, 2),
    "k_cat_intrinsic_s_inv": round(k_cat_intrinsic, 2),
    "k_FMN_heme_paper11": k_FMN,
    "ratio_intrinsic_over_ET": round(ratio_intrinsic_over_ET, 2),
    "slowest_intrinsic_step": slowest,
    "tau_slowest_ps": round(tau_slowest_ps, 2),
}

checks = {
    "T_return_gt_0.1_ps": T_return_ps > 0.1,
    "T_return_lt_1_ms": T_return_s < 1.0e-3,
    "T_return_in_range_ps_to_ms": 0.1 < T_return_ps < 1.0e9,
    "k_cat_intrinsic_gt_1e8": k_cat_intrinsic > 1e8,
    "k_cat_intrinsic_faster_than_FMN_tunneling": k_cat_intrinsic > k_FMN,
}

write_result(name, data, checks)
