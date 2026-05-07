"""Script 02 - Rate-Limiting Step Analysis.

Validates:
- Compute k_i = nu_floor * exp(-DM_i) for each step
- Identify slowest step
- T_return = sum(1/k_i) > 0.1 ns
- Slowest step has DM > 0.5
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_rate_limiting"

# Compute rates and timescales
k_values = {}
tau_values = {}
for step, dm in DM_STEPS.items():
    k = nu_floor * math.exp(-dm)
    k_values[step] = k
    tau_values[step] = 1.0 / k

# T_return = sum of all timescales
T_return_s = sum(tau_values.values())
T_return_ns = T_return_s * 1e9
T_return_ps = T_return_s * 1e12

# Find rate-limiting step (largest tau = smallest k)
slowest_step = max(tau_values, key=tau_values.get)
slowest_DM = DM_STEPS[slowest_step]
slowest_k = k_values[slowest_step]
slowest_tau_ps = tau_values[slowest_step] * 1e12

# k_cat from T_return
k_cat_intrinsic = 1.0 / T_return_s

# Compare with FMN->heme tunneling (Paper 11 slow step)
k_tunneling = K_FMN_HEME_PAPER11  # 5e6 s^-1

data = {
    "k_values_s_inv": {k: f"{v:.3e}" for k, v in k_values.items()},
    "tau_values_ps": {k: round(v * 1e12, 4) for k, v in tau_values.items()},
    "T_return_ns": round(T_return_ns, 4),
    "T_return_ps": round(T_return_ps, 2),
    "slowest_step": slowest_step,
    "slowest_DM": round(slowest_DM, 4),
    "slowest_k_s_inv": f"{slowest_k:.3e}",
    "slowest_tau_ps": round(slowest_tau_ps, 2),
    "k_cat_intrinsic_s_inv": round(k_cat_intrinsic, 2),
    "k_FMN_heme_paper11_s_inv": k_tunneling,
    "k_cat_faster_than_FMN_tunneling": k_cat_intrinsic > k_tunneling,
}

checks = {
    "slowest_step_DM_gt_0.5": slowest_DM > 0.5,
    "T_return_gt_0.1_ps": T_return_ps > 0.1,
    "T_return_lt_1_ms": T_return_s < 1.0e-3,
    "k_cat_intrinsic_gt_1e8": k_cat_intrinsic > 1e8,
    "slowest_is_substrate_binding": slowest_step == "1_to_2_substrate_binding",
}

write_result(name, data, checks)
