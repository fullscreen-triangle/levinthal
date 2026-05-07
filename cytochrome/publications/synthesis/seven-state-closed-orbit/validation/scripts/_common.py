"""Shared constants and utilities for Paper 12 validation."""
from __future__ import annotations
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

kB = 1.380649e-23    # J/K
hbar = 1.054572e-34  # J*s
NA = 6.022141e23
T = 310.0
kBT = kB * T
T_PART = 65.0        # kJ/mol per depth unit
nu_floor = 1.0e10    # s^-1
ln2 = math.log(2.0)
R = 8.314            # J/(mol*K)

# Seven-state cycle ΔM values (Paper 12 simplified model)
# These are the state-to-state categorical depths used in the closed-orbit framework.
# Note: The CPR FMN->heme ET rate (5e6 s^-1) is characterized in Paper 11.
# In the closed-orbit synthesis (Paper 12), the ET steps use the effective
# categorical depths for the full CPR->P450 electron delivery event,
# not just the FMN->heme tunneling barrier.
DM_STEPS = {
    "1_to_2_substrate_binding":   0.92,   # Paper 3
    "2_to_3_first_electron":      0.68,   # effective ET depth for full CPR->P450 delivery
    "3_to_4_O2_binding":          0.55,
    "4_to_5_second_electron":     0.72,   # second ET event
    "5_to_Cpd0_protonation":      0.45,
    "Cpd0_to_CpdI_heterolysis":   0.693,  # ln(2), from Paper 5
    "CpdI_HAT_activation":        0.65,   # from Paper 6
    "product_release":            0.30,
}

DM_LIST = list(DM_STEPS.values())
DM_SUM = sum(DM_LIST)

# Rate constants for each step: k_i = nu_floor * exp(-DM_i)
K_STEPS = {name: nu_floor * math.exp(-dm) for name, dm in DM_STEPS.items()}

# For rate hierarchy comparison (from Paper 11): FMN->heme ET
K_FMN_HEME_PAPER11 = 5.0e6   # s^-1 (Paper 11 slow tunneling step)
DM_FMN_HEME = math.log(nu_floor / K_FMN_HEME_PAPER11)  # ~7.60


def write_result(name, data, checks):
    passed = all(checks.values())
    out = {"script": name, "verdict": "PASS" if passed else "FAIL", "checks": checks, **data}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(out, indent=2))
    verdict = "PASS" if passed else "FAIL"
    print(f"{name}: {verdict}")
    for k, v in checks.items():
        mark = "OK" if v else "XX"
        print(f"  {mark} {k}")
    return out
