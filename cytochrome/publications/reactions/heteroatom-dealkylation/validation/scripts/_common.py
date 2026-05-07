"""Shared constants and utilities for Paper 7 validation."""
from __future__ import annotations
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

kB = 1.380649e-23    # J/K
hbar = 1.054572e-34  # J*s
c_cms = 2.997924e10  # cm/s
h_J = 6.62607e-34    # J*s
NA = 6.022141e23
T = 310.0            # K (physiological)
kBT = kB * T

T_PART = 65.0        # kJ/mol per depth unit
nu_floor = 1.0e10    # s^-1
ln2 = math.log(2.0)

# Heteroatom oxidation parameters (Paper 7)
# alpha-C-H bond dissociation energies (kcal/mol)
BDE_N_CH3 = 87.0    # alpha-C to N-methyl (weakened by N lone pair)
BDE_O_CH3 = 92.0    # alpha-C to O-methyl
BDE_aliphatic = 100.0  # unactivated C-H

# Activation depths from BDE scaling
# Reference: aliphatic BDE=100 -> DeltaM=0.65; scale proportionally
DELTA_M_N_DEALK = 0.50    # faster due to weaker alpha-C-H
DELTA_M_O_DEALK = 0.58    # intermediate
DELTA_M_ALIPHATIC = 0.65  # reference aliphatic

# Direct O-atom transfer (no HAT step)
DELTA_M_S_OX = 0.28     # sulfoxidation
DELTA_M_N_OX = 0.32     # N-oxide formation

# C-H stretch frequencies for alpha-C near heteroatom
NU_N_CH_CM1 = 2800.0    # softer than aliphatic 3000 due to N induction
NU_ALIPHATIC_CM1 = 3000.0

# Rates
K_N_DEALK = nu_floor * math.exp(-DELTA_M_N_DEALK)
K_O_DEALK = nu_floor * math.exp(-DELTA_M_O_DEALK)
K_S_OX = nu_floor * math.exp(-DELTA_M_S_OX)
K_N_OX = nu_floor * math.exp(-DELTA_M_N_OX)
K_ALIPHATIC = nu_floor * math.exp(-DELTA_M_ALIPHATIC)


def write_result(name: str, data: dict, checks: dict) -> dict:
    passed = all(checks.values())
    out = {
        "script": name,
        "verdict": "PASS" if passed else "FAIL",
        "checks": checks,
        **data,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(out, indent=2))
    verdict = "PASS" if passed else "FAIL"
    print(f"{name}: {verdict}")
    for k, v in checks.items():
        mark = "OK" if v else "XX"
        print(f"  {mark} {k}")
    return out
