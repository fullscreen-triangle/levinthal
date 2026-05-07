"""Shared constants and utilities for Paper 8 validation."""
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
T = 310.0            # K
kBT = kB * T

T_PART = 65.0        # kJ/mol per depth unit
nu_floor = 1.0e10    # s^-1
ln2 = math.log(2.0)

# Paper 8: Atypical reaction parameters
DELTA_M_DESATURATION_1 = 0.65   # first HAT (aliphatic)
DELTA_M_DESATURATION_2 = 0.55   # second HAT (radical-stabilized, beta-carbon)
DELTA_M_EPOXIDATION     = 0.35   # direct O insertion to pi system
DELTA_M_NIH_SHIFT       = 0.18   # cationic 1,2-H migration (spontaneous)
DELTA_M_NUCLEOPHILIC    = 0.42   # nucleophilic O-atom transfer to aldehyde C=O
DELTA_M_CARBENE         = 0.20   # carbene insertion (engineered P450s)

# Rebound rate from Paper 6
K_REBOUND = 7.4e9    # s^-1

# Intrinsic rates for each mechanism
K_DESAT_1 = nu_floor * math.exp(-DELTA_M_DESATURATION_1)
K_DESAT_2 = nu_floor * math.exp(-DELTA_M_DESATURATION_2)
K_EPOX    = nu_floor * math.exp(-DELTA_M_EPOXIDATION)
K_NIH     = nu_floor * math.exp(-DELTA_M_NIH_SHIFT)
K_NUC     = nu_floor * math.exp(-DELTA_M_NUCLEOPHILIC)
K_CARBENE = nu_floor * math.exp(-DELTA_M_CARBENE)

# Effective desaturation rate (competes with rebound at radical intermediate)
K_DESAT_EFF = K_DESAT_1 * K_DESAT_2 / (K_DESAT_2 + K_REBOUND)


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
