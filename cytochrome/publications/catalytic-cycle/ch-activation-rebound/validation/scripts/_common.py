"""Shared constants and utilities for Paper 6 validation."""
from __future__ import annotations
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Physical constants
kB = 1.380649e-23    # J/K
hbar = 1.054572e-34  # J·s
c_cms = 2.997924e10  # cm/s
NA = 6.022141e23     # mol^-1
T = 310.0            # K (physiological)
kBT = kB * T         # J

# Framework parameters
T_PART = 65.0        # kJ/mol per depth unit (from biological-partition-landscape)
nu_floor = 1.0e10    # s^-1  (categorical clock floor = 10^10)
ln2 = math.log(2.0)

# C-H activation parameters (Paper 6)
DELTA_M_HAT = 0.65                          # activation depth for HAT
DELTA_M_REBOUND = 0.30                      # activation depth for oxygen rebound
E_A_HAT_KCAL = T_PART * 1e3 / (4184 * NA) * DELTA_M_HAT * 1e3  # kcal/mol
# E_A = T_PART [kJ/mol] * delta_M
E_A_HAT_KCAL = T_PART * DELTA_M_HAT / 4.184  # kcal/mol

# C-H stretch frequencies
NU_CH_CM1 = 3000.0                          # cm^-1
NU_CD_CM1 = NU_CH_CM1 / math.sqrt(2)        # cm^-1

# HAT rate (categorical floor × depth correction)
K_HAT = nu_floor * math.exp(-DELTA_M_HAT)   # s^-1
K_REBOUND = nu_floor * math.exp(-DELTA_M_REBOUND)  # s^-1

# Testosterone regioselectivity
TESTOSTERONE_POSITIONS = {
    "6beta":  {"delta_M": 0.55, "g": 1.00},
    "2beta":  {"delta_M": 0.68, "g": 0.40},
    "15beta": {"delta_M": 0.80, "g": 0.30},
    "16beta": {"delta_M": 0.62, "g": 0.50},
}

# Reaction types
REACTION_TYPES = {
    "aliphatic":   {"delta_M": 0.65, "has_kie": True},
    "benzylic":    {"delta_M": 0.50, "has_kie": True},
    "allylic":     {"delta_M": 0.45, "has_kie": True},
    "aromatic":    {"delta_M": 0.38, "has_kie": False},
    "epoxidation": {"delta_M": 0.35, "has_kie": False},
}


def write_result(name: str, data: dict, checks: dict) -> None:
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
