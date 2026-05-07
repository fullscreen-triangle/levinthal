"""Shared constants and utilities for Paper 11 validation."""
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

# Paper 11 constants
DM_TM_INSERT = 0.42            # TM helix insertion depth
DG_INSERT_KCAL = -10.0         # kcal/mol

KD_CPR = 1.0e-7                # M (0.1 uM)
KD_B5 = 0.5e-7                 # M (0.05 uM)

K_FMN_HEME = 5.0e6             # s^-1 (FMN->heme ET rate)
K_B5_HEME = 3.0e7              # s^-1 (cytb5->heme ET rate)

DM_FMN_HEME = math.log(nu_floor / K_FMN_HEME)  # ln(1e10/5e6) ~ 7.60

MEMBRANE_ENRICHMENT_LOGP3 = 10.0  # 10^(3-2) = 10


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
