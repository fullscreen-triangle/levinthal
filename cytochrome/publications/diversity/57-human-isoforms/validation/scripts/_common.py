"""Shared constants and utilities for Paper 9 validation."""
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
NA = 6.022141e23
T = 310.0            # K
kBT = kB * T
T_PART = 65.0        # kJ/mol per depth unit
nu_floor = 1.0e10    # s^-1
ln2 = math.log(2.0)

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
