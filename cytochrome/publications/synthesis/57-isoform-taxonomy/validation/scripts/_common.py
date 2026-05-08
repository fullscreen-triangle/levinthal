"""Shared constants for Paper 9: 57 human CYP isoform taxonomy."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Categorical mechanics constants
nu_floor = 1.0e10   # s^-1
T_PART   = 65.0     # kJ/mol per depth unit

# 57 human CYP isoforms grouped into 18 Nelson families
N_HUMAN_CYPS  = 57
N_FAMILIES    = 18
N_SUBFAMILIES = 22  # representative count (CYP1A, 1B, 2A, 2B, 2C, 2D, 2E, 2F, 2J, 2R,
                    # 2S, 2U, 2W, 3A, 4A, 4B, 4F, 4V, 5A, 7A, 8A, 11A, 17A, 19A, 21A, 24A, 26A, 27A, 51A)

# Ternary address depth thresholds (from Paper 2)
DEPTH_FAMILY   = 3   # k=3 separates 18 families
DEPTH_ISOFORM  = 6   # k=6 separates 57 isoforms
DEPTH_ALLELE   = 9   # k=9 separates allelic variants

# Amino-acid distinctness thresholds
FAMILY_RECALL    = 0.94  # recall at k=3 family level
ISOFORM_DISTINCT = 0.97  # distinctness at k=6 isoform level

# Substrate selectivity ΔM ranges per family
DELTA_M_CYP1  = (0.50, 0.65)  # polycyclic/planar aromatics
DELTA_M_CYP2C = (0.48, 0.60)  # medium lipophilicity
DELTA_M_CYP2D = (0.52, 0.68)  # basic nitrogen substrates
DELTA_M_CYP3A = (0.40, 0.70)  # broad; largest isoform

# Capacity formula: C(n) = 2n^2 for shell n
def capacity(n: int) -> int:
    return 2 * n * n

# Trit-depth to families: 3^depth >= N_families
def min_depth_for(n_classes: int) -> int:
    d = 1
    while 3**d < n_classes:
        d += 1
    return d


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
