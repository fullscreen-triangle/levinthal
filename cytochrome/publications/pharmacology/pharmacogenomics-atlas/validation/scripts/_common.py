"""Shared constants for Paper 10: Pharmacogenomics atlas."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

nu_floor = 1.0e10   # s^-1
T_PART   = 65.0     # kJ/mol per depth unit
DEPTH_ALLELE = 9    # ternary depth separating allelic variants

# CYP2D6 allele pharmacogenomic parameters
# Poor Metabolizer (PM): loss-of-function mutations -> higher ΔM (slower)
# Extensive Metabolizer (EM): wild-type
# Ultrarapid Metabolizer (UM): gene duplication -> lower ΔM (faster)
DELTA_M_EM   = 0.55   # extensive metabolizer (wild-type CYP2D6*1)
DELTA_M_PM   = 2.50   # poor metabolizer (*4, *5: null alleles; ~8% residual rate)
DELTA_M_IM   = 0.75   # intermediate metabolizer (*10, *17)
DELTA_M_UM   = 0.27   # ultrarapid metabolizer (*1xN: ~2x gene copies -> lower ΔM)

K_EM   = nu_floor * math.exp(-DELTA_M_EM)
K_PM   = nu_floor * math.exp(-DELTA_M_PM)
K_IM   = nu_floor * math.exp(-DELTA_M_IM)
K_UM   = nu_floor * math.exp(-DELTA_M_UM)

# CYP2C9 alleles (*2, *3 reduced function)
DELTA_M_2C9_EM = 0.48
DELTA_M_2C9_2  = 0.62   # *2: R144C, ~30% activity reduction
DELTA_M_2C9_3  = 3.60   # *3: I359L, ~5% residual activity (K_star3 < 5% of K_EM)

K_2C9_EM = nu_floor * math.exp(-DELTA_M_2C9_EM)
K_2C9_2  = nu_floor * math.exp(-DELTA_M_2C9_2)
K_2C9_3  = nu_floor * math.exp(-DELTA_M_2C9_3)

# Population frequencies (approximate, European ancestry)
FREQ_PM  = 0.07   # 7% PM for CYP2D6 in Europeans
FREQ_IM  = 0.30
FREQ_EM  = 0.55
FREQ_UM  = 0.08


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
