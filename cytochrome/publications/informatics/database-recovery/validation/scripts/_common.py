"""Shared constants for Paper 14: P450 Database Recovery via Ternary Encoding."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

nu_floor = 1.0e10
T_PART   = 65.0

# Database recovery parameters
# The categorical address space can recover a complete P450 sequence from a
# partial address (e.g., 70% of positions known, 30% corrupted)

N_HUMAN_CYPS = 57
N_FAMILIES   = 18
DEPTH_FAMILY = 3
DEPTH_ISOFORM= 6
DEPTH_ALLELE = 9

# Recovery thresholds
MIN_KNOWN_FRACTION = 0.70    # 70% of address known -> full recovery
RECOVERY_DEPTH     = DEPTH_ISOFORM  # depth at which recovery is unique

# Information capacity (bits) of ternary encoding
# Each trit = log2(3) ≈ 1.585 bits
BITS_PER_TRIT = math.log2(3)

def bits_at_depth(k: int) -> float:
    return k * BITS_PER_TRIT

# Shannon entropy of the P450 sequence space
# H = sum(-p_i * log2(p_i)) over 57 isoforms
# Assume uniform distribution for a lower bound
H_uniform_57 = math.log2(57)
H_uniform_18 = math.log2(18)

# Recovery accuracy: fraction of sequence positions correctly inferred
# from partial address. Model: accuracy = 1 - exp(-k * BITS_PER_TRIT / H)
def recovery_accuracy(k: int, H: float = H_uniform_57) -> float:
    return 1.0 - math.exp(-bits_at_depth(k) / H)


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
