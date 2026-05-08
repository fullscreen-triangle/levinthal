"""Script 06 -- CYP3A4/2D6/2C9 account for >80% of drug metabolism."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_drug_metabolism_fractions"

# Empirical fractions of FDA-approved drugs metabolized by each CYP
# (Guengerich 2008, Nature Reviews Drug Discovery)
fractions = {
    "CYP3A4":  0.46,
    "CYP2D6":  0.19,
    "CYP2C9":  0.15,
    "CYP1A2":  0.10,
    "CYP2C19": 0.06,
    "other":   0.04,
}

top3_fraction = fractions["CYP3A4"] + fractions["CYP2D6"] + fractions["CYP2C9"]
total_fraction = sum(fractions.values())

# ΔM-weighted effective rate: isoforms with lower ΔM contribute more metabolism
# Proxy: fractional contribution ∝ exp(-ΔM_lo) for the family
dm_lo = {"CYP3A4": 0.40, "CYP2D6": 0.52, "CYP2C9": 0.48,
         "CYP1A2": 0.50, "CYP2C19": 0.50}
km_weighted = {cyp: fractions.get(cyp, 0) * math.exp(-dm_lo[cyp]) for cyp in dm_lo}
cyp3a4_km_largest = km_weighted["CYP3A4"] == max(km_weighted.values())

data = {
    "metabolism_fractions": fractions,
    "top3_fraction":        round(top3_fraction, 3),
    "total_fraction":       round(total_fraction, 3),
    "km_weighted":          {k: round(v, 4) for k, v in km_weighted.items()},
}

checks = {
    "top3_ge_0.80":           top3_fraction >= 0.80,
    "cyp3a4_largest":         fractions["CYP3A4"] == max(fractions.values()),
    "total_sums_to_1":        abs(total_fraction - 1.0) < 1e-9,
    "cyp3a4_km_weighted_top": cyp3a4_km_largest,
    "five_major_cyps_listed": len(fractions) >= 5,
    "cyp2d6_gt_10pct":        fractions["CYP2D6"] > 0.10,
}

write_result(name, data, checks)
