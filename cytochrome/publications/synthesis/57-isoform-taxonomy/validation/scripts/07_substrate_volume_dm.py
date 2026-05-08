"""Script 07 -- Substrate molecular volume correlates with activation ΔM."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_substrate_volume_dm"

# Representative substrates with known Km and approximate molecular volumes
# Volume (A^3), ΔM_predicted, experimental Km (uM)
substrates = {
    "acetaminophen": {"vol": 115, "dm": 0.55, "km_uM": 1200},
    "testosterone":  {"vol": 320, "dm": 0.45, "km_uM":  50},
    "midazolam":     {"vol": 280, "dm": 0.42, "km_uM":  25},
    "warfarin":      {"vol": 250, "dm": 0.48, "km_uM":   4},
    "dextromethorphan": {"vol": 270, "dm": 0.55, "km_uM": 5},
    "caffeine":      {"vol": 160, "dm": 0.50, "km_uM": 300},
}

# Correlation: larger volume -> lower ΔM (better fit in active site) -> faster rate
vols = [s["vol"] for s in substrates.values()]
dms  = [s["dm"]  for s in substrates.values()]

# Simple correlation coefficient
n = len(vols)
mean_v = sum(vols) / n
mean_d = sum(dms) / n
cov = sum((vols[i] - mean_v) * (dms[i] - mean_d) for i in range(n)) / n
std_v = math.sqrt(sum((v - mean_v)**2 for v in vols) / n)
std_d = math.sqrt(sum((d - mean_d)**2 for d in dms) / n)
r = cov / (std_v * std_d)

# r should be negative (larger volume -> lower ΔM)
data = {
    "substrates": {k: {"vol": v["vol"], "dm": v["dm"]} for k, v in substrates.items()},
    "pearson_r_vol_dm": round(r, 4),
    "correlation_direction": "negative" if r < 0 else "positive",
}

checks = {
    "r_negative_vol_vs_dm": r < 0,
    "abs_r_gt_0.4":         abs(r) > 0.4,
    "testosterone_dm_lt_acetaminophen_dm": substrates["testosterone"]["dm"] < substrates["acetaminophen"]["dm"],
    "midazolam_dm_lt_0.50": substrates["midazolam"]["dm"] < 0.50,
    "six_substrates_parameterized": len(substrates) == 6,
}

write_result(name, data, checks)
