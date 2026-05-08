"""Script 07 -- Ethnic variation in CYP allele frequencies shifts population ΔM."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_ethnic_variation"

# Explicit phenotype frequencies per ancestry (sum to 1.0)
# East Asians have high CYP2D6*10 (IM) prevalence → high IM fraction
ancestry_freqs = {
    "European":   {"PM": 0.07, "IM": 0.15, "EM": 0.70, "UM": 0.08},
    "East_Asian": {"PM": 0.01, "IM": 0.50, "EM": 0.41, "UM": 0.08},
    "African":    {"PM": 0.02, "IM": 0.35, "EM": 0.55, "UM": 0.08},
    "South_Asian":{"PM": 0.05, "IM": 0.20, "EM": 0.67, "UM": 0.08},
}

def pop_rate(freqs):
    return (freqs["PM"] * K_PM + freqs["IM"] * K_IM +
            freqs["EM"] * K_EM + freqs["UM"] * K_UM)

pop_rates = {anc: pop_rate(f) for anc, f in ancestry_freqs.items()}
freq_sums = {anc: sum(f.values()) for anc, f in ancestry_freqs.items()}

# Europeans have most PM → highest inter-individual variation
# East Asians have most IM (*10) → lower mean population rate than Europeans
euro_pm    = ancestry_freqs["European"]["PM"]
asian_pm   = ancestry_freqs["East_Asian"]["PM"]
asian_im   = ancestry_freqs["East_Asian"]["IM"]
euro_im    = ancestry_freqs["European"]["IM"]

data = {
    "ancestry_freqs": ancestry_freqs,
    "pop_rates_s": {k: round(v, 2) for k, v in pop_rates.items()},
}

checks = {
    "euro_pm_highest":            euro_pm == max(a["PM"] for a in ancestry_freqs.values()),
    "east_asian_pm_lowest":       asian_pm == min(a["PM"] for a in ancestry_freqs.values()),
    "east_asian_im_highest":      asian_im == max(a["IM"] for a in ancestry_freqs.values()),
    "euro_pop_rate_gt_asian":     pop_rates["European"] > pop_rates["East_Asian"],
    "all_freqs_sum_to_1":         all(abs(s - 1.0) < 1e-9 for s in freq_sums.values()),
}

write_result(name, data, checks)
