"""Script 08 -- Full pharmacogenomics table: alleles, rates, clinical implications."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "08_full_pgx_table"

# All alleles parameterized
alleles = {
    "CYP2D6*1_EM":  {"dm": DELTA_M_EM,   "k": K_EM,   "function": "normal"},
    "CYP2D6*4_PM":  {"dm": DELTA_M_PM,   "k": K_PM,   "function": "none"},
    "CYP2D6*10_IM": {"dm": DELTA_M_IM,   "k": K_IM,   "function": "reduced"},
    "CYP2D6*1xN_UM":{"dm": DELTA_M_UM,   "k": K_UM,   "function": "increased"},
    "CYP2C9*1_EM":  {"dm": DELTA_M_2C9_EM, "k": K_2C9_EM, "function": "normal"},
    "CYP2C9*3_PM":  {"dm": DELTA_M_2C9_3,  "k": K_2C9_3,  "function": "none"},
}

all_rates_positive = all(a["k"] > 0 for a in alleles.values())
um_fastest = alleles["CYP2D6*1xN_UM"]["k"] > alleles["CYP2D6*1_EM"]["k"]
pm_slowest_2d6 = alleles["CYP2D6*4_PM"]["k"] < alleles["CYP2D6*10_IM"]["k"]
star3_much_slower = alleles["CYP2C9*3_PM"]["k"] < 0.05 * alleles["CYP2C9*1_EM"]["k"]

data = {
    "alleles": {k: {"dm": round(v["dm"],3), "k_s": round(v["k"],2), "function": v["function"]}
                for k, v in alleles.items()},
    "n_alleles": len(alleles),
    "all_rates_positive": all_rates_positive,
}

checks = {
    "all_rates_positive":           all_rates_positive,
    "UM_faster_than_EM":            um_fastest,
    "PM_slower_than_IM_2d6":        pm_slowest_2d6,
    "2c9_star3_lt_5pct_em":         star3_much_slower,
    "six_alleles_covered":          len(alleles) == 6,
    "frequency_sum_approx_1": abs(FREQ_PM + FREQ_IM + FREQ_EM + FREQ_UM - 1.0) < 1e-9,
}

write_result(name, data, checks)
