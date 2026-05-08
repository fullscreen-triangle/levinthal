"""Script 03 -- Population frequency-weighted metabolic rate for CYP2D6."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_population_phenotype_frequencies"

# Population-weighted effective rate
k_pop = FREQ_PM * K_PM + FREQ_IM * K_IM + FREQ_EM * K_EM + FREQ_UM * K_UM
total_freq = FREQ_PM + FREQ_IM + FREQ_EM + FREQ_UM

# EM dominates the population rate
k_em_contribution = FREQ_EM * K_EM
em_fraction_of_pop_rate = k_em_contribution / k_pop

data = {
    "freq_PM": FREQ_PM,
    "freq_IM": FREQ_IM,
    "freq_EM": FREQ_EM,
    "freq_UM": FREQ_UM,
    "total_freq": round(total_freq, 4),
    "k_pop_s": round(k_pop, 2),
    "k_EM_s": round(K_EM, 2),
    "em_fraction_of_pop_rate": round(em_fraction_of_pop_rate, 4),
}

checks = {
    "frequencies_sum_to_1":       abs(total_freq - 1.0) < 1e-9,
    "em_dominates_pop_rate":      em_fraction_of_pop_rate > 0.4,
    "k_pop_close_to_k_EM":        abs(k_pop - K_EM) / K_EM < 0.5,
    "pm_freq_between_0.05_0.10":  0.05 < FREQ_PM < 0.10,
    "um_freq_gt_0.05":            FREQ_UM > 0.05,
}

write_result(name, data, checks)
