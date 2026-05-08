"""Script 03 -- Recovery from a 70%-complete ternary address."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_partial_address_recovery"

# If 70% of the k=6 address positions are known (4.2 of 6 trits effectively),
# can we uniquely identify the isoform?
known_fraction = MIN_KNOWN_FRACTION   # 0.70
effective_depth = known_fraction * DEPTH_ISOFORM  # 4.2 trits
bits_available  = bits_at_depth(effective_depth)  # 4.2 * 1.585 = 6.66 bits

bits_needed = math.log2(N_HUMAN_CYPS)  # 5.83 bits for 57 isoforms

recovery_possible = bits_available > bits_needed

# Probability of incorrect identification at 70% address:
# P_error = exp(-(bits_available - bits_needed))  # exponential drop
p_error = math.exp(-(bits_available - bits_needed))
p_correct = 1.0 - p_error

data = {
    "known_fraction":      known_fraction,
    "effective_depth":     round(effective_depth, 2),
    "bits_available":      round(bits_available, 4),
    "bits_needed":         round(bits_needed, 4),
    "recovery_possible":   recovery_possible,
    "p_correct":           round(p_correct, 4),
    "p_error":             round(p_error, 4),
}

checks = {
    "70pct_address_sufficient":  recovery_possible,
    "bits_available_gt_needed":  bits_available > bits_needed,
    "p_correct_gt_0.85":         p_correct > 0.85,
    "p_error_lt_0.15":           p_error < 0.15,
    "known_fraction_ge_0.70":    known_fraction >= 0.70,
}

write_result(name, data, checks)
