"""Script 01 -- Information capacity of ternary address at each depth."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_information_capacity"

bits_at_3 = bits_at_depth(3)
bits_at_6 = bits_at_depth(6)
bits_at_9 = bits_at_depth(9)

# Compare with minimum bits needed to identify each classification level
bits_needed_18  = math.log2(18)   # ~4.17 bits for families
bits_needed_57  = math.log2(57)   # ~5.83 bits for isoforms
bits_needed_1000= math.log2(1000) # ~9.97 bits for alleles

data = {
    "bits_per_trit":  round(BITS_PER_TRIT, 4),
    "bits_at_k3":     round(bits_at_3, 4),
    "bits_at_k6":     round(bits_at_6, 4),
    "bits_at_k9":     round(bits_at_9, 4),
    "bits_needed_18_families": round(bits_needed_18, 4),
    "bits_needed_57_isoforms": round(bits_needed_57, 4),
    "bits_needed_1000_alleles":round(bits_needed_1000, 4),
}

checks = {
    "bits_k3_exceeds_families":   bits_at_3 > bits_needed_18,
    "bits_k6_exceeds_isoforms":   bits_at_6 > bits_needed_57,
    "bits_k9_exceeds_alleles":    bits_at_9 > bits_needed_1000,
    "bits_per_trit_approx_1.585": abs(BITS_PER_TRIT - 1.585) < 0.001,
    "monotonic_bits_with_depth":  bits_at_3 < bits_at_6 < bits_at_9,
}

write_result(name, data, checks)
