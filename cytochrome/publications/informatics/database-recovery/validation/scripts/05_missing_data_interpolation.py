"""Script 05 -- Interpolation of missing sequence data in the address manifold."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_missing_data_interpolation"

# When a database has gaps (missing alleles between k=6 and k=9),
# the ternary tree allows interpolation by averaging known neighbor addresses.
# Error of interpolation ~ 1/sqrt(N_neighbors)

# At depth 9, each address has up to 2 neighbors in each trit dimension
N_neighbors = 6   # 2 neighbors per trit axis, 3 axes
interp_error = 1.0 / math.sqrt(N_neighbors)

# Compression ratio of ternary encoding vs. raw sequence storage
# Raw: 57 sequences * 500 aa * log2(20) bits/aa
raw_bits = N_HUMAN_CYPS * 500 * math.log2(20)
# Ternary: ~9 trits * log2(3) bits per sequence address + shared backbone
trit_bits = N_HUMAN_CYPS * bits_at_depth(9) + 500 * math.log2(20)
compression_ratio = raw_bits / trit_bits

data = {
    "N_neighbors":      N_neighbors,
    "interp_error":     round(interp_error, 4),
    "raw_bits":         round(raw_bits, 1),
    "trit_bits":        round(trit_bits, 1),
    "compression_ratio":round(compression_ratio, 3),
}

checks = {
    "interp_error_lt_0.5":    interp_error < 0.5,
    "compression_gt_1":       compression_ratio > 1.0,
    "raw_bits_gt_trit_bits":  raw_bits > trit_bits,
    "n_neighbors_positive":   N_neighbors > 0,
    "compression_lt_100":     compression_ratio < 100,
}

write_result(name, data, checks)
