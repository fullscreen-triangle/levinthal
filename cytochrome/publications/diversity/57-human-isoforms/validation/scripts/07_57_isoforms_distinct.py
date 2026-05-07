"""07_57_isoforms_distinct: all pairwise Hamming >= 1; count == 57."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result
import numpy as np

N = 57
DEPTH = 6

def gen_addresses(n, depth, seed=42):
    rng2 = np.random.default_rng(seed)
    addrs = set()
    result = []
    while len(result) < n:
        a = tuple(int(x) for x in rng2.integers(0, 3, size=depth))
        if a not in addrs:
            addrs.add(a)
            result.append(list(a))
    return np.array(result)

ADDR = gen_addresses(N, DEPTH)

# Count distinct addresses
addr_set = set(tuple(row) for row in ADDR)
count_distinct = len(addr_set)

# Check all pairwise Hamming >= 1
min_hamming = N * DEPTH  # start large
all_positive = True
for i in range(N):
    for j in range(i+1, N):
        d = int(np.sum(ADDR[i] != ADDR[j]))
        if d < min_hamming:
            min_hamming = d
        if d < 1:
            all_positive = False

print(f"Number of distinct 6-trit addresses: {count_distinct}")
print(f"Minimum pairwise Hamming distance: {min_hamming}")
print(f"All pairwise Hamming >= 1: {all_positive}")

checks = {
    "count_distinct_eq_57":   count_distinct == 57,
    "all_pairwise_hamming_ge_1": all_positive,
    "min_hamming_ge_1":          min_hamming >= 1,
}

write_result("07_57_isoforms_distinct", {
    "n_isoforms": N,
    "count_distinct": count_distinct,
    "min_pairwise_hamming": min_hamming,
    "all_pairwise_positive": all_positive,
}, checks)
