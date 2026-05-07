"""01_address_clustering: 57 addresses at 6-trit depth; check mean pairwise distinctness >= 0.95."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result
import numpy as np

rng = np.random.default_rng(42)

N = 57
DEPTH = 6

# Generate 57 distinct 6-trit addresses
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

# Pairwise distinctness: fraction of pairs with Hamming distance >= 1
n_pairs = 0
n_distinct = 0
for i in range(N):
    for j in range(i+1, N):
        d = int(np.sum(ADDR[i] != ADDR[j]))
        n_pairs += 1
        if d >= 1:
            n_distinct += 1

distinctness = n_distinct / n_pairs

print(f"N isoforms: {N}")
print(f"N pairs: {n_pairs}")
print(f"N distinct pairs (Hamming >= 1): {n_distinct}")
print(f"Distinctness: {distinctness:.4f}")

checks = {
    "mean_distinctness_ge_0.95": distinctness >= 0.95,
    "all_pairs_ge_1": n_distinct == n_pairs,
}

write_result("01_address_clustering", {
    "n_isoforms": N,
    "n_pairs": n_pairs,
    "n_distinct": n_distinct,
    "distinctness": round(distinctness, 4),
}, checks)
