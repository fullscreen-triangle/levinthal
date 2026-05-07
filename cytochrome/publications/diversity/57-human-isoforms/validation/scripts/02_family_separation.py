"""02_family_separation: check inter-family distance > intra-family at k=3."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result
import numpy as np

rng = np.random.default_rng(42)

N = 57
DEPTH = 6

FAMILIES = {
    "CYP1":   list(range(0, 3)),
    "CYP2":   list(range(3, 17)),
    "CYP3":   list(range(17, 21)),
    "CYP4-51": list(range(21, 57)),
}

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

def hamming(a, b, depth):
    return int(np.sum(a[:depth] != b[:depth]))

k = 3
intra_dists = []
inter_dists = []

fam_list = list(FAMILIES.keys())
fam_idxs = list(FAMILIES.values())

for fi in range(len(fam_list)):
    idxs_i = fam_idxs[fi]
    for a in range(len(idxs_i)):
        for b in range(a+1, len(idxs_i)):
            intra_dists.append(hamming(ADDR[idxs_i[a]], ADDR[idxs_i[b]], k))
    for fj in range(fi+1, len(fam_list)):
        idxs_j = fam_idxs[fj]
        for a in idxs_i:
            for b in idxs_j:
                inter_dists.append(hamming(ADDR[a], ADDR[b], k))

intra_mean = np.mean(intra_dists)
inter_mean = np.mean(inter_dists)
ratio = inter_mean / max(intra_mean, 1e-9)

print(f"k=3 intra-family mean Hamming: {intra_mean:.3f}")
print(f"k=3 inter-family mean Hamming: {inter_mean:.3f}")
print(f"Ratio inter/intra: {ratio:.3f}")

checks = {
    "inter_gt_intra_k3": float(inter_mean) > float(intra_mean),
    "ratio_gt_1.5": ratio > 1.5,
}

write_result("02_family_separation", {
    "k": k,
    "intra_family_mean_hamming": round(float(intra_mean), 3),
    "inter_family_mean_hamming": round(float(inter_mean), 3),
    "ratio_inter_over_intra": round(float(ratio), 3),
}, checks)
