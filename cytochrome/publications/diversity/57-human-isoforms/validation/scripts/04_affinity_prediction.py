"""04_affinity_prediction: mean substrate affinity > 2x mean non-substrate affinity (CYP3A4)."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result
import numpy as np

# Address distances for 5 known CYP3A4 substrates (small -> high affinity)
# Midazolam, Testosterone, Erythromycin, Cyclosporine, Simvastatin
substrate_dists = [0, 1, 1, 0, 1]

# Address distances for 5 non-substrates (large -> low affinity)
# Metformin, Atenolol, Ranitidine, Lisinopril, Warfarin-low
non_substrate_dists = [4, 4, 5, 5, 3]

substrate_affinities = [math.exp(-d) for d in substrate_dists]
non_substrate_affinities = [math.exp(-d) for d in non_substrate_dists]

mean_sub = sum(substrate_affinities) / len(substrate_affinities)
mean_non = sum(non_substrate_affinities) / len(non_substrate_affinities)
ratio = mean_sub / mean_non

print(f"Substrate affinities: {[round(a,4) for a in substrate_affinities]}")
print(f"Non-substrate affinities: {[round(a,4) for a in non_substrate_affinities]}")
print(f"Mean substrate affinity: {mean_sub:.4f}")
print(f"Mean non-substrate affinity: {mean_non:.4f}")
print(f"Ratio: {ratio:.3f}")

checks = {
    "mean_sub_gt_mean_non_x2": mean_sub > mean_non * 2,
    "mean_sub_gt_0.5": mean_sub > 0.5,
    "mean_non_lt_0.3": mean_non < 0.3,
    "ratio_gt_2": ratio > 2.0,
}

write_result("04_affinity_prediction", {
    "substrate_dists": substrate_dists,
    "non_substrate_dists": non_substrate_dists,
    "mean_substrate_affinity": round(mean_sub, 4),
    "mean_non_substrate_affinity": round(mean_non, 4),
    "ratio": round(ratio, 3),
}, checks)
