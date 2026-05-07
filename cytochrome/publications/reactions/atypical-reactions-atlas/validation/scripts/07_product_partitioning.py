"""Script 07 -- Arene oxide product partitioning: phenol (NIH) vs dihydrodiol."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_product_partitioning"

# After arene oxide forms, two competing fates:
# 1. NIH shift -> cationic rearrangement -> keto form -> tautomerize to phenol
# 2. Enzymatic/spontaneous hydration -> trans-dihydrodiol (via epoxide hydrolase)
#    k_NIH >> k_hydration (nonenzymatic) -> phenol dominant
#    k_NIH > k_epoxide_hydrolase_at_low_EH -> mixed products

k_nih = K_NIH          # 8.35e9 s^-1 (fast cationic rearrangement)

# Competing pathway: ring-opening to dihydrodiol
# Without EH: nonenzymatic hydration is slow (DeltaM_hydration_nonenzymatic = 0.60)
k_hydration_slow = nu_floor * math.exp(-0.60)    # ~5.49e9 s^-1

fraction_phenol = k_nih / (k_nih + k_hydration_slow)
fraction_dihydrodiol = k_hydration_slow / (k_nih + k_hydration_slow)

# Check that phenol is the majority product (> 45%)
data = {
    "k_nih_s": round(k_nih, 2),
    "k_hydration_competing_s": round(k_hydration_slow, 2),
    "fraction_phenol": round(fraction_phenol, 4),
    "fraction_dihydrodiol": round(fraction_dihydrodiol, 4),
    "phenol_dominant": fraction_phenol > fraction_dihydrodiol,
}

checks = {
    "fraction_phenol_gt_0.45": fraction_phenol > 0.45,
    "phenol_dominant": fraction_phenol > fraction_dihydrodiol,
    "fractions_sum_to_1": abs(fraction_phenol + fraction_dihydrodiol - 1.0) < 1e-9,
    "k_nih_gt_k_competing": k_nih > k_hydration_slow,
    "fraction_dihydrodiol_nonzero": fraction_dihydrodiol > 0,
}

write_result(name, data, checks)
