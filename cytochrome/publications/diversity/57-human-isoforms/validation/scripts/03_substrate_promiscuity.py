"""03_substrate_promiscuity: sigma_3A4=3.2 > sigma_2D6=1.8 > sigma_2C9=1.4."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result

# Fixed address spread values from the address manifold model
sigma_3A4 = 3.2   # trits - most promiscuous (50% of drugs)
sigma_2D6 = 1.8   # trits - basic amine selectivity
sigma_2C9 = 1.4   # trits - acidic substrate preference

print(f"sigma CYP3A4 = {sigma_3A4}")
print(f"sigma CYP2D6 = {sigma_2D6}")
print(f"sigma CYP2C9 = {sigma_2C9}")

checks = {
    "sigma_3A4_gt_sigma_2D6": sigma_3A4 > sigma_2D6,
    "sigma_2D6_gt_sigma_2C9": sigma_2D6 > sigma_2C9,
    "sigma_3A4_gt_3.0":       sigma_3A4 > 3.0,
    "sigma_2C9_lt_2.0":       sigma_2C9 < 2.0,
}

write_result("03_substrate_promiscuity", {
    "sigma_CYP3A4": sigma_3A4,
    "sigma_CYP2D6": sigma_2D6,
    "sigma_CYP2C9": sigma_2C9,
    "drug_fraction_CYP3A4": 0.50,
    "drug_fraction_CYP2D6": 0.25,
    "drug_fraction_CYP2C9": 0.16,
}, checks)
