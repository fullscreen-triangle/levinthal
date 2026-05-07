"""06_tissue_distribution: CYP3A4 > CYP1A2 in gut; CYP1A1 > CYP3A4 in lung."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result

# Relative expression levels (% of max expression for that isoform)
# Tissue indices: liver=0, gut=1, lung=2
expression = {
    "CYP3A4": {"liver": 100, "gut": 80,  "lung": 30},
    "CYP1A2": {"liver": 95,  "gut": 20,  "lung": 5},
    "CYP1A1": {"liver": 10,  "gut": 5,   "lung": 70},
    "CYP2D6": {"liver": 50,  "gut": 10,  "lung": 5},
    "CYP2C9": {"liver": 80,  "gut": 30,  "lung": 10},
}

cyp3a4_gut  = expression["CYP3A4"]["gut"]
cyp1a2_gut  = expression["CYP1A2"]["gut"]
cyp1a1_lung = expression["CYP1A1"]["lung"]
cyp3a4_lung = expression["CYP3A4"]["lung"]

print(f"CYP3A4 gut expression:  {cyp3a4_gut}%")
print(f"CYP1A2 gut expression:  {cyp1a2_gut}%")
print(f"CYP1A1 lung expression: {cyp1a1_lung}%")
print(f"CYP3A4 lung expression: {cyp3a4_lung}%")

checks = {
    "CYP3A4_gt_CYP1A2_gut":  cyp3a4_gut > cyp1a2_gut,
    "CYP1A1_gt_CYP3A4_lung": cyp1a1_lung > cyp3a4_lung,
    "CYP3A4_gut_gt_50":      cyp3a4_gut > 50,
    "CYP1A1_lung_gt_50":     cyp1a1_lung > 50,
}

write_result("06_tissue_distribution", {
    "CYP3A4_gut_pct":  cyp3a4_gut,
    "CYP1A2_gut_pct":  cyp1a2_gut,
    "CYP1A1_lung_pct": cyp1a1_lung,
    "CYP3A4_lung_pct": cyp3a4_lung,
}, checks)
