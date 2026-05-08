"""Script 07 -- PharmVar allele capacity: ternary address covers all known variants."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_pharmvar_allele_capacity"

# PharmVar database statistics (2023)
# CYP2D6: ~150 named alleles
# CYP2C9: ~75 named alleles
# CYP2C19: ~45 named alleles
# CYP3A4: ~40 named alleles
pharmvar_alleles = {
    "CYP2D6":  150,
    "CYP2C9":  75,
    "CYP2C19": 45,
    "CYP3A4":  40,
}

total_alleles = sum(pharmvar_alleles.values())

# Capacity at each depth
cap_9  = 3**9    # 19683
cap_6  = 3**6    # 729
cap_12 = 3**12   # 531441

# Minimum depth to cover total alleles across all isoforms
min_depth_total = math.ceil(math.log(total_alleles) / math.log(3))

# For a single gene (CYP2D6) with 150 alleles
min_depth_2d6 = math.ceil(math.log(pharmvar_alleles["CYP2D6"]) / math.log(3))

data = {
    "pharmvar_alleles":     pharmvar_alleles,
    "total_alleles":        total_alleles,
    "cap_at_k9":            cap_9,
    "min_depth_total":      min_depth_total,
    "min_depth_cyp2d6":     min_depth_2d6,
}

checks = {
    "k9_covers_all_alleles":      cap_9 >= total_alleles,
    "k6_covers_cyp2d6":           cap_6 >= pharmvar_alleles["CYP2D6"],
    "min_depth_total_le_9":       min_depth_total <= 9,
    "min_depth_2d6_le_5":         min_depth_2d6 <= 5,
    "total_alleles_gt_300":       total_alleles > 300,
}

write_result(name, data, checks)
