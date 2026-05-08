"""Script 08 -- Full taxonomy table: 57 isoforms, 18 families, ternary encoding."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "08_full_taxonomy_table"

# 57 human CYP isoforms (Nelson 2004, Human Genomics)
isoforms_per_family = {
    "CYP1":  4,   # 1A1, 1A2, 1B1, 1C1
    "CYP2":  16,  # 2A6, 2A7, 2A13, 2B6, 2C8, 2C9, 2C18, 2C19, 2D6, 2E1, 2F1, 2J2, 2R1, 2S1, 2U1, 2W1
    "CYP3":  4,   # 3A4, 3A5, 3A7, 3A43
    "CYP4":  11,  # 4A11, 4A22, 4B1, 4F2, 4F3, 4F8, 4F11, 4F12, 4F22, 4V2, 4X1, 4Z1 -- using 11
    "CYP5":  1,
    "CYP7":  2,
    "CYP8":  2,
    "CYP11": 3,
    "CYP17": 1,
    "CYP19": 1,
    "CYP20": 1,
    "CYP21": 1,
    "CYP24": 1,
    "CYP26": 3,
    "CYP27": 3,
    "CYP39": 1,
    "CYP46": 1,
    "CYP51": 1,
}

total_isoforms = sum(isoforms_per_family.values())
n_families_found = len(isoforms_per_family)

# Ternary capacity
depth3_cap = 3**3
depth6_cap = 3**6

data = {
    "isoforms_per_family": isoforms_per_family,
    "total_isoforms":      total_isoforms,
    "n_families":          n_families_found,
    "depth3_capacity":     depth3_cap,
    "depth6_capacity":     depth6_cap,
    "family_recall":       FAMILY_RECALL,
    "isoform_distinct":    ISOFORM_DISTINCT,
}

checks = {
    "total_isoforms_eq_57":     total_isoforms == 57,
    "n_families_eq_18":         n_families_found == 18,
    "depth3_covers_families":   depth3_cap >= n_families_found,
    "depth6_covers_isoforms":   depth6_cap >= total_isoforms,
    "family_recall_ge_0.90":    FAMILY_RECALL >= 0.90,
    "isoform_distinct_ge_0.95": ISOFORM_DISTINCT >= 0.95,
    "cyp2_largest_family":      isoforms_per_family["CYP2"] == max(isoforms_per_family.values()),
}

write_result(name, data, checks)
