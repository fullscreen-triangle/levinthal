"""Script 01 -- Minimum ternary depth required to separate 18 CYP families."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_ternary_depth_families"

# 3^k >= 18 families  => k >= 3  (3^2=9 < 18, 3^3=27 >= 18)
depth_needed_families  = min_depth_for(N_FAMILIES)
depth_needed_isoforms  = min_depth_for(N_HUMAN_CYPS)
depth_needed_alleles   = min_depth_for(1000)  # >1000 alleles in PharmVar

# Verify trit capacity at each depth
cap_3 = 3**3   # 27 >= 18 families
cap_6 = 3**6   # 729 >= 57 isoforms
cap_9 = 3**9   # 19683 >> alleles

data = {
    "depth_families":      depth_needed_families,
    "depth_isoforms":      depth_needed_isoforms,
    "depth_alleles":       depth_needed_alleles,
    "trit_cap_at_3":       cap_3,
    "trit_cap_at_6":       cap_6,
    "trit_cap_at_9":       cap_9,
    "n_families":          N_FAMILIES,
    "n_isoforms":          N_HUMAN_CYPS,
}

checks = {
    "depth3_separates_families":     depth_needed_families == 3,
    "depth6_separates_isoforms":     depth_needed_isoforms <= 6,
    "trit_cap3_ge_18":               cap_3 >= N_FAMILIES,
    "trit_cap6_ge_57":               cap_6 >= N_HUMAN_CYPS,
    "trit_cap9_ge_1000_alleles":     cap_9 >= 1000,
    "family_recall_ge_0.9":          FAMILY_RECALL >= 0.9,
    "isoform_distinct_ge_0.95":      ISOFORM_DISTINCT >= 0.95,
}

write_result(name, data, checks)
