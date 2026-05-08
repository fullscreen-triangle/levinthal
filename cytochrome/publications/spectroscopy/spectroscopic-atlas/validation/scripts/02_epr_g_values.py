"""Script 02 -- EPR g-values for high-spin and low-spin P450 states."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_epr_g_values"

# LS rhombic EPR: g1 > g2 > g3; g1 * g2 * g3 ≈ 8 (for S=1/2)
g_ls = EPR_G_LS
g_hs = EPR_G_HS

g_ls_product = g_ls[0] * g_ls[1] * g_ls[2]
g_hs_max     = max(g_hs)
rhombicity_ls = (g_ls[0] - g_ls[1]) / (g_ls[0] - g_ls[2])  # 0 = axial, 1 = rhombic

# Spin state determination: HS EPR appears at low field (g > 6)
hs_low_field_signal = g_hs[0] > 6.0

data = {
    "g_ls":                list(g_ls),
    "g_hs":                list(g_hs),
    "g_ls_product":        round(g_ls_product, 3),
    "rhombicity_ls":       round(rhombicity_ls, 4),
    "hs_max_g":            g_hs_max,
}

checks = {
    "ls_g1_gt_g2_gt_g3":   g_ls[0] > g_ls[1] > g_ls[2],
    "ls_g1_near_2.4":      2.3 < g_ls[0] < 2.6,
    "ls_g3_near_1.9":      1.8 < g_ls[2] < 2.0,
    "hs_low_field_gt_6":   hs_low_field_signal,
    "hs_g_max_near_7.7":   abs(g_hs_max - 7.7) < 0.5,
    "rhombicity_between_0_1": 0 < rhombicity_ls < 1,
}

write_result(name, data, checks)
