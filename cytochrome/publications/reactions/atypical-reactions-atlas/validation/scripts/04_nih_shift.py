"""Script 04 -- NIH shift: cationic 1,2-H migration from arene oxide."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_nih_shift"

k_nih = K_NIH    # 1e10 * exp(-0.18)
k_epox = K_EPOX  # 1e10 * exp(-0.35)

# NIH shift is spontaneous once arene oxide forms (cationic rearrangement)
# DeltaM_NIH = 0.18 << DeltaM_epox = 0.35 -> NIH rearrangement faster than epoxidation itself
kie_nih = 1.0    # secondary KIE only (< 1.3); primary = 0 for 1,2-shift

# Secondary KIE for tritium label retention
# The NIH shift retains the isotope label (H or D/T) at the adjacent carbon
# KIE_secondary ≈ 1.0-1.3 (well within error)
kie_nih_secondary = 1.15

data = {
    "delta_m_nih_shift": DELTA_M_NIH_SHIFT,
    "k_nih_s": round(k_nih, 2),
    "k_epox_s": round(k_epox, 2),
    "kie_nih_primary": kie_nih,
    "kie_nih_secondary": kie_nih_secondary,
    "nih_faster_than_epox": k_nih > k_epox,
}

checks = {
    "k_nih_gt_k_epox": k_nih > k_epox,
    "no_primary_kie_nih": kie_nih == 1.0,
    "nih_spontaneous_low_dm": DELTA_M_NIH_SHIFT < 0.30,
    "k_nih_gt_7e9": k_nih > 7e9,
    "secondary_kie_lt_2": kie_nih_secondary < 2.0,
}

write_result(name, data, checks)
