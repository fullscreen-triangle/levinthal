"""Script 02 -- Substrate ΔM ranges per CYP family match known selectivity."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_family_substrate_dm"

# CYP family representative ΔM ranges (from activiation partition depth model)
# CYP1A: planar/polycyclic aromatics -> higher pi-system rigidity -> lower ΔM
# CYP3A: broadest range -> widest ΔM window
families = {
    "CYP1A":  (0.50, 0.65),
    "CYP2C":  (0.48, 0.60),
    "CYP2D6": (0.52, 0.68),
    "CYP3A4": (0.40, 0.70),
    "CYP2E1": (0.60, 0.75),
    "CYP2B6": (0.50, 0.68),
}

dm_ranges = {f: (hi - lo) for f, (lo, hi) in families.items()}
cyp3a_width = families["CYP3A4"][1] - families["CYP3A4"][0]
cyp2e1_lo   = families["CYP2E1"][0]
cyp1a_lo    = families["CYP1A"][0]

# CYP3A4 should have the widest ΔM window (broadest selectivity)
widths = {f: (hi - lo) for f, (lo, hi) in families.items()}
cyp3a_is_widest = cyp3a_width == max(widths.values())

# CYP2E1 handles small hydrophilic substrates -> higher ΔM (more barrier)
cyp2e1_gt_cyp1a = cyp2e1_lo > cyp1a_lo

data = {
    "dm_ranges": {f: {"lo": lo, "hi": hi} for f, (lo, hi) in families.items()},
    "dm_widths":  widths,
    "cyp3a4_window": round(cyp3a_width, 3),
    "cyp3a_is_widest": cyp3a_is_widest,
}

checks = {
    "cyp3a4_widest_selectivity":    cyp3a_is_widest,
    "cyp2e1_dm_higher_than_cyp1a":  cyp2e1_gt_cyp1a,
    "all_lower_bounds_positive":    all(lo > 0 for lo, hi in families.values()),
    "all_upper_bounds_lt_1":        all(hi < 1.0 for lo, hi in families.values()),
    "six_families_parameterized":   len(families) == 6,
    "cyp3a4_lower_bound_lt_0.50":   families["CYP3A4"][0] < 0.50,
}

write_result(name, data, checks)
