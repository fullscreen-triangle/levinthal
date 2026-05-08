"""Script 08 -- Full spectral atlas table: all 7 states, all techniques."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "08_full_spectral_table"

atlas = {
    "resting_FeIII_LS": {
        "soret_nm":   417, "spin": "LS", "epr_g1": 2.42, "raman": None
    },
    "substrate_bound_HS": {
        "soret_nm":   392, "spin": "HS", "epr_g1": 7.70, "raman": None
    },
    "ferrous_FeII": {
        "soret_nm":   408, "spin": "LS", "epr_g1": None, "raman": None
    },
    "oxy_complex": {
        "soret_nm":   418, "spin": "LS", "epr_g1": 2.00, "raman": None
    },
    "peroxo": {
        "soret_nm":   440, "spin": "LS", "epr_g1": 2.31, "raman": None
    },
    "compound0": {
        "soret_nm":   367, "spin": "LS", "epr_g1": None, "raman": None
    },
    "compound_I": {
        "soret_nm":   370, "spin": "HS_radical", "epr_g1": 2.00, "raman": RAMAN_FEO_CM1
    },
}

all_soret_positive = all(v["soret_nm"] > 0 for v in atlas.values())
cpd1_has_raman = atlas["compound_I"]["raman"] is not None
hs_soret_lt_ls = atlas["substrate_bound_HS"]["soret_nm"] < atlas["resting_FeIII_LS"]["soret_nm"]
iso_shift_pred = round(RAMAN_FEO_CM1 - RAMAN_FEO_18O, 1)

data = {
    "atlas":             {k: {kk: vv for kk, vv in v.items() if vv is not None}
                          for k, v in atlas.items()},
    "raman_16O_cm1":     RAMAN_FEO_CM1,
    "raman_18O_shift":   iso_shift_pred,
    "n_states":          len(atlas),
}

checks = {
    "seven_states_in_atlas":  len(atlas) == 7,
    "all_soret_positive":     all_soret_positive,
    "cpd1_has_raman_signal":  cpd1_has_raman,
    "hs_soret_blueshifted":   hs_soret_lt_ls,
    "raman_shift_gt_30cm1":   iso_shift_pred > 30,
    "all_soret_between_350_470": all(350 <= v["soret_nm"] <= 470 for v in atlas.values()),
}

write_result(name, data, checks)
