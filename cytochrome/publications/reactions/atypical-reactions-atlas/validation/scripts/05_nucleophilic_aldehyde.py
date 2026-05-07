"""Script 05 -- Nucleophilic O-atom transfer to aldehyde C=O."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_nucleophilic_aldehyde"

k_nuc = K_NUC    # 1e10 * exp(-0.42)
kie_nuc = 1.0    # no H transferred

ea_nuc_kcal = T_PART * DELTA_M_NUCLEOPHILIC / 4.184

# Nucleophilic mechanism: Fe=O acts as electrophile toward aldehyde
# Peracid-like: [Fe-OOH] attacks aldehyde (alternative to standard Cpd I mechanism)
# This applies to CYP11A1 (cholesterol side-chain cleavage) and some steroid P450s

data = {
    "delta_m_nucleophilic": DELTA_M_NUCLEOPHILIC,
    "k_nuc_s": round(k_nuc, 2),
    "kie_nuc": kie_nuc,
    "ea_nuc_kcal": round(ea_nuc_kcal, 3),
}

checks = {
    "k_nuc_in_range_1e9_to_1e10": 1e9 < k_nuc < 1e10,
    "no_kie_nucleophilic": kie_nuc == 1.0,
    "ea_nuc_lt_8_kcal": ea_nuc_kcal < 8.0,
    "k_nuc_gt_k_desat_eff": k_nuc > K_DESAT_EFF,
    "delta_m_nuc_between_epox_and_n_dealk": DELTA_M_EPOXIDATION < DELTA_M_NUCLEOPHILIC < 0.55,
}

write_result(name, data, checks)
