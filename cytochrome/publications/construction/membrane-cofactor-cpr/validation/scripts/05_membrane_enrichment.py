"""Script 05 - Membrane Substrate Enrichment.

Validates:
- Membrane enrichment factor = 10^(logP - 2) for logP > 2
- For logP=3: enrichment = 10
- Apparent K_m reduced by enrichment factor
- Enrichment >= 5 for logP = 3
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_membrane_enrichment"

# Membrane thickness: ER bilayer ~40 Ang
membrane_thickness_ang = 40.0

# Enrichment factor for lipophilic substrates
# Model: enrichment = 10^(logP - 2) for logP > 2
def enrichment_factor(logP):
    if logP <= 2.0:
        return 1.0
    return 10.0 ** (logP - 2.0)

# Test substrates
substrates = {
    "hydrophilic": {"logP": 0.5, "K_m_intrinsic_uM": 50.0},
    "borderline":  {"logP": 2.0, "K_m_intrinsic_uM": 20.0},
    "lipophilic":  {"logP": 3.0, "K_m_intrinsic_uM": 10.0},
    "very_lipo":   {"logP": 4.5, "K_m_intrinsic_uM": 5.0},
}

results = {}
for name_sub, props in substrates.items():
    ef = enrichment_factor(props["logP"])
    K_m_app = props["K_m_intrinsic_uM"] / ef
    results[name_sub] = {
        "logP": props["logP"],
        "enrichment_factor": round(ef, 2),
        "K_m_intrinsic_uM": props["K_m_intrinsic_uM"],
        "K_m_apparent_uM": round(K_m_app, 3),
    }

# Key check: logP=3 -> enrichment >= 5
ef_logP3 = enrichment_factor(3.0)

# For logP=3: 10^(3-2) = 10
# For logP=4.5: 10^(4.5-2) = 10^2.5 ~ 316
ef_logP45 = enrichment_factor(4.5)

data = {
    "membrane_thickness_ang": membrane_thickness_ang,
    "enrichment_model": "10^(logP - 2) for logP > 2",
    "enrichment_logP3": round(ef_logP3, 2),
    "enrichment_logP45": round(ef_logP45, 2),
    "substrate_results": results,
}

checks = {
    "enrichment_logP3_ge_5": ef_logP3 >= 5.0,
    "enrichment_logP3_equals_10": abs(ef_logP3 - 10.0) < 0.01,
    "enrichment_logP2_equals_1": abs(enrichment_factor(2.0) - 1.0) < 0.01,
    "enrichment_increases_with_logP": ef_logP45 > ef_logP3,
    "K_m_apparent_less_than_intrinsic": results["lipophilic"]["K_m_apparent_uM"] < results["lipophilic"]["K_m_intrinsic_uM"],
}

write_result(name, data, checks)
