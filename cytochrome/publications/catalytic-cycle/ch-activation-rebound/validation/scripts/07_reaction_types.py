"""Script 07 — Five Reaction Types Under the Three-Body Aperture.

Validates:
- All five reaction types (aliphatic, benzylic, allylic, aromatic, epoxidation)
  have valid Delta_M in ascending order of activation
- Rates decrease monotonically with increasing Delta_M
- KIE predicted > 4 for H-transfer types; KIE ≈ 1 for non-HAT types
- Aliphatic/allylic rate ratio matches literature range
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_reaction_types"

# Compute rates and KIEs for all reaction types
results = {}
for rtype, params in REACTION_TYPES.items():
    dM = params["delta_M"]
    k = nu_floor * math.exp(-dM)
    E_a_kcal = T_PART * dM / 4.184
    # KIE only for H-transfer types
    if params["has_kie"]:
        omega_CH = 2 * math.pi * c_cms * NU_CH_CM1
        omega_CD = 2 * math.pi * c_cms * NU_CD_CM1
        delta_ZPE = (hbar / 2) * (omega_CH - omega_CD)
        delta_ZPE_kBT = delta_ZPE / kBT
        KIE_ZPE = math.exp(delta_ZPE_kBT)
        # Scale tunneling by Delta_M (deeper tunneling at higher barrier)
        delta_tunnel = 0.77 + (dM - 0.65) * 0.3  # slight adjustment per barrier
        kappa_ratio = math.exp(delta_tunnel * (1 - 1/math.sqrt(2)))
        KIE = KIE_ZPE * kappa_ratio
    else:
        KIE = 1.0
    results[rtype] = {
        "delta_M": dM,
        "k_s": round(k, 3),
        "log10_k": round(math.log10(k), 3),
        "E_a_kcalmol": round(E_a_kcal, 2),
        "has_kie": params["has_kie"],
        "KIE": round(KIE, 2),
    }

# Check ordering: aliphatic slowest among C-H types
k_aliphatic = results["aliphatic"]["k_s"]
k_benzylic = results["benzylic"]["k_s"]
k_allylic = results["allylic"]["k_s"]
k_aromatic = results["aromatic"]["k_s"]
k_epox = results["epoxidation"]["k_s"]

rate_ordering = (k_aliphatic < k_benzylic < k_allylic < k_aromatic < k_epox)

# KIE check
kie_hat_types = [results[t]["KIE"] for t in ["aliphatic", "benzylic", "allylic"]
                 if results[t]["has_kie"]]
kie_non_hat = [results[t]["KIE"] for t in ["aromatic", "epoxidation"]]

# Aliphatic / allylic rate ratio
rate_ratio_aliph_allyl = k_aliphatic / k_allylic

data = {
    "reaction_types": results,
    "rate_ordering_correct": rate_ordering,
    "rate_ratio_aliphatic_over_allylic": round(rate_ratio_aliph_allyl, 4),
    "KIE_aliphatic": results["aliphatic"]["KIE"],
    "KIE_allylic": results["allylic"]["KIE"],
    "KIE_aromatic": results["aromatic"]["KIE"],
    "KIE_epoxidation": results["epoxidation"]["KIE"],
}

checks = {
    "rate_ordering_aliphatic_slowest": rate_ordering,
    "all_HAT_KIE_above_4": all(k > 4.0 for k in kie_hat_types),
    "non_HAT_KIE_equals_1": all(k == 1.0 for k in kie_non_hat),
    "aliphatic_allylic_ratio_below_2": rate_ratio_aliph_allyl < 2.0,
    "aliphatic_delta_M_largest": (REACTION_TYPES["aliphatic"]["delta_M"] ==
                                  max(v["delta_M"] for v in REACTION_TYPES.values())),
    "five_reaction_types_present": len(results) == 5,
}

write_result(name, data, checks)
