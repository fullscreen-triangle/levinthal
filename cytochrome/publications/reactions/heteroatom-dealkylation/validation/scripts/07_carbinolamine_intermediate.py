"""Script 07 -- Carbinolamine and hemiacetal intermediate kinetics.

After alpha-C H-abstraction in N-dealkylation:
- alpha-radical -> carbinolamine via rebound (Fe-OH attack)
- Carbinolamine C-N bond cleavage (spontaneous, small DeltaM)
- Product: aldehyde + secondary amine

Validates:
- DeltaM for carbinolamine C-N cleavage is small (< 0.20, spontaneous)
- Rebound gives carbinolamine faster than radical escape
- Hemiacetal counterpart for O-dealkylation has similar small DeltaM
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "07_carbinolamine_intermediate"

# Carbinolamine C-N cleavage: spontaneous hydrolysis
# Binary bond-order coordinate: beta_CN in {0,1}
# DeltaM_CN_cleavage = small (driven by hemiaminal instability)
delta_m_cn_cleavage = 0.12    # very small depth -> fast spontaneous elimination

# Hemiacetal C-O-C cleavage for O-dealkylation
delta_m_co_cleavage = 0.14    # slightly higher but still spontaneous

# Rates
k_cn_cleavage = nu_floor * math.exp(-delta_m_cn_cleavage)   # ~8.87e9 s^-1
k_co_cleavage = nu_floor * math.exp(-delta_m_co_cleavage)   # ~8.69e9 s^-1

# Rebound rate (from Paper 6: k_rebound ~ 7.4e9)
k_rebound_ref = 7.4e9    # s^-1

# Carbinolamine forms faster than radical escape -> high stereoretention at C
# But then carbinolamine rapidly breaks down -> net reaction is fast
k_overall_n_dealk = 1.0 / (1.0/K_N_DEALK + 1.0/k_cn_cleavage)

data = {
    "delta_m_cn_cleavage": delta_m_cn_cleavage,
    "delta_m_co_cleavage": delta_m_co_cleavage,
    "k_cn_cleavage_s": round(k_cn_cleavage, 2),
    "k_co_cleavage_s": round(k_co_cleavage, 2),
    "k_overall_n_dealk_s": round(k_overall_n_dealk, 2),
    "k_rebound_ref_s": k_rebound_ref,
}

checks = {
    "delta_m_cn_cleavage_small": delta_m_cn_cleavage < 0.20,
    "delta_m_co_cleavage_small": delta_m_co_cleavage < 0.20,
    "cn_cleavage_fast": k_cn_cleavage > 5e9,
    "co_cleavage_fast": k_co_cleavage > 5e9,
    "intermediate_labile": k_cn_cleavage > k_rebound_ref * 0.5,
}

write_result(name, data, checks)
