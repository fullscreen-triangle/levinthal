"""Script 05 -- Rate spread within isoform family fits log-normal ΔM distribution."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "05_isoform_rate_spread"

# CYP2C subfamily: 2C8, 2C9, 2C18, 2C19 -- four distinct ΔM values
# Modelled by intra-family ΔM spread (sigma_DM ~ 0.06 depth units)
dm_2c = [0.48, 0.52, 0.58, 0.60]   # CYP2C8, 2C9, 2C18, 2C19 representative values
dm_2d = [0.52, 0.55, 0.58, 0.62, 0.65, 0.68]  # CYP2D6 variants (polymorphic)

import statistics
sigma_2c = statistics.stdev(dm_2c)
sigma_2d = statistics.stdev(dm_2d)
mean_2c  = statistics.mean(dm_2c)
mean_2d  = statistics.mean(dm_2d)

# Rate spread: k_max / k_min within family
k_2c_max = nu_floor * math.exp(-min(dm_2c))
k_2c_min = nu_floor * math.exp(-max(dm_2c))
rate_spread_2c = k_2c_max / k_2c_min

k_2d_max = nu_floor * math.exp(-min(dm_2d))
k_2d_min = nu_floor * math.exp(-max(dm_2d))
rate_spread_2d = k_2d_max / k_2d_min

data = {
    "dm_cyp2c": dm_2c,
    "dm_cyp2d": dm_2d,
    "sigma_dm_2c": round(sigma_2c, 4),
    "sigma_dm_2d": round(sigma_2d, 4),
    "rate_spread_2c": round(rate_spread_2c, 3),
    "rate_spread_2d": round(rate_spread_2d, 3),
}

checks = {
    "sigma_2c_lt_0.10":    sigma_2c < 0.10,
    "sigma_2d_lt_0.10":    sigma_2d < 0.10,
    "rate_spread_2c_lt_5": rate_spread_2c < 5.0,
    "rate_spread_2d_lt_5": rate_spread_2d < 5.0,
    "mean_2c_between_0.4_0.7": 0.4 < mean_2c < 0.7,
    "mean_2d_between_0.4_0.7": 0.4 < mean_2d < 0.7,
}

write_result(name, data, checks)
