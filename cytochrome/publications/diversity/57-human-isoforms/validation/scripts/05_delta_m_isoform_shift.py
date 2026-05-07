"""05_delta_m_isoform_shift: k_2D6/k_3A4 = exp(-0.08) in [0.85, 0.97]."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result, nu_floor

# CYP2D6 shift: +0.08 relative to CYP3A4
delta_m_2D6 = 0.08
k_ratio_2D6 = math.exp(-delta_m_2D6)   # = exp(-0.08) = 0.9231

# CYP2C9 shift: +0.05 relative to CYP3A4
delta_m_2C9 = 0.05
k_ratio_2C9 = math.exp(-delta_m_2C9)   # = exp(-0.05) = 0.9512

print(f"CYP2D6: Delta_M = +{delta_m_2D6}, k_2D6/k_3A4 = exp(-{delta_m_2D6}) = {k_ratio_2D6:.4f}")
print(f"CYP2C9: Delta_M = +{delta_m_2C9}, k_2C9/k_3A4 = exp(-{delta_m_2C9}) = {k_ratio_2C9:.4f}")

checks = {
    "k_2D6_k_3A4_in_0.85_0.97": 0.85 <= k_ratio_2D6 <= 0.97,
    "k_2C9_k_3A4_in_0.85_0.99": 0.85 <= k_ratio_2C9 <= 0.99,
    "k_2D6_lt_k_2C9": k_ratio_2D6 < k_ratio_2C9,
    "k_2D6_gt_0.90": k_ratio_2D6 > 0.90,
}

write_result("05_delta_m_isoform_shift", {
    "delta_m_2D6": delta_m_2D6,
    "delta_m_2C9": delta_m_2C9,
    "k_ratio_2D6_over_3A4": round(k_ratio_2D6, 4),
    "k_ratio_2C9_over_3A4": round(k_ratio_2C9, 4),
}, checks)
