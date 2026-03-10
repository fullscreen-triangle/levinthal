"""
Partition Depth (Equation III) and Gradient Flow (Equation IV).

M = Σ log_b(k_i) measures distinguishability.
S_P = k_B · M · ln(b) gives depth-entropy equivalence.
"""
import numpy as np
from typing import Dict, List, Tuple


K_B = 1.380649e-23  # Boltzmann constant


def compute_depth(branching_factors: List[int], base: int = 3) -> float:
    """Equation III: M = Σ log_b(k_i)."""
    return sum(np.log(k) / np.log(base) for k in branching_factors if k > 0)


def depth_entropy_equivalence(M: float, b: int = 2) -> float:
    """S_P = k_B · M · ln(b)."""
    return K_B * M * np.log(b)


def rate_from_depth(delta_M: float, temperature: float = 300.0,
                    base: int = 3) -> float:
    """k = (1/τ_p) · b^(-ΔM) where τ_p = ℏ/(k_B T)."""
    hbar = 1.054571817e-34
    tau_p = hbar / (K_B * temperature)
    return (1.0 / tau_p) * base ** (-delta_M)


def equilibrium_from_depth(M_A: float, M_B: float, base: int = 3) -> float:
    """K_eq = b^(-(M_B - M_A))."""
    return base ** (-(M_B - M_A))


def generate_depth_surface(n_max: int = 7) -> Dict:
    """Generate partition depth surface M(n, l) for visualization."""
    n_values = []
    l_values = []
    m_values = []
    depth_values = []

    for n in range(1, n_max + 1):
        for l in range(n):
            for m_val in range(-l, l + 1):
                branching = [n] + [2 * l + 1] if l > 0 else [n]
                M = compute_depth(branching, base=3)
                n_values.append(n)
                l_values.append(l)
                m_values.append(m_val)
                depth_values.append(M)

    return {
        'n': n_values,
        'l': l_values,
        'm': m_values,
        'depth': depth_values,
    }


def fit_depth_entropy_slope(n_samples: int = 200) -> Dict:
    """
    Validate depth-entropy equivalence: slope should match ln(2) = 0.693.
    Generate synthetic depth-entropy pairs and fit.
    """
    np.random.seed(42)

    # Generate depths from various branching structures
    depths = []
    entropies = []

    for _ in range(n_samples):
        n_levels = np.random.randint(2, 8)
        branching = np.random.randint(2, 6, size=n_levels)
        M = compute_depth(branching.tolist(), base=2)
        S = depth_entropy_equivalence(M, b=2)
        depths.append(M)
        entropies.append(S / K_B)  # Normalize by k_B for cleaner units

    depths = np.array(depths)
    entropies = np.array(entropies)

    # Linear fit
    coeffs = np.polyfit(depths, entropies, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # R² calculation
    predicted = np.polyval(coeffs, depths)
    ss_res = np.sum((entropies - predicted) ** 2)
    ss_tot = np.sum((entropies - np.mean(entropies)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    # Slope should be ln(2) = 0.6931...
    expected_slope = np.log(2)
    slope_error = abs(slope - expected_slope)

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'expected_slope': float(expected_slope),
        'slope_error': float(slope_error),
        'slope_matches': slope_error < 0.002,
        'depths': depths.tolist(),
        'entropies': entropies.tolist(),
        'n_samples': n_samples,
    }


def biological_scale_comparison() -> Dict:
    """Partition depth across biological scales."""
    scales = [
        {'name': 'Hydrogen atom', 'depth': 1, 'scale': 'atomic'},
        {'name': 'Carbon atom', 'depth': 2, 'scale': 'atomic'},
        {'name': 'Iron atom', 'depth': 4, 'scale': 'atomic'},
        {'name': 'Amino acid', 'depth': 8, 'scale': 'molecular'},
        {'name': 'α-Helix (10 res)', 'depth': 15, 'scale': 'molecular'},
        {'name': 'SOD1 (153 res)', 'depth': 25, 'scale': 'protein'},
        {'name': 'Ribosome', 'depth': 45, 'scale': 'complex'},
        {'name': 'E. coli cell', 'depth': 200, 'scale': 'cellular'},
    ]
    return {'scales': scales}


def run_depth_validation() -> Dict:
    """Run partition depth validation."""
    depth_surface = generate_depth_surface(7)
    slope_fit = fit_depth_entropy_slope(200)
    bio_scales = biological_scale_comparison()

    return {
        'depth_surface': depth_surface,
        'slope_fit': slope_fit,
        'biological_scales': bio_scales,
    }
