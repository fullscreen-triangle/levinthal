"""
Ternary Trisection (supports Equation VII).

Position = Trajectory = Proof via ternary encoding.
Resolution: Δx = L/3^k after k trisection steps.
"""
import numpy as np
from typing import List, Tuple, Dict


def trisection_step(region_min: float, region_max: float,
                    trit: int) -> Tuple[float, float]:
    """Perform one trisection step, selecting sub-region by trit value."""
    L = region_max - region_min
    third = L / 3.0
    if trit == 0:
        return region_min, region_min + third
    elif trit == 1:
        return region_min + third, region_min + 2 * third
    else:
        return region_min + 2 * third, region_max


def trisection_localize(L: float, k_iterations: int,
                        target: float = None) -> Dict:
    """
    Perform k iterations of ternary trisection on interval [0, L].
    If target is given, localize it; otherwise use random trits.
    """
    np.random.seed(42)
    region_min, region_max = 0.0, L
    trits = []
    resolutions = []
    positions = []

    for i in range(k_iterations):
        third = (region_max - region_min) / 3.0

        if target is not None:
            # Determine which third contains the target
            if target < region_min + third:
                trit = 0
            elif target < region_min + 2 * third:
                trit = 1
            else:
                trit = 2
        else:
            trit = np.random.randint(0, 3)

        region_min, region_max = trisection_step(region_min, region_max, trit)
        trits.append(trit)
        resolutions.append(region_max - region_min)
        positions.append((region_min + region_max) / 2.0)

    return {
        'trits': trits,
        'final_position': positions[-1] if positions else 0.0,
        'final_resolution': resolutions[-1] if resolutions else L,
        'resolutions': resolutions,
        'positions': positions,
        'ternary_string': ''.join(str(t) for t in trits),
        'n_iterations': k_iterations,
        'initial_size': L,
    }


def resolution_after_k(L: float, k: int) -> float:
    """Resolution after k trisection steps: Δx = L/3^k."""
    return L / (3.0 ** k)


def ternary_to_position(trit_sequence: List[int], L: float) -> float:
    """Convert trit sequence to position: x = Σ t_i · L/3^(i+1)."""
    x = 0.0
    for i, t in enumerate(trit_sequence):
        x += t * L / (3.0 ** (i + 1))
    return x


def speedup_vs_binary() -> Dict:
    """Compare ternary vs binary search efficiency."""
    # log_3(N) / log_2(N) = log(2) / log(3) = 0.631
    ratio = np.log(2) / np.log(3)
    speedup = 1.0 / ratio  # 1.585

    return {
        'ternary_log_ratio': float(ratio),
        'speedup_factor': float(speedup),
        'percent_improvement': float((1 - ratio) * 100),
    }
