"""
Hydrogen Bond Network Detection and Coupling Matrix.

Generates H-bond connectivity for Kuramoto oscillator networks.
"""
import numpy as np
from typing import Dict, List, Tuple


def generate_hbond_positions(n_hbonds: int, protein_radius: float = 15.0,
                             seed: int = 42) -> np.ndarray:
    """Generate realistic H-bond midpoint positions within a protein."""
    rng = np.random.RandomState(seed)
    positions = rng.randn(n_hbonds, 3) * (protein_radius / 3.0)
    return positions


def compute_coupling_matrix(positions: np.ndarray, K0: float = 2.0,
                            r0: float = 5.0) -> np.ndarray:
    """Compute distance-dependent coupling: K_ij = K0·exp(-r_ij/r0)."""
    n = len(positions)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            k_ij = K0 * np.exp(-r_ij / r0)
            K[i, j] = k_ij
            K[j, i] = k_ij
    return K


def identify_regions(positions: np.ndarray, n_regions: int = 4,
                     seed: int = 42) -> Dict:
    """Assign H-bonds to structural regions using simple clustering."""
    rng = np.random.RandomState(seed)
    n = len(positions)

    # Simple region assignment based on position
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        # Use angular position to assign regions
        phi = np.arctan2(positions[i, 1], positions[i, 0])
        labels[i] = int((phi + np.pi) / (2 * np.pi) * n_regions) % n_regions

    regions = {}
    region_names = ['beta_barrel', 'dimer_interface', 'loop_region', 'cu_ligand']
    for r in range(n_regions):
        name = region_names[r] if r < len(region_names) else f'region_{r}'
        indices = np.where(labels == r)[0].tolist()
        regions[name] = indices

    return regions


def generate_sod1_network(seed: int = 42) -> Dict:
    """Generate SOD1-like H-bond network with 165 bonds."""
    n_hbonds = 165
    positions = generate_hbond_positions(n_hbonds, seed=seed)
    coupling = compute_coupling_matrix(positions)
    regions = identify_regions(positions, seed=seed)

    return {
        'n_hbonds': n_hbonds,
        'positions': positions,
        'coupling_matrix': coupling,
        'regions': regions,
    }


def generate_loop_network(n_hbonds: int = 8, seed: int = 42) -> Dict:
    """Generate a small loop sub-network (for conformational dynamics)."""
    rng = np.random.RandomState(seed)
    # Loop H-bonds are roughly linear
    positions = np.zeros((n_hbonds, 3))
    for i in range(n_hbonds):
        positions[i] = [i * 2.0, rng.randn() * 0.5, rng.randn() * 0.5]

    coupling = compute_coupling_matrix(positions, K0=2.0, r0=5.0)

    return {
        'n_hbonds': n_hbonds,
        'positions': positions,
        'coupling_matrix': coupling,
    }
