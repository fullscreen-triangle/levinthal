"""
Receiver R_bio applied to GLB-derived structural data.

Implements the morphism chain access ∘ fuse ∘ catalyze* ∘ observe on
a parsed GLB structure, producing:
  - partition signatures (per-atom and pairwise)
  - contact predictions
  - S-entropy address
  - heme-iron neighbourhood characterisation
"""

from __future__ import annotations

import math

import numpy as np

from levinthal_glb.cpk import element_radius_A
from levinthal_glb.parser import GLBStructure as Structure
from levinthal_glb.s_entropy import (
    F_CB,
    atom_to_s_entropy,
    structure_centroid_s_entropy,
    trit_address,
)
from levinthal_glb.structure import (
    centre_of_mass,
    contact_map_from_atoms,
    distance_matrix,
    element_composition,
    find_iron,
    neighbours_of,
    radius_of_gyration,
)


class RbioGLBEvaluator:
    """
    Apply R_bio to a GLB-parsed Structure.

    Provides the four morphism-chain operations and convenience methods for
    extracting receiver-evaluation outputs.
    """

    def __init__(self, structure: Structure, K0: float = 1.0,
                 sigma_S: float = 0.30):
        self.structure = structure
        self.K0 = K0
        self.sigma_S = sigma_S
        self._distance_matrix: np.ndarray | None = None
        self._coupling_matrix: np.ndarray | None = None

    # ============================================================
    # Geometric properties (precondition for receiver evaluation)
    # ============================================================

    def distance_matrix(self) -> np.ndarray:
        if self._distance_matrix is None:
            self._distance_matrix = distance_matrix(self.structure)
        return self._distance_matrix

    def s_coords_per_atom(self) -> np.ndarray:
        return np.array([atom_to_s_entropy(a.element) for a in self.structure.atoms])

    def coupling_matrix(self) -> np.ndarray:
        """K_ij = K0 * exp(-d_S^2 / 2 sigma^2) * exp(-r_ij / r0)."""
        if self._coupling_matrix is not None:
            return self._coupling_matrix
        S = self.s_coords_per_atom()
        D = self.distance_matrix()
        n = len(S)
        K = np.zeros((n, n))
        r0 = 5.0  # Å
        for i in range(n):
            for j in range(i + 1, n):
                d_S = float(np.linalg.norm(S[i] - S[j]))
                k_ij = (self.K0
                        * math.exp(-d_S ** 2 / (2 * self.sigma_S ** 2))
                        * math.exp(-D[i, j] / r0))
                K[i, j] = k_ij
                K[j, i] = k_ij
        self._coupling_matrix = K
        return K

    # ============================================================
    # Morphism chain operations
    # ============================================================

    def observe(self) -> np.ndarray:
        """
        Pass 1 (observe): produce partition signature Σ.

        For each atom pair, computes a partition-coordinate-like value from
        the coupling magnitude. Returns N×N matrix.
        """
        K = self.coupling_matrix()
        return K.copy()

    def catalyze(self, sigma: np.ndarray, contact_cutoff_A: float = 4.0) -> np.ndarray:
        """
        Pass 2 (catalyze): apply contact-distance kernel.

        Boost signature entries for atom pairs within van-der-Waals contact
        distance (i.e., bonded or directly interacting).
        """
        D = self.distance_matrix()
        out = sigma.copy()
        n = sigma.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < contact_cutoff_A:
                    boost = 2.0 if D[i, j] < 2.5 else 1.5  # bond vs contact
                    out[i, j] *= boost
                    out[j, i] *= boost
        return out

    def fuse(self, *sigmas: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
        """Pass 3 (fuse): weighted average of multiple signatures."""
        if not sigmas:
            return np.zeros((0, 0))
        if weights is None:
            weights = [1.0 / len(sigmas)] * len(sigmas)
        result = np.zeros_like(sigmas[0])
        for s, w in zip(sigmas, weights):
            result = result + w * s
        return result

    def access_contact_map(self, sigma: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Pass 4 (access): threshold the signature into a binary contact map."""
        if threshold is None:
            threshold = float(sigma[sigma > 0].mean()) if (sigma > 0).any() else 0.0
        cm = (sigma > threshold).astype(int)
        np.fill_diagonal(cm, 0)
        return cm

    # ============================================================
    # Top-level convenience
    # ============================================================

    def evaluate(self) -> dict:
        """Run the full morphism chain and return all derived quantities."""
        composition = element_composition(self.structure)
        iron_idx = find_iron(self.structure)
        com = centre_of_mass(self.structure)
        rg = radius_of_gyration(self.structure)

        # Receiver evaluation
        sigma_observed = self.observe()
        sigma_catalyzed = self.catalyze(sigma_observed)
        sigma_fused = self.fuse(sigma_observed, sigma_catalyzed,
                                 weights=[0.5, 0.5])
        contact_map = self.access_contact_map(sigma_fused)

        # Whole-structure S-entropy
        s_centroid = structure_centroid_s_entropy(self.structure)
        partition = F_CB(s_centroid)
        address = trit_address(s_centroid, depth=9)

        # Heme iron neighbourhood
        iron_neighbours: list[dict] = []
        if iron_idx is not None:
            nbr_idxs = neighbours_of(self.structure, iron_idx, cutoff_A=3.0)
            for j in nbr_idxs:
                a = self.structure.atoms[j]
                d = float(np.linalg.norm(a.position - self.structure.atoms[iron_idx].position))
                iron_neighbours.append({
                    "index": j,
                    "element": a.element,
                    "distance_A": d,
                })

        return {
            "n_atoms": self.structure.n_atoms,
            "composition": composition,
            "centre_of_mass_A": com.tolist(),
            "radius_of_gyration_A": rg,
            "iron_atom_index": iron_idx,
            "iron_first_shell_neighbours": iron_neighbours,
            "structure_S_centroid": list(s_centroid),
            "partition_depth_M": partition["M"],
            "partition_n": partition["n"],
            "partition_l": partition["l"],
            "trit_address_depth9": address,
            "contact_map_n_contacts": int(contact_map.sum() // 2),
            "contact_map_density": float(
                contact_map.sum() / (contact_map.size - contact_map.shape[0])
            ) if contact_map.size > 0 else 0.0,
        }
