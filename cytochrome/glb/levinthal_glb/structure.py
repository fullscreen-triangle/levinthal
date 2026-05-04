"""
Structural primitives derived from GLB-parsed atom lists:
  - Atom (alias for GLBAtom)
  - Structure (alias for GLBStructure)
  - distance_matrix
  - contact_map_from_atoms
  - bond_inference (simple distance-based bond detection)
  - centre_of_mass
  - radius_of_gyration
"""

from __future__ import annotations

import numpy as np

from levinthal_glb.parser import GLBAtom as Atom, GLBStructure as Structure
from levinthal_glb.cpk import element_radius_A


def distance_matrix(structure: Structure) -> np.ndarray:
    """Pairwise atom-atom Euclidean distances in angstroms."""
    P = structure.positions
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(P[i] - P[j]))
            D[i, j] = d
            D[j, i] = d
    return D


def contact_map_from_atoms(structure: Structure, cutoff_A: float = 5.0) -> np.ndarray:
    """
    Binary contact matrix between atoms within cutoff_A angstroms.
    Diagonal excluded (an atom never 'contacts' itself).
    """
    D = distance_matrix(structure)
    n = D.shape[0]
    cm = ((D > 0) & (D < cutoff_A)).astype(int)
    return cm


def bond_inference(structure: Structure, scale: float = 0.6) -> list[tuple[int, int, float]]:
    """
    Simple distance-based bond detection: a bond exists between atoms i and j
    if the distance is less than scale * (vdW_i + vdW_j).

    Returns list of (i, j, distance_A) tuples.
    """
    bonds: list[tuple[int, int, float]] = []
    P = structure.positions
    n = len(P)
    for i in range(n):
        ri = element_radius_A(structure.atoms[i].element)
        for j in range(i + 1, n):
            rj = element_radius_A(structure.atoms[j].element)
            d = float(np.linalg.norm(P[i] - P[j]))
            if 0.5 < d < scale * (ri + rj):
                bonds.append((i, j, d))
    return bonds


def centre_of_mass(structure: Structure) -> np.ndarray:
    """Geometric centre of all atomic positions (no mass weighting)."""
    if structure.n_atoms == 0:
        return np.zeros(3)
    return structure.positions.mean(axis=0)


def radius_of_gyration(structure: Structure) -> float:
    """Radius of gyration about the centre of mass."""
    com = centre_of_mass(structure)
    P = structure.positions
    if len(P) == 0:
        return 0.0
    return float(np.sqrt(((P - com) ** 2).sum(axis=1).mean()))


def element_composition(structure: Structure) -> dict[str, int]:
    """Count of atoms by element."""
    counts: dict[str, int] = {}
    for a in structure.atoms:
        counts[a.element] = counts.get(a.element, 0) + 1
    return counts


def find_iron(structure: Structure) -> int | None:
    """Return index of the Fe atom (heme iron). Returns None if not found.
    If multiple Fe atoms exist, returns the one closest to the structure centre."""
    fe_indices = [i for i, a in enumerate(structure.atoms) if a.element == "Fe"]
    if not fe_indices:
        return None
    if len(fe_indices) == 1:
        return fe_indices[0]
    com = centre_of_mass(structure)
    best = fe_indices[0]
    best_d = float(np.linalg.norm(structure.atoms[best].position - com))
    for i in fe_indices[1:]:
        d = float(np.linalg.norm(structure.atoms[i].position - com))
        if d < best_d:
            best = i
            best_d = d
    return best


def neighbours_of(structure: Structure, atom_idx: int, cutoff_A: float = 4.0) -> list[int]:
    """Indices of atoms within cutoff of the given atom (excluding self)."""
    P = structure.positions
    centre = P[atom_idx]
    out: list[int] = []
    for j, p in enumerate(P):
        if j == atom_idx:
            continue
        d = float(np.linalg.norm(p - centre))
        if d < cutoff_A:
            out.append(j)
    return out
