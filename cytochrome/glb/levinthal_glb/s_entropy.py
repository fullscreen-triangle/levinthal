"""
Atomic centroids -> S-entropy coordinates and addresses.

Maps a GLB-parsed Structure to S-entropy coordinates suitable for input to
the receiver R_bio. Two layers of mapping:

  1. Per-atom: element -> S-coord triple (from CPK / atomic data)
  2. Whole-structure: centroid + composition -> structure-level S-coord
"""

from __future__ import annotations

import math

import numpy as np

from levinthal_glb.cpk import ELEMENT_S_ENTROPY
from levinthal_glb.parser import GLBStructure as Structure


def atom_to_s_entropy(element: str) -> tuple[float, float, float]:
    """Map element symbol to S-entropy coordinate."""
    return ELEMENT_S_ENTROPY.get(element, ELEMENT_S_ENTROPY["?"])


def structure_centroid_s_entropy(structure: Structure) -> tuple[float, float, float]:
    """Average S-entropy coordinate across all atoms in the structure."""
    if structure.n_atoms == 0:
        return (0.5, 0.5, 0.5)
    coords = [atom_to_s_entropy(a.element) for a in structure.atoms]
    sk = sum(c[0] for c in coords) / len(coords)
    st = sum(c[1] for c in coords) / len(coords)
    se = sum(c[2] for c in coords) / len(coords)
    return (sk, st, se)


def trit_address(s: tuple[float, float, float], depth: int = 9) -> str:
    """
    Interleaved-ternary address (Paper 1, Eq. 1).

    For each trit position j in [0, depth), refines axis (j mod 3) by
    one ternary bit. Returns string of length `depth`.
    """
    r = list(s)
    out = []
    for j in range(depth):
        axis = j % 3
        digit = int(r[axis] * 3)
        digit = max(0, min(2, digit))
        out.append(str(digit))
        r[axis] = r[axis] * 3 - digit
    return "".join(out)


def structure_to_address(structure: Structure, depth: int = 9) -> str:
    """Convenience: structure -> centroid -> trit address."""
    return trit_address(structure_centroid_s_entropy(structure), depth)


def F_CB(s: tuple[float, float, float], regularize: bool = True) -> dict:
    """Closed-form F_CB partition-coordinate map (Paper 1, Construction 6.2)."""
    Sk, St, Se = s
    norm = math.sqrt(Sk * Sk + St * St + Se * Se)
    if regularize and norm >= 1.0:
        epsilon = math.exp(-7.13)
        norm_clipped = min(norm, 1.0 - epsilon)
    else:
        norm_clipped = norm
    if norm_clipped >= 1.0:
        return {"M": float("inf"), "n": -1, "l": -1, "norm": norm}
    M = -math.log(1.0 - norm_clipped)
    n = max(1, int(math.ceil(math.sqrt(3.0 * M))))
    if norm > 1e-12:
        cos_angle = max(-1.0, min(1.0, Se / norm))
        l = max(0, min(n - 1,
                       int(math.floor((n - 1) * math.acos(cos_angle) / math.pi))))
    else:
        l = 0
    return {"M": M, "n": n, "l": l, "norm": norm}


def per_atom_addresses(structure: Structure, depth: int = 9) -> list[str]:
    """Per-atom trit address (each atom contributes its own element-derived trit)."""
    return [trit_address(atom_to_s_entropy(a.element), depth) for a in structure.atoms]
