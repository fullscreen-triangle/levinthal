"""
CPK color mapping (JMol/Rasmol convention).

Maps RGB colors found in GLB primitives to chemical elements.
Tolerant matching: any color within Euclidean distance < 30 of a CPK
reference colour is assigned the corresponding element.
"""

from __future__ import annotations

import math

# Standard JMol CPK colors (RGB 0-255)
CPK_COLORS: dict[str, tuple[int, int, int]] = {
    "H":  (255, 255, 255),
    "He": (217, 255, 255),
    "Li": (204, 128, 255),
    "Be": (194, 255,   0),
    "B":  (255, 181, 181),
    "C":  (144, 144, 144),
    "N":  ( 48,  80, 248),
    "O":  (255,  13,  13),
    "F":  (144, 224,  80),
    "Ne": (179, 227, 245),
    "Na": (171,  92, 242),
    "Mg": (138, 255,   0),
    "Al": (191, 166, 166),
    "Si": (240, 200, 160),
    "P":  (255, 128,   0),
    "S":  (255, 255,  48),
    "Cl": ( 31, 240,  31),
    "Ar": (128, 209, 227),
    "K":  (143,  64, 212),
    "Ca": ( 61, 255,   0),
    "Fe": (224, 102,  51),
    "Cu": (200, 128,  51),
    "Zn": (125, 128, 176),
    "Br": (165,  42,  42),
    "I":  (148,   0, 148),
}

# Common alternative shades sometimes used in GLB exports
CPK_ALIASES: dict[tuple[int, int, int], str] = {
    (211, 211, 211): "H",   # light grey alternate H
    (210, 180, 140): "P",   # tan, sometimes used for phosphate
    (147, 112, 219): "X",   # custom "ligand" colour (substrate atoms)
    (180,  90,  20): "Fe",  # darker orange variant
}


def cpk_color_to_element(rgb: tuple[int, int, int] | None,
                          tolerance: float = 30.0) -> str:
    """
    Map RGB color to chemical element symbol.

    Returns "?" if unmatched (color too far from any CPK reference).
    Returns "X" for the custom "ligand" violet (147, 112, 219) that some
    visualisations use for highlighted substrate atoms.
    """
    if rgb is None:
        return "?"

    rgb = tuple(int(c) for c in rgb[:3])

    # Direct alias match
    if rgb in CPK_ALIASES:
        return CPK_ALIASES[rgb]

    # Nearest-neighbour CPK match
    best_element = "?"
    best_distance = tolerance
    for element, ref in CPK_COLORS.items():
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, ref)))
        if d < best_distance:
            best_distance = d
            best_element = element
    return best_element


# van der Waals radii in angstroms (Bondi 1964 + element-specific updates)
ELEMENT_VDW_RADIUS_A: dict[str, float] = {
    "H":  1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B":  1.92,
    "C":  1.70, "N":  1.55, "O":  1.52, "F":  1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P":  1.80,
    "S":  1.80, "Cl": 1.75, "Ar": 1.88, "K":  2.75, "Ca": 2.31,
    "Fe": 1.94, "Cu": 1.40, "Zn": 1.39, "Br": 1.85, "I":  1.98,
    "X":  1.70,  # custom ligand atom default (treat as carbon-like)
    "?":  1.70,  # unknown
}


def element_radius_A(element: str) -> float:
    """Return van der Waals radius in angstroms."""
    return ELEMENT_VDW_RADIUS_A.get(element, 1.70)


# Element-to-S-entropy mapping (calibrated to atomic spectroscopic-derivation
# paper's element coordinates; values in [0,1]^3)
ELEMENT_S_ENTROPY: dict[str, tuple[float, float, float]] = {
    # (Sk = electronegativity / hydrophobicity proxy,
    #  St = atomic radius proxy,
    #  Se = electrostatic/ionisation proxy)
    "H":  (0.50, 0.20, 0.30),
    "C":  (0.70, 0.55, 0.30),
    "N":  (0.30, 0.50, 0.70),
    "O":  (0.20, 0.45, 0.85),
    "F":  (0.10, 0.40, 0.95),
    "P":  (0.45, 0.65, 0.45),
    "S":  (0.60, 0.65, 0.40),
    "Cl": (0.20, 0.55, 0.80),
    "Fe": (0.78, 0.55, 0.50),
    "Mg": (0.40, 0.60, 0.30),
    "Cu": (0.78, 0.55, 0.50),
    "Zn": (0.60, 0.55, 0.40),
    "X":  (0.70, 0.55, 0.30),  # ligand default
    "?":  (0.50, 0.50, 0.50),
}
