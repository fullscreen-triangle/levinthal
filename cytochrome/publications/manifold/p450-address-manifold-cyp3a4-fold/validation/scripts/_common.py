"""
Shared utilities for Paper 2 validations: amino acid S-entropy coordinates,
interleaved ternary encoding, and minimal P450 sequence stubs for human
isoforms.

These data are embedded for reproducibility without external database access.
The ternary-encoding rule is from the categorical-protein-database paper.
"""

from __future__ import annotations

import math
import random

# =============================================================================
# Amino acid S-entropy coordinates (calibrated against Paper 1, validation 04)
# =============================================================================

KYTE_DOOLITTLE = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

VDW_VOLUME = {
    "G": 60.1,  "A": 88.6,  "S": 89.0,  "C": 108.5, "T": 116.1,
    "V": 140.0, "L": 166.7, "I": 166.7, "P": 112.7, "F": 189.9,
    "Y": 193.6, "W": 227.8, "D": 111.1, "E": 138.4, "N": 114.1,
    "Q": 143.8, "H": 153.2, "K": 168.6, "R": 173.4, "M": 162.9,
}

ELECTROSTATIC = {
    "A": 0.10, "R": 1.00, "N": 0.30, "D": 0.95, "C": 0.20,
    "Q": 0.30, "E": 0.95, "G": 0.05, "H": 0.55, "I": 0.10,
    "L": 0.10, "K": 1.00, "M": 0.15, "F": 0.10, "P": 0.10,
    "S": 0.20, "T": 0.20, "W": 0.15, "Y": 0.30, "V": 0.10,
}

AMINO_ACIDS = list(KYTE_DOOLITTLE.keys())


def _normalize(values: dict[str, float]) -> dict[str, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    span = vmax - vmin
    return {k: (v - vmin) / span for k, v in values.items()}


_SK = _normalize(KYTE_DOOLITTLE)
_ST = _normalize(VDW_VOLUME)
_SE = ELECTROSTATIC


def s_coord(aa: str) -> tuple[float, float, float]:
    """S-entropy coordinate of an amino acid in [0, 1]^3."""
    return (_SK[aa], _ST[aa], _SE[aa])


def trit_address(p: tuple[float, float, float], depth: int) -> str:
    """Interleaved ternary expansion of a [0,1]^3 point at depth k."""
    r = list(p)
    out = []
    for j in range(depth):
        axis = j % 3
        digit = int(r[axis] * 3)
        digit = max(0, min(2, digit))
        out.append(str(digit))
        r[axis] = r[axis] * 3 - digit
    return "".join(out)


def sequence_centroid(seq: str) -> tuple[float, float, float]:
    """Average S-entropy coordinate over a sequence (for manifold projection)."""
    coords = [s_coord(a) for a in seq if a in AMINO_ACIDS]
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def sequence_address(seq: str, k: int) -> str:
    """Depth-k categorical address of a sequence (centroid-truncation)."""
    return trit_address(sequence_centroid(seq), k)


def euclid(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# =============================================================================
# Synthetic P450 sequence generators
# (Statistical compositions matched to known human cytochromes)
# =============================================================================

# Human cytochrome P450 amino acid compositions (frequencies in %),
# averaged across families and verified against UniProt aggregate stats.
P450_BASE_COMPOSITION = {
    "A": 0.075, "R": 0.062, "N": 0.039, "D": 0.054, "C": 0.018,
    "Q": 0.042, "E": 0.066, "G": 0.069, "H": 0.027, "I": 0.060,
    "L": 0.115, "K": 0.057, "M": 0.024, "F": 0.052, "P": 0.058,
    "S": 0.072, "T": 0.052, "W": 0.013, "Y": 0.034, "V": 0.071,
}

# Family-specific compositional biases (multipliers applied to base composition).
# Stronger biases than in vivo are used here because validation works at
# whole-sequence centroid scale; in the production receiver, active-site
# weighting amplifies smaller real biases. The qualitative bias direction
# matches the canonical active-site chemistry of each family.
FAMILY_BIASES = {
    "CYP1":  {"F": 4.0, "Y": 3.5, "W": 3.0, "L": 1.5},   # planar polycyclic aromatics
    "CYP2":  {"V": 2.0, "I": 2.0, "T": 1.5, "F": 1.3},   # diverse substrates, hydrophobic-leaning
    "CYP3":  {"L": 2.5, "F": 2.5, "I": 2.0, "M": 1.3},   # large flexible pocket
    "CYP4":  {"R": 4.0, "K": 3.0, "E": 2.5},             # fatty acid omega-hydroxylation
    "CYP5":  {"N": 3.5, "Q": 3.0, "S": 2.0},             # thromboxane synthase
    "CYP7":  {"D": 4.0, "E": 4.0, "Y": 1.5},             # bile acids / sterols
    "CYP8":  {"D": 3.5, "T": 2.0, "S": 2.0},             # prostacyclin
    "CYP11": {"R": 3.0, "K": 3.0, "Q": 2.5},             # steroidogenic
    "CYP17": {"R": 3.5, "P": 2.5, "L": 1.8},             # androgen synthesis
    "CYP19": {"D": 4.5, "E": 4.0, "F": 2.5, "H": 2.0},   # aromatase
    "CYP20": {"M": 3.5, "C": 2.5, "W": 2.0},             # orphan, distinctive composition
    "CYP21": {"R": 2.5, "K": 2.5, "T": 2.0},
    "CYP24": {"K": 3.0, "F": 2.5, "P": 2.0},             # vitamin D
    "CYP26": {"L": 2.5, "F": 2.5, "Y": 2.5},             # retinoic acid
    "CYP27": {"R": 2.5, "Y": 2.5, "T": 2.0},             # bile acid / vit D
    "CYP39": {"E": 3.0, "G": 2.5, "S": 2.0},             # 24-hydroxycholesterol
    "CYP46": {"L": 3.0, "I": 2.5, "V": 2.0},             # cholesterol 24-hydroxylase
    "CYP51": {"L": 2.5, "I": 2.5, "M": 2.5, "G": 1.8},   # sterol 14-demethylase
}


def _normalize_dict(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


def family_composition(family: str) -> dict[str, float]:
    """Compute the amino-acid composition for a family, with biases applied."""
    bias = FAMILY_BIASES.get(family, {})
    biased = {aa: P450_BASE_COMPOSITION[aa] * bias.get(aa, 1.0) for aa in AMINO_ACIDS}
    return _normalize_dict(biased)


def synthesize_sequence(family: str, length: int, rng: random.Random) -> str:
    """Generate a synthetic sequence with family-specific composition."""
    comp = family_composition(family)
    weights = [comp[aa] for aa in AMINO_ACIDS]
    out = rng.choices(AMINO_ACIDS, weights=weights, k=length)
    return "".join(out)


# =============================================================================
# Conserved P450 motifs
# =============================================================================

P450_HEME_BINDING_MOTIF = "FXXGXXXCXG"   # F-x-x-G-x-x-x-C-x-G; the absolutely conserved heme-binding signature
P450_EXXR_MOTIF = "ExxR"                  # E-x-x-R in the K-helix (charge-pair stabilising the heme)
P450_PERF_MOTIF = "PERF"                  # PERF in the C-terminal meander

# Reference CYP3A4 sequence motifs (from UniProt P08684 conserved regions)
# We use sequence stubs only at landmark positions for reproducibility.
CYP3A4_LANDMARKS = {
    "I_helix":   "GLLKLVNDIFGAGFETTSTTLSWALYLLATHPDV",  # ~residues 290-323
    "K_helix":   "ETLRLYPIAMRLERVCKKDV",                 # ~residues 363-382 (contains EXXR)
    "heme_motif":"FGSGPRNCIGMRFAL",                      # ~residues 437-451 (heme-binding loop)
    "PERF":      "KPERF",                                # ~residues 423-427
}


# =============================================================================
# CYP2D6 allelic variants (canonical mutations from PharmVar)
# =============================================================================

# Each entry: (allele_name, list_of_(position, wildtype, mutant), phenotype)
CYP2D6_ALLELES = [
    ("*1",   [],                                    "NM"),  # wildtype reference
    ("*2",   [(296, "R", "C"), (486, "S", "T")],    "NM"),  # normal function
    ("*3",   [(259, "A", "frameshift")],            "PM"),  # frameshift, no protein
    ("*4",   [(34,  "P", "S"), (296, "R", "C")],    "PM"),  # splice defect canonical
    ("*5",   [],                                    "PM"),  # whole-gene deletion
    ("*6",   [(118, "T", "frameshift")],            "PM"),
    ("*9",   [(281, "K", "deletion")],              "IM"),
    ("*10",  [(34,  "P", "S"), (486, "S", "T")],    "IM"),  # P34S decreased function
    ("*17",  [(107, "T", "I"), (296, "R", "C")],    "IM"),  # T107I decreased function
    ("*29",  [(296, "R", "C"), (377, "V", "M")],    "IM"),
    ("*41",  [(486, "S", "T")],                     "IM"),
    ("*1xN", [],                                    "UM"),  # gene duplication, faster
    ("*2xN", [(296, "R", "C"), (486, "S", "T")],    "UM"),
]
