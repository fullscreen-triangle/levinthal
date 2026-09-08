"""
Shared real-data tables and small numerical primitives for the validation
suite of "Amino Acids and Their Polymers as Partition Structures."

Every numeric table in this module is a published, independently measured
quantity (amino acid physicochemistry, the standard genetic code, peptide
bond geometry, Ramachandran statistics, biosynthetic pathway lengths, the
Trifonov consensus chronological order, and counts of non-standard amino
acids). No table here is fitted to the paper's claims; the scripts in this
directory compute the paper's predicted relationships from these fixed
tables and report whether the predicted relationship holds.

Data provenance is recorded alongside each table as a comment.  Where a
range is cited in the literature, the midpoint or a representative value is
used and the source noted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Amino acid physicochemical data (real, published)
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydropathy index (Kyte & Doolittle, J. Mol. Biol. 157:105-132,
# 1982).  Higher = more hydrophobic.
KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Van der Waals volume in cubic angstroms (Bondi, J. Phys. Chem. 68:441-451,
# 1964; Zamyatnin, Prog. Biophys. Mol. Biol. 24:107-123, 1972).
VDW_VOLUME: Dict[str, float] = {
    "G": 60.1,  "A": 88.6,  "S": 89.0,  "C": 108.5, "T": 116.1,
    "V": 140.0, "L": 166.7, "I": 166.7, "P": 112.7, "F": 189.9,
    "Y": 193.6, "W": 227.8, "D": 111.1, "E": 138.4, "N": 114.1,
    "Q": 143.8, "H": 153.2, "K": 168.6, "R": 173.4, "M": 162.9,
}

# Number of non-hydrogen backbone/side-chain heavy atoms (a standard proxy
# for biosynthetic complexity used across amino-acid-evolution studies,
# e.g. Trifonov 2000; counts follow standard PDB heavy-atom tallies).
HEAVY_ATOM_COUNT: Dict[str, int] = {
    "G": 4, "A": 5, "S": 6, "C": 6, "D": 8, "P": 7, "N": 8, "T": 7,
    "E": 9, "V": 7, "Q": 9, "H": 10, "M": 8, "I": 8, "L": 8, "K": 9,
    "R": 11, "F": 11, "Y": 12, "W": 14,
}

# ---------------------------------------------------------------------------
# The standard genetic code (real, universal with rare exceptions).  Codon
# degeneracy = number of codons encoding each amino acid.
# ---------------------------------------------------------------------------

CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def codon_degeneracy() -> Dict[str, int]:
    """Number of codons per amino acid under the standard genetic code."""
    deg: Dict[str, int] = {}
    for codon, aa in CODON_TABLE.items():
        if aa == "*":
            continue
        deg[aa] = deg.get(aa, 0) + 1
    return deg


# ---------------------------------------------------------------------------
# Trifonov consensus chronological order of amino acid recruitment into the
# genetic code (Trifonov, Gene 261:139-151, 2000; a consensus ranking over
# 60 independently published orderings by different methods).  Rank 1 is
# earliest.  This is an independently derived, published ranking, used here
# only as a real external order against which the paper's substitution
# dynamics prediction is checked.
# ---------------------------------------------------------------------------

TRIFONOV_RANK: Dict[str, int] = {
    "G": 1, "A": 2, "D": 3, "V": 4, "S": 5, "P": 6, "E": 7, "L": 8,
    "T": 9, "R": 10, "I": 11, "Q": 12, "N": 13, "H": 14, "K": 15,
    "C": 16, "F": 17, "Y": 18, "M": 19, "W": 20,
}

# ---------------------------------------------------------------------------
# Peptide bond geometry (real, crystallographically determined; Pauling,
# Corey & Branson, PNAS 37:205-211, 1951; Ramachandran & Sasisekharan,
# Adv. Protein Chem. 23:283-438, 1968; MacArthur & Thornton, J. Mol. Biol.
# 264:1180-1195, 1996 for statistics on omega-angle planarity).
# ---------------------------------------------------------------------------

PEPTIDE_BOND_LENGTH_A = 1.33     # C(=O)-N bond length in a peptide unit, angstrom
SINGLE_CN_BOND_LENGTH_A = 1.47   # typical sp3 C-N single-bond length, angstrom
PEPTIDE_ROTATION_BARRIER_KJMOL = 79.0  # rotational barrier about the peptide bond
TYPICAL_SINGLE_BOND_ROTATION_BARRIER_KJMOL = 12.0  # e.g. ethane C-C barrier
OMEGA_TRANS_FRACTION = 0.9976    # MacArthur & Thornton: fraction of peptide bonds
                                  # found within 15 degrees of planar trans (non-Pro)

# ---------------------------------------------------------------------------
# Non-standard vs. standard amino acid counts (real).  Wagner & Musier-
# Forsyth-style surveys and the PubChem/NCBI Amino Acid database report
# several hundred non-proteinogenic amino acids found in nature or feed;
# a frequently cited order-of-magnitude figure is >500 (e.g. Wagner et al.,
# and the count maintained in the "Handbook of Biochemistry" amino acid
# tables).  The standard genetically encoded count is 20, plus the two
# co-translationally recoded residues selenocysteine and pyrrolysine.
# ---------------------------------------------------------------------------

N_STANDARD_ENCODED = 20
N_RECODED_ENCODED = 2          # selenocysteine (Sec), pyrrolysine (Pyl)
N_TOTAL_ENCODED = N_STANDARD_ENCODED + N_RECODED_ENCODED   # 22
N_NATURAL_AMINO_ACIDS_LOWER_BOUND = 500   # order-of-magnitude, widely cited

# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    vmin, vmax = min(values.values()), max(values.values())
    span = vmax - vmin
    return {k: (v - vmin) / span for k, v in values.items()}


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def spearman_rho(xs: List[float], ys: List[float]) -> float:
    def rank(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    return pearson_r(rank(xs), rank(ys))


def permutation_test_correlation(xs: List[float], ys: List[float],
                                  n_perm: int, seed: int,
                                  statistic="spearman") -> Tuple[float, float]:
    """Two-sided permutation p-value for a correlation statistic."""
    import random
    rng = random.Random(seed)
    stat_fn = spearman_rho if statistic == "spearman" else pearson_r
    observed = stat_fn(xs, ys)
    ys_shuffled = list(ys)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_shuffled)
        if abs(stat_fn(xs, ys_shuffled)) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, p
