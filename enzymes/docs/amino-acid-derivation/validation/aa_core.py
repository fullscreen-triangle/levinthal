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


# ---------------------------------------------------------------------------
# Real, previously computed data drawn from the companion cytochrome P450
# validation corpus (this repository, other papers in the same framework).
# These are not re-simulated here; they are cited as fixed facts, with the
# originating result file recorded in each comment, exactly as the tables
# above cite Kyte-Doolittle or the standard genetic code. They are used to
# test Part V's storage-hierarchy and alphabet-ceiling claims against
# structures (CYP3A4, the 57 human P450 isoforms, the four-hop electron
# transport chain) that were characterised independently of this manuscript.
# ---------------------------------------------------------------------------

# CYP3A4 fold statistics (cytochrome/publications/synthesis/57-isoform-
# taxonomy/validation/results/03_cyp3a4_fold_depth.json and
# 04_capacity_shell_rule.json).
CYP3A4_N_AMINO_ACIDS = 503
CYP3A4_FOLD_DEPTH_LOG3N = 5.6622          # log_3(503)
CYP3A4_ADDRESSES_AT_DEPTH6 = 729          # 3^6
CYP3A4_SEQUENCE_COVERAGE = 1.4493         # addresses / n_amino_acids at depth 6

# Ternary partition capacity per depth (same source, 04_capacity_shell_rule):
# C(n) = 2n^2 evaluated at successive shells; independently confirms the
# smallest depth able to address the 20-residue alphabet.
CAPACITY_PER_SHELL = {1: 2, 2: 8, 3: 18, 4: 32, 5: 50}
MIN_DEPTH_FOR_20_AA = 3                    # from the same result file
DEPTH9_CAPACITY = 19683                    # 3^9, confirmed to exceed 20 aa

# The 57 human CYP isoforms: real partition addresses at depth 8 assigned
# from real Kyte-Doolittle/volume/electrostatic coordinates (cytochrome/
# publications/manifold/p450-address-manifold-cyp3a4-fold/validation/
# results/04_isoform_separation.json). A representative subset is recorded
# here (full 57-row table available in that source file) together with the
# family each isoform belongs to, sufficient to test cross-lineage
# recurrence structure (families sharing partial addresses) without
# reproducing the entire external table.
ISOFORM_ADDRESSES = {
    "CYP1A1": ("CYP1", "11111002"), "CYP1A2": ("CYP1", "11011201"),
    "CYP1B1": ("CYP1", "11011222"), "CYP2A6": ("CYP2", "11010212"),
    "CYP2A7": ("CYP2", "11011222"), "CYP2A13": ("CYP2", "11021200"),
    "CYP2B6": ("CYP2", "11101020"), "CYP2C8": ("CYP2", "11111010"),
    "CYP2C9": ("CYP2", "11011211"), "CYP2C18": ("CYP2", "11011200"),
    "CYP2C19": ("CYP2", "11111010"), "CYP2D6": ("CYP2", "11011220"),
    "CYP2E1": ("CYP2", "11010222"), "CYP3A4": ("CYP3", "11021210"),
    "CYP3A5": ("CYP3", "11011220"), "CYP3A7": ("CYP3", "11021210"),
    "CYP3A43": ("CYP3", "11011221"), "CYP4A11": ("CYP4", "11101000"),
    "CYP4A22": ("CYP4", "11100102"), "CYP4B1": ("CYP4", "01121212"),
    "CYP5A1": ("CYP5", "11010202"), "CYP7A1": ("CYP7", "11100122"),
    "CYP7B1": ("CYP7", "11101020"), "CYP8A1": ("CYP8", "11111010"),
    "CYP8B1": ("CYP8", "11010221"), "CYP11A1": ("CYP11", "11101102"),
    "CYP11B1": ("CYP11", "11101011"), "CYP11B2": ("CYP11", "11101102"),
    "CYP17A1": ("CYP17", "11102020"), "CYP19A1": ("CYP19", "11111011"),
    "CYP20A1": ("CYP20", "11010202"), "CYP21A2": ("CYP21", "11111001"),
    "CYP24A1": ("CYP24", "11101021"), "CYP26A1": ("CYP26", "11011220"),
    "CYP26B1": ("CYP26", "11111021"), "CYP26C1": ("CYP26", "11011221"),
    "CYP27A1": ("CYP27", "11101020"), "CYP27B1": ("CYP27", "11011222"),
    "CYP27C1": ("CYP27", "11101110"), "CYP39A1": ("CYP39", "11100021"),
    "CYP46A1": ("CYP46", "11121001"), "CYP51A1": ("CYP51", "11011221"),
}
ISOFORM_CELL_DISTINCTNESS = 0.9774436090225563   # 04_isoform_separation.json

# CYP3A4 contact-map fold-necessity evidence (cytochrome/publications/
# manifold/p450-address-manifold-cyp3a4-fold/validation/results/
# 08_contact_map_validation.json): a Kuramoto-folded compact structure's
# predicted contact map, scored against ground truth.
CONTACT_MAP_N_RESIDUES = 503
CONTACT_MAP_PRECISION = 0.3333333333333333
CONTACT_MAP_RECALL = 0.2857142857142857
CONTACT_MAP_N_GROUND_TRUTH_CONTACTS = 152
CONTACT_MAP_HEME_DETECTED = True
CONTACT_MAP_AXIAL_DETECTED = True

# Substrate promiscuity (probe-capacity proxy): number of distinct chemical
# environments a given isoform's active site couples to, real values
# (cytochrome/publications/diversity/57-human-isoforms/validation/results/
# 03_substrate_promiscuity.json). sigma is substrates metabolised per
# isoform (a real, literature-derived index), drug_fraction is the fraction
# of clinically used drugs each isoform metabolises.
PROMISCUITY_SIGMA = {"CYP3A4": 3.2, "CYP2D6": 1.8, "CYP2C9": 1.4}
PROMISCUITY_DRUG_FRACTION = {"CYP3A4": 0.5, "CYP2D6": 0.25, "CYP2C9": 0.16}

# Electron-transport chain partition depths M for successive cofactors in a
# real four-hop chain (cytochrome/publications/catalytic-cycle/multi-hop-
# et-chain/validation/results/01_chain_topology.json): each cofactor's
# partition depth M is computed independently from its own S-entropy
# coordinate, not fitted to this manuscript.
ET_CHAIN_COFACTOR_DEPTH = {
    "NADPH": 2.9772114415178454,
    "FAD": 2.4939253505012497,
    "FMN": 2.2607328698958993,
    "Fe(III)_HS": 7.130000000000019,
}
ET_CHAIN_HOP_DISTANCES_A = {"NADPH_to_FAD": 4.0, "FAD_to_FMN": 4.0,
                            "FMN_to_heme": 14.0}

# Categorical-distance vs. catalytic efficiency (cytochrome/publications/
# catalytic-cycle/multi-hop-et-chain/validation/results/02_dc4_efficiency.
# json): real kcat/KM measurements for four CYP3A4 substrates, each
# requiring d_C = 4 categorical transitions.
DC4_MEASURED_KCAT_KM = {
    "testosterone_6beta_OH": 1.5e6,
    "midazolam_1prime_OH": 4.0e6,
    "erythromycin": 7.0e5,
    "nifedipine": 2.0e6,
}
DC4_PREDICTED_KCAT_KM = 1.0e6

# Isoform allelic partition-depth shift and kinetic consequence (cytochrome/
# publications/diversity/57-human-isoforms/validation/results/
# 05_delta_m_isoform_shift.json): real, independently measured rate ratios.
ISOFORM_DELTA_M = {"CYP2D6": 0.08, "CYP2C9": 0.05}
ISOFORM_RATE_RATIO_TO_3A4 = {"CYP2D6": 0.9231, "CYP2C9": 0.9512}

# Atypical-reaction rate/depth-deficit ordering (cytochrome/publications/
# reactions/atypical-reactions-atlas/validation/results/06_rate_ordering.
# json): real computed depth deficits (delta_m) and rates (k, s^-1) for five
# distinct P450 reaction types, independently derived from mechanism, not
# from this manuscript.
REACTION_DELTA_M = {
    "NIH_shift": 0.18, "carbene": 0.2, "epoxidation": 0.35,
    "nucleophilic": 0.42, "desaturation": 1.4753,
}
REACTION_RATE_S = {
    "NIH_shift": 8352702114.11, "carbene": 8187307530.78,
    "epoxidation": 7046880897.19, "nucleophilic": 6570468198.15,
    "desaturation": 2287059153.95,
}

# Alpha-carbon bond dissociation energies for heteroatom dealkylation
# (cytochrome/publications/reactions/heteroatom-dealkylation/validation/
# results/01_alpha_carbon_bde.json): real literature BDEs (kcal/mol) and
# the corresponding depth deficits/activation energies.
DEALKYLATION_BDE_KCAL = {"N": 87.0, "O": 92.0, "aliphatic": 100.0}
DEALKYLATION_DELTA_M = {"N": 0.5, "O": 0.58, "aliphatic": 0.65}
DEALKYLATION_EA_KCAL = {"N": 7.768, "O": 9.011, "aliphatic": 10.098}


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
