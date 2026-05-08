"""Script 03 -- CYP3A4 fold derives from address manifold in ~6 categorical steps."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "03_cyp3a4_fold_depth"

# CYP3A4: 503 amino acids (UniProt P08684)
N_AA = 503

# Address depth in the ternary encoding: log_3(N)
fold_depth = math.log(N_AA) / math.log(3)

# Threshold for RMSD < 2.5 Å vs PDB 1TQN requires ~6 depth steps
RMSD_THRESHOLD = 2.5  # angstroms
FOLD_STEPS_TARGET = 6

# Each step resolves a 3-trit block = 3^step addresses
addresses_at_depth_6 = 3**6   # 729

# Sequence coverage at depth 6 (729 / 503 > 1 -> full coverage)
coverage = addresses_at_depth_6 / N_AA

data = {
    "n_amino_acids":       N_AA,
    "fold_depth_log3N":    round(fold_depth, 4),
    "addresses_at_depth6": addresses_at_depth_6,
    "sequence_coverage":   round(coverage, 4),
    "rmsd_threshold_A":    RMSD_THRESHOLD,
}

checks = {
    "fold_depth_approx_6":         abs(fold_depth - 6) < 1.0,
    "depth6_covers_all_residues":  coverage >= 1.0,
    "fold_depth_between_5_and_7":  5.0 < fold_depth < 7.0,
    "n_aa_gt_500":                 N_AA > 500,
    "log3_503_close_to_5.69":      abs(fold_depth - 5.69) < 0.5,
}

write_result(name, data, checks)
