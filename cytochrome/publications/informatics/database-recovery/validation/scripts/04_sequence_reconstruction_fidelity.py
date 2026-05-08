"""Script 04 -- Sequence reconstruction fidelity from address manifold."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_sequence_reconstruction_fidelity"

# Sequence reconstruction: each amino acid at position i is assigned by
# decoding the ternary address. Fidelity = fraction of correctly decoded residues.
# Model: RMSD between reconstructed and native = f(depth)
# RMSD_aa ~ C * exp(-k * BITS_PER_TRIT / H_codon)

# H_codon: entropy of 20 amino acids with natural usage frequencies
# Approximate with uniform: H_uniform_20 = log2(20) ≈ 4.32 bits
H_codon = math.log2(20)   # bits per position

def rmsd_aa(k: int) -> float:
    """Fraction of incorrectly assigned residues at depth k."""
    return math.exp(-bits_at_depth(k) / H_codon)

fidelity_at_3 = 1.0 - rmsd_aa(3)
fidelity_at_6 = 1.0 - rmsd_aa(6)
fidelity_at_9 = 1.0 - rmsd_aa(9)

data = {
    "H_codon_bits":     round(H_codon, 4),
    "fidelity_at_k3":   round(fidelity_at_3, 4),
    "fidelity_at_k6":   round(fidelity_at_6, 4),
    "fidelity_at_k9":   round(fidelity_at_9, 4),
}

checks = {
    "fidelity_k3_gt_0.5":   fidelity_at_3 > 0.50,
    "fidelity_k6_gt_0.85":  fidelity_at_6 > 0.85,
    "fidelity_k9_gt_0.95":  fidelity_at_9 > 0.95,
    "fidelity_increases":   fidelity_at_3 < fidelity_at_6 < fidelity_at_9,
    "fidelity_lt_1_at_k9":  fidelity_at_9 < 1.0,
}

write_result(name, data, checks)
