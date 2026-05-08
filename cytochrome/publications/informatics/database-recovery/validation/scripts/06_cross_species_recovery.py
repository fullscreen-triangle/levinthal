"""Script 06 -- Cross-species P450 recovery: bacterial vs mammalian address."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_cross_species_recovery"

# CYP101A1 (P450cam) vs CYP3A4: ~20% identity, diverged ~2 billion years ago
identity_cross = 0.20   # cross-species

# Within-human: 40% average identity across 57 isoforms
# But each isoform has ~5 close neighbors (>60% identity in same family)
# Recovery accuracy ~ mean identity among closest neighbors in address space
identity_within_family = 0.65   # within-subfamily average

# Within-human recovery: uses closest family members as reference
# Accuracy ~ 1 - exp(-identity * k) for k = depth of comparison
acc_cross   = 1.0 - math.exp(-identity_cross * DEPTH_ISOFORM)
acc_within  = 1.0 - math.exp(-identity_within_family * DEPTH_ISOFORM)

# Divergence depth: minimum depth where addresses differ
k_diverge_cross  = math.ceil(-math.log(identity_cross)  / math.log(3))
k_diverge_within = math.ceil(-math.log(identity_within_family) / math.log(3))

data = {
    "identity_cross_species":   identity_cross,
    "identity_within_family":   identity_within_family,
    "acc_cross_species":        round(acc_cross, 4),
    "acc_within_family":        round(acc_within, 4),
    "k_diverge_cross":          k_diverge_cross,
    "k_diverge_within":         k_diverge_within,
}

checks = {
    "k_diverge_cross_ge_2":      k_diverge_cross >= 2,
    "human_family_recovery_better": acc_within > acc_cross,
    "cross_accuracy_lt_0.7":     acc_cross < 0.7,
    "within_accuracy_gt_0.5":    acc_within > 0.5,
    "k_diverge_cross_le_6":      k_diverge_cross <= 6,
}

write_result(name, data, checks)
