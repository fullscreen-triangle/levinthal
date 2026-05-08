"""Script 02 -- Recovery accuracy as a function of ternary depth."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "02_recovery_accuracy_vs_depth"

depths = list(range(1, 13))
acc_isoform = [recovery_accuracy(k, H_uniform_57) for k in depths]
acc_family  = [recovery_accuracy(k, H_uniform_18) for k in depths]

# At k=6, accuracy for isoform recovery
acc_at_6 = recovery_accuracy(6, H_uniform_57)
acc_at_9 = recovery_accuracy(9, H_uniform_57)
acc_at_3  = recovery_accuracy(3, H_uniform_18)

data = {
    "acc_isoform_at_k6": round(acc_at_6, 4),
    "acc_isoform_at_k9": round(acc_at_9, 4),
    "acc_family_at_k3":  round(acc_at_3, 4),
    "H_57_bits":         round(H_uniform_57, 4),
    "H_18_bits":         round(H_uniform_18, 4),
}

checks = {
    "acc_k6_gt_0.85":     acc_at_6 > 0.85,
    "acc_k9_gt_0.95":     acc_at_9 > 0.95,
    "acc_k3_family_gt_0.75": acc_at_3 > 0.75,
    "acc_increases_with_k": all(acc_isoform[i] < acc_isoform[i+1] for i in range(len(depths)-1)),
    "acc_lt_1_at_k12":    acc_isoform[-1] < 1.0,
}

write_result(name, data, checks)
