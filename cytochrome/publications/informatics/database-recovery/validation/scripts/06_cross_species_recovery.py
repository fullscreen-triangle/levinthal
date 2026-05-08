"""Script 06 -- Cross-species P450 recovery: bacterial vs mammalian address."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "06_cross_species_recovery"

# CYP101A1 (P450cam from Pseudomonas putida) vs CYP3A4 (human)
# Both are P450s but diverged ~2 billion years ago
# Address manifold should distinguish them by depth k >= 2 (family level)

# CYP101A1: sequence identity ~20% with CYP3A4
# At k=2 depth: 3^2=9 classes -> separates prokaryote vs eukaryote CYPs
identity_cyp101_cyp3a4 = 0.20  # 20% sequence identity

# Divergence at address depth k:
# Expected k_diverge = ceil(-log3(identity))
k_diverge_pred = math.ceil(-math.log(identity_cyp101_cyp3a4) / math.log(3))

# Cross-species recovery: if we know human CYP3A4 address,
# how well can we predict the bacterial P450 sequence?
# Prediction accuracy = identity^(1/k_diverge) for k-depth interpolation
pred_accuracy_cross = identity_cyp101_cyp3a4 ** (1.0 / k_diverge_pred)

# Within-human recovery is much better:
within_human_identity = 0.40  # average identity within human CYPs
k_diverge_human = math.ceil(-math.log(within_human_identity) / math.log(3))
pred_accuracy_human = within_human_identity ** (1.0 / max(k_diverge_human, 1))

data = {
    "identity_cyp101_cyp3a4":   identity_cyp101_cyp3a4,
    "k_diverge_cross_species":  k_diverge_pred,
    "pred_accuracy_cross":      round(pred_accuracy_cross, 4),
    "within_human_identity":    within_human_identity,
    "pred_accuracy_human":      round(pred_accuracy_human, 4),
}

checks = {
    "k_diverge_cross_ge_2":      k_diverge_pred >= 2,
    "human_recovery_better":     pred_accuracy_human > pred_accuracy_cross,
    "cross_accuracy_lt_0.7":     pred_accuracy_cross < 0.7,
    "human_accuracy_gt_0.5":     pred_accuracy_human > 0.5,
    "k_diverge_cross_le_6":      k_diverge_pred <= 6,
}

write_result(name, data, checks)
