"""Script 08 -- Full database recovery table: capacity, accuracy, compression."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "08_full_recovery_table"

recovery_table = {
    "k3_families":  {
        "depth": 3, "capacity": 3**3, "classes": N_FAMILIES,
        "bits": round(bits_at_depth(3), 3),
        "accuracy": round(recovery_accuracy(3, H_uniform_18), 4),
    },
    "k6_isoforms":  {
        "depth": 6, "capacity": 3**6, "classes": N_HUMAN_CYPS,
        "bits": round(bits_at_depth(6), 3),
        "accuracy": round(recovery_accuracy(6, H_uniform_57), 4),
    },
    "k9_alleles":   {
        "depth": 9, "capacity": 3**9, "classes": 310,  # ~310 PharmVar alleles total
        "bits": round(bits_at_depth(9), 3),
        "accuracy": round(recovery_accuracy(9, math.log2(310)), 4),
    },
}

all_bits_monotonic = (
    recovery_table["k3_families"]["bits"] <
    recovery_table["k6_isoforms"]["bits"] <
    recovery_table["k9_alleles"]["bits"]
)
all_accuracies_lt_1 = all(rt["accuracy"] < 1.0 for rt in recovery_table.values())
all_capacities_cover = all(
    rt["capacity"] >= rt["classes"] for rt in recovery_table.values()
)

data = {
    "recovery_table": recovery_table,
    "total_pharmvar_alleles": 310,
}

checks = {
    "all_bits_monotonic":      all_bits_monotonic,
    "all_accuracies_lt_1":     all_accuracies_lt_1,
    "all_capacities_cover":    all_capacities_cover,
    "k6_accuracy_gt_0.75":     recovery_table["k6_isoforms"]["accuracy"] > 0.75,
    "k9_accuracy_gt_0.80":     recovery_table["k9_alleles"]["accuracy"] > 0.80,
    "three_levels_parameterized": len(recovery_table) == 3,
}

write_result(name, data, checks)
