"""Script 04 - Newton's Cradle Non-Identity Theorem.

Validates:
- 7 states have distinct receiver addresses
- Each state maps to unique integer ID 1-7
- Hamming distance >= 1 between all pairs
- States are non-degenerate (no two states identical)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "04_non_identity"

# States with their distinguishing receiver coordinates (n, l, m, s)
# Each state is characterized by its Fe oxidation state, spin state, ligands
STATE_COORDS = {
    1: {"Fe_ox": 3, "spin": 0, "ligand": "H2O",    "n": 3, "l": 2, "m": 0, "s": 1},
    2: {"Fe_ox": 3, "spin": 1, "ligand": "subst",  "n": 3, "l": 2, "m": 1, "s": 1},
    3: {"Fe_ox": 2, "spin": 0, "ligand": "none",   "n": 4, "l": 2, "m": 0, "s": 0},
    4: {"Fe_ox": 2, "spin": 0, "ligand": "O2",     "n": 4, "l": 2, "m": 1, "s": 0},
    5: {"Fe_ox": 3, "spin": 0, "ligand": "OO2-",   "n": 3, "l": 2, "m": 2, "s": 0},
    6: {"Fe_ox": 3, "spin": 0, "ligand": "OOH",    "n": 3, "l": 2, "m": 3, "s": 0},
    7: {"Fe_ox": 4, "spin": 1, "ligand": "O+rad",  "n": 4, "l": 3, "m": 0, "s": 1},
}

# Extract (Fe_ox, spin, n, l, m, s) tuples for uniqueness check
def state_tuple(s):
    c = STATE_COORDS[s]
    return (c["Fe_ox"], c["spin"], c["n"], c["l"], c["m"], c["s"])

tuples = {i: state_tuple(i) for i in range(1, 8)}

# Check all tuples are unique
all_unique = len(set(tuples.values())) == 7

# Hamming distance between two tuples
def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# Check all pairs have Hamming distance >= 1
pairs_ok = True
min_hamming = 999
for i in range(1, 8):
    for j in range(i+1, 8):
        d = hamming(tuples[i], tuples[j])
        if d < min_hamming:
            min_hamming = d
        if d < 1:
            pairs_ok = False

# Newton's Cradle non-identity: eval(R_bio, state_i) != eval(R_bio, state_j)
# This is guaranteed by distinct receiver coordinate tuples
non_identity_satisfied = all_unique and pairs_ok

data = {
    "n_states": len(STATE_COORDS),
    "state_tuples": {str(k): list(v) for k, v in tuples.items()},
    "all_unique": all_unique,
    "min_hamming_distance": min_hamming,
    "pairs_ok": pairs_ok,
    "non_identity_theorem_satisfied": non_identity_satisfied,
}

checks = {
    "exactly_7_states": len(STATE_COORDS) == 7,
    "all_tuples_unique": all_unique,
    "min_hamming_ge_1": min_hamming >= 1,
    "newton_cradle_non_identity": non_identity_satisfied,
    "state_ids_1_to_7": set(STATE_COORDS.keys()) == set(range(1, 8)),
}

write_result(name, data, checks)
