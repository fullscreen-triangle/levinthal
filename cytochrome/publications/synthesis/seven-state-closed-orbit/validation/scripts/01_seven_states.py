"""Script 01 - Seven-State Catalytic Cycle Definition.

Validates:
- 7 unique states defined
- 8 transitions (7->1 closes the orbit)
- All DM values > 0 and < 10
- States are non-degenerate
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

name = "01_seven_states"

# Define 7 states
STATES = {
    1: "Resting_Fe3plus_H2O_LS",
    2: "Substrate_bound_Fe3plus_HS",
    3: "Reduced_Fe2plus",
    4: "Oxy_Fe2plus_O2",
    5: "Peroxo_Fe3plus_OO2minus",
    6: "Compound0_Fe3plus_OOH",
    7: "CompoundI_Fe4plus_O_radical",
}

# 8 transitions (7->1 is product release + return to resting)
TRANSITIONS = list(DM_STEPS.keys())

n_states = len(STATES)
n_transitions = len(TRANSITIONS)

# Verify all DM > 0 and < 10
all_DM_valid = all(0 < dm < 10 for dm in DM_LIST)

# Verify state names are unique
state_names = list(STATES.values())
states_unique = len(set(state_names)) == len(state_names)

# Verify 7 unique state IDs
state_ids_unique = len(set(STATES.keys())) == 7

data = {
    "n_states": n_states,
    "n_transitions": n_transitions,
    "states": STATES,
    "DM_values": {k: round(v, 4) for k, v in DM_STEPS.items()},
    "DM_sum": round(DM_SUM, 4),
    "all_DM_positive": all(dm > 0 for dm in DM_LIST),
    "all_DM_lt_10": all(dm < 10 for dm in DM_LIST),
}

checks = {
    "exactly_7_states": n_states == 7,
    "exactly_8_transitions": n_transitions == 8,
    "all_DM_valid_range": all_DM_valid,
    "states_unique": states_unique,
    "orbit_closed_by_7_to_1": "product_release" in TRANSITIONS,
}

write_result(name, data, checks)
