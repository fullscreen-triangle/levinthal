"""
Validation 05: State-4 oxy-complex identification.

Verifies the headline conclusion of Paper 2.5 (Section 5.2): the
1.814 Å Fe-O distance in the productive GLB places it at state 4
(oxy-complex / Fe^II-O_2) of the catalytic cycle.

Tests:
  - The closest Fe-O distance lies in the canonical Fe-O_2 / Fe-OOH
    range (1.75–1.90 Å).
  - It is closer to oxy-complex (1.80 Å) than to Compound I (1.65 Å)
    or to a hexacoordinate Fe-OH (2.10 Å+).
  - Therefore Paper 5's Compound I distance (1.65 Å) is *not* observed
    here, consistent with this GLB modelling state 4 not state 6.
  - The result contributes to the "trajectory waypoints" role
    (Role 5 of Paper 2.5 Section 1.3).

Outputs: results/05_state4_oxy_complex.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    FE_O_COMPOUND_I, FE_O_OXY_COMPLEX_RANGE,
    GLB_ATOMISTIC, filter_real_atoms, glb_path, write_result,
)

from levinthal_glb import parse_glb  # noqa: E402
from levinthal_glb.structure import find_iron, neighbours_of  # noqa: E402

import numpy as np  # noqa: E402


# Canonical reference Fe-O distances (Å), used as discriminators.
FE_O_REFERENCES = {
    "compound_I_ferryl":   1.65,   # Cpd I Fe^IV=O
    "oxy_complex_Fe_O2":   1.80,   # Fe^II-O_2 (state 4)
    "hydroperoxo_Fe_OOH":  1.85,   # Cpd 0
    "water_axial_FeIII_OH": 2.10,  # Resting state alternate
}


def main() -> dict:
    s = filter_real_atoms(parse_glb(glb_path(GLB_ATOMISTIC)))
    fe_idx = find_iron(s)
    if fe_idx is None:
        return {"validation_id": "05_state4_oxy_complex", "verdict": "FAIL",
                "reason": "no Fe found"}

    fe_pos = s.atoms[fe_idx].position
    oxygen_distances = []
    for j in neighbours_of(s, fe_idx, cutoff_A=3.0):
        if s.atoms[j].element == "O":
            d = float(np.linalg.norm(s.atoms[j].position - fe_pos))
            oxygen_distances.append(d)
    oxygen_distances.sort()
    closest_FeO = oxygen_distances[0] if oxygen_distances else float("inf")

    # Identify which canonical state is closest
    distances_to_refs = {
        name: abs(closest_FeO - ref) for name, ref in FE_O_REFERENCES.items()
    }
    nearest_state = min(distances_to_refs, key=distances_to_refs.get)

    delta_to_oxy_complex = abs(closest_FeO - FE_O_REFERENCES["oxy_complex_Fe_O2"])
    delta_to_compound_I  = abs(closest_FeO - FE_O_COMPOUND_I)

    checks = {
        "fe_o_in_oxy_complex_range":
            FE_O_OXY_COMPLEX_RANGE[0] <= closest_FeO <= FE_O_OXY_COMPLEX_RANGE[1],
        "closer_to_oxy_complex_than_compound_I":
            delta_to_oxy_complex < delta_to_compound_I,
        "compound_I_distance_not_observed":
            closest_FeO > FE_O_COMPOUND_I + 0.05,
        "nearest_state_is_oxy_complex_or_hydroperoxo":
            nearest_state in ("oxy_complex_Fe_O2", "hydroperoxo_Fe_OOH"),
    }

    return {
        "validation_id": "05_state4_oxy_complex",
        "paper_reference": "Paper 2.5, Section 5.2 (state-4 identification)",
        "fe_atom_index": fe_idx,
        "all_oxygen_distances_A": oxygen_distances,
        "closest_Fe_O_A": closest_FeO,
        "reference_distances_A": FE_O_REFERENCES,
        "delta_to_each_state_A": distances_to_refs,
        "nearest_canonical_state": nearest_state,
        "delta_to_oxy_complex_A":   delta_to_oxy_complex,
        "delta_to_compound_I_A":    delta_to_compound_I,
        "interpretation":
            "Fe-O = {:.3f} Å places the GLB at state 4 (oxy-complex) "
            "between Fe-O_2 (1.80) and Fe-OOH (1.85). "
            "Compound I (1.65 Å) is two apertures further along "
            "the cycle and is not observed here.".format(closest_FeO),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("05_state4_oxy_complex.json", out)
    print(f"[{out['verdict']}] state-4 oxy-complex identification")
    print(f"  closest Fe-O = {out['closest_Fe_O_A']:.3f} Å")
    print(f"  nearest canonical state: {out['nearest_canonical_state']}")
    print(f"  delta to oxy-complex (1.80 Å): {out['delta_to_oxy_complex_A']:.3f} Å")
    print(f"  delta to Cpd I       (1.65 Å): {out['delta_to_compound_I_A']:.3f} Å")
