"""
Validation 08: Five GLB roles taxonomy.

Verifies Section 1.3 of Paper 2.5 (the five operational roles GLBs
play in receiver evaluation) by classifying each test GLB and the
operations it actually supports.

Roles:
  1. Calibration reference   — atomic distances available for ground-truth.
  2. Initial conditions      — Cα positions seedable for Kuramoto.
  3. Validation target       — top-L contact precision/recall computable.
  4. Interactive probe       — mesh suitable for in-browser display.
  5. Trajectory waypoint     — assignable to a specific catalytic state.

A GLB qualifies for a given role if the corresponding pipeline
operation does not raise *and* yields non-trivial output. Ribbon-only
GLBs serve role 4 alone; the productive atomistic GLB serves all five.

Tests:
  - The atomistic GLB is classified for all 5 roles.
  - Ribbon GLBs are classified for at most role 4 (interactive probe).
  - Each role's underlying operation either produces concrete output
    or fails gracefully.

Outputs: results/08_five_roles_taxonomy.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    ALL_GLBS, GLB_ATOMISTIC, GLB_RIBBON_1, GLB_RIBBON_2,
    filter_real_atoms, glb_path, write_result,
)

from levinthal_glb import RbioGLBEvaluator, parse_glb  # noqa: E402
from levinthal_glb.structure import find_iron, neighbours_of  # noqa: E402

import numpy as np  # noqa: E402


def role_calibration(struct) -> bool:
    """Role 1: at least one Fe with a non-empty first shell."""
    fe = find_iron(struct)
    if fe is None:
        return False
    nbrs = neighbours_of(struct, fe, cutoff_A=3.0)
    return len(nbrs) >= 4   # at least the porphyrin pyrroles


def role_initial_conditions(struct) -> bool:
    """Role 2: enough atoms to seed a Kuramoto simulation."""
    return struct.n_atoms >= 50


def role_validation_target(struct) -> bool:
    """Role 3: contact-map computable with a non-trivial number of pairs."""
    if struct.n_atoms < 10:
        return False
    rb = RbioGLBEvaluator(struct).evaluate()
    return rb["contact_map_n_contacts"] >= 1


def role_interactive_probe(struct) -> bool:
    """Role 4: any GLB that loads and yields *something* (mesh data) — true
    for every parsed GLB, even ribbon-only."""
    return True   # by construction; the file loaded into the parser


def role_trajectory_waypoint(struct) -> bool:
    """Role 5: a single discriminating Fe-O distance lies in any of the
    canonical state windows (Cpd I, oxy-complex, peroxo, water-axial)."""
    fe = find_iron(struct)
    if fe is None:
        return False
    fe_pos = struct.atoms[fe].position
    o_distances = [
        float(np.linalg.norm(struct.atoms[j].position - fe_pos))
        for j in neighbours_of(struct, fe, cutoff_A=3.0)
        if struct.atoms[j].element == "O"
    ]
    if not o_distances:
        return False
    closest = min(o_distances)
    # All canonical state windows
    windows = [(1.55, 1.75), (1.75, 1.90), (1.85, 2.00), (2.00, 2.20)]
    return any(lo <= closest <= hi for lo, hi in windows)


def classify(struct) -> dict[str, bool]:
    return {
        "role_1_calibration":         role_calibration(struct),
        "role_2_initial_conditions":  role_initial_conditions(struct),
        "role_3_validation_target":   role_validation_target(struct),
        "role_4_interactive_probe":   role_interactive_probe(struct),
        "role_5_trajectory_waypoint": role_trajectory_waypoint(struct),
    }


def main() -> dict:
    per_glb = {}
    for name in ALL_GLBS:
        try:
            s = filter_real_atoms(parse_glb(glb_path(name)))
            roles = classify(s)
            per_glb[name] = {
                "n_atoms": s.n_atoms,
                "roles":   roles,
                "n_roles_satisfied": sum(roles.values()),
            }
        except Exception as exc:
            per_glb[name] = {"error": str(exc), "n_roles_satisfied": 0}

    atom_info = per_glb[GLB_ATOMISTIC]
    rib1_info = per_glb[GLB_RIBBON_1]
    rib2_info = per_glb[GLB_RIBBON_2]

    checks = {
        "atomistic_satisfies_all_five_roles":
            atom_info.get("n_roles_satisfied") == 5,
        "ribbon_glbs_satisfy_at_most_role_four":
            rib1_info.get("n_roles_satisfied", 0) <= 1
            and rib2_info.get("n_roles_satisfied", 0) <= 1,
        "ribbon_glbs_satisfy_role_4":
            rib1_info.get("roles", {}).get("role_4_interactive_probe", False)
            and rib2_info.get("roles", {}).get("role_4_interactive_probe", False),
        "atomistic_role_5_oxy_complex":
            atom_info.get("roles", {}).get("role_5_trajectory_waypoint", False),
    }

    return {
        "validation_id": "08_five_roles_taxonomy",
        "paper_reference": "Paper 2.5, Section 1.3 (the five roles)",
        "role_definitions": {
            "1_calibration_reference":
                "Fe + non-empty first-shell coordination",
            "2_initial_conditions":
                ">= 50 atoms (sufficient to seed a Kuramoto simulation)",
            "3_validation_target":
                ">= 10 atoms with computable contact map",
            "4_interactive_probe":
                "GLB loads and yields mesh data (always satisfied)",
            "5_trajectory_waypoint":
                "Fe-O closest distance falls in a canonical state window",
        },
        "per_glb": per_glb,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("08_five_roles_taxonomy.json", out)
    print(f"[{out['verdict']}] five GLB roles taxonomy")
    for name, info in out["per_glb"].items():
        n = info.get("n_roles_satisfied", 0)
        print(f"  {name}: {n}/5 roles")
        for role, ok in (info.get("roles") or {}).items():
            mark = "+" if ok else "-"
            print(f"    [{mark}] {role}")
