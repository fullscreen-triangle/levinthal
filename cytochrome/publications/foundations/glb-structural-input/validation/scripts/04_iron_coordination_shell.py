"""
Validation 04: Iron coordination shell.

Verifies Section 4 of Paper 2.5 (Fe coordination auto-detection
matches canonical CYP450 active-site distances).

For the productive atomistic GLB, the parser must:
  - Detect exactly one Fe atom.
  - Find at least 4 N neighbours within 1.95–2.10 Å (porphyrin pyrroles).
  - Find at least 1 S neighbour within 2.20–2.35 Å (proximal Cys
    thiolate).
  - Find at least 1 O neighbour within 1.75–1.90 Å (axial dioxygen).
  - All first-shell elements lie within 3.0 Å of Fe.

These ranges come directly from the cytochrome P450 crystallographic
literature and are the operational definition of "physically correct
heme coordination" used in the paper.

Outputs: results/04_iron_coordination_shell.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    FE_N_PORPHYRIN_RANGE, FE_S_THIOLATE_RANGE, FE_O_OXY_COMPLEX_RANGE,
    GLB_ATOMISTIC, filter_real_atoms, glb_path, in_range, write_result,
)

from levinthal_glb import RbioGLBEvaluator, parse_glb  # noqa: E402
from levinthal_glb.structure import find_iron, neighbours_of  # noqa: E402

import numpy as np  # noqa: E402


def main() -> dict:
    s = filter_real_atoms(parse_glb(glb_path(GLB_ATOMISTIC)))
    iron_idx = find_iron(s)

    nbrs = []
    if iron_idx is not None:
        for j in neighbours_of(s, iron_idx, cutoff_A=3.0):
            d = float(np.linalg.norm(s.atoms[j].position - s.atoms[iron_idx].position))
            nbrs.append({
                "index": j,
                "element": s.atoms[j].element,
                "distance_A": round(d, 4),
            })
        nbrs.sort(key=lambda x: x["distance_A"])

    n_neighbors_within_porphyrin = sum(
        1 for n in nbrs
        if n["element"] == "N" and in_range(n["distance_A"], *FE_N_PORPHYRIN_RANGE)
    )
    s_within_thiolate = sum(
        1 for n in nbrs
        if n["element"] == "S" and in_range(n["distance_A"], *FE_S_THIOLATE_RANGE)
    )
    o_within_oxycomplex = sum(
        1 for n in nbrs
        if n["element"] == "O" and in_range(n["distance_A"], *FE_O_OXY_COMPLEX_RANGE)
    )

    closest_o = min(
        (n for n in nbrs if n["element"] == "O"),
        key=lambda n: n["distance_A"],
        default=None,
    )

    checks = {
        "exactly_one_iron":            iron_idx is not None,
        "first_shell_nonempty":        len(nbrs) > 0,
        "all_neighbours_within_3A":    all(n["distance_A"] <= 3.0 for n in nbrs),
        "four_or_more_porphyrin_N":    n_neighbors_within_porphyrin >= 4,
        "one_or_more_thiolate_S":      s_within_thiolate >= 1,
        "axial_O_in_oxy_complex_range":o_within_oxycomplex >= 1,
        "closest_O_in_oxy_complex_range":
            closest_o is not None
            and in_range(closest_o["distance_A"], *FE_O_OXY_COMPLEX_RANGE),
    }

    # Receiver evaluator path also reports neighbours; cross-check
    rb = RbioGLBEvaluator(s).evaluate()

    return {
        "validation_id": "04_iron_coordination_shell",
        "paper_reference": "Paper 2.5, Section 4 (Construction 4.1)",
        "iron_atom_index": iron_idx,
        "first_shell_neighbours": nbrs,
        "summary_counts": {
            "n_porphyrin_N":    n_neighbors_within_porphyrin,
            "n_thiolate_S":     s_within_thiolate,
            "n_axial_oxycomplex_O": o_within_oxycomplex,
            "total_first_shell": len(nbrs),
        },
        "closest_O": closest_o,
        "ranges_used": {
            "Fe_N_porphyrin_A":    list(FE_N_PORPHYRIN_RANGE),
            "Fe_S_thiolate_A":     list(FE_S_THIOLATE_RANGE),
            "Fe_O_oxy_complex_A":  list(FE_O_OXY_COMPLEX_RANGE),
        },
        "rbio_iron_atom_index_cross_check": rb["iron_atom_index"] == iron_idx,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("04_iron_coordination_shell.json", out)
    print(f"[{out['verdict']}] iron coordination shell")
    print(f"  Fe at index {out['iron_atom_index']}, "
          f"{len(out['first_shell_neighbours'])} first-shell neighbours")
    for n in out["first_shell_neighbours"][:8]:
        print(f"    {n['element']:2s}  at {n['distance_A']:.3f} Å")
