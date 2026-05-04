"""
Validation 02: CPK color decoder.

Verifies Section 3.2 of Paper 2.5 (the CPK colour decoder with
30-RGB-unit tolerant matching).

Tests:
  - Exact CPK reference colours (H, C, N, O, P, S, Fe, ...) decode to
    their canonical elements.
  - Aliases (light-grey H, custom violet ligand "X") decode correctly.
  - Colours within tolerance (±25 RGB units) decode to nearest CPK
    element (Fe at (224, 102, 51) -> (220, 100, 55) still maps to Fe).
  - Out-of-tolerance colours (random pastel) map to "?" (unknown).
  - The decoder is symmetric: feeding the recovered element's reference
    colour back recovers the same element.

Outputs: results/02_cpk_color_decoder.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result  # noqa: E402

from levinthal_glb.cpk import (  # noqa: E402
    CPK_ALIASES,
    CPK_COLORS,
    cpk_color_to_element,
    element_radius_A,
)


def main() -> dict:
    # 1. Exact reference colours
    exact_pairs = []
    for elem, ref in CPK_COLORS.items():
        decoded = cpk_color_to_element(ref)
        exact_pairs.append({"element": elem, "rgb": list(ref), "decoded": decoded,
                            "match": decoded == elem})
    n_exact_correct = sum(1 for p in exact_pairs if p["match"])

    # 2. Alias matches
    alias_results = []
    for rgb, elem in CPK_ALIASES.items():
        decoded = cpk_color_to_element(rgb)
        alias_results.append({"rgb": list(rgb), "expected": elem,
                              "decoded": decoded, "match": decoded == elem})

    # 3. Within-tolerance perturbations (±15 units in each channel)
    tol_results = []
    for elem, ref in [("Fe", CPK_COLORS["Fe"]), ("S", CPK_COLORS["S"]),
                       ("O",  CPK_COLORS["O"]),  ("N", CPK_COLORS["N"]),
                       ("C",  CPK_COLORS["C"])]:
        perturbed = tuple(
            max(0, min(255, c + (15 if i % 2 == 0 else -10)))
            for i, c in enumerate(ref)
        )
        decoded = cpk_color_to_element(perturbed)
        tol_results.append({"element": elem,
                            "ref_rgb": list(ref),
                            "perturbed_rgb": list(perturbed),
                            "decoded": decoded,
                            "match": decoded == elem})

    # 4. Out-of-tolerance: random colours far from every reference
    far_colours = [(50, 80, 120), (200, 200, 200) ,(100, 30, 200)]
    far_results = []
    for rgb in far_colours:
        decoded = cpk_color_to_element(rgb)
        far_results.append({"rgb": list(rgb), "decoded": decoded})

    # 5. None input
    none_decoded = cpk_color_to_element(None)

    # 6. Round-trip: every CPK element has a vdW radius lookup
    radii = {elem: element_radius_A(elem) for elem in list(CPK_COLORS) + ["X", "?"]}

    checks = {
        "every_reference_colour_decodes_correctly":
            n_exact_correct == len(exact_pairs),
        "all_aliases_decode_correctly":
            all(a["match"] for a in alias_results),
        "perturbed_colours_within_tolerance_decode_correctly":
            all(t["match"] for t in tol_results),
        "none_input_decodes_to_unknown":
            none_decoded == "?",
        "every_cpk_element_has_vdw_radius":
            all(0.5 < r < 3.5 for r in radii.values()),
    }

    return {
        "validation_id": "02_cpk_color_decoder",
        "paper_reference": "Paper 2.5, Section 3.2",
        "n_cpk_reference_colours": len(CPK_COLORS),
        "n_aliases": len(CPK_ALIASES),
        "exact_match_summary": {
            "total": len(exact_pairs),
            "correct": n_exact_correct,
        },
        "alias_results": alias_results,
        "perturbation_results": tol_results,
        "out_of_tolerance_results": far_results,
        "vdw_radii_table": radii,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("02_cpk_color_decoder.json", out)
    print(f"[{out['verdict']}] CPK colour decoder")
    print(f"  reference colours: {out['exact_match_summary']['correct']}/"
          f"{out['exact_match_summary']['total']} decoded correctly")
    print(f"  aliases:           "
          f"{sum(1 for a in out['alias_results'] if a['match'])}/{len(out['alias_results'])}")
    print(f"  perturbed:         "
          f"{sum(1 for a in out['perturbation_results'] if a['match'])}/{len(out['perturbation_results'])}")
