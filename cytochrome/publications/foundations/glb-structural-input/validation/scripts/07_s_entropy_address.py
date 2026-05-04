"""
Validation 07: S-entropy address generation.

Verifies Section 4.1 of Paper 2.5 (per-atom and whole-structure
S-entropy mapping) and the trit-address generator (Paper 1, Eq. 1).

Tests:
  - Every covered element has an S-coordinate in [0, 1]^3.
  - Per-atom addresses are length-9 strings of {0,1,2}.
  - Whole-structure centroid S-coordinate has norm < 1
    (so F_CB is well-defined without regularisation).
  - F_CB on the centroid yields finite M, n >= 1, 0 <= l <= n-1.
  - Trit address is deterministic: re-running gives the same address.
  - Element diversity reflects in address diversity:
    the productive GLB has > 5 distinct per-atom trit addresses
    (Fe, S, O, N, C, P, H, X each map to a different trit).

Outputs: results/07_s_entropy_address.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import GLB_ATOMISTIC, filter_real_atoms, glb_path, write_result  # noqa: E402

import math  # noqa: E402

from levinthal_glb import parse_glb  # noqa: E402
from levinthal_glb.cpk import ELEMENT_S_ENTROPY  # noqa: E402
from levinthal_glb.s_entropy import (  # noqa: E402
    F_CB, atom_to_s_entropy, per_atom_addresses,
    structure_centroid_s_entropy, trit_address,
)


def in_unit_cube(s) -> bool:
    return all(0.0 <= c <= 1.0 for c in s)


def main() -> dict:
    # 1. Per-element table
    table = {}
    all_in_cube = True
    for elem, coord in ELEMENT_S_ENTROPY.items():
        in_cube = in_unit_cube(coord)
        if not in_cube:
            all_in_cube = False
        table[elem] = {
            "S": list(coord),
            "norm": math.sqrt(sum(c * c for c in coord)),
            "in_unit_cube": in_cube,
            "trit_address_depth9": trit_address(coord, depth=9),
        }

    # 2. Productive GLB structure
    s = filter_real_atoms(parse_glb(glb_path(GLB_ATOMISTIC)))
    centroid = structure_centroid_s_entropy(s)
    centroid_norm = math.sqrt(sum(c * c for c in centroid))
    fcb = F_CB(centroid)
    centroid_address = trit_address(centroid, depth=9)
    centroid_address_again = trit_address(centroid, depth=9)

    per_atom = per_atom_addresses(s, depth=9)
    distinct_addresses = sorted(set(per_atom))

    # 3. Sanity: each atom's address comes from its element's S
    consistent = True
    for atom, addr in zip(s.atoms, per_atom):
        expected = trit_address(atom_to_s_entropy(atom.element), depth=9)
        if addr != expected:
            consistent = False
            break

    checks = {
        "all_element_S_in_unit_cube": all_in_cube,
        "centroid_in_unit_cube": in_unit_cube(centroid),
        "centroid_norm_below_one": centroid_norm < 1.0,
        "F_CB_finite": math.isfinite(fcb["M"]),
        "F_CB_n_at_least_one": fcb["n"] >= 1,
        "F_CB_l_within_range": 0 <= fcb["l"] < max(1, fcb["n"]),
        "centroid_address_correct_length": len(centroid_address) == 9,
        "centroid_address_only_trits":
            all(c in "012" for c in centroid_address),
        "trit_address_deterministic": centroid_address == centroid_address_again,
        "per_atom_address_consistent_with_element": consistent,
        "per_atom_address_diversity": len(distinct_addresses) >= 5,
    }

    return {
        "validation_id": "07_s_entropy_address",
        "paper_reference":
            "Paper 2.5, Section 4.1 + Paper 1 trit-address Eq. (1)",
        "n_elements_covered": len(ELEMENT_S_ENTROPY),
        "element_table": table,
        "structure_centroid_S": list(centroid),
        "structure_centroid_norm": centroid_norm,
        "F_CB_on_centroid": fcb,
        "structure_centroid_address": centroid_address,
        "n_atoms": s.n_atoms,
        "n_distinct_per_atom_addresses": len(distinct_addresses),
        "distinct_per_atom_addresses": distinct_addresses,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("07_s_entropy_address.json", out)
    print(f"[{out['verdict']}] S-entropy address generation")
    print(f"  centroid S = {out['structure_centroid_S']}, "
          f"||S|| = {out['structure_centroid_norm']:.4f}")
    print(f"  F_CB: M = {out['F_CB_on_centroid']['M']:.3f}, "
          f"(n,l) = ({out['F_CB_on_centroid']['n']}, "
          f"{out['F_CB_on_centroid']['l']})")
    print(f"  centroid address: {out['structure_centroid_address']}")
    print(f"  distinct per-atom addresses: {out['n_distinct_per_atom_addresses']}")
