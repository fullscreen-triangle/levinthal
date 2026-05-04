"""
Validation 03: Artifact filtering.

Verifies Section 3.3 of Paper 2.5 (the artifact filter that removes
oversized "wrapping" meshes and zero-position overlay markers).

Tests:
  - On the productive GLB: 171 raw -> 146 atoms after filtering
    (the package's headline number).
  - filter_oversized(max_size=5.0) drops meshes larger than any real
    atom's vdW diameter (real atoms <= 4 Å).
  - The zero-position drop catches origin-marker artifacts.
  - The filter never drops Fe (the heme iron must survive).
  - Filtering is idempotent: applying twice gives the same structure.

Outputs: results/03_artifact_filtering.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    GLB_ATOMISTIC, GLB_RIBBON_1, GLB_RIBBON_2, ALL_GLBS,
    filter_real_atoms, glb_path, write_result,
)

from levinthal_glb import parse_glb  # noqa: E402


def summarise(struct):
    sizes = [a.sphere_size for a in struct.atoms]
    fe_count = sum(1 for a in struct.atoms if a.element == "Fe")
    zero_pos = sum(1 for a in struct.atoms
                   if a.position[0] == 0 and a.position[1] == 0 and a.position[2] == 0)
    return {
        "n_atoms": struct.n_atoms,
        "max_sphere_size": max(sizes) if sizes else 0.0,
        "min_sphere_size": min(sizes) if sizes else 0.0,
        "fe_count": fe_count,
        "zero_position_count": zero_pos,
    }


def main() -> dict:
    per_glb = {}
    for name in ALL_GLBS:
        raw = parse_glb(glb_path(name))
        filtered = filter_real_atoms(raw)
        twice = filter_real_atoms(filtered)  # idempotent
        per_glb[name] = {
            "raw":       summarise(raw),
            "filtered":  summarise(filtered),
            "twice":     summarise(twice),
        }

    atomistic = per_glb[GLB_ATOMISTIC]
    ribbon1   = per_glb[GLB_RIBBON_1]
    ribbon2   = per_glb[GLB_RIBBON_2]

    checks = {
        # Atomistic GLB matches the headline number
        "atomistic_filtered_count_matches_published":
            atomistic["filtered"]["n_atoms"] == 146,
        "atomistic_drops_oversized":
            atomistic["raw"]["max_sphere_size"]
            > atomistic["filtered"]["max_sphere_size"],
        "atomistic_keeps_fe":
            atomistic["filtered"]["fe_count"] >= 1,
        "filtered_max_size_le_5":
            atomistic["filtered"]["max_sphere_size"] <= 5.0,
        "filtered_no_zero_positions":
            atomistic["filtered"]["zero_position_count"] == 0,
        # Idempotency
        "filter_idempotent_atomistic":
            atomistic["filtered"]["n_atoms"] == atomistic["twice"]["n_atoms"],
        "filter_idempotent_ribbon1":
            ribbon1["filtered"]["n_atoms"] == ribbon1["twice"]["n_atoms"],
        "filter_idempotent_ribbon2":
            ribbon2["filtered"]["n_atoms"] == ribbon2["twice"]["n_atoms"],
        # Ribbon GLBs reduce to <= 1 atom
        "ribbon_glbs_yield_at_most_one_atom_after_filter":
            ribbon1["filtered"]["n_atoms"] <= 1
            and ribbon2["filtered"]["n_atoms"] <= 1,
    }

    return {
        "validation_id": "03_artifact_filtering",
        "paper_reference": "Paper 2.5, Section 3.3",
        "filter_settings": {
            "max_sphere_size_A": 5.0,
            "drop_zero_position": True,
        },
        "per_glb": per_glb,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("03_artifact_filtering.json", out)
    print(f"[{out['verdict']}] artifact filtering")
    for name, info in out["per_glb"].items():
        print(f"  {name}:")
        print(f"    raw:      {info['raw']['n_atoms']:4d} atoms, "
              f"max_size={info['raw']['max_sphere_size']:.2f} Å")
        print(f"    filtered: {info['filtered']['n_atoms']:4d} atoms, "
              f"max_size={info['filtered']['max_sphere_size']:.2f} Å, "
              f"Fe={info['filtered']['fe_count']}")
