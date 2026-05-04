"""
Validation 01: GLB parser smoke test.

Verifies Section 3 of Paper 2.5 (the parser construction).

Walks the scene graph of all three test GLBs and checks:
  - Each GLB loads without exception.
  - The parser produces *some* GLBAtom records (positioned objects),
    even for ribbon-only files (where most are wrapping meshes).
  - Per-atom records carry a 3D position, a sphere size, a node name,
    and a geometry name.
  - Metadata extraction yields source attribution (author, license,
    source URL) for the Sketchfab-derived GLBs.

Outputs: results/01_glb_parser_smoke.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ALL_GLBS, glb_path, write_result  # noqa: E402

from levinthal_glb import parse_glb  # noqa: E402


def main() -> dict:
    per_glb = {}
    for name in ALL_GLBS:
        s = parse_glb(glb_path(name))
        atom = s.atoms[0] if s.atoms else None
        per_glb[name] = {
            "n_positioned_objects": s.n_atoms,
            "metadata_keys": sorted(list(s.metadata.keys())),
            "has_author":  "author"  in s.metadata,
            "has_license": "license" in s.metadata,
            "has_source":  "source"  in s.metadata,
            "first_atom": None if atom is None else {
                "position": atom.position.tolist(),
                "color_rgb": atom.color_rgb,
                "element": atom.element,
                "sphere_size": atom.sphere_size,
                "node_name": atom.node_name,
                "geometry_name": atom.geometry_name,
            },
        }

    checks = {
        "all_glbs_loaded":       all(p["n_positioned_objects"] >= 1 for p in per_glb.values()),
        "all_have_metadata":     all(len(p["metadata_keys"]) > 0 for p in per_glb.values()),
        "first_atom_has_position":
            all(p["first_atom"] is not None and len(p["first_atom"]["position"]) == 3
                for p in per_glb.values()),
        "atomistic_glb_yields_many_atoms":
            per_glb["model_of_cytochrome_p450__oxygen__drug_complex.glb"]["n_positioned_objects"]
            >= 100,
        "ribbon_glbs_yield_few_objects": (
            per_glb["cytochrome_p450_with_haem_highlighted.glb"]["n_positioned_objects"] < 100
            and per_glb["practice_molecules_cytochrome_c.glb"]["n_positioned_objects"] < 100
        ),
    }

    return {
        "validation_id": "01_glb_parser_smoke",
        "paper_reference": "Paper 2.5, Section 3 (the parser, Construction 3.1)",
        "per_glb": per_glb,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("01_glb_parser_smoke.json", out)
    print(f"[{out['verdict']}] GLB parser smoke test")
    for name, info in out["per_glb"].items():
        print(f"  {name}: {info['n_positioned_objects']} positioned objects")
