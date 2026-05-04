"""
Smoke test: run the full GLB → R_bio pipeline on all three GLBs in
cytochrome/glb/ and emit a JSON report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add this directory to sys.path so the package import works
sys.path.insert(0, str(Path(__file__).parent))

from levinthal_glb import (  # noqa: E402
    GLBProteinParser,
    RbioGLBEvaluator,
    parse_glb,
)
from levinthal_glb.cpk import cpk_color_to_element  # noqa: E402
from levinthal_glb.structure import (  # noqa: E402
    bond_inference,
    contact_map_from_atoms,
    element_composition,
)

GLB_DIR = Path(__file__).parent
GLB_FILES = [
    "cytochrome_p450_with_haem_highlighted.glb",
    "model_of_cytochrome_p450__oxygen__drug_complex.glb",
    "practice_molecules_cytochrome_c.glb",
]


def test_glb(glb_filename: str) -> dict:
    """Process one GLB file end-to-end."""
    glb_path = GLB_DIR / glb_filename
    print(f"\n=== {glb_filename} ===")

    # 1. Parse
    structure = parse_glb(glb_path)
    print(f"   parsed {structure.n_atoms} positioned objects")

    # 2. Filter to real atoms (drop zero-position artifacts and oversize meshes)
    structure_filtered = structure.filter_oversized(max_size=5.0)
    structure_filtered = type(structure_filtered)(
        atoms=[a for a in structure_filtered.atoms
               if not (a.position[0] == 0 and a.position[1] == 0 and a.position[2] == 0)],
        metadata=structure_filtered.metadata,
        file_path=structure_filtered.file_path,
    )
    print(f"   after filtering: {structure_filtered.n_atoms} atoms")

    # 3. Element composition
    composition = element_composition(structure_filtered)
    print(f"   composition: {dict(sorted(composition.items(), key=lambda x: -x[1]))}")

    # 4. Bond inference
    bonds = bond_inference(structure_filtered)
    print(f"   bonds inferred: {len(bonds)}")

    # 5. R_bio evaluation
    if structure_filtered.n_atoms == 0:
        print("   skipping R_bio: no atoms")
        return {
            "glb_file": glb_filename,
            "n_atoms_total": structure.n_atoms,
            "n_atoms_filtered": 0,
            "skipped": True,
        }

    evaluator = RbioGLBEvaluator(structure_filtered)
    receiver_output = evaluator.evaluate()
    print(f"   M = {receiver_output['partition_depth_M']:.3f}")
    print(f"   address (depth 9): {receiver_output['trit_address_depth9']}")
    print(f"   contact map: {receiver_output['contact_map_n_contacts']} contacts, "
          f"density {receiver_output['contact_map_density']:.4f}")

    # 6. Iron neighbourhood
    if receiver_output["iron_atom_index"] is not None:
        print(f"   Fe at index {receiver_output['iron_atom_index']}")
        for nbr in receiver_output["iron_first_shell_neighbours"]:
            print(f"     - {nbr['element']} at {nbr['distance_A']:.3f} Å")

    return {
        "glb_file": glb_filename,
        "metadata": structure.metadata,
        "n_atoms_total": structure.n_atoms,
        "n_atoms_filtered": structure_filtered.n_atoms,
        "composition": composition,
        "n_bonds_inferred": len(bonds),
        "receiver_evaluation": receiver_output,
    }


def main():
    results = {}
    for glb_filename in GLB_FILES:
        results[glb_filename] = test_glb(glb_filename)

    # Save full report
    out_path = GLB_DIR / "test_glb_pipeline_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== SUMMARY ===")
    for filename, r in results.items():
        if r.get("skipped"):
            print(f"  {filename}: skipped")
            continue
        print(f"  {filename}:")
        print(f"    {r['n_atoms_filtered']} atoms ({r['composition']})")
        if "receiver_evaluation" in r:
            re_ = r["receiver_evaluation"]
            print(f"    M={re_['partition_depth_M']:.3f}, "
                  f"R_g={re_['radius_of_gyration_A']:.2f} Å, "
                  f"Fe={'yes' if re_['iron_atom_index'] is not None else 'no'}")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
