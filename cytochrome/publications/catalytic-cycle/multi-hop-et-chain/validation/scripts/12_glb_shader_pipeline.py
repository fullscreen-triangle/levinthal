"""
Validation 12: GLB-grounded shader pipeline integration.

Verifies that the apparatus described in Paper 4 (Section 2) and Paper 2.5
(GLB-based structural input) compose correctly: the productive cytochrome
P450 GLB is loaded as analyte, the heme-Fe position is taken DIRECTLY from
the GLB scene graph, the four cofactor centres are placed along the
extension axis at literature distances, and the Layer 5 hologram pipeline
runs on the resulting bounding box.

This is the apparatus's integration test: it confirms the chain
GLB -> parser -> Fe-anchored cofactor placement -> shader pipeline ->
electron-density grid -> diffraction pattern -> Marcus lambda. Failure of
any link breaks the experimental claim of Paper 4.

Outputs: results/12_glb_shader_pipeline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import LAMBDA_REORG_EV  # noqa: E402

# Wire the levinthal_glb package into sys.path
_HERE = Path(__file__).resolve()
_GLB_DIR = _HERE.parents[5] / "glb"
if str(_GLB_DIR) not in sys.path:
    sys.path.insert(0, str(_GLB_DIR))

from levinthal_glb import (  # noqa: E402
    run_pipeline_glb_grounded,
    hop_occupancies,
)

GLB_FILE = "model_of_cytochrome_p450__oxygen__drug_complex.glb"


def main() -> dict:
    glb_path = _GLB_DIR / GLB_FILE

    pipeline = run_pipeline_glb_grounded(
        glb_path=str(glb_path),
        t_fs_frames=(0.0, 100.0, 250.0, 500.0, 800.0),
        grid_shape=(40, 40, 40),
    )
    frames = pipeline["frames"]

    fe_pos = pipeline["fe_position_A"]
    cof_pos = np.array(pipeline["cofactor_positions_A"])

    # The heme cofactor must equal the GLB Fe position exactly.
    fe_anchored = bool(
        np.allclose(np.asarray(fe_pos), cof_pos[3], atol=1e-9)
    )

    # The chain length should match the literature-anchored intercofactor
    # distances (4 + 4 + 14 = 22 A from NADPH to heme-Fe).
    chain_length_A = float(np.linalg.norm(cof_pos[3] - cof_pos[0]))
    chain_length_expected = 22.0
    chain_length_match = abs(chain_length_A - chain_length_expected) < 1.0

    # Final-frame |psi|^2 peak must be near the heme-Fe at the categorical
    # clock timescale: at t=800 fs the heme occupancy should dominate.
    final_occ = frames[-1]["occupancy"]
    heme_dominates_final = final_occ[3] > 0.5

    # Centre-of-density should travel from NADPH to heme across frames
    axis_vec = cof_pos[3] - cof_pos[0]
    axis_vec /= max(np.linalg.norm(axis_vec), 1e-12)
    bbox_min = np.array(pipeline["bbox_min_A"])
    bbox_max = np.array(pipeline["bbox_max_A"])

    centroids = []
    for f in frames:
        density = f["density"]
        nx, ny, nz = density.shape
        xs = np.linspace(bbox_min[0], bbox_max[0], nx)
        ys = np.linspace(bbox_min[1], bbox_max[1], ny)
        zs = np.linspace(bbox_min[2], bbox_max[2], nz)
        XG, YG, ZG = np.meshgrid(xs, ys, zs, indexing="ij")
        rel = np.stack([XG - cof_pos[0][0],
                        YG - cof_pos[0][1],
                        ZG - cof_pos[0][2]], axis=-1)
        proj = (rel * axis_vec).sum(axis=-1)
        if density.sum() > 0:
            centroids.append(float((proj * density).sum() / density.sum()))
        else:
            centroids.append(0.0)

    monotonic_advance = all(
        centroids[i + 1] >= centroids[i] - 0.5
        for i in range(len(centroids) - 1)
    )
    centroid_advance_A = centroids[-1] - centroids[0]
    centroid_advance_at_least_half_chain = (
        centroid_advance_A >= chain_length_A * 0.5
    )

    # Marcus lambda from the diffraction pattern at the final frame
    # must lie within 20% of the framework canonical 0.85 eV
    final_lambda = frames[-1]["lambda_eV"]
    lambda_within_20_percent = (
        final_lambda is not None
        and abs(final_lambda - LAMBDA_REORG_EV) / LAMBDA_REORG_EV <= 0.20
    )

    # Apparatus-level check: number of GLB atoms recovered matches Paper 2.5
    n_atoms_correct = pipeline["n_glb_atoms"] == 146

    checks = {
        "fe_position_anchored_to_glb": fe_anchored,
        "chain_length_matches_literature": chain_length_match,
        "heme_occupancy_dominates_final_frame": heme_dominates_final,
        "centroid_advances_monotonically": monotonic_advance,
        "centroid_traverses_at_least_half_chain":
            centroid_advance_at_least_half_chain,
        "marcus_lambda_within_20_percent_of_canonical":
            lambda_within_20_percent,
        "glb_n_atoms_matches_paper2_5_baseline": n_atoms_correct,
    }

    return {
        "validation_id": "12_glb_shader_pipeline",
        "paper_reference":
            "Paper 4, Section 2 (apparatus) + Paper 2.5 (GLB structural input)",
        "glb_file": GLB_FILE,
        "n_glb_atoms": pipeline["n_glb_atoms"],
        "fe_position_A": fe_pos,
        "axis_vec":      pipeline["axis_vec"],
        "cofactor_positions_A": pipeline["cofactor_positions_A"],
        "chain_length_A": chain_length_A,
        "centroids_along_chain_A": centroids,
        "centroid_advance_A": centroid_advance_A,
        "final_frame_occupancy_NADPH_FAD_FMN_heme": list(final_occ),
        "final_frame_marcus_lambda_eV": final_lambda,
        "framework_canonical_lambda_eV": LAMBDA_REORG_EV,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "12_glb_shader_pipeline.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[{out['verdict']}] GLB-grounded shader pipeline integration")
    print(f"  GLB:                {out['glb_file']}")
    print(f"  GLB atoms:          {out['n_glb_atoms']}")
    print(f"  Fe at:              {out['fe_position_A']}")
    print(f"  Chain length:       {out['chain_length_A']:.2f} A")
    print(f"  Centroid advance:   {out['centroid_advance_A']:.2f} A")
    print(f"  Heme occupancy[-1]: {out['final_frame_occupancy_NADPH_FAD_FMN_heme'][3]:.3f}")
    print(f"  Marcus lambda:      {out['final_frame_marcus_lambda_eV']:.3f} eV "
          f"(canonical {LAMBDA_REORG_EV} eV)")
    print(f"  -> wrote {out_path}")
