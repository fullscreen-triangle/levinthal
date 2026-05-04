"""
Validation 06: Morphism chain on a real GLB structure.

Verifies Section 4.3 of Paper 2.5: the four-stage morphism chain
(observe -> catalyze -> fuse -> access) runs end-to-end on a parsed
GLB structure and produces sensible derived quantities.

Tests:
  - observe(): coupling matrix is symmetric, non-negative, with zero
    diagonal.
  - catalyze(): boost is monotonic — no entry decreases.
  - fuse(): output is a convex combination — bounded by the
    component min/max element-wise.
  - access(): contact map is binary, symmetric, with zero diagonal.
  - Whole-pipeline: partition depth M and trit address are finite
    and well-defined.
  - Number of contacts is plausibly << N(N-1)/2.

Outputs: results/06_morphism_chain.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import GLB_ATOMISTIC, filter_real_atoms, glb_path, write_result  # noqa: E402

from levinthal_glb import RbioGLBEvaluator, parse_glb  # noqa: E402
import numpy as np  # noqa: E402
import math  # noqa: E402


def is_symmetric(M: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.allclose(M, M.T, atol=atol))


def main() -> dict:
    s = filter_real_atoms(parse_glb(glb_path(GLB_ATOMISTIC)))
    rb = RbioGLBEvaluator(s)

    # Stage 1: observe
    sigma_obs = rb.observe()
    obs_diag_zero = bool(np.allclose(np.diag(sigma_obs), 0))
    obs_nonneg    = bool((sigma_obs >= 0).all())
    obs_sym       = is_symmetric(sigma_obs)

    # Stage 2: catalyze (must not decrease any entry)
    sigma_cat = rb.catalyze(sigma_obs)
    cat_monotonic = bool((sigma_cat >= sigma_obs - 1e-12).all())
    cat_sym       = is_symmetric(sigma_cat)

    # Stage 3: fuse (with weights summing to 1)
    sigma_fused = rb.fuse(sigma_obs, sigma_cat, weights=[0.5, 0.5])
    fuse_within_min_max = bool(
        ((sigma_fused >= np.minimum(sigma_obs, sigma_cat) - 1e-9)
         & (sigma_fused <= np.maximum(sigma_obs, sigma_cat) + 1e-9)).all()
    )
    fuse_sym = is_symmetric(sigma_fused)

    # Stage 4: access (threshold to binary contact map)
    cm = rb.access_contact_map(sigma_fused)
    cm_binary = bool(((cm == 0) | (cm == 1)).all())
    cm_sym    = is_symmetric(cm)
    cm_diag_zero = bool((np.diag(cm) == 0).all())

    n = s.n_atoms
    max_pairs = n * (n - 1) // 2
    n_contacts = int(cm.sum() // 2)
    contact_density = n_contacts / max_pairs if max_pairs > 0 else 0.0

    # Whole-pipeline
    rb_full = RbioGLBEvaluator(s).evaluate()
    M = rb_full["partition_depth_M"]
    trit = rb_full["trit_address_depth9"]

    checks = {
        "observe_zero_diagonal":    obs_diag_zero,
        "observe_nonnegative":      obs_nonneg,
        "observe_symmetric":        obs_sym,
        "catalyze_monotonic_boost": cat_monotonic,
        "catalyze_symmetric":       cat_sym,
        "fuse_within_min_max":      fuse_within_min_max,
        "fuse_symmetric":           fuse_sym,
        "access_binary":            cm_binary,
        "access_symmetric":         cm_sym,
        "access_zero_diagonal":     cm_diag_zero,
        "partition_depth_finite":   math.isfinite(M),
        "trit_address_correct_length": len(trit) == 9 and all(c in "012" for c in trit),
        "contact_density_reasonable":
            0.0 < contact_density < 0.5,  # not all-on, not all-off
    }

    return {
        "validation_id": "06_morphism_chain",
        "paper_reference": "Paper 2.5, Section 4.3 (the four-stage chain)",
        "n_atoms": n,
        "matrix_diagnostics": {
            "observe":    {"symmetric": obs_sym,  "diag_zero": obs_diag_zero,
                           "nonneg": obs_nonneg, "max_entry": float(sigma_obs.max())},
            "catalyze":   {"symmetric": cat_sym,  "monotonic": cat_monotonic,
                           "max_entry": float(sigma_cat.max())},
            "fuse":       {"symmetric": fuse_sym, "within_min_max": fuse_within_min_max,
                           "max_entry": float(sigma_fused.max())},
            "access":     {"symmetric": cm_sym,   "binary": cm_binary,
                           "diag_zero": cm_diag_zero},
        },
        "n_contacts": n_contacts,
        "max_possible_contacts": max_pairs,
        "contact_density": contact_density,
        "partition_depth_M": M,
        "partition_n": rb_full["partition_n"],
        "partition_l": rb_full["partition_l"],
        "trit_address": trit,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    write_result("06_morphism_chain.json", out)
    print(f"[{out['verdict']}] morphism chain on real GLB")
    print(f"  N = {out['n_atoms']}, contacts = {out['n_contacts']} "
          f"(density {out['contact_density']:.4f})")
    print(f"  M = {out['partition_depth_M']:.3f}, "
          f"(n,l) = ({out['partition_n']}, {out['partition_l']})")
    print(f"  trit address: {out['trit_address']}")
