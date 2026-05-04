"""
Shared utilities for Paper 2.5 (GLB-based structural input) validations.

These tests run the levinthal_glb pipeline against the three test GLBs
shipped in cytochrome/glb/. The package itself lives one directory up
from the test data; we add it to sys.path here so each script can simply
import from levinthal_glb.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Repo / package roots.
#   _common.py lives at:
#     cytochrome/publications/foundations/glb-structural-input/validation/scripts/_common.py
#   parents: 0=scripts, 1=validation, 2=glb-structural-input, 3=foundations,
#            4=publications, 5=cytochrome.
THIS = Path(__file__).resolve()
GLB_DIR = THIS.parents[5] / "glb"                # cytochrome/glb/
PACKAGE_ROOT = GLB_DIR                            # contains the levinthal_glb pkg
RESULTS_DIR = THIS.parents[1] / "results"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


# Test GLB filenames
GLB_RIBBON_1 = "cytochrome_p450_with_haem_highlighted.glb"   # ribbon, no atoms
GLB_ATOMISTIC = "model_of_cytochrome_p450__oxygen__drug_complex.glb"  # productive
GLB_RIBBON_2 = "practice_molecules_cytochrome_c.glb"          # ribbon, no atoms

ALL_GLBS = [GLB_RIBBON_1, GLB_ATOMISTIC, GLB_RIBBON_2]


# Canonical CYP450 first-shell distances (crystallographic literature)
FE_N_PORPHYRIN_RANGE = (1.95, 2.10)   # Å — pyrrole N
FE_S_THIOLATE_RANGE = (2.20, 2.35)    # Å — proximal Cys
FE_O_OXY_COMPLEX_RANGE = (1.75, 1.90) # Å — Fe-O2 / Fe-OOH
FE_O_COMPOUND_I = 1.65                # Å — Cpd I ferryl


def glb_path(name: str) -> Path:
    """Absolute path of a GLB file in cytochrome/glb/."""
    return GLB_DIR / name


def filter_real_atoms(structure):
    """
    Apply the same filtering as test_glb_pipeline.py:
      - drop oversized "wrapping" meshes (> 5 Å bounding box)
      - drop zero-position artifacts (origin overlay markers)
    """
    from levinthal_glb.parser import GLBStructure
    s = structure.filter_oversized(max_size=5.0)
    kept = [
        a for a in s.atoms
        if not (a.position[0] == 0 and a.position[1] == 0 and a.position[2] == 0)
    ]
    return GLBStructure(atoms=kept, metadata=s.metadata, file_path=s.file_path)


def in_range(value: float, lo: float, hi: float) -> bool:
    """Closed-interval membership."""
    return lo <= value <= hi


def write_result(filename: str, result: dict) -> Path:
    """Write a result dict to results/<filename>.json."""
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with path.open("w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    """Tolerate numpy scalars and non-finite floats in JSON dumps."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, float):
        if math.isfinite(o):
            return o
        return str(o)
    return str(o)
