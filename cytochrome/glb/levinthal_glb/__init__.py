"""
levinthal_glb: GLB-based structural input to the receiver R_bio.

Modules:
  - parser: GLB scene-graph traversal -> atomic positions + CPK colors
  - cpk: CPK color -> chemical element mapping
  - structure: atoms, bonds, residues, contact maps
  - s_entropy: atomic centroids -> S-entropy coordinates
  - rbio: receiver R_bio applied to GLB-derived structural data
  - shader_pipeline: numpy prototype of the 5-pass fragment shader
                     producing electron-trajectory visualisations
                     (Layer 5 of the apparatus stack, Paper 4)
"""

from levinthal_glb.parser import GLBProteinParser, parse_glb
from levinthal_glb.cpk import cpk_color_to_element, element_radius_A
from levinthal_glb.structure import Atom, Structure, contact_map_from_atoms
from levinthal_glb.s_entropy import atom_to_s_entropy, structure_to_address
from levinthal_glb.rbio import RbioGLBEvaluator
from levinthal_glb.shader_pipeline import (
    CofactorPlacement,
    electron_density_grid,
    hop_occupancies,
    diffraction_pattern_2d,
    lambda_from_diffraction,
    run_pipeline_glb_grounded,
)

__all__ = [
    "GLBProteinParser",
    "parse_glb",
    "cpk_color_to_element",
    "element_radius_A",
    "Atom",
    "Structure",
    "contact_map_from_atoms",
    "atom_to_s_entropy",
    "structure_to_address",
    "RbioGLBEvaluator",
    "CofactorPlacement",
    "electron_density_grid",
    "hop_occupancies",
    "diffraction_pattern_2d",
    "lambda_from_diffraction",
    "run_pipeline_glb_grounded",
]
