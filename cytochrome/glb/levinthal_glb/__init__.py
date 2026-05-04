"""
levinthal_glb: GLB-based structural input to the receiver R_bio.

Modules:
  - parser: GLB scene-graph traversal -> atomic positions + CPK colors
  - cpk: CPK color -> chemical element mapping
  - structure: atoms, bonds, residues, contact maps
  - s_entropy: atomic centroids -> S-entropy coordinates
  - rbio: receiver R_bio applied to GLB-derived structural data
"""

from levinthal_glb.parser import GLBProteinParser, parse_glb
from levinthal_glb.cpk import cpk_color_to_element, element_radius_A
from levinthal_glb.structure import Atom, Structure, contact_map_from_atoms
from levinthal_glb.s_entropy import atom_to_s_entropy, structure_to_address
from levinthal_glb.rbio import RbioGLBEvaluator

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
]
