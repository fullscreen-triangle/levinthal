"""
GLB scene-graph parser.

Walks the GLB scene graph, extracts per-mesh transforms, derives
atomic positions and CPK colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from levinthal_glb.cpk import cpk_color_to_element


@dataclass
class GLBAtom:
    """A single atom extracted from a GLB scene graph."""

    position: np.ndarray  # (3,) Å
    color_rgb: Optional[tuple[int, int, int]]
    element: str
    sphere_size: float
    node_name: str
    geometry_name: str


@dataclass
class GLBStructure:
    """Aggregated structure from a GLB file."""

    atoms: list[GLBAtom]
    metadata: dict
    file_path: Path

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def positions(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms])

    @property
    def elements(self) -> list[str]:
        return [a.element for a in self.atoms]

    def filter_atoms(self, keep_unknown: bool = False) -> "GLBStructure":
        """Return a new structure excluding atoms with element == '?' (unidentified)."""
        kept = [a for a in self.atoms if keep_unknown or a.element != "?"]
        return GLBStructure(atoms=kept, metadata=self.metadata, file_path=self.file_path)

    def filter_oversized(self, max_size: float = 5.0) -> "GLBStructure":
        """Return atoms with sphere size below threshold (real atoms ~ 2 Å)."""
        kept = [a for a in self.atoms if a.sphere_size <= max_size]
        return GLBStructure(atoms=kept, metadata=self.metadata, file_path=self.file_path)


class GLBProteinParser:
    """
    GLB parser specialised for atomistic ball-and-stick models.

    Walks the scene graph, treats each transformed unit-sphere mesh as one
    atom positioned at the node's translation. Extracts CPK colour from the
    mesh's PBR baseColorFactor. Derives chemical element via colour matching.
    """

    def __init__(self, glb_path: str | Path):
        self.glb_path = Path(glb_path)
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")
        self.scene = trimesh.load(self.glb_path, process=False)

    def metadata(self) -> dict:
        """Extract source-file metadata."""
        meta = dict(self.scene.metadata) if hasattr(self.scene, "metadata") else {}
        # pygltflib already added asset.extras to scene.metadata in some versions;
        # also load via pygltflib for robustness
        try:
            from pygltflib import GLTF2
            gltf = GLTF2().load(self.glb_path)
            if gltf.asset and gltf.asset.extras:
                meta.update(gltf.asset.extras)
            if gltf.asset:
                meta["generator"] = gltf.asset.generator
        except Exception:
            pass
        return meta

    def parse_atoms(self) -> list[GLBAtom]:
        """Walk the scene graph and extract one atom per positioned primitive."""
        atoms: list[GLBAtom] = []
        if not hasattr(self.scene, "graph") or not hasattr(self.scene.graph, "nodes"):
            return atoms

        for node_name in self.scene.graph.nodes:
            try:
                transform, geometry_name = self.scene.graph[node_name]
            except (KeyError, ValueError):
                continue

            if geometry_name is None or geometry_name not in self.scene.geometry:
                continue

            mesh = self.scene.geometry[geometry_name]

            # The position is the translation component of the 4x4 transform
            position = np.array(transform[:3, 3], dtype=float)

            # Colour from PBR material's baseColorFactor
            color_rgb: Optional[tuple[int, int, int]] = None
            if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                material = mesh.visual.material
                if hasattr(material, "baseColorFactor"):
                    bcf = material.baseColorFactor
                    if bcf is not None and len(bcf) >= 3:
                        color_rgb = (int(bcf[0]), int(bcf[1]), int(bcf[2]))

            element = cpk_color_to_element(color_rgb)

            # Sphere size as max-extent of the mesh's local bounds
            sphere_size = float((mesh.bounds[1] - mesh.bounds[0]).max())

            atoms.append(GLBAtom(
                position=position,
                color_rgb=color_rgb,
                element=element,
                sphere_size=sphere_size,
                node_name=node_name,
                geometry_name=geometry_name,
            ))
        return atoms

    def to_structure(self) -> GLBStructure:
        atoms = self.parse_atoms()
        return GLBStructure(
            atoms=atoms,
            metadata=self.metadata(),
            file_path=self.glb_path,
        )


def parse_glb(glb_path: str | Path) -> GLBStructure:
    """Convenience function: parse a GLB and return a GLBStructure."""
    return GLBProteinParser(glb_path).to_structure()
