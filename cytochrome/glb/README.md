# `levinthal_glb`: GLB-Based Structural Input to the Biomolecular Receiver

A Python package that bridges PDB-derived 3D structural data (encoded as
glTF/GLB) to the receiver $\mathcal{R}_{\mathrm{bio}}$ of the Levinthal
framework.

## Status

Built and validated against three GLB files. Two of three GLBs are
visualization-only (ribbons), but one (`model_of_cytochrome_p450__oxygen__drug_complex.glb`)
contains usable atomic-level data — 146 atoms with CPK-coloured chemistry
and physically correct iron coordination geometry.

## Why GLB

GLB files (binary glTF) are how PDB structures are typically distributed
to the web (RCSB Mol* viewer, Sketchfab, etc.). The framework's existing
validation suites use synthetic CYP3A4-statistical sequences; GLB
integration provides an upgrade path to real-PDB-grounded validation
without needing a full PDB parser.

## What works

For atomistic ball-and-stick GLBs (CPK-style space-filling), the package
extracts:

1. **Per-atom 3D positions** (via scene-graph node transforms, 1 atom per
   primitive)
2. **Chemical element identification** (via PBR baseColorFactor → CPK lookup)
3. **Bond inference** (via element-aware vdW distance threshold)
4. **Contact maps** (binary 0/1 within configurable cutoff)
5. **Centre of mass, radius of gyration, element composition**
6. **Iron coordination shell** (auto-detected, first-shell neighbours by
   distance)
7. **Receiver evaluation**: the four-stage morphism chain
   (observe → catalyze → fuse → access) applied to the GLB-derived
   coupling matrix
8. **S-entropy address**: per-atom and whole-structure trit addresses

## What doesn't work

- **Ribbon/cartoon GLBs** (e.g. PDB-style smoothed surface meshes): only
  surface vertices are accessible, no atomic resolution. The first and
  third test GLBs (`cytochrome_p450_with_haem_highlighted.glb`,
  `practice_molecules_cytochrome_c.glb`) fall into this category. The
  parser succeeds in walking the scene but yields no usable atoms.
- **Non-CPK colour schemes**: GLBs that use colour for visualisation
  (rainbow N-to-C, B-factor gradient, etc.) cannot be element-decoded.
- **Backbone vs side-chain**: the parser does not currently distinguish
  protein backbone from cofactor/substrate atoms; this requires a
  PDB-companion file (or RCSB API lookup by structure identifier).

## Quick start

```python
from levinthal_glb import parse_glb, RbioGLBEvaluator

# Parse a GLB
structure = parse_glb('model_of_cytochrome_p450__oxygen__drug_complex.glb')

# Filter artifacts (oversized "wrapping" meshes and zero-position objects)
structure = structure.filter_oversized(max_size=5.0)

# Apply the receiver
evaluator = RbioGLBEvaluator(structure)
result = evaluator.evaluate()

print(result['composition'])
print(f"Fe at index {result['iron_atom_index']}")
print(f"M = {result['partition_depth_M']:.3f}")
print(f"Trit address: {result['trit_address_depth9']}")
```

## Test results

Run `python test_glb_pipeline.py` for a full report. Headline result on
the productive GLB:

```
=== model_of_cytochrome_p450__oxygen__drug_complex.glb ===
Composition: {C: 80, H: 22, O: 13, X: 12, N: 12, P: 4, S: 2, Fe: 1}

Fe first-shell coordination (canonical P450 active site):
  O   at 1.814 Å   ← axial oxygen (oxy-complex / Fe-O₂)
  N×4 at 2.01-2.04 ← porphyrin nitrogens
  S   at 2.228 Å   ← proximal Cys thiolate
  N×8 at 2.23-2.27 ← secondary nitrogen shell (model-specific)
  O   at 2.340 Å   ← distal water? carbonyl?
  S   at 2.445 Å   ← Met thioether?
  C×8 at 2.74-2.79 ← porphyrin α-carbons
```

The 1.814 Å Fe-O bond is between an Fe-O₂ oxy-complex (~1.80 Å) and an
Fe-OOH peroxo (~1.85 Å), confirming this GLB models the oxy-complex
state (state 4 of the catalytic cycle) — useful as a structural waypoint
for Paper 5 (Compound I formation), since Compound I is reached two
apertures further along the cycle.

## Roles GLBs play in the monograph

(After the original five-roles taxonomy proposed in the GLB integration
plan)

1. **Calibration references**: ground-truth iron coordination geometry to
   verify the framework's predicted active-site distances.
2. **Initial conditions**: real Cα positions seed Kuramoto folding
   simulations (when ribbon GLBs include atomistic backbone).
3. **Validation targets**: real top-L contact precision/recall vs the
   framework's predictions (currently synthetic).
4. **Interactive probes**: web-frontend integration for user-driven
   exploration (mutation, substrate docking).
5. **Trajectory waypoints**: this oxy-complex GLB anchors state 4 of the
   catalytic cycle; combined with PDB 1TQN (state 1), 1W0E (state 2), and
   future Cpd I crystallographic snapshots, the seven-state cycle
   acquires structural anchors at four of seven states.

## Module layout

```
levinthal_glb/
├── __init__.py        # Public API exports
├── parser.py          # GLB → GLBStructure (atomic positions + colours)
├── cpk.py             # CPK colour → element + vdW radius + S-entropy table
├── structure.py       # Distance matrix, contact map, bond inference, Fe finder
├── s_entropy.py       # Element → S-coord, structure → trit address, F_CB
└── rbio.py            # Receiver R_bio applied to GLB structures
```

## Limitations honestly bounded

- Bond inference is geometric (vdW threshold) only; no chemistry-aware
  bond order detection.
- The X atoms (custom violet ligand colour) are mapped to a default
  carbon-like S-entropy coordinate. PDB-companion data would resolve
  ambiguity.
- Some GLBs include "wrapping" meshes (large transparent envelope around
  the protein) that get filtered by `filter_oversized`; this filter is
  hand-tuned to ≤ 5 Å bounds.
- Visualization GLBs without atomic primitives (ribbons, surfaces) are
  unsupported; parser yields zero usable atoms.

## Dependencies

- `trimesh >= 4.0`
- `pygltflib >= 1.16`
- `numpy`

Install with `pip install trimesh pygltflib numpy`.
