# Paper 2.5 Validation Suite — GLB-Based Structural Input

Validation scripts for *GLB-Based Structural Input to the Biomolecular
Receiver: Bridging PDB Geometry and Categorical Evaluation*.

## Structure

```
validation/
├── README.md
├── run_all.py
├── scripts/
│   ├── _common.py
│   ├── 01_glb_parser_smoke.py         # scene-graph traversal across all 3 GLBs
│   ├── 02_cpk_color_decoder.py        # CPK colour -> element with tolerance
│   ├── 03_artifact_filtering.py       # oversized + zero-position filters
│   ├── 04_iron_coordination_shell.py  # Fe-N, Fe-S, Fe-O canonical distances
│   ├── 05_state4_oxy_complex.py       # 1.814 Å Fe-O = state 4 (oxy-complex)
│   ├── 06_morphism_chain.py           # observe -> catalyze -> fuse -> access
│   ├── 07_s_entropy_address.py        # element table, F_CB, trit address
│   └── 08_five_roles_taxonomy.py      # which GLB serves which role
└── results/
```

## What is validated

| # | Paper section | Tested |
|---|---|---|
| 01 | Sec 3 (Cons. 3.1) | Parser walks all 3 test GLBs; metadata extracted |
| 02 | Sec 3.2          | CPK exact, alias, perturbed, and out-of-tolerance colours |
| 03 | Sec 3.3          | filter\_oversized + zero-position; 171 → 146 atoms |
| 04 | Sec 4 (Cons. 4.1)| Fe + 4 N (porphyrin) + S (Cys) + O (axial) within canonical ranges |
| 05 | Sec 5.2          | Closest Fe-O = 1.814 Å; nearest canonical state = oxy-complex |
| 06 | Sec 4.3          | Symmetry, monotonicity, binarity of the four morphism stages |
| 07 | Sec 4.1 + Paper 1| Element-S table in [0,1]³; F\_CB finite; trit address deterministic |
| 08 | Sec 1.3          | Atomistic GLB satisfies all 5 roles; ribbon GLBs satisfy only role 4 |

## Calibration notes

- **04 iron coordination shell**: ranges used are
  Fe-N\_porphyrin = 1.95–2.10 Å, Fe-S\_thiolate = 2.20–2.35 Å,
  Fe-O\_oxy-complex = 1.75–1.90 Å, taken from the cytochrome P450
  crystallographic literature (Denisov et al. 2005). The productive
  GLB satisfies all three with ≥ 4 N at 2.013–2.040 Å, S at 2.228 Å,
  and O at 1.814 Å.

- **05 state-4 oxy-complex**: the test discriminates against Compound I
  (Fe=O = 1.65 Å) by requiring the closest Fe-O distance to be ≥ 1.75 Å
  *and* nearest to either oxy-complex (1.80) or hydroperoxo (1.85).
  This is a classification, not a fit — there are no free parameters.

- **08 five roles**: the productive GLB satisfies all five operational
  criteria; the two ribbon GLBs satisfy only role 4 (interactive probe),
  consistent with the paper's claim that ribbon GLBs cannot be parsed
  to atomic resolution.

## What's NOT validated here (deferred)

- **PDB-companion lookups** — for ribbon-only GLBs the natural extension
  is RCSB API integration (Section 7.1 of the paper). Deferred.
- **Backbone vs side-chain distinction** — requires PDB residue
  metadata. Deferred.
- **Bond-order detection** — current `bond_inference` is geometric only;
  chemistry-aware (single/double/aromatic) bond classification deferred.
- **Cross-GLB consistency of Fe placement across CYP isoforms** — only
  one productive GLB is currently available; multi-GLB consistency is
  reserved for Paper 14 (database recovery).

## Run

```
python validation/run_all.py
```

Each script in `scripts/` can also be run standalone; results are
written to `results/<script_stem>.json`.
