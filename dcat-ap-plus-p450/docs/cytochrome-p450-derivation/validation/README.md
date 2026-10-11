# Validation experiments

Experiments behind the two manuscripts in this folder:

- `f*`: *positive-occupation-propagation* (the foundation)
- `g*`: *p450-derivation*

Each experiment does three things:
- writes one JSON record to `results/`;
- writes its figure panels (PDF and PNG) to the `figures/` directory of its manuscript;
- caches what it downloads under `data/`, with source URL, sha256 and retrieval date recorded in `data/manifest.json`.

```bash
uv sync
uv run python run_all.py          # everything, about 4 minutes
uv run python run_all.py g6       # one experiment, by prefix
```

| Experiment | Question | Data |
|---|---|---|
| `f1_partial_sums` | Do counted costs give strictly increasing, unbounded partial sums? Do real costs? What does the dichotomy cost in commitments? | simulated |
| `f2_individuation` | Does the separation cost from the medium ever decrease as a contact graph grows? Does the smallest weight keep a floor? | simulated graphs |
| `f3_recurrence` | In reversible finite dynamics, does the configuration recur while a register only grows? | Kac ring, cat map |
| `f4_ladders` | Does a ladder's gap close in finitely many rungs? What happens without a floor on rung power? | exact and floating |
| `g1_atoms` | Do counted subshell capacities and the filling order reproduce the period lengths and the largest ionization-energy drops? What is iron's d shell? | NIST data via `mendeleev` |
| `g2_backbone` | Peptide bond length, planarity, backbone dihedrals | 12 P450 PDB entries |
| `g3_boundary` | Do ER-membrane P450s carry a bilayer-spanning N-terminal segment that bacterial P450s lack? | 1,919 reviewed UniProt P450s |
| `g4_record` | Shortest codon word; substitution tolerance of the standard code against 200,000 permuted codes | NCBI table 1 |
| `g5_fold` | Compaction, burial of the heme, separation cost in an integer-weighted contact graph | 20 PDB entries |
| `g6_ligation` | Proximal ligand of every heme iron; the cysteine-loop signature across the P450 family and other heme proteins | 20 PDB entries, PROSITE PS00086, UniProt |

## What the data can and cannot show

- **Restrained geometry.** Bond lengths and angles in crystal structures are restrained during refinement. `g2` therefore reports them to fix geometry, not as independent tests.
- **ω versus φ/ψ.** The peptide dihedral ω is weakly restrained. The backbone dihedrals φ and ψ are not restrained.
- **The filling order is an input.** The (n+l, n) filling order used in `g1` is the empirical Madelung rule. It is not derived anywhere in these experiments.
- **Location annotations.** Subcellular locations and taxonomy in `g3` are UniProt annotations taken as given.
- **Which iron is analysed.** `g5` and `g6` use the first heme iron of each entry, except where all sites are listed.
