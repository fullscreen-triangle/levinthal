# Paper 1 Validation Suite

Validation scripts for *An Expression Algebra for Biomolecules: The Receiver
$\mathcal{R}_{\mathrm{bio}}$ as a Type-Safe Morphism Chain*.

## Structure

```
validation/
├── README.md
├── run_all.py
├── scripts/
│   ├── 01_floor_theorem.py            # Theorem 7.1: Floor positive, ~3.7e-4
│   ├── 02_capacity_selection.py       # Theorem 2.6: C(n) = 2n^2; Eq. (1) selection rules
│   ├── 03_cycle_closure.py            # Theorem 6.4: Phi cycle closes mod Floor
│   ├── 04_amino_acid_coords.py        # 20 AA coords in [0,1]^3, depth-9 unique
│   ├── 05_tau_assignment.py           # Theorem 11.1: tau = sign(Delta_Pi)
│   ├── 06_spin_crossover.py           # Theorem 12.1, Cor. 12.2: 7 catalytic states
│   ├── 07_kuramoto_sync.py            # Sec. 5.2: temporal axis sub-evaluation
│   └── 08_morphism_chain.py           # Theorems 8.1, 8.2: type safety + S-conservation
└── results/
    ├── _summary.json                  # Aggregate report
    └── NN_*.json                      # Per-validation JSON outputs
```

## Running

```bash
# Run everything
python run_all.py

# Run a single validation
python scripts/01_floor_theorem.py
```

## Dependencies

- Python 3.10+
- numpy
- (no other external dependencies; all PDB-derived data is embedded or
  synthetic for the foundation-paper validations)

## What is validated

Each script's docstring states the theorem/equation it validates. The
scripts themselves are self-contained and emit structured JSON to
`results/<script_name>.json`.

The aggregate `_summary.json` reports per-script verdicts (PASS/FAIL/ERROR)
plus per-check counts.

## What is NOT validated here

Three categories of claim are deferred to subsequent monograph papers:

1. **Crystal-structure-grounded receiver evaluation** (CYP3A4 vs PDB 1TQN).
   Requires implementation of `levinthal-sexpr-bio` (Rust) and PDB ingestion.
   Defer to Paper 4 (CYP3A4 folding) and Paper 7 (resting/substrate-bound
   states).

2. **Multi-hop electron trajectory** (NADPH -> FAD -> FMN -> heme).
   Requires the trajectory-tracing machinery from azurin/SOD papers extended
   to chains. Defer to Paper 7 of the monograph.

3. **Compound I formation** (state 5 -> state 6).
   Requires bond-order partition coordinate and PCET trajectory machinery.
   Defer to Paper 8.

These deferrals are honestly bounded inside Paper 1's $\mathfrak{S}_{\mathrm{floor}}$ and
listed in Section 15 (Limitations).
