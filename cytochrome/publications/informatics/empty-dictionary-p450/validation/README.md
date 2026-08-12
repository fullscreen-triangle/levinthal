# Validation — The Empty Dictionary for Cytochrome P450

Run everything:

```bash
python run_all.py
```

Exit code is non-zero if any verdict is `FAIL`. No third-party dependencies;
standard library only (the rank correlation in `02` is implemented locally so
that SciPy is not required).

## Experiments

| # | Script | Claim under test | Control |
|---|--------|------------------|---------|
| 01 | `01_storage_scaling.py` | Resident state is O(1) in the number of addressable objects | The per-sequence arrangement must *grow* over the same corpora, or the comparison is empty |
| 02 | `02_query_without_entries.py` | Queries are answered for unstored objects, by evaluation, and the answers carry physicochemical content | **A**: an exact-match dictionary over a disjoint training corpus. **B**: 200 shuffled coordinate tables — same size, same address algebra, wrong assignment |
| 03 | `03_paper2_reconciliation.py` | Paper 2's O(N·k) figure measures an optional cache, not the resident state | Answers are compared byte-for-byte with the cache present and deleted; if they differ, the cache is load-bearing and the claim fails |

## Measured results

| Quantity | Value |
|---|---|
| Resident state | **561 bytes** (20 rows + encoding rule), constant across corpora from 1 to 10⁹ sequences |
| Paper 2 per-sequence figure, recomputed | 7.13 kbits (paper states ~7 kbits) |
| Paper 2 corpus figure, recomputed | 356.6 MB (paper states ~350 MB) |
| Answerability, empty scheme | 6/6 queries |
| Answerability, control A (dictionary) | 0/6 queries |
| Graded task, real table | Spearman ρ = **0.382** over 190 residue pairs |
| Graded task, control B (200 shuffles) | mean ρ = −0.004, max ρ = 0.252, empirical *p* < 0.005 |
| Answers with cache vs cache deleted | identical |

## A note on experiment 02

The first version of this script used nearest-neighbour retrieval over closely
related P450 fragments, and **failed its own control**: the shuffled coordinate
table scored a perfect 1.00, exactly matching the real table.

That was a defective task, not a refuted claim. The fragments sit at ~94%
sequence identity, so the nearest neighbour is decided by string identity
alone — *any* injective residue→address map ranks the near-identical pair
first, whether or not its coordinates mean anything.

The task was replaced with graded physicochemical proximity, which sequence
identity cannot carry and a shuffled table must destroy. The original task is
retained in the output under `task_identity_NON_DISCRIMINATING`, reporting its
own *p* = 1.000, so the record shows which tasks discriminate and which do not.
