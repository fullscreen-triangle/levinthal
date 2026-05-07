# Paper 9 Validation

**Paper**: The 57 Human CYP Isoforms as Address-Manifold Variants

## Running

```bash
python validation/run_all.py
```

## Scripts

| Script | Claim | Expected |
|--------|-------|----------|
| 01_address_clustering | Mean pairwise distinctness at k=6 | >= 0.95 |
| 02_family_separation | Inter-family > intra-family at k=3 | ratio > 1.5 |
| 03_substrate_promiscuity | sigma ordering: 3A4 > 2D6 > 2C9 | 3.2 > 1.8 > 1.4 |
| 04_affinity_prediction | Substrate/non-substrate affinity ratio | > 2.0 |
| 05_delta_m_isoform_shift | k_2D6/k_3A4 in [0.85, 0.97] | 0.923 |
| 06_tissue_distribution | CYP3A4 > CYP1A2 gut; CYP1A1 > CYP3A4 lung | 80>20; 70>30 |
| 07_57_isoforms_distinct | All pairwise Hamming >= 1; count == 57 | min=1; n=57 |
| 08_validation_summary | 8/8 PASS | 8/8 |

## Expected result

```
Paper 9 validation: 8/8 PASS
Overall: PASS
```
