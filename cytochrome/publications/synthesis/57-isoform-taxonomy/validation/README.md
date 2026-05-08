# Paper 9 Validation — 57 Human CYP Isoform Taxonomy

Eight scripts validate the ternary address encoding of the 57 human CYP isoforms
across 18 gene families, their substrate ΔM ranges, CYP3A4 fold depth, and drug
metabolism contributions.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_ternary_depth_families.py | Trit depth for 18 families / 57 isoforms | 3^3≥18, 3^6≥57 |
| 02_family_substrate_dm.py | Per-family ΔM ranges | CYP3A4 widest, CYP2E1 highest |
| 03_cyp3a4_fold_depth.py | CYP3A4 fold from log₃(503 aa) | ≈5.69 steps |
| 04_capacity_shell_rule.py | C(n)=2n² shell capacity | exact for n=1..5 |
| 05_isoform_rate_spread.py | ΔM spread within CYP2C/2D | σ < 0.10 |
| 06_drug_metabolism_fractions.py | 3A4+2D6+2C9 ≥ 80% of drugs | empirical fractions |
| 07_substrate_volume_dm.py | Substrate volume vs ΔM | negative correlation |
| 08_full_taxonomy_table.py | All 57 isoforms, 18 families | ternary capacity |

## Parameters (_common.py)

| Symbol | Value | Description |
|--------|-------|-------------|
| N_HUMAN_CYPS | 57 | Human CYP isoforms |
| N_FAMILIES | 18 | Nelson gene families |
| DEPTH_FAMILY | 3 | Trit depth for family separation |
| DEPTH_ISOFORM | 6 | Trit depth for isoform separation |
| FAMILY_RECALL | 0.94 | Taxonomy recall at k=3 |
| ISOFORM_DISTINCT | 0.97 | Distinctness at k=6 |
