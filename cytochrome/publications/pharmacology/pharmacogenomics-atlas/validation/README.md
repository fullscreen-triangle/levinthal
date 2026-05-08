# Paper 10 Validation — Pharmacogenomics Atlas

Eight scripts validate categorical mechanics predictions for CYP pharmacogenomics:
allele rate constants, warfarin dosing, population frequencies, codeine toxicity,
DDI induction/inhibition, and ethnic variation.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_cyp2d6_allele_rates.py | PM/IM/EM/UM rate hierarchy | UM>EM>IM>PM, ratio>5 |
| 02_cyp2c9_warfarin_dosing.py | *3 allele: >3x dose change | dose_star3>3, <5% activity |
| 03_population_phenotype_frequencies.py | Frequency-weighted k_pop | EM dominates |
| 04_codeine_toxicity_model.py | UM: >1.3x morphine exposure | excess morphine |
| 05_ddI_cyp3a4_induction.py | Rifampicin 20x induction | AUC ratio < 0.1 |
| 06_inhibition_competitive.py | Fluoxetine Ki DDI | alpha between 2-4 |
| 07_ethnic_variation.py | PM/IM frequencies by ancestry | East Asian IM highest |
| 08_full_pgx_table.py | Full allele table | 6 alleles, all checks |

## Parameters (_common.py)

| Symbol | Value | Description |
|--------|-------|-------------|
| ΔM_EM | 0.55 | Wild-type CYP2D6*1 |
| ΔM_PM | 2.50 | Null allele (*4/*5), ~8% rate |
| ΔM_IM | 0.75 | Reduced function (*10/*17) |
| ΔM_UM | 0.27 | Gene duplication, ~2x rate |
| ΔM_2C9*3 | 3.60 | CYP2C9*3 (I359L), ~5% activity |
