# Paper 8 Validation — Atypical Reactions Atlas

Eight scripts validate all quantitative predictions for desaturation, epoxidation,
NIH shift, nucleophilic aldehyde oxidation, and carbene insertion.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_desaturation_two_step.py | Two-step HAT effective rate | k_eff < k_single, fractions sum |
| 02_desaturation_kie.py | KIE from ZPE for desaturation | KIE in 3–9 range |
| 03_arene_oxide.py | Epoxidation rate and no-KIE | k_epox in range, KIE=1 |
| 04_nih_shift.py | NIH shift cationic rearrangement | k_NIH > k_epox, secondary KIE only |
| 05_nucleophilic_aldehyde.py | Nucleophilic O-atom transfer | k_nuc in range, no KIE |
| 06_rate_ordering.py | ΔM monotonic ordering | NIH fastest, desaturation slowest |
| 07_product_partitioning.py | Phenol vs dihydrodiol partitioning | phenol dominant |
| 08_full_validation_table.py | All 5 types vs literature ranges | all within lit range, KIE correct |

## Parameters (_common.py)

| Symbol | Value | Description |
|--------|-------|-------------|
| ν_floor | 1×10¹⁰ s⁻¹ | Attempt frequency |
| ΔM_desat_1 | 0.65 | First HAT activation depth |
| ΔM_desat_2 | 0.55 | Second HAT activation depth |
| ΔM_epox | 0.35 | Epoxidation activation depth |
| ΔM_NIH | 0.18 | NIH shift activation depth |
| ΔM_nuc | 0.42 | Nucleophilic O-transfer depth |
| ΔM_carbene | 0.20 | Carbene insertion depth |
| K_rebound | 7.4×10⁹ s⁻¹ | Rebound rate (from Paper 6) |
