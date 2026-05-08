# Paper 14 Validation — P450 Database Recovery

Eight scripts validate the ternary address encoding's capacity to recover
P450 sequence and classification information from partial or corrupted databases.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_information_capacity.py | Trit bits exceed log₂N at each level | k=3,6,9 sufficient |
| 02_recovery_accuracy_vs_depth.py | Accuracy = 1 - exp(-bits/H) | monotonic, <1 |
| 03_partial_address_recovery.py | 70% address → unique isoform | bits exceed threshold |
| 04_sequence_reconstruction_fidelity.py | Fidelity at k=6,9 | >85% at k=6 |
| 05_missing_data_interpolation.py | Interpolation error + compression | 40x compression |
| 06_cross_species_recovery.py | Human CYPs > bacterial recovery | within-family better |
| 07_pharmvar_allele_capacity.py | k=9 covers 310 PharmVar alleles | 3^9=19683≫310 |
| 08_full_recovery_table.py | Three-level summary table | all checks |
