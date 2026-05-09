# Paper 15 Validation — Polymorphisms, DDI, and Inhibitors

Eight scripts validate the categorical mechanics framework applied to CYP
polymorphism ΔM shifts, competitive and mechanism-based inhibition, CYP
induction, and compound phenotype–inhibitor effects.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_polymorphism_dm_shift.py | Allele ΔM values for CYP2D6/2C9/3A4 | k ratios, <5% activity for *3 |
| 02_competitive_inhibition_alpha.py | α modulus; apparent ΔM shift = ln(α) | ketoconazole/quinidine strong DDI |
| 03_mbi_inactivation.py | Kitz-Wilson MBI kinetics; t½ under MBI | kobs > kdeg; f_active < 50% |
| 04_induction_fold.py | Rifampicin 20×; PXR E_max model | AUC ratio < 10%; ddm ordering |
| 05_inhibitor_ranking.py | Ki ranking; DDI risk classification | ≥3 strong inhibitors |
| 06_compound_phenotype_ddi.py | PM + inhibitor; population-weighted rate | EM+quin below PM rate |
| 07_tdi_ic50_shift.py | TDI IC50 shift at 60 min preincubation | clarithromycin/diltiazem TDI+ |
| 08_full_ddi_table.py | Three-level summary table | all 13 checks |
