# Paper 7 Validation — Heteroatom Oxidation and Dealkylation

## Overview

8 validation scripts covering all quantitative claims in Paper 7.
Run from the paper root directory:

```
python validation/run_all.py
```

## Scripts

| Script | Topic | Result |
|--------|-------|--------|
| 01_alpha_carbon_bde | BDE ordering and DeltaM scaling | PASS |
| 02_n_dealkylation_rate | N-dealkylation intrinsic rate | PASS |
| 03_kie_dealkylation | KIE comparison N-dealk vs aliphatic | PASS |
| 04_s_oxidation_direct | S-oxidation direct O-atom transfer | PASS |
| 05_n_oxide_formation | N-oxide formation kinetics | PASS |
| 06_rate_ordering | Full rate hierarchy (5 reactions) | PASS |
| 07_carbinolamine_intermediate | Carbinolamine lability | PASS |
| 08_competitive_inhibition | Ketoconazole vs lidocaine | PASS |

## Key Results

- BDE ordering: N-CH3 (87) < O-CH3 (92) < aliphatic (100 kcal/mol)
- Rate ordering: k_S-ox (7.6e9) > k_N-ox (7.3e9) > k_N-dealk (6.1e9) > k_O-dealk (5.6e9) > k_aliphatic (5.2e9) s^-1
- KIE_N-dealk ≈ 6.7 (< KIE_aliphatic 7.7, due to softer alpha-C-H at 2800 cm^-1)
- S-oxidation: KIE = 1.0 (no H motion)
- Ki_ketoconazole < 100 nM vs CYP3A4

## Overall: 8/8 PASS
