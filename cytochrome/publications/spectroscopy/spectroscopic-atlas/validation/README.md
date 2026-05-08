# Paper 13 Validation — Spectroscopic Atlas of P450 States

Eight scripts validate spectroscopic predictions across all 7 P450 catalytic
cycle states: UV-Vis Soret peaks, EPR g-values, resonance Raman Fe=O stretch,
spin-state equilibria, and circular dichroism.

## Run

```
cd validation
python run_all.py
```

Expected output: `8/8 PASS`

## Scripts

| Script | Topic | Checks |
|--------|-------|--------|
| 01_soret_band_positions.py | Soret peaks for 7 states | 414–420 nm resting, HS blueshifted |
| 02_epr_g_values.py | EPR g-values LS/HS | g1>g2>g3, HS at g>6 |
| 03_raman_feo_stretch.py | Fe=O Raman 795 cm⁻¹ + ¹⁸O shift | shift 30–50 cm⁻¹ |
| 04_spin_state_equilibrium.py | LS/HS ΔG model | HS favored after substrate binding |
| 05_absorbance_dm_correlation.py | Soret energy vs ΔM_spec | r > 0.9 |
| 06_cd_spectrum_chirality.py | CD secondary structure | θ_222 < 0, θ_195 > 0 |
| 07_spectral_discrimination.py | Can UV-Vis alone distinguish all 7? | EPR/Raman needed for spin/Cpd I |
| 08_full_spectral_table.py | Complete atlas table | 7 states, Raman signal for Cpd I |
