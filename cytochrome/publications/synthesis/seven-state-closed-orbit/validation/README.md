# Paper 12 Validation

**Paper**: The Seven-State Catalytic Cycle as Closed Categorical Orbit

## Running the validation

```bash
cd cytochrome/publications/synthesis/seven-state-closed-orbit
python validation/run_all.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `01_seven_states.py` | Define 7 states with DM values; verify range and uniqueness |
| `02_rate_limiting.py` | Rate analysis; T_return > 100 ns; slowest step DM > 0.5 |
| `03_orbit_closure.py` | Closed orbit; DM sum in [4.5, 6.0]; all d_C=1 |
| `04_non_identity.py` | Newton's Cradle non-identity; Hamming distance >= 1 |
| `05_poincare_return.py` | T_return in [1 ns, 1 ms]; k_cat_intrinsic > 1e5 s^-1 |
| `06_anharmonic_check.py` | No sink points; all DM < 10; protein lifetime check |
| `07_rate_hierarchy.py` | k_chem/k_ET >= 100; chemistry not rate-limiting |
| `08_full_cycle_summary.py` | Comprehensive 8/8 summary |

## Expected output

All 8 scripts pass (8/8 PASS).
