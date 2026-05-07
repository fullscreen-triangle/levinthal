# Paper 11 Validation

**Paper**: Membrane Anchoring and Partner Coupling in Cytochrome P450

## Running the validation

```bash
cd cytochrome/publications/construction/membrane-cofactor-cpr
python validation/run_all.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `01_tm_helix_insertion.py` | TM helix insertion energy and categorical depth |
| `02_cpr_binding.py` | CPR-P450 binding affinity (K_d = 0.1 uM) |
| `03_fmn_heme_distance.py` | FMN to heme electron transfer rate |
| `04_cytb5_comparison.py` | Cytochrome b5 vs CPR kinetics |
| `05_membrane_enrichment.py` | Substrate enrichment near ER membrane |
| `06_complex_stoichiometry.py` | CPR:P450 ratio and turnover analysis |
| `07_proximal_face_electrostatics.py` | P450 proximal face charge complementarity |
| `08_full_complex_validation.py` | Comprehensive 8/8 summary check |

## Expected output

All 8 scripts pass (8/8 PASS).
