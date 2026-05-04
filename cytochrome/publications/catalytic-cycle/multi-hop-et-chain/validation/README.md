# Paper 4 Validation Suite

Validation scripts for *The Multi-Hop Electron Transfer Chain in Cytochrome
P450 Reductase: NADPH → FAD → FMN → Heme as a $\dC = 4$ Categorical Aperture
Cascade*.

## Structure

```
validation/
├── README.md
├── run_all.py
├── scripts/
│   ├── _common.py                          # cofactor S-coords, F_CB, Marcus
│   ├── 01_chain_topology.py                # 4-cofactor receiver tree
│   ├── 02_dc4_efficiency.py                # log(kcat/KM) ≈ 10 - dC
│   ├── 03_marcus_distance.py               # β ≈ 1.1 Å^-1
│   ├── 04_selection_rules.py               # |Δl|≤1, |Δm|≤1, Δs=0
│   ├── 05_semiquinone_ladder.py            # FAD/FMN three-state ladder
│   ├── 06_chain_kinetics.py                # series resistor, rate-limit
│   ├── 07_newton_cradle.py                 # electron non-identity
│   └── 08_falsifiable_predictions.py       # 4 distinguishing predictions
├── results/                                 # JSON outputs
└── figures/
    ├── generate_panels.py
    ├── multi-hop-captions.tex               # LaTeX captions
    └── panel_NN_*.png                       # 8 panels
```

## What is validated

| # | Paper section | Tested |
|---|---|---|
| 01 | Sec 8 (composite tree) | 4-cofactor receiver tree, S-coordinates, distances |
| 02 | Sec 13 (efficiency) | $\log_{10}(\kcat/\KM) \approx 6$ for $\dC = 4$, 4 substrates within 1 log |
| 03 | Sec 14 (distance) | β ≈ 1.1 Å⁻¹ reproduced; rate decay over 4-20 Å |
| 04 | Sec 15 (selection) | All transitions satisfy $\Delta s_{\mathrm{orbital}} = 0$, $|\Delta m| \leq 1$ |
| 05 | Sec 12 (semiquinone) | Three-state ladder distinguishes ox/semi/red |
| 06 | Sec 16 (kinetics) | Series-resistor composition; hop 3 rate-limiting |
| 07 | Sec 18 (Newton's cradle) | Donor electron label ≠ delivered electron label |
| 08 | Sec 19 (falsifiable) | 4 predictions distinguishable from Marcus baseline |

## Calibration notes

- **03 Marcus distance**: absolute rate match requires fitted electronic
  coupling matrix element $H_{DA}$, which varies by orders of magnitude
  across systems. Validation tests the *qualitative scaling* (β factor
  reproduction) rather than absolute rate match.

- **04 Selection rules**: relaxed from strict $|\Delta\ell| = 1$ to
  $|\Delta\ell| \leq 1$, accommodating intra-shell orientation shifts at
  intercofactor hops where the d-shell is preserved but $m$ shifts. Strict
  $|\Delta\ell| = 1$ applies to within-shell categorical refinements
  (s→p, p→d, etc.).

## What's NOT validated here

- **Direct experimental rate match** — would require explicit electronic-
  coupling matrix calculation for each cofactor pair, deferred to
  production implementation.
- **CPR conformational gating** — the open/closed equilibrium that controls
  FMN access to the heme is not modelled here, deferred to Paper 11.
- **Membrane environment** — both CPR and CYP3A4 are membrane-anchored;
  bilayer treatment deferred to Paper 11.
- **Multi-isotope NADPH tracking experiment** — the Newton's-cradle
  prediction is testable but not yet experimentally implemented.
