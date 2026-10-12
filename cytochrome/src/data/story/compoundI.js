// Numbers for the Compound I page, read from
// publications/catalytic-cycle/compound-i-formation/validation (scripts/_common.py
// and results/0N_*.json). "set" = a constant typed into the scripts; "computed"
// = produced by the scripts from those constants; "lit" = a value the paper
// compares with, as cited there.

// Morse potential for O–O used by script 05 (set): D_e kcal/mol, r_e Å, a 1/Å.
export const MORSE = { De: 60, re: 1.45, a: 2.0 };

// Minimum distance of return to the start, ten trajectories of script 05
// (velocity Verlet, dt 0.1, 1000 steps), in Å (computed).
export const RETURN_MIN = [3.891e-5, 9.316e-6, 4.59e-7, 1.447e-5, 1.282e-5, 2.241e-7, 1.185e-5, 1.964e-6, 2.059e-6, 2.91e-5];

// Rate rule of the paper: log10 k = 10 − d_C (set).
export const DC_RULE = [1, 2, 3, 4, 5, 6].map((d) => ({ d, logk: 10 - d }));

// KIE: concerted and sequential values are set in script 04; the paper
// compares with 1.5–2.0, attributed to Vatsis 2002 (citation to be checked).
export const KIE = { concerted: 2.0, sequential: 6.0, lit: [1.5, 2.0] };

// Lifetime τ = (h/k_B T)·exp(E_a/k_B T) with E_a = 11.0 kcal/mol (set), ms.
export const LIFETIME_EA = 11.0;
export const LIFETIME_PTS = [
  { T: 193, model: 113.158, lit: 210, litNote: "CYP119, Rittle & Green 2010" },
  { T: 310, model: 0.00140169, lit: 1.0, litNote: "an estimate from turnover, not a measurement", estimate: true },
];

// Oxidation potential E = (k_B T/e)·n_eff·ΣΔM, T = 310 K, n_eff = 5 (set).
export const POTENTIAL = {
  parts: [
    ["spin-state change", 0.92, 0.1229],
    ["oxidation-state change", 1.0, 0.1336],
    ["Fe=O bond formation", 1.5, 0.2004],
    ["porphyrin radical", 0.6, 0.0801],
  ],
  total: 0.5369,
  lit: 0.9, // Green 2009, as cited
  neff: 5,
  perNeff: 0.10738, // V per unit n_eff at ΣΔM = 4.02
};

// The paper's observable table: "listed" is the value in the prediction
// column, "computed" what the scripts produce where they produce anything.
export const OBSERVABLES = [
  { name: "Soret band", unit: "nm", listed: 367, lit: 367, computed: null },
  { name: "Mössbauer δ", unit: "mm/s", listed: 0.1, lit: 0.11, computed: null },
  { name: "Mössbauer ΔE_Q", unit: "mm/s", listed: 0.85, lit: 0.9, computed: null },
  { name: "EPR g", unit: "", listed: 1.99, lit: 1.99, computed: null },
  { name: "total spin", unit: "", listed: 0.5, lit: 0.5, computed: null },
  { name: "oxidation potential", unit: "V", listed: 0.9, lit: 0.9, computed: 0.537 },
  { name: "lifetime, 193 K", unit: "ms", listed: 200, lit: 210, computed: 113 },
  { name: "lifetime, 310 K", unit: "ms", listed: 1.0, lit: 1.0, computed: 0.0014, litEstimate: true },
];
