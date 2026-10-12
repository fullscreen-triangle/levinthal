// Values for the application slides, taken from the monograph papers in
// cytochrome/publications. Each value is marked by where it comes from:
// "M" a value computed in the paper's model, "L" a literature value the
// paper compares with, "C" measured in this work from public data.

export const PAPERS = {
  glb: "GLB-based structural input",
  manifold: "P450 address manifold + CYP3A4 fold",
  isoforms: "The 57 human P450 isoforms",
  taxonomy: "57-isoform taxonomy",
  equilibrium: "Resting and substrate-bound CYP3A4",
  membrane: "Membrane anchoring and partner coupling",
  et: "Electron-transfer chain through CPR",
  cpdI: "Compound I formation",
  ch: "C–H activation and rebound",
  hetero: "Heteroatom oxidation and dealkylation",
  atypical: "Atypical reactions atlas",
  orbit: "Seven-state closed cycle",
  spectra: "Spectroscopic atlas",
  ddi: "Polymorphisms, DDI and inhibitors",
  pgx: "Pharmacogenomics atlas",
  database: "Database recovery",
  dictionary: "Empty dictionary for P450",
};

// Human CYP genes by family (Nelson nomenclature), 18 families, 57 genes.
export const HUMAN_CYPS = {
  CYP1: ["1A1", "1A2", "1B1"],
  CYP2: ["2A6", "2A7", "2A13", "2B6", "2C8", "2C9", "2C18", "2C19", "2D6", "2E1", "2F1", "2J2", "2R1", "2S1", "2U1", "2W1"],
  CYP3: ["3A4", "3A5", "3A7", "3A43"],
  CYP4: ["4A11", "4A22", "4B1", "4F2", "4F3", "4F8", "4F11", "4F12", "4F22", "4V2", "4X1", "4Z1"],
  CYP5: ["5A1"],
  CYP7: ["7A1", "7B1"],
  CYP8: ["8A1", "8B1"],
  CYP11: ["11A1", "11B1", "11B2"],
  CYP17: ["17A1"],
  CYP19: ["19A1"],
  CYP20: ["20A1"],
  CYP21: ["21A2"],
  CYP24: ["24A1"],
  CYP26: ["26A1", "26B1", "26C1"],
  CYP27: ["27A1", "27B1", "27C1"],
  CYP39: ["39A1"],
  CYP46: ["46A1"],
  CYP51: ["51A1"],
};

export const GLB_FE = [
  // label, GLB model value(s) in the paper, PDB measured in this work (12 P450s)
  { bond: "Fe–N (pyrrole)", glb: [2.01, 2.04], pdb: [1.98, 2.10] },
  { bond: "Fe–S (Cys)", glb: [2.228, 2.228], pdb: [1.90, 2.52], pdbMean: 2.32 },
  { bond: "Fe–O (axial, oxy state)", glb: [1.814, 1.814], lit: [1.80, 1.85] },
];

export const DEPTHS = [
  // smallest depths whose capacity 3^k covers each set (a counting bound)
  { k: 3, cap: 27, need: 18, what: "families" },
  { k: 4, cap: 81, need: 57, what: "human isoforms" },
  { k: 9, cap: 19683, need: null, what: "allelic variants" },
];

export const STATES = [
  // name, short, iron, spin, Soret (nm) as assigned in the spectroscopic atlas
  { id: 1, name: "Resting", label: "Fe(III)–OH₂", spin: "low spin", soret: 417, lit: "417–420" },
  { id: 2, name: "Substrate-bound", label: "Fe(III)", spin: "high spin", soret: 390, lit: "≈390" },
  { id: 3, name: "Ferrous", label: "Fe(II)", spin: "high spin", soret: 408 },
  { id: 4, name: "Oxy", label: "Fe(II)–O₂", spin: "", soret: 418 },
  { id: 5, name: "Peroxo", label: "Fe(III)–OO⁻", spin: "", soret: 440 },
  { id: 6, name: "Compound 0", label: "Fe(III)–OOH", spin: "", soret: 367 },
  { id: 7, name: "Compound I", label: "Fe(IV)=O π•⁺", spin: "S = ½", soret: 367, lit: "367 (Rittle & Green)" },
];

export const ORBIT_DM = [
  ["1→2", "substrate binding", 0.92],
  ["2→3", "first electron", 0.68],
  ["3→4", "O₂ binding", 0.55],
  ["4→5", "second electron", 0.72],
  ["5→6", "protonation", 0.45],
  ["6→7", "O–O heterolysis", 0.693],
  ["7→P", "C–H activation", 0.65],
  ["P→1", "product release", 0.30],
];

export const HETERO_DM = [
  ["S-oxidation", 0.28, "direct O transfer"],
  ["N-oxidation", 0.32, "direct O transfer"],
  ["N-dealkylation", 0.50, "via H abstraction"],
  ["O-dealkylation", 0.58, "via H abstraction"],
];

export const ATYPICAL = [
  { name: "NIH shift", dm: 0.18, kie: "≈1", primary: false },
  { name: "Carbene insertion", dm: 0.20, kie: "≈1", primary: false },
  { name: "Epoxidation", dm: 0.35, kie: "≈1", primary: false },
  { name: "Nucleophilic O-transfer", dm: 0.42, kie: "≈1", primary: false },
  { name: "Desaturation (two HATs)", dm: null, keff: 2.3e9, kie: "≈ 5", primary: true },
];

export const CYP2D6 = [
  { phenotype: "Poor (PM)", alleles: "*4/*5", dm: 2.5, rel: 0.14, freq: 0.07 },
  { phenotype: "Intermediate (IM)", alleles: "*10/*17", dm: 0.75, rel: 0.82, freq: 0.30 },
  { phenotype: "Normal (EM)", alleles: "*1/*2", dm: 0.55, rel: 1.0, freq: 0.55 },
  { phenotype: "Ultrarapid (UM)", alleles: "*1×N", dm: 0.27, rel: 1.32, freq: 0.08 },
];

export const INHIBITORS = [
  { name: "Ketoconazole", target: "CYP3A4", ki: 0.037, conc: 0.2 },
  { name: "Quinidine", target: "CYP2D6", ki: 0.027, conc: 0.5 },
  { name: "Fluoxetine", target: "CYP2D6", ki: 0.24, conc: 0.5 },
];

export const CLARITHROMYCIN = { KI: 3.7, kinact: 0.04, conc: 4 };
