// ---------------------------------------------------------------------
// VENDORED, UNMODIFIED except for import paths.
// Source: cytochrome/src/data/lessons.js
//
// This is the existing Shakespeare interpreter, not a reimplementation.
// Keeping it byte-faithful means the notebook and the cytochrome IDE
// perform the same plays and get the same clock, which is the point:
// two front ends over one receiver.
// ---------------------------------------------------------------------

// =====================================================================
//  Shakespeare Sandbox — lesson ladder
//
//  Eleven .shk plays of increasing categorical depth, each performing
//  one paper of the Cytochrome P450 monograph. Every lesson carries:
//    - src      : the Shakespeare program (the play)
//    - paper    : the monograph paper it performs
//    - oracle   : the validated numbers it is checked against (baked
//                 from cytochrome/publications/.../validation/results)
//    - charts   : which D3 chart components mount for this lesson
//    - requires : the lesson whose receiver state must precede it
//
//  State accumulates DOWN the ladder: the committed-cut count M is a
//  monotone clock (never resets), and measurement conserves the medium,
//  so lesson N+1 runs on the receiver lesson N left behind.
//
//  Oracle numbers are the monograph's own (verdict: PASS). See
//  shakespear/docs/shakespeare-sandbox-design.md.
// =====================================================================

export const RECEIVER_FLOOR = 3.7e-4; // R_bio floor estimate (expression-algebra paper)

export const LESSONS = [
  // ------------------------------------------------------------------ 0
  {
    id: "00_hello-receiver",
    n: 0,
    title: "Hello, receiver",
    subtitle: "the floor, one leaf, one measurement",
    paper: "Prefatory · monograph-introduction",
    verbs: ["receiver", "floor", "cut", "observe"],
    charts: [],
    requires: null,
    src: `-- 00_hello-receiver.shk
-- A Shakespeare play is performed, not simulated. Running it
-- MEASURES the protein. The cut count M is a clock that only
-- moves forward; the floor beta > 0 says nothing is free.

receiver bio            -- fix the biomolecular receiver R_bio
floor 3.7e-4            -- the contact floor; no separation is free

Fe := cut "Fe"          -- individuate one iron leaf against the medium
observe Fe              -- force the measurement now
`,
    // oracle: this lesson just establishes the clock + floor
    oracle: {
      verdict: "PASS",
      floor: 3.7e-4,
      note: "one leaf individuated; M advances by 1",
    },
  },

  // ------------------------------------------------------------------ 1
  {
    id: "01_sequence-address",
    n: 1,
    title: "The sequence address",
    subtitle: "CYP3A4 at categorical depth 3, 6, 9",
    paper: "Paper 1–2 · p450-sequence-space / address-manifold",
    verbs: ["residue", "address"],
    charts: ["depth-ladder", "s-entropy"],
    requires: "00_hello-receiver",
    src: `-- 01_sequence-address.shk
-- A protein's identity is an ADDRESS in S-entropy space [0,1]^3,
-- read at increasing categorical depth. CYP3A4 (UniProt P08684,
-- 503 residues) compresses from 4527 raw trits to a structure
-- address by secondary-structure aggregation.

receiver bio
floor 3.7e-4

CYP3A4 := residue "P08684"     -- the 503-residue chain as a leaf-tree

observe CYP3A4 as address
-- expect: family (depth 3) = 110
--         full   (depth 9) = 110112212
`,
    oracle: {
      verdict: "PASS",
      family_address_d3: "110",
      full_address_d9: "110112212",
      n_residues: 503,
      raw_trit_count: 4527,
      centroid: [0.545, 0.483, 0.316],
      note: "503 residues -> 4527 trits, compresses to a depth-9 address",
    },
  },

  // ------------------------------------------------------------------ 2
  {
    id: "02_fold-cyp3a4",
    n: 2,
    title: "The fold",
    subtitle: "native structure in six categorical steps",
    paper: "Paper 9 · address-manifold-cyp3a4-fold",
    verbs: ["fold"],
    charts: ["contact-map", "order-parameter"],
    requires: "01_sequence-address",
    src: `-- 02_fold-cyp3a4.shk
-- Folding is not a 3^N search. It is frequency matching: the
-- residue oscillators phase-lock, and the native state is the
-- unique global minimum of the Kuramoto energy. 'fold' runs the
-- morphism chain access o fuse o catalyze* o observe.

receiver bio
floor 3.7e-4

C := fold "1TQN"        -- the CYP3A4 crystal structure (PDB 1TQN)
observe C
-- expect: 13 helices, 5 sheets from the coupling eigenstructure
`,
    oracle: {
      verdict: "PASS",
      helices: 13,
      sheets: 5,
      loops: 8,
      n_elements: 26,
      coherence: 0.83,
      note: "native fold = Kuramoto energy minimum; 13 helix / 5 sheet",
    },
  },

  // ------------------------------------------------------------------ 3
  {
    id: "03_resting-state",
    n: 3,
    title: "The resting state",
    subtitle: "heme, thiolate, axial water — the receiver tree",
    paper: "Paper 3 · cyp3a4-resting-substrate-bound",
    verbs: ["cofactor", "solvent", "complex"],
    charts: ["contact-map"],
    requires: "02_fold-cyp3a4",
    src: `-- 03_resting-state.shk
-- The resting enzyme is a CLOSURE of cuts: the fold plus its
-- prosthetic heme, the absolutely-conserved Cys442 proximal
-- thiolate, and the axial water. 'complex' drives every
-- constituent to closure and returns the receiver tree.

receiver bio
floor 3.7e-4

E := complex CYP3A4(
       cofactor "heme",       -- prosthetic group leaf
       residue  "CYS442",     -- proximal thiolate ligand
       solvent  "H2O_axial"   -- sixth axial ligand, low-spin Fe3+
     ) by CYS442

observe E
-- resting state: Fe3+ low-spin, S = 1/2
`,
    oracle: {
      verdict: "PASS",
      spin_state: "low-spin",
      S_total: 0.5,
      fe_oxidation: 3,
      axial_ligand: "H2O",
      proximal: "CYS442 thiolate",
      note: "resting Fe3+ LS receiver tree closes on Cys442",
    },
  },

  // ------------------------------------------------------------------ 4
  {
    id: "04_catalytic-cycle",
    n: 4,
    title: "The catalytic cycle",
    subtitle: "the seven-state closed orbit",
    paper: "Paper 14 · seven-state-closed-orbit",
    verbs: ["substrate", "track", "until", "converge", "yield"],
    charts: ["seven-state-orbit", "dm-bars"],
    requires: "03_resting-state",
    src: `-- 04_catalytic-cycle.shk
-- 'track ... until converge' IS the causal propagation table over
-- proteins. Dioxygen is tracked through the enzyme as a residue-
-- chain of cuts; the chain is admissible iff it S-entropy converges
-- and the orbit closes (state 7 -> state 1, product released).
-- Intermediate steps are free; only convergence is judged.

receiver bio
floor 3.7e-4

O2 := substrate "O2"

cycle := track O2 in E
           with reps mass, charge, spin   -- representation switching
           until converge
           yield amalgamation

observe cycle
-- expect: 7 states, 8 transitions, sum(delta M) = 4.963, closed
`,
    oracle: {
      verdict: "PASS",
      n_states: 7,
      n_transitions: 8,
      DM_sum: 4.963,
      orbit_closed: true,
      states: [
        "Resting Fe3+ H2O (LS)",
        "Substrate-bound Fe3+ (HS)",
        "Reduced Fe2+",
        "Oxy Fe2+-O2",
        "Peroxo Fe3+-OO2-",
        "Compound 0 Fe3+-OOH",
        "Compound I Fe4+=O radical",
      ],
      DM_values: [0.92, 0.68, 0.55, 0.72, 0.45, 0.693, 0.65, 0.3],
      DM_labels: [
        "substrate binding",
        "first electron",
        "O2 binding",
        "second electron",
        "protonation",
        "O-O heterolysis",
        "H-atom transfer",
        "product release",
      ],
      note: "closed orbit; DM sum in [4.5, 6.0]; Poincare return defined",
    },
  },

  // ------------------------------------------------------------------ 5
  {
    id: "05_compound-i",
    n: 5,
    title: "Compound I",
    subtitle: "a transient intermediate, licensed by closure",
    paper: "Paper 5 · compound-i-formation",
    verbs: ["track"],
    charts: ["dm-bars"],
    requires: "04_catalytic-cycle",
    src: `-- 05_compound-i.shk
-- Compound I (Fe4+=O porphyrin radical cation) lives ~ms and is
-- locally MAXIMALLY infeasible. The Miracle Principle licenses it:
-- a sub-step may be locally impossible provided the whole cycle
-- closes. We isolate its formation from Compound 0 by O-O
-- heterolysis and read its lifetime and oxidation potential.

receiver bio
floor 3.7e-4

cpdI := track cycle in E
          until converge
          yield state6      -- the Compound I excursion

observe cpdI
-- Fe4+=O, oxidation +4 (the unique excursion above +3)
`,
    oracle: {
      verdict: "PASS",
      fe_oxidation: 4,
      species: "Fe(IV)=O porphyrin radical cation",
      heterolysis_DM: 0.693,
      lifetime_ms: 1.0,
      licensed_by: "cycle closure (Miracle Principle)",
      note: "locally infeasible; global S = floor because orbit closes",
    },
  },

  // ------------------------------------------------------------------ 6
  {
    id: "06_ch-rebound",
    n: 6,
    title: "C–H activation & rebound",
    subtitle: "reaction and measurement are one cut",
    paper: "Paper 6 · ch-activation-rebound",
    verbs: ["catalyze"],
    charts: ["dm-bars"],
    requires: "05_compound-i",
    src: `-- 06_ch-rebound.shk
-- 'catalyze' turns a substrate over. A reaction step and a
-- measurement step are the SAME cut, differing only in whether the
-- residue's causal link is resolved. Compound I abstracts an H atom,
-- then the hydroxyl rebounds onto the carbon radical.

receiver bio
floor 3.7e-4

sub := substrate "R-H"          -- a C-H hydroxylation substrate

prod := catalyze sub in E yield product
-- H-atom transfer (abstraction) then oxygen rebound -> R-OH

observe prod
`,
    oracle: {
      verdict: "PASS",
      mechanism: "H-atom abstraction + oxygen rebound",
      HAT_DM: 0.65,
      product: "R-OH",
      reading: "reaction (link unresolved) == measurement (link resolved)",
      note: "one symmetric contact op read two ways",
    },
  },

  // ------------------------------------------------------------------ 7
  {
    id: "07_electron-chain",
    n: 7,
    title: "The electron-transfer chain",
    subtitle: "femtosecond multi-hop electronic leaves",
    paper: "Paper 4 · multi-hop-et-chain",
    verbs: ["electronic", "track"],
    charts: ["et-trace"],
    requires: "06_ch-rebound",
    src: `-- 07_electron-chain.shk
-- Electrons are their own leaf class, addressed by partition
-- coordinates (n, l, m, s). The reductase delivers two electrons
-- to the heme through a multi-hop chain, resolved to femtoseconds.

receiver bio
floor 3.7e-4

e1 := electronic "FAD -> FMN -> heme"

trace := track e1 in E
           until converge
           yield trajectory

observe trace
-- a time-indexed sequence of partition cells (fs resolution)
`,
    oracle: {
      verdict: "PASS",
      hops: ["FAD", "FMN", "heme-Fe"],
      timescale: "femtosecond",
      selection_rule: "delta s_orbital = 0 (conserved)",
      note: "electronic-leaf trajectory; QND under categorical observation",
    },
  },

  // ------------------------------------------------------------------ 8
  {
    id: "08_isoform-diversity",
    n: 8,
    title: "Isoform diversity",
    subtitle: "57 human isoforms at categorical depth 6",
    paper: "Paper 12 · 57-isoform-taxonomy",
    verbs: ["residue", "address"],
    charts: ["s-entropy", "depth-ladder"],
    requires: "02_fold-cyp3a4",
    src: `-- 08_isoform-diversity.shk
-- The 57 human P450 isoforms are the receiver evaluated at depth 6:
-- the Nelson family taxonomy resolves at depth 3, the individual
-- isoforms at depth 6. Each is one address in [0,1]^3.

receiver bio
floor 3.7e-4

CYP3A4 := residue "P08684"
CYP2D6 := residue "P10635"
CYP2C9 := residue "P11712"

observe CYP3A4 as address    -- family 3A
observe CYP2D6 as address    -- family 2D
observe CYP2C9 as address    -- family 2C
-- 57 isoforms separate cleanly at depth 6
`,
    oracle: {
      verdict: "PASS",
      n_isoforms: 57,
      family_depth: 3,
      isoform_depth: 6,
      examples: {
        CYP3A4: "110",
        CYP2D6: "021",
        CYP2C9: "020",
      },
      note: "families at depth 3, isoforms at depth 6",
    },
  },

  // ------------------------------------------------------------------ 9
  {
    id: "09_variant-effect",
    n: 9,
    title: "The variant effect",
    subtitle: "a point mutation as a re-cut",
    paper: "Paper 15 · polymorphisms-ddi / pharmacogenomics",
    verbs: ["mutate"],
    charts: ["contact-map", "dm-bars"],
    requires: "02_fold-cyp3a4",
    src: `-- 09_variant-effect.shk
-- A polymorphism re-cuts the coupling graph. The variant effect is
-- read from the change in folding convergence and coupling
-- eigenstructure -- an O(N^2) query against the folded receiver,
-- not a re-fold from scratch. (PharmVar variants live at depth 9.)

receiver bio
floor 3.7e-4

wt  := fold "1TQN"
mut := mutate wt by ARG -> CYS at 144    -- a CYP2C9-style *2 variant

observe mut
assert mut.coherence >= wt.coherence  emit "variant destabilises fold"
`,
    oracle: {
      verdict: "PASS",
      mutation: "ARG144CYS",
      wt_coherence: 0.83,
      mut_coherence: 0.79,
      effect: "reduced catalytic efficiency",
      note: "re-cut lowers coherence; PharmVar depth-9 resolution",
    },
  },

  // ------------------------------------------------------------------ 10
  {
    id: "10_db-recovery",
    n: 10,
    title: "Database recovery",
    subtitle: "the empty-dictionary engine",
    paper: "Paper 13 · database-recovery",
    verbs: ["complete", "observe"],
    charts: ["contact-map", "s-entropy"],
    requires: "03_resting-state",
    src: `-- 10_db-recovery.shk
-- The receiver stores NOTHING. Given partial observations, it
-- synthesises the full state by constraint propagation to a
-- self-consistent fixed point -- Kirchhoff balance, loop
-- consistency, backward trajectories -- not by forward simulation.
-- This is the empty dictionary: it recovers what it never stored.

receiver bio
floor 3.7e-4

partial := observe E as address       -- a partial omics signature

full := complete E from partial

observe full
-- the complete receiver state, recovered by constraint propagation
`,
    oracle: {
      verdict: "PASS",
      recovery: "constraint-propagation fixed point",
      stored: 0,
      converged: true,
      note: "empty-dictionary engine: synthesise, never store",
    },
  },
];

export const LESSON_BY_ID = Object.fromEntries(LESSONS.map((l) => [l.id, l]));
