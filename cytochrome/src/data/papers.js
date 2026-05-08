// Single source of truth for the cytochrome P450 monograph papers.
// Used by index.js (paper cards), per-paper pages, and gallery.js.

export const PAPERS = [
  {
    id: "1",
    slug: "foundations",
    href: "/foundations",
    title: "An S-Expression Algebra for Biomolecules",
    short: "Foundations",
    part: "Part I — Foundations",
    role: "Receiver and instrument bindings",
    abstract: `Establishes the receiver R_bio, the leaf algebra,
      the closed-form conversion functors F_OC, F_CB, F_BO, the τ-assignment
      protocol, and the spin-crossover resolution via two-tier chirality.
      Foundation for every subsequent paper.`,
    headline: [
      { label: "Floor S(R_bio)", computed: "3.43 × 10⁻⁴", target: "≈ 3.7 × 10⁻⁴" },
      { label: "Capacity C(n) = 2n²", computed: "exact", target: "electron shells" },
      { label: "Amino-acid uniqueness", computed: "20 / 20", target: "depth 9" },
    ],
    panelDir: "/panels/paper-1",
    panels: [
      ["panel_01_floor_theorem.png",       "Floor theorem: 3.43 × 10⁻⁴ ≈ 3.7 × 10⁻⁴"],
      ["panel_02_capacity_selection.png",  "C(n) = 2n² capacity"],
      ["panel_03_cycle_closure.png",       "Cycle closure under S₃"],
      ["panel_04_amino_acid_coords.png",   "20 amino-acid coordinates, depth 9"],
      ["panel_05_tau_assignment.png",      "τ-assignment rule"],
      ["panel_06_spin_crossover.png",      "Spin-crossover via two-tier chirality"],
      ["panel_07_kuramoto_sync.png",       "Kuramoto synchronisation under R_bio"],
      ["panel_08_morphism_chain.png",      "Morphism chain access ∘ fuse ∘ catalyze ∘ observe"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "2",
    slug: "manifold",
    href: "/manifold",
    title: "The P450 Address Manifold + CYP3A4 Fold",
    short: "Manifold",
    part: "Part II — The P450 Manifold",
    role: "Sequence taxonomy + CYP3A4 fold from sequence",
    abstract: `Apply the empty-dictionary engine and ternary encoding to
      ~400 000 P450 sequences. The 18-family taxonomic structure
      corresponds to clustering at trit-depth k=3; the 57 human isoforms
      separate at k=6; allelic variants separate at k=9. CYP3A4 folds
      against PDB 1TQN in O(log₃ N) ≈ 6 categorical steps.`,
    headline: [
      { label: "Family clustering", computed: "recall 0.94", target: "Nelson nomenclature" },
      { label: "Isoform separation k=6", computed: "0.97 distinctness", target: "57 human CYPs" },
      { label: "Fold log₃ N",            computed: "5.69",            target: "≈ 6" },
    ],
    panelDir: "/panels/paper-2",
    panels: [
      ["panel_01_address_encoding.png",    "Ternary address encoding"],
      ["panel_02_manifold_density.png",    "Manifold density in P450 subregion"],
      ["panel_03_family_clustering.png",   "18 family clusters at k=3"],
      ["panel_04_isoform_separation.png",  "57 human isoforms at k=6"],
      ["panel_05_allele_resolution.png",   "CYP2D6 alleles at k=9"],
      ["panel_06_cyp3a4_address.png",      "CYP3A4 address assembly"],
      ["panel_07_kuramoto_folding.png",    "Kuramoto folding trajectory"],
      ["panel_08_contact_map.png",         "Predicted vs PDB 1TQN contact map"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "2.5",
    slug: "glb-input",
    href: "/glb-input",
    title: "GLB-Based Structural Input to the Receiver",
    short: "GLB",
    part: "Part I (methods) — GLB Bridge",
    role: "Methods companion: real PDB geometry into R_bio",
    abstract: `The package levinthal_glb bridges binary-glTF (GLB) 3D
      structural files to R_bio. Atomistic GLBs yield per-atom positions
      and elements via scene-graph traversal and CPK colour decoding.
      The productive cytochrome P450 GLB recovers Fe coordination at
      canonical CYP450 distances; the 1.814 Å Fe–O specifically identifies
      the GLB as state 4 (oxy-complex) of the catalytic cycle.`,
    headline: [
      { label: "Fe–N (porphyrin)",     computed: "2.01 – 2.04 Å", target: "≈ 2.0 Å" },
      { label: "Fe–S (Cys thiolate)",  computed: "2.228 Å",        target: "≈ 2.2 Å" },
      { label: "Fe–O (axial, state 4)", computed: "1.814 Å",       target: "1.80 – 1.85 Å" },
    ],
    panelDir: "/panels/paper-2.5",
    panels: [
      ["panel_01_glb_parser_smoke.png",     "GLB parser smoke test, 3 GLBs"],
      ["panel_02_cpk_color_decoder.png",    "CPK colour decoder, ±30 RGB tolerance"],
      ["panel_03_artifact_filtering.png",   "Artifact filtering: 171 → 146 atoms"],
      ["panel_04_iron_coordination.png",    "Iron first-shell coordination"],
      ["panel_05_state4_oxy_complex.png",   "1.814 Å Fe–O = state 4 oxy-complex"],
      ["panel_06_morphism_chain.png",       "Morphism chain on real GLB"],
      ["panel_07_s_entropy_address.png",    "Per-element S-entropy + trit address"],
      ["panel_08_five_roles.png",           "Five GLB roles taxonomy"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "3",
    slug: "equilibrium",
    href: "/equilibrium",
    title: "The Resting and Substrate-Bound States of CYP3A4",
    short: "Equilibrium",
    part: "Part II — Equilibrium States",
    role: "States 1 and 2 of the catalytic cycle",
    abstract: `Characterise resting Fe³⁺-H₂O low-spin and substrate-bound
      Fe³⁺ high-spin as receiver evaluations. The 1 → 2 transition is one
      categorical aperture (d_C = 1) with ΔM = 0.92. The +120 mV redox
      shift gating CPR-mediated electron acceptance emerges directly from
      ΔM via ΔE_{1/2} = (k_BT/e) · n_eff · ΔM · ln b. Heme-pocket
      capacitor model: C ≈ 5.7 × 10⁻²⁰ F, U ≈ 1.4 eV, τ_RC ≈ 60 ps.`,
    headline: [
      { label: "ΔM (Fe LS → HS)",      computed: "0.918",      target: "0.92" },
      { label: "Heme capacitance",      computed: "56.7 aF",    target: "≈ 57 aF" },
      { label: "Redox shift +120 mV",   computed: "122 mV",     target: "120 mV (Daff 1997)" },
    ],
    panelDir: "/panels/paper-3",
    panels: [
      ["panel_01_closed_form_functors.png",  "Closed-form F_OC / F_CB / F_BO"],
      ["panel_02_resting_state.png",          "Resting-state coherent regime"],
      ["panel_03_heme_capacitor.png",         "Heme-pocket capacitor model"],
      ["panel_04_variance_free_energy.png",   "F = k_BT · σ²(φ)"],
      ["panel_05_spin_crossover.png",         "Spin-crossover ΔM = 0.92"],
      ["panel_06_substrate_bound.png",        "Substrate-bound locked regime"],
      ["panel_07_chamber.png",                "Electrostatic chamber confinement"],
      ["panel_08_redox_shift.png",            "ΔE_{1/2} = 122 mV vs 120 mV measured"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "4",
    slug: "transfer",
    href: "/transfer",
    title: "Observing Electron Transfer through Cytochrome P450 Reductase",
    short: "Transfer",
    part: "Part III — The Headline Observation",
    role: "HEADLINE: NADPH → FAD → FMN → heme observed via 5-layer apparatus",
    headline_paper: true,
    abstract: `The headline paper of the monograph. With the protein
      constructed by Papers 1–3 and grounded in real PDB coordinates by
      Paper 2.5, the four-cofactor electron transfer chain is observed —
      not simulated — by a five-layer instrument stack already specified
      in the source papers. Hardware oscillators at the bottom (CPU/bus/
      LED/refresh) resolving (n, ℓ, m, s); triple-equivalence theorem as
      calibration certificate; ensemble strobes at fs / ns / μs+;
      harmonic molecular resonator giving cycle-rank cross-validation
      loops; a five-pass GPU hologram pipeline at the top, whose six
      observables include Marcus reorganisation energy λ. The headline
      deliverable is per-frame |ψ(r,t)|² snapshots of the electron
      moving through the real GLB-anchored CPR–CYP3A4 geometry.`,
    headline: [
      { label: "k_cat/K_M (d_C = 4)",       computed: "≈ 10⁶ M⁻¹s⁻¹", target: "measured CPR–P450" },
      { label: "Marcus λ (Layer 5)",         computed: "0.85 eV",       target: "0.7 – 1.0 eV (lit.)" },
      { label: "Cofactor recall (χ²)",       computed: "100 %",         target: "azurin precedent" },
    ],
    panelDir: "/panels/paper-4",
    panels: [
      ["panel_01_chain_topology.png",          "Four-cofactor chain topology"],
      ["panel_02_dc4_efficiency.png",          "d_C = 4 efficiency relation"],
      ["panel_03_marcus_distance.png",         "Marcus distance scaling (β)"],
      ["panel_04_selection_rules.png",         "Selection rules at each hop"],
      ["panel_05_semiquinone_ladder.png",      "Flavin three-state semiquinone ladder"],
      ["panel_06_chain_kinetics.png",          "Chain rate composition"],
      ["panel_07_newton_cradle.png",           "Newton's-cradle non-identity"],
      ["panel_08_falsifiable_predictions.png", "Falsifiable predictions"],
      ["panel_09_apparatus_stack.png",         "Five-layer instrument stack"],
      ["panel_10_cofactor_self_selection.png", "Cofactor self-selection by χ²"],
      ["panel_11_electron_visualisations.png", "HEADLINE: |ψ(r,t)|² across the chain"],
      ["panel_12_glb_shader_integration.png",  "GLB + shader pipeline integration"],
    ],
    status: "Validation 12/12 PASS",
  },
  {
    id: "5",
    slug: "compound-i",
    href: "/compound-i",
    title: "Compound I Formation via O–O Heterolysis",
    short: "Compound I",
    part: "Part III — Downstream Chemistry",
    role: "What happens after the electrons arrive at heme",
    abstract: `Compound I (Fe⁴⁺=O porphyrin•⁺) as a single d_C=1 aperture
      with ΔM = ln 2 driven by an O–O bond-order trajectory from 1
      (peroxo) to 0 (cleaved). Anharmonic Poincaré non-recurrence
      replaces the energy-barrier picture; bond-breaking is structurally
      guaranteed, not a rare event. PCET concerted (d_C=1) vs sequential
      (d_C=2) discrimination predicts 10× rate ratio. 11 spectroscopic
      observables match Rittle–Green within 20%.`,
    headline: [
      { label: "ΔM bond-order cleave",   computed: "ln 2 ≈ 0.693",    target: "binary bond coord" },
      { label: "PCET concerted/seq.",    computed: "10×",              target: "Marcus ≠ 1" },
      { label: "Spectroscopic match",    computed: "8/8 within 20 %",  target: "Rittle–Green" },
    ],
    panelDir: "/panels/paper-5",
    panels: [
      ["panel_01_peroxo_state.png",     "Cpd 0 peroxo state"],
      ["panel_02_bond_order.png",       "Binary bond-order coordinate"],
      ["panel_03_aperture.png",         "d_C = 1 single aperture"],
      ["panel_04_pcet.png",             "PCET concerted vs sequential"],
      ["panel_05_anharmonic.png",       "Anharmonic non-recurrence"],
      ["panel_06_lifetime.png",         "Cpd I lifetime"],
      ["panel_07_potential.png",        "Oxidation potential"],
      ["panel_08_spectroscopy.png",     "Eight spectroscopic observables"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "6",
    slug: "ch-activation",
    href: "/ch-activation",
    title: "C–H Activation by Compound I: H-Atom Transfer and Oxygen Rebound",
    short: "C–H Activation",
    part: "Part III — Catalytic Chemistry",
    role: "Three-body trajectory: substrate C, H atom, and Fe=O",
    abstract: `C–H bond activation by Compound I treated as a three-body
      categorical trajectory: substrate C, abstracted H, and ferryl O.
      Three new pieces: the C–H bond-order partition coordinate β_{CH} ∈ {0,1},
      the three-body aperture selection rule (d_C=1 requires simultaneous
      ΔβCH=1, ΔβOH=−1, Δs_orbital=0), and the oxygen-rebound aperture with
      ΔM_rebound < ΔM_HAT ensuring rebound is intrinsically faster than
      abstraction. KIE ≈ 7.2 predicted (range 4–11). Stereoretention
      40–90 % from rebound/escape competition. Testosterone 6β ≈ 49 %
      (literature 50–70 %). Five reaction types unified under one framework.`,
    headline: [
      { label: "KIE (aliphatic C–H)",   computed: "7.2",             target: "4 – 11 (literature)" },
      { label: "k_rebound / k_HAT",     computed: "1.4 ×",           target: "> 1 (rebound faster)" },
      { label: "6β regioselectivity",   computed: "49 %",            target: "50 – 70 % (CYP3A4)" },
    ],
    panelDir: "/panels/paper-6",
    panels: [
      ["panel_01_substrate_positioning.png",  "Substrate class activation depths"],
      ["panel_02_hat_coordinate.png",         "C–H bond-order coordinate + TS geometry"],
      ["panel_03_kie.png",                    "KIE: ZPE + tunneling → 7.2"],
      ["panel_04_radical_intermediate.png",   "Radical intermediate partition cell"],
      ["panel_05_rebound.png",                "Oxygen-rebound aperture"],
      ["panel_06_regioselectivity.png",       "Testosterone 6β selectivity 49 %"],
      ["panel_07_reaction_types.png",         "Five reaction types unified"],
      ["panel_08_validation.png",             "Full validation: 8/8 PASS"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "7",
    slug: "heteroatom",
    href: "/heteroatom",
    title: "Heteroatom Oxidation and Dealkylation by Cytochrome P450",
    short: "Heteroatom",
    part: "Part III — Catalytic Chemistry",
    role: "N-, O-, S-oxidation under a unified activation-depth hierarchy",
    abstract: `N-dealkylation, O-dealkylation, S-oxidation, and N-oxide
      formation unified under one activation partition-depth hierarchy.
      Direct O-atom transfers (S-ox ΔM=0.28, N-ox ΔM=0.32) are intrinsically
      faster than HAT-based dealkylations (N-dealk ΔM=0.50, O-dealk ΔM=0.58).
      Alpha-C–H softer frequency (2800 cm⁻¹) reduces KIE_N-dealk to ≈6.7
      vs 7.7 aliphatic. Carbinolamine intermediates (ΔM<0.15) are
      kinetically labile and non-accumulating.`,
    headline: [
      { label: "ΔM (N-dealkylation α-C)", computed: "0.50",     target: "BDE-derived, α-C weaker" },
      { label: "KIE (N-dealkylation)",    computed: "6.7",      target: "< 7.2 aliphatic" },
      { label: "k_S-ox / k_N-dealk",     computed: "1.25 ×",   target: "direct transfer faster" },
    ],
    panelDir: "/panels/paper-7",
    panels: [
      ["panel_01_alpha_carbon_bde.png",          "Alpha-C BDE ordering and ΔM mapping"],
      ["panel_02_n_dealkylation_mechanism.png",  "N-dealkylation pathway and KIE"],
      ["panel_03_kie_comparison.png",            "KIE comparison across heteroatom types"],
      ["panel_04_s_oxidation.png",               "S-oxidation: direct O-atom transfer, KIE=1"],
      ["panel_05_n_oxide.png",                   "N-oxide formation via lone-pair O-transfer"],
      ["panel_06_rate_ordering.png",             "Complete rate hierarchy (5 reactions)"],
      ["panel_07_carbinolamine.png",             "Carbinolamine/hemiacetal cascade"],
      ["panel_08_validation.png",                "Validation summary: 8/8 PASS"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "11",
    slug: "membrane",
    href: "/membrane",
    title: "Membrane Anchoring and Partner Coupling in Cytochrome P450",
    short: "Membrane",
    part: "Part IV — Structural Biology",
    role: "ER membrane insertion, CPR interface, electron delivery to heme",
    abstract: `CYP3A4 anchors to the ER membrane via a 20-residue N-terminal
      transmembrane helix (ΔG_insert ≈ −10 kcal/mol, ΔM_TM = 0.42).
      The FMN-binding domain of CPR contacts the proximal face of P450;
      K_d ≈ 0.1 μM. FMN→heme electron tunnelling (r = 14 Å, β = 1.4 Å⁻¹)
      gives k_ET ≈ 5×10⁶ s⁻¹ (ΔM_ET = 7.60). Cytochrome b5 binds tighter
      (K_d ≈ 0.05 μM) and transfers faster (k_b5 ≈ 3×10⁷ s⁻¹). Membrane
      enrichment enhances lipophilic substrate effective concentration by
      ~10× for logP = 3 substrates.`,
    headline: [
      { label: "K_d (CPR–P450)",          computed: "0.1 μM",          target: "0.05 – 0.5 μM" },
      { label: "k_FMN→heme (ET rate)",    computed: "5 × 10⁶ s⁻¹",    target: "10⁶ – 10⁷ s⁻¹" },
      { label: "Enrichment (logP=3)",     computed: "10 ×",            target: "> 5 × lipophilic" },
    ],
    panelDir: "/panels/paper-11",
    panels: [
      ["panel_01_tm_helix.png",               "TM helix hydrophobicity + membrane insertion"],
      ["panel_02_cpr_interface.png",          "CPR–P450 interface electrostatics"],
      ["panel_03_et_pathway.png",             "FMN→heme ET pathway, distance vs rate"],
      ["panel_04_cytb5_competition.png",      "CPR vs Cyt b5 kinetics comparison"],
      ["panel_05_membrane_partitioning.png",  "Substrate enrichment near ER vs logP"],
      ["panel_06_cpr_p450_stoichiometry.png", "CPR:P450 ratio effect on turnover"],
      ["panel_07_proximal_face.png",          "3D proximal face charge distribution"],
      ["panel_08_validation.png",             "Validation summary: 8/8 PASS"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "8",
    slug: "atypical",
    href: "/atypical",
    title: "Atypical Reactions of Cytochrome P450: Desaturation, Epoxidation, NIH Shift, Nucleophilic O-Atom Transfer, and Carbene Insertion",
    short: "Atypical Reactions",
    part: "Part III — Reaction Mechanisms",
    role: "Five non-HAT reaction modes; rate hierarchy; KIE predictions",
    abstract: `Beyond canonical hydroxylation, P450 Compound I (and related
      iron-oxo species) catalyzes five mechanistically distinct atypical
      reactions. Within the categorical mechanics framework each is assigned
      a single activation partition depth ΔM: NIH shift (0.18), carbene
      insertion (0.20), epoxidation (0.35), nucleophilic O-atom transfer (0.42),
      and desaturation (two-step HAT; k_eff ≈ 1.3×10⁸ s⁻¹). The predicted
      rate hierarchy NIH > carbene > epoxidation > nucleophilic > desaturation
      matches experimental literature for all five classes. Only desaturation
      carries a primary KIE (≈ 4–6); the others have KIE ≈ 1. Eight validation
      scripts confirm all 40 quantitative checks (100% PASS).`,
    headline: [
      { label: "NIH shift rate",         computed: "8.4 × 10⁹ s⁻¹",  target: "5×10⁹ – 10¹⁰ s⁻¹" },
      { label: "Desaturation k_eff",     computed: "1.3 × 10⁸ s⁻¹",  target: "10⁸ – 5×10⁹ s⁻¹" },
      { label: "KIE desaturation",       computed: "4 – 6",           target: "> 3 (primary KIE)" },
    ],
    panelDir: "/panels/paper-8",
    panels: [
      ["panel_01_desaturation_two_step.png", "Desaturation: two-step HAT vs. rebound competition"],
      ["panel_02_desaturation_kie.png",      "KIE analysis: desaturation vs. single HAT"],
      ["panel_03_arene_oxide.png",           "Arene epoxidation: rate and KIE"],
      ["panel_04_nih_shift.png",             "NIH shift: fastest atypical mechanism"],
      ["panel_05_nucleophilic_aldehyde.png", "Nucleophilic O-atom transfer to aldehyde C=O"],
      ["panel_06_rate_ordering.png",         "Rate ordering of all five atypical mechanisms"],
      ["panel_07_product_partitioning.png",  "Product partitioning: phenol vs. dihydrodiol"],
      ["panel_08_validation.png",            "Validation summary: 8/8 PASS"],
    ],
    status: "Validation 8/8 PASS",
  },
  {
    id: "12",
    slug: "closed-orbit",
    href: "/closed-orbit",
    title: "The Seven-State Catalytic Cycle as Closed Categorical Orbit",
    short: "Closed Orbit",
    part: "Part IV — Synthesis",
    role: "Seven states as non-degenerate orbit; Poincaré return; rate hierarchy",
    abstract: `The P450 catalytic cycle — resting → substrate-bound → reduced →
      oxy-complex → peroxo → Cpd0 → Cpd I → product release — is formulated
      as a closed categorical orbit of eight d_C=1 transitions with ΣΔM ≈ 4.96.
      The Newton's-cradle non-identity theorem guarantees seven structurally
      distinct states. Chemical steps (ΔM ≤ 0.92) are 1000× faster than the
      rate-limiting FMN→heme electron tunnelling (ΔM = 7.60), confirming that
      intrinsic chemistry is not the bottleneck in vivo. Poincaré return time
      ≈ 400 ps (ET-limited).`,
    headline: [
      { label: "States in closed orbit",   computed: "7",           target: "structurally distinct" },
      { label: "Poincaré return time",      computed: "≈ 400 ps",   target: "ET-limited > 100 ps" },
      { label: "k_chem / k_ET ratio",       computed: "1000 ×",     target: "chemistry not limiting" },
    ],
    panelDir: "/panels/paper-12",
    panels: [
      ["panel_01_seven_states.png",        "Seven-state cycle diagram + energy profile"],
      ["panel_02_state_properties.png",    "ΔM, k, τ for each transition"],
      ["panel_03_closed_orbit.png",        "Poincaré section showing orbit closure"],
      ["panel_04_rate_limiting.png",       "Rate hierarchy: chemistry vs ET vs binding"],
      ["panel_05_anharmonic_closure.png",  "ΔM trajectory showing no sink points"],
      ["panel_06_non_identity.png",        "3D address space: 7 distinct state points"],
      ["panel_07_poincare_return.png",     "Return time distribution + cycle period"],
      ["panel_08_validation.png",          "Validation summary: 8/8 PASS"],
    ],
    status: "Validation 8/8 PASS",
  },
];

export const PAPER_BY_SLUG = Object.fromEntries(PAPERS.map(p => [p.slug, p]));

// All panels flattened, for the gallery
export const ALL_PANELS = PAPERS.flatMap(p =>
  p.panels.map(([file, caption], idx) => ({
    paper: p.short,
    paperId: p.id,
    paperHref: p.href,
    src: `${p.panelDir}/${file}`,
    caption,
    panelIndex: idx + 1,
  }))
);
