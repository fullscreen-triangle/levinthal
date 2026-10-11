# Cytochrome P450 Library — Plan

Status: draft for review, 2026-10-08. Supersedes the monograph-as-papers approach.

The library has four layers, and each later layer is built from the ones before it:

1. **Papers.** They argue each claim and validate it.
2. **Records.** Every claim the papers make is emitted as triples in a DCAT-AP-PLUS extension, with one
   named graph per basis (stated, supplied, derived, refined, admitted).
3. **Queries.** HFQ plans over those records and over snapshots of external sources. Each plan returns a
   verdict with a blocker, never a bare empty answer.
4. **Model.** A domain LLM trained on the first three layers. It writes plans and explains verdicts. It
   is never a source of values (the okgg rule: the model proposes, the witness decides).

The capability map behind this plan is in memory (`reference-p450-capability-map`). It records where each
reusable module lives across levinthal, lavoisier, gospel, borgia, hegel, conspirator, ultraviolet and
buhera.

## Where it runs: Buhera OS

The OS-wide implementation already exists: `architecture/buhera/long-grass`, deployed at
`/protein-modelling`. The library does not build a new runtime; it supplies records, plans and a model
to this one.

**What Buhera already provides:**

- **A federation.** About 40 modules are dispatched through one registry (`src/lib/modules/registry.js`).
  Science modules are byte-exact vendored copies with pinned upstream commits; each has a spec in
  `buhera/specifications/specs/*.md` listing what is bound, what is not, and the known defects.
  - Checked 2026-10-08: the vendored `hfq` matches hegel `consequences/src/lib/hfq` exactly, ladder step
    included. The vendored `ladder` engine matches levinthal `enzymes/web/src/lib/engine.js` HEAD.
- **A CKG runtime** (`ckg-module.js`). Catalytic states are nodes; modules fold value-deltas onto them.
  `/protein-modelling` runs the seven-state cycle with three contributors: `cytochrome`, `sbs`
  (the CPR→heme redox circuit) and `shapeshifter` (an MS acquisition at Compound I). It reports ≥ 11
  facts across 7 vertices.
- **A P450 module** (`cytochrome-module.js`, 566 lines). Its ops are participants, electron-transfer,
  compound-i, pathway, states, spectroscopy, isoform and floor. Every number in it is **copied as a
  constant** from the monograph.
- **Natural language to DSL** (`dsl-writer-module.js`). Instructions in natural language become a
  script for a named DSL, checked by that DSL's real compiler and repaired on failure, then dispatched
  to the same executor a hand-written script reaches. `zangalewa` and `interceptor` are siblings.
- **Compute.** `lattice` runs jobs on AppHub; `spec` reads DCAT-AP or LinkML specifications.

**What the library changes in Buhera:**

1. **`cytochrome-module` reads records, not constants.** Its ops become queries over the emitted named
   graphs. A changed paper then changes the OS without anyone editing JS.
2. **Every CKG fact carries a basis.** The CKG currently mixes three kinds of value: measured
   literature values (Soret 417→392 nm, EPR g-tensor, Raman 795→758 cm⁻¹), framework-derived values
   (ΔM, ΣΔM = 4.963, the conditioned floor β), and values that look clinical but are framework-derived
   (CYP2D6 phenotype ΔM, which is not a CPIC activity score). A fact folded onto a vertex should keep
   its basis (stated, supplied, derived, refined) and its semiotic verdict, so the report can say which
   is which.
3. **A `p450` world in HFQ.** It is added upstream in hegel and re-vendored, following the OS's own
   vendoring rule (spec rule M1).
4. **`ladder` feeds `hfq`.** `specs/ladder.md` §1 already says the engine exists to make a plan's
   `ladder over … power P` a computed quantity. Phase 2.5 wires that hand-off.
5. **The domain LLM is the model behind `dsl-writer` for `dslId: "hfq"`.** The compiler check and
   repair loop are already there, so the model never needs to be trusted to produce a valid plan.
6. **`/protein-modelling` becomes the library's front page.** Its report numbers come from the records.

---

## Phase 0 — Prerequisites

Every later phase depends on these. None of them is P450-specific.

| # | Task | Why it comes first |
|---|---|---|
| 0.1 | **Stop tracking `.claude/settings.local.json`, and rotate the gateway credential.** | The repo is public and the credential is in pushed history. Nothing should be committed until this is done. |
| 0.2 | **Extract a shared core package** from `borgia/dcat-ap-plus-chromatography`: `identifiers`, `census`, `refinement`, `verdicts`, `linkage`, `comparability`, `provenance`, emission helpers, and the artefact generator calls. | P450 would be the fourth extension. Proteomics already vendors two of these modules at `98f0c47`; without a shared core, P450 would be the fifth copy. |
| 0.3 | **Reconcile the verdict algebras.** There are three: the semiotic five (symbol, collapsed, index, icon, composite) in genomics `ground.py`; the nine labels in chromatography `verdicts.py`; and the HFQ outcomes R1–R6 with blockers. | These answer different questions: the ground of a record, the status of a comparison, and the outcome of a query step. They can coexist, but one document has to state how they relate. A P450 answer will carry all three. |
| 0.4 | **Make the ladder semantics consistent.** The paper (`enzymes/docs/catalytic-ladders-federated-querying`) says `subfloor`, the code says `starved`. The paper refuses an empty ladder as `empty`, the code raises a ParseError. No test anywhere uses a ladder. | Ladders replace oscillations throughout the library, so they must be correct and tested before papers are rewritten on top of them. |
| 0.5 | **Make HFQ canonical.** Buhera already treats hegel's JS (`consequences/src/lib/hfq`) as the upstream and vendors it byte-exact. What is missing is a conformance check between that JS and the Python in `hegel/ckg/validation-federated/hfq`, which holds the validation (V1–V16). `borgia/honjo-masamune/src/lib/hfq` is a third copy: either re-vendor it from hegel or retire it. | Three copies today, and only one pair (hegel JS ↔ Buhera) is known to agree. |
| 0.6 | **Fix the levinthal-core defects Buhera's spec records** (`specs/levinthal.md`). U-lev-2: residue masses are average masses labelled monoisotopic, about 18.01 Da heavy each. U-lev-3: `levinthal-folding` is unseeded and fails on wasm32. U-lev-4: `to_sentropy` is a constant placeholder. | Peptide mapping for CYP abundance (N4, Phase 3) would inherit U-lev-2. |

Open question for 0.2: does the core package become `dcat-ap-plus-core` inside `conspirator/dcat-ap-plus`, or a new repo?

---

## Phase 1 — Papers

### 1A. Revise the existing papers (21 in `publications/`)

Four kinds of revision:

- **Ladder rewrite (L).** Replace the oscillator, phase-lock and hardware dynamics with ladders: rungs,
  composite power `1-∏(1-pᵢ)`, `subfloor`.
- **Records section (R).** Every paper gains a section "Records and queries" stating three things:
  (a) which triples the paper's claims emit, and under which basis; (b) the HFQ plans that answer its
  claims, with the expected verdicts; (c) what the schema cannot yet separate (its cells).
- **Merge (M).** Two papers that cover the same ground become one.
- **Light (·).** Only the records section is added.

| Paper | Oscillator refs | Revision |
|---|---|---|
| foundations/expression-algebra-for-biomolecules | 40 | **L + R.** The receiver R_bio stays; its dynamics become ladders. This is the heaviest rewrite and must be done first, because the others cite it. |
| catalytic-cycle/multi-hop-et-chain | 22 (12 ladder already) | **L + R.** It is already half-converted. The five instrument layers become rungs. |
| manifold/p450-address-manifold-cyp3a4-fold | 22 | **L + R** |
| equilibrium-states/cyp3a4-resting-substrate-bound | 18 | **L + R** |
| catalytic-cycle/compound-i-formation | 6 | **L + R** |
| catalytic-cycle/ch-activation-rebound | 0 | R |
| foundations/glb-structural-input | 1 | R |
| construction/membrane-cofactor-cpr | 0 | R |
| reactions/heteroatom-dealkylation, reactions/atypical-reactions-atlas | 0 | R, and these feed the metabolite rules (N4) |
| spectroscopy/spectroscopic-atlas | 0 | R, and it absorbs the minimal-probe result (N7) |
| diversity/polymorphisms-ddi-inhibitors, pharmacology/pharmacogenomics-atlas | 0 | R, and these feed N5 and N6 |
| synthesis/seven-state-closed-orbit | 0 | Recast the closed orbit as a closed ladder (N2), or merge it into N2 |
| diversity/57-human-isoforms + synthesis/57-isoform-taxonomy | 0 | **M.** One isoform paper. |
| informatics/database-recovery + informatics/empty-dictionary-p450 | 0 | **M.** One paper, which becomes the bridge to N3: "the empty dictionary is answered by plans". |
| foundations/p450-sequence-space | — | The file is empty. Either write it as the sequence and phylogeny paper for the 57 CYPs (with the CYP2D6/CYP2D7 hybrid case), or drop it. |
| introduction/monograph-introduction | 5 | Rewrite it **last**. Its current title, "Geometric Consequences of Bounded Phase Space", reflects the old framing. |

**Theory base (`cytochrome/sources/`).** These nine files are oscillator-heavy, with 1,000+ lines and up
to 105 oscillator references each. Proposal: stop citing them as the foundation. Ground the library on
`levinthal/models/sources/` instead — the contact floor, the causal propagation table and Shakespeare —
together with the ladder papers in `enzymes/docs/`. Keep the old sources as historical references only.

### 1B. New papers

| # | Paper | Built on | Validated by |
|---|---|---|---|
| N1 | **The P450 schema.** A DCAT-AP-PLUS extension: enzyme as a catalyst role with its own IRI, isoform, allele and diplotype, incubation (NADPH, microsome lot), kinetics, inhibition (reversible and time-dependent), metabolite, catalytic state, spectral observation. It includes a census and a semiotic receiver audit. | Chemistry-DCAT-AP, the shared core | census (zero lost pairs); every measured-vs-computed distinction survives emission |
| N2 | **The catalytic cycle as a ladder.** Seven states as rungs; what a missing state does to the composite. | ladder papers, hegel `exp_04_panel.py` | ladder tests; agreement with the closed-orbit results |
| N3 | **Federated querying of P450.** Source snapshots, the predicates `inhibits`, `induces`, `substrate_of` and `metabolite_of`, and many-to-many isoform maps (EC 1.14.14.1 maps to many CYPs). It must also state the library's position on substrate scope: recorded substrates are an extension, scope is an intension, and borgia `chem-dcat-ap-querying.tex:738` and refined-triples tex:1816 already argue the gap cannot be closed by a similarity axiom. | HFQ, `biocat.py` | plan verdicts; refusal tests |
| N4 | **Metabolite identification.** Biotransformation mass rules, answering only when two checks agree, corroborated by retention time. | lavoisier CASMI and ModifiedCosine; chromatography corroboration | precision against a held-out set, the same reporting style as proteomics' 79.2% |
| N5 | **Pharmacogenomic admission.** Star alleles and activity scores; the ClinVar drug-response significance; which genotype fields to admit. | genomics ground and corpora; schema admission | admission conditions A1–A6 |
| N6 | **DDI and clearance as propagation.** Inhibition and induction as ladder and cascade steps over drug-metabolism pathways. | hegel SBS (`cascade`), causal propagation table | known DDI pairs from FDA tables |
| N7 | **Spectroscopic discrimination.** The minimal set of probes that separates the seven states, plus instrument readers. | hegel `exp_04_panel.py`, ultraviolet JASCO reader | exhaustive subset search |
| N8 | **The domain model.** The model as a compiled probe. | Phase 4 | the Phase 4 evaluation |

### 1C. Order

N1 is written alongside the Phase 2 schema work, so the paper describes what exists. N2 comes after
0.4. The ladder rewrites (L) come after N2, with expression-algebra first. The records sections (R)
are added as each paper's claims become emittable. The introduction is written last. A paper is done
when its validation runs, its records emit and conform to SHACL, and its plans return the verdicts the
paper states.

---

## Phase 2 — Querying

1. **Package `dcat-ap-plus-p450`**, on the shared core, following the proteomics layout: `schema/`,
   `artefacts/`, `docs/specification/NN-*.md`, `notebooks/`, `tests/` with synthetic fixtures, and a
   uv-based CLI. Its location is an open question (see below).
2. **Snapshot sources.** HFQ adapters are forbidden from network I/O by design (`adapters.py:15`), so
   each source has a `fetch` step that writes a dated, content-addressed snapshot (`identifiers.Stage`),
   and an adapter that reads only the snapshot. Starting sources: UniProt (the 57 human CYPs), PDB
   (structures), Rhea and KEGG (reactions; hsa00982, 00983, 00140, 00590), ChEMBL (activities: Ki, IC50,
   time-dependent inhibition), PharmVar and CPIC (alleles, activity scores), ClinVar (already parsed by
   genomics), and FDA DDI tables. The live clients in `hegel/backend/api/services/data_sources.py` can
   seed the fetchers.
3. **Predicates and features.** Add `inhibits`, `induces`, `substrate_of`, `metabolite_of`,
   `allele_of` and `activity_score` to `PREDICATE_FEATURES`. Add numeric range filtering with units —
   HFQ `filter` currently allows one comparison and no units.
4. **Translation maps.** UniProt ↔ EC ↔ KEGG ↔ PDB ↔ ChEMBL target, measured with HFQ's existing
   retention and amplification.
5. **Data-derived ladders.** Today rung powers are literals in the plan. Allow a rung to take its power
   from a retrieved value, such as a kinetic fraction or a map's retention, or from the `ladder`
   engine's computed intensive power (`specs/ladder.md` §1 describes this hand-off). This is the one
   real extension to the HFQ language, so it lands upstream in hegel first and is then re-vendored.
6. **Plan library.** One `.hfq` plan per claim in the papers, kept in `plans/`, with expected verdicts
   as tests.
7. **Into Buhera.** Add a `p450` world to HFQ's `WORLDS` (hegel upstream, then re-vendored), so plans
   run in the OS terminal and in `/protein-modelling`. Rewrite `cytochrome-module.js` so its ops query
   the records instead of holding constants.

## Phase 3 — Downstream processes

| Process | Reuses | New work |
|---|---|---|
| Emission of named graphs, conformance to SHACL, round-trip | proteomics `emission.py`, `output.py` | P450 classes |
| Refined triples and the gate (`+`, `-`, `a`, `open`) | conspirator `store.py`, `gate.py` | — |
| Schema admission | proteomics `admission.py`, `goals.py` | P450 goals |
| Semiotic grounding profile beside each answer | genomics `ground.py` | P450 witness-class declaration |
| Ontology over the papers and records | okgg (sections and records) | — |
| Metabolite ID pipeline | lavoisier CASMI, ModifiedCosine | biotransformation rules; adding F, Br, I and Fe; isotope model |
| Pharmacogenomics | genomics ClinVar corpus | fix `clinvar.py:121` (drug response collapsed to "other"); star-allele caller (gospel's is broken at `analysis.py:161`) |
| Spectroscopy | ultraviolet JASCO | EPR, Raman and Mössbauer readers (none exist) |
| SPARQL endpoint | long-grass `transaminase_kg/api.py` | swap the enzyme |
| CKG runtime | Buhera `ckg-module.js`, `/protein-modelling` | each folded fact carries its basis and semiotic verdict; the report projects them |
| Consolidation | — | P450 content in purpose (knowledge pack), long-grass (`cytochrome-module.js`), bloodhound (`cytochrome.json`) and kwisatz-haderach (papers) reads from the library instead of holding its own copy |

## Phase 4 — Domain LLM

**What it is for.** The model is a compiled probe, consistent with `models/publications/purpose-based-protein-model`.
It does four things:

1. translate a question into an HFQ plan;
2. read the plan's verdicts and blame chain, and explain them, including why an empty answer is
   `starved`, `refused` or `open`;
3. propose distinctions for okgg and admission;
4. answer from the papers with citations to records.

**It never states a value that a record does not hold.**

**Training data,** all generated from Phases 1–3:

- question → plan pairs, from the plan library;
- plan → verdict → explanation, including deliberate refusals;
- paper sections → Q&A, using the corpus builder in `greifswald/nfdi4cat-model/build_corpus.py`;
- record graphs → descriptions.

**Deployment.** The model serves Buhera's `dsl-writer` for `dslId: "hfq"`, and later for `ladder` and
`shapeshifter`. That module already validates every generated script with the real compiler and
repairs it on failure, so a plan the model gets wrong is caught before it runs. The model never needs
to be trusted to produce a valid plan; it needs to be good enough that repairs are rare. The rate of
repairs is a deployment metric.

**Recipe.** Reuse doerr-lab-model: a Qwen2.5 base, LoRA rank 16, and TIES merging (`ties_merge.py`,
`eval.py`). Train on AppHub via `nfdi4cat-model/bundle/` or pylon `lattice/`. Choose the size after a
0.5B baseline.

**Evaluation.** Not text similarity. Measure:

- whether the plans it writes parse and pass the check;
- whether running its plans gives the same verdicts as reference plans;
- whether it refuses when it should;
- zero values not backed by a record.

**Licences constrain the training data.** ChEMBL is CC BY-SA, which is fine. PharmGKB and DrugBank have
licences restricting redistribution and commercial use, so check them before any of their content
enters training data or a released model.

---

## Decisions needed

1. Where `dcat-ap-plus-p450` lives: levinthal, a new repo next to the other extensions, or the
   university GitLab.
2. Where the shared core lives (0.2).
3. Whether the old `cytochrome/sources/` stop being cited as the foundation (1A).
4. The library's position on substrate scope (N3).
5. Which sources are in scope for the first release, given the licences.
6. The target size of the model and the compute budget on AppHub.
7. Whether `cytochrome-module.js` is rewritten in place to read the records, or replaced by a new
   `p450` module, keeping the current one as the transaminase-comparison demo that
   `/protein-modelling` describes.
