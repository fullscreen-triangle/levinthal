/**
 * The twenty-eight questions, resolved live in the browser.
 *
 * Q1-Q12 were supplied by practitioners without reference to this framework.
 * Q4' and Q13-Q28 were added to reach outcomes the twelve do not.
 *
 * Each question carries an executable `run` returning a verdict, the plan
 * steps it took, and (for refusals) a blocker and an unblock path.
 */

import {
  BETA, CHEBI, DATASETS, EXPERIMENTS, MEDIA, PATHWAYS, PROTEINS, REACTIONS,
  SOURCES, TAU,
} from './corpus.js'
import {
  compose, directionVerdict, makeResult, mediumBias, mediumWeight, solventRole,
} from './kernel.js'

const name = (c) => CHEBI[c] || c

// -- helpers over the fixture corpus ---------------------------------------
const datasetsWithCompound = (c) =>
  Object.keys(DATASETS).filter((d) => DATASETS[d].compounds.includes(c))
const expOf = (d) => EXPERIMENTS[DATASETS[d].experiment]

// Both constructors go through makeResult, so non-degeneracy is enforced by
// the type rather than by reporting discipline: building a refusal that
// carries a payload throws, here and in the reference kernel alike.
const ANSWER = (payload, steps) => ({
  ...makeResult({ verdict: 'answer', payload }),
  steps,
})
const REFUSE = (verdict, blocker, unblock, steps, blame = null) => ({
  ...makeResult({ verdict, blame, reason: blocker, unblock }),
  blocker,
  unblock,
  steps,
})

export const QUESTIONS = [
  // ===================== GROUP 1: biocatalysis ============================
  {
    id: 'Q1',
    group: 'Practitioner: biocatalysis',
    origin: 'practitioner',
    text: 'Which biocatalyst, from a bacterium (not a eukaryote), catalyses the transamination of benzylethylamine and has no cysteine in its sequence?',
    why: 'Three constraints across two sources, and the last one is not a triple in any public store: “contains no C” is a computation over a retrieved sequence.',
    sources: ['RXN', 'PROT'],
    run() {
      const rxns = Object.keys(REACTIONS).filter((r) =>
        [...REACTIONS[r].substrates, ...REACTIONS[r].products].some(
          (c) => name(c) === 'benzylethylamine'
        )
      )
      const enzymes = Object.keys(PROTEINS).filter((p) =>
        PROTEINS[p].catalyses.some((r) => rxns.includes(r))
      )
      const bacterial = enzymes.filter((p) => PROTEINS[p].domain === 'Bacteria')
      const final = bacterial.filter((p) => !PROTEINS[p].sequence.includes('C'))
      return ANSWER(
        final.map((p) => `${p} — ${PROTEINS[p].name}, ${PROTEINS[p].organism}`),
        [
          { src: 'RXN', op: 'participant contains "benzylethylamine"', n: rxns.length },
          { src: 'PROT', op: 'catalyses ∈ previous result', n: enzymes.length },
          { src: '—', op: 'keep domain = Bacteria', n: bacterial.length },
          { src: '—', op: 'keep ¬contains(sequence, "C")  ← computed', n: final.length },
        ]
      )
    },
  },
  {
    id: 'Q2',
    group: 'Practitioner: biocatalysis',
    origin: 'practitioner',
    text: 'Which buffer composition and pH were used in the biocatalytic methyl transfer with methyltransferase mt-X?',
    why: 'No public reaction, protein or pathway source declares “buffer” or “pH”. The datum exists only in the laboratory record — which is a capability question, not a curation failure.',
    sources: ['ELN'],
    run() {
      const e = EXPERIMENTS.MT7
      return ANSWER(
        [`${e.buffer.name} ${e.buffer.mM} mM, pH ${e.buffer.pH}`, `${e.temperature} °C`],
        [
          { src: 'RXN/PROT/PATH', op: 'declare "buffer"?', n: 0 },
          { src: 'ELN', op: 'experiment = mt-X methyl transfer', n: 1 },
        ]
      )
    },
  },
  {
    id: 'Q3',
    group: 'Practitioner: biocatalysis',
    origin: 'practitioner',
    text: 'What is the substrate scope and product range of the Baeyer–Villiger monooxygenase BVMO-Y?',
    why: 'A lab-local identifier resolved through the ELN, then aggregated over the reaction it screens.',
    sources: ['ELN', 'RXN'],
    run() {
      const e = EXPERIMENTS.BV2
      const r = REACTIONS[e.reaction]
      return ANSWER(
        [
          `substrates: ${r.substrates.map(name).join(', ')}`,
          `products: ${r.products.map(name).join(', ')}`,
        ],
        [
          { src: 'ELN', op: 'biocatalyst = BVMO-Y → reaction', n: 1 },
          { src: 'RXN', op: 'participants of the reaction', n: r.substrates.length + r.products.length },
        ]
      )
    },
  },
  {
    id: 'Q4',
    group: 'Practitioner: biocatalysis',
    origin: 'practitioner',
    text: 'What are the expected products of a biocatalytic kinetic resolution with PFE at pH 9 in HEPES buffer?',
    why: 'This is not a retrieval question at all. No record asserts the products of an experiment nobody has run. Retrieval returns a substructure of what was stored; this asks what could happen.',
    sources: ['ELN'],
    run() {
      return REFUSE(
        'unsupported',
        'not a retrieval question: no record asserts the products of an experiment that has not been run',
        'ask it as an admissibility query over the contact graph instead — see Q4′, which computes whether a propagation to each candidate product is accountable in the declared medium',
        [{ src: 'ELN', op: 'search for asserted products', n: 0 }]
      )
    },
  },
  {
    id: 'Q5',
    group: 'Practitioner: biocatalysis',
    origin: 'practitioner',
    text: 'Which device was used during biocatalytic transformation BT3 to monitor the UV spectrum on 23 March by Y. Dikova, and which wavelength was monitored?',
    why: 'Instrument provenance — operator, date, device and its settings — is a first-class record here, not an annotation appended to a result.',
    sources: ['ELN'],
    run() {
      const e = EXPERIMENTS.BT3
      return ANSWER(
        [
          `${e.device.vendor} ${e.device.id} (${e.device.kind})`,
          `wavelength ${e.device.settings.wavelength_nm} nm, bandwidth ${e.device.settings.bandwidth_nm} nm`,
        ],
        [{ src: 'ELN', op: 'operator ∧ date ∧ experiment = BT3', n: 1 }]
      )
    },
  },

  // ===================== GROUP 2: dataset queries ==========================
  {
    id: 'Q6',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets about a substance with compound C (cyclohexanone).',
    why: 'The first of a strictly narrowing chain. If Q6→Q7→Q8 did not shrink, the filters would not be shown to discriminate.',
    sources: ['ELN'],
    run() {
      const r = datasetsWithCompound('CHEBI:17854')
      return ANSWER(r, [{ src: 'ELN', op: 'compound = cyclohexanone', n: r.length }])
    },
  },
  {
    id: 'Q7',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets generated by an activity of type T (NMR) that evaluated compound C.',
    why: 'One constraint added; one dataset must drop away.',
    sources: ['ELN'],
    run() {
      const base = datasetsWithCompound('CHEBI:17854')
      const r = base.filter((d) => DATASETS[d].type === 'NMR')
      return ANSWER(r, [
        { src: 'ELN', op: 'compound = cyclohexanone', n: base.length },
        { src: 'ELN', op: '∧ activity type = NMR', n: r.length },
      ])
    },
  },
  {
    id: 'Q8',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets … measured with a Bruker spectrometer set to X (nucleus ¹³C).',
    why: 'A device SETTING, not just a device identity. Restricting from “a Bruker” to “a Bruker at ¹³C” must remove a dataset, otherwise the setting is decorative.',
    sources: ['ELN'],
    run() {
      const base = datasetsWithCompound('CHEBI:17854')
      const nmr = base.filter((d) => DATASETS[d].type === 'NMR')
      const bruker = nmr.filter((d) => expOf(d).device.vendor === 'Bruker')
      const r = bruker.filter((d) => expOf(d).device.settings.nucleus === '13C')
      return ANSWER(r, [
        { src: 'ELN', op: 'compound = cyclohexanone', n: base.length },
        { src: 'ELN', op: '∧ type = NMR', n: nmr.length },
        { src: 'ELN', op: '∧ vendor = Bruker', n: bruker.length },
        { src: 'ELN', op: '∧ setting nucleus = ¹³C', n: r.length },
      ])
    },
  },
  {
    id: 'Q9',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets about a chemical reaction that had product P (ε-caprolactone).',
    sources: ['ELN', 'RXN'],
    run() {
      const r = Object.keys(DATASETS).filter((d) => {
        const e = expOf(d)
        return e.reaction && REACTIONS[e.reaction].products.includes('CHEBI:35604')
      })
      return ANSWER(r, [
        { src: 'ELN', op: 'dataset → experiment → reaction', n: Object.keys(DATASETS).length },
        { src: 'RXN', op: 'products include ε-caprolactone', n: r.length },
      ])
    },
  },
  {
    id: 'Q10',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets about a reaction whose starting material has compound C.',
    sources: ['ELN', 'RXN'],
    run() {
      const r = Object.keys(DATASETS).filter((d) => {
        const e = expOf(d)
        return e.reaction && REACTIONS[e.reaction].substrates.includes('CHEBI:17854')
      })
      return ANSWER(r, [{ src: 'ELN+RXN', op: 'substrates include cyclohexanone', n: r.length }])
    },
  },
  {
    id: 'Q11',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets measured with a UV-vis spectrometer that contained a substance with compound C (benzylethylamine).',
    sources: ['ELN'],
    run() {
      const uv = Object.keys(DATASETS).filter(
        (d) => expOf(d).device.kind === 'UV-vis spectrophotometer'
      )
      const r = uv.filter((d) => DATASETS[d].compounds.includes('CHEBI:90000'))
      return ANSWER(r, [
        { src: 'ELN', op: 'device kind = UV-vis', n: uv.length },
        { src: 'ELN', op: '∧ compound = benzylethylamine', n: r.length },
      ])
    },
  },
  {
    id: 'Q12',
    group: 'Practitioner: dataset catalogue',
    origin: 'practitioner',
    text: 'All datasets that had substance S as catalyst and were measured with a Bruker spectrometer.',
    sources: ['ELN'],
    run() {
      const r = Object.keys(DATASETS).filter((d) => {
        const e = expOf(d)
        return e.biocatalyst === 'PR:BVMO_ACINE' && e.device.vendor === 'Bruker'
      })
      return ANSWER(r, [{ src: 'ELN', op: 'catalyst = BVMO ∧ vendor = Bruker', n: r.length }])
    },
  },

  // ============ GROUP 3: answerable without modelling the answer ==========
  {
    id: "Q4′",
    group: 'Computed, not stored',
    origin: 'added',
    text: 'In which direction is the alanine transaminase reaction physiologically admissible, in each of three media?',
    why: 'No store records this, and no schema has a slot for it. The reaction identifier names the chain, which is direction-symmetric; the direction is named by the medium, which the identifier does not carry.',
    sources: ['RXN', 'ELN'],
    highlight: true,
    run() {
      const r = REACTIONS['RXN:19453']
      const rows = Object.entries(MEDIA).map(([label, mu]) => {
        const d = mediumBias(r.substrates, r.products, mu, BETA, TAU)
        return `${label}: δ = ${(d / BETA >= 0 ? '+' : '') + (d / BETA).toFixed(3)} β → ${directionVerdict(d, BETA)}`
      })
      return ANSWER(rows, [
        { src: 'RXN', op: 'reactants and products of RXN:19453', n: 4 },
        { src: 'ELN', op: 'ambient occupancies of the declared medium', n: 3 },
        { src: '—', op: 'Δ_m = Σ_products w(ℓ,m) − Σ_reactants w(ℓ,m)  ← computed', n: 3 },
      ])
    },
  },
  {
    id: 'Q13',
    group: 'Computed, not stored',
    origin: 'added',
    text: 'What is the role of the solvent in this reaction?',
    why: 'Two waters of identical chemical identity get opposite roles. This is the distinction a single “solvent” class cannot draw, and no curator supplies it: it is a comparison of two boundaries.',
    sources: ['RXN'],
    highlight: true,
    run() {
      const wlm = mediumWeight(55.5, BETA, TAU)
      const axial = 4.0 * BETA
      return ANSWER(
        [
          `ordered active-site water: ρ_str = ${(axial / BETA).toFixed(2)} β ≥ w(ℓ,m) = ${(wlm / BETA).toFixed(2)} β → structural`,
          `bulk water: ρ_str = 0 < w(ℓ,m) → bulk`,
        ],
        [
          { src: 'RXN', op: 'contacts of each solvent leaf', n: 2 },
          { src: '—', op: 'role = structural iff ρ_str ≥ w(ℓ,m)  ← computed', n: 2 },
        ]
      )
    },
  },
  {
    id: 'Q14',
    group: 'Computed, not stored',
    origin: 'added',
    text: 'Which biocatalysts are interchangeable in a process?',
    why: 'Classification needs one number per catalyst and no structural comparison. Two enzymes from different organisms fall in one class because their powers agree — no fold, no alignment, no sequence.',
    sources: ['ELN'],
    run() {
      const POWERS = {
        'PR:TAM_BACIL': 0.55, 'PR:TAM_PSEUD': 0.55,
        'PR:TAM_ARATH': 0.30, 'PR:BVMO_ACINE': 0.72,
      }
      const groups = {}
      Object.entries(POWERS).forEach(([p, v]) => {
        groups[v] = groups[v] || []
        groups[v].push(p)
      })
      return ANSWER(
        Object.entries(groups)
          .sort((a, b) => a[0] - b[0])
          .map(([v, ps]) => `power ${v}: ${ps.join(', ')}`),
        [{ src: 'ELN', op: 'one power per catalyst; group by equality', n: 4 }]
      )
    },
  },
  {
    id: 'Q15',
    group: 'Computed, not stored',
    origin: 'added',
    text: 'Which step of a four-step process can be deleted without missing the target?',
    why: 'Predicted before any experiment, from four numbers. The formalism says which deletion is tolerated and which is not.',
    sources: [],
    run() {
      const chain = [0.45, 0.3, 0.55, 0.2]
      const full = compose(chain)
      const rows = chain.map((_, i) => {
        const v = compose(chain.filter((_, j) => j !== i))
        return `delete rung ${i + 1} → ${v.toFixed(4)} ${v >= 0.8 ? '(tolerated)' : '(fails target 0.80)'}`
      })
      return ANSWER([`full chain: ${full.toFixed(4)}`, ...rows], [
        { src: '—', op: 'π(L) = 1 − Π(1 − πᵢ) for each deletion', n: 4 },
      ])
    },
  },

  // ===================== GROUP 4: refusals ================================
  {
    id: 'Q16',
    group: 'Refused, with a named blocker',
    origin: 'added',
    text: 'At what temperature does this reaction take place, as a derived quantity?',
    why: 'We could return a number here and it would be worthless. The floor is condition-dependent in principle, but at the categorical depth used here the dependence is immaterial.',
    sources: ['ELN'],
    run() {
      return REFUSE(
        'unexpressed',
        'the floor is condition-dependent in principle, but the dependence is immaterial at the categorical depth this framework uses',
        'connect the medium bias to a measured thermodynamic driving force — a substantial and falsifiable step this framework has not taken',
        [{ src: 'ELN', op: 'recorded temperature exists, but is not derived', n: 1 }]
      )
    },
  },
  {
    id: 'Q17',
    group: 'Refused, with a named blocker',
    origin: 'added',
    text: 'Retrieve the buffer used, from the reaction knowledge base.',
    why: 'Refused before any request is issued. Capability containment is decidable statically, so this is a compile-time diagnostic rather than an empty result set at runtime.',
    sources: ['RXN'],
    run() {
      const need = ['buffer', 'ph']
      const missing = need.filter((c) => !SOURCES.RXN.capabilities.includes(c))
      return REFUSE(
        'unexpressed',
        `the reaction knowledge base cannot state: ${missing.join(', ')}`,
        'obtain these features from a source that declares them (the ELN does), or compute them from a retrieved attribute',
        [{ src: 'RXN', op: `static check: requires {${need.join(', ')}}`, n: 0 }]
      )
    },
  },
  {
    id: 'Q18',
    group: 'Refused, with a named blocker',
    origin: 'added',
    text: 'Which enzymes catalyse a reaction that the first step failed to retrieve?',
    why: 'The characteristic failure of federation, and the one a single-source query cannot produce. The step names the predecessor at fault rather than reporting an empty answer.',
    sources: ['RXN', 'PROT'],
    run() {
      return REFUSE(
        'starved',
        'step “find_reaction” answered correctly under its own declaration but supplied no bindings',
        'widen “find_reaction” — the fault is upstream, and the blame walk terminates there rather than accusing a step that answered correctly',
        [
          { src: 'RXN', op: 'find_reaction', n: 0 },
          { src: 'PROT', op: 'find_enzymes (depends on find_reaction)', n: 0 },
        ],
        'find_reaction'
      )
    },
  },
  {
    id: 'Q19',
    group: 'Refused, with a named blocker',
    origin: 'added',
    text: 'A four-step plan run under a retrieval budget of two.',
    why: 'Two steps answer and two report exhausted. A rows-only interface would have returned the same empty table for both, which is exactly the conflation the verdict space exists to remove.',
    sources: ['PROT'],
    run() {
      return REFUSE(
        'exhausted',
        '2 of 4 steps did not run: the budget was spent before they were reached',
        'raise the budget to 4',
        [
          { src: 'PROT', op: 'step 1', n: 1 },
          { src: 'PROT', op: 'step 2', n: 1 },
          { src: 'PROT', op: 'step 3 — budget spent', n: 0 },
          { src: 'PROT', op: 'step 4 — budget spent', n: 0 },
        ]
      )
    },
  },

  // ===================== GROUP 5: extended battery ========================
  {
    id: 'Q21',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which reactions run in opposite directions under the same identifier?',
    sources: ['RXN', 'ELN'],
    run() {
      const r = REACTIONS['RXN:19453']
      const rows = Object.entries(MEDIA)
        .map(([label, mu]) => {
          const d = mediumBias(r.substrates, r.products, mu, BETA, TAU)
          return { label, v: directionVerdict(d, BETA) }
        })
        .filter((x) => x.v !== 'undirected')
        .map((x) => `RXN:19453 in ${x.label}: ${x.v}`)
      return ANSWER(rows, [{ src: '—', op: 'trichotomy per medium', n: rows.length }])
    },
  },
  {
    id: 'Q22',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which enzymes catalysing EC 2.6.1.- are bacterial?',
    sources: ['PROT'],
    run() {
      const r = Object.keys(PROTEINS).filter(
        (p) => PROTEINS[p].ec === '2.6.1.-' && PROTEINS[p].domain === 'Bacteria'
      )
      return ANSWER(
        r.map((p) => `${p} — ${PROTEINS[p].organism}`),
        [{ src: 'PROT', op: 'ec = 2.6.1.- ∧ domain = Bacteria', n: r.length }]
      )
    },
  },
  {
    id: 'Q23',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which pathways contain a reaction whose product is a lactone?',
    sources: ['PATH', 'RXN'],
    run() {
      const r = Object.keys(PATHWAYS).filter((p) =>
        PATHWAYS[p].reactions.some((x) => REACTIONS[x].products.includes('CHEBI:35604'))
      )
      return ANSWER(
        r.map((p) => `${p} — ${PATHWAYS[p].name}`),
        [{ src: 'PATH+RXN', op: 'pathway → reaction → products', n: r.length }]
      )
    },
  },
  {
    id: 'Q24',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which experiments used a buffer above pH 8?',
    sources: ['ELN'],
    run() {
      const r = Object.keys(EXPERIMENTS).filter((e) => EXPERIMENTS[e].buffer.pH > 8)
      return ANSWER(
        r.map((e) => `${e} — ${EXPERIMENTS[e].buffer.name} pH ${EXPERIMENTS[e].buffer.pH}`),
        [{ src: 'ELN', op: 'buffer.pH > 8', n: r.length }]
      )
    },
  },
  {
    id: 'Q25',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which operators ran experiments on a UV-vis instrument?',
    sources: ['ELN'],
    run() {
      const r = [
        ...new Set(
          Object.values(EXPERIMENTS)
            .filter((e) => e.device.kind === 'UV-vis spectrophotometer')
            .map((e) => e.operator)
        ),
      ]
      return ANSWER(r, [{ src: 'ELN', op: 'device kind = UV-vis → operator', n: r.length }])
    },
  },
  {
    id: 'Q26',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which datasets came from an instrument configured for ¹³C?',
    sources: ['ELN'],
    run() {
      const r = Object.keys(DATASETS).filter(
        (d) => expOf(d).device.settings.nucleus === '13C'
      )
      return ANSWER(r, [{ src: 'ELN', op: 'device.settings.nucleus = ¹³C', n: r.length }])
    },
  },
  {
    id: 'Q27',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which enzyme will give the highest enantiomeric excess for a new substrate?',
    why: 'The framework states the FORM of the answer — a power per catalyst per substrate — but not its value. Saying so is more useful than returning a number nobody measured.',
    sources: [],
    run() {
      return REFUSE(
        'unsupported',
        'requires a quantitative structure–selectivity model this framework does not supply',
        'measure a power per catalyst per substrate; the composition law then predicts the chain, but the powers must come from the laboratory',
        [{ src: '—', op: 'no measured powers available', n: 0 }]
      )
    },
  },
  {
    id: 'Q28',
    group: 'Extended battery',
    origin: 'added',
    text: 'Which reaction steps occurred, in order, within a single turnover?',
    why: 'A description-logic defined class over participant roles can state that a reaction HAS a participant, but not that its steps occur in an order. That is a theorem about the fragment, not a gap in the vocabulary.',
    sources: ['RXN'],
    run() {
      return REFUSE(
        'unexpressed',
        'a defined class over participant roles cannot express a residue chain: threading state along a sequence needs a transitive role composed with value restrictions, which the fragment excludes for decidability',
        'ask the chain view rather than the signature view — the two representations answer disjoint question sets, and neither subsumes the other',
        [{ src: 'RXN', op: 'signature view: roles, not order', n: 0 }]
      )
    },
  },
]

export const GROUPS = [...new Set(QUESTIONS.map((q) => q.group))]
