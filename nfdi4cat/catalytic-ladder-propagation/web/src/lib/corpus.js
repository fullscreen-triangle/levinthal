/**
 * The fixture corpus, mirroring validation/fixtures/corpus.py.
 *
 * These are small hand-checkable stand-ins with the same record SHAPE as the
 * public resources, not copies of them.  Every query on this page runs against
 * these records, in the browser, so the results are reproducible and none of
 * them is evidence about biology.
 */

export const CHEBI = {
  'CHEBI:57972': 'L-alanine',
  'CHEBI:16810': '2-oxoglutarate',
  'CHEBI:15361': 'pyruvate',
  'CHEBI:29985': 'L-glutamate',
  'CHEBI:90000': 'benzylethylamine',
  'CHEBI:90001': 'benzylethylketone',
  'CHEBI:17854': 'cyclohexanone',
  'CHEBI:35604': 'ε-caprolactone',
  'CHEBI:17790': 'methanol',
  'CHEBI:33308': 'carboxylic ester',
  'CHEBI:15377': 'water',
  'CHEBI:15379': 'dioxygen',
  'CHEBI:57783': 'NADPH',
  'CHEBI:58349': 'NADP+',
}

export const REACTIONS = {
  'RXN:19453': {
    equation: 'L-alanine + 2-oxoglutarate = pyruvate + L-glutamate',
    ec: '2.6.1.2',
    status: 'approved',
    substrates: ['CHEBI:57972', 'CHEBI:16810'],
    products: ['CHEBI:15361', 'CHEBI:29985'],
    directional: ['RXN:19454', 'RXN:19455'],
    note: 'master reaction; two directional children, as public resources store it',
  },
  'RXN:20504': {
    equation: 'benzylethylamine + pyruvate = benzylethylketone + L-alanine',
    ec: '2.6.1.-',
    status: 'approved',
    substrates: ['CHEBI:90000', 'CHEBI:15361'],
    products: ['CHEBI:90001', 'CHEBI:57972'],
    directional: ['RXN:20505', 'RXN:20506'],
  },
  'RXN:31000': {
    equation: 'cyclohexanone + NADPH + O₂ = ε-caprolactone + NADP⁺ + H₂O',
    ec: '1.14.13.22',
    status: 'approved',
    substrates: ['CHEBI:17854', 'CHEBI:57783', 'CHEBI:15379'],
    products: ['CHEBI:35604', 'CHEBI:58349', 'CHEBI:15377'],
    directional: ['RXN:31001', 'RXN:31002'],
  },
}

export const PROTEINS = {
  'PR:AlaA_ECOLI': {
    name: 'alanine transaminase AlaA',
    organism: 'Escherichia coli',
    domain: 'Bacteria',
    ec: '2.6.1.2',
    catalyses: ['RXN:19453'],
    sequence: 'MADTRPERLSAFGSSFLDAMRLKAQGHDVLNFSAGEPDF',
  },
  'PR:ALT1_HUMAN': {
    name: 'alanine aminotransferase 1',
    organism: 'Homo sapiens',
    domain: 'Eukaryota',
    ec: '2.6.1.2',
    catalyses: ['RXN:19453'],
    sequence: 'MASSTGDRSQAVRHGLRAKVLTLDGMNPRVRRVEYAVRGPIC',
  },
  'PR:TAM_BACIL': {
    name: 'ω-transaminase',
    organism: 'Bacillus megaterium',
    domain: 'Bacteria',
    ec: '2.6.1.-',
    catalyses: ['RXN:20504'],
    sequence: 'MSFNAEQLNQIDAAHHLHPFTDMKSLNQAGARVMTRGEGVYLWD',
  },
  'PR:TAM_PSEUD': {
    name: 'ω-transaminase',
    organism: 'Pseudomonas fluorescens',
    domain: 'Bacteria',
    ec: '2.6.1.-',
    catalyses: ['RXN:20504'],
    sequence: 'MTQPLNVAECRALDAAHHLHPFTSLKALNEQGACVITKAEGAYIYD',
  },
  'PR:TAM_ARATH': {
    name: 'transaminase',
    organism: 'Arabidopsis thaliana',
    domain: 'Eukaryota',
    ec: '2.6.1.-',
    catalyses: ['RXN:20504'],
    sequence: 'MSLNTEQLNAIDAAHHLHPFTDMKSLNEKGSRVITRAEGVYLWD',
  },
  'PR:BVMO_ACINE': {
    name: 'cyclohexanone monooxygenase',
    organism: 'Acinetobacter calcoaceticus',
    domain: 'Bacteria',
    ec: '1.14.13.22',
    catalyses: ['RXN:31000'],
    sequence: 'MSQKMDFDAIVIGGGFGGLYAVKKLRDELELKVQAFDKATDVGGTWYWNRYPGA',
  },
}

export const PATHWAYS = {
  'PW:00250': {
    name: 'Alanine, aspartate and glutamate metabolism',
    reactions: ['RXN:19453'],
  },
  'PW:00930': { name: 'Caprolactam degradation', reactions: ['RXN:31000'] },
}

export const EXPERIMENTS = {
  BT3: {
    title: 'biocatalytic transformation BT3',
    operator: 'Y. Dikova',
    date: '2026-03-23',
    reaction: 'RXN:20504',
    biocatalyst: 'PR:TAM_BACIL',
    buffer: { name: 'HEPES', mM: 50, pH: 7.5 },
    temperature: 30,
    device: {
      id: 'DEV:UV-1900i',
      kind: 'UV-vis spectrophotometer',
      vendor: 'Shimadzu',
      settings: { wavelength_nm: 245, bandwidth_nm: 1.0 },
    },
    datasets: ['DS:0031'],
  },
  MT7: {
    title: 'methyl transfer with mt-X',
    operator: 'M. Doerr',
    date: '2026-02-11',
    reaction: null,
    biocatalyst: 'PR:MTX_BACSU',
    buffer: { name: 'Tris-HCl', mM: 100, pH: 8.0 },
    temperature: 37,
    device: {
      id: 'DEV:AVANCE-400',
      kind: 'NMR spectrometer',
      vendor: 'Bruker',
      settings: { field_MHz: 400, nucleus: '1H' },
    },
    datasets: ['DS:0017'],
  },
  KR9: {
    title: 'kinetic resolution with PFE',
    operator: 'Y. Dikova',
    date: '2026-04-02',
    reaction: null,
    biocatalyst: 'PR:PFE_PSEFL',
    buffer: { name: 'HEPES', mM: 50, pH: 9.0 },
    temperature: 25,
    device: {
      id: 'DEV:UV-1900i',
      kind: 'UV-vis spectrophotometer',
      vendor: 'Shimadzu',
      settings: { wavelength_nm: 410, bandwidth_nm: 2.0 },
    },
    datasets: ['DS:0044'],
  },
  BV2: {
    title: 'BVMO-Y substrate scope screen',
    operator: 'D. Linke',
    date: '2026-05-14',
    reaction: 'RXN:31000',
    biocatalyst: 'PR:BVMO_ACINE',
    buffer: { name: 'phosphate', mM: 50, pH: 7.0 },
    temperature: 30,
    device: {
      id: 'DEV:AVANCE-400',
      kind: 'NMR spectrometer',
      vendor: 'Bruker',
      settings: { field_MHz: 400, nucleus: '13C' },
    },
    datasets: ['DS:0052', 'DS:0053'],
  },
  BV3: {
    title: 'BVMO-Y conversion time course',
    operator: 'D. Linke',
    date: '2026-05-15',
    reaction: 'RXN:31000',
    biocatalyst: 'PR:BVMO_ACINE',
    buffer: { name: 'phosphate', mM: 50, pH: 7.0 },
    temperature: 30,
    device: {
      id: 'DEV:UV-1900i',
      kind: 'UV-vis spectrophotometer',
      vendor: 'Shimadzu',
      settings: { wavelength_nm: 340, bandwidth_nm: 1.0 },
    },
    datasets: ['DS:0061'],
  },
  BV4: {
    title: 'BVMO-Y proton check',
    operator: 'D. Linke',
    date: '2026-05-16',
    reaction: 'RXN:31000',
    biocatalyst: 'PR:BVMO_ACINE',
    buffer: { name: 'phosphate', mM: 50, pH: 7.0 },
    temperature: 30,
    device: {
      id: 'DEV:AVANCE-400',
      kind: 'NMR spectrometer',
      vendor: 'Bruker',
      settings: { field_MHz: 400, nucleus: '1H' },
    },
    datasets: ['DS:0062'],
  },
}

export const DATASETS = {
  'DS:0017': { experiment: 'MT7', type: 'NMR', compounds: ['CHEBI:17790'] },
  'DS:0031': {
    experiment: 'BT3',
    type: 'UV-vis',
    compounds: ['CHEBI:90000', 'CHEBI:15361'],
  },
  'DS:0044': { experiment: 'KR9', type: 'UV-vis', compounds: ['CHEBI:33308'] },
  'DS:0052': { experiment: 'BV2', type: 'NMR', compounds: ['CHEBI:17854'] },
  'DS:0053': { experiment: 'BV2', type: 'NMR', compounds: ['CHEBI:35604'] },
  'DS:0061': { experiment: 'BV3', type: 'UV-vis', compounds: ['CHEBI:17854'] },
  'DS:0062': { experiment: 'BV4', type: 'NMR', compounds: ['CHEBI:17854'] },
}

/** Ambient occupancies. Illustrative, as the paper states — not measured. */
export const MEDIA = {
  'glutamate-depleted cytosol': {
    'CHEBI:57972': 2.0e-3,
    'CHEBI:16810': 1.0e-4,
    'CHEBI:15361': 5.0e-4,
    'CHEBI:29985': 1.0e-6,
    note: '2-oxoglutarate ambient, L-glutamate held low by downstream demand',
  },
  '2-oxoglutarate-depleted cytosol': {
    'CHEBI:57972': 2.0e-3,
    'CHEBI:16810': 1.0e-6,
    'CHEBI:15361': 5.0e-4,
    'CHEBI:29985': 1.0e-4,
    note: 'the same enzyme used the other way, to feed nitrogen',
  },
  'balanced medium': {
    'CHEBI:57972': 1.0e-4,
    'CHEBI:16810': 1.0e-4,
    'CHEBI:15361': 1.0e-4,
    'CHEBI:29985': 1.0e-4,
    note: 'no imbalance: the medium declines to orient the chain',
  },
}

export const SOURCES = {
  RXN: {
    label: 'Reaction knowledge base',
    shape: 'SPARQL endpoint',
    capabilities: ['reaction', 'participant', 'ec', 'equation', 'status', 'direction-asserted'],
  },
  PROT: {
    label: 'Protein resource',
    shape: 'SPARQL + REST',
    capabilities: ['protein', 'organism', 'lineage', 'ec', 'sequence', 'catalyses'],
  },
  PATH: {
    label: 'Pathway resource',
    shape: 'flat-file REST',
    capabilities: ['pathway', 'reaction-membership'],
  },
  ELN: {
    label: 'Electronic lab notebook',
    shape: 'local records (LARA-shaped)',
    capabilities: [
      'experiment', 'operator', 'date', 'device', 'device-settings',
      'buffer', 'ph', 'temperature', 'dataset', 'compound',
    ],
  },
}

export const BETA = 3.7e-4
export const TAU = 1.0e-3
