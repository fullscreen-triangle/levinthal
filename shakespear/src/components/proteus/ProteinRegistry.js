/**
 * Protein Registry
 * =================
 * Maps GLB models to their PDB/CIF structures.
 * Each entry pairs a visual model with its physical coordinates.
 *
 * The GLB is the canvas. The CIF is the physics. Together: shakespear.
 */

const PROTEINS = [
  {
    id: 'troponin',
    name: 'Troponin C',
    glb: '/models/conformational_transition_of_troponin.glb',
    cif: '/pdb/1QO1.cif',
    pdbId: '1QO1',
    description: 'Calcium-binding protein in muscle regulation',
    organism: 'Gallus gallus',
    residues: 162,
    classification: 'Calcium-binding',
  },
  {
    id: 'gfp',
    name: 'Green Fluorescent Protein',
    glb: '/models/gfp__green_fluorescent_protein.glb',
    cif: '/pdb/1SKY.cif',
    pdbId: '1SKY',
    description: 'Fluorescent protein from Aequorea victoria',
    organism: 'Aequorea victoria',
    residues: 238,
    classification: 'Fluorescent protein',
  },
  {
    id: 'pokeweed',
    name: 'Pokeweed Antiviral Protein',
    glb: '/models/pokeweed_antiviral_protein.glb',
    cif: '/pdb/2QLE.cif',
    pdbId: '2QLE',
    description: 'Ribosome-inactivating protein',
    organism: 'Phytolacca americana',
    residues: 262,
    classification: 'Antiviral',
  },
  {
    id: 'glucokinase',
    name: 'Glucokinase',
    glb: '/models/glucokinase__glu__atp.glb',
    cif: '/pdb/3TZ1.cif',
    pdbId: '3TZ1',
    description: 'Glucose phosphorylation enzyme with glucose and ATP',
    organism: 'Homo sapiens',
    residues: 448,
    classification: 'Transferase',
  },
  {
    id: 'atp_synthase',
    name: 'ATP Synthase',
    glb: '/models/atp_synthase.glb',
    cif: '/pdb/5FIL.cif',
    pdbId: '5FIL',
    description: 'Mitochondrial ATP synthase complex',
    organism: 'Bos taurus',
    residues: 2780,
    classification: 'Hydrolase/Synthase',
  },
  {
    id: 'atp_synthase_v2',
    name: 'ATP Synthase (variant)',
    glb: '/models/atp_synthase_colour_change.glb',
    cif: '/pdb/9D7U.cif',
    pdbId: '9D7U',
    description: 'ATP synthase structural variant',
    organism: 'Various',
    residues: null,
    classification: 'Hydrolase/Synthase',
  },
];

export default PROTEINS;
