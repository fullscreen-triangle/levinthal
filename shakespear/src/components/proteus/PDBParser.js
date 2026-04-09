/**
 * PDB File Parser
 * ===============
 * Parses PDB format text into atom coordinates and residue identities.
 * Pure JS, no dependencies. 50 lines of actual parsing.
 *
 * PDB format: fixed-width columns
 *   ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
 *   cols 1-6: record type
 *   cols 7-11: atom serial
 *   cols 13-16: atom name
 *   cols 18-20: residue name
 *   cols 22: chain ID
 *   cols 23-26: residue sequence number
 *   cols 31-38: x coordinate
 *   cols 39-46: y coordinate
 *   cols 47-54: z coordinate
 */

const AA_3TO1 = {
  ALA:'A', ARG:'R', ASN:'N', ASP:'D', CYS:'C', GLN:'Q', GLU:'E', GLY:'G',
  HIS:'H', ILE:'I', LEU:'L', LYS:'K', MET:'M', PHE:'F', PRO:'P', SER:'S',
  THR:'T', TRP:'W', TYR:'Y', VAL:'V',
};

export function parsePDB(text) {
  const atoms = [];
  const residues = new Map(); // resId -> { name, chain, atoms: [{x,y,z,name}] }

  for (const line of text.split('\n')) {
    const record = line.substring(0, 6).trim();
    if (record !== 'ATOM' && record !== 'HETATM') continue;

    const atomName = line.substring(12, 16).trim();
    const resName = line.substring(17, 20).trim();
    const chain = line.substring(21, 22).trim();
    const resSeq = parseInt(line.substring(22, 26));
    const x = parseFloat(line.substring(30, 38));
    const y = parseFloat(line.substring(38, 46));
    const z = parseFloat(line.substring(46, 54));
    const element = line.substring(76, 78).trim() || atomName[0];

    if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

    const atom = { atomName, resName, chain, resSeq, x, y, z, element };
    atoms.push(atom);

    const resId = `${chain}:${resSeq}`;
    if (!residues.has(resId)) {
      residues.set(resId, {
        name: resName,
        code1: AA_3TO1[resName] || 'X',
        chain, resSeq,
        atoms: [],
      });
    }
    residues.get(resId).atoms.push({ x, y, z, name: atomName, element });
  }

  // Compute CA positions (alpha carbons) for each residue
  const caPositions = [];
  const sequence = [];
  for (const [, res] of residues) {
    const ca = res.atoms.find(a => a.name === 'CA');
    if (ca) {
      caPositions.push([ca.x, ca.y, ca.z]);
      sequence.push(res.code1);
    }
  }

  return {
    atoms,
    residues: Array.from(residues.values()),
    caPositions,         // Nx3 array of alpha-carbon coords
    sequence: sequence.join(''),
    nResidues: sequence.length,
    nAtoms: atoms.length,
  };
}

/** Compute bounding box and center of atom coordinates. */
export function computeBounds(atoms) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const a of atoms) {
    minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x);
    minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
    minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z);
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2],
    size: Math.max(maxX-minX, maxY-minY, maxZ-minZ),
  };
}
