/**
 * Structure File Parser (PDB + mmCIF)
 * =====================================
 * Parses both PDB format and mmCIF format into atom coordinates and residues.
 * Pure JS, zero dependencies.
 */

const AA_3TO1 = {
  ALA:'A', ARG:'R', ASN:'N', ASP:'D', CYS:'C', GLN:'Q', GLU:'E', GLY:'G',
  HIS:'H', ILE:'I', LEU:'L', LYS:'K', MET:'M', PHE:'F', PRO:'P', SER:'S',
  THR:'T', TRP:'W', TYR:'Y', VAL:'V',
};

/** Detect format from file content. */
function detectFormat(text) {
  const first = text.trim().substring(0, 200);
  if (first.startsWith('data_')) return 'cif';
  if (first.match(/^(HEADER|ATOM|HETATM|REMARK)/m)) return 'pdb';
  if (first.includes('_atom_site.')) return 'cif';
  return 'pdb'; // default
}

/** Parse PDB format (fixed-width columns). */
function parsePDBFormat(text) {
  const atoms = [];
  const residues = new Map();

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

    atoms.push({ atomName, resName, chain, resSeq, x, y, z, element });

    const resId = `${chain}:${resSeq}`;
    if (!residues.has(resId)) {
      residues.set(resId, {
        name: resName, code1: AA_3TO1[resName] || 'X',
        chain, resSeq, atoms: [],
      });
    }
    residues.get(resId).atoms.push({ x, y, z, name: atomName, element });
  }

  return { atoms, residues };
}

/** Parse mmCIF format (whitespace-delimited with column headers). */
function parseCIFFormat(text) {
  const atoms = [];
  const residues = new Map();

  // Find the _atom_site loop
  const lines = text.split('\n');
  let inAtomSite = false;
  let columns = [];
  let colMap = {};

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Detect start of atom_site loop
    if (line === 'loop_') {
      // Check if next lines are _atom_site columns
      const nextLine = (lines[i + 1] || '').trim();
      if (nextLine.startsWith('_atom_site.')) {
        inAtomSite = true;
        columns = [];
        colMap = {};
        continue;
      }
    }

    if (inAtomSite && line.startsWith('_atom_site.')) {
      columns.push(line);
      colMap[line] = columns.length - 1;
      continue;
    }

    // Data lines in atom_site loop
    if (inAtomSite && columns.length > 0 && !line.startsWith('_') && !line.startsWith('#') && line.length > 0 && !line.startsWith('loop_')) {
      // Split by whitespace, respecting quoted strings
      const fields = splitCIFLine(line);
      if (fields.length < columns.length) {
        // End of atom_site block
        if (line.startsWith('#') || line.startsWith('loop_') || line === '') {
          inAtomSite = false;
          continue;
        }
        // Might be a continuation or short line, skip
        continue;
      }

      const get = (col) => fields[colMap[col]] || '';

      const group = get('_atom_site.group_PDB');
      if (group !== 'ATOM' && group !== 'HETATM') continue;

      const atomName = get('_atom_site.label_atom_id') || get('_atom_site.auth_atom_id');
      const resName = get('_atom_site.label_comp_id') || get('_atom_site.auth_comp_id');
      const chain = get('_atom_site.auth_asym_id') || get('_atom_site.label_asym_id');
      const resSeq = parseInt(get('_atom_site.auth_seq_id') || get('_atom_site.label_seq_id'));
      const x = parseFloat(get('_atom_site.Cartn_x'));
      const y = parseFloat(get('_atom_site.Cartn_y'));
      const z = parseFloat(get('_atom_site.Cartn_z'));
      const element = get('_atom_site.type_symbol');
      const modelNum = get('_atom_site.pdbx_PDB_model_num');

      // Only take model 1
      if (modelNum && modelNum !== '1') continue;
      if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

      atoms.push({ atomName, resName, chain, resSeq, x, y, z, element });

      const resId = `${chain}:${resSeq}`;
      if (!residues.has(resId)) {
        residues.set(resId, {
          name: resName, code1: AA_3TO1[resName] || 'X',
          chain, resSeq, atoms: [],
        });
      }
      residues.get(resId).atoms.push({ x, y, z, name: atomName, element });
    }

    // End of atom_site block
    if (inAtomSite && columns.length > 0 && (line.startsWith('#') || line.startsWith('loop_'))) {
      inAtomSite = false;
    }
  }

  return { atoms, residues };
}

/** Split a CIF data line by whitespace, handling single-quoted strings. */
function splitCIFLine(line) {
  const fields = [];
  let i = 0;
  while (i < line.length) {
    // Skip whitespace
    while (i < line.length && line[i] === ' ') i++;
    if (i >= line.length) break;

    if (line[i] === "'") {
      // Quoted string
      i++;
      let start = i;
      while (i < line.length && !(line[i] === "'" && (i + 1 >= line.length || line[i + 1] === ' '))) i++;
      fields.push(line.substring(start, i));
      i++; // skip closing quote
    } else {
      // Unquoted field
      let start = i;
      while (i < line.length && line[i] !== ' ') i++;
      fields.push(line.substring(start, i));
    }
  }
  return fields;
}

/** Parse any structure file (PDB or mmCIF). */
export function parsePDB(text) {
  const format = detectFormat(text);
  const { atoms, residues } = format === 'cif' ? parseCIFFormat(text) : parsePDBFormat(text);

  // Compute CA positions
  const caPositions = [];
  const sequence = [];
  for (const [, res] of residues) {
    const ca = res.atoms.find(a => a.name === 'CA');
    if (ca && res.code1 !== 'X') {
      caPositions.push([ca.x, ca.y, ca.z]);
      sequence.push(res.code1);
    }
  }

  return {
    atoms,
    residues: Array.from(residues.values()),
    caPositions,
    sequence: sequence.join(''),
    nResidues: sequence.length,
    nAtoms: atoms.length,
    format,
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
