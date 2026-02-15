#!/usr/bin/env python3
"""
PDB Loader: Download and parse protein structures from RCSB PDB.

This module provides functionality to:
1. Download PDB files from RCSB
2. Parse atom coordinates and metadata
3. Identify secondary structures (alpha helices, beta sheets)
4. Extract residue and chain information

Author: Kundai Farai Sachikonye
Date: February 2026
"""

import urllib.request
import urllib.error
import gzip
import io
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# =============================================================================
# Data Classes
# =============================================================================

class SecondaryStructure(Enum):
    """Secondary structure types."""
    HELIX = "helix"
    SHEET = "sheet"
    COIL = "coil"
    TURN = "turn"
    UNKNOWN = "unknown"


@dataclass
class Atom:
    """Represents an atom in the protein structure."""
    serial: int
    name: str
    alt_loc: str
    residue_name: str
    chain_id: str
    residue_seq: int
    insertion_code: str
    position: np.ndarray  # x, y, z in Angstroms
    occupancy: float
    temp_factor: float
    element: str
    charge: str

    # Ternary state for spectrometer framework
    ternary_state: int = 1  # 0=ground, 1=natural, 2=excited

    @property
    def residue_id(self) -> str:
        """Unique residue identifier."""
        return f"{self.chain_id}:{self.residue_name}{self.residue_seq}"

    def distance_to(self, other: 'Atom') -> float:
        """Calculate distance to another atom in Angstroms."""
        return np.linalg.norm(self.position - other.position)


@dataclass
class Residue:
    """Represents a residue in the protein."""
    name: str
    seq_num: int
    chain_id: str
    atoms: List[Atom] = field(default_factory=list)
    secondary_structure: SecondaryStructure = SecondaryStructure.UNKNOWN

    @property
    def ca_atom(self) -> Optional[Atom]:
        """Get alpha carbon atom."""
        for atom in self.atoms:
            if atom.name.strip() == "CA":
                return atom
        return None

    @property
    def centroid(self) -> np.ndarray:
        """Calculate centroid of all atoms."""
        if not self.atoms:
            return np.zeros(3)
        positions = np.array([a.position for a in self.atoms])
        return positions.mean(axis=0)

    @property
    def residue_id(self) -> str:
        """Unique residue identifier."""
        return f"{self.chain_id}:{self.name}{self.seq_num}"


@dataclass
class Helix:
    """Represents an alpha helix."""
    serial: int
    helix_id: str
    start_residue: str
    start_chain: str
    start_seq: int
    end_residue: str
    end_chain: str
    end_seq: int
    helix_class: int
    length: int

    def contains_residue(self, chain: str, seq: int) -> bool:
        """Check if residue is in this helix."""
        return (chain == self.start_chain and
                self.start_seq <= seq <= self.end_seq)


@dataclass
class Sheet:
    """Represents a beta sheet strand."""
    strand: int
    sheet_id: str
    num_strands: int
    start_residue: str
    start_chain: str
    start_seq: int
    end_residue: str
    end_chain: str
    end_seq: int
    sense: int

    def contains_residue(self, chain: str, seq: int) -> bool:
        """Check if residue is in this strand."""
        return (chain == self.start_chain and
                self.start_seq <= seq <= self.end_seq)


@dataclass
class ProteinStructure:
    """Complete protein structure from PDB."""
    pdb_id: str
    title: str
    atoms: List[Atom] = field(default_factory=list)
    residues: Dict[str, Residue] = field(default_factory=dict)
    helices: List[Helix] = field(default_factory=list)
    sheets: List[Sheet] = field(default_factory=list)
    chains: Set[str] = field(default_factory=set)
    resolution: Optional[float] = None

    def get_atom(self, serial: int) -> Optional[Atom]:
        """Get atom by serial number."""
        for atom in self.atoms:
            if atom.serial == serial:
                return atom
        return None

    def get_residue(self, chain: str, seq: int) -> Optional[Residue]:
        """Get residue by chain and sequence number."""
        key = f"{chain}:{seq}"
        return self.residues.get(key)

    def get_chain_atoms(self, chain_id: str) -> List[Atom]:
        """Get all atoms in a chain."""
        return [a for a in self.atoms if a.chain_id == chain_id]

    def get_backbone(self, chain_id: Optional[str] = None) -> List[Atom]:
        """Get backbone atoms (N, CA, C, O)."""
        backbone_names = {"N", "CA", "C", "O"}
        atoms = self.atoms if chain_id is None else self.get_chain_atoms(chain_id)
        return [a for a in atoms if a.name.strip() in backbone_names]

    def get_ca_trace(self, chain_id: Optional[str] = None) -> np.ndarray:
        """Get CA atom positions as array."""
        atoms = self.atoms if chain_id is None else self.get_chain_atoms(chain_id)
        ca_atoms = [a for a in atoms if a.name.strip() == "CA"]
        if not ca_atoms:
            return np.array([])
        return np.array([a.position for a in ca_atoms])

    def get_helices_atoms(self) -> List[Atom]:
        """Get all atoms in alpha helices."""
        helix_atoms = []
        for atom in self.atoms:
            for helix in self.helices:
                if helix.contains_residue(atom.chain_id, atom.residue_seq):
                    helix_atoms.append(atom)
                    break
        return helix_atoms

    def get_sheets_atoms(self) -> List[Atom]:
        """Get all atoms in beta sheets."""
        sheet_atoms = []
        for atom in self.atoms:
            for sheet in self.sheets:
                if sheet.contains_residue(atom.chain_id, atom.residue_seq):
                    sheet_atoms.append(atom)
                    break
        return sheet_atoms

    def get_helix_by_id(self, helix_id: str) -> Optional[Helix]:
        """Get helix by ID."""
        for helix in self.helices:
            if helix.helix_id.strip() == helix_id.strip():
                return helix
        return None

    def get_helix_atoms(self, helix_id: str) -> List[Atom]:
        """Get atoms in specific helix."""
        helix = self.get_helix_by_id(helix_id)
        if not helix:
            return []
        return [a for a in self.atoms
                if helix.contains_residue(a.chain_id, a.residue_seq)]

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of structure."""
        positions = np.array([a.position for a in self.atoms])
        return positions.min(axis=0), positions.max(axis=0)

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        # Approximate atomic masses
        masses = {"C": 12.0, "N": 14.0, "O": 16.0, "S": 32.0, "H": 1.0}
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        for atom in self.atoms:
            m = masses.get(atom.element, 12.0)
            weighted_pos += m * atom.position
            total_mass += m
        return weighted_pos / total_mass if total_mass > 0 else np.zeros(3)

    def assign_secondary_structure(self):
        """Assign secondary structure to residues."""
        for key, residue in self.residues.items():
            # Check helices
            for helix in self.helices:
                if helix.contains_residue(residue.chain_id, residue.seq_num):
                    residue.secondary_structure = SecondaryStructure.HELIX
                    break
            # Check sheets
            if residue.secondary_structure == SecondaryStructure.UNKNOWN:
                for sheet in self.sheets:
                    if sheet.contains_residue(residue.chain_id, residue.seq_num):
                        residue.secondary_structure = SecondaryStructure.SHEET
                        break
            # Default to coil
            if residue.secondary_structure == SecondaryStructure.UNKNOWN:
                residue.secondary_structure = SecondaryStructure.COIL

    def summary(self) -> str:
        """Return summary string."""
        n_helix_res = sum(1 for r in self.residues.values()
                         if r.secondary_structure == SecondaryStructure.HELIX)
        n_sheet_res = sum(1 for r in self.residues.values()
                         if r.secondary_structure == SecondaryStructure.SHEET)
        return (f"PDB: {self.pdb_id}\n"
                f"Title: {self.title[:60]}...\n"
                f"Atoms: {len(self.atoms)}\n"
                f"Residues: {len(self.residues)}\n"
                f"Chains: {', '.join(sorted(self.chains))}\n"
                f"Helices: {len(self.helices)} ({n_helix_res} residues)\n"
                f"Sheets: {len(self.sheets)} strands ({n_sheet_res} residues)\n"
                f"Resolution: {self.resolution or 'N/A'} Å")


# =============================================================================
# PDB Downloader
# =============================================================================

def download_pdb(pdb_id: str, save_dir: Optional[Path] = None) -> str:
    """
    Download PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB ID (e.g., "4AZU")
        save_dir: Optional directory to save file

    Returns:
        PDB file content as string
    """
    pdb_id = pdb_id.upper()

    # Check cache first
    if save_dir:
        save_dir = Path(save_dir)
        cache_file = save_dir / f"{pdb_id}.pdb"
        if cache_file.exists():
            print(f"  Loading cached: {cache_file}")
            return cache_file.read_text()

    # RCSB PDB URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"

    print(f"  Downloading from RCSB: {pdb_id}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            compressed = response.read()

        # Decompress
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
            content = f.read().decode('utf-8')

        # Save to cache
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(content)
            print(f"  Saved to: {cache_file}")

        return content

    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"PDB ID '{pdb_id}' not found in RCSB")
        raise
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to RCSB: {e}")


# =============================================================================
# PDB Parser
# =============================================================================

def parse_pdb(content: str) -> ProteinStructure:
    """
    Parse PDB file content.

    Args:
        content: PDB file content as string

    Returns:
        ProteinStructure object
    """
    structure = ProteinStructure(pdb_id="", title="")

    for line in content.split('\n'):
        record = line[:6].strip()

        if record == "HEADER":
            # Extract PDB ID from header
            if len(line) >= 66:
                structure.pdb_id = line[62:66].strip()

        elif record == "TITLE":
            structure.title += line[10:80].strip() + " "

        elif record == "REMARK":
            # Look for resolution
            if "RESOLUTION" in line and "ANGSTROMS" in line:
                match = re.search(r'(\d+\.?\d*)\s*ANGSTROMS', line)
                if match:
                    structure.resolution = float(match.group(1))

        elif record == "HELIX":
            helix = _parse_helix(line)
            if helix:
                structure.helices.append(helix)

        elif record == "SHEET":
            sheet = _parse_sheet(line)
            if sheet:
                structure.sheets.append(sheet)

        elif record == "ATOM" or record == "HETATM":
            atom = _parse_atom(line)
            if atom:
                structure.atoms.append(atom)
                structure.chains.add(atom.chain_id)

                # Add to residue dictionary
                res_key = f"{atom.chain_id}:{atom.residue_seq}"
                if res_key not in structure.residues:
                    structure.residues[res_key] = Residue(
                        name=atom.residue_name,
                        seq_num=atom.residue_seq,
                        chain_id=atom.chain_id
                    )
                structure.residues[res_key].atoms.append(atom)

    # Assign secondary structure
    structure.assign_secondary_structure()

    return structure


def _parse_atom(line: str) -> Optional[Atom]:
    """Parse ATOM/HETATM record."""
    try:
        return Atom(
            serial=int(line[6:11].strip()),
            name=line[12:16].strip(),
            alt_loc=line[16].strip(),
            residue_name=line[17:20].strip(),
            chain_id=line[21].strip() or "A",
            residue_seq=int(line[22:26].strip()),
            insertion_code=line[26].strip(),
            position=np.array([
                float(line[30:38].strip()),
                float(line[38:46].strip()),
                float(line[46:54].strip())
            ]),
            occupancy=float(line[54:60].strip()) if line[54:60].strip() else 1.0,
            temp_factor=float(line[60:66].strip()) if line[60:66].strip() else 0.0,
            element=line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0],
            charge=line[78:80].strip() if len(line) > 78 else ""
        )
    except (ValueError, IndexError):
        return None


def _parse_helix(line: str) -> Optional[Helix]:
    """Parse HELIX record."""
    try:
        return Helix(
            serial=int(line[7:10].strip()),
            helix_id=line[11:14].strip(),
            start_residue=line[15:18].strip(),
            start_chain=line[19].strip() or "A",
            start_seq=int(line[21:25].strip()),
            end_residue=line[27:30].strip(),
            end_chain=line[31].strip() or "A",
            end_seq=int(line[33:37].strip()),
            helix_class=int(line[38:40].strip()) if line[38:40].strip() else 1,
            length=int(line[71:76].strip()) if len(line) > 71 and line[71:76].strip() else 0
        )
    except (ValueError, IndexError):
        return None


def _parse_sheet(line: str) -> Optional[Sheet]:
    """Parse SHEET record."""
    try:
        return Sheet(
            strand=int(line[7:10].strip()),
            sheet_id=line[11:14].strip(),
            num_strands=int(line[14:16].strip()) if line[14:16].strip() else 1,
            start_residue=line[17:20].strip(),
            start_chain=line[21].strip() or "A",
            start_seq=int(line[22:26].strip()),
            end_residue=line[28:31].strip(),
            end_chain=line[32].strip() or "A",
            end_seq=int(line[33:37].strip()),
            sense=int(line[38:40].strip()) if line[38:40].strip() else 0
        )
    except (ValueError, IndexError):
        return None


# =============================================================================
# High-Level API
# =============================================================================

def load_protein(pdb_id: str, cache_dir: Optional[Path] = None) -> ProteinStructure:
    """
    Download and parse a protein structure from PDB.

    Args:
        pdb_id: 4-character PDB ID
        cache_dir: Optional directory for caching PDB files

    Returns:
        ProteinStructure object
    """
    content = download_pdb(pdb_id, cache_dir)
    structure = parse_pdb(content)

    # Ensure PDB ID is set
    if not structure.pdb_id:
        structure.pdb_id = pdb_id.upper()

    return structure


def load_local_pdb(filepath: Path) -> ProteinStructure:
    """
    Load protein structure from local PDB file.

    Args:
        filepath: Path to PDB file

    Returns:
        ProteinStructure object
    """
    content = Path(filepath).read_text()
    return parse_pdb(content)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Load azurin
    print("Loading protein structure...")
    protein = load_protein("4AZU", cache_dir=Path("data/pdb"))
    print(protein.summary())

    print(f"\nHelices:")
    for helix in protein.helices:
        print(f"  {helix.helix_id}: {helix.start_residue}{helix.start_seq} - "
              f"{helix.end_residue}{helix.end_seq} (length: {helix.length})")

    print(f"\nSheets:")
    for sheet in protein.sheets:
        print(f"  {sheet.sheet_id} strand {sheet.strand}: "
              f"{sheet.start_residue}{sheet.start_seq} - "
              f"{sheet.end_residue}{sheet.end_seq}")
