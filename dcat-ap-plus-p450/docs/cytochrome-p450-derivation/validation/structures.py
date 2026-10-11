"""Fetch and parse the structures and sequences used by the P450 experiments.

Structures are read from the PDB format (fixed columns), first model only,
highest-occupancy altloc only. Identities are not taken on trust: every
entry is checked against its own COMPND record, and the check is recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from common import DATA, fetch

# PDB entries; "group" is the hypothesis being tested, "expect" a substring
# the entry's own COMPND/HEADER must contain for it to be used.
P450_ENTRIES = {
    "2CPP": ("CYP101A1, camphor-bound", "P450"),
    "1PHC": ("CYP101A1, substrate-free", "P450"),
    "1TQN": ("CYP3A4", "P450"),
    "1OG2": ("CYP2C9", "P450"),
    "2F9Q": ("CYP2D6", "P450"),
    "2HI4": ("CYP1A2", "P450"),
    "3E6I": ("CYP2E1", "P450"),
    "1BU7": ("CYP102A1 heme domain", "P450"),
    "3LD6": ("CYP51A1", "P450"),
    "3EQM": ("CYP19A1", "AROMATASE"),
    "3RUK": ("CYP17A1", "P450"),
    "3N9Y": ("CYP11A1", "P450"),
}
OTHER_HEME_ENTRIES = {
    "1A6M": ("myoglobin", "MYOGLOBIN"),
    "2HHB": ("hemoglobin", "HEMOGLOBIN"),
    "2CYP": ("cytochrome c peroxidase", "PEROXIDASE"),
    "1ATJ": ("horseradish peroxidase", "PEROXIDASE"),
    "7CAT": ("catalase", "CATALASE"),
    "1CPO": ("chloroperoxidase", "PEROXIDASE"),
    "1NOD": ("nitric oxide synthase oxygenase", "NITRIC"),
    "1HRC": ("cytochrome c", "CYTOCHROME C"),
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}
HEME_NAMES = {"HEM", "HEC", "HEA", "HEB"}


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    element: str
    hetero: bool


@dataclass
class Structure:
    pdb_id: str
    title: str
    compound: str
    resolution: float | None
    atoms: list[Atom] = field(default_factory=list)

    def coords(self, sel=None) -> np.ndarray:
        atoms = self.atoms if sel is None else [a for a in self.atoms if sel(a)]
        return np.array([a.xyz for a in atoms])

    def residues(self, chain: str):
        out, seen = [], set()
        for a in self.atoms:
            key = (a.chain, a.resseq, a.icode)
            if a.chain == chain and not a.hetero and a.resname in THREE_TO_ONE and key not in seen:
                seen.add(key)
                out.append((a.resseq, a.icode, a.resname))
        return out

    def sequence(self, chain: str) -> str:
        return "".join(THREE_TO_ONE[r[2]] for r in self.residues(chain))


def load(pdb_id: str) -> Structure:
    path = fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb", DATA / "pdb" / f"{pdb_id}.pdb")
    title, compound, resolution = [], [], None
    atoms, best = [], {}
    for line in path.read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec.startswith("TITLE"):
            title.append(line[10:80].strip())
        elif rec.startswith("COMPND"):
            compound.append(line[10:80].strip())
        elif rec.startswith("REMARK") and line[7:10] == "  2" and "RESOLUTION." in line:
            m = re.search(r"([\d.]+)\s+ANGSTROM", line)
            resolution = float(m.group(1)) if m else None
        elif rec.startswith("ENDMDL"):
            break
        elif rec in ("ATOM  ", "HETATM"):
            alt = line[16]
            occ = float(line[54:60] or 1.0)
            key = (line[21], line[22:27], line[12:16])
            atom = Atom(int(line[6:11]), line[12:16].strip(), line[17:20].strip(), line[21],
                        int(line[22:26]), line[26].strip(),
                        np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                        (line[76:78].strip() or line[12:14].strip()).upper(), rec == "HETATM")
            if alt.strip():
                if key in best and best[key][0] >= occ:
                    continue
                if key in best:
                    atoms.remove(best[key][1])
                best[key] = (occ, atom)
            atoms.append(atom)
    return Structure(pdb_id, " ".join(title), " ".join(compound), resolution, atoms)


def heme_sites(s: Structure):
    """Every heme iron with its axial environment."""
    sites = []
    for fe in (a for a in s.atoms if a.resname in HEME_NAMES and a.name == "FE"):
        own = [a for a in s.atoms if a.resname == fe.resname and a.chain == fe.chain and a.resseq == fe.resseq]
        pyrrole = np.array([a.xyz for a in own if a.name in ("NA", "NB", "NC", "ND")])
        others = [a for a in s.atoms if a not in own and a.element != "H"]
        d = np.linalg.norm(np.array([a.xyz for a in others]) - fe.xyz, axis=1)
        order = np.argsort(d)
        near = [(others[i], float(d[i])) for i in order[:12]]
        protein = [(a, r) for a, r in near if not a.hetero or a.resname == "MSE"]
        waters = [(a, r) for a, r in near if a.resname == "HOH" and r < 2.9]
        # displacement of Fe from the mean plane of the four pyrrole nitrogens
        oop = None
        fe_np = None
        if len(pyrrole) == 4:
            c = pyrrole.mean(axis=0)
            _, _, vt = np.linalg.svd(pyrrole - c)
            oop = float(abs(np.dot(fe.xyz - c, vt[2])))
            fe_np = float(np.linalg.norm(pyrrole - fe.xyz, axis=1).mean())
        prox = protein[0] if protein else None
        sites.append({
            "chain": fe.chain, "heme_resseq": fe.resseq,
            "proximal_resname": prox[0].resname if prox else None,
            "proximal_atom": prox[0].name if prox else None,
            "proximal_resseq": prox[0].resseq if prox else None,
            "proximal_distance": prox[1] if prox else None,
            "distal_water_distance": waters[0][1] if waters else None,
            "fe_pyrrole_n_mean": fe_np,
            "fe_out_of_plane": oop,
            "fe_xyz": fe.xyz,
        })
    return sites


def uniprot_fasta(acc: str) -> tuple[str, str]:
    path = fetch(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", DATA / "uniprot" / f"{acc}.fasta")
    lines = path.read_text().splitlines()
    return lines[0], "".join(lines[1:])


def uniprot_tsv(query: str, fields: str, name: str) -> list[dict]:
    from urllib.parse import quote
    url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
           f"&query={quote(query)}&fields={fields}")
    path = fetch(url, DATA / "uniprot" / f"{name}.tsv")
    rows = path.read_text(encoding="utf-8").splitlines()
    head = rows[0].split("\t")
    return [dict(zip(head, r.split("\t"))) for r in rows[1:] if r.strip()]


def prosite_pattern(ac: str) -> tuple[str, str]:
    """PROSITE pattern from the PA lines of the entry, and its regex."""
    path = fetch(f"https://prosite.expasy.org/{ac}.txt", DATA / f"{ac}.txt")
    pa = "".join(l[5:].strip() for l in path.read_text().splitlines() if l.startswith("PA"))
    pa = pa.rstrip(".")
    rx = []
    for el in pa.split("-"):
        m = re.fullmatch(r"(.+?)(\((\d+)(,(\d+))?\))?", el)
        core, rep = m.group(1), m.group(2)
        core = core.replace("x", ".").replace("{", "[^").replace("}", "]")
        if core.startswith("<"):
            core = "^" + core[1:]
        if core.endswith(">"):
            core = core[:-1] + "$"
        if rep:
            lo, hi = m.group(3), m.group(5)
            core += "{%s,%s}" % (lo, hi) if hi else "{%s}" % lo
        rx.append(core)
    return pa, "".join(rx)
