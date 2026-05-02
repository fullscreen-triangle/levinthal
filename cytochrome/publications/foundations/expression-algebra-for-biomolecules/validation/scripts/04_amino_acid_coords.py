"""
Validation 04: S-entropy coordinates for the 20 standard amino acids.

Verifies the canonical mapping (Paper 1, Sec. 4.2; categorical-protein-database
paper Table 1):

    S_k = (h - h_min) / (h_max - h_min)     (Kyte-Doolittle hydrophobicity)
    S_t = (v - v_min) / (v_max - v_min)     (van der Waals volume)
    S_e = (e - e_min) / (e_max - e_min)     (electrostatic / charge index)

Checks:
  - All 20 amino acids land strictly in [0, 1]^3
  - Pairwise S-entropy distances distinguish all 190 pairs at trit-depth 9
  - Hydrophobic family (V, I, L, F) clusters at high S_k
  - Charged family (D, E, K, R) clusters at extreme S_e
  - Small residues (G, A) cluster at low S_t

Outputs: results/04_amino_acid_coords.json
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

# Kyte-Doolittle hydropathy index (J. Mol. Biol. 157, 1982)
KYTE_DOOLITTLE = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Van der Waals volume in cubic angstroms (Bondi 1964 / Voss & Gerstein 2005)
VDW_VOLUME = {
    "G": 60.1,  "A": 88.6,  "S": 89.0,  "C": 108.5, "T": 116.1,
    "V": 140.0, "L": 166.7, "I": 166.7, "P": 112.7, "F": 189.9,
    "Y": 193.6, "W": 227.8, "D": 111.1, "E": 138.4, "N": 114.1,
    "Q": 143.8, "H": 153.2, "K": 168.6, "R": 173.4, "M": 162.9,
}

# Electrostatic / charge index in [0, 1] (positive abs of formal charge at pH 7,
# scaled so charged extremes are 1.0; per categorical-protein-database paper)
ELECTROSTATIC = {
    "A": 0.10, "R": 1.00, "N": 0.30, "D": 0.95, "C": 0.20,
    "Q": 0.30, "E": 0.95, "G": 0.05, "H": 0.55, "I": 0.10,
    "L": 0.10, "K": 1.00, "M": 0.15, "F": 0.10, "P": 0.10,
    "S": 0.20, "T": 0.20, "W": 0.15, "Y": 0.30, "V": 0.10,
}


AMINO_ACIDS = list(KYTE_DOOLITTLE.keys())


def normalize(values: dict[str, float]) -> dict[str, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    span = vmax - vmin
    return {k: (v - vmin) / span for k, v in values.items()}


def s_coord(aa: str, sk: dict, st: dict, se: dict) -> tuple[float, float, float]:
    return (sk[aa], st[aa], se[aa])


def euclidean(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def trit_address(p: tuple[float, float, float], depth: int) -> str:
    """Interleaved ternary expansion (categorical-protein-database, Def 2.4)."""
    r = list(p)
    out = []
    for j in range(depth):
        axis = j % 3
        digit = int(r[axis] * 3)
        digit = max(0, min(2, digit))
        out.append(str(digit))
        r[axis] = r[axis] * 3 - digit
    return "".join(out)


def main() -> dict:
    sk = normalize(KYTE_DOOLITTLE)
    st = normalize(VDW_VOLUME)
    se = ELECTROSTATIC  # already in [0, 1]

    coords = {aa: s_coord(aa, sk, st, se) for aa in AMINO_ACIDS}

    # Check 1: all coordinates in [0, 1]^3
    in_unit_cube = all(
        0.0 <= c[i] <= 1.0
        for c in coords.values()
        for i in range(3)
    )

    # Check 2: pairwise distinguishability at depth 9
    addresses = {aa: trit_address(coords[aa], 9) for aa in AMINO_ACIDS}
    unique_at_depth_9 = len(set(addresses.values())) == len(AMINO_ACIDS)

    # Check 3: minimum pairwise euclidean distance
    min_dist = math.inf
    min_pair = None
    for a, b in itertools.combinations(AMINO_ACIDS, 2):
        d = euclidean(coords[a], coords[b])
        if d < min_dist:
            min_dist = d
            min_pair = (a, b)

    # Check 4: hydrophobic family clusters at high S_k
    hydrophobic = ["V", "I", "L", "F", "M"]
    sk_hydrophobic_mean = sum(sk[aa] for aa in hydrophobic) / len(hydrophobic)
    hydrophobic_high = sk_hydrophobic_mean > 0.7

    # Check 5: charged family clusters at extreme S_e
    charged = ["D", "E", "K", "R"]
    se_charged_mean = sum(se[aa] for aa in charged) / len(charged)
    charged_extreme = se_charged_mean > 0.9

    # Check 6: small residues at low S_t
    small = ["G", "A", "S"]
    st_small_mean = sum(st[aa] for aa in small) / len(small)
    small_low = st_small_mean < 0.3

    # Check 7: depth-3 chemical family clustering
    depth_3 = {aa: trit_address(coords[aa], 3) for aa in AMINO_ACIDS}
    family_table = {}
    for aa, addr in depth_3.items():
        family_table.setdefault(addr, []).append(aa)

    coord_table = [
        {
            "aa": aa,
            "Sk": round(coords[aa][0], 4),
            "St": round(coords[aa][1], 4),
            "Se": round(coords[aa][2], 4),
            "trit_d9": addresses[aa],
        }
        for aa in AMINO_ACIDS
    ]

    checks = {
        "all_in_unit_cube": in_unit_cube,
        "unique_at_depth_9": unique_at_depth_9,
        "hydrophobic_at_high_Sk": hydrophobic_high,
        "charged_at_extreme_Se": charged_extreme,
        "small_at_low_St": small_low,
    }

    result = {
        "validation_id": "04_amino_acid_coords",
        "paper_reference": "Paper 1, Sec. 4.2; categorical-protein-database Sec. 5",
        "coordinates": coord_table,
        "min_pairwise_distance": {
            "value": min_dist,
            "pair": list(min_pair) if min_pair else None,
        },
        "depth_3_family_clusters": {
            addr: aas for addr, aas in sorted(family_table.items())
        },
        "family_means": {
            "hydrophobic_Sk_mean": sk_hydrophobic_mean,
            "charged_Se_mean": se_charged_mean,
            "small_St_mean": st_small_mean,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "04_amino_acid_coords.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] amino acid S-coords")
    print(f"  unique at depth-9: {out['checks']['unique_at_depth_9']}")
    print(f"  min pairwise dist: {out['min_pairwise_distance']['value']:.4f}")
    print(f"  -> wrote {out_path}")
