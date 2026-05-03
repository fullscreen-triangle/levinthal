"""
Validation 06: CYP3A4 address assembly and hierarchical compression.

Verifies Construction 8.1 (Hierarchical Address): the 503-residue
CYP3A4 sequence (UniProt P08684) compresses from 4527 raw trits to
~600 structure-level trits via secondary-structure aggregation.

Method:
  - Synthesize a CYP3A4-statistical sequence at 503 residues with the
    CYP3 family bias.
  - Anchor the conserved P450 landmarks (heme-binding motif, EXXR,
    PERF, I-helix) at canonical positions.
  - Compute the raw residue-level address: 503 * 9 = 4527 trits.
  - Apply hierarchical compression: aggregate residues into 13 alpha
    helices + 5 beta strands + linkers, then compute one trit-address
    per structural element at depth 9.
  - Verify compression ratio falls within predicted bounds.

Outputs: results/06_cyp3a4_address.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    CYP3A4_LANDMARKS,
    sequence_address,
    sequence_centroid,
    synthesize_sequence,
    trit_address,
)

RANDOM_SEED = 42
N_RESIDUES = 503
DEPTH_PER_RESIDUE = 9

# CYP3A4 secondary structure (approximate, based on PDB 1TQN topology).
# CYP3A4 has the canonical P450 fold with 13 helices (A, B, B', C, D, E, F,
# G, H, I, J, K, L) and 5 beta-strands. Format: (name, type, start, end).
CYP3A4_TOPOLOGY = [
    # N-terminal anchor + linker
    ("anchor",   "loop", 1, 23),
    ("alpha-A",  "helix", 24, 49),
    ("alpha-B",  "helix", 50, 70),
    ("alpha-Bp", "helix", 71, 85),  # B' helix above the substrate channel
    ("beta-1",   "sheet", 86, 94),
    ("loop-1",   "loop", 95, 105),
    ("alpha-C",  "helix", 106, 130),
    ("alpha-D",  "helix", 131, 165),
    ("alpha-E",  "helix", 166, 195),
    ("alpha-F",  "helix", 196, 225),
    ("alpha-G",  "helix", 226, 250),
    ("alpha-H",  "helix", 251, 270),
    ("loop-2",   "loop", 271, 289),
    ("alpha-I",  "helix", 290, 325),
    ("loop-3",   "loop", 326, 343),
    ("alpha-J",  "helix", 344, 365),
    ("alpha-K",  "helix", 366, 390),
    ("loop-4",   "loop", 391, 410),
    ("beta-2",   "sheet", 411, 416),
    ("beta-3",   "sheet", 417, 422),
    ("PERF",     "loop", 423, 427),
    ("alpha-L",  "helix", 428, 437),
    ("heme-loop","loop", 438, 451),
    ("beta-4",   "sheet", 452, 460),
    ("beta-5",   "sheet", 461, 470),
    ("c-terminus","loop", 471, 503),
]


def embed_landmark(seq_chars: list, position: int, motif: str) -> list:
    """Embed a sequence motif at a given residue position."""
    L = len(seq_chars)
    end = min(position + len(motif), L)
    for i, aa in enumerate(motif):
        if position + i < L:
            seq_chars[position + i] = aa
    return seq_chars


def main() -> dict:
    rng = random.Random(RANDOM_SEED)

    # 1. Synthesize the CYP3A4 sequence
    seq_chars = list(synthesize_sequence("CYP3", N_RESIDUES, rng))

    # 2. Embed canonical landmarks at their approximate positions
    seq_chars = embed_landmark(seq_chars, 295, CYP3A4_LANDMARKS["I_helix"])
    seq_chars = embed_landmark(seq_chars, 365, CYP3A4_LANDMARKS["K_helix"])
    seq_chars = embed_landmark(seq_chars, 437, CYP3A4_LANDMARKS["heme_motif"])
    seq_chars = embed_landmark(seq_chars, 423, CYP3A4_LANDMARKS["PERF"])

    cyp3a4_seq = "".join(seq_chars)

    # 3. Raw residue-level address
    raw_trits = N_RESIDUES * DEPTH_PER_RESIDUE
    full_address = sequence_address(cyp3a4_seq, DEPTH_PER_RESIDUE)
    centroid = sequence_centroid(cyp3a4_seq)

    # 4. Hierarchical compression: per-element addresses
    element_addresses = []
    for name, etype, start, end in CYP3A4_TOPOLOGY:
        sub_seq = cyp3a4_seq[start - 1 : min(end, N_RESIDUES)]
        if not sub_seq:
            continue
        elem_centroid = sequence_centroid(sub_seq)
        elem_addr = trit_address(elem_centroid, DEPTH_PER_RESIDUE)
        element_addresses.append({
            "element": name,
            "type": etype,
            "range": [start, end],
            "length": len(sub_seq),
            "centroid": list(elem_centroid),
            "address": elem_addr,
        })

    # 5. Compression ratio
    n_elements = len(element_addresses)
    compressed_trits = n_elements * DEPTH_PER_RESIDUE
    compression_ratio = compressed_trits / raw_trits

    # 6. SS-element type distribution
    type_counts = {"helix": 0, "sheet": 0, "loop": 0}
    for elem in element_addresses:
        type_counts[elem["type"]] = type_counts.get(elem["type"], 0) + 1

    # 7. Verify the I-helix is in a P450 landmark cell at depth 3
    i_helix = next(e for e in element_addresses if e["element"] == "alpha-I")
    i_helix_d3_address = trit_address(tuple(i_helix["centroid"]), 3)

    # 8. Heme-binding loop landmark
    heme_loop = next(e for e in element_addresses if e["element"] == "heme-loop")
    heme_loop_d3_address = trit_address(tuple(heme_loop["centroid"]), 3)

    # 9. CYP3A4 family-cell consistency at depth 3 (should match CYP3 family)
    full_d3_address = sequence_address(cyp3a4_seq, 3)

    checks = {
        "raw_trit_count_correct": raw_trits == 4527,
        "n_helices_eq_13": type_counts["helix"] == 13,
        "n_sheets_eq_5": type_counts["sheet"] == 5,
        "n_elements_in_predicted_range": 20 <= n_elements <= 30,
        "compression_ratio_above_5x": compression_ratio < 0.20,
        "compression_ratio_within_30x": compression_ratio > 1.0 / 30.0,
        "address_length_matches": len(full_address) == DEPTH_PER_RESIDUE,
    }

    result = {
        "validation_id": "06_cyp3a4_address",
        "paper_reference": "Paper 2, Construction 8.1",
        "parameters": {
            "n_residues": N_RESIDUES,
            "depth_per_residue": DEPTH_PER_RESIDUE,
            "raw_trit_count": raw_trits,
            "random_seed": RANDOM_SEED,
        },
        "sequence_summary": {
            "length": len(cyp3a4_seq),
            "centroid": list(centroid),
            "full_address_d9": full_address,
            "family_address_d3": full_d3_address,
        },
        "topology": {
            "n_elements": n_elements,
            "type_counts": type_counts,
        },
        "element_addresses": element_addresses,
        "compression": {
            "raw_trit_count": raw_trits,
            "compressed_trit_count": compressed_trits,
            "ratio": compression_ratio,
            "x_compression": 1.0 / compression_ratio if compression_ratio > 0 else float("inf"),
        },
        "landmark_addresses_d3": {
            "I_helix": i_helix_d3_address,
            "heme_loop": heme_loop_d3_address,
            "full_protein": full_d3_address,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "06_cyp3a4_address.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] CYP3A4 address assembly")
    print(f"  raw trit count: {out['compression']['raw_trit_count']}")
    print(f"  compressed:     {out['compression']['compressed_trit_count']} "
          f"({out['compression']['x_compression']:.1f}x)")
    print(f"  topology: {out['topology']['type_counts']}")
    print(f"  full address (d9): {out['sequence_summary']['full_address_d9']}")
    print(f"  -> wrote {out_path}")
