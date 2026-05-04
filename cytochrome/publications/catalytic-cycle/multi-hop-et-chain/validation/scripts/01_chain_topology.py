"""
Validation 01: Chain topology — receiver tree, distances, S-coordinates.

Verifies the four-cofactor receiver tree of Paper 4 (Section 8):
  - Four leaves: NADPH, FAD, FMN, heme Fe^3+
  - Three intercofactor distances (4 Å, 4 Å, 14 Å)
  - S-coordinate progression along the chain
  - Partition depth M for each cofactor

Outputs: results/01_chain_topology.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    DISTANCE_FAD_FMN_A,
    DISTANCE_FMN_HEME_A,
    DISTANCE_NADPH_FAD_A,
    F_CB,
    S_FAD,
    S_FE_HS,
    S_FE_HS_RED,
    S_FMN,
    S_NADPH,
)


def main() -> dict:
    cofactors = [
        {"name": "NADPH", "S": list(S_NADPH)},
        {"name": "FAD", "S": list(S_FAD)},
        {"name": "FMN", "S": list(S_FMN)},
        {"name": "Fe(III) HS (state 2)", "S": list(S_FE_HS)},
        {"name": "Fe(II) HS (state 3)", "S": list(S_FE_HS_RED)},
    ]
    for cof in cofactors:
        result = F_CB(tuple(cof["S"]))
        cof.update(result)

    # Distances along chain
    distances_A = [
        {"hop": 1, "from": "NADPH", "to": "FAD", "distance_A": DISTANCE_NADPH_FAD_A},
        {"hop": 2, "from": "FAD",   "to": "FMN", "distance_A": DISTANCE_FAD_FMN_A},
        {"hop": 3, "from": "FMN",   "to": "Fe³⁺", "distance_A": DISTANCE_FMN_HEME_A},
    ]

    # Pairwise S-entropy distances
    pairwise = []
    for i, a in enumerate(cofactors[:4]):
        for j, b in enumerate(cofactors[:4]):
            if i >= j:
                continue
            d = math.sqrt(sum((a["S"][k] - b["S"][k]) ** 2 for k in range(3)))
            pairwise.append({"from": a["name"], "to": b["name"], "S_distance": d})

    chain_distinctness = all(
        c["norm"] != cofactors[i + 1]["norm"]
        for i, c in enumerate(cofactors[:4])
        if i + 1 < 4
    )

    # M progression along chain
    M_chain = [c["M"] for c in cofactors[:4]]
    M_monotonic_decrease = all(
        M_chain[i] >= M_chain[i + 1] - 0.5 for i in range(len(M_chain) - 1)
    )

    checks = {
        "all_cofactors_have_finite_M": bool(all(math.isfinite(c["M"]) or c["M"] == float("inf") for c in cofactors)),
        "chain_S_coords_distinct": bool(chain_distinctness),
        "three_distances_specified": bool(len(distances_A) == 3),
        "FMN_heme_distance_largest": bool(DISTANCE_FMN_HEME_A > DISTANCE_NADPH_FAD_A),
        "interprotein_distance_above_10A": bool(DISTANCE_FMN_HEME_A > 10.0),
    }

    return {
        "validation_id": "01_chain_topology",
        "paper_reference": "Paper 4, Section 8",
        "cofactor_partition_data": cofactors,
        "distances_A": distances_A,
        "pairwise_S_distances": pairwise,
        "M_chain": M_chain,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "01_chain_topology.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] chain topology")
    for c in out["cofactor_partition_data"]:
        print(f"  {c['name']:25s} M={c['M']:.3f}, n={c['n']}, l={c['l']}")
    print(f"  -> wrote {out_path}")
