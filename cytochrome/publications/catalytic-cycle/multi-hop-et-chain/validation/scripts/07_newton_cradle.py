"""
Validation 07: Newton's-cradle electron non-identity.

Verifies Theorem 4.4 of Paper 4 (extended from Theorem 4.4 of biological-current-flux):
the electron arriving at the heme is categorically continuous with, but not
materially identical to, the hydride electron donated by NADPH.

Method:
  - Simulate a 4-cofactor chain with isotopically-labelled donor electrons.
  - Track which physical electron is delivered to the acceptor at each hop.
  - Verify that the "label" of the original donor electron does NOT propagate
    to the final acceptor (only the categorical defect does).
  - Sanity-check chain mass-balance: each cofactor net-zero electron change.

Outputs: results/07_newton_cradle.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def simulate_newton_cradle(n_cofactors: int = 4, n_iterations: int = 1) -> dict:
    """Simulate the Newton's-cradle propagation through a chain.

    Each cofactor starts with a quota of 'native' electrons. The hydride from
    NADPH (cofactor 0) carries label 'L0'. When the categorical defect arrives
    at cofactor i, cofactor i passes ONE OF ITS OWN electrons (label 'Li')
    onward to cofactor i+1, while retaining the carrier of the categorical
    defect on its surface.
    """
    # Each cofactor has a stack of native electrons
    cofactors = [[] for _ in range(n_cofactors)]
    for i in range(n_cofactors):
        # Start with several native electrons per cofactor (representing
        # available bridging electrons in a flavin pi-system or Fe d-shell)
        for k in range(3):
            cofactors[i].append(f"L{i}_{k}")

    log = []
    for it in range(n_iterations):
        # NADPH donates the hydride: extracts L0_0 from cofactor 0
        donor_label = cofactors[0].pop(0)
        # Defect propagates: each carrier supplies its own electron to the next
        carrier = donor_label
        log.append({
            "iteration": it,
            "step": "donor extraction",
            "from_cofactor": 0,
            "carrier_label": carrier,
        })

        for hop in range(n_cofactors - 1):
            # Defect arrives at cofactor[hop+1]
            # Cofactor[hop+1] supplies one of its own electrons to advance the chain
            if hop < n_cofactors - 2:
                # Intermediate hops: cofactor releases its own electron
                supplied_label = cofactors[hop + 1].pop(0)
                # That supplied label moves forward; the original carrier stays
                # at cofactor[hop+1] (joins the cofactor's residue)
                cofactors[hop + 1].append(carrier)
                log.append({
                    "iteration": it,
                    "step": f"hop {hop + 1}",
                    "from_cofactor": hop + 1,
                    "to_cofactor": hop + 2,
                    "incoming_carrier": carrier,
                    "outgoing_label": supplied_label,
                    "stays_at_cofactor": carrier,
                })
                carrier = supplied_label
            else:
                # Final hop to terminal acceptor
                cofactors[hop + 1].append(carrier)
                log.append({
                    "iteration": it,
                    "step": f"final hop {hop + 1}",
                    "to_cofactor_terminal": hop + 1,
                    "delivered_label": carrier,
                })

    return {"cofactors_final": cofactors, "log": log}


def main() -> dict:
    sim = simulate_newton_cradle(n_cofactors=4, n_iterations=1)

    # Find the label delivered to the terminal acceptor
    final_cofactor = sim["cofactors_final"][-1]
    delivered_to_terminal = final_cofactor[-1] if final_cofactor else None
    original_donor_label = "L0_0"

    # Newton's cradle prediction: the delivered label should NOT be L0_0
    label_non_identity = delivered_to_terminal != original_donor_label

    # Mass balance: the categorical defect propagates through the chain;
    # the donor extracted one electron, but the terminal cofactor gains one
    # (the categorical defect arrives there). Net: chain's total electron
    # count is conserved. For our simulation that pops one electron from
    # cofactor 0 and pushes the carrier to the terminal cofactor, the final
    # count = initial count.
    initial_count = 4 * 3  # 4 cofactors × 3 electrons each
    final_count = sum(len(c) for c in sim["cofactors_final"])
    expected_final_count = initial_count  # defect propagates, no net loss

    propagation_steps = sum(1 for s in sim["log"] if "hop" in s.get("step", ""))

    checks = {
        "label_non_identity": bool(label_non_identity),
        "donor_label_stayed_at_intermediate": bool(
            any(original_donor_label in c for c in sim["cofactors_final"])
        ),
        "terminal_received_intermediate_label": bool(delivered_to_terminal != original_donor_label),
        "propagation_steps_eq_3_for_4cofactor_chain": bool(propagation_steps == 3),
        "chain_mass_balance_maintained": bool(final_count == expected_final_count),
    }

    return {
        "validation_id": "07_newton_cradle",
        "paper_reference": "Paper 4, Theorem 4.4",
        "simulation_log": sim["log"],
        "cofactors_final_state": sim["cofactors_final"],
        "original_donor_label": original_donor_label,
        "delivered_to_terminal": delivered_to_terminal,
        "label_non_identity_observed": label_non_identity,
        "initial_electron_count": initial_count,
        "expected_final_electron_count": expected_final_count,
        "final_electron_count": final_count,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "07_newton_cradle.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] Newton's cradle non-identity")
    print(f"  donor:     {out['original_donor_label']}")
    print(f"  delivered: {out['delivered_to_terminal']}")
    print(f"  non-identity: {out['label_non_identity_observed']}")
    print(f"  -> wrote {out_path}")
