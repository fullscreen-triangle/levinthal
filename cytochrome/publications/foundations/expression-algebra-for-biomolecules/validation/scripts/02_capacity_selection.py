"""
Validation 02: Partition capacity formula and selection rules.

Verifies:
  - Theorem 2.6 (Paper 1, inherited from spectroscopic-derivation):
    C(n) = 2 * sum_{l=0..n-1}(2l+1) = 2n^2

  - Equation (1) selection rules:
    Delta_l = +-1, |Delta_m| <= 1, Delta_s_orbital = 0

  - Capacity matches electron shell occupancies (K=2, L=8, M=18, N=32, O=50)

Outputs: results/02_capacity_selection.json
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


def capacity_direct(n: int) -> int:
    """Compute C(n) by direct enumeration of (l, m, s) cells."""
    count = 0
    for l in range(0, n):
        for _m in range(-l, l + 1):
            for _s in (-1, 1):  # 2 spin states
                count += 1
    return count


def capacity_formula(n: int) -> int:
    """C(n) = 2n^2."""
    return 2 * n * n


def is_allowed_transition(
    state1: tuple[int, int, int, int],
    state2: tuple[int, int, int, int],
) -> bool:
    """Check Delta_l = +-1, |Delta_m| <= 1, Delta_s_orbital = 0."""
    n1, l1, m1, s1 = state1
    n2, l2, m2, s2 = state2
    return (
        abs(l2 - l1) == 1
        and abs(m2 - m1) <= 1
        and s2 == s1
        and -l1 <= m1 <= l1
        and -l2 <= m2 <= l2
    )


def main() -> dict:
    # 1. Capacity formula consistency
    capacity_table = []
    n_max = 7
    for n in range(1, n_max + 1):
        c_direct = capacity_direct(n)
        c_formula = capacity_formula(n)
        capacity_table.append({
            "n": n,
            "C_direct": c_direct,
            "C_formula_2n2": c_formula,
            "agree": c_direct == c_formula,
        })

    capacity_consistent = all(row["agree"] for row in capacity_table)

    # 2. Match against known electron shell occupancies
    shells = {1: 2, 2: 8, 3: 18, 4: 32, 5: 50, 6: 72, 7: 98}
    shell_match = all(
        capacity_formula(n) == shells[n] for n in shells
    )

    # 3. Selection rule sanity: enumerate transitions for n=1..3 and report
    # how many are allowed vs forbidden.
    all_states = []
    for n in range(1, 4):
        for l in range(0, n):
            for m in range(-l, l + 1):
                for s in (-1, 1):
                    all_states.append((n, l, m, s))

    allowed_count = 0
    forbidden_examples = []
    allowed_examples = []
    for s1, s2 in itertools.combinations(all_states, 2):
        ok = is_allowed_transition(s1, s2)
        if ok:
            allowed_count += 1
            if len(allowed_examples) < 5:
                allowed_examples.append({"from": s1, "to": s2})
        else:
            if len(forbidden_examples) < 5:
                forbidden_examples.append({"from": s1, "to": s2})

    total_pairs = len(all_states) * (len(all_states) - 1) // 2

    # 4. Verify Delta_l rule rejects same-l transitions explicitly
    same_l_rejected = not is_allowed_transition((2, 1, 0, 1), (2, 1, 1, 1))

    # 5. Verify Delta_l = 2 forbidden
    delta_l_2_rejected = not is_allowed_transition((3, 0, 0, 1), (3, 2, 0, 1))

    # 6. Verify spin-flip forbidden (orbital chirality conservation)
    spin_flip_rejected = not is_allowed_transition((2, 0, 0, 1), (2, 1, 0, -1))

    checks = {
        "capacity_formula_consistent_n1_to_7": capacity_consistent,
        "matches_electron_shells_K_to_Q": shell_match,
        "same_l_transition_rejected": same_l_rejected,
        "delta_l_eq_2_rejected": delta_l_2_rejected,
        "spin_flip_rejected": spin_flip_rejected,
    }

    result = {
        "validation_id": "02_capacity_selection",
        "paper_reference": "Paper 1, Theorem 2.6 and Eq. (1)",
        "capacity_table": capacity_table,
        "shell_match_table": [
            {"n": n, "C(n)": capacity_formula(n), "shell_occupancy": shells[n]}
            for n in sorted(shells)
        ],
        "transition_audit": {
            "total_state_pairs_n_le_3": total_pairs,
            "allowed_count": allowed_count,
            "forbidden_count": total_pairs - allowed_count,
            "allowed_examples": allowed_examples,
            "forbidden_examples": forbidden_examples,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "02_capacity_selection.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] capacity formula + selection rules")
    print(f"  C(1..5) = {[capacity_formula(n) for n in range(1, 6)]}")
    print(f"  shells  = [2, 8, 18, 32, 50]")
    print(f"  -> wrote {out_path}")
