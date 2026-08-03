r"""
EXP-B -- The derived weighting and the identified floor.

Targets `thm:weighting-derived`, `thm:floor-is-resolution`,
`thm:aa-resolution`, `rem:guarantee-slack`, `rem:floor-measured`.

Five independent claims, each checked against the residue table
`tab:aa` rather than against a re-derivation of it:

  B1. d_min = 0.0756, attained at Lys--Arg, with Ile--Leu next at
      0.0780. (The table caption previously named a different pair;
      this recomputes it from the tabulated coordinates.)

  B2. The guarantee of `thm:aa-resolution`: all twenty residues have
      distinct depth-k addresses whenever delta(k) < d_min, i.e. k >= 9.
      TIGHTNESS: the guarantee must FAIL to be provable at k = 8,
      i.e. delta(8) > d_min -- otherwise the bound is not the least one.

  B3. `rem:guarantee-slack`: the guarantee is conservative. The ACTUAL
      separation depth (all twenty addresses distinct) is k = 7, two
      levels below the guaranteed 9. A run reporting actual == guaranteed
      would contradict the remark.

  B4. `thm:floor-is-resolution`: with w = Lambda/||.||,
      min_e w = Lambda/sqrt3 and max_e w = Lambda/d_min, so the
      strongest/weakest contact ratio is sqrt3/d_min = 22.9,
      independent of Lambda.

  B5. `rem:floor-measured` -- Lambda cancels. Sweep Lambda over six
      orders of magnitude; every RATIO the theory predicts must be
      invariant to machine precision. This is what makes the theory
      parameter-free, so it is worth testing rather than asserting.

Also recorded (not a pass/fail): the reciprocal profile of
`rem:reciprocal` -- that w is a separation COST, so the closest pair is
the most expensive to separate. We verify the ordering is exactly
reversed between distance and weight, since that inversion is the step
that makes contact and similarity one measurement.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp_b_weighting_and_floor.json"

# tab:aa, transcribed from the manuscript (sorted by Sk there).
AA = {
    "I": (1.000, 0.636, 0.000), "V": (0.967, 0.482, 0.000),
    "L": (0.922, 0.636, 0.000), "F": (0.811, 0.774, 0.000),
    "C": (0.778, 0.319, 0.150), "M": (0.711, 0.629, 0.000),
    "A": (0.700, 0.164, 0.000), "G": (0.456, 0.000, 0.000),
    "T": (0.422, 0.372, 0.100), "S": (0.411, 0.166, 0.100),
    "W": (0.400, 1.000, 0.000), "Y": (0.356, 0.836, 0.150),
    "P": (0.322, 0.399, 0.000), "H": (0.144, 0.653, 0.500),
    "E": (0.111, 0.545, 1.000), "Q": (0.111, 0.558, 0.200),
    "D": (0.111, 0.383, 1.000), "N": (0.111, 0.381, 0.200),
    "K": (0.067, 0.641, 1.000), "R": (0.000, 0.676, 1.000),
}

SQRT3 = math.sqrt(3.0)


def delta(k: int) -> float:
    """Cell diagonal at depth k: sqrt3 * 3^{-floor(k/3)}."""
    return SQRT3 * 3.0 ** (-(k // 3))


def address(pt, k: int):
    """Round-robin ternary positional encoding to depth k.

    Digit i refines coordinate (i mod 3); after k digits, coordinate c
    has been refined floor(k/3) times (plus one more for the first
    k mod 3 coordinates)."""
    lo = [0.0, 0.0, 0.0]
    hi = [1.0, 1.0, 1.0]
    digits = []
    for i in range(k):
        c = i % 3
        span = (hi[c] - lo[c]) / 3.0
        # clamp guards the closed upper endpoint x = 1.0
        d = min(2, int((pt[c] - lo[c]) / span)) if span > 0 else 0
        digits.append(d)
        lo[c] = lo[c] + d * span
        hi[c] = lo[c] + span
    return tuple(digits)


# ----------------------------------------------------------------- B1

def b1_dmin():
    ds = sorted(
        (math.dist(a, b), x, y)
        for (x, a), (y, b) in itertools.combinations(AA.items(), 2)
    )
    d_min, x0, y0 = ds[0]
    d_2nd, x1, y1 = ds[1]
    return {
        "claim": "d_min = 0.0756 at Lys-Arg; Ile-Leu next at 0.0780",
        "d_min": d_min,
        "closest_pair": [x0, y0],
        "second_distance": d_2nd,
        "second_pair": [x1, y1],
        "closest_matches_LysArg": {x0, y0} == {"K", "R"},
        "second_matches_IleLeu": {x1, y1} == {"I", "L"},
        "d_min_rounds_to_0_0756": abs(d_min - 0.0756) < 5e-5,
        "eight_smallest": [{"pair": [x, y], "d": d} for d, x, y in ds[:8]],
        "passed": (
            {x0, y0} == {"K", "R"}
            and {x1, y1} == {"I", "L"}
            and abs(d_min - 0.0756) < 5e-5
        ),
    }


# ----------------------------------------------------------------- B2, B3

def b2_b3_resolution(d_min: float):
    # guaranteed depth: least k with delta(k) < d_min
    k_guar = next(k for k in range(1, 40) if delta(k) < d_min)
    # actual depth: least k at which all twenty addresses are distinct
    k_act = None
    per_depth = []
    for k in range(1, 16):
        addrs = {r: address(p, k) for r, p in AA.items()}
        n_dist = len(set(addrs.values()))
        per_depth.append({"k": k, "distinct_addresses": n_dist})
        if n_dist == len(AA) and k_act is None:
            k_act = k
    return {
        "claim": ("guaranteed depth k=9 from delta(k) < d_min, tight at "
                  "k=8; actual separation at k=7 (rem:guarantee-slack)"),
        "guaranteed_depth": k_guar,
        "delta_at_9": delta(9),
        "delta_at_8": delta(8),
        "tight_at_8": delta(8) > d_min,   # bound must FAIL one level down
        "actual_separation_depth": k_act,
        "slack_levels": (k_guar - k_act) if k_act is not None else None,
        "guarantee_is_conservative": k_act is not None and k_act < k_guar,
        "distinct_addresses_by_depth": per_depth,
        "passed": (
            k_guar == 9
            and delta(8) > d_min
            and k_act == 7
        ),
    }


# ----------------------------------------------------------------- B4, B5

def weights(lam: float):
    return {
        (x, y): lam / math.dist(a, b)
        for (x, a), (y, b) in itertools.combinations(AA.items(), 2)
    }


def b4_b5_floor(d_min: float):
    lambdas = [1e-3, 1e-2, 1.0, 7.5, 1e3, 1e6]
    rows, ratios = [], []
    for lam in lambdas:
        W = weights(lam)
        w_min, w_max = min(W.values()), max(W.values())
        ratio = w_max / w_min
        ratios.append(ratio)
        rows.append({
            "Lambda": lam,
            "min_weight": w_min,
            "max_weight": w_max,
            "min_matches_Lambda_over_sqrt3":
                abs(w_min - lam / SQRT3) / (lam / SQRT3) < 1e-12,
            "max_matches_Lambda_over_dmin":
                abs(w_max - lam / d_min) / (lam / d_min) < 1e-12,
            "ratio": ratio,
        })
    spread = max(ratios) - min(ratios)
    predicted = SQRT3 / d_min
    return {
        "claim": ("min w = Lambda/sqrt3, max w = Lambda/d_min, ratio "
                  "sqrt3/d_min = 22.9 independent of Lambda"),
        "predicted_ratio": predicted,
        "predicted_rounds_to_22_9": abs(predicted - 22.9) < 0.05,
        "Lambda_sweep": rows,
        "ratio_spread_over_sweep": spread,
        "Lambda_cancels": spread < 1e-9,
        "note": ("min/max here are over item-item edges only; the "
                 "bound flo >= Lambda/sqrt3 of thm:floor-is-resolution "
                 "is a bound on the residual, which is a SUM of at "
                 "least one such edge, hence no tighter than this."),
        "passed": (
            all(r["min_matches_Lambda_over_sqrt3"] for r in rows)
            and all(r["max_matches_Lambda_over_dmin"] for r in rows)
            and spread < 1e-9
            and abs(predicted - 22.9) < 0.05
        ),
    }


# ----------------------------------------------------------------- reciprocal

def reciprocal_profile():
    """rem:reciprocal -- w is a separation COST, so the distance order
    and the weight order must be exactly reversed."""
    pairs = [
        ((x, y), math.dist(a, b))
        for (x, a), (y, b) in itertools.combinations(AA.items(), 2)
    ]
    by_d = [p for p, _ in sorted(pairs, key=lambda t: t[1])]
    W = weights(1.0)
    by_w = [p for p, _ in sorted(
        ((p, W[p]) for p, _ in pairs), key=lambda t: -t[1])]
    return {
        "claim": "ordering by distance is exactly reversed by weight",
        "n_pairs": len(pairs),
        "orderings_are_reverse": by_d == by_w,
        "most_expensive_to_separate": list(by_w[0]),
        "cheapest_to_separate": list(by_w[-1]),
        "passed": by_d == by_w,
    }


def main() -> int:
    b1 = b1_dmin()
    d_min = b1["d_min"]
    b23 = b2_b3_resolution(d_min)
    b45 = b4_b5_floor(d_min)
    rec = reciprocal_profile()

    parts = {"B1_dmin": b1, "B2B3_resolution": b23,
             "B4B5_floor_and_Lambda": b45, "reciprocal_profile": rec}
    passed = all(p["passed"] for p in parts.values())

    payload = {
        "experiment": "EXP-B",
        "target": ("thm:weighting-derived / thm:floor-is-resolution / "
                   "thm:aa-resolution / rem:guarantee-slack / "
                   "rem:floor-measured / rem:reciprocal"),
        "source_of_coordinates": "tab:aa (transcribed, not re-derived)",
        "summary": {
            "parts": len(parts),
            "parts_passed": sum(1 for p in parts.values() if p["passed"]),
            "d_min": d_min,
            "guaranteed_depth": b23["guaranteed_depth"],
            "actual_depth": b23["actual_separation_depth"],
            "contact_ratio": b45["predicted_ratio"],
            "Lambda_cancels": b45["Lambda_cancels"],
            "passed": passed,
        },
        **parts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"[EXP-B] d_min = {d_min:.4f} at "
          f"{'-'.join(b1['closest_pair'])} "
          f"(next {'-'.join(b1['second_pair'])} at {b1['second_distance']:.4f})")
    print(f"[EXP-B] depth: guaranteed {b23['guaranteed_depth']}, "
          f"actual {b23['actual_separation_depth']}, "
          f"slack {b23['slack_levels']}; tight at k=8: {b23['tight_at_8']}")
    print(f"[EXP-B] contact ratio sqrt3/d_min = "
          f"{b45['predicted_ratio']:.2f}; Lambda cancels: "
          f"{b45['Lambda_cancels']} "
          f"(spread {b45['ratio_spread_over_sweep']:.2e})")
    print(f"[EXP-B] distance/weight orderings reversed: "
          f"{rec['orderings_are_reverse']}")
    for name, p in parts.items():
        if not p["passed"]:
            print(f"[EXP-B]   FAILING PART: {name}")
    print(f"[EXP-B] {'PASS' if passed else 'FAIL'} -> {OUT.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
