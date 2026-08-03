r"""
EXP-A -- Theorem `thm:subtrie-cut`.

CLAIM. If the contact weighting factors through the trie,
    w({u,v}) = f(lcp(addr(u), addr(v))),
then for every item v the minimum v--medium cut is attained on a
SUBTRIE BLOCK B_d(v) = {u : addr_d(u) = addr_d(v)}, so

    str(v) = min_{0<=d<=k} w(cut(B_d(v)))                      (eq:str-over-depths)

which is a minimisation over k+1 nested candidates rather than over
2^{|V|-1} subsets.

TEST. Exhaustive enumeration of every admissible S (containing v,
omitting the medium) versus the nested chain. The quantity of interest
is the GAP

    gap(v) = min over chain  -  min over all subsets  >= 0.

gap == 0 for every item and every instance <=> theorem holds on that
instance. A single positive gap falsifies it.

WHAT MAKES THIS A TEST AND NOT A TAUTOLOGY. Two degenerate readings had
to be excluded, and both bit during development:

  1. S = V\{med} ("separate everything from solvent at once") cuts only
     the medium edges and is a subtrie block (d=0). If medium weights
     are small it is optimal for every v, and the gap is trivially 0
     regardless of the weighting. We therefore also record results with
     the medium scaled so this cut does NOT dominate.
  2. Singletons {v} are subtrie blocks (d=k). If medium weights are
     large, {v} is optimal for every v -- again trivially 0.

So the interesting regime is intermediate medium weight, and the
reported `nontrivial_fraction` is the share of items whose optimal
depth d* is strictly between 0 and k. A run in which that fraction is
0 proves nothing and is flagged.

CONTROL. The same enumeration with weights NOT factoring through the
trie (drawn i.i.d. per pair) must produce a strictly positive gap,
otherwise the hypothesis of the theorem is vacuous.

The theorem does NOT assume f is monotone -- see `rem:factoring-not-monotone`.
We therefore sweep increasing, decreasing, non-monotone and constant f.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp_a_subtrie_cut.json"

MED = "MED"


# ----------------------------------------------------------------- graph

def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(k: int, b: int, f, med_w):
    """Items are all b^k addresses of depth k; MED joins every item."""
    items = list(itertools.product(range(b), repeat=k))
    W = {}
    for i, u in enumerate(items):
        for v in items[i + 1:]:
            W[(u, v)] = f(lcp(u, v))
        W[(u, MED)] = med_w(u)
    return items, W


def cut_weight(W, S) -> float:
    return sum(w for (u, v), w in W.items() if (u in S) != (v in S))


def exhaustive_min(items, W, v, proper=False):
    """min over ALL S with v in S, MED not in S.

    `proper=True` additionally excludes S = V\\{MED}, the "separate
    everything from the medium at once" cut. That cut severs only the
    |V| medium edges, is itself a subtrie block (d=0), and for a wide
    range of medium weights it is optimal for EVERY item under EVERY
    weighting -- including weightings that do not factor through the
    trie. Leaving it in therefore drives the measured gap to zero
    unconditionally and destroys the test's ability to discriminate;
    the control below confirms this empirically."""
    others = [x for x in items if x != v]
    best, arg = float("inf"), None
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            S = frozenset((v,) + sub)
            if proper and len(S) == len(items):
                continue
            c = cut_weight(W, S)
            if c < best - 1e-12:
                best, arg = c, S
    return best, arg


def exhaustive_min_both(items, W, v):
    """One enumeration, two answers: the unrestricted minimum and the
    minimum with the whole-set cut S = V\\{MED} removed. Enumerating
    twice doubles a 2^(n-1) loop for no reason."""
    others = [x for x in items if x != v]
    n = len(items)
    best, arg = float("inf"), None
    best_p, arg_p = float("inf"), None
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            S = frozenset((v,) + sub)
            c = cut_weight(W, S)
            if c < best - 1e-12:
                best, arg = c, S
            if len(S) != n and c < best_p - 1e-12:
                best_p, arg_p = c, S
    return (best, arg), (best_p, arg_p)


def chain_min(items, W, v, k, proper=False):
    """min over the k+1 nested subtrie blocks B_0(v) >= ... >= B_k(v)."""
    best, arg_d = float("inf"), None
    for d in range(k + 1):
        S = frozenset(x for x in items if x[:d] == v[:d])
        if proper and len(S) == len(items):
            continue
        c = cut_weight(W, S)
        if c < best - 1e-12:
            best, arg_d = c, d
    return best, arg_d


# ----------------------------------------------------------------- profiles

def profiles(k: int, rng: random.Random):
    """f : {0..k} -> (0,inf).  Monotonicity is deliberately NOT assumed."""
    return {
        "increasing_3^l": lambda l: 3.0 ** l,
        "increasing_1+3l": lambda l: 1.0 + 3.0 * l,
        "decreasing_3^-l": lambda l: 3.0 ** (-l),
        "nonmonotone": (lambda tab: (lambda l: tab[min(l, len(tab) - 1)]))(
            [1.0, 9.0, 1.0, 4.0, 2.0][: k + 1]),
        "constant": lambda l: 1.0,
        "random_f": (lambda tab: (lambda l: tab[min(l, len(tab) - 1)]))(
            [rng.uniform(0.05, 20.0) for _ in range(k + 1)]),
    }


# ----------------------------------------------------------------- runs

def typical_internal(k, b, f):
    """Mean item-item weight, used to scale the medium so that the
    whole-set cut and the singleton cut actually compete.

    Without this the medium range is an absolute number while the
    item-item weights span orders of magnitude across profiles, so every
    case collapses to d*=0 or d*=k and the test is vacuous."""
    items = list(itertools.product(range(b), repeat=k))
    vals = [f(lcp(u, v))
            for i, u in enumerate(items) for v in items[i + 1:]]
    return sum(vals) / len(vals)


def run_case(k, b, profile_name, f, med_lo, med_hi, rng):
    # medium weights are expressed as MULTIPLES of the typical internal
    # weight, times the number of items each cut must sever, so that the
    # regimes are comparable across profiles.
    scale = typical_internal(k, b, f) * (b ** k) / 4.0
    med_w = {}

    def m(u):
        if u not in med_w:
            med_w[u] = rng.uniform(med_lo, med_hi) * scale
        return med_w[u]

    items, W = build(k, b, f, m)
    worst_gap = 0.0
    worst_gap_proper = 0.0
    depths, depths_proper = [], []
    witness = None
    for v in items:
        (ex, ex_S), (exp_, exp_S) = exhaustive_min_both(items, W, v)
        ch, d = chain_min(items, W, v, k)
        depths.append(d)
        gap = ch - ex
        if gap > worst_gap + 1e-12:
            worst_gap = gap
            witness = {"item": list(v), "optimal_S": sorted(list(x) for x in ex_S),
                       "exhaustive": ex, "chain": ch}

        # same comparison with the degenerate whole-set cut removed from
        # BOTH sides -- this is the reading with content
        chp, dp = chain_min(items, W, v, k, proper=True)
        depths_proper.append(dp)
        gp = chp - exp_
        if gp > worst_gap_proper + 1e-12:
            worst_gap_proper = gp
            witness = {"item": list(v), "optimal_S": sorted(list(x) for x in exp_S),
                       "exhaustive": exp_, "chain": chp, "proper": True}
    n = len(items)
    nontrivial = sum(1 for d in depths_proper if 0 < d < k)
    return {
        "k": k, "b": b, "n_items": n,
        "profile": profile_name,
        "medium_range": [med_lo, med_hi],
        "worst_gap": worst_gap,
        "worst_gap_excluding_whole_set": worst_gap_proper,
        "theorem_holds": max(worst_gap, worst_gap_proper) <= 1e-9,
        "optimal_depths": depths,
        "optimal_depths_excluding_whole_set": depths_proper,
        "nontrivial_fraction": nontrivial / n,
        "degenerate": nontrivial == 0,
        "witness": witness,
    }


def run_control(k, b, rng, trials):
    """Weights NOT a function of lcp. The gap MUST become positive."""
    items = list(itertools.product(range(b), repeat=k))
    n = len(items)
    worst, witness = 0.0, None
    positive_trials = 0
    for t in range(trials):
        W = {}
        # medium scaled the same way as in run_case, so the control sits
        # in the informative regime rather than at a degenerate optimum
        med_scale = 5.05 * n / 4.0
        for i, u in enumerate(items):
            for v in items[i + 1:]:
                W[(u, v)] = rng.uniform(0.1, 10.0)
            W[(u, MED)] = rng.uniform(0.05, 0.4) * med_scale
        trial_gap = 0.0
        for v in items:
            ex, ex_S = exhaustive_min(items, W, v, proper=True)
            ch, _ = chain_min(items, W, v, k, proper=True)
            trial_gap = max(trial_gap, ch - ex)
            if ch - ex > worst + 1e-12:
                worst = ch - ex
                witness = {"trial": t, "item": list(v),
                           "optimal_S": sorted(list(x) for x in ex_S),
                           "exhaustive": ex, "chain": ch}
        if trial_gap > 1e-9:
            positive_trials += 1
    return {
        "k": k, "b": b, "trials": trials,
        "whole_set_cut_excluded": True,
        "worst_gap": worst,
        "trials_with_positive_gap": positive_trials,
        "hypothesis_is_nonvacuous": worst > 1e-9,
        "witness": witness,
    }


def main() -> int:
    rng = random.Random(20260803)
    cases, controls = [], []

    # medium ranges chosen to span the three regimes:
    #   tiny   -> d*=0 dominates (whole-set cut)
    #   mid    -> intermediate depths appear  <-- the informative regime
    #   large  -> d*=k dominates (singleton cut)
    # expressed as multiples of the typical internal weight (see run_case)
    med_ranges = [(0.005, 0.05), (0.05, 0.4), (0.2, 1.5),
                  (0.8, 4.0), (3.0, 20.0)]

    # (4,2) is 16 items => 2^15 admissible subsets per item per case.
    # Exhaustive enumeration there is affordable only for a subset of the
    # profiles; we keep one increasing, one decreasing and one
    # non-monotone f, which is what the "monotonicity is irrelevant"
    # claim actually needs. The full six-profile sweep runs at (3,2) and
    # (2,3). This restriction is a compute bound, not a selection of
    # favourable cases.
    subset_at_16 = ("increasing_3^l", "decreasing_3^-l", "nonmonotone")

    for (k, b) in [(3, 2), (2, 3), (4, 2)]:
        for pname, f in profiles(k, rng).items():
            if (k, b) == (4, 2) and pname not in subset_at_16:
                continue
            for lo, hi in med_ranges:
                cases.append(run_case(k, b, pname, f, lo, hi, rng))

    for (k, b) in [(3, 2), (2, 3)]:
        controls.append(run_control(k, b, rng, trials=25))

    n_cases = len(cases)
    n_hold = sum(1 for c in cases if c["theorem_holds"])
    informative = [c for c in cases if not c["degenerate"]]
    n_inf_hold = sum(1 for c in informative if c["theorem_holds"])
    worst = max(max(c["worst_gap"], c["worst_gap_excluding_whole_set"])
                for c in cases)

    controls_ok = all(c["hypothesis_is_nonvacuous"] for c in controls)

    passed = (n_hold == n_cases) and controls_ok and len(informative) > 0

    payload = {
        "experiment": "EXP-A",
        "target": "thm:subtrie-cut / eq:str-over-depths",
        "claim": ("min v-med cut is attained on a subtrie block when the "
                  "weighting factors through the trie"),
        "method": ("exhaustive enumeration of all 2^(n-1) admissible cuts "
                   "vs the k+1 nested subtrie blocks"),
        "summary": {
            "cases": n_cases,
            "cases_theorem_holds": n_hold,
            "informative_cases": len(informative),
            "informative_cases_theorem_holds": n_inf_hold,
            "worst_gap_over_all_cases": worst,
            "monotonicity_assumed": False,
            "control_gap_positive": controls_ok,
            "passed": passed,
        },
        "cases": cases,
        "controls": controls,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"[EXP-A] {n_hold}/{n_cases} cases hold; "
          f"{len(informative)} informative (nontrivial d*); "
          f"worst gap {worst:.2e}")
    print(f"[EXP-A] control (non-factoring weights) gap positive: {controls_ok}")
    for c in controls:
        print(f"         b={c['b']} k={c['k']}: worst control gap "
              f"{c['worst_gap']:.4f} in {c['trials_with_positive_gap']}"
              f"/{c['trials']} trials")
    print(f"[EXP-A] {'PASS' if passed else 'FAIL'} -> {OUT.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
