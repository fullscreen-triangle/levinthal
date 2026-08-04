r"""
EXP-C -- Graceful degradation under non-factoring edges.

Targets `cor:sepcost-cheap` and `rem:complexity-honest`.

`thm:subtrie-cut` (EXP-A) assumes the weighting factors through the
trie. Real contact graphs do not: a disulfide bond couples residues that
may be far apart in S-entropy space, so the graph carries edges
violating eq:trie-factoring. `rem:complexity-honest` claims the theorem
then DEGRADES GRACEFULLY rather than failing, and makes two specific,
falsifiable claims:

  C1 (soundness). The trie chain still furnishes an UPPER BOUND:

          chain_min(v)  >=  true_min(v)          for every v.

      This is the claim that matters operationally -- an upper bound on
      separation cost is still usable; a quantity of unknown sign is not.
      Trivially true by construction (the chain minimises over a subset
      of the admissible cuts), so it is asserted here mainly as a guard
      against implementation error, and it would catch any bug that let
      the chain consider an inadmissible S.

  C2 (the quantitative claim). The gap is bounded by the total weight of
      the non-factoring edges:

          chain_min(v) - true_min(v)  <=  sum of perturbation weights.

      This is NOT trivial and is the real content of the remark. We
      perturb a factoring graph by adding m "disulfide" edges of total
      weight P and check the bound holds, then report how TIGHT it is.

  C3 (the complexity claim of cor:sepcost-cheap). The chain evaluates
      k+1 candidates versus 2^(n-1) admissible subsets. We count both
      directly rather than quoting the asymptotics, and confirm the
      chain's answer equals the exhaustive answer on the unperturbed
      graph (which is EXP-A's result, re-checked here as a control on
      this harness).

DESIGN NOTE. As in EXP-A, the degenerate whole-set cut S = V\{MED} is
excluded from both sides: it is optimal for every v under a wide range
of medium weights regardless of the weighting, which would drive every
gap to zero and make the degradation claim untestable.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp_c_degradation.json"

MED = "MED"


def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cut_weight(W, S) -> float:
    return sum(w for (u, v), w in W.items() if (u in S) != (v in S))


def exhaustive_min(items, W, v):
    """Minimum over all admissible S, excluding the whole-set cut."""
    others = [x for x in items if x != v]
    n = len(items)
    best, arg = float("inf"), None
    n_eval = 0
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            S = frozenset((v,) + sub)
            if len(S) == n:
                continue
            n_eval += 1
            c = cut_weight(W, S)
            if c < best - 1e-12:
                best, arg = c, S
    return best, arg, n_eval


def chain_min(items, W, v, k):
    """Minimum over the nested subtrie blocks, excluding the whole set."""
    n = len(items)
    best, arg_d = float("inf"), None
    n_eval = 0
    for d in range(k + 1):
        S = frozenset(x for x in items if x[:d] == v[:d])
        if len(S) == n:
            continue
        n_eval += 1
        c = cut_weight(W, S)
        if c < best - 1e-12:
            best, arg_d = c, d
    return best, arg_d, n_eval


def build(k, b, f, med_w):
    items = list(itertools.product(range(b), repeat=k))
    W = {}
    for i, u in enumerate(items):
        for v in items[i + 1:]:
            W[(u, v)] = f(lcp(u, v))
        W[(u, MED)] = med_w(u)
    return items, W


def perturb(items, W, rng, m, w_lo, w_hi):
    """Add m non-factoring ("disulfide") couplings.

    We ADD weight to m randomly chosen item-item pairs. Adding rather
    than replacing keeps the perturbation's total weight P unambiguous:
    P is exactly the weight that does not factor through the trie, which
    is the quantity rem:complexity-honest's bound is stated in terms of."""
    Wp = dict(W)
    pairs = [(u, v) for i, u in enumerate(items) for v in items[i + 1:]]
    chosen = rng.sample(pairs, min(m, len(pairs)))
    P = 0.0
    for p in chosen:
        dw = rng.uniform(w_lo, w_hi)
        Wp[p] = Wp[p] + dw
        P += dw
    return Wp, P, chosen


def run_trial(k, b, f, rng, m, w_lo, w_hi):
    n = b ** k
    scale = 5.0 * n / 4.0
    mw = {}

    def med(u):
        if u not in mw:
            mw[u] = rng.uniform(0.05, 0.4) * scale
        return mw[u]

    items, W = build(k, b, f, med)
    Wp, P, chosen = perturb(items, W, rng, m, w_lo, w_hi)

    worst_gap = 0.0
    sound = True
    exact = 0
    ev_chain = ev_exh = 0
    for v in items:
        ex, _, ne = exhaustive_min(items, Wp, v)
        ch, _, nc = chain_min(items, Wp, v, k)
        ev_exh, ev_chain = ne, nc
        if ch < ex - 1e-9:
            sound = False          # chain below true min => impossible
        gap = ch - ex
        if gap <= 1e-9:
            exact += 1
        worst_gap = max(worst_gap, gap)

    return {
        "n_items": n,
        "n_perturbed_edges": len(chosen),
        "perturbation_total_weight": P,
        "worst_gap": worst_gap,
        "chain_is_upper_bound": sound,
        "gap_within_perturbation_bound": worst_gap <= P + 1e-9,
        "tightness": (worst_gap / P) if P > 0 else 0.0,
        "items_still_exact": exact,
        "fraction_still_exact": exact / n,
        "cut_evals_chain": ev_chain,
        "cut_evals_exhaustive": ev_exh,
    }


def main() -> int:
    rng = random.Random(20260804)
    f = lambda l: 3.0 ** l          # noqa: E731  (one of EXP-A's profiles)

    # Perturbation magnitudes are swept across two orders of magnitude.
    # A disulfide is not a small correction to a van der Waals contact,
    # so the large end is the physically relevant one -- and a bound
    # tested only where the perturbation never changes an optimum would
    # be evidence of insensitivity, not of the bound. `bit` below counts
    # how often it actually changes the answer.
    trials = []
    for (k, b) in [(3, 2), (2, 3)]:
        for m in [1, 2, 3, 5]:
            for lo, hi in [(0.5, 6.0), (5.0, 40.0), (30.0, 200.0)]:
                for _ in range(6):
                    trials.append({
                        "k": k, "b": b, "m_requested": m,
                        "perturbation_range": [lo, hi],
                        **run_trial(k, b, f, rng, m, lo, hi),
                    })

    # C3 control: with NO perturbation the chain must be exact (EXP-A),
    # re-checked here so a harness bug cannot masquerade as degradation.
    controls = []
    for (k, b) in [(3, 2), (2, 3)]:
        c = run_trial(k, b, f, rng, 0, 0.0, 0.0)
        c.update({"k": k, "b": b, "m_requested": 0})
        controls.append(c)

    c1 = all(t["chain_is_upper_bound"] for t in trials + controls)
    c2 = all(t["gap_within_perturbation_bound"] for t in trials)
    c3 = all(t["worst_gap"] <= 1e-9 for t in controls)

    # how often the perturbation actually bites -- a run where the gap is
    # always 0 would not be evidence for the bound, only for insensitivity
    bit = sum(1 for t in trials if t["worst_gap"] > 1e-9)
    tight = [t["tightness"] for t in trials if t["perturbation_total_weight"] > 0]

    speedup = [
        {"k": t["k"], "b": t["b"], "n_items": t["n_items"],
         "cut_evals_chain": t["cut_evals_chain"],
         "cut_evals_exhaustive": t["cut_evals_exhaustive"],
         "ratio": t["cut_evals_exhaustive"] / t["cut_evals_chain"]}
        for t in controls
    ]

    passed = c1 and c2 and c3 and bit > 0

    payload = {
        "experiment": "EXP-C",
        "target": "cor:sepcost-cheap / rem:complexity-honest",
        "claim": ("under non-factoring (disulfide-like) edges the trie "
                  "chain remains an upper bound on str(v), with gap "
                  "bounded by the total non-factoring weight"),
        "summary": {
            "trials": len(trials),
            "C1_chain_is_upper_bound": c1,
            "C2_gap_within_perturbation_bound": c2,
            "C3_exact_when_unperturbed": c3,
            "trials_where_perturbation_bit": bit,
            "mean_tightness_gap_over_P": (sum(tight) / len(tight)) if tight else 0.0,
            "max_tightness_gap_over_P": max(tight) if tight else 0.0,
            "cut_eval_counts": speedup,
            "passed": passed,
        },
        "trials": trials,
        "controls": controls,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"[EXP-C] {len(trials)} trials; perturbation bit in {bit}")
    print(f"[EXP-C] C1 chain is upper bound: {c1}")
    print(f"[EXP-C] C2 gap <= total non-factoring weight: {c2} "
          f"(mean gap/P {payload['summary']['mean_tightness_gap_over_P']:.3f}, "
          f"max {payload['summary']['max_tightness_gap_over_P']:.3f})")
    print(f"[EXP-C] C3 exact when unperturbed: {c3}")
    for s in speedup:
        print(f"         n={s['n_items']:3d}: {s['cut_evals_chain']} chain "
              f"evals vs {s['cut_evals_exhaustive']} exhaustive "
              f"({s['ratio']:.0f}x)")
    print(f"[EXP-C] {'PASS' if passed else 'FAIL'} -> {OUT.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
