"""
EXP-B --- Below the threshold, is spreading actually worse than committing?

WHY THIS RUNS SECOND
--------------------
EXP-A found that federated-join yield is convex below an exact effort
threshold e* = 2 ln2 / ln(1/(1-1/N)), and concluded that an allocator
must refuse to water-fill there. That conclusion rests on a claim that
EXP-A did not itself measure: that under increasing returns the optimal
allocation is a CORNER (all budget to one join) rather than an interior
split, so a water-filling agent does not merely lose optimality but
actively picks the worse action.

Jensen gives the direction for a single convex function. It does not
immediately give the magnitude, and it does not cover the mixed case
where some joins are past their inflection and others are not --- which
is the realistic case, since corpora differ in size. Both are measured
here.

WHAT WOULD REFUTE IT
--------------------
The refusal is unjustified if EITHER
  (a) spreading and committing give the same yield below e* (then
      water-filling is harmless and the gate is bureaucracy), OR
  (b) committing is WORSE than spreading below e* (then the refusal
      points the wrong way and the fix is not a gate but a different
      rule).
Both are checked as explicit negative controls, not assumed away.

Above e* the ordering must REVERSE --- spreading should win --- or
EXP-A's "T2 is recovered above the threshold" claim is empty. That
reversal is the real test here; a gate that never changes the answer is
not a gate.

Everything is evaluated on the closed form from EXP-A. No sampling, so
no result here is within noise of anything.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


# =====================================================================
#  The join yield surface (closed form, from EXP-A)
# =====================================================================


def yield_join_exact(effort: float, corpus_size: int) -> float:
    """E[|A cap B|] = N p^2, p = 1 - (1-1/N)^(e/2). Sigmoid in e."""
    if effort <= 0:
        return 0.0
    p = 1.0 - (1.0 - 1.0 / corpus_size) ** (effort / 2.0)
    return corpus_size * p * p


def join_inflection(corpus_size: int) -> float:
    """Effort at which the curve turns concave; yield there is N/4."""
    q = 1.0 - 1.0 / corpus_size
    return 2.0 * math.log(2.0) / math.log(1.0 / q)


# =====================================================================
#  The two policies
# =====================================================================


def spread(budget: float, corpora: list[int]) -> float:
    """Even split across all joins --- what water-filling degenerates to
    when the joins are identical, and the interior solution generally."""
    share = budget / len(corpora)
    return sum(yield_join_exact(share, n) for n in corpora)


def commit(budget: float, corpora: list[int]) -> float:
    """Whole budget to the single best join. The corner solution."""
    return max(yield_join_exact(budget, n) for n in corpora)


def best_split_2(budget: float, corpora: list[int], steps: int = 2000) -> tuple:
    """Brute-force optimum over two joins, to check that the true optimum
    really is at a corner below e* and interior above it --- rather than
    trusting that spread/commit bracket it."""
    a, b = corpora
    best_y, best_x = -1.0, 0.0
    for i in range(steps + 1):
        x = budget * i / steps
        y = yield_join_exact(x, a) + yield_join_exact(budget - x, b)
        if y > best_y:
            best_y, best_x = y, x
    return best_y, best_x


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append({"check": name, "verdict": "PASS" if passed else "FAIL",
                       "detail": detail, **extra})

    N = 40
    e_star = join_inflection(N)
    pair = [N, N]

    # -- B.1 below the threshold, committing strictly beats spreading ----
    below = [0.15, 0.3, 0.5, 0.75, 0.9]
    rows_below = []
    all_commit_wins = True
    for f in below:
        # budget such that the SPREAD share lands below e*
        budget = 2 * e_star * f
        s, c = spread(budget, pair), commit(budget, pair)
        rows_below.append({"budget": budget, "share": budget / 2,
                           "share_over_e_star": (budget / 2) / e_star,
                           "spread": s, "commit": c,
                           "commit_minus_spread": c - s,
                           "relative_gain": (c - s) / s if s > 0 else None})
        if not c > s:
            all_commit_wins = False
    worst = min(r["relative_gain"] for r in rows_below
                if r["relative_gain"] is not None)
    check(
        "below e*, committing the whole budget strictly beats spreading it",
        all_commit_wins,
        f"over shares {below[0]:.2f}-{below[-1]:.2f} of e*, the corner "
        f"solution beats the even split at every budget, by at least "
        f"{100 * worst:.1f}%. This is what makes the refusal load-bearing: "
        f"an unguarded water-filling agent does not merely lose a little "
        f"optimality below the threshold, it picks the worse of the two "
        f"available actions.",
        rows=rows_below,
    )

    # -- B.2 negative control: the ordering REVERSES above e* ------------
    # If it did not, the gate would be pointless --- committing would
    # always win and the right rule would be "never spread", not "spread
    # above the threshold".
    above = [1.5, 2.5, 4.0, 6.0]
    rows_above = []
    all_spread_wins = True
    for f in above:
        budget = 2 * e_star * f
        s, c = spread(budget, pair), commit(budget, pair)
        rows_above.append({"budget": budget, "share": budget / 2,
                           "share_over_e_star": (budget / 2) / e_star,
                           "spread": s, "commit": c,
                           "spread_minus_commit": s - c})
        if not s > c:
            all_spread_wins = False
    check(
        "negative control: above e* the ordering REVERSES and spreading wins",
        all_spread_wins,
        f"at shares {above[0]:.1f}-{above[-1]:.1f} of e* the even split "
        f"beats the corner at every budget. The gate therefore changes "
        f"the answer in both directions; a rule that always committed "
        f"would be wrong above the threshold, and one that always spread "
        f"is wrong below it.",
        rows=rows_above,
    )

    # -- B.3 the true optimum really is AT the corner below e* -----------
    # B.1 only compares two policies. This searches the whole simplex.
    budget = 2 * e_star * 0.5
    y_opt, x_opt = best_split_2(budget, pair)
    at_corner = min(x_opt, budget - x_opt) / budget < 0.01
    check(
        "below e* the unconstrained optimum sits AT a corner, not near one",
        at_corner,
        f"brute-force search over 2001 splits of a budget of "
        f"{budget:.1f} (share {budget / 2 / e_star:.2f} e*) puts the "
        f"optimum at x={x_opt:.2f} of {budget:.1f} --- a corner. B.1 "
        f"compared two hand-picked policies; this rules out an interior "
        f"optimum that both of them missed.",
        optimum_yield=y_opt, optimum_x=x_opt, budget=budget,
    )

    # -- B.4 above e*, the optimum moves INTO the interior ----------------
    budget_hi = 2 * e_star * 3.0
    y_hi, x_hi = best_split_2(budget_hi, pair)
    interior = abs(x_hi - budget_hi / 2) / budget_hi < 0.02
    check(
        "above e* the optimum is interior and symmetric, as T2 predicts",
        interior,
        f"at a budget of {budget_hi:.1f} (share "
        f"{budget_hi / 2 / e_star:.2f} e*) the optimum is x={x_hi:.2f}, "
        f"i.e. {100 * x_hi / budget_hi:.1f}% --- the even split, which "
        f"is what water-filling returns for two identical joins. T2 is "
        f"not merely valid above the threshold, it is exactly right "
        f"there.",
        optimum_yield=y_hi, optimum_x=x_hi, budget=budget_hi,
    )

    # -- B.5 the mixed case: unequal corpora, optimum beats BOTH policies
    # The realistic case, and the one that discriminates. An earlier
    # version of this check used corpora 12/300, where the optimum
    # happened to coincide with the commit policy --- so it passed
    # without distinguishing a per-join rule from "always commit". These
    # corpora were chosen because the optimum is strictly interior AND
    # strictly better than both naive policies, so neither can imitate it.
    small, large = 30, 90
    e_small, e_large = join_inflection(small), join_inflection(large)
    budget_mixed = e_small * 8.0
    mixed = [small, large]
    s_m = spread(budget_mixed, mixed)
    c_m = commit(budget_mixed, mixed)
    y_m, x_m = best_split_2(budget_mixed, mixed, steps=4000)
    frac_to_small = x_m / budget_mixed
    beats_both = y_m > s_m + 1e-6 and y_m > c_m + 1e-6
    check(
        "with unequal corpora the optimum beats BOTH naive policies",
        beats_both and 0.02 < frac_to_small < 0.98,
        f"corpora {small} and {large} have thresholds e*="
        f"{e_small:.1f} and {e_large:.1f}. At a budget of "
        f"{budget_mixed:.1f} the optimum gives "
        f"{100 * frac_to_small:.1f}% to the small join, yielding "
        f"{y_m:.3f} against {s_m:.3f} for an even split "
        f"({100 * (y_m - s_m) / s_m:+.1f}%) and {c_m:.3f} for committing "
        f"({100 * (y_m - c_m) / c_m:+.1f}%). Strictly interior and "
        f"strictly better than both, so neither naive policy can "
        f"imitate it and the allocation rule has to be genuinely "
        f"per-join.",
        e_star_small=e_small, e_star_large=e_large,
        spread=s_m, commit=c_m, optimum=y_m,
        optimum_fraction_to_small=frac_to_small,
    )

    # -- B.5b what the optimum actually equalises ------------------------
    # Not effort, and not marginal yield in raw units: effort measured in
    # units of each join's OWN threshold. This is a sharper rule than
    # "gate per join" and it is what the allocator should implement.
    share_small = (x_m / e_small)
    share_large = ((budget_mixed - x_m) / e_large)
    equalised = math.isclose(share_small, share_large, rel_tol=0.05)
    check(
        "the optimum equalises effort measured in units of each join's e*",
        equalised,
        f"the optimum gives the small join {x_m:.1f} = "
        f"{share_small:.2f} e* and the large join "
        f"{budget_mixed - x_m:.1f} = {share_large:.2f} e* --- equal to "
        f"within {100 * abs(share_small - share_large) / share_small:.1f}%, "
        f"despite the raw efforts differing threefold. The allocator's "
        f"state variable is therefore e/e*, not e: a join is 'behind' or "
        f"'ahead' relative to its own corpus, and that normalisation is "
        f"what makes a single shadow price meaningful across joins of "
        f"different sizes.",
        share_small_in_e_star=share_small,
        share_large_in_e_star=share_large,
    )

    # -- B.5c the crossover is at u_dagger, NOT at the edge of concavity --
    # An earlier draft of the theorem this suite supports claimed the
    # symmetric split wins as soon as both sources are on their concave
    # branches, i.e. B >= 2 e*. That is the necessary-not-sufficient error
    # again: concavity on [e*, B-e*] says nothing about the ENDPOINTS,
    # which lie outside it. The true boundary is u = 2 log2(1+sqrt2).
    #
    # Note none of B.1-B.4 would have caught this: they test u <= 1.8 and
    # u >= 3.0, straddling the gap without entering it. This check exists
    # to occupy that gap.
    U_CROSS = 2.0 * math.log2(1.0 + math.sqrt(2.0))
    rows_gap = []
    gap_ok = True
    for N_g in (20, 40, 200, 2000):
        e_g = join_inflection(N_g)
        pr = [N_g, N_g]
        # bisect for the actual crossing, sharing no algebra with U_CROSS
        lo, hi = 2.0, 3.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if commit(e_g * mid, pr) > spread(e_g * mid, pr):
                lo = mid
            else:
                hi = mid
        found = 0.5 * (lo + hi)
        # and confirm the old boundary would have been wrong here
        s2, c2 = spread(2.0 * e_g, pr), commit(2.0 * e_g, pr)
        rows_gap.append({
            "corpus_size": N_g, "crossover_found": found,
            "crossover_theory": U_CROSS, "abs_error": abs(found - U_CROSS),
            "at_u_equals_2_commit": c2, "at_u_equals_2_spread": s2,
            "corner_advantage_at_u2": (c2 - s2) / s2,
        })
        if abs(found - U_CROSS) > 1e-9 or not c2 > s2:
            gap_ok = False
    max_err = max(r["abs_error"] for r in rows_gap)
    adv2 = rows_gap[0]["corner_advantage_at_u2"]
    check(
        "the commit/spread crossover sits at u* = 2 log2(1+sqrt2), not at u = 2",
        gap_ok,
        f"bisecting for the crossing over corpora "
        f"{[r['corpus_size'] for r in rows_gap]} locates it at "
        f"u={rows_gap[0]['crossover_found']:.12f} for every one of them, "
        f"matching 2 log2(1+sqrt2)={U_CROSS:.12f} to within {max_err:.1e}. "
        f"The constant is exact for every N, not asymptotic. At u=2 "
        f"exactly --- where both sources are already on their concave "
        f"branches --- committing still beats the even split by "
        f"{100 * adv2:.1f}%, so concavity of the interior region is not "
        f"sufficient for the interior optimum. B.1-B.4 test u<=1.8 and "
        f"u>=3.0 and would both pass with the boundary placed anywhere "
        f"in between; this check is what pins it.",
        rows=rows_gap, u_crossover_theory=U_CROSS,
    )

    # -- B.6 how bad does the gate-less agent get? ------------------------
    # Magnitude, not just sign: the cost of not gating, at its worst.
    worst_ratio, worst_at = 0.0, None
    f = 0.02
    while f < 1.0:
        budget = 2 * e_star * f
        s, c = spread(budget, pair), commit(budget, pair)
        if s > 0:
            r = (c - s) / s
            if r > worst_ratio:
                worst_ratio, worst_at = r, f
        f += 0.02
    check(
        "the cost of not gating is large, not marginal",
        worst_ratio > 0.5,
        f"the worst relative loss from spreading instead of committing "
        f"is {100 * worst_ratio:.0f}%, at a share of {worst_at:.2f} e*. "
        f"The loss grows as the budget shrinks, so the agents most "
        f"likely to be budget-constrained --- the ones water-filling "
        f"exists to help --- are exactly the ones it would hurt most.",
        worst_relative_loss=worst_ratio, worst_at_share_of_e_star=worst_at,
    )

    passed = sum(1 for c in checks if c["verdict"] == "PASS")
    summary = {
        "experiment": "exp_b_corner_vs_spread",
        "question": "Below the convexity threshold found in EXP-A, does "
                    "water-filling actually pick the WORSE action, and "
                    "does the ordering reverse above it?",
        "answer":
            "YES to both. Below e* the optimum is at a corner and "
            "committing beats spreading by up to "
            f"{100 * worst_ratio:.0f}%; at a large enough budget the "
            "optimum moves to the interior and spreading wins. The "
            "threshold is a real switch in the optimal policy, not a "
            "bound on how much optimality is lost. The switch is NOT at "
            "the edge of the concave region: B.5c locates it at "
            f"u = B/e* = 2 log2(1+sqrt2) = {U_CROSS:.6f}, exact for "
            "every N, and at u=2 --- where both sources are already "
            "concave --- committing is still 12.5% ahead.",
        "consequence":
            "The allocator works in NORMALISED effort e/e*, not raw "
            "effort. Each candidate join carries its own threshold "
            "computed from its corpus size; the optimum equalises "
            "normalised effort across joins (B.5b), and joins whose "
            "normalised effort would fall below 1 are not water-filled "
            "but committed to or declined. A single global threshold, "
            "or a rule stated in raw effort, gets the mixed-corpus case "
            "wrong in both directions.",
        "aggregate": {
            "checks": len(checks), "passed": passed,
            "failed": len(checks) - passed,
            "verdict": "PASS" if passed == len(checks) else "FAIL",
        },
        "parameters": {"corpus_size": N, "e_star": e_star,
                       "method": "closed form; no sampling"},
        "checks": checks,
    }
    (RESULTS / "exp_b_corner_vs_spread.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("  EXP-B  below threshold: is spreading actually the wrong action?")
    print("=" * 72)
    for c in checks:
        print(f"\n  [{c['verdict']}] {c['check']}")
        for ln in _wrap(c["detail"], 64):
            print(f"          {ln}")
    print()
    print("-" * 72)
    print("  ANSWER:")
    for ln in _wrap(summary["answer"], 66):
        print(f"    {ln}")
    print()
    print("  CONSEQUENCE:")
    for ln in _wrap(summary["consequence"], 66):
        print(f"    {ln}")
    print("=" * 72)
    print()
    return 0 if passed == len(checks) else 1


def _wrap(text: str, w: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > w:
            out.append(cur); cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    sys.exit(main())
