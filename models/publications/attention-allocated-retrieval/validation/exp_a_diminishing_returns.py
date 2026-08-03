"""
EXP-A --- Does information retrieval actually have diminishing returns?

WHY THIS RUNS FIRST
-------------------
The split-attention agent framework proves (T2) that optimal division of a
finite attention budget over concurrent scenes is a water-filling rule with
a single shadow price. That theorem is *conditional*: it assumes
diminishing returns of attention within a scene, and the paper flags this
explicitly as an ENVIRONMENTAL assumption --- a property of the world, not
of the agent.

If information sources do NOT have diminishing returns, T2 does not apply
to retrieval and the whole "agent allocates queries across sources"
architecture is unfounded. So this is tested before anything is built on
it.

WHAT WOULD REFUTE IT
--------------------
Diminishing returns means gamma(a) is concave: the second unit of effort
yields less new information than the first. It is refuted if yield is
linear (every query returns as much as the last) or convex (later queries
return MORE --- e.g. if results only become useful in combination).

Three regimes are tested, and they are not all expected to be concave.
That is the point: a test where every arm passes is not a test.

  1. DISTINCT-ENTITY retrieval. Querying a source for facts about a fixed
     entity set. Modelled by coupon-collector: each query returns a random
     record, and yield is the number of DISTINCT records seen. Concavity
     here is a theorem, not an assumption --- included as a positive
     control that the measurement can detect concavity at all.

  2. JOIN retrieval. The federated case: results are useful only when a
     Rhea record joins to a UniProt record. Yield is the number of
     COMPLETED joins. This is the regime tacat actually operates in, and
     it is NOT obviously concave --- early queries produce unjoinable
     fragments, making the early curve convex.

  3. ADVERSARIAL/paginated retrieval. A source that returns results in
     fixed pages with no overlap. Yield is exactly linear in effort. This
     is the refuting case and it MUST come back non-concave, or the
     concavity detector is broken.

  4. JOIN, CLOSED FORM. The same join regime evaluated analytically, with
     no sampling at all. Added after the sampled join arm failed, to
     settle whether the failure was Monte Carlo noise or the world. It is
     the world: see below.

WHAT THIS EXPERIMENT ACTUALLY FOUND
-----------------------------------
The join regime is NOT concave, and this is structural rather than
statistical. With effort split evenly between two sources,

    E[|A cap B|] = N * p(e)^2,   p(e) = 1 - (1 - 1/N)^(e/2),

because an identifier joins only if it was drawn on BOTH sides, and the
two draws are independent. Squaring a concave saturating function makes
it sigmoid: convex through the whole rise, concave only near saturation.
The closed form is convex at 54 of 59 second differences over the tested
range --- with zero sampling noise --- so no amount of extra trials would
have rescued it.

The inflection is exact. Setting d^2/de^2 [N p^2] = 0 gives p(e*) = 1/2,
hence

    e* = 2 ln 2 / ln(1/(1-1/N))  ->  2 N ln 2   as N grows,

and at that point the yield is N/4: the curve turns concave precisely
when ONE QUARTER of the joinable corpus has been recovered, independently
of N. Verified numerically for N in {10, 20, 40, 80, 160, 400, 1000};
e*/(2N ln 2) rises from 0.949 to 0.999 and the recovered fraction is
25.0% in every case.

CONSEQUENCE FOR THE ALLOCATOR
-----------------------------
T2's water-filling is invalid below e*. Under increasing returns the
optimum is a corner, not an interior split: the shadow price argument
inverts and the correct policy is to commit the whole budget to a single
join rather than spread it. An allocator run below threshold would spread,
and spreading is exactly wrong there.

This does not kill the allocator; it bounds it. It yields a REFUSAL
condition of the same shape as the two refusals in the medium-vertex
paper: below a computable threshold the agent must decline to allocate
and say why, rather than return a confidently wrong split.

Nothing here queries a live endpoint. The structure being tested is the
combinatorics of retrieval, which is source-independent; live latency
(measured at 1.1s vs 43s for Rhea summary vs detail) enters the later
allocation experiment as cost, not as yield.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 20260731
# 400 trials leaves Monte Carlo noise (~0.08) five times larger than the
# true curvature (~0.016), which made the first run report the ANALYTICALLY
# concave positive control as non-concave. Convergence was checked against
# the closed form E[distinct] = N(1-(1-1/N)^e): at 400 trials the maximum
# second difference is +0.1075 (22/59 violations), at 4000 it is +0.0148
# (7/59), and at 40000 it is -0.0025 (0/59) against an exact -0.0058.
N_TRIALS = 40000
MAX_EFFORT = 60


# =====================================================================
#  Yield models
# =====================================================================


def yield_distinct(effort: int, corpus_size: int, rng: random.Random) -> int:
    """Regime 1: distinct records seen after `effort` random draws."""
    seen = set()
    for _ in range(effort):
        seen.add(rng.randrange(corpus_size))
    return len(seen)


def yield_join(effort: int, corpus_size: int, rng: random.Random) -> int:
    """Regime 2: completed joins after `effort` draws across TWO sources.

    Effort is split between source A and source B. A record is only
    counted when its counterpart has also been drawn --- which is exactly
    what "identifiers were reused so the join works" buys you, and
    exactly what fails when they were not.
    """
    a_seen, b_seen = set(), set()
    for i in range(effort):
        if i % 2 == 0:
            a_seen.add(rng.randrange(corpus_size))
        else:
            b_seen.add(rng.randrange(corpus_size))
    return len(a_seen & b_seen)


def yield_join_exact(effort: float, corpus_size: int) -> float:
    """Regime 4: the join curve in closed form, with no sampling.

    E[|A cap B|] = N p^2 with p = 1 - (1-1/N)^(e/2): an identifier joins
    only if drawn on both sides, and the two events are independent.
    """
    p = 1.0 - (1.0 - 1.0 / corpus_size) ** (effort / 2.0)
    return corpus_size * p * p


def join_inflection(corpus_size: int) -> float:
    """Effort e* at which the join curve turns from convex to concave.

    d^2/de^2 [N p^2] = 0 exactly when p = 1/2, so
    e* = 2 ln2 / ln(1/(1-1/N)), which tends to 2 N ln 2. Yield there is
    N/4 for every N.
    """
    q = 1.0 - 1.0 / corpus_size
    return 2.0 * math.log(2.0) / math.log(1.0 / q)


def yield_paginated(effort: int, corpus_size: int, rng: random.Random) -> int:
    """Regime 3: a paginated source with no repeats. Exactly linear.

    Deliberately NOT capped at corpus_size over the tested range: a cap
    would make the curve flatten at the end and manufacture the very
    diminishing returns this arm exists to lack. A source that keeps
    paginating is the honest refuting case.
    """
    per_page = 3
    return effort * per_page


REGIMES = {
    "distinct-entity": (yield_distinct, "positive control: should be concave"),
    "federated-join": (yield_join, "the regime tacat operates in: unknown"),
    "paginated": (yield_paginated, "refuting control: must be NON-concave"),
}


# =====================================================================
#  Concavity measurement
# =====================================================================


def mean_curve(fn, corpus_size: int, seed: int) -> list[float]:
    """Expected yield at each effort level, averaged over trials.

    COMMON RANDOM NUMBERS. The first version of this drew an independent
    sample at every effort level, which made adjacent points independent
    and left Monte Carlo noise in the second differences. That noise
    (std ~0.081) was five times the true curvature (~0.016), so the
    concavity test was measuring noise and reported the positive control
    as non-concave. Reusing one RNG stream per trial across ALL effort
    levels makes the curve monotone within each trial and cancels the
    noise in the differences.
    """
    totals = [0.0] * (MAX_EFFORT + 1)
    for t in range(N_TRIALS):
        rng = random.Random(seed + t)
        state = rng.getstate()
        for e in range(MAX_EFFORT + 1):
            rng.setstate(state)  # same draws, longer prefix
            totals[e] += fn(e, corpus_size, rng)
    return [x / N_TRIALS for x in totals]


def second_differences(curve: list[float]) -> list[float]:
    """Discrete second derivative. Concave <=> all <= 0."""
    return [curve[i + 1] - 2 * curve[i] + curve[i - 1]
            for i in range(1, len(curve) - 1)]


def concavity_report(curve: list[float]) -> dict:
    """Concavity, tested against the sampling noise rather than against 0.

    A finite-sample curve never has exactly non-positive second
    differences. The honest question is whether positive ones exceed what
    sampling error explains. We use a tolerance of 3 standard errors of
    the second-difference sequence itself, and additionally report the
    marginal-decay ratio, which is the practically meaningful quantity:
    diminishing returns means late effort yields less than early effort.
    """
    d2 = second_differences(curve)
    n = len(d2)
    mean_d2 = sum(d2) / n if n else 0.0
    var = sum((v - mean_d2) ** 2 for v in d2) / n if n else 0.0
    noise = math.sqrt(var)
    tol = 3.0 * noise / math.sqrt(N_TRIALS) + 1e-12
    violations = [i for i, v in enumerate(d2) if v > tol]
    d1 = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    # Measure decay from the PEAK marginal yield, not from index 0. The
    # join regime has zero marginal yield at the origin --- no join can
    # complete before both sides are sampled --- so a ratio taken from
    # index 0 divides by zero and returns nan, which is not evidence of
    # anything. The meaningful question is whether marginal yield falls
    # once it has started.
    peak_i = max(range(len(d1)), key=lambda i: d1[i]) if d1 else 0
    peak = d1[peak_i] if d1 else 0.0
    ratio = d1[-1] / peak if peak > 1e-12 else float("nan")
    return {
        "concave": len(violations) == 0,
        "diminishing": (not math.isnan(ratio)) and ratio < 0.9,
        "peak_marginal_at_effort": peak_i,
        "peak_marginal": peak,
        "n_second_differences": n,
        "n_positive": len(violations),
        "tolerance_used": tol,
        "noise_std_of_d2": noise,
        "mean_second_difference": mean_d2,
        "max_second_difference": max(d2) if d2 else 0.0,
        "marginal_first": d1[0] if d1 else 0.0,
        "marginal_last": d1[-1] if d1 else 0.0,
        "marginal_ratio_last_over_first": ratio,
        "total_yield_at_max_effort": curve[-1],
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    curves: dict = {}

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append({"check": name, "verdict": "PASS" if passed else "FAIL",
                       "detail": detail, **extra})

    corpus = 40
    for rname, (fn, note) in REGIMES.items():
        curve = mean_curve(fn, corpus, SEED)
        rep = concavity_report(curve)
        curves[rname] = {"curve": curve, "report": rep, "note": note}

    # the closed form, for the record: no sampling, no seed, no trials
    curves["federated-join-exact"] = {
        "curve": [yield_join_exact(e, corpus) for e in range(MAX_EFFORT + 1)],
        "note": "analytic N(1-(1-1/N)^(e/2))^2; no sampling",
    }

    # -- A.1 positive control: distinct-entity retrieval diminishes ------
    r = curves["distinct-entity"]["report"]
    check(
        "positive control: distinct-entity retrieval shows diminishing returns",
        r["diminishing"] and r["concave"],
        f"marginal yield falls from {r['marginal_first']:.3f} to "
        f"{r['marginal_last']:.3f} "
        f"({100 * r['marginal_ratio_last_over_first']:.1f}% of the first); "
        f"{r['n_positive']}/{r['n_second_differences']} second differences "
        f"exceed the sampling tolerance {r['tolerance_used']:.2e}. "
        f"Coupon-collector yield is analytically concave, so a failure "
        f"here would mean the detector is broken, not the world.",
        **r,
    )

    # -- A.2 refuting control: paginated retrieval does NOT diminish -----
    # A linear curve has zero second differences, so it is *weakly*
    # concave and a pointwise concavity test passes it vacuously. The
    # operative test is marginal DECAY, which must not occur here.
    r = curves["paginated"]["report"]
    check(
        "refuting control: paginated retrieval shows NO diminishing returns",
        not r["diminishing"],
        f"marginal yield is {r['marginal_first']:.3f} at the start and "
        f"{r['marginal_last']:.3f} at the end (ratio "
        f"{r['marginal_ratio_last_over_first']:.3f}). A source that keeps "
        f"paginating violates the diminishing-returns assumption, so T2 "
        f"does NOT apply to it. Note this curve is weakly concave "
        f"(all second differences are 0), which is why concavity alone "
        f"is the wrong criterion and marginal decay is the right one.",
        **r,
    )

    # -- A.3 THE ACTUAL QUESTION: the federated join regime does NOT ------
    # This is the finding. It was written as a PASS-if-diminishing check
    # and it failed; the failure survived a 100x increase in trials, a
    # parity-free resampling of the source split, and finally an exact
    # closed-form evaluation with no sampling at all. It is the world,
    # not the measurement, so the check now asserts the true statement.
    r = curves["federated-join"]["report"]
    check(
        "federated-join retrieval does NOT show diminishing returns",
        not r["diminishing"],
        f"marginal yield {r['marginal_first']:.3f} -> "
        f"{r['marginal_last']:.3f}; "
        f"{r['n_positive']}/{r['n_second_differences']} second differences "
        f"are positive beyond the sampling tolerance. Joins require BOTH "
        f"sides to be drawn, so yield goes as p^2 with p concave, and "
        f"squaring a saturating function gives a sigmoid. T2 therefore "
        f"does NOT apply to the regime tacat's Rhea/UniProt join operates "
        f"in --- at least not at every budget. A.4 says where it does.",
        **r,
    )

    # -- A.4 not sampling: the closed form is convex too ------------------
    # The decisive check. If the sampled curve were merely noisy, the
    # exact curve would be concave. It is not.
    exact = [yield_join_exact(e, corpus) for e in range(MAX_EFFORT + 1)]
    d2x = second_differences(exact)
    n_convex_exact = sum(1 for v in d2x if v > 0)
    check(
        "the join non-concavity is structural, not Monte Carlo noise",
        n_convex_exact > len(d2x) // 2,
        f"the CLOSED FORM E[|A cap B|] = N(1-(1-1/N)^(e/2))^2, evaluated "
        f"with no sampling whatsoever, is convex at "
        f"{n_convex_exact}/{len(d2x)} second differences "
        f"(max {max(d2x):+.5f}). No number of trials would have made the "
        f"sampled arm concave. This is why the failing A.3 was promoted "
        f"to a finding rather than debugged further.",
        n_convex_exact=n_convex_exact, n_total=len(d2x),
        max_second_difference=max(d2x),
    )

    # -- A.5 the threshold is exact and scales as 2 N ln 2 ---------------
    # p = 1/2 at the inflection, so yield there is N/4 for EVERY N. This
    # is what makes the refusal condition computable at run time rather
    # than a tuned constant.
    scaling = []
    ok_scaling = True
    for n in (10, 20, 40, 80, 160, 400, 1000):
        e_star = join_inflection(n)
        asymptote = 2 * n * math.log(2)
        frac = yield_join_exact(e_star, n) / n
        scaling.append({"corpus_size": n, "e_star": e_star,
                        "two_n_ln2": asymptote,
                        "ratio_to_asymptote": e_star / asymptote,
                        "fraction_of_corpus_joined": frac})
        if not math.isclose(frac, 0.25, abs_tol=1e-9):
            ok_scaling = False
    check(
        "the convex/concave threshold sits at exactly 25% of the corpus",
        ok_scaling,
        f"e* = 2 ln2 / ln(1/(1-1/N)) -> 2N ln2 (ratio rises "
        f"{scaling[0]['ratio_to_asymptote']:.3f} -> "
        f"{scaling[-1]['ratio_to_asymptote']:.3f} over N=10..1000), and "
        f"the yield at e* is N/4 for every N tested. The threshold is "
        f"therefore computable from corpus size alone, with no fitted "
        f"constant --- which is what lets an agent decide at run time "
        f"whether it is allowed to allocate.",
        scaling=scaling,
    )

    # -- A.6 above the threshold, T2 is recovered ------------------------
    # The negative result is bounded, and this measures the bound. If the
    # curve above e* were still convex the allocator would be dead
    # outright; it is not.
    e_star = join_inflection(corpus)
    hi = [yield_join_exact(e, corpus)
          for e in range(math.ceil(e_star), math.ceil(e_star) + 60)]
    d2hi = second_differences(hi)
    n_convex_hi = sum(1 for v in d2hi if v > 1e-12)
    check(
        "above the threshold the join regime IS concave, so T2 is recovered",
        n_convex_hi == 0,
        f"for effort >= e* = {e_star:.1f} (corpus {corpus}) the exact "
        f"curve has {n_convex_hi}/{len(d2hi)} positive second "
        f"differences. The negative result is therefore a BOUND, not a "
        f"refutation: water-filling is valid above e* and invalid below "
        f"it, and an allocator that checks the bound is well-founded on "
        f"the side where it acts.",
        e_star=e_star, n_convex_above=n_convex_hi, n_total=len(d2hi),
    )

    passed = sum(1 for c in checks if c["verdict"] == "PASS")
    join_diminishes = curves["federated-join"]["report"]["diminishing"]

    summary = {
        "experiment": "exp_a_diminishing_returns",
        "question": "Does information retrieval satisfy the diminishing-"
                    "returns assumption that T2 (water-filling) requires?",
        "answer":
            "PARTLY, and the boundary is exact. Distinct-entity retrieval "
            "diminishes everywhere. Paginated retrieval never diminishes. "
            "Federated JOIN retrieval --- the regime that matters --- is "
            "CONVEX below a threshold effort e* = 2 ln2 / ln(1/(1-1/N)) "
            "-> 2N ln2 and concave above it, because yield goes as the "
            "SQUARE of a saturating coverage. The threshold is where 25% "
            "of the joinable corpus has been recovered, for every N.",
        "consequence":
            "Build the allocator, but gate it. Water-filling (T2) is "
            "valid above e* and INVALID below it: under increasing "
            "returns the optimum is a corner, so an unguarded allocator "
            "would spread budget across joins at exactly the budgets "
            "where committing it to one is correct. Below e* the agent "
            "must refuse to allocate and say so --- a refusal of the "
            "same shape as the two in the medium-vertex paper, and "
            "computable from corpus size with no fitted constant. "
            "Paginated sources are excluded at every budget.",
        "join_regime": {
            "diminishing_over_tested_range": join_diminishes,
            "inflection_effort": join_inflection(corpus),
            "inflection_yield_fraction": 0.25,
            "asymptotic_form": "e* -> 2 N ln 2",
        },
        "aggregate": {
            "checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "verdict": "PASS" if passed == len(checks) else "FAIL",
        },
        "parameters": {"seed": SEED, "trials": N_TRIALS,
                       "max_effort": MAX_EFFORT, "corpus_size": corpus},
        "curves": curves,
        "checks": checks,
    }
    (RESULTS / "exp_a_diminishing_returns.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("  EXP-A  does retrieval have diminishing returns?")
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
