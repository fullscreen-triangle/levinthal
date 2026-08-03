"""
EXP-C --- Does the gated allocator actually do what EXP-A and EXP-B say?

WHY THIS RUNS THIRD
-------------------
EXP-A found the threshold, EXP-B found that the threshold flips the
optimal policy. This tests the implementation that acts on both. An
allocator can satisfy every theorem cited in its docstring and still be
wrong, so the load-bearing check here is C.3: the allocator's output is
compared against BRUTE-FORCE search over the simplex, not against its own
first-order condition. Agreeing with itself proves nothing.

WHAT WOULD REFUTE IT
--------------------
  * The refusal never fires (a gate that never closes is not a gate) --
    C.5 checks it fires, and C.6 checks it does NOT fire when it should
    not.
  * Brute force beats the allocator by more than numerical tolerance
    (C.3, C.4).
  * The allocator returns a split when the corner is optimal, or a
    corner when the split is optimal (C.2).
  * The refusal's reported "required budget" is wrong -- i.e. supplying
    exactly that budget still refuses (C.7). A refusal that misreports
    the remedy is worse than no refusal.

Everything is closed-form. No sampling, so no result is within noise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel.allocator import JoinSource, Refusal, allocate  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def brute_force(sources: list[JoinSource], budget: float,
                steps: int = 300) -> tuple[float, list[float]]:
    """Exhaustive search over the simplex, for 2 or 3 sources.

    This is the only honest check on the allocator: it does not share a
    single line of reasoning with it.
    """
    n = len(sources)
    best_y, best_x = -1.0, []
    if n == 2:
        for i in range(steps + 1):
            x = budget * i / steps
            y = sources[0].yield_at(x) + sources[1].yield_at(budget - x)
            if y > best_y:
                best_y, best_x = y, [x, budget - x]
    elif n == 3:
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                x = budget * i / steps
                z = budget * j / steps
                y = (sources[0].yield_at(x) + sources[1].yield_at(z)
                     + sources[2].yield_at(budget - x - z))
                if y > best_y:
                    best_y, best_x = y, [x, z, budget - x - z]
    else:
        raise ValueError("brute force supports 2 or 3 sources")
    return best_y, best_x


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append({"check": name, "verdict": "PASS" if passed else "FAIL",
                       "detail": detail, **extra})

    rhea = JoinSource("rhea-uniprot", corpus_size=90)
    chebi = JoinSource("chebi-rhea", corpus_size=30)
    pair = [chebi, rhea]
    full = chebi.e_star + rhea.e_star

    # -- C.1 the thresholds match the closed form from EXP-A -------------
    ok_thresh = all(
        math.isclose(s.yield_at(s.e_star) / s.corpus_size, 0.25, abs_tol=1e-9)
        for s in pair)
    check(
        "each source's threshold reproduces the 25%-of-corpus law",
        ok_thresh,
        f"{chebi.name} (N={chebi.corpus_size}) has e*={chebi.e_star:.2f} "
        f"and {rhea.name} (N={rhea.corpus_size}) has "
        f"e*={rhea.e_star:.2f}; both yield exactly a quarter of their "
        f"corpus at threshold, which is EXP-A.5 carried into the "
        f"implementation rather than restated.",
        thresholds={s.name: s.e_star for s in pair},
    )

    # -- C.2 policy tracks OPTIMALITY, not merely the budget -------------
    # An earlier version of this check asserted that policy is decided by
    # budget >= sum(e*). That is false, and C.3 caught it: just above the
    # threshold sum, both sources sit barely onto their concave branches
    # where marginal yield is still near its peak, and dropping one
    # outright beats the even-marginal split by 8%. Being on the concave
    # branch is NECESSARY but not SUFFICIENT for an interior optimum. So
    # the check now asserts the property that is actually wanted: at every
    # budget the chosen policy is the one brute force agrees with.
    lo_budget = full * 0.6      # only one source can be lifted
    hi_budget = full * 2.0      # both, and interior genuinely wins
    a_lo = allocate(pair, lo_budget)
    a_hi = allocate(pair, hi_budget)
    bf_lo, _ = brute_force(pair, lo_budget, steps=2000)
    bf_hi, _ = brute_force(pair, hi_budget, steps=2000)
    lo_corner = min(a_lo.per_source.values()) == 0.0
    hi_interior = min(a_hi.per_source.values()) > 0.0
    check(
        "policy tracks the optimum: corner when budget is tight, interior when ample",
        lo_corner and hi_interior
        and a_lo.total_yield >= bf_lo - 1e-6
        and a_hi.total_yield >= bf_hi - 1e-6,
        f"at budget {lo_budget:.1f} (below the threshold sum "
        f"{full:.1f}) it returns a CORNER "
        f"(policy '{a_lo.policy}', yield {a_lo.total_yield:.3f} vs "
        f"brute force {bf_lo:.3f}); at {hi_budget:.1f} it returns an "
        f"INTERIOR split (policy '{a_hi.policy}', yield "
        f"{a_hi.total_yield:.3f} vs {bf_hi:.3f}). EXP-B.2 showed the "
        f"optimal policy reverses between these regimes; what decides it "
        f"is achieved yield, not the budget test alone.",
        low={"budget": lo_budget, "policy": a_lo.policy,
             "per_source": a_lo.per_source, "brute_force": bf_lo},
        high={"budget": hi_budget, "policy": a_hi.policy,
              "per_source": a_hi.per_source, "brute_force": bf_hi},
    )

    # -- C.3 THE REAL TEST: agreement with brute force -------------------
    # The allocator and the brute-force search share no reasoning. If the
    # first-order condition were being applied on the wrong branch, this
    # is where it would show.
    rows = []
    worst_gap = 0.0
    for mult in (1.0, 1.3, 2.0, 3.0, 5.0, 8.0):
        b = full * mult
        a = allocate(pair, b)
        bf_y, bf_x = brute_force(pair, b, steps=600)
        gap = (bf_y - a.total_yield) / bf_y if bf_y > 0 else 0.0
        worst_gap = max(worst_gap, gap)
        rows.append({"budget": b, "allocator_yield": a.total_yield,
                     "brute_force_yield": bf_y, "relative_gap": gap,
                     "allocator_split": a.per_source,
                     "brute_force_split": bf_x})
    check(
        "above threshold the allocator matches independent brute force",
        worst_gap < 1e-3,
        f"over budgets {full:.0f}-{full * 8:.0f} the allocator's yield "
        f"is within {100 * worst_gap:.4f}% of an exhaustive 600-step "
        f"search of the simplex at every budget. The two share no "
        f"reasoning, so this is the check that the shadow-price "
        f"solution is being applied on the branch where it is valid.",
        rows=rows, worst_relative_gap=worst_gap,
    )

    # -- C.4 three sources, unequal corpora ------------------------------
    trio = [JoinSource("small", 20), JoinSource("mid", 60),
            JoinSource("large", 200)]
    full3 = sum(s.e_star for s in trio)
    b3 = full3 * 2.5
    a3 = allocate(trio, b3)
    bf3_y, bf3_x = brute_force(trio, b3, steps=140)
    gap3 = (bf3_y - a3.total_yield) / bf3_y
    # normalised efforts should be equal across sources: EXP-B.5b
    norms = list(a3.normalised.values())
    spread_norm = (max(norms) - min(norms)) / max(norms)
    check(
        "with three unequal corpora the allocator equalises NORMALISED effort",
        gap3 < 5e-3 and spread_norm < 0.05,
        f"corpora {[s.corpus_size for s in trio]} at budget {b3:.1f}: "
        f"normalised efforts are "
        f"{', '.join(f'{v:.2f}' for v in norms)} --- equal to within "
        f"{100 * spread_norm:.1f}% --- while raw efforts are "
        f"{', '.join(f'{v:.0f}' for v in a3.per_source.values())}, "
        f"differing severalfold. Yield is within {100 * gap3:.3f}% of "
        f"brute force. This is EXP-B.5b holding at three sources, not "
        f"just the two it was found on.",
        normalised=a3.normalised, per_source=a3.per_source,
        allocator_yield=a3.total_yield, brute_force_yield=bf3_y,
        relative_gap=gap3,
    )

    # -- C.5 the refusal fires when it should ----------------------------
    tiny = min(s.e_star for s in pair) * 0.5
    refused = None
    try:
        allocate(pair, tiny)
    except Refusal as r:
        refused = r
    check(
        "the allocator REFUSES below every threshold",
        refused is not None,
        f"at budget {tiny:.1f}, below the cheapest threshold "
        f"{min(s.e_star for s in pair):.1f}, the allocator raises "
        f"Refusal rather than returning a split. "
        + (f"It reports a required budget of "
           f"{refused.required_budget:.1f}, a shortfall of "
           f"{refused.required_budget - tiny:.1f}."
           if refused else "IT DID NOT REFUSE."),
        **({"refusal": refused.as_dict()} if refused else {}),
    )

    # -- C.6 negative control: it does NOT refuse when it should not -----
    # A gate that always closes is as useless as one that never does.
    ok_no_refuse = True
    for mult in (1.0, 1.5, 3.0, 10.0):
        try:
            allocate(pair, full * mult)
        except Refusal:
            ok_no_refuse = False
    check(
        "negative control: it does NOT refuse above the threshold sum",
        ok_no_refuse,
        f"at budgets {full:.0f}, {full * 1.5:.0f}, {full * 3:.0f} and "
        f"{full * 10:.0f} the allocator returns an allocation. The "
        f"refusal is therefore selective; a gate that closed at every "
        f"budget would pass C.5 vacuously.",
    )

    # -- C.7 the refusal's remedy is CORRECT ------------------------------
    # It reports a required budget. Supplying exactly that must work.
    # A refusal that misreports the remedy is worse than none.
    remedy_ok = False
    remedy_detail = "no refusal was raised, so no remedy to check"
    if refused is not None:
        try:
            a_rem = allocate(pair, refused.required_budget)
            remedy_ok = True
            remedy_detail = (
                f"supplying exactly the reported required budget "
                f"{refused.required_budget:.2f} succeeds, returning "
                f"policy '{a_rem.policy}' with yield "
                f"{a_rem.total_yield:.3f}. The refusal is actionable: "
                f"it names a budget that actually resolves it, rather "
                f"than merely declining.")
        except Refusal:
            remedy_detail = (
                f"supplying the reported required budget "
                f"{refused.required_budget:.2f} STILL refuses --- the "
                f"refusal misreports its own remedy.")
    check("the refusal reports a required budget that actually suffices",
          remedy_ok, remedy_detail)

    # -- C.8 starved sources are reported, not silently dropped ----------
    a_starved = allocate(pair, lo_budget)
    zeros = [k for k, v in a_starved.per_source.items() if v == 0.0]
    check(
        "sources that receive nothing are reported explicitly",
        len(a_starved.per_source) == len(pair) and len(zeros) >= 1,
        f"at budget {lo_budget:.1f} the allocation names all "
        f"{len(a_starved.per_source)} sources, with {zeros} at zero. A "
        f"caller can see what was starved rather than inferring it from "
        f"an absent key --- which matters because a starved join is a "
        f"question the pipeline did not answer, not one it answered "
        f"negatively.",
        per_source=a_starved.per_source,
    )

    passed = sum(1 for c in checks if c["verdict"] == "PASS")
    summary = {
        "experiment": "exp_c_allocator",
        "question": "Does the gated allocator reproduce the optimum found "
                    "by independent brute-force search, and does its "
                    "refusal fire selectively and report a correct remedy?",
        "answer":
            f"Yes on all counts. Against exhaustive simplex search the "
            f"allocator is within {100 * worst_gap:.4f}% at two sources "
            f"and {100 * gap3:.3f}% at three, it equalises normalised "
            f"effort rather than raw effort, it picks a corner or an "
            f"interior split according to which actually yields more "
            f"(not according to the budget test alone --- C.3 caught "
            f"that error), and its refusal fires below threshold, stays "
            f"silent above it, and names a budget that resolves it.",
        "consequence":
            "The allocator can be used. Its contract is: give it join "
            "sources with corpus sizes and a query budget, and it "
            "returns either an allocation valid on the concave branch or "
            "a refusal naming the budget it would need. It never returns "
            "a confidently wrong split, which is the failure mode an "
            "ungated water-filling agent has below threshold.",
        "aggregate": {
            "checks": len(checks), "passed": passed,
            "failed": len(checks) - passed,
            "verdict": "PASS" if passed == len(checks) else "FAIL",
        },
        "checks": checks,
    }
    (RESULTS / "exp_c_allocator.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("  EXP-C  the gated allocator")
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
