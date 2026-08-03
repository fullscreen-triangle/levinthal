"""
The gated source allocator.

Its shape is dictated by EXP-A and EXP-B, not chosen:

  EXP-A  federated-join yield is N p^2 with p = 1-(1-1/N)^(e/2), which is
         a SIGMOID: convex below e* = 2 ln2/ln(1/(1-1/N)) -> 2N ln2, and
         concave above. Yield at e* is N/4 for every N.

  EXP-B  below e* the optimum is at a CORNER, and committing beats
         spreading by up to 97%. Above e* the ordering reverses. The
         optimum equalises effort in units of each join's OWN e*, not
         raw effort.

So this allocator:

  * works in NORMALISED effort u = e/e*, never raw effort;
  * gates PER JOIN, because a global threshold gets mixed corpora wrong
    in both directions (EXP-B.5);
  * REFUSES rather than guessing when the budget cannot lift every
    admitted join past its own threshold.

The refusal is the point. The split-attention paper's T2 is a theorem
about an environment that satisfies diminishing returns; where the
environment does not, the honest output is "I decline to allocate, and
here is the budget at which I could" --- not a confidently wrong split.
This is the same move as the two refusals in the medium-vertex paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# =====================================================================
#  Sources
# =====================================================================


@dataclass(frozen=True)
class JoinSource:
    """One federated join: a pair of endpoints joined on a shared key.

    `corpus_size` is the number of joinable keys --- NOT the number of
    records. For Rhea x UniProt that is the count of reactions with a
    resolvable protein link, which is knowable before querying (from a
    count query) and is the only parameter the threshold needs.
    """

    name: str
    corpus_size: int
    cost_per_query: float = 1.0

    @property
    def e_star(self) -> float:
        """Convex/concave inflection. Yield here is corpus_size/4."""
        q = 1.0 - 1.0 / self.corpus_size
        return 2.0 * math.log(2.0) / math.log(1.0 / q)

    def yield_at(self, effort: float) -> float:
        """Expected completed joins after `effort` queries."""
        if effort <= 0:
            return 0.0
        p = 1.0 - (1.0 - 1.0 / self.corpus_size) ** (effort / 2.0)
        return self.corpus_size * p * p

    def marginal_at(self, effort: float) -> float:
        """d(yield)/d(effort). Positive everywhere; peaks at e*."""
        if effort <= 0:
            effort = 1e-9
        q = 1.0 - 1.0 / self.corpus_size
        r = q ** (effort / 2.0)
        p = 1.0 - r
        # d/de [N p^2] = N * 2p * (-r' ) with r' = r ln(q) / 2
        return self.corpus_size * 2.0 * p * (-r * math.log(q) / 2.0)


class Refusal(Exception):
    """Raised when allocation is not well-founded at the given budget.

    Carries the budget that WOULD suffice, so the caller can act on it
    rather than merely being told no.
    """

    def __init__(self, message: str, required_budget: float,
                 offered_budget: float, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.required_budget = required_budget
        self.offered_budget = offered_budget
        self.detail = detail or {}

    def as_dict(self) -> dict:
        return {
            "refused": True,
            "reason": self.message,
            "offered_budget": self.offered_budget,
            "required_budget": self.required_budget,
            "shortfall": self.required_budget - self.offered_budget,
            **self.detail,
        }


@dataclass
class Allocation:
    """The result of a successful allocation."""

    per_source: dict[str, float] = field(default_factory=dict)
    normalised: dict[str, float] = field(default_factory=dict)
    shadow_price: float = 0.0
    total_yield: float = 0.0
    budget_used: float = 0.0
    policy: str = "water-fill"

    def as_dict(self) -> dict:
        return {
            "refused": False,
            "policy": self.policy,
            "per_source": self.per_source,
            "normalised_effort": self.normalised,
            "shadow_price": self.shadow_price,
            "total_yield": self.total_yield,
            "budget_used": self.budget_used,
        }


# =====================================================================
#  Allocation
# =====================================================================


def _water_fill(sources: list[JoinSource], budget: float,
                tol: float = 1e-10) -> Allocation:
    """Bisection on the shadow price, over the CONCAVE branch only.

    Every source here is guaranteed (by the caller) to receive at least
    e*, so marginal yield is decreasing over the whole feasible region
    and the first-order condition is sufficient. Without that guarantee
    this routine would be solving the wrong problem --- which is exactly
    what EXP-B measured the cost of.
    """
    lo, hi = 0.0, max(s.marginal_at(s.e_star) for s in sources)

    def effort_at_price(s: JoinSource, price: float) -> float:
        """Invert marginal_at on [e*, inf) by bisection."""
        a, b = s.e_star, s.e_star
        # expand until marginal drops below the price
        for _ in range(200):
            if s.marginal_at(b) <= price:
                break
            b *= 2.0
        else:
            return b
        for _ in range(200):
            m = 0.5 * (a + b)
            if s.marginal_at(m) > price:
                a = m
            else:
                b = m
            if b - a < tol:
                break
        return 0.5 * (a + b)

    price = 0.0
    for _ in range(200):
        price = 0.5 * (lo + hi)
        total = sum(effort_at_price(s, price) for s in sources)
        if total > budget:
            lo = price
        else:
            hi = price
        if hi - lo < tol:
            break

    price = 0.5 * (lo + hi)
    efforts = {s.name: effort_at_price(s, price) for s in sources}

    # Bisection converges to the price, leaving a small budget residue.
    # Hand it to the source with the highest marginal yield rather than
    # discarding it, so the reported budget_used is the real one.
    residue = budget - sum(efforts.values())
    if residue > 0:
        top = max(sources, key=lambda s: s.marginal_at(efforts[s.name]))
        efforts[top.name] += residue

    return Allocation(
        per_source=efforts,
        normalised={s.name: efforts[s.name] / s.e_star for s in sources},
        shadow_price=price,
        total_yield=sum(s.yield_at(efforts[s.name]) for s in sources),
        budget_used=sum(efforts.values()),
        policy="water-fill",
    )


def allocate(sources: list[JoinSource], budget: float,
             allow_commit: bool = True) -> Allocation:
    """Allocate `budget` queries across `sources`, or refuse.

    Three outcomes, in the order they are tested:

      1. budget >= sum of all e*  ->  WATER-FILL. Every source is on its
         concave branch, T2 applies, the shadow price is meaningful.

      2. budget >= the largest single e*  ->  COMMIT to the subset that
         can be lifted past threshold, chosen by achieved yield. This is
         the corner solution EXP-B found optimal below threshold.

      3. budget < every e*  ->  REFUSE. No source can be brought onto
         its concave branch, so no allocation rule here is founded. The
         refusal reports the budget that would suffice.
    """
    if not sources:
        raise ValueError("no sources given")
    if budget <= 0:
        raise Refusal("budget is non-positive", 0.0, budget)

    full = sum(s.e_star for s in sources)
    smallest = min(s.e_star for s in sources)

    # --- case 3: nothing can be lifted past its own threshold ---------
    if budget < smallest:
        cheapest = min(sources, key=lambda s: s.e_star)
        raise Refusal(
            f"budget {budget:.1f} is below every source's convexity "
            f"threshold (cheapest is {cheapest.name} at "
            f"{cheapest.e_star:.1f}). Join yield is convex there, so the "
            f"optimum is a corner and no interior allocation is "
            f"well-founded; water-filling would spread budget across "
            f"joins at exactly the budgets where committing is correct.",
            required_budget=smallest,
            offered_budget=budget,
            detail={"thresholds": {s.name: s.e_star for s in sources},
                    "yield_if_committed_anyway":
                        max(s.yield_at(budget) for s in sources)},
        )

    # --- case 1: everything is on the concave branch ------------------
    if budget >= full:
        interior = _water_fill(sources, budget)
        # `budget >= full` puts every source on its concave branch, but
        # that is NECESSARY, not SUFFICIENT, for the interior solution to
        # be optimal. Just past the boundary each source sits barely onto
        # its concave branch, where its marginal yield is still near the
        # PEAK it reaches at e*; dropping one source entirely then frees
        # budget to push another far enough up its curve to win outright.
        # Measured: at budget = sum(e*) exactly, funding only the larger
        # source beats the even-marginal split by 8%.
        #
        # The first-order condition cannot see this, because it compares
        # interior points and the winner is a corner. So the interior
        # solution is scored against every drop-one-source corner and the
        # better is returned. Found by EXP-C.3, which checks against
        # brute-force search rather than against the allocator's own
        # optimality condition.
        best = interior
        for drop in sources:
            rest = [s for s in sources if s is not drop]
            if not rest:
                continue
            if budget < sum(s.e_star for s in rest):
                continue
            cand = (_water_fill(rest, budget) if len(rest) > 1
                    else Allocation(
                        per_source={rest[0].name: budget},
                        normalised={rest[0].name: budget / rest[0].e_star},
                        shadow_price=rest[0].marginal_at(budget),
                        total_yield=rest[0].yield_at(budget),
                        budget_used=budget))
            if cand.total_yield > best.total_yield + 1e-12:
                cand.policy = "commit-subset"
                for s in sources:
                    cand.per_source.setdefault(s.name, 0.0)
                    cand.normalised.setdefault(s.name, 0.0)
                best = cand
        return best

    # --- case 2: some but not all can be lifted -----------------------
    if not allow_commit:
        raise Refusal(
            f"budget {budget:.1f} cannot lift every source past its "
            f"threshold (sum of thresholds is {full:.1f}), and "
            f"committing was disallowed.",
            required_budget=full, offered_budget=budget,
            detail={"thresholds": {s.name: s.e_star for s in sources}},
        )

    # Choose the subset to fund by ENUMERATION, scored on achieved yield.
    # A greedy rule (rank by yield-per-threshold, take while affordable)
    # was tried first and is not optimal: which subset wins depends on
    # how far the budget pushes each survivor up its own curve, not on
    # any per-source ratio computed before the split is known. Subsets
    # are few --- a pipeline federates a handful of sources, not
    # hundreds --- so enumeration is affordable and correct.
    best_alloc: Allocation | None = None
    n = len(sources)
    for mask in range(1, 1 << n):
        subset = [sources[i] for i in range(n) if mask & (1 << i)]
        if budget < sum(s.e_star for s in subset):
            continue  # cannot lift them all past threshold
        cand = (_water_fill(subset, budget) if len(subset) > 1
                else Allocation(
                    per_source={subset[0].name: budget},
                    normalised={subset[0].name: budget / subset[0].e_star},
                    shadow_price=subset[0].marginal_at(budget),
                    total_yield=subset[0].yield_at(budget),
                    budget_used=budget))
        if best_alloc is None or cand.total_yield > best_alloc.total_yield:
            best_alloc = cand
    if best_alloc is None:  # defensive: budget >= smallest guarantees one
        cheapest = min(sources, key=lambda s: s.e_star)
        best_alloc = Allocation(
            per_source={cheapest.name: budget},
            normalised={cheapest.name: budget / cheapest.e_star},
            shadow_price=cheapest.marginal_at(budget),
            total_yield=cheapest.yield_at(budget),
            budget_used=budget)

    alloc = best_alloc
    alloc.policy = "commit-subset"
    # Sources that got nothing are reported explicitly rather than
    # silently omitted: a caller must be able to see what was starved.
    for s in sources:
        alloc.per_source.setdefault(s.name, 0.0)
        alloc.normalised.setdefault(s.name, 0.0)
    return alloc
