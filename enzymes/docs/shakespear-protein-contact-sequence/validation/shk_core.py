#!/usr/bin/env python3
"""
Core kernel for

    "Shakespeare: A Receiver Language with Ladder Composition"

This module is the whole machine: contact graph, floor, cut key, rung,
ladder, the resolution-indexed power derivation, and the small-step
operational semantics with the ladder construct.

DESIGN NOTES THAT ARE PART OF THE FORMALISM, NOT COMMENTARY
-----------------------------------------------------------
1.  A Rung carries exactly one datum: its power.  There is no identity
    field.  A test of label-independence that compared objects carrying
    names would be testing the names.

2.  Power is DERIVED from a cut key, not declared.  The prior formulation
    left powers as measured inputs; here a rung's power is computed from
    the separation cost of the region it acts on.  This is what makes the
    intensivity question answerable rather than assumed.

3.  `sigma` is the separation cost against the medium: the minimum weight
    of a cut placing an item on one side and the medium on the other.
    It is INTENSIVE -- it does not depend on what else is in the graph
    beyond the item's own neighbourhood at the declared radius.  The
    intensivity of the power is what licenses order-independent
    composition, and it is tested, not asserted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Edge = FrozenSet[str]


# ===========================================================================
# Contact graph, floor, cut, separation cost
# ===========================================================================
@dataclass
class ContactGraph:
    """G = (V, E, w, m) with a medium adjacent to every item."""
    vertices: Set[str]
    weights: Dict[Edge, float]
    medium: str = "m"

    def __post_init__(self) -> None:
        for e, w in self.weights.items():
            if w <= 0.0:
                raise ValueError(f"weight must be strictly positive: {e}={w}")

    @property
    def floor(self) -> float:
        """beta = min over a FINITE edge set of strictly positive weights."""
        if not self.weights:
            raise ValueError("floor undefined on an empty edge set")
        return min(self.weights.values())

    @property
    def total(self) -> float:
        return sum(self.weights.values())

    def items(self) -> Set[str]:
        return self.vertices - {self.medium}

    def neighbours(self, v: str) -> Set[str]:
        out = set()
        for e in self.weights:
            a, b = tuple(e)
            if a == v:
                out.add(b)
            elif b == v:
                out.add(a)
        return out

    def ball(self, v: str, radius: int) -> Set[str]:
        """Vertices within `radius` hops of v, excluding the medium.

        radius 0 = {v} itself.  The medium is adjacent to everything, so
        including it would make every ball the whole graph at radius 2 and
        destroy the resolution parameter.
        """
        seen = {v}
        frontier = {v}
        for _ in range(radius):
            nxt: Set[str] = set()
            for u in frontier:
                nxt |= (self.neighbours(u) - {self.medium} - seen)
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        return seen

    def cut(self, S: Set[str]) -> List[Edge]:
        out = []
        for e in self.weights:
            u, v = tuple(e)
            if (u in S) != (v in S):
                out.append(e)
        return out

    def residue(self, S: Set[str]) -> float:
        return sum(self.weights[e] for e in self.cut(S))

    def admissible(self, S: Set[str]) -> bool:
        return bool(S & self.items()) and self.medium not in S

    def sigma(self, v: str) -> float:
        """
        Separation cost against the medium:
            sigma(v) = min over admissible S containing v of w(cut(S)).

        Exhaustive over subsets: correct by construction, used as ground
        truth.  Exponential, so callers must keep graphs small.
        """
        others = sorted(self.items() - {v})
        best = math.inf
        for mask in range(1 << len(others)):
            S = {v} | {others[i] for i in range(len(others)) if mask >> i & 1}
            r = self.residue(S)
            if r < best:
                best = r
        return best

    def sigma_local(self, v: str, radius: int) -> float:
        """
        Separation cost computed on the ball of the given radius around v.

        This is the RESOLUTION-INDEXED cut key.  radius is a resolution
        parameter, not a threshold on a score: raising it admits more of
        the graph into the computation and therefore discriminates more.
        """
        ball = self.ball(v, radius)
        others = sorted(ball - {v})
        best = math.inf
        for mask in range(1 << len(others)):
            S = {v} | {others[i] for i in range(len(others)) if mask >> i & 1}
            r = self.residue(S)
            if r < best:
                best = r
        return best


def complete_graph(n_items: int, weights: Sequence[float],
                   medium: str = "m") -> ContactGraph:
    verts = {medium} | {f"v{i}" for i in range(n_items)}
    edges: Dict[Edge, float] = {}
    k = 0
    names = sorted(verts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            edges[frozenset((names[i], names[j]))] = weights[k % len(weights)]
            k += 1
    return ContactGraph(verts, edges, medium)


def chain_graph(n_items: int, rng: random.Random,
                medium_weight: float = 1.0,
                lo: float = 0.5, hi: float = 3.0,
                medium: str = "m") -> ContactGraph:
    """
    A path v0 - v1 - ... - v(n-1), plus a medium adjacent to every item.

    Locality is real here: v0 and v(n-1) are far apart, so a ball of small
    radius genuinely omits part of the graph.  In a complete graph every
    ball is everything at radius 1 and the resolution parameter is inert.
    """
    verts = {medium} | {f"v{i}" for i in range(n_items)}
    edges: Dict[Edge, float] = {}
    for i in range(n_items):
        edges[frozenset((f"v{i}", medium))] = medium_weight
    for i in range(n_items - 1):
        edges[frozenset((f"v{i}", f"v{i+1}"))] = rng.uniform(lo, hi)
    return ContactGraph(verts, edges, medium)


# ===========================================================================
# Rung, ladder, composition
# ===========================================================================
@dataclass(frozen=True)
class Rung:
    """
    A rung carries exactly one datum: its power.

    No identity field exists.  This is not an omission for brevity; it is
    the formalism.  A test of label-independence that compared objects
    carrying names would be testing the names.
    """
    power: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.power <= 1.0):
            raise ValueError(f"power must lie in [0,1]: {self.power}")


@dataclass
class Ladder:
    rungs: List[Rung]

    @property
    def powers(self) -> List[float]:
        return [r.power for r in self.rungs]

    def composite_power(self) -> float:
        p = 1.0
        for r in self.rungs:
            p *= (1.0 - r.power)
        return 1.0 - p

    def residual_fraction(self) -> float:
        return 1.0 - self.composite_power()

    def gap_trajectory(self, gap0: float = 1.0) -> List[float]:
        g = gap0
        out = [g]
        for r in self.rungs:
            g *= (1.0 - r.power)
            out.append(g)
        return out

    def sensitivity(self) -> List[float]:
        """d(composite)/d(pi_j) = prod_{i != j} (1 - pi_i)."""
        out = []
        for j in range(len(self.rungs)):
            p = 1.0
            for i, r in enumerate(self.rungs):
                if i != j:
                    p *= (1.0 - r.power)
            out.append(p)
        return out


def compose_multiplicative(powers: Sequence[float]) -> float:
    p = 1.0
    for x in powers:
        p *= (1.0 - x)
    return 1.0 - p


def compose_additive(powers: Sequence[float]) -> float:
    return min(1.0, sum(powers))


def compose_max(powers: Sequence[float]) -> float:
    return max(powers) if powers else 0.0


def compose_mean(powers: Sequence[float]) -> float:
    return sum(powers) / len(powers) if powers else 0.0


# ===========================================================================
# Derived power: intensive and extensive candidates
# ===========================================================================
def local_floor(g: ContactGraph, v: str, radius: int) -> float:
    """
    The minimum edge weight WITHIN the ball of the given radius around v.

    This is the local analogue of the global floor.  It exists for the same
    reason the global floor does -- a finite set of strictly positive
    weights has a positive minimum -- but it is computed from the ball
    alone, so a distant edge cannot move it.
    """
    ball = g.ball(v, radius) | {g.medium}
    ws = [w for e, w in g.weights.items() if set(e) <= ball]
    return min(ws) if ws else g.floor


def power_intensive(g: ContactGraph, v: str, radius: int = 1) -> float:
    """
    INTENSIVE power.  Both the cut key AND its normaliser are local:

        pi(v) = 1 - beta_r(v) / sigma_r(v)        clipped to [0,1]

    where sigma_r is the separation cost computed on the ball of radius r
    and beta_r is the minimum edge weight inside that same ball.

    Every quantity is a function of the ball alone, so adding items outside
    the ball leaves pi unchanged exactly.  That exactness is the property
    under test in V2, and it is what licenses order-independent
    composition.
    """
    s = g.sigma_local(v, radius)
    if s <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - local_floor(g, v, radius) / s))


def power_globalfloor(g: ContactGraph, v: str, radius: int = 1) -> float:
    """
    CONTROL 1 -- a NEAR-MISS, and the more informative of the two controls.

        pi(v) = 1 - beta / sigma_r(v)

    The cut key is local but the normaliser is the GLOBAL floor.  Because
    beta is a minimum over every edge in the graph, a single distant edge
    of small weight lowers it and shifts this quantity everywhere.

    This control exists because we got it wrong first.  The local cut key
    alone does not buy intensivity; the normaliser must be local too.  A
    control that differs from the real candidate in one component is worth
    more than one that differs in all of them, because it localises the
    failure instead of merely exhibiting one.
    """
    s = g.sigma_local(v, radius)
    if s <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - g.floor / s))


def power_extensive(g: ContactGraph, v: str, radius: int = 1) -> float:
    """
    CONTROL 2 -- plainly EXTENSIVE.

        pi(v) = sigma_r(v) / Omega

    Omega is the total edge weight of the whole graph, so this changes when
    any edge is added anywhere.  If the intensivity test cannot separate
    this from the intensive candidate, the test is not measuring
    intensivity and must be reported as non-discriminating.
    """
    tot = g.total
    if tot <= 0.0:
        return 0.0
    return max(0.0, min(1.0, g.sigma_local(v, radius) / tot))


def ladder_from_graph(g: ContactGraph, order: Sequence[str],
                      radius: int = 1, intensive: bool = True) -> Ladder:
    """Build a ladder by deriving one rung power per item, in the given order."""
    f = power_intensive if intensive else power_extensive
    return Ladder([Rung(f(g, v, radius)) for v in order])


# ===========================================================================
# Sequential derivation -- the setting in which order can genuinely matter
# ===========================================================================
def derive_sequential(g: ContactGraph, order: Sequence[str],
                      power_fn, radius: int = 1) -> List[float]:
    """
    Derive one rung power per item, IN ORDER, where each derivation sees the
    graph as its predecessors left it.

    Committing a rung consumes the item's weakest item-item contact.  This
    is what makes the ordering test able to fail: if powers were derived
    from the pristine graph, every candidate would trivially agree across
    orderings because the composition law is symmetric in its arguments.
    """
    gg = ContactGraph(set(g.vertices), dict(g.weights), g.medium)
    out: List[float] = []
    for v in order:
        out.append(power_fn(gg, v, radius))
        cand = [(w, e) for e, w in gg.weights.items()
                if v in set(e) and gg.medium not in set(e)]
        if cand:
            gg.weights.pop(min(cand)[1])
    return out


# ===========================================================================
# Static analysis
# ===========================================================================
def static_reachable(powers: Sequence[float], target: float) -> bool:
    return compose_multiplicative(powers) >= target


def saturation_diagnostic(n: int, p_max: float, target: float) -> bool:
    """True if even the most favourable ladder with these bounds falls short."""
    return (1.0 - (1.0 - p_max) ** n) < target


def min_rungs_for(target: float, p: float) -> Optional[int]:
    if target >= 1.0 or p <= 0.0:
        return None
    return math.ceil(math.log(1.0 - target) / math.log(1.0 - p))


# ===========================================================================
# Verdicts
# ===========================================================================
VERDICTS = ("reached", "short", "subfloor", "refused", "empty")


@dataclass(frozen=True)
class Verdict:
    """
    A verdict carries a label and a payload.  Only `reached` and `short`
    carry a gap; the rest carry the reason they fired.

    `subfloor` is the floor refusal: a declared rung power that would
    require closing the gap below the floor.  The name is taken from the
    existing honjo verdict label of the same meaning rather than inventing
    a new one.
    """
    label: str
    payload: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VERDICTS:
            raise ValueError(f"unknown verdict label: {self.label}")
        carries_gap = self.label in ("reached", "short")
        has_gap = "gap" in self.payload
        if carries_gap != has_gap:
            raise ValueError(
                f"label {self.label!r} and payload disagree on carrying a gap")


# ===========================================================================
# Operational semantics
# ===========================================================================
@dataclass
class Config:
    """<G, M, gap, e> -- graph, commitment counter, above-floor gap, expr."""
    graph: ContactGraph
    M: int
    gap: float
    expr: str
    trace: List[str] = field(default_factory=list)


class Machine:
    """
    Small-step evaluator.

      E-Probe   region changes, M unchanged, no residue.
      E-Commit  boundary committed, M += 1, residue >= floor.
      E-Climb   unfold one rung of a ladder.
      E-Halt    gap within epsilon, or no rungs left.
      E-Refuse  declared target unreachable -> subfloor verdict.
    """

    def __init__(self, graph: ContactGraph, epsilon: float = 1e-9) -> None:
        self.graph = graph
        self.epsilon = epsilon
        self.residues: List[float] = []

    def probe(self, cfg: Config) -> Config:
        cfg.trace.append("E-Probe")
        cfg.expr = "done"
        return cfg                       # M untouched, graph untouched

    def commit(self, cfg: Config, rung: Rung) -> Config:
        residue = self.graph.floor
        if residue < self.graph.floor:
            raise AssertionError("commitment below floor")
        self.residues.append(residue)
        cfg.M += 1
        cfg.gap *= (1.0 - rung.power)
        cfg.expr = "done"
        cfg.trace.append("E-Commit")
        return cfg

    def climb(self, cfg: Config, ladder: Ladder) -> Config:
        for rung in ladder.rungs:
            if cfg.gap <= self.epsilon:
                cfg.trace.append("E-Halt")
                cfg.expr = "Reached"
                return cfg
            cfg.trace.append("E-Climb")
            self.commit(cfg, rung)
        cfg.trace.append("E-Halt")
        cfg.expr = "Reached" if cfg.gap <= self.epsilon else "Short"
        return cfg

    def run(self, ladder: Ladder, gap0: float = 1.0) -> Config:
        cfg = Config(self.graph, 0, gap0, "climb")
        return self.climb(cfg, ladder)

    def run_verdict(self, ladder: Ladder, target: float,
                    gap0: float = 1.0) -> Verdict:
        """Execute and return a verdict rather than a raw configuration."""
        if not ladder.rungs:
            return Verdict("empty", {"reason": "ladder declares no rungs"})
        if not static_reachable(ladder.powers, target):
            return Verdict("subfloor", {
                "reason": "declared rungs cannot reach declared target",
                "best_possible": ladder.composite_power(),
                "target": target})
        cfg = self.run(ladder, gap0)
        achieved = 1.0 - cfg.gap
        return Verdict("reached" if achieved >= target - 1e-12 else "short",
                       {"gap": cfg.gap, "achieved": achieved, "M": cfg.M})


def random_ladder(n: int, rng: random.Random,
                  lo: float = 0.0, hi: float = 0.95) -> Ladder:
    return Ladder([Rung(rng.uniform(lo, hi)) for _ in range(n)])
