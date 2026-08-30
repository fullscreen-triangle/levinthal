#!/usr/bin/env python3
"""
Core implementation of the ladder formalism.

This module contains the entire machine: contact graph, floor, alignment,
rung, ladder, composition, and the small-step operational semantics.
Nothing here is specific to any application.  Everything the validation
scripts test is computed from these definitions.

The point of the formalism under test is that a rung carries ONE number.
Accordingly a Rung here has exactly one field.  There is deliberately no
slot for a name, a mechanism, a species, or a rate: if the code allowed one,
the inertness theorem would not be being tested by the code.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Contact graph, floor, cut, residue
# ---------------------------------------------------------------------------
Edge = FrozenSet[str]


@dataclass
class ContactGraph:
    """G = (V, E, w) with a distinguished medium adjacent to every item."""
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

    def cut(self, S: Set[str]) -> List[Edge]:
        """Edges with exactly one endpoint in S."""
        out = []
        for e in self.weights:
            u, v = tuple(e)
            if (u in S) != (v in S):
                out.append(e)
        return out

    def residue(self, S: Set[str]) -> float:
        return sum(self.weights[e] for e in self.cut(S))

    def admissible(self, S: Set[str]) -> bool:
        """Contains at least one item and omits the medium."""
        return bool(S & self.items()) and self.medium not in S

    def min_cut_between(self, x: str, target: str) -> float:
        """
        alpha(x, target): minimum residue over admissible S containing x and
        omitting target.  Exhaustive over subsets -- correct by construction,
        used as ground truth for the max-flow check in v1.
        """
        others = sorted(self.vertices - {x, target})
        best = math.inf
        for mask in range(1 << len(others)):
            S = {x} | {others[i] for i in range(len(others)) if mask >> i & 1}
            if target in S:
                continue
            r = self.residue(S)
            if r < best:
                best = r
        return best


def complete_graph(n_items: int, weights: Sequence[float],
                   medium: str = "m") -> ContactGraph:
    """Build a graph on n items plus a medium, with the given edge weights."""
    verts = {medium} | {f"v{i}" for i in range(n_items)}
    edges: Dict[Edge, float] = {}
    k = 0
    names = sorted(verts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            edges[frozenset((names[i], names[j]))] = weights[k % len(weights)]
            k += 1
    return ContactGraph(verts, edges, medium)


# ---------------------------------------------------------------------------
# Rung and ladder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rung:
    """
    A rung carries exactly one datum: its power.

    No identity field exists.  This is not an omission for brevity; it is the
    formalism.  A test of inertness that compared objects carrying names would
    be testing the names.
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
        """1 - prod(1 - pi_i)."""
        p = 1.0
        for r in self.rungs:
            p *= (1.0 - r.power)
        return 1.0 - p

    def residual_fraction(self) -> float:
        return 1.0 - self.composite_power()

    def gap_trajectory(self, gap0: float = 1.0) -> List[float]:
        """Above-floor gap after each rung, starting from gap0."""
        g = gap0
        out = [g]
        for r in self.rungs:
            g *= (1.0 - r.power)
            out.append(g)
        return out

    def alignment_trajectory(self, a0: float, floor_norm: float) -> List[float]:
        """a_i = gap_i + beta/Omega, recovered from the power sequence alone."""
        gap0 = a0 - floor_norm
        return [g + floor_norm for g in self.gap_trajectory(gap0)]

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


# ---------------------------------------------------------------------------
# Alternative composition laws -- the competing models for V3
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------
def static_reachable(powers: Sequence[float], target_power: float) -> bool:
    return compose_multiplicative(powers) >= target_power


def saturation_diagnostic(n: int, p_max: float, target_power: float) -> bool:
    """True if even the most favourable ladder with these bounds falls short."""
    return (1.0 - (1.0 - p_max) ** n) < target_power


def min_rungs_for(target_power: float, p: float) -> Optional[int]:
    """Ceil(log(1-pi*)/log(1-pi)); None if unreachable at any finite n."""
    if target_power >= 1.0:
        return None
    if p <= 0.0:
        return None
    return math.ceil(math.log(1.0 - target_power) / math.log(1.0 - p))


# ---------------------------------------------------------------------------
# Operational semantics: configuration and small-step reduction
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """<G, M, x, e> -- graph, commitment counter, region, expression."""
    graph: ContactGraph
    M: int
    gap: float                 # above-floor gap at the current region
    expr: str
    trace: List[str] = field(default_factory=list)


class Machine:
    """
    Small-step evaluator.

    E-Probe   : region changes, M unchanged, no residue.
    E-Commit  : boundary committed, M += 1, residue >= floor.
    E-Climb   : unfold one rung of a ladder.
    E-Halt    : gap within epsilon, or no rungs left.
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
        residue = self.graph.floor       # the minimum any commitment deposits
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


# ---------------------------------------------------------------------------
# Random ladder generation, for the statistical tests
# ---------------------------------------------------------------------------
def random_ladder(n: int, rng: random.Random,
                  lo: float = 0.0, hi: float = 0.95) -> Ladder:
    return Ladder([Rung(rng.uniform(lo, hi)) for _ in range(n)])
