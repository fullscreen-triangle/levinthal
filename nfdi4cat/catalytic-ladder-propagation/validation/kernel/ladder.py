"""
Reference kernel for the catalytic ladder.

Everything the paper claims about the ladder algebra is computed here, from
the definitions given in the paper and nothing else.  No result in this file
depends on any external library beyond numpy, and none of it reads a network.

Sections map to the paper:
  contact graph, floor, cut          -> Sec. 3
  rung, power, composition           -> Sec. 4
  closed ladders                     -> Sec. 5
  medium, role, direction            -> Sec. 6
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Sec. 3 -- contact graph, separation cost, floor
# ----------------------------------------------------------------------------

MEDIUM = "m"


@dataclass
class ContactGraph:
    """Finite weighted graph with a distinguished medium vertex.

    Vertices are hashable labels.  ``w`` maps frozenset({u,v}) -> positive
    weight, read as the cost of separating u from v (never as a distance).
    """

    vertices: set
    w: dict = field(default_factory=dict)
    medium: str = MEDIUM

    def edge(self, u, v):
        return self.w.get(frozenset((u, v)), 0.0)

    def add(self, u, v, weight):
        assert weight > 0, "contact weights are strictly positive"
        self.vertices.add(u)
        self.vertices.add(v)
        self.w[frozenset((u, v))] = float(weight)

    def total(self):
        return sum(self.w.values())

    def neighbours(self, v):
        out = []
        for e in self.w:
            if v in e:
                (other,) = e - {v}
                out.append(other)
        return out

    # -- cuts -----------------------------------------------------------------

    def cut_weight(self, S):
        """Total weight of edges with exactly one endpoint in S."""
        S = set(S)
        tot = 0.0
        for e, wt in self.w.items():
            if len(e & S) == 1:
                tot += wt
        return tot

    def separation_cost(self, v, exhaustive=True):
        """min over S with v in S, medium not in S, of cut_weight(S).

        Exhaustive enumeration is used so that the quantity is the definition
        rather than an algorithm's output; the graphs here are small.
        """
        others = sorted(self.vertices - {v, self.medium}, key=str)
        best = self.cut_weight({v})
        if not exhaustive:
            return best
        for r in range(len(others) + 1):
            for extra in itertools.combinations(others, r):
                S = {v} | set(extra)
                c = self.cut_weight(S)
                if c < best:
                    best = c
        return best

    def floor(self):
        """beta := min over non-medium vertices of the separation cost."""
        vs = [v for v in self.vertices if v != self.medium]
        return min(self.separation_cost(v) for v in vs)


# ----------------------------------------------------------------------------
# Sec. 4 -- rungs, powers, linear ladders
# ----------------------------------------------------------------------------


def power_from_alignment(a_before, a_after, beta, omega):
    """pi = (a_before - a_after) / (a_before - beta/omega), clipped to [0,1]."""
    denom = a_before - beta / omega
    if denom <= 0:
        return 0.0
    pi = (a_before - a_after) / denom
    return float(min(1.0, max(0.0, pi)))


def compose(pis):
    """Composite power of a linear ladder: 1 - prod(1 - pi_i)."""
    prod = 1.0
    for p in pis:
        prod *= 1.0 - p
    return 1.0 - prod


def residual(pis):
    prod = 1.0
    for p in pis:
        prod *= 1.0 - p
    return prod


def compose_additive(pis):
    return min(1.0, sum(pis))


def compose_max(pis):
    return max(pis) if pis else 0.0


def compose_mean(pis):
    return sum(pis) / len(pis) if pis else 0.0


def sensitivity_additive(pis, j):
    """d pi(L) / d pi_j = prod_{i != j} (1 - pi_i)."""
    prod = 1.0
    for i, p in enumerate(pis):
        if i != j:
            prod *= 1.0 - p
    return prod


def sensitivity_proportional(pis, j):
    """Gain when rung j is improved by delta*(1 - pi_j): equals delta*P.

    Returned per unit delta, so the value is P = prod_i (1 - pi_i) for every j.
    """
    return sensitivity_additive(pis, j) * (1.0 - pis[j])


def rungs_needed(target, pi_max):
    """Least n with 1-(1-pi_max)^n >= target."""
    if pi_max <= 0:
        return math.inf
    if target >= 1:
        return math.inf
    return math.ceil(math.log(1 - target) / math.log(1 - pi_max))


# ----------------------------------------------------------------------------
# Sec. 5 -- closed ladders
# ----------------------------------------------------------------------------


def circulation(pis):
    """rho = -sum log(1 - pi_i).  Defined without a target."""
    return float(-sum(math.log(1.0 - p) for p in pis))


def uniformity(pis):
    """upsilon = max(0, 1 - sd/mean)."""
    a = np.asarray(pis, dtype=float)
    m = a.mean()
    if m == 0:
        return 0.0
    return float(max(0.0, 1.0 - a.std() / m))


# ----------------------------------------------------------------------------
# Sec. 6 -- medium, solvent role, direction
# ----------------------------------------------------------------------------


def medium_weight(mu, beta, tau, family="log"):
    """w(l, m) as a function of ambient occupancy mu.

    Four families are provided.  All are strictly decreasing in mu and bounded
    below by beta; the paper's structural theorems use only those two
    properties, and the suite re-verifies every one of them under all four.
    """
    if mu <= 0:
        return math.inf
    r = tau / mu
    if family == "log":
        return beta * (1.0 + math.log(1.0 + r))
    if family == "sqrt":
        return beta * (1.0 + math.sqrt(r) / (1.0 + math.sqrt(r)) * 3.0)
    if family == "rational":
        return beta * (1.0 + 3.0 * r / (1.0 + r))
    if family == "linear-cap":
        return beta * (1.0 + min(3.0, r))
    raise ValueError(family)


def solvent_role(rho_str, w_lm):
    """structural iff the boundary against the system >= boundary against medium."""
    return "structural" if rho_str >= w_lm else "bulk"


def medium_bias(initial_ids, terminal_ids, mu, beta, tau, family="log"):
    """Delta_m(C) = sum_{terminal} w(l_i,m) - sum_{initial} w(l_i,m)."""
    f = lambda i: medium_weight(mu[i], beta, tau, family)
    return sum(f(i) for i in terminal_ids) - sum(f(i) for i in initial_ids)


def direction_verdict(delta, beta):
    """Trichotomy on one inequality."""
    if delta > beta:
        return "forward"
    if delta < -beta:
        return "reverse"
    return "undirected"
