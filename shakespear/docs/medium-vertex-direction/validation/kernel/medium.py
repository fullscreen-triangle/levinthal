"""
The medium vertex as a reservoir.

Implements Definitions 2.1-2.3, 3.1 and 4.1-4.2 of
`medium-vertex-direction.tex`. Nothing here is fitted to data: the
constitutive choice is eq. (1), and every result the suite checks depends
on it only through the two properties named in Remark 2.2 --- strictly
decreasing in mu, bounded below by beta.

The suite in ../run_all.py exploits that: `robustness_family()` supplies
alternative weight functions with the same two properties, so the
structural theorems can be re-checked under each. A theorem that only
holds for the logarithm is a theorem about the logarithm, not about the
medium, and the paper claims the latter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

# Leaf classes of the five-class algebra.
RES, COF, ELE, SOL, SUB = "res", "cof", "ele", "sol", "sub"

# The published R_bio floor estimate (expression-algebra paper). Used as
# the default ambient floor; every function takes it as a parameter so
# nothing depends on this constant globally.
BETA_DEFAULT = 3.7e-4


# =========================================================================
#  Leaves and the contact graph
# =========================================================================


@dataclass(frozen=True)
class Leaf:
    """An oscillator leaf: the arity-one cut."""

    name: str
    cls: str
    identity: str = ""

    def __post_init__(self) -> None:
        if not self.identity:
            object.__setattr__(self, "identity", self.name)


@dataclass
class Medium:
    """The medium vertex, as a reservoir (Definition 2.1).

    `mu` maps a chemical identity to its ambient occupancy; `tau` is the
    exchange scale. Only the ratio tau/mu is observable, which the suite
    checks directly (see `test_scale_invariance`).
    """

    mu: dict[str, float] = field(default_factory=dict)
    tau: float = 1.0e-3
    label: str = "medium"

    def occupancy(self, identity: str) -> float:
        """Ambient occupancy of an identity. Absent identities are 0."""
        return self.mu.get(identity, 0.0)

    def weight_to(
        self,
        leaf: Leaf,
        beta: float = BETA_DEFAULT,
        weight_fn: Callable[[float, float, float], float] | None = None,
    ) -> float:
        """w(leaf, m) --- eq. (1).

        Diverges when the identity is absent from the medium: a leaf of an
        identity the medium does not supply cannot be told apart from the
        medium cheaply, because there is nothing there to tell it apart
        from.
        """
        mu = self.occupancy(leaf.identity)
        if mu <= 0.0:
            return math.inf
        fn = weight_fn or weight_log
        return fn(mu, self.tau, beta)

    def is_ambient(self, identity: str, beta: float = BETA_DEFAULT) -> bool:
        """Definition 2.3: w < 2*beta, equivalently mu > tau/(e-1)."""
        return self.occupancy(identity) > self.tau / (math.e - 1.0)

    def is_depleted(self, identity: str) -> bool:
        """Definition 2.3: mu < tau."""
        mu = self.occupancy(identity)
        return mu < self.tau  # includes mu == 0 (absent => depleted)


# =========================================================================
#  Weight families (Remark 2.2)
# =========================================================================
#
#  Every function here must be: (a) >= beta everywhere, (b) strictly
#  decreasing in mu. The paper claims its theorems hold for ANY such
#  function; the suite tests that claim rather than assuming it.


def weight_log(mu: float, tau: float, beta: float) -> float:
    """eq. (1): beta * (1 + log(1 + tau/mu)). The paper's choice."""
    return beta * (1.0 + math.log(1.0 + tau / mu))


def weight_sqrt(mu: float, tau: float, beta: float) -> float:
    """Alternative: beta * (1 + sqrt(tau/mu)). Same two properties."""
    return beta * (1.0 + math.sqrt(tau / mu))


def weight_rational(mu: float, tau: float, beta: float) -> float:
    """Alternative: beta * (1 + tau/(tau+mu)). Bounded, unlike the others."""
    return beta * (1.0 + tau / (tau + mu))


def weight_linear_cap(mu: float, tau: float, beta: float) -> float:
    """Alternative: beta * (1 + min(1, tau/mu)). Piecewise, non-smooth."""
    return beta * (1.0 + min(1.0, tau / mu))


def robustness_family() -> dict[str, Callable[[float, float, float], float]]:
    """The weight functions the structural theorems are re-checked under."""
    return {
        "log (eq. 1)": weight_log,
        "sqrt": weight_sqrt,
        "rational": weight_rational,
        "linear-cap": weight_linear_cap,
    }


# =========================================================================
#  Solvent role (Definition 3.1, Theorem 3.2)
# =========================================================================

STRUCTURAL = "structural"
BULK = "bulk"


@dataclass
class Contact:
    """An edge between two leaves: the arity-two cut."""

    a: str  # leaf name
    b: str
    weight: float


class ContactGraph:
    """A contact graph carrying a medium.

    Only the operations the paper's theorems need: system-adjacency for a
    leaf, solvent role, and the medium bias of a chain.
    """

    def __init__(self, medium: Medium, beta: float = BETA_DEFAULT):
        self.medium = medium
        self.beta = beta
        self.leaves: dict[str, Leaf] = {}
        self.contacts: list[Contact] = []

    def add_leaf(self, leaf: Leaf) -> "ContactGraph":
        self.leaves[leaf.name] = leaf
        return self

    def add_contact(self, a: str, b: str, weight: float) -> "ContactGraph":
        if weight < self.beta:
            raise ValueError(
                f"contact {a}~{b} has weight {weight:.3e} < floor "
                f"{self.beta:.3e}: the sharp cut is not representable"
            )
        self.contacts.append(Contact(a, b, weight))
        return self

    def structural_residue(self, leaf_name: str) -> float:
        """Boundary the leaf maintains against the SYSTEM (not the medium)."""
        return sum(
            c.weight
            for c in self.contacts
            if c.a == leaf_name or c.b == leaf_name
        )

    def medium_weight(
        self, leaf_name: str, weight_fn: Callable[..., float] | None = None
    ) -> float:
        """Boundary the leaf maintains against the SURROUNDINGS."""
        return self.medium.weight_to(
            self.leaves[leaf_name], self.beta, weight_fn
        )

    def role(
        self, leaf_name: str, weight_fn: Callable[..., float] | None = None
    ) -> str:
        """Definition 3.1. Structural iff system boundary >= medium boundary."""
        return (
            STRUCTURAL
            if self.structural_residue(leaf_name)
            >= self.medium_weight(leaf_name, weight_fn)
            else BULK
        )

    def role_report(
        self, leaf_name: str, weight_fn: Callable[..., float] | None = None
    ) -> dict:
        """Corollary 3.4: the answer with its derivation attached."""
        s = self.structural_residue(leaf_name)
        m = self.medium_weight(leaf_name, weight_fn)
        return {
            "leaf": leaf_name,
            "structural_residue": s,
            "medium_weight": m,
            "role": STRUCTURAL if s >= m else BULK,
            "because": (
                f"structural residue {s:.4e} "
                f"{'>=' if s >= m else '<'} medium weight {m:.4e}"
            ),
        }


# =========================================================================
#  Direction (Definitions 4.2-4.3, Theorem 4.4)
# =========================================================================

FORWARD, REVERSE, UNDIRECTED = "forward", "reverse", "undirected"


@dataclass
class Chain:
    """A residue chain from an initial identity multiset to a terminal one."""

    name: str
    initial: list[str]  # chemical identities consumed
    terminal: list[str]  # chemical identities produced
    residues: list[float] = field(default_factory=list)

    def reversed_chain(self) -> "Chain":
        """C^R: the reversal. Same boundary, opposite endpoints."""
        return Chain(
            name=f"{self.name} (reversed)",
            initial=list(self.terminal),
            terminal=list(self.initial),
            residues=list(reversed(self.residues)),
        )

    def total_boundary(self) -> float:
        return sum(self.residues)

    def cut_count(self) -> int:
        return len(self.residues)


def medium_bias(
    chain: Chain,
    medium: Medium,
    beta: float = BETA_DEFAULT,
    weight_fn: Callable[..., float] | None = None,
) -> float:
    """eq. (2): sum of terminal medium-weights minus initial medium-weights.

    Positive when products are scarcer in the medium than reactants --- the
    medium is depleted at the product end, so the chain is pulled forward.
    """
    def w(identity: str) -> float:
        return medium.weight_to(Leaf(identity, SUB, identity), beta, weight_fn)

    return sum(w(i) for i in chain.terminal) - sum(w(i) for i in chain.initial)


def direction(
    chain: Chain,
    medium: Medium,
    beta: float = BETA_DEFAULT,
    weight_fn: Callable[..., float] | None = None,
) -> dict:
    """Theorem 4.4: the trichotomy on delta against the floor."""
    delta = medium_bias(chain, medium, beta, weight_fn)
    if delta > beta:
        verdict = FORWARD
    elif delta < -beta:
        verdict = REVERSE
    else:
        verdict = UNDIRECTED
    return {
        "chain": chain.name,
        "medium": medium.label,
        "delta": delta,
        "floor": beta,
        "delta_over_floor": (
            delta / beta if math.isfinite(delta) else math.inf
        ),
        "direction": verdict,
        "physiologically_admissible": verdict != UNDIRECTED,
        "case": {FORWARD: "(a)", REVERSE: "(b)", UNDIRECTED: "(c)"}[verdict],
    }


# =========================================================================
#  Refusals (Definitions 5.2-5.3)
# =========================================================================


class Refusal(Exception):
    """Raised when the framework declines to individuate or orient.

    A refusal is a scientific outcome, not an error condition: it is the
    framework saying that the distinction asked for is one it cannot draw.
    """

    def __init__(self, kind: str, subject: str, reason: str):
        self.kind = kind
        self.subject = subject
        self.reason = reason
        super().__init__(f"{kind} refused for {subject}: {reason}")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "subject": self.subject,
                "reason": self.reason}


def individuate_solvent(
    graph: ContactGraph,
    leaf_name: str,
    weight_fn: Callable[..., float] | None = None,
) -> dict:
    """Definition 5.2. A bulk solvent leaf is refused individuation.

    Returns the leaf's committed cut on success; raises Refusal on bulk.
    Critically, a refusal commits NO cut --- the clock must not advance for
    something the framework declined to individuate.
    """
    report = graph.role_report(leaf_name, weight_fn)
    if report["role"] == BULK:
        raise Refusal(
            kind="individuation",
            subject=leaf_name,
            reason=(
                f"role is bulk ({report['because']}): a fluctuation of the "
                "medium, not a part of the system"
            ),
        )
    return {**report, "cut_committed": True, "residue": report["structural_residue"]}


def orient(
    chain: Chain,
    medium: Medium,
    beta: float = BETA_DEFAULT,
    weight_fn: Callable[..., float] | None = None,
) -> dict:
    """Definition 5.3. An unbiased medium is refused orientation."""
    d = direction(chain, medium, beta, weight_fn)
    if d["direction"] == UNDIRECTED:
        raise Refusal(
            kind="orientation",
            subject=chain.name,
            reason=(
                f"|delta| = {abs(d['delta']):.4e} <= floor {beta:.4e}: the "
                "medium does not favour either direction by an amount the "
                "receiver can resolve"
            ),
        )
    return d
