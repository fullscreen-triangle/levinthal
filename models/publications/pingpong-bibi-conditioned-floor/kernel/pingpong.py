"""
Ping-pong bi-bi as a two-half-reaction cut chain.

Why the existing `catalyze` cannot express this
-----------------------------------------------
`catalyze s in E yield r` assumes one substrate inside one enzyme, and the
sandbox interpreter commits a fixed two cuts for it. That shape fits the
cytochrome cycle, where the substrate enters, is turned over, and leaves
while the enzyme returns to its resting state unchanged.

Ping-pong bi-bi is a different topology:

  half 1:   E  + A  ->  E* + P        (first product leaves)
  half 2:   E* + B  ->  E  + Q        (second substrate then binds)

Three things break the single-substrate form:

  1. There is no ternary complex. E-A-B never exists. The first product
     departs before the second substrate arrives, so the two halves are
     sequential cuts through *different* enzyme states, not one cut.

  2. The enzyme is chemically modified between halves. E* is not E. For a
     transaminase, E carries pyridoxal 5'-phosphate (PLP) and E* carries
     pyridoxamine 5'-phosphate (PMP) -- the amine group is held *on the
     cofactor* across the gap between half-reactions.

  3. The cofactor is bound but is not a participant. PLP never leaves, so
     it appears in no reaction equation (Rhea confirms four participants
     for each transaminase, none of them PLP). It is part of the receiver,
     not part of the expression being evaluated.

Point 3 is the interesting one for the framework. It means `participant`
is not the same predicate as `leaf present in the complex`, and the
framework had no way to say so. Here the distinction is explicit: a
carrier leaf is committed once, at enzyme construction, and is *not*
re-cut per turnover, while participants are cut on every pass.

Everything below is computed from the residue chain. No baked oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conditioned_floor import Conditions, beta

# --- leaf classes, as in the five-class leaf algebra ---------------------
RES, COF, ELE, SOL, SUB = "res", "cof", "ele", "sol", "sub"


@dataclass
class Leaf:
    """An oscillator leaf: the arity-one cut."""

    name: str
    cls: str
    # A carrier is bound to the receiver rather than consumed by it. It is
    # cut once when the enzyme is built and never re-cut per turnover, which
    # is exactly why it appears in no reaction equation.
    carrier: bool = False


@dataclass
class CutEvent:
    """One committed cut: the residue it deposits is the next cut's cause."""

    label: str
    arity: int
    residue: float
    half: int  # 1 or 2; 0 for construction
    enzyme_state: str


@dataclass
class PingPongResult:
    events: list[CutEvent] = field(default_factory=list)
    M: int = 0  # committed-cut count, monotone
    participants: list[str] = field(default_factory=list)
    carriers: list[str] = field(default_factory=list)
    enzyme_states: list[str] = field(default_factory=list)
    closed: bool = False
    total_residue: float = 0.0
    floor: float = 0.0

    def residue_chain(self) -> list[float]:
        return [e.residue for e in self.events]

    def half(self, n: int) -> list[CutEvent]:
        return [e for e in self.events if e.half == n]


class PingPongCycle:
    """A ping-pong bi-bi cycle evaluated as a residue-chained cut sequence.

    The cut cost of a step is derived, not tabulated: each step's residue is
    the floor scaled by the number of boundaries the step actually commits,
    so the arithmetic is forced by the topology rather than chosen to match
    a published number.
    """

    def __init__(self, name: str, cond: Conditions | None = None):
        self.name = name
        self.cond = cond or Conditions()
        self.floor = beta(self.cond)
        self._carriers: list[Leaf] = []
        self._state = "E"
        self._events: list[CutEvent] = []
        self._M = 0

    # -- construction ----------------------------------------------------
    def bind_carrier(self, leaf: Leaf) -> "PingPongCycle":
        """Commit a carrier leaf once. Not a participant in any turnover."""
        leaf = Leaf(leaf.name, leaf.cls, carrier=True)
        self._carriers.append(leaf)
        # One arity-one cut, at construction (half 0).
        self._commit(f"bind carrier {leaf.name}", arity=1, boundaries=1, half=0)
        return self

    # -- the two halves --------------------------------------------------
    def half_reaction(
        self,
        substrate: Leaf,
        product: Leaf,
        new_state: str,
        group_transferred: str,
    ) -> "PingPongCycle":
        """One half-reaction: substrate in, product out, enzyme state changes.

        Commits three cuts, and the count is topological rather than chosen:

          1. substrate binding      -- arity 2 (substrate ~ active site)
          2. group transfer         -- arity 2 (substrate ~ carrier); this is
                                       the cut that modifies the enzyme, and
                                       it is why E* != E
          3. product release        -- arity 1 (product individuated from the
                                       complex and departs)

        No ternary complex is representable here: release happens inside the
        half, before the next half can begin.
        """
        half = 1 if self._state == "E" else 2
        prev_state = self._state

        self._commit(
            f"bind {substrate.name}", arity=2, boundaries=2, half=half
        )
        # The transfer cut crosses the substrate/carrier boundary for each
        # carrier present -- the group has to land somewhere.
        self._commit(
            f"transfer {group_transferred} -> carrier",
            arity=2,
            boundaries=1 + len(self._carriers),
            half=half,
        )
        self._state = new_state
        self._commit(
            f"release {product.name}", arity=1, boundaries=1, half=half
        )

        self._half_meta = getattr(self, "_half_meta", [])
        self._half_meta.append(
            {
                "half": half,
                "from_state": prev_state,
                "to_state": new_state,
                "substrate": substrate.name,
                "product": product.name,
                "group": group_transferred,
            }
        )
        return self

    def _commit(self, label: str, arity: int, boundaries: int, half: int) -> None:
        """Commit one cut. Residue = floor * boundaries: never below floor."""
        residue = self.floor * boundaries
        self._events.append(
            CutEvent(
                label=label,
                arity=arity,
                residue=residue,
                half=half,
                enzyme_state=self._state,
            )
        )
        self._M += 1

    # -- closure ---------------------------------------------------------
    def close(self) -> PingPongResult:
        """Finish the cycle and report.

        The cycle is *closed* iff the enzyme returned to its starting state.
        That is the ping-pong admissibility condition and it is the analogue
        of the seven-state orbit closing: the carrier must hand the group on
        and come back, or the enzyme is not a catalyst.
        """
        meta = getattr(self, "_half_meta", [])
        states = ["E"] + [m["to_state"] for m in meta]
        participants: list[str] = []
        for m in meta:
            participants.extend([m["substrate"], m["product"]])

        return PingPongResult(
            events=list(self._events),
            M=self._M,
            participants=participants,
            carriers=[c.name for c in self._carriers],
            enzyme_states=states,
            closed=self._state == "E" and len(meta) == 2,
            total_residue=sum(e.residue for e in self._events),
            floor=self.floor,
        )


def transaminase(
    donor: str,
    keto_product: str,
    acceptor: str = "2-oxoglutarate",
    amine_product: str = "L-glutamate",
    cond: Conditions | None = None,
) -> PingPongResult:
    """Build and run a PLP-dependent transaminase as ping-pong bi-bi.

    The two halves, for alanine transaminase (EC 2.6.1.2):

      half 1:  E-PLP  + L-alanine      -> E-PMP + pyruvate
      half 2:  E-PMP  + 2-oxoglutarate -> E-PLP + L-glutamate

    The amine group rides on the cofactor between the halves. PLP is bound
    throughout and is a participant in neither equation.
    """
    cyc = PingPongCycle(f"{donor} transaminase", cond=cond)
    cyc.bind_carrier(Leaf("pyridoxal 5'-phosphate", COF))
    cyc.half_reaction(
        substrate=Leaf(donor, SUB),
        product=Leaf(keto_product, SUB),
        new_state="E*",  # E-PMP: cofactor now carries the amine
        group_transferred="NH2",
    )
    cyc.half_reaction(
        substrate=Leaf(acceptor, SUB),
        product=Leaf(amine_product, SUB),
        new_state="E",  # back to E-PLP; the cycle closes
        group_transferred="NH2",
    )
    return cyc.close()
