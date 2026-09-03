"""
Reference kernel for the plan language: sources, capabilities, verdicts.

This implements Sec. 7 and Sec. 8 of the paper.  A plan is a sequence of
steps; each step names a source, declares the capabilities it requires, and
returns a verdict.  Nothing here reaches a network: sources resolve against
fixtures so that every reported number is reproducible offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    """Six outcomes, pairwise distinguishable.

    ANSWER    a result set, certified non-empty or certified empty
    EMPTY     the question is well posed and the answer is provably empty
    UNEXPRESSED  the question cannot be stated in the source's model
    UNSUPPORTED  statable in the model, not lowerable by this compiler
    STARVED   this step failed because an earlier step under-retrieved
    EXHAUSTED the budget ran out before the step completed
    """

    ANSWER = "answer"
    EMPTY = "empty"
    UNEXPRESSED = "unexpressed"
    UNSUPPORTED = "unsupported"
    STARVED = "starved"
    EXHAUSTED = "exhausted"


#: Verdicts that may never carry a non-empty payload (non-degeneracy).
NON_PAYLOAD = {
    Verdict.UNEXPRESSED,
    Verdict.UNSUPPORTED,
    Verdict.STARVED,
    Verdict.EXHAUSTED,
    Verdict.EMPTY,
}


@dataclass
class Result:
    verdict: Verdict
    payload: list = field(default_factory=list)
    blame: str | None = None          # for STARVED: the predecessor at fault
    reason: str = ""
    unblock: str = ""                 # what would have to change

    def __post_init__(self):
        # Non-degeneracy: only ANSWER may carry a non-empty payload.
        if self.verdict in NON_PAYLOAD and self.payload:
            raise ValueError(
                f"non-degeneracy violated: {self.verdict} carries a payload"
            )
        if self.verdict is Verdict.STARVED and self.blame is None:
            raise ValueError("a starved step must name its predecessor")


@dataclass
class Source:
    """A backend: an endpoint, a declared capability set, an extractor."""

    name: str
    capabilities: set
    records: dict = field(default_factory=dict)

    def supports(self, required: set):
        return required <= self.capabilities


@dataclass
class Step:
    name: str
    source: str
    requires: set
    fn: object = None
    depends_on: str | None = None


class Plan:
    def __init__(self, sources: dict):
        self.sources = sources
        self.steps: list[Step] = []

    def step(self, name, source, requires, fn=None, depends_on=None):
        self.steps.append(Step(name, source, set(requires), fn, depends_on))
        return self

    # -- static analysis ------------------------------------------------------

    def static_check(self):
        """Decide compilability before any request is issued.

        Returns a dict step-name -> None (ok) or a Result explaining refusal.
        """
        out = {}
        for st in self.steps:
            src = self.sources.get(st.source)
            if src is None:
                out[st.name] = Result(
                    Verdict.UNSUPPORTED,
                    reason=f"no source named {st.source!r}",
                    unblock="register an adapter for this source",
                )
                continue
            missing = st.requires - src.capabilities
            if missing:
                out[st.name] = Result(
                    Verdict.UNEXPRESSED,
                    reason=(
                        f"{src.name} cannot state "
                        + ", ".join(sorted(missing))
                    ),
                    unblock=(
                        "obtain these features from another source, or compute "
                        "them from a retrieved attribute"
                    ),
                )
            else:
                out[st.name] = None
        return out

    # -- execution ------------------------------------------------------------

    def run(self, budget=None):
        static = self.static_check()
        results = {}
        spent = 0
        for st in self.steps:
            if static[st.name] is not None:
                results[st.name] = static[st.name]
                continue
            if budget is not None and spent >= budget:
                results[st.name] = Result(
                    Verdict.EXHAUSTED,
                    reason="budget spent before this step",
                    unblock=f"raise budget above {spent}",
                )
                continue
            # starvation: a predecessor answered, but with nothing to work on
            if st.depends_on is not None:
                prev = results.get(st.depends_on)
                if prev is None or prev.verdict is not Verdict.ANSWER or not prev.payload:
                    results[st.name] = Result(
                        Verdict.STARVED,
                        blame=st.depends_on,
                        reason=f"step {st.depends_on!r} supplied no bindings",
                        unblock=f"widen {st.depends_on!r}",
                    )
                    continue
            src = self.sources[st.source]
            payload = st.fn(src, results) if st.fn else []
            spent += 1
            if payload:
                results[st.name] = Result(Verdict.ANSWER, payload=payload)
            else:
                results[st.name] = Result(
                    Verdict.EMPTY,
                    reason="source certified an empty extension",
                )
        return results
