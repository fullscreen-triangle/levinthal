"""
Experiment 2 --- Direction is a property of the medium (Lemma 4.1,
Corollary 4.2, Theorem 4.4).

The load-bearing negative result is Lemma 4.1: a chain and its reversal
commit identical boundary, so no intrinsic predicate can orient a process.
That is checked first, because everything after it is only interesting if
it holds.

Then Theorem 4.4's trichotomy, including the negative control the paper
mandates in Sec. 5.3 item 2 (an unbiased medium must return UNDIRECTED)
and item 3 (the bias must exceed the floor, not merely be non-zero).

The three transaminase reactions are the fixtures. Participant sets are
Rhea-verified; the media are constructed, not measured, and the paper says
so (Sec. 6, "No experimental validation").
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))

from medium import (  # noqa: E402
    BETA_DEFAULT,
    FORWARD,
    REVERSE,
    UNDIRECTED,
    Chain,
    Medium,
    Refusal,
    direction,
    medium_bias,
    orient,
    robustness_family,
)

BETA = BETA_DEFAULT

# Rhea-verified participant sets. PLP is in none of them --- it is a
# carrier, not a participant (Theorem 6.3).
REACTIONS = {
    "2.6.1.2": {
        "name": "alanine transaminase",
        "rhea": "RHEA:19453",
        "initial": ["L-alanine", "2-oxoglutarate"],
        "terminal": ["pyruvate", "L-glutamate"],
    },
    "2.6.1.1": {
        "name": "aspartate transaminase",
        "rhea": "RHEA:21824",
        "initial": ["L-aspartate", "2-oxoglutarate"],
        "terminal": ["oxaloacetate", "L-glutamate"],
    },
    "2.6.1.3": {
        "name": "cysteine transaminase",
        "rhea": "RHEA:17441",
        "initial": ["L-cysteine", "2-oxoglutarate"],
        "terminal": ["2-oxo-3-sulfanylpropanoate", "L-glutamate"],
    },
}

ALL_IDENTITIES = sorted(
    {i for r in REACTIONS.values() for i in r["initial"] + r["terminal"]}
)


def chain_for(ec: str) -> Chain:
    r = REACTIONS[ec]
    # Residues are illustrative; Lemma 4.1 is about their SUM being
    # reversal-invariant, which holds for any values.
    return Chain(
        name=r["name"],
        initial=list(r["initial"]),
        terminal=list(r["terminal"]),
        residues=[2.0 * BETA, 3.0 * BETA, 1.5 * BETA, 2.5 * BETA],
    )


def uniform_medium(mu: float = 1.0e-3, tau: float = 1.0e-3) -> Medium:
    """Every identity at the same occupancy -> delta must be exactly 0."""
    return Medium(mu={i: mu for i in ALL_IDENTITIES}, tau=tau,
                  label="uniform (unbiased)")


def glutamate_depleted() -> Medium:
    """Downstream demand holds L-glutamate low; 2-oxoglutarate ambient."""
    mu = {i: 1.0e-3 for i in ALL_IDENTITIES}
    mu["2-oxoglutarate"] = 1.0e-1   # ambient
    mu["L-glutamate"] = 1.0e-7      # depleted
    return Medium(mu=mu, tau=1.0e-3, label="glutamate-depleted (cytosol)")


def oxoglutarate_depleted() -> Medium:
    """The mirror organism: nitrogen flows the other way."""
    mu = {i: 1.0e-3 for i in ALL_IDENTITIES}
    mu["2-oxoglutarate"] = 1.0e-7   # depleted
    mu["L-glutamate"] = 1.0e-1      # ambient
    return Medium(mu=mu, tau=1.0e-3, label="2-oxoglutarate-depleted")


def run() -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append(
            {"check": name, "verdict": "PASS" if passed else "FAIL",
             "detail": detail, **extra}
        )

    # -- 2.1 Lemma 4.1: reversal invariance -----------------------------
    # The negative result everything else rests on.
    lemma_violations = []
    for ec in REACTIONS:
        c = chain_for(ec)
        cr = c.reversed_chain()
        if not math.isclose(c.total_boundary(), cr.total_boundary()):
            lemma_violations.append({"ec": ec, "why": "boundary differs"})
        if c.cut_count() != cr.cut_count():
            lemma_violations.append({"ec": ec, "why": "cut count differs"})
        if sorted(c.residues) != sorted(cr.residues):
            lemma_violations.append({"ec": ec, "why": "residue multiset differs"})
    check(
        "Lemma 4.1: chain and reversal commit identical boundary",
        not lemma_violations,
        f"3 chains; total boundary, cut count and residue multiset all "
        f"invariant under reversal. {len(lemma_violations)} violation(s). "
        f"=> no intrinsic predicate can orient a process (Cor. 4.2)",
        violations=lemma_violations,
    )

    # -- 2.2 Corollary 4.2, operationally --------------------------------
    # If an intrinsic quantity COULD orient, some function of the residues
    # would differ between C and C^R. Enumerate the obvious candidates.
    intrinsic_candidates = {
        "sum": lambda c: sum(c.residues),
        "max": lambda c: max(c.residues),
        "min": lambda c: min(c.residues),
        "count": lambda c: float(len(c.residues)),
        "mean": lambda c: sum(c.residues) / len(c.residues),
        "range": lambda c: max(c.residues) - min(c.residues),
    }
    discriminating = []
    for ec in REACTIONS:
        c, cr = chain_for(ec), chain_for(ec).reversed_chain()
        for cname, fn in intrinsic_candidates.items():
            if not math.isclose(fn(c), fn(cr)):
                discriminating.append({"ec": ec, "candidate": cname})
    check(
        "Cor. 4.2: no intrinsic residue statistic distinguishes C from C^R",
        not discriminating,
        f"{len(intrinsic_candidates)} candidate statistics x 3 chains = "
        f"{len(intrinsic_candidates) * 3} tests; "
        f"{len(discriminating)} discriminated. Any non-zero count would "
        f"refute Cor. 4.2.",
        discriminating=discriminating,
    )

    # -- 2.3 NEGATIVE CONTROL (Sec. 5.3 item 2): unbiased => undirected --
    unbiased = uniform_medium()
    undirected_results = []
    for ec in REACTIONS:
        d = direction(chain_for(ec), unbiased, BETA)
        undirected_results.append({"ec": ec, "delta": d["delta"],
                                   "direction": d["direction"]})
    all_undirected = all(
        r["direction"] == UNDIRECTED for r in undirected_results
    )
    check(
        "NEGATIVE CONTROL: unbiased medium refuses to orient",
        all_undirected,
        f"uniform occupancy across all identities -> delta = 0 exactly "
        f"for all 3 chains; every verdict UNDIRECTED. Without this, "
        f"Thm 4.4(c) is an unreachable third case.",
        results=undirected_results,
    )

    # ...and `orient` must actually raise.
    raised = 0
    refusals = []
    for ec in REACTIONS:
        try:
            orient(chain_for(ec), unbiased, BETA)
        except Refusal as r:
            raised += 1
            refusals.append(r.as_dict())
    check(
        "NEGATIVE CONTROL: orient() raises Refusal on an unbiased medium",
        raised == len(REACTIONS),
        f"{raised}/{len(REACTIONS)} chains refused orientation",
        refusals=refusals[:1],
    )

    # -- 2.4 Theorem 4.4(a): depleted product end => forward -------------
    fwd = []
    for ec in REACTIONS:
        d = direction(chain_for(ec), glutamate_depleted(), BETA)
        fwd.append({"ec": ec, "delta_over_floor": d["delta_over_floor"],
                    "direction": d["direction"], "case": d["case"]})
    check(
        "Thm 4.4(a): glutamate-depleted medium orients forward",
        all(r["direction"] == FORWARD for r in fwd),
        f"L-glutamate depleted (mu=1e-7), 2-oxoglutarate ambient (mu=1e-1); "
        f"delta/floor = {[round(r['delta_over_floor'], 2) for r in fwd]}",
        results=fwd,
    )

    # -- 2.5 NEGATIVE CONTROL (Sec. 5.3 item 3): margin, not just non-zero
    # A test asserting delta != 0 passes on floating-point noise. Demand a
    # real margin above the floor.
    margins = [r["delta_over_floor"] for r in fwd]
    check(
        "NEGATIVE CONTROL: bias exceeds the floor with a margin (>2x)",
        all(m > 2.0 for m in margins),
        f"min delta/floor = {min(margins):.2f}. Asserting merely "
        f"delta != 0 would pass on noise; Sec. 5.3 item 3 requires a "
        f"margin, after an earlier suite of ours passed on a "
        f"self-comparison.",
        margins=margins,
    )

    # -- 2.6 Theorem 4.4(b) + Cor. 4.5: same chain, opposite directions --
    # The AlaA case. One identifier, two organisms, two directions.
    mirror = []
    for ec in REACTIONS:
        d_fwd = direction(chain_for(ec), glutamate_depleted(), BETA)
        d_rev = direction(chain_for(ec), oxoglutarate_depleted(), BETA)
        mirror.append({
            "ec": ec, "rhea": REACTIONS[ec]["rhea"],
            "glutamate_depleted": d_fwd["direction"],
            "oxoglutarate_depleted": d_rev["direction"],
            "delta_fwd": d_fwd["delta"], "delta_rev": d_rev["delta"],
        })
    flipped = all(
        m["glutamate_depleted"] == FORWARD
        and m["oxoglutarate_depleted"] == REVERSE
        for m in mirror
    )
    check(
        "Cor. 4.5: the SAME chain runs opposite ways in two media",
        flipped,
        "identical chain and identical Rhea identifier; only the medium "
        "differs. This is the AlaA case (P0A959 vs human ALT1): the "
        "shared identifier is correct because reaction identity is a "
        "property of the chain and direction is not.",
        results=mirror,
    )

    # -- 2.7 antisymmetry: delta(C^R) = -delta(C) ------------------------
    antisym_violations = []
    for fname, fn in robustness_family().items():
        for ec in REACTIONS:
            c = chain_for(ec)
            d1 = medium_bias(c, glutamate_depleted(), BETA, fn)
            d2 = medium_bias(c.reversed_chain(), glutamate_depleted(), BETA, fn)
            if not math.isclose(d1, -d2, rel_tol=1e-12, abs_tol=1e-18):
                antisym_violations.append(
                    {"weight_fn": fname, "ec": ec, "d1": d1, "d2": d2}
                )
    check(
        "delta is antisymmetric under reversal, for all weight fns",
        not antisym_violations,
        f"{len(robustness_family()) * len(REACTIONS)} combinations; "
        f"{len(antisym_violations)} violation(s). This is the asymmetry "
        f"Lemma 4.1 showed the chain itself cannot supply.",
        violations=antisym_violations,
    )

    # -- 2.8 the trichotomy is exhaustive and exclusive -------------------
    # Sweep the depletion ratio and confirm exactly one case fires each
    # time, and that all three cases are reached.
    # The sweep must be TWO-SIDED. Depleting a product drives delta up
    # without bound (the medium weight diverges as mu -> 0), but making a
    # product abundant only saves log(2)*beta ~ 0.69*beta, because the
    # OTHER product is still at baseline. So a sweep that only varies the
    # product end can never reach case (b) -- it is bounded below by
    # -0.693*beta > -beta. The first version of this check swept exactly
    # that way and reported the trichotomy broken; the sweep was
    # one-sided, not the theorem. Reaching (b) requires depleting a
    # REACTANT, which is what the mirror organism does.
    cases_seen: dict[str, int] = {FORWARD: 0, REVERSE: 0, UNDIRECTED: 0}
    exclusivity_violations = []
    sweep_points = []
    for exponent in range(-9, 10):
        mu = {i: 1.0e-3 for i in ALL_IDENTITIES}
        if exponent < 0:
            # deplete a product -> products dearer -> delta > 0 -> forward
            mu["L-glutamate"] = 1.0e-3 * (10.0**exponent)
        elif exponent > 0:
            # deplete a reactant -> reactants dearer -> delta < 0 -> reverse
            mu["2-oxoglutarate"] = 1.0e-3 * (10.0 ** -exponent)
        med = Medium(mu=mu, tau=1.0e-3, label=f"sweep 1e{exponent}")
        d = direction(chain_for("2.6.1.2"), med, BETA)
        cases_seen[d["direction"]] += 1
        sweep_points.append({"exponent": exponent,
                             "delta_over_floor": d["delta_over_floor"],
                             "direction": d["direction"]})
        n_true = sum([
            d["delta"] > BETA,
            d["delta"] < -BETA,
            abs(d["delta"]) <= BETA,
        ])
        if n_true != 1:
            exclusivity_violations.append({"exponent": exponent,
                                           "n_cases_true": n_true})
    check(
        "Thm 4.4: trichotomy is exhaustive and mutually exclusive",
        not exclusivity_violations and all(v > 0 for v in cases_seen.values()),
        f"19-point two-sided sweep (deplete product / deplete reactant); "
        f"cases reached: {cases_seen}; {len(exclusivity_violations)} "
        f"point(s) where the three conditions were not exactly-one-true. "
        f"All three cases must be REACHED or the trichotomy is a "
        f"dichotomy.",
        cases_seen=cases_seen,
        violations=exclusivity_violations,
        sweep=sweep_points,
    )

    # -- 2.9 the one-sided bound, recorded as a finding -------------------
    # Making a single product arbitrarily abundant cannot flip the
    # direction: the saving is bounded by log(2)*beta because the other
    # product remains at baseline. This is a real, checkable property of
    # eq. (1) and it is why 2.8 had to be two-sided.
    mu_sat = {i: 1.0e-3 for i in ALL_IDENTITIES}
    mu_sat["L-glutamate"] = 1.0e30  # effectively infinite
    d_sat = direction(chain_for("2.6.1.2"),
                      Medium(mu=mu_sat, tau=1.0e-3), BETA)
    bound = -math.log(2.0)
    check(
        "saturating one product cannot reverse a chain (bounded by log 2)",
        math.isclose(d_sat["delta_over_floor"], bound, rel_tol=1e-6)
        and d_sat["direction"] == UNDIRECTED,
        f"L-glutamate at mu=1e30: delta/floor -> "
        f"{d_sat['delta_over_floor']:.4f}, bound = -log(2) = {bound:.4f}; "
        f"verdict {d_sat['direction']}. Reversal requires depleting a "
        f"reactant, not flooding a product --- an asymmetry of eq. (1) "
        f"worth stating.",
        delta_over_floor=d_sat["delta_over_floor"],
        theoretical_bound=bound,
    )

    passed = sum(1 for c in checks if c["verdict"] == "PASS")
    return {
        "experiment": "exp02_direction",
        "claim": "Lemma 4.1 / Theorem 4.4 --- cut structure cannot orient "
                 "a process; the medium can",
        "aggregate": {
            "checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "verdict": "PASS" if passed == len(checks) else "FAIL",
        },
        "beta": BETA,
        "reactions": {k: v["rhea"] for k, v in REACTIONS.items()},
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
