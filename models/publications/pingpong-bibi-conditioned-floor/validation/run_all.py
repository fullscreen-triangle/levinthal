"""
Validation for the ping-pong bi-bi extension and the conditioned floor.

Every check here is falsifiable and computed. Where a check asserts a
number that came from outside this repository, the source is named in the
check's own output so a reader can go and disagree with it.

Reference data: the three transaminase reactions, with participant sets
verified against Rhea and ChEBI (nfdi4cat-sources/reference/
verified-identifiers.csv). The load-bearing external fact is that PLP is
a participant in none of them.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

KERNEL = Path(__file__).resolve().parent.parent / "kernel"
sys.path.insert(0, str(KERNEL))

from conditioned_floor import (  # noqa: E402
    Conditions,
    T_REF_K,
    beta,
    beta_breakdown,
    commensurable,
    distinguishable_at,
)
from pingpong import transaminase  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# Rhea-verified participant sets. PLP appears in none of them; that is the
# point of the fixture, not an omission.
REACTIONS = {
    "2.6.1.2": {
        "name": "alanine transaminase",
        "rhea": "RHEA:19453",
        "donor": "L-alanine",
        "keto_product": "pyruvate",
        "participants": {
            "L-alanine", "2-oxoglutarate", "pyruvate", "L-glutamate",
        },
    },
    "2.6.1.1": {
        "name": "aspartate transaminase",
        "rhea": "RHEA:21824",
        "donor": "L-aspartate",
        "keto_product": "oxaloacetate",
        "participants": {
            "L-aspartate", "2-oxoglutarate", "oxaloacetate", "L-glutamate",
        },
    },
    "2.6.1.3": {
        "name": "cysteine transaminase",
        "rhea": "RHEA:17441",
        "donor": "L-cysteine",
        "keto_product": "2-oxo-3-sulfanylpropanoate",
        "participants": {
            "L-cysteine", "2-oxoglutarate",
            "2-oxo-3-sulfanylpropanoate", "L-glutamate",
        },
    },
}

checks: list[dict] = []


def check(name: str, passed: bool, detail: str, *, expect_fail: bool = False) -> None:
    """Record a check.

    `expect_fail=True` marks a check whose failure is a *result* rather than
    a defect -- a place where the framework's own numbers refute a plausible
    claim. Such a check is reported as XFAIL and does not fail the suite,
    but if it ever starts passing it is reported as XPASS and DOES fail,
    because that means the finding no longer holds and the surrounding
    commentary has gone stale.
    """
    if expect_fail:
        verdict = "XPASS" if passed else "XFAIL"
    else:
        verdict = "PASS" if passed else "FAIL"
    checks.append({"check": name, "verdict": verdict, "detail": detail})


# =========================================================================
# 1. The floor reduces to the published estimate at reference conditions
# =========================================================================
ref = Conditions()
b_ref = beta(ref)
bd = beta_breakdown(ref)

# The sandbox and the plays both hard-code 3.7e-4 (RECEIVER_FLOOR in
# cytochrome/src/data/lessons.js). The conditioned floor must land on the
# same order of magnitude at reference conditions or it has changed the
# theory rather than parameterised it.
PUBLISHED_FLOOR = 3.7e-4
same_order = abs(math.log10(b_ref / PUBLISHED_FLOOR)) < 1.0
check(
    "floor at reference conditions reproduces the published order",
    same_order,
    f"beta(ref) = {b_ref:.4e} vs published {PUBLISHED_FLOOR:.4e}; "
    f"log10 ratio {math.log10(b_ref / PUBLISHED_FLOOR):+.3f}. "
    f"dominant term: {bd['dominant_term']}",
)

# =========================================================================
# 2. The floor is genuinely condition-dependent (the whole claim)
# =========================================================================
cold = Conditions(temperature_K=277.15)   # 4 C, on ice
warm = Conditions(temperature_K=310.15)   # 37 C, physiological
b_cold, b_warm = beta(cold), beta(warm)

# Sign of the dependence. This is necessary but NOT sufficient -- see the
# materiality check immediately below, which is the one that bites.
check(
    "floor rises with temperature (sign)",
    b_warm > b_cold,
    f"beta(4C) = {b_cold:.12e} < beta(37C) = {b_warm:.12e}. "
    "Q falls as thermal occupation rises, so the coarsest resolvable "
    "boundary grows. Sign only -- magnitude is checked separately.",
)

# MATERIALITY. A dependence that exists only in the 12th decimal place is
# not a dependence anyone can measure, and a check that merely asserts
# `>` would pass on floating-point noise. Demand that varying temperature
# across the full biochemical range moves beta by at least 1% -- i.e. that
# the condition-dependence is detectable at all.
#
# At depth d=9 this FAILS, and the failure is the finding: floor_conv =
# 6/3^9 = 3.05e-4 swamps the Q term by ~4000x, so beta is effectively
# condition-INDEPENDENT at the depth the address manifold uses. The
# conditioned floor only becomes real at shallow depth, where the
# combinatorial terms are small enough for physics to show through.
rel_swing = (b_warm - b_cold) / b_cold
check(
    "temperature dependence is material (>1%) at depth 9",
    rel_swing > 0.01,
    f"relative swing 4C->37C = {rel_swing:.3e} ({rel_swing * 100:.6f}%). "
    f"At d=9 floor_conv={bd['floor_conv']:.3e} dominates "
    f"floor_Q={bd['floor_Q']:.3e} by "
    f"{bd['floor_conv'] / bd['floor_Q']:.0f}x, so beta barely moves. "
    "This is the finding: at the depth the address manifold uses, the "
    "floor is effectively condition-INDEPENDENT.",
    expect_fail=True,
)

# The same test at shallow depth, where the combinatorial floor is small.
# d=2: floor_conv = 6/9 = 0.67 -- still large. The Q term only competes
# when it is itself large, i.e. at short integration times. Establish
# where the crossover actually is rather than asserting one.
crossover = None
for d in range(1, 16):
    bc = beta(cold, d=d)
    bw = beta(warm, d=d)
    if (bw - bc) / bc > 0.01:
        crossover = d
        break
check(
    "a depth exists where temperature dependence is material",
    crossover is not None,
    f"crossover depth = {crossover}. Above it the combinatorial terms "
    "dominate and the floor is effectively condition-independent; below "
    "it, conditions govern."
    if crossover
    else "no depth in 1..15 gives a >1% swing at T_int=1e-3s: the Q term "
         "never competes with the combinatorial terms at this integration "
         "time, so the conditioned floor is not observable here.",
)

# Integration time is the one knob that *lowers* the floor: average longer,
# resolve finer. If this were not so, the Allan-deviation form is wrong.
longer = Conditions(integration_time_s=1.0e-1)
check(
    "floor falls with longer integration (sign)",
    beta(longer) < b_ref,
    f"beta(T_int=1e-1) = {beta(longer):.12e} < beta(ref) = {b_ref:.12e}",
)

# Where the Q term CAN dominate: very short integration. sigma ~ 1/sqrt(T_int),
# so shrinking T_int inflates the Q floor without touching the others.
short = Conditions(integration_time_s=1.0e-12)
bd_short = beta_breakdown(short)
check(
    "at short integration the Q term dominates and conditions govern",
    bd_short["dominant_term"] == "Q",
    f"T_int=1e-12s: floor_Q={bd_short['floor_Q']:.3e} vs "
    f"floor_conv={bd_short['floor_conv']:.3e}; dominant="
    f"{bd_short['dominant_term']}. This is the regime the conditioned "
    "floor is about.",
)

short_cold = Conditions(temperature_K=277.15, integration_time_s=1.0e-12)
short_warm = Conditions(temperature_K=310.15, integration_time_s=1.0e-12)
swing_short = (beta(short_warm) - beta(short_cold)) / beta(short_cold)
check(
    "temperature dependence is material in the Q-dominated regime",
    swing_short > 0.01,
    f"relative swing 4C->37C at T_int=1e-12s = {swing_short:.3e} "
    f"({swing_short * 100:.2f}%), vs {rel_swing * 100:.6f}% at T_int=1e-3s. "
    "The conditioned floor is real, but only where Q dominates.",
)

# =========================================================================
# 3. Ping-pong topology: the three structural facts
# =========================================================================
for ec, rx in REACTIONS.items():
    res = transaminase(donor=rx["donor"], keto_product=rx["keto_product"])

    # 3a. The cycle closes: E -> E* -> E
    check(
        f"[{ec}] cycle closes through a modified enzyme state",
        res.closed and res.enzyme_states == ["E", "E*", "E"],
        f"{rx['name']}: states {' -> '.join(res.enzyme_states)}, "
        f"closed={res.closed}",
    )

    # 3b. Participants match Rhea exactly -- and PLP is NOT among them
    got = set(res.participants)
    check(
        f"[{ec}] participants match {rx['rhea']}",
        got == rx["participants"],
        f"computed {sorted(got)}; expected {sorted(rx['participants'])}",
    )
    check(
        f"[{ec}] PLP is a carrier, never a participant",
        "pyridoxal 5'-phosphate" in res.carriers
        and not any("pyridoxal" in p for p in res.participants),
        f"carriers={res.carriers}; no PLP in participants "
        f"(Rhea gives 4 participants, none of them PLP)",
    )

    # 3c. No ternary complex: the first product is released before the
    # second substrate binds. Check the event order directly.
    labels = [e.label for e in res.events]
    rel1 = next(i for i, l in enumerate(labels) if l.startswith("release"))
    bind2 = next(
        i for i, l in enumerate(labels)
        if l.startswith("bind") and rx["keto_product"] not in l
        and i > rel1
    )
    check(
        f"[{ec}] no ternary complex (product released before 2nd substrate)",
        rel1 < bind2,
        f"release of {rx['keto_product']} at event {rel1}, "
        f"second binding at event {bind2}",
    )

    # 3d. Every residue is at or above the floor -- thm:no-zero-value
    chain = res.residue_chain()
    check(
        f"[{ec}] every cut residue >= floor",
        all(r >= res.floor for r in chain),
        f"min residue {min(chain):.6e} vs floor {res.floor:.6e}; "
        f"M={res.M} cuts committed",
    )

    # 3e. The carrier is cut once, not once per half
    carrier_cuts = [e for e in res.events if "carrier" in e.label
                    and e.label.startswith("bind")]
    check(
        f"[{ec}] carrier committed once, not per turnover",
        len(carrier_cuts) == 1,
        f"{len(carrier_cuts)} carrier binding cut(s) across {res.M} events",
    )

# =========================================================================
# 4. Cross-condition comparability -- the operation the framework lacked
# =========================================================================
# Two measurements of the same quantity, taken at different temperatures.
# Their difference is smaller than the coarser floor, so they are the same
# measurement and any ranking between them is an artefact.
cmp_close = commensurable(0.500, cold, 0.500 + 0.4 * b_warm, warm)
check(
    "sub-floor difference across conditions is not commensurable",
    not cmp_close["commensurable"],
    f"delta {cmp_close['delta']:.3e} <= governing floor "
    f"{cmp_close['governing_floor']:.3e} (set by condition "
    f"'{cmp_close['limited_by']}'): the two values are one measurement",
)

cmp_far = commensurable(0.500, cold, 0.600, warm)
check(
    "supra-floor difference across conditions is commensurable",
    cmp_far["commensurable"],
    f"delta {cmp_far['delta']:.3e} > governing floor "
    f"{cmp_far['governing_floor']:.3e}",
)

# The coarser floor governs. This is the claim that makes reporting
# conditions non-optional: a precise measurement compared to a sloppy one
# inherits the sloppy resolution.
#
# Comparing 4C against 37C at T_int=1e-3 is useless here -- the two floors
# are equal to 11 decimal places (see the materiality finding above), so
# `max` picks arbitrarily and the check would pass on nothing. Use two
# conditions whose floors genuinely differ: a long integration (fine floor)
# against a short one (coarse floor).
fine = Conditions(integration_time_s=1.0e-1)
coarse = Conditions(integration_time_s=1.0e-15)  # femtosecond gate
b_fine, b_coarse = beta(fine), beta(coarse)
# Guard against the vacuity that bit the first version of this suite: if the
# two floors are equal, `max` picks arbitrarily and everything below passes
# on nothing. One order of magnitude is the bar -- enough that the two
# regimes are unambiguously distinct.
check(
    "the two floors being compared are actually different",
    b_coarse / b_fine > 10.0,
    f"beta(coarse, T_int=1e-15s) = {b_coarse:.3e} vs "
    f"beta(fine, T_int=1e-1s) = {b_fine:.3e}; ratio "
    f"{b_coarse / b_fine:.0f}x. Without this the governance check below "
    "would be vacuous -- an earlier version of this suite compared 4C to "
    "37C, whose floors are equal to 11 decimal places, and passed on noise.",
)

mixed = commensurable(0.500, fine, 0.500 + 2 * b_fine, coarse)
check(
    "the coarser floor governs comparability",
    mixed["governing_floor"] == b_coarse and mixed["limited_by"] == "b",
    f"governing floor {mixed['governing_floor']:.3e} = the COARSE one "
    f"({b_coarse:.3e}), not the fine one ({b_fine:.3e}); "
    f"limited_by='{mixed['limited_by']}'",
)
check(
    "a fine-floor difference is erased by a coarse-floor comparator",
    not mixed["commensurable"],
    f"delta {mixed['delta']:.3e} clears the fine floor "
    f"({b_fine:.3e}) but not the governing coarse floor "
    f"({b_coarse:.3e}): pairing a precise measurement with a sloppy one "
    "destroys the precision. This is why conditions must be reported.",
)

# =========================================================================
# 5. Screening: variants that are not distinguishable at the floor
# =========================================================================
# A plate of variant activities. Three are spaced well apart; four are
# clustered inside one floor of each other and must come back as one group.
b = beta(ref)
variants = [
    1.000,            # wild-type
    1.000 + 0.2 * b,  # +--- these four differ by less than the floor
    1.000 + 0.4 * b,  # |    from one another, so no screen at these
    1.000 + 0.6 * b,  # |    conditions can rank them
    1.000 + 0.8 * b,  # +---
    1.500,            # genuinely better
    0.400,            # genuinely worse
]
groups = distinguishable_at(variants, ref)
clustered = [g for g in groups if len(g) > 1]
check(
    "floor-indistinguishable variants are grouped, not ranked",
    len(groups) == 3 and any(len(g) == 5 for g in groups),
    f"{len(variants)} variants -> {len(groups)} distinguishable group(s), "
    f"sizes {sorted(len(g) for g in groups)}. "
    f"The 5-member group differs by < beta={b:.3e} and cannot be ranked.",
)
check(
    "genuinely separated variants stay separate",
    all(len(g) == 1 for g in groups if 5 in g or 6 in g),
    f"groups: {groups}",
)

# The same plate read at a genuinely coarser floor loses resolution. Using
# 37C would prove nothing (its floor is equal to 25C's at this depth), so
# read the plate at a short integration time instead, where the floor is
# ~1000x coarser and the collapse is real.
groups_coarse = distinguishable_at(variants, coarse)
# beta(coarse) = 7.4e-2, and the wild-type cluster sits within 1.0 +/- 8e-4
# while 1.5 and 0.4 are ~0.5 away. So the coarse floor should absorb the
# cluster but still separate the two genuine outliers: 3 groups, not 1.
# Asserting the exact partition rather than just `fewer groups` -- a check
# that only counts groups would pass on an unrelated collapse.
check(
    "a coarser floor still separates outliers but absorbs the cluster",
    len(groups_coarse) == 3 and sorted(len(g) for g in groups_coarse) == [1, 1, 5],
    f"at beta={b:.3e}: {len(groups)} groups "
    f"{sorted(len(g) for g in groups)}; at beta={b_coarse:.3e}: "
    f"{len(groups_coarse)} groups {sorted(len(g) for g in groups_coarse)}. "
    f"The 0.4/1.5 outliers survive a {b_coarse / b:.0f}x coarser floor "
    "because they are ~0.5 apart; the cluster was never resolvable.",
)

# Now a floor coarse enough to erase even the outliers. This is the check
# that shows resolution genuinely collapses rather than being clamped.
absurd = Conditions(integration_time_s=1.0e-18)
groups_absurd = distinguishable_at(variants, absurd)
check(
    "a sufficiently coarse floor erases all distinctions",
    len(groups_absurd) == 1,
    f"at beta={beta(absurd):.3e} (> the full 1.1 spread of the plate): "
    f"{len(groups_absurd)} group of {len(variants)} variants. Every "
    "variant is the same measurement; the plate carries no information.",
)

# Monotonicity: resolution can never increase as the floor coarsens.
check(
    "group count is monotone in the floor",
    len(groups_coarse) <= len(groups) <= len(distinguishable_at(variants, fine)),
    f"fine {len(distinguishable_at(variants, fine))} >= "
    f"ref {len(groups)} >= coarse {len(groups_coarse)}",
)

# =========================================================================
# Report
# =========================================================================
tally = {v: sum(1 for c in checks if c["verdict"] == v)
         for v in ("PASS", "FAIL", "XFAIL", "XPASS")}
total = len(checks)
# XFAIL is an accepted result; FAIL and XPASS both break the suite.
broken = tally["FAIL"] + tally["XPASS"]

RESULTS.mkdir(parents=True, exist_ok=True)
summary = {
    "module": "pingpong-bibi-conditioned-floor",
    "aggregate": {
        "checks": total,
        **{k.lower(): v for k, v in tally.items()},
        "verdict": "PASS" if broken == 0 else "FAIL",
    },
    "reference_conditions": str(ref),
    "beta_reference": b_ref,
    "beta_breakdown": bd,
    "findings": [
        "At categorical depth 9 the conversion floor (6/3^9 = 3.05e-4) "
        "dominates the oscillator floor by ~4100x, so beta is effectively "
        "condition-independent there. The conditioned floor only governs in "
        "the Q-dominated regime (short integration times).",
    ],
    "checks": checks,
}
(RESULTS / "_summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n{'=' * 72}")
print("  ping-pong bi-bi + conditioned floor — validation")
print(f"{'=' * 72}\n")
for c in checks:
    print(f"  [{c['verdict']:5}] {c['check']}")
    print(f"          {c['detail']}\n")
print(f"{'=' * 72}")
print(f"  {tally['PASS']} pass · {tally['XFAIL']} xfail (expected) · "
      f"{tally['FAIL']} fail · {tally['XPASS']} unexpected pass   "
      f"[{total} checks]")
if tally["XPASS"]:
    print("  XPASS: a documented finding no longer holds — update the "
          "commentary.")
print(f"{'=' * 72}\n")

sys.exit(0 if broken == 0 else 1)
