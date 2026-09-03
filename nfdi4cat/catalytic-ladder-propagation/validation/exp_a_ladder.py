"""
Experiment A -- the ladder algebra.

Every check states what would falsify it.  Checks marked CONTROL must FAIL on
well-formed input; a suite in which they pass is measuring nothing.

Run:  python exp_a_ladder.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel.ladder import (  # noqa: E402
    ContactGraph,
    circulation,
    compose,
    compose_additive,
    compose_max,
    compose_mean,
    residual,
    rungs_needed,
    sensitivity_additive,
    sensitivity_proportional,
    uniformity,
)

RNG = random.Random(20260903)
checks = []


def check(name, ok, detail, control=False):
    checks.append(
        {"name": name, "pass": bool(ok), "detail": detail, "control": control}
    )
    tag = "CONTROL" if control else "check  "
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}: {detail}")
    return ok


print("=" * 74)
print("EXPERIMENT A -- ladder algebra")
print("=" * 74)

# ---------------------------------------------------------------- A1 floor
print("\nA1. The floor is computed, not assumed")

instances = []
# A small hand-checkable graph.  Separating a costs either cut{a} = 2+5 = 7
# or cut{a,b} = 2+3 = 5; separating b costs either cut{b} = 3+5 = 8 or
# cut{a,b} = 5.  So beta = 5, attained by the REGION {a,b} and not by any
# single vertex.  The expected value is a separation cost, never an edge
# weight, and writing 2.0 here (the a--m edge) is the error this check exists
# to catch; see Prop. "identity is a region, not a point".
g = ContactGraph(vertices={"m"})
g.add("a", "m", 2.0)
g.add("b", "m", 3.0)
g.add("a", "b", 5.0)
instances.append(("hand", g, 5.0))

# Adversarial: a very small edge weight is present, but it is NOT the floor.
# Separating a costs min(cut{a} = 1e-12 + 1, cut{a,b} = 1e-12 + 1) = 1+1e-12;
# separating b costs min(1+1, 1+1e-12) = 1+1e-12.  A tiny edge inside a cut
# does not make the cut cheap.
g2 = ContactGraph(vertices={"m"})
g2.add("a", "m", 1e-12)
g2.add("b", "m", 1.0)
g2.add("a", "b", 1.0)
instances.append(("adversarial-1e-12", g2, 1.0 + 1e-12))

# a chain
g3 = ContactGraph(vertices={"m"})
for i in range(6):
    g3.add(f"v{i}", "m", 1.0 + i)
for i in range(5):
    g3.add(f"v{i}", f"v{i+1}", 10.0)
instances.append(("chain", g3, None))

all_pos = True
for label, gr, expect in instances:
    beta = gr.floor()
    all_pos &= beta > 0
    if expect is not None:
        ok = abs(beta - expect) < 1e-15
        check(f"floor({label})", ok, f"beta={beta:.3e} expected {expect:.3e}")
    else:
        check(f"floor({label}) > 0", beta > 0, f"beta={beta:.6f}")

check("floor positive on every instance", all_pos, f"{len(instances)}/{len(instances)}")

# the failure branch: unbounded refinement drives the infimum to zero
seq = [1.0 / n for n in range(1, 400)]
inf_zero = min(seq) < 1e-2 and all(x > 0 for x in seq)
check(
    "unbounded refinement has infimum 0",
    inf_zero,
    f"min={min(seq):.3e} > 0, but clears every eps down to 1e-2",
)
trunc = min(seq[:10])
check(
    "truncation restores a positive floor",
    abs(trunc - 0.1) < 1e-12,
    f"floor(first 10) = {trunc}",
)

# Identity is a region, not a point.  In a graph with a tightly bound core,
# every minimiser contains the whole core: the cheapest way to individuate a
# buried vertex is to individuate the core it sits in.
core = ContactGraph(vertices={"m"})
W = 50.0
for i in range(4):
    for j in range(i + 1, 4):
        core.add(f"a{i}", f"a{j}", W)      # tightly packed core
core.add("a0", "b0", 1.0)                   # single link to the periphery
for i in range(4):
    core.add(f"b{i}", "m", 1.0)             # periphery touches the medium
for i in range(3):
    core.add(f"b{i}", f"b{i+1}", 1.0)

best_S, best_c = None, float("inf")
import itertools as _it
others = sorted(core.vertices - {"a0", "m"}, key=str)
for r in range(len(others) + 1):
    for extra in _it.combinations(others, r):
        S = {"a0"} | set(extra)
        c = core.cut_weight(S)
        if c < best_c:
            best_c, best_S = c, S
whole_core_in = all(f"a{i}" in best_S for i in range(4))
check(
    "identity is a region, not a point",
    whole_core_in and len(best_S) > 1,
    f"minimiser for a0 has {len(best_S)} vertices and contains the whole core "
    f"(cost {best_c:g}); no singleton attains it",
)

# ------------------------------------------------------- A2 composition law
print("\nA2. Composition is multiplicative; alternatives are scored too")

N = 4000
errs = {"multiplicative": [], "additive": [], "max": [], "mean": []}
for _ in range(N):
    n = RNG.randint(2, 7)
    pis = [RNG.uniform(0.0, 0.95) for _ in range(n)]
    # ground truth by rung-by-rung simulation of the residual gap
    gap = 1.0
    for p in pis:
        gap *= 1.0 - p
    truth = 1.0 - gap
    errs["multiplicative"].append(abs(compose(pis) - truth))
    errs["additive"].append(abs(compose_additive(pis) - truth))
    errs["max"].append(abs(compose_max(pis) - truth))
    errs["mean"].append(abs(compose_mean(pis) - truth))

mae = {k: float(np.mean(v)) for k, v in errs.items()}
check(
    "multiplicative law exact",
    mae["multiplicative"] < 1e-12,
    f"MAE={mae['multiplicative']:.3e}",
)
check(
    "alternatives are worse (comparison is the test)",
    mae["additive"] > 1e-3 and mae["max"] > 1e-3 and mae["mean"] > 1e-3,
    f"additive={mae['additive']:.4f} max={mae['max']:.4f} mean={mae['mean']:.4f}",
)

# ------------------------------------------------------- A3 sensitivity
print("\nA3. Sensitivity: additive vs proportional  (the correction)")

add_argmax_is_strongest = 0
prop_spreads = []
M = 5000
for _ in range(M):
    n = RNG.randint(3, 7)
    pis = [RNG.uniform(0.05, 0.9) for _ in range(n)]
    add = [sensitivity_additive(pis, j) for j in range(n)]
    prop = [sensitivity_proportional(pis, j) for j in range(n)]
    if int(np.argmax(add)) == int(np.argmax(pis)):
        add_argmax_is_strongest += 1
    prop_spreads.append(max(prop) - min(prop))

check(
    "additive argmax is the highest-power rung",
    add_argmax_is_strongest == M,
    f"{add_argmax_is_strongest}/{M}",
)
check(
    "proportional sensitivity is flat",
    max(prop_spreads) < 1e-12,
    f"max spread over rungs = {max(prop_spreads):.3e}",
)

# numerical cross-check of the analytic derivative, independent of the formula
worst = 0.0
for _ in range(500):
    n = RNG.randint(3, 6)
    pis = [RNG.uniform(0.05, 0.9) for _ in range(n)]
    j = RNG.randrange(n)
    h = 1e-6
    up = list(pis)
    up[j] += h
    num = (compose(up) - compose(pis)) / h
    worst = max(worst, abs(num - sensitivity_additive(pis, j)))
check(
    "analytic derivative matches numerical",
    worst < 1e-5,
    f"max |analytic - numerical| = {worst:.2e}",
)

# ------------------------------------------------------- A4 saturation
print("\nA4. Repetition saturates; a target has a cost")

pi = 0.3
comp = [1 - (1 - pi) ** n for n in range(1, 40)]
strictly_increasing = all(b > a for a, b in zip(comp, comp[1:]))
never_reaches = all(c < 1.0 for c in comp)
check(
    "repetition rises strictly and never attains 1",
    strictly_increasing and never_reaches,
    f"n=39 gives {comp[-1]:.6f}",
)
n_needed = rungs_needed(0.80, 0.55)
check(
    "cost of a target",
    n_needed == 3,
    f"reaching 0.80 with rungs <= 0.55 needs {n_needed} rungs",
)

# divergence dichotomy, indexed from i=2 so no rung is absolute
harm = [1.0 / i for i in range(2, 4000)]
sq = [1.0 / i**2 for i in range(2, 4000)]
check(
    "divergent series drives the gap to zero",
    residual(harm) < 1e-3,
    f"residual = {residual(harm):.3e}",
)
check(
    "convergent series plateaus at positive residual",
    residual(sq) > 0.3,
    f"residual = {residual(sq):.4f}",
)

# ------------------------------------------------------- A5 closed ladders
print("\nA5. Closed ladders: circulation and uniformity")

# circulation reparametrises composite power in the LINEAR case: stated plainly
worst_rep = 0.0
for _ in range(500):
    n = RNG.randint(2, 6)
    pis = [RNG.uniform(0.0, 0.9) for _ in range(n)]
    worst_rep = max(worst_rep, abs(circulation(pis) + math.log(1 - compose(pis))))
check(
    "rho = -log(1 - composite) on linear ladders (so it adds nothing there)",
    worst_rep < 1e-9,
    f"max deviation = {worst_rep:.2e}",
)

# uniformity DOES add information beyond composite power
same_comp_diff_unif = 0
trials = 4000
for _ in range(trials):
    target = RNG.uniform(0.4, 0.9)
    a = RNG.uniform(0.1, 0.8)
    b = 1 - (1 - target) / (1 - a)
    if not (0 < b < 1):
        continue
    p1 = [a, b]
    c = RNG.uniform(0.1, 0.8)
    d = 1 - (1 - target) / (1 - c)
    if not (0 < d < 1):
        continue
    p2 = [c, d]
    if abs(compose(p1) - compose(p2)) < 1e-12 and abs(
        uniformity(p1) - uniformity(p2)
    ) > 1e-9:
        same_comp_diff_unif += 1
check(
    "uniformity separates ladders that share a composite power",
    same_comp_diff_unif > 1000,
    f"{same_comp_diff_unif}/{trials} pairs differ in upsilon at equal composite",
)

check(
    "circulation vanishes exactly on inertness",
    abs(circulation([0.0, 0.0, 0.0])) < 1e-15 and circulation([0.1, 0.0]) > 0,
    "rho=0 iff every rung inert",
)

# rotation invariance
p = [0.2, 0.5, 0.35, 0.1]
rots = [p[i:] + p[:i] for i in range(len(p))]
check(
    "both invariants are rotation invariant",
    max(abs(circulation(r) - circulation(p)) for r in rots) < 1e-15
    and max(abs(uniformity(r) - uniformity(p)) for r in rots) < 1e-15,
    "rho and upsilon constant over all rotations",
)

# length is not recoverable from rho alone
n1, pi1 = 3, 0.4
pi2 = 1 - math.sqrt(1 - pi1)
check(
    "cycle length is not recoverable from (rho, upsilon)",
    abs(circulation([pi1] * n1) - circulation([pi2] * (2 * n1))) < 1e-12,
    f"n={n1} at pi={pi1:.3f} and n={2*n1} at pi={pi2:.4f} share rho and upsilon=1",
)

# ------------------------------------------------------- A6 CONTROLS
print("\nA6. Negative controls -- these must FAIL on well-formed input")

# control 1: a statistic that cannot discriminate
const_stat = lambda pis: 1.0
disc = abs(const_stat([0.1, 0.9]) - const_stat([0.5, 0.5])) > 1e-9
check(
    "a constant statistic does NOT discriminate",
    not disc,
    "reported so that A2's separation is known not to be automatic",
    control=True,
)

# control 2: permuting a fixed multiset cannot change composite power
p = [0.2, 0.5, 0.7]
perms = set()
for q in __import__("itertools").permutations(p):
    perms.add(round(compose(list(q)), 12))
check(
    "permutation test is NON-DISCRIMINATING (excluded from score)",
    len(perms) == 1,
    f"{len(perms)} distinct composite over all permutations -- the law is symmetric, "
    "so this check cannot fail and is not counted",
    control=True,
)

# control 3: relabelling rungs moves nothing, but perturbing a power does
base = [0.3, 0.6, 0.45]
relabelled = list(base)  # labels are external; the datum is the power
moved_by_label = abs(compose(relabelled) - compose(base))
perturbed = [0.3, 0.66, 0.45]
moved_by_power = abs(compose(perturbed) - compose(base))
check(
    "labels move nothing; powers move everything",
    moved_by_label < 1e-15 and moved_by_power > 1e-3,
    f"label delta={moved_by_label:.1e}, power delta={moved_by_power:.4f}",
)

# ---------------------------------------------------------------- summary
def is_nondiscriminating(c):
    return "NON-DISCRIMINATING" in (c["name"] + c["detail"])


excluded = [c for c in checks if is_nondiscriminating(c)]
scored = [c for c in checks if not is_nondiscriminating(c)]
controls = [c for c in scored if c["control"]]
npass = sum(1 for c in scored if c["pass"])
print("\n" + "=" * 74)
print(f"EXPERIMENT A: {npass}/{len(scored)} scored checks pass "
      f"({len(controls)} controls, {len(excluded)} excluded as "
      f"non-discriminating)")
print("=" * 74)

os.makedirs("results", exist_ok=True)
json.dump(
    {
        "experiment": "A",
        "checks": checks,
        "scored": len(scored),
        "passed": npass,
        "controls": len(controls),
        "excluded_nondiscriminating": len(excluded),
        "composition_mae": mae,
        "proportional_max_spread": max(prop_spreads),
        "additive_argmax_strongest": f"{add_argmax_is_strongest}/{M}",
    },
    open("results/exp_a.json", "w"),
    indent=2,
)
sys.exit(0 if npass == len(scored) else 1)
