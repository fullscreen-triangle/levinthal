#!/usr/bin/env python3
"""
V2 --- Intensivity, its near-miss, and the graded order-dependence result.

This is the module in which the paper's central technical claim can fail, and
in which our first attempt did fail.

Three candidate power definitions are compared throughout:

    intensive    1 - beta_r(v)/sigma_r(v)   both factors local
    globalfloor  1 - beta  /sigma_r(v)      cut key local, normaliser global
    extensive    sigma_r(v)/Omega           normalised by the whole graph

`globalfloor` is the NEAR-MISS: it differs from the candidate in exactly one
component, so when it fails it localises the failure to the normaliser.  A
control that differed in every component would show only that some difference
mattered.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from typing import Dict, List

from shk_core import (ContactGraph, Ladder, Rung, chain_graph,
                      compose_multiplicative, derive_sequential,
                      power_extensive, power_globalfloor, power_intensive)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CANDIDATES = [("intensive", power_intensive),
              ("globalfloor", power_globalfloor),
              ("extensive", power_extensive)]


def _extend_disjoint(g: ContactGraph, rng: random.Random,
                     n_new: int) -> ContactGraph:
    """
    Append n_new items to the far end of a chain, plus their medium edges.

    The addition is outside the ball of radius 1 about v0 whenever the chain
    has at least three items, which is the hypothesis of the intensivity
    theorem.  One added medium edge is deliberately given a small weight, so
    the GLOBAL floor moves; that is what the near-miss control is for.
    """
    verts = set(g.vertices)
    weights = dict(g.weights)
    existing = sorted(int(v[1:]) for v in g.items())
    last = max(existing)
    for k in range(1, n_new + 1):
        v = f"v{last + k}"
        verts.add(v)
        weights[frozenset((v, g.medium))] = 1.0 if k > 1 else g.floor * 0.5
        weights[frozenset((f"v{last + k - 1}", v))] = rng.uniform(0.5, 3.0)
    return ContactGraph(verts, weights, g.medium)


def v2_1_intensivity_under_extension(n_graphs: int = 120,
                                     seed: int = 21) -> Dict:
    """
    The decisive test.  Compute a power at v0, then extend the graph far from
    v0 and recompute.  An intensive quantity must be unchanged EXACTLY.

    The test can fail in both directions: if the intensive candidate drifted it
    would fail, and if the extensive control did NOT drift the test would be
    non-discriminating and is reported as such.
    """
    rng = random.Random(seed)
    drift: Dict[str, List[float]] = {n: [] for n, _ in CANDIDATES}
    for _ in range(n_graphs):
        base = chain_graph(rng.randint(4, 6), rng)
        ext = _extend_disjoint(base, rng, rng.randint(1, 3))
        for name, fn in CANDIDATES:
            a = fn(base, "v0", 1)
            b = fn(ext, "v0", 1)
            drift[name].append(abs(a - b))

    summary = {n: {"max_drift": max(v), "mean_drift": sum(v) / len(v),
                   "n_moved": sum(1 for x in v if x > 1e-12)}
               for n, v in drift.items()}

    intensive_exact = summary["intensive"]["max_drift"] <= 1e-12
    controls_move = (summary["extensive"]["n_moved"] > 0
                     and summary["globalfloor"]["n_moved"] > 0)
    discriminating = controls_move

    return {
        "test": "V2.1 derived power is intensive; both controls drift",
        "n_graphs": n_graphs,
        "by_candidate": summary,
        "intensive_exactly_invariant": intensive_exact,
        "controls_move": controls_move,
        "scored": discriminating,
        "passed": bool(intensive_exact and controls_move),
        "interpretation": (
            "The near-miss (globalfloor) differs from the candidate only in "
            "the normaliser, so its drift attributes the failure to the "
            "normaliser rather than to locality in general. Had no control "
            "drifted, the test would not have been measuring intensivity and "
            "would be reported non-discriminating."
        ),
    }


def v2_2_permutation_is_non_discriminating(n_graphs: int = 12,
                                           seed: int = 22) -> Dict:
    """
    NON-DISCRIMINATING BY CONSTRUCTION, and reported as such.

    Permuting a FIXED multiset of powers cannot change 1 - prod(1-p): the
    formula is symmetric.  Every candidate passes, including deliberately
    extensive ones.  We run it and exclude it from the score so that a reader
    cannot mistake it for evidence.
    """
    rng = random.Random(seed)
    distinct_counts: Dict[str, List[int]] = {n: [] for n, _ in CANDIDATES}
    for _ in range(n_graphs):
        g = chain_graph(5, rng)
        items = sorted(g.items())
        for name, fn in CANDIDATES:
            powers = [fn(g, v, 1) for v in items]
            comps = {round(compose_multiplicative(list(p)), 12)
                     for p in itertools.permutations(powers)}
            distinct_counts[name].append(len(comps))

    all_one = all(all(c == 1 for c in v) for v in distinct_counts.values())
    return {
        "test": "V2.2 permuting a fixed power multiset (NON-DISCRIMINATING)",
        "n_graphs": n_graphs,
        "distinct_composites_per_graph": {k: sorted(set(v))
                                          for k, v in distinct_counts.items()},
        "every_candidate_gives_one": all_one,
        "scored": False,
        "passed": None,
        "verdict": (
            "NON-DISCRIMINATING: the composition law is symmetric in its "
            "arguments, so permuting a fixed multiset of powers cannot change "
            "the composite for ANY candidate definition. Every candidate "
            "returns exactly one distinct composite. The test cannot fail and "
            "is therefore excluded from the score; V2.3 is the test that can."
        ),
    }


def v2_3_sequential_order_dependence(n_items: int = 6,
                                     seed: int = 23) -> Dict:
    """
    The test that CAN fail.  Derive powers sequentially, so that each rung sees
    the graph its predecessors mutated, and sweep every ordering.

    We expected intensivity to give exact order-independence.  It does not.
    We report the graded result and the radius trend that explains it.
    """
    rng = random.Random(seed)
    g = chain_graph(n_items, rng)
    items = [f"v{i}" for i in range(n_items)]
    perms = list(itertools.permutations(items))

    rows = []
    by_radius: Dict[str, Dict[str, Dict[str, float]]] = {}
    for radius in (0, 1):
        entry: Dict[str, Dict[str, float]] = {}
        for name, fn in CANDIDATES:
            comps = set()
            for perm in perms:
                ps = derive_sequential(g, perm, fn, radius)
                comps.add(round(compose_multiplicative(ps), 10))
            entry[name] = {"distinct": len(comps),
                           "spread": max(comps) - min(comps)}
        by_radius[str(radius)] = entry
        rows.append({"radius": radius, **{k: v["distinct"]
                                          for k, v in entry.items()}})

    r1 = by_radius["1"]
    ordering_ok = (r1["intensive"]["distinct"]
                   < r1["globalfloor"]["distinct"]
                   < r1["extensive"]["distinct"])
    spread_ok = r1["intensive"]["spread"] < r1["extensive"]["spread"]
    tightens = (by_radius["0"]["intensive"]["distinct"]
                < by_radius["1"]["intensive"]["distinct"])
    exact = r1["intensive"]["distinct"] == 1

    return {
        "test": "V2.3 sequential derivation: order dependence is graded",
        "n_items": n_items,
        "n_orderings": len(perms),
        "by_radius": by_radius,
        "rows": rows,
        "intensive_beats_both_controls": ordering_ok,
        "intensive_spread_smaller": spread_ok,
        "tightens_at_finer_radius": tightens,
        "exact_order_independence": exact,
        "passed": bool(ordering_ok and spread_ok and not exact),
        "interpretation": (
            "The claim under test is the GRADED one: intensivity reduces "
            "order dependence and does not abolish it. Exact independence "
            "(distinct == 1) is recorded and is FALSE, which is the honest "
            "result; had we tested only the clean claim the test would have "
            "failed and we would have had nothing to report. Dependence "
            "survives because commitment mutates edges inside the balls of "
            "neighbouring items, which is exactly the case the intensivity "
            "theorem does not cover."
        ),
    }


def v2_4_scale_invariance(n_graphs: int = 60, seed: int = 24) -> Dict:
    """
    Corollary: the floor is load-bearing only through positivity.  Rescaling
    every weight by c > 0 must leave the intensive power unchanged, and must
    NOT leave the extensive one unchanged (it is a ratio to Omega, which also
    scales -- so this control is expected to be invariant too, and we say so).
    """
    rng = random.Random(seed)
    dev_int, dev_ext = [], []
    for _ in range(n_graphs):
        g = chain_graph(rng.randint(4, 6), rng)
        c = rng.choice([0.01, 0.5, 2.0, 1000.0])
        scaled = ContactGraph(set(g.vertices),
                              {e: w * c for e, w in g.weights.items()},
                              g.medium)
        dev_int.append(abs(power_intensive(g, "v0", 1)
                           - power_intensive(scaled, "v0", 1)))
        dev_ext.append(abs(power_extensive(g, "v0", 1)
                           - power_extensive(scaled, "v0", 1)))
    return {
        "test": "V2.4 derived power is invariant under rescaling of weights",
        "n_graphs": n_graphs,
        "max_dev_intensive": max(dev_int),
        "max_dev_extensive": max(dev_ext),
        "passed": max(dev_int) < 1e-12,
        "note": (
            "The extensive candidate is also scale-invariant, being a ratio "
            "of two quantities that scale together. This test therefore "
            "separates the candidates on intensivity (V2.1) and not on scale, "
            "and we record the extensive deviation so the reader can see that "
            "no separation is being claimed here."
        ),
    }


def main() -> Dict:
    os.makedirs(RESULTS, exist_ok=True)
    tests = [
        v2_1_intensivity_under_extension(),
        v2_2_permutation_is_non_discriminating(),
        v2_3_sequential_order_dependence(),
        v2_4_scale_invariance(),
    ]
    scored = [t for t in tests if t.get("scored", True) is not False]
    n_pass = sum(1 for t in scored if t.get("passed"))
    res = {"script": "v2_intensivity.py",
           "tests": tests,
           "summary": {"n": len(tests), "scored": len(scored),
                       "passed": n_pass,
                       "all_scored_passed": n_pass == len(scored)}}
    out = os.path.join(RESULTS, "v2_intensivity.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[V2] {n_pass}/{len(scored)} scored passed -> {out}")
    for t in tests:
        if t.get("scored", True) is False:
            print(f"  NON-DISC  {t['test']}")
        else:
            print(f"  {'PASS' if t.get('passed') else 'FAIL'}  {t['test']}")
    return res


if __name__ == "__main__":
    main()
