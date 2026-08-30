#!/usr/bin/env python3
"""
V1 --- The substrate: floor positivity, the finiteness hypothesis, the local
       separation cost against brute force, and the radius sweep.

The radius sweep tests Principle "resolution, not threshold": the radius must
do something monotone, and the DIRECTION it moves in is a measurement rather
than an assumption.  A sweep in which nothing changes would mean the radius is
inert and would be reported as such.  Our predicted direction was wrong; V1.4
records the correction and the mechanism that explains it.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

from shk_core import (ContactGraph, chain_graph, complete_graph, local_floor,
                      power_extensive, power_globalfloor, power_intensive)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def v1_1_floor_positive() -> Dict:
    """The floor is computed, not assumed, over instances spanning decades."""
    rng = random.Random(11)
    rows = []
    instances = [
        ("uniform", complete_graph(4, [1.0, 1.0, 1.0, 1.0])),
        ("mixed", complete_graph(4, [1.0, 2.0, 0.5, 3.0])),
        ("chain", chain_graph(6, random.Random(2))),
        ("adversarial_1e-12", complete_graph(4, [1e-12, 1.0, 2.0, 0.5])),
    ]
    ok = True
    for name, g in instances:
        f = g.floor
        rows.append({"instance": name, "n_edges": len(g.weights),
                     "floor_computed": f, "positive": f > 0.0,
                     "total": g.total})
        ok = ok and f > 0.0
    return {
        "test": "V1.1 floor is strictly positive on every finite instance",
        "instances": rows,
        "passed": ok,
    }


def v1_2_finiteness_required() -> Dict:
    """
    The failure branch.  An unbounded refinement 1/n has infimum zero: every
    epsilon is eventually cleared.  Truncating to a finite system restores a
    positive floor.  This test exists so the hypothesis is visible rather than
    tacit; it checks a mathematical fact about the sequence, and would fail if
    some epsilon were never cleared.
    """
    cleared = []
    all_cleared = True
    for k in range(1, 10):
        eps = 10.0 ** (-k)
        n = next((n for n in range(1, 10 ** 10) if 1.0 / n < eps), None)
        cleared.append({"epsilon": eps, "first_n_below": n})
        all_cleared = all_cleared and n is not None
    truncated_floor = min(1.0 / n for n in range(1, 11))
    return {
        "test": "V1.2 unbounded refinement has infimum zero; truncation restores a floor",
        "epsilons_cleared": cleared,
        "truncated_to_10_stages_floor": truncated_floor,
        "truncated_floor_positive": truncated_floor > 0.0,
        "passed": bool(all_cleared and truncated_floor > 0.0),
    }


def v1_3_local_sepcost_vs_bruteforce(n_graphs: int = 40,
                                     seed: int = 13) -> Dict:
    """
    At a radius large enough to cover the graph, the local separation cost must
    equal the global one computed by exhaustive enumeration.  This checks the
    implementation against ground truth rather than restating the definition.
    """
    rng = random.Random(seed)
    max_dev = 0.0
    rows = []
    for k in range(n_graphs):
        n = rng.randint(3, 6)
        g = chain_graph(n, rng)
        v = f"v{rng.randrange(n)}"
        wide = g.sigma_local(v, radius=n + 2)
        truth = g.sigma(v)
        dev = abs(wide - truth)
        max_dev = max(max_dev, dev)
        if k < 6:
            rows.append({"n_items": n, "item": v, "local_wide": wide,
                         "bruteforce": truth, "abs_dev": dev})
    return {
        "test": "V1.3 local separation cost at full radius equals brute force",
        "n_graphs": n_graphs,
        "max_abs_deviation": max_dev,
        "sample_rows": rows,
        "passed": max_dev < 1e-12,
    }


def v1_4_radius_is_a_resolution(n_graphs: int = 25, seed: int = 14) -> Dict:
    """
    Principle "resolution, not threshold".  Sweep the radius and count how many
    pairs of items share a derived power to a fixed tolerance.

    WE PREDICTED THE WRONG DIRECTION AND THE SWEEP CORRECTED US.  The first
    version of this test asserted that a COARSE radius identifies more pairs,
    by analogy with a neighbourhood-refinement matcher where a wider view
    discriminates more.  The measurement is the reverse, and the reason is
    structural rather than incidental: balls of radius 0 are disjoint
    singletons, so each item minimises over its own edges and gets its own
    value, whereas balls of large radius OVERLAP and eventually coincide, so
    distinct items minimise over nearly the same edge set and converge to a
    common value.  We therefore measure the mean pairwise ball overlap
    alongside the identification count, because the overlap is the mechanism
    and reporting the count alone would leave the direction unexplained.

    The test can still fail three ways: flat counts (the radius is inert),
    non-monotone counts, or counts moving opposite to the overlap.
    """
    rng = random.Random(seed)
    tol = 1e-9
    sweep: Dict[int, int] = {}
    distinct_by_radius: Dict[int, int] = {}
    overlap_by_radius: Dict[int, float] = {}
    for radius in (0, 1, 2, 3):
        identified = 0
        distinct_total = 0
        overlaps: List[float] = []
        for _ in range(n_graphs):
            g = chain_graph(rng.randint(5, 6), rng)
            items = sorted(g.items())
            ps = [power_intensive(g, v, radius) for v in items]
            balls = [g.ball(v, radius) for v in items]
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if abs(ps[i] - ps[j]) <= tol:
                        identified += 1
                    a, b = balls[i], balls[j]
                    overlaps.append(len(a & b) / len(a | b))
            distinct_total += len({round(p, 12) for p in ps})
        sweep[radius] = identified
        distinct_by_radius[radius] = distinct_total
        overlap_by_radius[radius] = sum(overlaps) / len(overlaps)

    radii = sorted(sweep)
    counts = [sweep[r] for r in radii]
    overl = [overlap_by_radius[r] for r in radii]
    distinct = [distinct_by_radius[r] for r in radii]

    counts_non_decreasing = all(counts[i] <= counts[i + 1]
                                for i in range(len(counts) - 1))
    overlap_increasing = all(overl[i] < overl[i + 1]
                             for i in range(len(overl) - 1))
    distinct_non_increasing = all(distinct[i] >= distinct[i + 1]
                                  for i in range(len(distinct) - 1))
    flat = len(set(counts)) == 1

    return {
        "test": "V1.4 radius is a resolution parameter (fine radius identifies more)",
        "tolerance": tol,
        "pairs_identified_by_radius": {str(r): sweep[r] for r in radii},
        "distinct_powers_by_radius": {str(r): distinct_by_radius[r]
                                      for r in radii},
        "mean_ball_overlap_by_radius": {str(r): overlap_by_radius[r]
                                        for r in radii},
        "counts_non_decreasing": counts_non_decreasing,
        "overlap_strictly_increasing": overlap_increasing,
        "distinct_non_increasing": distinct_non_increasing,
        "sweep_is_flat": flat,
        "passed": bool(counts_non_decreasing and overlap_increasing
                       and distinct_non_increasing and not flat),
        "interpretation": (
            "Identification rises with the radius because the balls overlap "
            "more, not less: at radius 0 the balls are disjoint singletons "
            "and every item gets its own value; at large radius the balls "
            "coincide and distinct items minimise over the same edges. The "
            "measured overlap is reported so that the direction is explained "
            "rather than merely recorded. Our stated prediction was the "
            "opposite and this sweep corrected it."
        ),
    }


def v1_5_local_floor_bounded() -> Dict:
    """beta <= beta_r(v) <= sigma_r(v), checked over many graphs and radii."""
    rng = random.Random(15)
    violations = 0
    checked = 0
    for _ in range(200):
        g = chain_graph(rng.randint(4, 7), rng)
        for v in sorted(g.items()):
            for radius in (0, 1, 2):
                lf = local_floor(g, v, radius)
                sc = g.sigma_local(v, radius)
                checked += 1
                if not (g.floor - 1e-15 <= lf <= sc + 1e-15):
                    violations += 1
    return {
        "test": "V1.5 floor <= local floor <= local separation cost",
        "n_checked": checked,
        "violations": violations,
        "passed": violations == 0,
    }


def main() -> Dict:
    os.makedirs(RESULTS, exist_ok=True)
    tests = [
        v1_1_floor_positive(),
        v1_2_finiteness_required(),
        v1_3_local_sepcost_vs_bruteforce(),
        v1_4_radius_is_a_resolution(),
        v1_5_local_floor_bounded(),
    ]
    n_pass = sum(1 for t in tests if t.get("passed"))
    res = {"script": "v1_substrate_and_resolution.py",
           "tests": tests,
           "summary": {"n": len(tests), "passed": n_pass,
                       "all_passed": n_pass == len(tests)}}
    out = os.path.join(RESULTS, "v1_substrate_and_resolution.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[V1] {n_pass}/{len(tests)} passed -> {out}")
    for t in tests:
        print(f"  {'PASS' if t.get('passed') else 'FAIL'}  {t['test']}")
    return res


if __name__ == "__main__":
    main()
