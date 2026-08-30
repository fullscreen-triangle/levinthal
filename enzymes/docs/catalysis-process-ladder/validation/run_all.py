#!/usr/bin/env python3
"""
Master runner for the validation suite of

    "A Process Is a Ladder: Catalysis Without Identities"

SCORING POLICY
    A test contributes to the score only if it could have failed.  Tests whose
    own control shows the statistic does not discriminate are marked
    NON-DISCRIMINATING and excluded, with the reason recorded.  Descriptive
    outputs carry no weight.

    This is stated because a reader cannot distinguish a passing control from
    a non-discriminating one by inspecting a table of successes.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

MODULES = [
    ("v1_floor_and_asymmetry",
     "Floor, its failure branch, and the probe/commit asymmetry"),
    ("v3_composition_and_inertness",
     "Composition (P1-P3), sensitivity (P4), inertness (P5)"),
    ("v6_semantics_and_static",
     "Semantics (P6), static analysis (P7), worked demonstration"),
]


def classify(t: Dict) -> str:
    if str(t.get("status", "")).startswith("SKIPPED"):
        return "skipped"
    if t.get("scored", True) is False:
        return "non_discriminating"
    if "not scored" in str(t.get("test", "")):
        return "descriptive"
    return "passed" if t.get("passed") else "failed"


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    summary: Dict[str, object] = {
        "paper": "A Process Is a Ladder: Catalysis Without Identities",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modules": [],
    }
    tally = {"passed": 0, "failed": 0, "skipped": 0,
             "non_discriminating": 0, "descriptive": 0}
    failures: List[Dict] = []
    nondisc: List[Dict] = []

    for name, desc in MODULES:
        print(f"\n=== {name}: {desc} ===")
        entry: Dict[str, object] = {"module": name, "description": desc}
        try:
            mod = importlib.import_module(name)
            res = mod.main()
            counts = {k: 0 for k in tally}
            for t in res.get("tests", []):
                k = classify(t)
                counts[k] += 1
                tally[k] += 1
                if k == "failed":
                    failures.append({"module": name, "test": t.get("test")})
                if k == "non_discriminating":
                    nondisc.append({"module": name, "test": t.get("test"),
                                    "reason": t.get("verdict", "")})
            entry["status"] = "ok"
            entry["counts"] = counts
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-1200:]
            print(f"  ERROR: {exc}")
        summary["modules"].append(entry)

    scored = tally["passed"] + tally["failed"]
    summary["tally"] = tally
    summary["scored_tests"] = scored
    summary["pass_rate_of_scored"] = (tally["passed"] / scored
                                      if scored else None)
    summary["failures"] = failures
    summary["non_discriminating"] = nondisc
    summary["elapsed_seconds"] = round(time.time() - started, 2)

    out = os.path.join(RESULTS, "SUMMARY.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 66)
    print("VALIDATION SUMMARY")
    print("=" * 66)
    print(f"  scored tests      : {scored}")
    print(f"    passed          : {tally['passed']}")
    print(f"    failed          : {tally['failed']}")
    print(f"  non-discriminating: {tally['non_discriminating']}")
    print(f"  skipped           : {tally['skipped']}")
    print(f"  descriptive only  : {tally['descriptive']}")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    - {f['module']}: {f['test']}")
    if nondisc:
        print("\n  NON-DISCRIMINATING:")
        for f in nondisc:
            print(f"    - {f['module']}: {f['test']}")
    print(f"\n  summary -> {out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
