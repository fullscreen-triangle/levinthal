#!/usr/bin/env python3
"""
Master runner for the validation suite of

    "Shakespeare with Ladders"

SCORING POLICY
    A test contributes to the score only if it could have failed.  Tests whose
    own construction shows the statistic cannot discriminate are marked
    NON-DISCRIMINATING and excluded, with the reason recorded.  Claims the
    suite cannot bear on are marked NOT TESTED and excluded.

    This is stated because a reader cannot distinguish a passing control from
    a vacuous one by inspecting a table of successes.

    Exit code is 0 only if every scored check passes.
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
    ("v1_substrate_and_resolution",
     "Floor, finiteness, local separation cost, the radius sweep"),
    ("v2_intensivity",
     "Intensivity, the near-miss control, graded order dependence"),
    ("v3_semantics_and_refusal",
     "Composition, sensitivity, freeness, the clock, the refusal"),
]


def classify(t: Dict) -> str:
    if str(t.get("verdict", "")).startswith("NOT TESTED"):
        return "not_tested"
    if t.get("scored", True) is False:
        return "non_discriminating"
    return "passed" if t.get("passed") else "failed"


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    summary: Dict[str, object] = {
        "paper": "Shakespeare with Ladders",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modules": [],
    }
    tally = {"passed": 0, "failed": 0, "non_discriminating": 0,
             "not_tested": 0}
    failures: List[Dict] = []
    nondisc: List[Dict] = []
    negatives: List[str] = []

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
                if str(t.get("test", "")).startswith("NEGATIVE CONTROL"):
                    negatives.append(t["test"])
            entry["status"] = "ok"
            entry["counts"] = counts
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-1200:]
            tally["failed"] += 1
            failures.append({"module": name, "test": "MODULE CRASHED"})
            print(f"  ERROR: {exc}")
        summary["modules"].append(entry)

    scored = tally["passed"] + tally["failed"]
    summary["tally"] = tally
    summary["scored_tests"] = scored
    summary["pass_rate_of_scored"] = (tally["passed"] / scored
                                      if scored else None)
    summary["failures"] = failures
    summary["non_discriminating"] = nondisc
    summary["negative_controls"] = {"count": len(negatives),
                                    "checks": negatives}
    summary["elapsed_seconds"] = round(time.time() - started, 2)

    out = os.path.join(RESULTS, "SUMMARY.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 68)
    print("VALIDATION SUMMARY")
    print("=" * 68)
    print(f"  scored tests      : {scored}")
    print(f"    passed          : {tally['passed']}")
    print(f"    failed          : {tally['failed']}")
    print(f"  non-discriminating: {tally['non_discriminating']}")
    print(f"  not tested        : {tally['not_tested']}")
    print(f"  negative controls : {len(negatives)}")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    - {f['module']}: {f['test']}")
    if nondisc:
        print("\n  NON-DISCRIMINATING:")
        for f in nondisc:
            print(f"    - {f['module']}: {f['test']}")
    print(f"\n  summary -> {out}")
    print("=" * 68)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
