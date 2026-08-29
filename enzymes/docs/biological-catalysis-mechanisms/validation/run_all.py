#!/usr/bin/env python3
"""
Master runner for the validation suite of

    "Why Enzymes Exist: Catalysis as Categorical Provision Upon Contact"

Runs every validation module, aggregates the JSON outputs, and writes a
single summary file.

SCORING POLICY
    A test contributes to the score only if it COULD have failed.  Tests whose
    own negative control shows the statistic does not discriminate are marked
    NON-DISCRIMINATING and excluded from the score, with the reason recorded.
    Skipped tests (missing data) are reported separately and are not counted
    as passes.

    This is deliberate.  A reader cannot distinguish a passing control from a
    non-discriminating one by inspecting a table of successes, so the
    distinction is made explicit here and in every module.

Usage:
    python run_all.py            # run everything (fetches if cache absent)
    python run_all.py --offline  # skip network retrieval, use cache
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

MODULES = [
    ("v1_abstract_system",
     "Abstract system: category/residue independence"),
    ("v2_floor_and_orthogonality",
     "Floor positivity/uniformity; configurational-kinetic orthogonality"),
    ("v3_haldane_closure",
     "Haldane closure; equilibrium invariance (P1)"),
    ("v4_specificity_window",
     "Bounded specificity window (P2)"),
    ("v5_aperture_counting",
     "Aperture counting and efficiency law (P3, P4)"),
    ("v9_inhibition_taxonomy",
     "Inhibition dichotomy (P5)"),
    ("v8_database_scale_analysis",
     "Scale analysis over KEGG/Reactome (P7 context)"),
]

NETWORK_MODULE = ("v6_fetch_reaction_data",
                  "Retrieve KEGG and Reactome records")


def classify(test: Dict) -> str:
    if str(test.get("status", "")).startswith("SKIPPED"):
        return "skipped"
    if test.get("scored", True) is False:
        return "non_discriminating"
    if "not scored" in str(test.get("test", "")):
        return "descriptive"
    return "passed" if test.get("passed") else "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip network retrieval; use existing cache")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.time()

    summary: Dict[str, object] = {
        "paper": "Why Enzymes Exist: Catalysis as Categorical Provision Upon Contact",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline": args.offline,
        "modules": [],
    }

    # ---- optional network retrieval -------------------------------------
    if not args.offline:
        name, desc = NETWORK_MODULE
        print(f"\n=== {name}: {desc} ===")
        try:
            mod = importlib.import_module(name)
            mod.main()
            summary["modules"].append(
                {"module": name, "description": desc, "status": "ok"})
        except Exception as exc:
            print(f"  retrieval failed: {exc}")
            summary["modules"].append(
                {"module": name, "description": desc,
                 "status": "failed", "error": str(exc)})

    # ---- analysis modules -----------------------------------------------
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
            counts = {"passed": 0, "failed": 0, "skipped": 0,
                      "non_discriminating": 0, "descriptive": 0}
            for t in res.get("tests", []):
                k = classify(t)
                counts[k] += 1
                tally[k] += 1
                if k == "failed":
                    failures.append({"module": name, "test": t.get("test")})
                if k == "non_discriminating":
                    nondisc.append({"module": name, "test": t.get("test"),
                                    "reason": t.get("verdict")
                                    or "control shows no separation"})
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
    summary["elapsed_seconds"] = round(time.time() - started, 1)

    out = os.path.join(RESULTS_DIR, "SUMMARY.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 68)
    print("VALIDATION SUMMARY")
    print("=" * 68)
    print(f"  scored tests      : {scored}")
    print(f"    passed          : {tally['passed']}")
    print(f"    failed          : {tally['failed']}")
    print(f"  non-discriminating: {tally['non_discriminating']}  "
          f"(reported, excluded from score)")
    print(f"  skipped (no data) : {tally['skipped']}")
    print(f"  descriptive only  : {tally['descriptive']}")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    - {f['module']}: {f['test']}")
    if nondisc:
        print("\n  NON-DISCRIMINATING (statistic could not separate):")
        for f in nondisc:
            print(f"    - {f['module']}: {f['test']}")
    print(f"\n  summary -> {out}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
