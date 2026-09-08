#!/usr/bin/env python3
"""
Master runner for the validation suite of

    "Amino Acids and Their Polymers as Partition Structures: A
    First-Principles Derivation of the Peptide Backbone, Protein Folding,
    and the Genetically Encoded Residue Alphabet"

SCORING POLICY
    A test contributes to the score only if it could have failed. Tests
    marked non-scored (descriptive) because the manuscript makes no
    falsifiable numerical claim there are excluded, with the reason
    recorded in the test's own "interpretation" field.

    This is stated explicitly because a reader cannot distinguish a
    passing test from a merely descriptive one by inspecting a table of
    successes alone.
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
    ("v1_floor_and_depth_sharing",
     "Boundary-thickness floor, its bounded-refinement failure branch, "
     "and depth reduction under coordinate sharing"),
    ("v2_backbone_planarity_and_folding",
     "Peptide bond planarity/depth-sharing signature and folding-forced "
     "compaction, tested against measured structural data"),
    ("v3_substitution_and_recurrence",
     "Substitution tolerance and cross-lineage recurrence, tested "
     "against the standard genetic code and the Trifonov consensus order"),
    ("v4_encoded_count_and_boundary",
     "The encoded alphabet as a recurrence statistic, and the recoded "
     "residues as past-threshold cases"),
    ("v5_hierarchy_and_ceiling",
     "The storage-resolution hierarchy and the alphabet-bounded ceiling, "
     "tested against real CYP3A4 fold, isoform, and electron-transport data"),
    ("v6_reaction_depth_deficit",
     "Partition-depth deficit vs. real catalytic rate and bond energetics"),
]


def classify(t: Dict) -> str:
    if str(t.get("status", "")).startswith("SKIPPED"):
        return "skipped"
    if t.get("scored", True) is False:
        return "descriptive"
    return "passed" if t.get("passed") else "failed"


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    summary: Dict[str, object] = {
        "paper": "Amino Acids and Their Polymers as Partition Structures",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modules": [],
    }
    tally = {"passed": 0, "failed": 0, "skipped": 0, "descriptive": 0}
    failures: List[Dict] = []
    descriptive: List[Dict] = []

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
                if k == "descriptive":
                    descriptive.append({"module": name, "test": t.get("test")})
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
    summary["descriptive_only"] = descriptive
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
    print(f"  descriptive only  : {tally['descriptive']}")
    print(f"  skipped           : {tally['skipped']}")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    - {f['module']}: {f['test']}")
    if descriptive:
        print("\n  DESCRIPTIVE (not scored):")
        for f in descriptive:
            print(f"    - {f['module']}: {f['test']}")
    print(f"\n  summary -> {out}")
    print("=" * 66)
    return 0 if tally["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
