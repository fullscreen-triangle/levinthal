"""
Run every attention-allocated-retrieval experiment and report.

Order matters and is not alphabetical by accident:

  EXP-A  finds the convexity threshold e* and shows it is structural.
  EXP-B  shows the threshold flips the OPTIMAL policy, not just the
         bound on lost optimality.
  EXP-C  tests the allocator built on both, against brute force.

Each script exits non-zero if any of its checks fail. This runner
propagates that, so the whole suite is a single pass/fail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SCRIPTS = [
    ("exp_a_diminishing_returns.py", "exp_a_diminishing_returns.json",
     "does retrieval have diminishing returns?"),
    ("exp_b_corner_vs_spread.py", "exp_b_corner_vs_spread.json",
     "below threshold, is spreading the wrong action?"),
    ("exp_c_allocator.py", "exp_c_allocator.json",
     "does the gated allocator match brute force?"),
]


def main() -> int:
    rows = []
    failed = 0
    for script, result_file, question in SCRIPTS:
        proc = subprocess.run([sys.executable, script], cwd=HERE,
                              capture_output=True, text=True)
        agg = None
        rf = RESULTS / result_file
        if rf.exists():
            agg = json.loads(rf.read_text(encoding="utf-8")).get("aggregate")
        rows.append({"script": script, "question": question,
                     "exit_code": proc.returncode, "aggregate": agg})
        if proc.returncode != 0:
            failed += 1
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)

    print()
    print("=" * 72)
    print("  ATTENTION-ALLOCATED RETRIEVAL --- full suite")
    print("=" * 72)
    total_checks = total_passed = 0
    for r in rows:
        a = r["aggregate"] or {}
        c, p = a.get("checks", 0), a.get("passed", 0)
        total_checks += c
        total_passed += p
        mark = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(f"\n  [{mark}] {r['script']}")
        print(f"         {r['question']}")
        print(f"         {p}/{c} checks")
    print()
    print("-" * 72)
    print(f"  TOTAL: {total_passed}/{total_checks} checks across "
          f"{len(rows)} experiments")
    print(f"  SUITE: {'PASS' if failed == 0 else f'FAIL ({failed} broken)'}")
    print("=" * 72)
    print()

    (RESULTS / "run_all.json").write_text(
        json.dumps({"experiments": rows,
                    "total_checks": total_checks,
                    "total_passed": total_passed,
                    "verdict": "PASS" if failed == 0 else "FAIL"},
                   indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
