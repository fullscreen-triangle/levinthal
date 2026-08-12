#!/usr/bin/env python3
"""Run every validation script for the Empty Dictionary (P450) paper.

Exit code is non-zero if any experiment's verdict is FAIL, so this can gate
a build. Each script writes its own JSON into results/.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
RESULTS = os.path.join(HERE, "results")

ORDER = [
    "01_storage_scaling.py",
    "02_query_without_entries.py",
    "03_paper2_reconciliation.py",
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    failures, summary = [], []

    for name in ORDER:
        path = os.path.join(SCRIPTS, name)
        print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
        proc = subprocess.run([sys.executable, path], cwd=SCRIPTS)
        if proc.returncode != 0:
            failures.append(name)

        rpath = os.path.join(RESULTS, name.replace(".py", ".json"))
        verdict = "MISSING"
        if os.path.exists(rpath):
            with open(rpath) as f:
                verdict = json.load(f).get("verdict", "UNKNOWN")
        summary.append((name, verdict))

    print(f"\n{'=' * 68}\nSUMMARY\n{'=' * 68}")
    for name, verdict in summary:
        print(f"  {verdict:<8} {name}")

    if failures:
        print(f"\n{len(failures)} experiment(s) FAILED: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(ORDER)} experiments passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
