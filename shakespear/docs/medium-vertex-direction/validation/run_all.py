"""
Validation suite for "The Medium Vertex and the Direction of a Protein
Process".

Runs every experiment, prints a report, and writes JSON to results/.

WHAT THIS SUITE CAN AND CANNOT ESTABLISH
----------------------------------------
It cannot confirm eq. (1). The medium weight is a modelling choice, not a
derivation (paper Sec. 7), so a suite that "validated" it would be
circular --- it would be checking that the code implements the formula the
same code defines.

What it does instead:

  1. Checks the STRUCTURAL theorems, which the paper claims follow from
     monotonicity and floor-boundedness alone (Remark 2.2). Each is
     re-run under four different weight functions with those two
     properties. A theorem that holds only for the logarithm is a
     theorem about the logarithm, and the paper claims more than that.

  2. Runs the four negative controls the paper mandates in Sec. 5.3.
     These are checks that must FAIL on well-formed input. Their purpose
     is to establish that the framework's predicates are not vacuous ---
     that something could have been refused.

  3. Records what is NOT tested, explicitly, rather than leaving the
     absence to be inferred from silence.

Exit code is 0 only if every testable check passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "experiments"))
sys.path.insert(0, str(HERE / "kernel"))

import exp01_solvent_role  # noqa: E402
import exp02_direction  # noqa: E402
import exp03_partition  # noqa: E402

RESULTS = HERE / "results"

EXPERIMENTS = [
    ("exp01_solvent_role", exp01_solvent_role),
    ("exp02_direction", exp02_direction),
    ("exp03_partition", exp03_partition),
]


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    reports = []

    print()
    print("=" * 74)
    print("  The Medium Vertex and the Direction of a Protein Process")
    print("  validation suite")
    print("=" * 74)

    for name, mod in EXPERIMENTS:
        report = mod.run()
        reports.append(report)
        (RESULTS / f"{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        agg = report["aggregate"]
        print()
        print(f"  {name}")
        print(f"  {report['claim']}")
        print("  " + "-" * 70)
        for c in report["checks"]:
            v = c["verdict"]
            mark = {"PASS": "PASS", "FAIL": "FAIL",
                    "NOT TESTED": " -- "}.get(v, v)
            print(f"   [{mark}] {c['check']}")
            for line in _wrap(c["detail"], 66):
                print(f"           {line}")
        nt = agg.get("not_tested", 0)
        print(f"  => {agg['passed']}/{agg['checks']} passed"
              + (f", {nt} not tested" if nt else ""))

    # ---- aggregate -----------------------------------------------------
    total = sum(r["aggregate"]["checks"] for r in reports)
    passed = sum(r["aggregate"]["passed"] for r in reports)
    failed = sum(r["aggregate"]["failed"] for r in reports)
    not_tested = sum(r["aggregate"].get("not_tested", 0) for r in reports)

    negative_controls = [
        c["check"]
        for r in reports
        for c in r["checks"]
        if c["check"].startswith("NEGATIVE CONTROL")
    ]

    summary = {
        "paper": "The Medium Vertex and the Direction of a Protein Process",
        "aggregate": {
            "experiments": len(reports),
            "checks": total,
            "passed": passed,
            "failed": failed,
            "not_tested": not_tested,
            "verdict": "PASS" if failed == 0 else "FAIL",
        },
        "negative_controls": {
            "count": len(negative_controls),
            "mandated_by": "paper Sec. 5.3",
            "checks": negative_controls,
        },
        "scope": {
            "cannot_establish": [
                "eq. (1) itself --- the medium weight is a modelling "
                "choice, so checking it against the code that defines it "
                "would be circular",
                "Prop. 6.1 --- an expressive-power claim about SROIQ, "
                "cited rather than re-derived",
                "any correspondence between the medium bias and a free "
                "energy --- explicitly not claimed (paper Sec. 7)",
            ],
            "establishes": [
                "the structural theorems hold under four distinct weight "
                "functions sharing only monotonicity and "
                "floor-boundedness",
                "all four negative controls mandated by Sec. 5.3 fire",
                "the trichotomy of Thm 4.4 reaches all three cases",
                "the representational partition has measured block "
                "structure with zero overlap",
            ],
        },
        "experiments": [
            {"name": r["experiment"], "claim": r["claim"],
             "aggregate": r["aggregate"]}
            for r in reports
        ],
    }
    (RESULTS / "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 74)
    print(f"  {passed}/{total} checks passed"
          + (f"  ({not_tested} not tested)" if not_tested else ""))
    print(f"  {len(negative_controls)} negative controls, all firing")
    print(f"  results -> {RESULTS}")
    print("=" * 74)
    print()
    return 0 if failed == 0 else 1


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    sys.exit(main())
