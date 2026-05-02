"""
Run all Paper 1 validation scripts and emit a consolidated summary.

Usage:
    python run_all.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    RESULTS.mkdir(exist_ok=True)
    summary = {
        "monograph_paper": "Paper 1: An Expression Algebra for Biomolecules",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validations": [],
    }

    script_files = sorted(SCRIPTS.glob("[0-9][0-9]_*.py"))

    for script in script_files:
        print(f"\n=== Running {script.name} ===")
        try:
            t0 = time.time()
            module = load_module(script)
            result = module.main()
            elapsed = time.time() - t0

            out_path = RESULTS / f"{script.stem}.json"
            with out_path.open("w") as f:
                json.dump(result, f, indent=2)

            summary["validations"].append({
                "script": script.name,
                "validation_id": result.get("validation_id"),
                "paper_reference": result.get("paper_reference"),
                "verdict": result.get("verdict"),
                "checks_passed": sum(
                    1 for v in result.get("checks", {}).values() if v is True
                ),
                "checks_total": len(result.get("checks", {})),
                "elapsed_s": round(elapsed, 3),
                "output_file": str(out_path.relative_to(ROOT)),
            })
            print(f"   verdict: {result.get('verdict')}  ({elapsed:.2f}s)")
        except Exception as exc:
            summary["validations"].append({
                "script": script.name,
                "verdict": "ERROR",
                "error": str(exc),
            })
            print(f"   ERROR: {exc}")

    # Aggregate
    n_pass = sum(1 for v in summary["validations"] if v.get("verdict") == "PASS")
    n_fail = sum(1 for v in summary["validations"] if v.get("verdict") == "FAIL")
    n_error = sum(1 for v in summary["validations"] if v.get("verdict") == "ERROR")
    summary["aggregate"] = {
        "total": len(summary["validations"]),
        "pass": n_pass,
        "fail": n_fail,
        "error": n_error,
        "overall_verdict": "PASS" if n_fail == 0 and n_error == 0 else "FAIL",
    }

    summary_path = RESULTS / "_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"OVERALL: {summary['aggregate']['overall_verdict']}  "
          f"(PASS={n_pass} / FAIL={n_fail} / ERROR={n_error} / TOTAL={len(script_files)})")
    print(f"Summary written to: {summary_path}")
    return 0 if summary["aggregate"]["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
