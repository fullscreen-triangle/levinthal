"""Run all Paper 7 validation scripts and write summary."""
import subprocess, sys, json
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

scripts = sorted(
    p for p in SCRIPTS.glob("0*.py") if not p.name.startswith("_")
)

verdicts = {}
for s in scripts:
    result = subprocess.run(
        [sys.executable, str(s)],
        capture_output=True, text=True
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    name = s.stem
    rfile = RESULTS / f"{name}.json"
    if rfile.exists():
        data = json.loads(rfile.read_text())
        verdicts[name] = data.get("verdict", "FAIL")
    else:
        verdicts[name] = "FAIL"

total = len(verdicts)
passed = sum(1 for v in verdicts.values() if v == "PASS")
summary = {
    "paper": "Paper 7: Heteroatom Oxidation and Dealkylation",
    "total": total,
    "passed": passed,
    "failed": total - passed,
    "verdict": "PASS" if passed == total else "FAIL",
    "scripts": verdicts,
}

(RESULTS / "_summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n{'='*50}")
print(f"Paper 7 validation: {passed}/{total} PASS")
print(f"Overall: {summary['verdict']}")
