"""Run all 8 validation scripts for Paper 10: Pharmacogenomics Atlas."""
import subprocess, sys
from pathlib import Path

scripts_dir = Path(__file__).parent / "scripts"
scripts = sorted(scripts_dir.glob("[0-9][0-9]_*.py"))

results = []
for s in scripts:
    r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr)
    results.append((s.stem, r.returncode == 0 and "FAIL" not in r.stdout))

print("\n" + "="*50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"Result: {passed}/{total} PASS")
if passed < total:
    sys.exit(1)
