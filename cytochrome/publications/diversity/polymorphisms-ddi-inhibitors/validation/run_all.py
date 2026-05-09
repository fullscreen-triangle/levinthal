"""Run all 8 validation scripts for Paper 15 — Polymorphisms, DDI, Inhibitors."""
import subprocess, sys, os

SCRIPTS = [
    "01_polymorphism_dm_shift.py",
    "02_competitive_inhibition_alpha.py",
    "03_mbi_inactivation.py",
    "04_induction_fold.py",
    "05_inhibitor_ranking.py",
    "06_compound_phenotype_ddi.py",
    "07_tdi_ic50_shift.py",
    "08_full_ddi_table.py",
]

base = os.path.join(os.path.dirname(__file__), "scripts")
passed = 0
for s in SCRIPTS:
    r = subprocess.run([sys.executable, os.path.join(base, s)],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    passed += ok
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {s}")
    if not ok:
        print(r.stdout[-400:] if r.stdout else "")
        print(r.stderr[-200:] if r.stderr else "")

print(f"\n{passed}/{len(SCRIPTS)} PASS")
sys.exit(0 if passed == len(SCRIPTS) else 1)
