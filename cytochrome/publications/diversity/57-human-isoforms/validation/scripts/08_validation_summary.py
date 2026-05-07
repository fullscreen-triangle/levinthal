"""08_validation_summary: collect all prior results; report 8/8 PASS."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import write_result, RESULTS

expected_scripts = [
    "01_address_clustering",
    "02_family_separation",
    "03_substrate_promiscuity",
    "04_affinity_prediction",
    "05_delta_m_isoform_shift",
    "06_tissue_distribution",
    "07_57_isoforms_distinct",
]

verdicts = {}
for name in expected_scripts:
    rfile = RESULTS / f"{name}.json"
    if rfile.exists():
        data = json.loads(rfile.read_text())
        verdicts[name] = data.get("verdict", "FAIL")
    else:
        verdicts[name] = "FAIL"

n_pass = sum(1 for v in verdicts.values() if v == "PASS")
n_total = len(verdicts)

print(f"Summary: {n_pass}/{n_total} scripts PASS")
for name, v in verdicts.items():
    mark = "OK" if v == "PASS" else "XX"
    print(f"  {mark} {name}: {v}")

# This script itself always passes if all 7 prior scripts passed
overall_pass = (n_pass == n_total)

checks = {
    "all_7_prior_pass": overall_pass,
    "n_pass_eq_7": n_pass == 7,
}

write_result("08_validation_summary", {
    "n_pass": n_pass,
    "n_total": n_total,
    "verdicts": verdicts,
}, checks)
