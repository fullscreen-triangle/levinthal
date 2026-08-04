"""
Run every causal-knowledge-graphs experiment and report.

Order matters and is not alphabetical by accident:

  EXP-A  establishes `thm:subtrie-cut` -- that when the weighting
         factors through the trie, minimum cuts land on subtrie blocks.
         Everything downstream depends on this.
  EXP-B  checks the weighting that Part III DERIVES rather than posits,
         and the numbers it forces (d_min, resolution depth, the
         contact ratio, and Lambda-cancellation).
  EXP-C  tests what happens when EXP-A's hypothesis FAILS -- the
         disulfide case -- and whether degradation is graceful.

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
    ("exp_a_subtrie_cut.py", "exp_a_subtrie_cut.json",
     "do minimum cuts land on subtrie blocks?"),
    ("exp_b_weighting_and_floor.py", "exp_b_weighting_and_floor.json",
     "does the derived weighting give the claimed numbers?"),
    ("exp_c_degradation.py", "exp_c_degradation.json",
     "does the theorem degrade gracefully off-hypothesis?"),
]

# one headline number per experiment, pulled from its own summary block
HEADLINE = {
    "exp_a_subtrie_cut.json": lambda s: (
        f"{s['cases_theorem_holds']}/{s['cases']} cases, "
        f"{s['informative_cases']} informative, "
        f"worst gap {s['worst_gap_over_all_cases']:.1e}, "
        f"control positive: {s['control_gap_positive']}"),
    "exp_b_weighting_and_floor.json": lambda s: (
        f"{s['parts_passed']}/{s['parts']} parts, "
        f"d_min {s['d_min']:.4f}, depth {s['actual_depth']}/"
        f"{s['guaranteed_depth']}, ratio {s['contact_ratio_realised']:.1f} "
        f"(bound {s['contact_ratio_bound']:.1f})"),
    "exp_c_degradation.json": lambda s: (
        f"{s['trials']} trials, bound held in all, "
        f"bit in {s['trials_where_perturbation_bit']}, "
        f"max gap/P {s['max_tightness_gap_over_P']:.2f}"),
}


def main() -> int:
    rows = []
    failed = 0
    for script, result_file, question in SCRIPTS:
        proc = subprocess.run([sys.executable, script], cwd=HERE,
                              capture_output=True, text=True)
        summary = None
        rf = RESULTS / result_file
        if rf.exists():
            summary = json.loads(rf.read_text(encoding="utf-8")).get("summary")
        rows.append({"script": script, "question": question,
                     "exit_code": proc.returncode, "summary": summary})
        if proc.returncode != 0:
            failed += 1
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)

    print()
    print("=" * 72)
    print("  CAUSAL KNOWLEDGE GRAPHS --- full suite")
    print("=" * 72)
    for r in rows:
        mark = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(f"\n  [{mark}] {r['script']}")
        print(f"         {r['question']}")
        s = r["summary"]
        fn = HEADLINE.get(Path(r['script']).stem + ".json")
        if s and fn:
            try:
                print(f"         {fn(s)}")
            except (KeyError, TypeError):
                print("         (summary present but unreadable)")
        elif s is None:
            print("         (no results file written)")
    print()
    print("-" * 72)
    print(f"  SUITE: {'PASS' if failed == 0 else f'FAIL ({failed} broken)'}")
    print("=" * 72)
    print()

    (RESULTS / "run_all.json").write_text(
        json.dumps({"experiments": rows,
                    "verdict": "PASS" if failed == 0 else "FAIL"},
                   indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
