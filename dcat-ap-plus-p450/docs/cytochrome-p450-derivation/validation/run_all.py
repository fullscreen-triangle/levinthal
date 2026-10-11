"""Run every validation experiment in order and report which succeeded.

    uv run python run_all.py            # all
    uv run python run_all.py g5 g6      # by prefix
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    "f1_partial_sums", "f2_individuation", "f3_recurrence", "f4_ladders",
    "g1_atoms", "g2_backbone", "g3_boundary", "g4_record", "g5_fold", "g6_ligation",
]

wanted = sys.argv[1:]
failed = []
for name in EXPERIMENTS:
    if wanted and not any(name.startswith(w) for w in wanted):
        continue
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=HERE,
                          capture_output=True, text=True, encoding="utf-8")
    status = "ok" if proc.returncode == 0 else "FAILED"
    print(f"{name:20s} {status:7s} {time.time() - t0:6.1f}s")
    if proc.returncode:
        failed.append(name)
        print(proc.stderr[-2000:])
sys.exit(1 if failed else 0)
