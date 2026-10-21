"""Compare regenerated results with the committed ones.

After `run_all.py`, each `results/<name>.json` should equal the version committed at
HEAD, except for the `generated` timestamp. Numbers may differ by floating-point noise
between platforms, so they are compared with a relative tolerance of 1e-9 and no
absolute one: several results are gaps near 1e-50, which any absolute tolerance would hide.

    uv run python run_all.py f
    uv run python check_reproduced.py f1_partial_sums f2_individuation f3_recurrence f4_ladders

Exit 0 if every result reproduces, 1 if any differs, 2 if one cannot be read.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IGNORED = {"generated"}
RTOL = 1e-9


def diff(a: object, b: object, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in IGNORED:
                continue
            if k not in a or k not in b:
                out.append(f"{where}/{k}: only in {'committed' if k in a else 'regenerated'}")
            else:
                out += diff(a[k], b[k], f"{where}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{where}[{i}]")]
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)
        return [] if same else [f"{where}: {a!r} != {b!r}"]
    return [] if a == b else [f"{where}: {a!r} != {b!r}"]


def committed(rel: str) -> object:
    top = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True).stdout.strip()
    path = (HERE / rel).resolve().relative_to(Path(top).resolve()).as_posix()
    blob = subprocess.run(["git", "-C", top, "show", f"HEAD:{path}"], capture_output=True, check=True).stdout
    return json.loads(blob)


def main(names: list[str]) -> int:
    if not names:
        print(__doc__)
        return 2
    code = 0
    for name in names:
        rel = f"results/{name}.json"
        try:
            old, new = committed(rel), json.loads((HERE / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, subprocess.CalledProcessError) as e:
            print(f"{name:20s} UNREADABLE {e}")
            return 2
        found = diff(old, new)
        print(f"{name:20s} {'reproduced' if not found else f'DIFFERS in {len(found)} place(s)'}")
        for d in found[:10]:
            print(f"    {d}")
        code = code or (1 if found else 0)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
