"""A stand-in for lattice in the integration tests: `plan`, `wrap` and `--version`.

It reproduces what the driver depends on, as lattice 0.2.0 does it:

- shard ids from the matrix, keys sorted, as `key-value_key-value` (lattice/src/unit.rs);
- refusal of a command with a `{key}` placeholder that no matrix key fills;
- `wrap` commits only `.lattice/<name>/` and pushes the current branch, unless `--no-push`.

The unit is written as `.lattice/<name>/unit.json` (lattice writes unit.toml and run.sh).
`run_on_apphub` in test_pipeline.py reads it to execute the shards. Every invocation is
appended to the file named by STUB_LATTICE_LOG, so tests can see what the driver passed.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"(^|[^$])\{([a-z_][a-z0-9_]*)\}")


def values(spec: str) -> list[str]:
    if re.fullmatch(r"-?\d+\.\.-?\d+", spec):
        lo, hi = (int(x) for x in spec.split(".."))
        return [str(i) for i in range(lo, hi + 1)]
    return spec.split(",")


def sanitize(name: str) -> str:
    s = re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9._-]", "-", name)).strip("-.")
    return s or "unit"


def parse(args: list[str]) -> dict:
    words = []
    if "--" in args:
        i = args.index("--")
        args, words = args[:i], args[i + 1:]
    o = {"repo": ".", "verb": "", "task": "", "name": "", "matrix": {}, "outputs": [], "env": {},
         "secrets": [], "flags": set(), "words": words}
    it = iter(args)
    for a in it:
        if a == "-C":
            o["repo"] = next(it)
        elif a in {"plan", "wrap"} and not o["verb"]:
            o["verb"] = a
        elif a == "--name":
            o["name"] = next(it)
        elif a == "--matrix":
            k, v = next(it).split("=", 1)
            o["matrix"][k] = values(v)
        elif a == "--output":
            o["outputs"].append(next(it))
        elif a == "--env":
            k, v = next(it).split("=", 1)
            o["env"][k] = v
        elif a == "--secret":
            o["secrets"].append(next(it))
        elif a in {"--gpu", "--parallel", "--timeout", "--each"}:
            next(it)
        elif a in {"--force", "--no-push"}:
            o["flags"].add(a)
        elif o["verb"] and not a.startswith("-"):
            o["task"] = a
        else:
            sys.exit(f"lattice: error: unexpected argument {a!r}")
    return o


def shards(matrix: dict[str, list[str]]) -> list[dict]:
    keys = sorted(matrix)
    out = []
    for combo in itertools.product(*(matrix[k] for k in keys)):
        pairs = list(zip(keys, combo))
        sid = sanitize("_".join(f"{k}-{v}" for k, v in pairs)) if pairs else "main"
        out.append({"id": sid, "vars": dict(pairs)})
    return out


def git(repo: str, *a: str) -> None:
    subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True)


def main(argv: list[str]) -> int:
    log = os.environ.get("STUB_LATTICE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + "\n")
    if argv == ["--version"]:
        print("lattice 0.2.0-stub")
        return 0
    o = parse(argv)
    if not o["verb"]:
        sys.exit("lattice: error: give plan or wrap")
    script = " ".join(o["words"])
    missing = sorted({m.group(2) for m in PLACEHOLDER.finditer(script)} - set(o["matrix"]))
    if missing:
        print(f"lattice: error: the command uses {{{missing[0]}}} but no --matrix gives a value for it",
              file=sys.stderr)
        return 1
    units = shards(o["matrix"])
    name = o["name"] or o["task"] or "unit"
    print(f"unit         {name} — {len(units)} shard(s)")
    if o["verb"] == "plan":
        return 0
    unit_dir = Path(o["repo"]) / ".lattice" / name
    if unit_dir.exists() and "--force" not in o["flags"]:
        print(f"lattice: error: {unit_dir} exists (use --force)", file=sys.stderr)
        return 1
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = {"name": name, "words": o["words"], "shards": units, "outputs": o["outputs"], "env": o["env"]}
    (unit_dir / "unit.json").write_text(json.dumps(unit, indent=2), encoding="utf-8")
    git(o["repo"], "add", str(Path(".lattice") / name))
    git(o["repo"], "commit", "-q", "-m", f"lattice: unit {name}")
    if "--no-push" not in o["flags"]:
        remote = os.environ.get("STUB_LATTICE_REMOTE", "origin")
        git(o["repo"], "push", "-q", remote, "HEAD")
    print(f"wrapped {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
