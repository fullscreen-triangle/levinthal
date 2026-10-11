"""The record an experiment keeps of itself: `experiments/<id>/runs/<run>/`.

Append-only. A file is created once and never rewritten: a later status or a second
collection is a new file, so the record shows everything that happened, in order.

    runs/<run>/wrap.json        what was wrapped: spec digest, unit, repository, commits, command line
    runs/<run>/wrap.txt         lattice's own output (not .log: levinthal ignores *.log)
    runs/<run>/status-<t>.json  shard states read from the results branch at time t
    runs/<run>/collect-<t>.json collected files with digests, and the verdict of every check
    runs/<run>/collect-<t>/...  the collected files (shard files, logs, outputs)
    plans/<t>.txt               `lattice plan` output; a plan is not a run
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_once(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:  # "x": never overwrite
        f.write(text)
    return path


def write_json(path: Path, data: object) -> Path:
    return write_once(path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def runs(exp_dir: Path) -> list[Path]:
    d = exp_dir / "runs"
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []


def latest_run(exp_dir: Path) -> Path | None:
    r = runs(exp_dir)
    return r[-1] if r else None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(run_dir: Path, prefix: str) -> Path | None:
    files = sorted(run_dir.glob(f"{prefix}-*.json"))
    return files[-1] if files else None
