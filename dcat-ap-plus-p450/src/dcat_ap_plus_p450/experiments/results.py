"""Reading a unit's results from its `lattice-results/<unit>` branch.

AppHub pushes, per shard, `shards/<id>/status.json` (state, exit code, timing, host),
`shards/<id>/log`, and `shards/<id>/files/...` (what the shard wrote to $LATTICE_OUT);
outputs matching the unit's `--output` globs land under `outputs/<path>`
(lattice README, "Results branch"). Everything here reads that branch with git, so
a collection records exactly the revision it read.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .repos import git

STATES_FINAL = {"done", "failed", "timeout"}


@dataclass(frozen=True)
class Fetched:
    rev: str = ""
    error: str = ""


def fetch(repo: Path, remote: str, unit: str) -> Fetched:
    ref = f"lattice-results/{unit}"
    r = git(repo, "fetch", "--quiet", remote, f"+refs/heads/{ref}:refs/remotes/{remote}/{ref}")
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["git fetch failed"])[-1]
        return Fetched(error=f"{ref} not on {remote} yet ({msg})")
    return Fetched(rev=git(repo, "rev-parse", f"refs/remotes/{remote}/{ref}").stdout.strip())


def files(repo: Path, rev: str, prefix: str = "") -> list[str]:
    out = git(repo, "ls-tree", "-r", "--name-only", rev, *([prefix] if prefix else [])).stdout
    return [line for line in out.splitlines() if line]


def blob(repo: Path, rev: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, check=True).stdout


def shard_states(repo: Path, rev: str) -> list[dict]:
    states = []
    for path in files(repo, rev, "shards"):
        if path.endswith("/status.json") and path.count("/") == 2:
            try:
                states.append(json.loads(blob(repo, rev, path)))
            except (json.JSONDecodeError, subprocess.CalledProcessError):
                states.append({"shard": path.split("/")[1], "state": "unreadable"})
    return sorted(states, key=lambda s: str(s.get("shard", "")))


def summarise(states: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in states:
        counts[str(s.get("state", "?"))] = counts.get(str(s.get("state", "?")), 0) + 1
    return counts


def collect(repo: Path, rev: str, dest: Path) -> list[dict]:
    """Copy shard files and outputs from the results branch into `dest`, with digests.

    `shards/<id>/files/<p>` lands at `dest/shards/<id>/<p>`; `outputs/<p>` at
    `dest/outputs/<p>`. Logs are copied too: a failed shard is judged by its log.
    """
    manifest = []
    for path in files(repo, rev):
        parts = path.split("/")
        if parts[0] == "shards" and len(parts) >= 4 and parts[2] == "files":
            target = Path("shards", parts[1], *parts[3:])
        elif parts[0] == "shards" and len(parts) == 3 and parts[2] == "log":
            target = Path("shards", parts[1], "log")
        elif parts[0] == "outputs" and len(parts) >= 2:
            target = Path(*parts)
        else:
            continue
        data = blob(repo, rev, path)
        out = dest / target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        manifest.append({"path": target.as_posix(), "source": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return manifest
