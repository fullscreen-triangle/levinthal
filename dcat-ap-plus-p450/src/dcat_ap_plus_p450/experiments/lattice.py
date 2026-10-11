"""Turning an experiment into lattice command lines, and running them.

lattice's run.sh executes a unit's command as `bash -c` over the shell-quoted
words given after `--` (lattice/src/render.rs, run.sh). A single string such as
"cd x && y" would therefore become one quoted word and fail, so a command is
always handed over as `bash -c '<script>'`. That script is also where the clones
of required repositories go: each at a pinned commit, under an flock because all
shards of a unit share one checkout and start at once.

Shards see LATTICE_ROOT, LATTICE_OUT and LATTICE_<KEY> for every matrix key; use
those in commands rather than lattice's `{key}` placeholders.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass

from .spec import Experiment, Require


@dataclass(frozen=True)
class Resolved:
    """A required repository, resolved on the laptop at plan or wrap time."""

    require: Require
    url: str
    sha: str


def bootstrap(resolved: list[Resolved]) -> str:
    """Shell lines that put each required repository beside the target at its pinned commit."""
    if not resolved:
        return ""
    lines = ["set -euo pipefail", "(", "  flock 9"]
    for r in resolved:
        d = f'"$LATTICE_ROOT"/{shlex.quote(r.require.dir)}'
        url, sha = shlex.quote(r.url), shlex.quote(r.sha)
        lines += [
            f"  [ -d {d}/.git ] || git clone --quiet {url} {d}",
            # `^0` rather than `^{commit}`: lattice would read `{commit}` as a placeholder.
            f"  git -C {d} cat-file -e {sha}^0 2>/dev/null || git -C {d} fetch --quiet origin",
            f"  git -C {d} checkout --quiet --detach {sha}",
        ]
    lines.append(') 9>"$LATTICE_ROOT/.lattice-requires.lock"')
    return "\n".join(lines) + "\n"


def command_words(exp: Experiment, resolved: list[Resolved]) -> list[str]:
    """What follows `--` on the lattice command line."""
    return ["bash", "-c", bootstrap(resolved) + exp.command]


def options(exp: Experiment) -> list[str]:
    """The lattice options a specification sets, shared by plan and wrap."""
    a = ["--name", exp.unit]
    for k, v in exp.matrix.items():
        a += ["--matrix", f"{k}={v}"]
    if exp.each:
        a += ["--each", exp.each]
    for o in exp.outputs:
        a += ["--output", o]
    for k, v in exp.env.items():
        a += ["--env", f"{k}={v}"]
    for s in exp.secrets:
        a += ["--secret", s]
    if exp.gpu:
        a += ["--gpu", exp.gpu]
    if exp.parallel:
        a += ["--parallel", str(exp.parallel)]
    if exp.timeout_secs:
        a += ["--timeout", str(exp.timeout_secs)]
    return a


def argv(verb: str, exp: Experiment, repo_path: str, resolved: list[Resolved], extra: list[str] | None = None) -> list[str]:
    """The full command line for `lattice plan` or `lattice wrap`."""
    if verb not in {"plan", "wrap"}:
        raise ValueError(verb)
    a = ["lattice", "-C", repo_path, verb]
    if exp.task:
        a.append(exp.task)
    a += options(exp) + (extra or [])
    if exp.command:
        a += ["--", *command_words(exp, resolved)]
    return a


def available() -> bool:
    return shutil.which("lattice") is not None


def version() -> str:
    if not available():
        return ""
    r = subprocess.run(["lattice", "--version"], capture_output=True, text=True)
    return r.stdout.strip()


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")


def display(args: list[str]) -> str:
    """A copy-pasteable rendering of a command line."""
    return shlex.join(args)
