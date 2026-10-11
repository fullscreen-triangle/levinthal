"""Experiment specifications: `experiments/<id>/experiment.toml`.

An experiment is planned here and run on AppHub through lattice. Its specification
says what it asks, which components of the graph it exercises, which repository it
runs in, the command, the shards, what it sends back, which other repositories it
needs at pinned commits, and how its result is judged.

Loading never raises on a bad specification: `load` returns the experiment and the
list of problems, so `check` can report all of them at once.
"""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Rules lattice itself enforces (lattice/src/unit.rs), checked here so a bad
# specification is refused before lattice is called.
ID_RE = re.compile(r"^e\d{3}-[a-z0-9][a-z0-9-]*$")
UNIT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MATRIX_KEY_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GPU_VALUES = {"", "none", "recommended", "required"}
# lattice reads every `{key}` not preceded by `$` as a matrix placeholder and refuses a
# command whose placeholders no matrix fills (lattice/src/unit.rs `unfilled`). That
# includes braces in code such as a Python f-string's `{e}`.
PLACEHOLDER_RE = re.compile(r"(^|[^$])\{([a-z_][a-z0-9_]*)\}")


def unfilled_placeholders(command: str, keys: set[str]) -> list[str]:
    """The `{key}` placeholders lattice would find in `command` that no shard variable fills."""
    return sorted({m.group(2) for m in PLACEHOLDER_RE.finditer(command)} - keys)
CHECK_KINDS = {"shards", "file", "json"}
JSON_OPS = {"exists", "eq", "ne", "lt", "le", "gt", "ge"}


@dataclass(frozen=True)
class Require:
    """Another repository the command needs, cloned beside the target on AppHub."""

    repo: str
    ref: str
    dir: str


@dataclass(frozen=True)
class Check:
    """One acceptance check, judged by `collect`.

    `shards`: every shard finished with state `done`.
    `file`: a collected file exists (`path` is relative to the run's collected files).
    `json`: a key in a collected JSON file satisfies `op` against `value`.
    """

    kind: str
    path: str = ""
    key: str = ""
    op: str = "exists"
    value: object = None
    note: str = ""


@dataclass(frozen=True)
class Experiment:
    id: str
    dir: Path
    digest: str
    title: str = ""
    question: str = ""
    components: tuple[str, ...] = ()
    repo: str = ""
    task: str = ""
    command: str = ""
    unit: str = ""
    matrix: dict[str, str] = field(default_factory=dict)
    each: str = ""
    outputs: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    secrets: tuple[str, ...] = ()
    gpu: str = ""
    parallel: int = 0
    timeout_secs: int = 0
    requires: tuple[Require, ...] = ()
    checks: tuple[Check, ...] = ()


def _matrix_values(value: object) -> str:
    """`[1, 2, 3]` or `"1..3"` or `"a,b"` -> the string lattice's `--matrix` takes."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def load(directory: Path) -> tuple[Experiment, list[str]]:
    """Read `experiment.toml` in `directory`; return the experiment and its problems."""
    path = directory / "experiment.toml"
    problems: list[str] = []
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        stub = Experiment(id=directory.name, dir=directory, digest="")
        return stub, [f"experiment.toml does not parse: {e}"]

    exp_t = data.get("experiment", {})
    tgt = data.get("target", {})
    run = data.get("run", {})
    known = {"experiment", "target", "run", "requires", "expect"}
    for extra in sorted(set(data) - known):
        problems.append(f"unknown table [{extra}] (known: {', '.join(sorted(known))})")

    requires = tuple(
        Require(repo=str(r.get("repo", "")), ref=str(r.get("ref", "HEAD")), dir=str(r.get("dir", "")))
        for r in data.get("requires", [])
    )
    checks = tuple(
        Check(
            kind=str(c.get("kind", "")),
            path=str(c.get("path", "")),
            key=str(c.get("key", "")),
            op=str(c.get("op", "exists")),
            value=c.get("value"),
            note=str(c.get("note", "")),
        )
        for c in data.get("expect", [])
    )
    exp_id = str(exp_t.get("id", directory.name))
    experiment = Experiment(
        id=exp_id,
        dir=directory,
        digest=hashlib.sha256(raw).hexdigest(),
        title=str(exp_t.get("title", "")),
        question=str(exp_t.get("question", "")),
        components=tuple(str(c) for c in exp_t.get("components", [])),
        repo=str(tgt.get("repo", "")),
        task=str(tgt.get("task", "")),
        command=str(tgt.get("command", "")),
        unit=str(tgt.get("unit", "") or f"p450-{exp_id}"),
        matrix={str(k): _matrix_values(v) for k, v in run.get("matrix", {}).items()},
        each=str(run.get("each", "")),
        outputs=tuple(str(o) for o in run.get("outputs", [])),
        env={str(k): str(v) for k, v in run.get("env", {}).items()},
        secrets=tuple(str(s) for s in run.get("secrets", [])),
        gpu=str(run.get("gpu", "")),
        parallel=int(run.get("parallel", 0)),
        timeout_secs=int(run.get("timeout_secs", 0)),
        requires=requires,
        checks=checks,
    )
    if exp_id != directory.name:
        problems.append(f"experiment.id {exp_id!r} differs from its directory {directory.name!r}")
    return experiment, problems


def validate(exp: Experiment, component_ids: set[str], repo_keys: set[str]) -> list[str]:
    """Problems that make the experiment unrunnable or unanchored. Empty means valid."""
    p: list[str] = []
    if not ID_RE.match(exp.id):
        p.append(f"id {exp.id!r} must look like e001-short-name (e + 3 digits, lowercase, hyphens)")
    if not exp.title:
        p.append("experiment.title is empty")
    if not exp.question:
        p.append("experiment.question is empty: say what the run is meant to decide")
    if not exp.components:
        p.append("experiment.components is empty: name the graph components it exercises")
    for c in exp.components:
        if c not in component_ids:
            p.append(f"component {c!r} is not in components.json")
    if exp.repo not in repo_keys:
        p.append(f"target.repo {exp.repo!r} is not in experiments/repos.toml")
    if bool(exp.task) == bool(exp.command):
        p.append("give exactly one of target.task (a lattice task) or target.command")
    if exp.task and exp.requires:
        p.append("[[requires]] needs target.command: a lattice task cannot be given a clone step")
    if not UNIT_RE.match(exp.unit):
        p.append(f"unit {exp.unit!r} must be letters, digits, '.', '_' or '-'")
    for k in exp.matrix:
        if not MATRIX_KEY_RE.match(k):
            p.append(f"matrix key {k!r} must be lowercase letters, digits or '_'")
    if exp.matrix and exp.each:
        p.append("use run.matrix or run.each, not both")
    keys = set(exp.matrix) | ({"file"} if exp.each else set())
    for k in unfilled_placeholders(exp.command, keys):
        p.append(f"command contains {{{k}}}, which lattice reads as a matrix placeholder that nothing fills; "
                 f"avoid {{name}} in code (e.g. no f-strings), or use ${{{k}}} for a shell variable")
    for name in [*exp.env, *exp.secrets]:
        if not ENV_NAME_RE.match(name):
            p.append(f"environment name {name!r} is not a valid variable name")
    if exp.gpu not in GPU_VALUES:
        p.append(f"run.gpu {exp.gpu!r} must be one of none, recommended, required")
    if exp.parallel < 0 or exp.timeout_secs < 0:
        p.append("run.parallel and run.timeout_secs cannot be negative")
    seen_dirs: set[str] = set()
    for r in exp.requires:
        if r.repo not in repo_keys:
            p.append(f"requires.repo {r.repo!r} is not in experiments/repos.toml")
        if r.repo == exp.repo:
            p.append(f"requires.repo {r.repo!r} is the target itself")
        if not r.dir or r.dir.startswith("/") or ".." not in Path(r.dir).parts[:1]:
            p.append(f"requires.dir {r.dir!r} must be relative and outside the target, e.g. ../{r.repo}")
        if r.dir in seen_dirs:
            p.append(f"requires.dir {r.dir!r} is used twice")
        seen_dirs.add(r.dir)
    if not exp.checks:
        p.append("no [[expect]] checks: an experiment with no acceptance criterion cannot conclude")
    for c in exp.checks:
        if c.kind not in CHECK_KINDS:
            p.append(f"expect.kind {c.kind!r} must be one of {sorted(CHECK_KINDS)}")
        if c.kind in {"file", "json"} and not c.path:
            p.append(f"expect.kind {c.kind!r} needs a path")
        if c.kind == "json":
            if not c.key:
                p.append("expect.kind 'json' needs a key")
            if c.op not in JSON_OPS:
                p.append(f"expect.op {c.op!r} must be one of {sorted(JSON_OPS)}")
            if c.op != "exists" and c.value is None:
                p.append(f"expect.op {c.op!r} needs a value")
    return p


def discover(root: Path) -> list[Path]:
    """Experiment directories under `root`, skipping `_template` and anything without a spec."""
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "experiment.toml").exists())


def component_ids(components_json: Path) -> set[str]:
    data = json.loads(components_json.read_text(encoding="utf-8"))
    return {n["id"] for n in data["nodes"]}
