"""Tests for the experiment layer. No network, no lattice, no AppHub: git repositories
are built in temporary directories, and lattice's rules are checked as the driver
mirrors them (lattice/src/unit.rs, render.rs, publish.rs)."""
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from dcat_ap_plus_p450 import paths
from dcat_ap_plus_p450.experiments import checks, ledger, lattice, repos, results, spec

COMPONENTS = {"compute.lattice", "up.okgg"}
REPOS = {"levinthal", "conspirator"}

VALID = """
[experiment]
id = "e010-sample"
title = "A sample"
question = "Does it run?"
components = ["compute.lattice"]

[target]
repo = "levinthal"
command = "python3 run.py --seed $LATTICE_SEED"

[run]
matrix = { seed = [1, 2, 3] }
outputs = ["results/**"]
gpu = "none"
timeout_secs = 60

[[expect]]
kind = "shards"
"""


def write_exp(tmp: Path, text: str, name: str = "e010-sample") -> Path:
    d = tmp / name
    d.mkdir()
    (d / "experiment.toml").write_text(textwrap.dedent(text), encoding="utf-8")
    return d


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    sh(path, "init", "-q", "-b", "main")
    sh(path, "config", "user.email", "t@example.org")
    sh(path, "config", "user.name", "t")
    return path


# ---- specifications ----------------------------------------------------------------------

def test_valid_spec_has_no_problems(tmp_path):
    exp, problems = spec.load(write_exp(tmp_path, VALID))
    assert problems == []
    assert spec.validate(exp, COMPONENTS, REPOS) == []
    assert exp.unit == "p450-e010-sample"
    assert exp.matrix == {"seed": "1,2,3"}


@pytest.mark.parametrize("change, expected", [
    (('question = "Does it run?"', 'question = ""'), "question is empty"),
    (('components = ["compute.lattice"]', 'components = ["nope"]'), "not in components.json"),
    (('repo = "levinthal"', 'repo = "elsewhere"'), "not in experiments/repos.toml"),
    (('command = "python3 run.py --seed $LATTICE_SEED"', 'command = "x"\ntask = "build"'), "exactly one of"),
    (('matrix = { seed = [1, 2, 3] }', 'matrix = { Seed = [1] }'), "matrix key 'Seed'"),
    (('gpu = "none"', 'gpu = "maybe"'), "run.gpu"),
    (('[[expect]]\nkind = "shards"', ''), "no [[expect]] checks"),
])
def test_invalid_specs_are_reported(tmp_path, change, expected):
    exp, problems = spec.load(write_exp(tmp_path, VALID.replace(*change)))
    assert any(expected in p for p in problems + spec.validate(exp, COMPONENTS, REPOS))


def test_directory_and_id_must_agree(tmp_path):
    _, problems = spec.load(write_exp(tmp_path, VALID, name="e011-other"))
    assert any("differs from its directory" in p for p in problems)


# ---- lattice's placeholder rule -----------------------------------------------------------

def test_placeholders_mirror_lattice():
    assert spec.unfilled_placeholders('print(f"x {e}")', set()) == ["e"]
    assert spec.unfilled_placeholders("echo ${e} $e", set()) == []        # shell variables
    assert spec.unfilled_placeholders("run --seed {seed}", {"seed"}) == []
    assert spec.unfilled_placeholders("{t: 1 for t in x}", set()) == []   # not a bare {name}
    assert spec.unfilled_placeholders("{Upper}", set()) == []             # lattice keys are lowercase


def test_fstring_in_command_is_a_problem(tmp_path):
    text = VALID.replace("python3 run.py --seed $LATTICE_SEED", 'python3 -c \\"print(f\'{e}\')\\"')
    exp, _ = spec.load(write_exp(tmp_path, text))
    assert any("{e}" in p for p in spec.validate(exp, COMPONENTS, REPOS))


# ---- command lines --------------------------------------------------------------------------

def test_argv_passes_command_through_bash(tmp_path):
    exp, _ = spec.load(write_exp(tmp_path, VALID))
    a = lattice.argv("plan", exp, "/repo", [])
    assert a[:4] == ["lattice", "-C", "/repo", "plan"]
    assert ["--matrix", "seed=1,2,3"] == a[a.index("--matrix"):a.index("--matrix") + 2]
    assert a[a.index("--") + 1:] == ["bash", "-c", "python3 run.py --seed $LATTICE_SEED"]
    for opt in ("--output", "--gpu", "--timeout", "--name"):
        assert opt in a


def test_requires_bootstrap_pins_serialises_and_has_no_placeholders():
    req = spec.Require(repo="conspirator", ref="HEAD", dir="../conspirator")
    script = lattice.bootstrap([lattice.Resolved(req, "https://git.uni-greifswald.de/x/conspirator.git", "a" * 40)])
    assert "flock 9" in script and "checkout --quiet --detach" in script and "a" * 40 in script
    # Regression: `^{commit}` made lattice refuse every multi-repo unit.
    assert spec.unfilled_placeholders(script, set()) == []


def test_wrap_rejects_unknown_verb(tmp_path):
    exp, _ = spec.load(write_exp(tmp_path, VALID))
    with pytest.raises(ValueError):
        lattice.argv("results", exp, "/repo", [])


# ---- reachability ------------------------------------------------------------------------

def test_reach_routes(tmp_path):
    repo = repos.Repo("r", git_init(tmp_path / "r"))
    assert repos.reach(repo, "git.uni-greifswald.de").route == "unreachable"
    sh(repo.path, "remote", "add", "gh", "https://github.com/x/r.git")
    assert repos.reach(repo, "git.uni-greifswald.de").route == "unreachable"
    (repo.path / ".sync.toml").write_text('[[origin]]\nurl = "https://git.uni-greifswald.de/x/r.git"\n')
    assert repos.reach(repo, "git.uni-greifswald.de").route == "sync"
    sh(repo.path, "remote", "add", "uni", "git@git.uni-greifswald.de:x/r.git")
    r = repos.reach(repo, "git.uni-greifswald.de")
    assert (r.route, r.remote) == ("direct", "uni")


def test_unpushed_is_none_without_a_tracking_branch(tmp_path):
    path = git_init(tmp_path / "r")
    (path / "a").write_text("a")
    sh(path, "add", "a")
    sh(path, "commit", "-qm", "a")
    assert repos.unpushed(path, "uni", "main") is None
    assert repos.resolve(path, "HEAD") == repos.head(path)


# ---- results branch ------------------------------------------------------------------------

def make_results(path: Path, states: dict[str, str]) -> str:
    """A repository whose branch lattice-results/u looks like what AppHub pushes."""
    git_init(path)
    (path / "seed").write_text("x")
    sh(path, "add", "seed")
    sh(path, "commit", "-qm", "seed")
    sh(path, "checkout", "-q", "--orphan", "lattice-results/u")
    sh(path, "rm", "-rq", "--cached", ".")
    for shard, state in states.items():
        d = path / "shards" / shard
        (d / "files").mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"shard": shard, "state": state, "exit_code": 0 if state == "done" else 1}))
        (d / "log").write_text("log")
        (d / "files" / "probe.json").write_text(json.dumps({"cpus": 4, "tools": {"flock": True}}))
    (path / "outputs" / "res").mkdir(parents=True)
    (path / "outputs" / "res" / "summary.json").write_text(json.dumps({"failing": 0, "total": 16}))
    sh(path, "add", "shards", "outputs")
    sh(path, "commit", "-qm", "results")
    return sh(path, "rev-parse", "HEAD").strip()


def test_collect_and_judge(tmp_path):
    rev = make_results(tmp_path / "r", {"a": "done", "b": "done"})
    states = results.shard_states(tmp_path / "r", rev)
    assert results.summarise(states) == {"done": 2}
    manifest = results.collect(tmp_path / "r", rev, tmp_path / "out")
    got = {m["path"] for m in manifest}
    assert {"shards/a/probe.json", "shards/a/log", "outputs/res/summary.json"} <= got
    assert all(len(m["sha256"]) == 64 for m in manifest)
    want = (
        spec.Check("shards"),
        spec.Check("file", path="shards/*/probe.json"),
        spec.Check("json", path="shards/*/probe.json", key="cpus", op="gt", value=0),
        spec.Check("json", path="shards/*/probe.json", key="tools.flock", op="eq", value=True),
        spec.Check("json", path="outputs/res/summary.json", key="failing", op="eq", value=0),
    )
    verdicts = checks.judge_all(want, states, tmp_path / "out", sorted(got))
    assert [v.verdict for v in verdicts] == ["pass"] * 5
    assert checks.overall(verdicts) == "concluded"


def test_failed_shard_fails_and_missing_input_is_skipped(tmp_path):
    rev = make_results(tmp_path / "r", {"a": "done", "b": "failed"})
    states = results.shard_states(tmp_path / "r", rev)
    v_shards = checks.judge(spec.Check("shards"), states, tmp_path, [])
    v_missing = checks.judge(spec.Check("json", path="nothing/*.json", key="k"), states, tmp_path, [])
    assert v_shards.verdict == "fail" and "b=failed" in v_shards.detail
    assert v_missing.verdict == "skipped"
    assert checks.overall([v_missing]) == "incomplete"
    assert checks.overall([v_missing, v_shards]) == "failed"


# ---- ledger ------------------------------------------------------------------------------

def test_ledger_never_overwrites(tmp_path):
    ledger.write_json(tmp_path / "runs" / "1" / "wrap.json", {"a": 1})
    with pytest.raises(FileExistsError):
        ledger.write_json(tmp_path / "runs" / "1" / "wrap.json", {"a": 2})
    assert ledger.latest_run(tmp_path).name == "1"


# ---- the experiments actually in this repository -------------------------------------------

def test_committed_experiments_are_valid():
    root = paths.experiments_dir()
    ids = spec.component_ids(paths.components_json())
    keys = set(repos.load_registry(root / "repos.toml"))
    dirs = spec.discover(root)
    assert dirs, "no experiments found"
    for d in dirs:
        exp, problems = spec.load(d)
        assert problems + spec.validate(exp, ids, keys) == [], d.name


# ---- status and collect, end to end, against a stand-in for the forge -------------------------

def test_status_and_collect_end_to_end(tmp_path, monkeypatch, capsys):
    from dcat_ap_plus_p450.experiments import cli

    # The forge: a bare repository holding lattice-results/u as AppHub would push it.
    work = tmp_path / "apphub"
    make_results(work, {"a": "done"})
    (work / "shards" / "a" / "files" / "table.csv").write_text("x,y\n")
    sh(work, "add", "shards")
    sh(work, "commit", "-qm", "csv")
    forge = tmp_path / "forge.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(forge)], check=True)

    # The target repository on the laptop, with the forge as its push remote.
    target = git_init(tmp_path / "target")
    sh(target, "remote", "add", "uni", str(forge))

    # A package root holding one experiment and one wrapped run pointing at the target.
    root = tmp_path / "pkg"
    (root / "docs" / "specifications").mkdir(parents=True)
    (root / "docs" / "specifications" / "components.json").write_text(json.dumps({"nodes": [{"id": "compute.lattice"}]}))
    (root / "experiments").mkdir()
    (root / "experiments" / "repos.toml").write_text(f'[repos.levinthal]\npath = "{target.as_posix()}"\n')
    exp_dir = write_exp(root / "experiments", VALID + textwrap.dedent("""
        [[expect]]
        kind = "json"
        path = "shards/*/probe.json"
        key = "cpus"
        op = "ge"
        value = 1
    """))
    git_init(root)  # so collected files can be checked against a .gitignore
    (root / ".gitignore").write_text("*.csv\n")
    ledger.write_json(exp_dir / "runs" / "20260101T000000Z" / "wrap.json", {
        "repo_path": str(target), "route": "direct", "remote": "uni", "unit": "u", "returncode": 0,
    })
    monkeypatch.setenv("P450_ROOT", str(root))

    assert cli.main(["status", "e010-sample"]) == cli.OK
    assert "done 1" in capsys.readouterr().out

    code = cli.main(["collect", "e010-sample"])
    out = capsys.readouterr().out
    assert code == cli.OK, out
    assert "overall: concluded" in out
    assert "ignored by git" in out and "table.csv" in out
    record = ledger.read_json(ledger.latest_file(exp_dir / "runs" / "20260101T000000Z", "collect"))
    assert record["overall"] == "concluded"
    assert record["ignored_by_git"] == ["shards/a/table.csv"]
    assert cli._state(exp_dir) == "concluded"
