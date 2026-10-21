"""The experiment route end to end: check, plan, wrap, the shards, status, collect.

What is real: git (a forge of bare repositories on disk, the laptop checkouts, the
results branch), the driver's CLI, and the shell script it generates. That script runs
under bash on the stand-in AppHub below, so the flock-guarded clone of a required
repository at its pinned commit is executed, not just inspected. What is stood in:
lattice (stub_lattice.py) and AppHub (`run_on_apphub`), which follow lattice's README
for the unit and the results branch layout.

The forge is a host-less one (`forge = ""` in LATTICE_HOME/config.toml), so plain-path
remotes count as forge remotes; that is the only way the route is bent for the test.

    uv run --group dev pytest -m integration
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dcat_ap_plus_p450 import paths
from dcat_ap_plus_p450.experiments import cli, lattice, ledger, spec

pytestmark = pytest.mark.integration

STUB = Path(__file__).with_name("stub_lattice.py")
BASH = os.environ.get("P450_TEST_BASH") or shutil.which("bash")
EXP_ID = "e010-pipeline"

EXPERIMENT = '''
[experiment]
id = "e010-pipeline"
title = "Pipeline"
question = "Does the route carry a compare check from specification to verdict?"
components = ["compute.lattice"]

[target]
repo = "target"
command = """
base=$(cat ../dep/base.txt)
python3 - "$LATTICE_ARM" "$base" "$REAL_BONUS" > "$LATTICE_OUT/report.json" <<'PY'
import json, sys
arm, base, bonus = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
score = base + (bonus if arm == "real" else 0.1 * int(arm[-1]))
print(json.dumps(dict(value=score, arm=arm)))
PY
"""

[run]
matrix = { arm = ["real", "ctl1", "ctl2"], case = ["a", "b"] }
env = { REAL_BONUS = "@BONUS@" }
timeout_secs = 60

[[requires]]
repo = "dep"
ref = "HEAD"
dir = "../dep"

[[expect]]
kind = "shards"

[[expect]]
kind = "json"
path = "shards/*/report.json"
key = "value"
op = "ge"
value = 0

[[expect]]
kind = "compare"
path = "shards/arm-real_case-*/report.json"
key = "value"
against = "shards/arm-ctl*_case-$1/report.json"
better = "higher"
margin = 0.05
'''


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def checkout(path: Path, forge: Path, files: dict[str, str]) -> Path:
    """A laptop checkout with one commit, pushed to and fetched from its bare repository on the forge."""
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(forge)], check=True)
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@example.org")
    git(path, "config", "user.name", "t")
    for rel, text in files.items():
        (path / rel).write_text(text, encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-qm", "init")
    git(path, "remote", "add", "uni", forge.as_posix())
    git(path, "push", "-q", "uni", "main")
    git(path, "fetch", "-q", "uni")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Forge, laptop checkouts, a package root with one experiment, and the stand-in lattice."""
    def make(bonus: str) -> dict:
        forge = tmp_path / "forge"
        target = checkout(tmp_path / "laptop" / "target", forge / "target.git", {"README.md": "target\n"})
        dep = checkout(tmp_path / "laptop" / "dep", forge / "dep.git", {"base.txt": "0.0\n"})
        root = tmp_path / "pkg"
        (root / "docs" / "specifications").mkdir(parents=True)
        (root / "docs" / "specifications" / "components.json").write_text(
            json.dumps({"nodes": [{"id": "compute.lattice"}]}), encoding="utf-8")
        exp_dir = root / "experiments" / EXP_ID
        exp_dir.mkdir(parents=True)
        (root / "experiments" / "repos.toml").write_text(
            f'[repos.target]\npath = "{target.as_posix()}"\n\n[repos.dep]\npath = "{dep.as_posix()}"\n',
            encoding="utf-8")
        (exp_dir / "experiment.toml").write_text(EXPERIMENT.replace("@BONUS@", bonus), encoding="utf-8")
        home = tmp_path / "lattice-home"
        home.mkdir()
        (home / "config.toml").write_text('forge = ""\n', encoding="utf-8")
        log = tmp_path / "stub-lattice.jsonl"
        monkeypatch.setenv("P450_ROOT", str(root))
        monkeypatch.setenv("LATTICE_HOME", str(home))
        monkeypatch.setenv("P450_LATTICE", f'"{Path(sys.executable).as_posix()}" "{STUB.as_posix()}"')
        monkeypatch.setenv("STUB_LATTICE_LOG", str(log))
        monkeypatch.setenv("STUB_LATTICE_REMOTE", "uni")
        return {"tmp": tmp_path, "forge": forge, "target": target, "dep": dep, "exp_dir": exp_dir, "log": log}
    return make


def run_on_apphub(w: dict, unit_name: str) -> None:
    """Clone the target from the forge, run every shard of the unit, push lattice-results/<unit>."""
    site = w["tmp"] / "apphub"
    clone = site / "target"  # LATTICE_ROOT; the unit's `../dep` lands beside it
    subprocess.run(["git", "clone", "-q", str(w["forge"] / "target.git"), str(clone)], check=True)
    unit = json.loads((clone / ".lattice" / unit_name / "unit.json").read_text(encoding="utf-8"))
    results = site / "results"
    results.mkdir()
    git(results, "init", "-q", "-b", f"lattice-results/{unit_name}")
    git(results, "config", "user.email", "apphub@example.org")
    git(results, "config", "user.name", "apphub")
    for shard in unit["shards"]:
        sid = shard["id"]
        out = site / "out" / sid
        out.mkdir(parents=True)
        env = {**os.environ, **unit["env"], "LATTICE_ROOT": clone.as_posix(), "LATTICE_OUT": out.as_posix(),
               "LATTICE_SHARD": sid, **{f"LATTICE_{k.upper()}": v for k, v in shard["vars"].items()}}
        words = [BASH, *unit["words"][1:]] if unit["words"][0] == "bash" else unit["words"]
        p = subprocess.run(words, cwd=clone, env=env, capture_output=True, text=True, timeout=120, check=False)
        dest = results / "shards" / sid
        (dest / "files").mkdir(parents=True)
        (dest / "log").write_text(p.stdout + p.stderr, encoding="utf-8")
        (dest / "status.json").write_text(json.dumps({
            "shard": sid, "state": "done" if p.returncode == 0 else "failed",
            "exit_code": p.returncode, "host": "stand-in"}), encoding="utf-8")
        for f in out.rglob("*"):
            if f.is_file():
                to = dest / "files" / f.relative_to(out)
                to.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, to)
    git(results, "add", ".")
    git(results, "commit", "-qm", f"results of {unit_name}")
    git(results, "push", "-q", str(w["forge"] / "target.git"), f"HEAD:refs/heads/lattice-results/{unit_name}")


# The shards run under bash, take an flock and call python3. Where one is missing the
# shard tests skip, except under P450_INTEGRATION_STRICT=1 (set in CI), where they fail.
MISSING = [t for t, found in [("bash", BASH), ("flock", shutil.which("flock")),
                              ("python3", shutil.which("python3"))] if not found]
STRICT = os.environ.get("P450_INTEGRATION_STRICT") == "1"
needs_shell = pytest.mark.skipif(bool(MISSING) and not STRICT, reason=f"the shards need {', '.join(MISSING)}")


def route(w: dict, capsys) -> tuple[int, dict]:
    """check, plan, wrap, run, status, collect; returns collect's exit code and its record."""
    assert not MISSING, f"P450_INTEGRATION_STRICT is set but {', '.join(MISSING)} is missing"
    assert cli.main(["check", EXP_ID]) == cli.OK, capsys.readouterr().out
    assert cli.main(["plan", EXP_ID]) == cli.OK, capsys.readouterr().out
    assert len(list((w["exp_dir"] / "plans").glob("*.txt"))) == 1
    assert cli.main(["wrap", EXP_ID, "--yes"]) == cli.OK, capsys.readouterr().out
    run = ledger.latest_run(w["exp_dir"])
    wrap = ledger.read_json(run / "wrap.json")
    assert wrap["route"] == "direct" and wrap["returncode"] == 0
    assert wrap["requires"][0]["sha"] == git(w["dep"], "rev-parse", "HEAD").strip()
    run_on_apphub(w, wrap["unit"])
    assert cli.main(["status", EXP_ID]) == cli.OK, capsys.readouterr().out
    code = cli.main(["collect", EXP_ID])
    capsys.readouterr()
    return code, ledger.read_json(ledger.latest_file(run, "collect"))


@needs_shell
def test_route_concludes_when_the_real_arm_beats_its_best_control(world, capsys):
    w = world("0.3")
    code, record = route(w, capsys)
    assert code == cli.OK and record["overall"] == "concluded", record["checks"]
    compare = record["checks"][-1]
    assert compare["kind"] == "compare" and compare["verdict"] == "pass"
    assert [p["observed"] for p in compare["pairs"]] == [
        "shards/arm-real_case-a/report.json", "shards/arm-real_case-b/report.json"]
    for p in compare["pairs"]:
        case = p["observed"].split("case-")[1].split("/")[0]
        assert p["best_file"] == f"shards/arm-ctl2_case-{case}/report.json"
        assert p["alternatives"] == 2 and p["lead"] == pytest.approx(0.1)
    # the committed unit went to the forge with the branch, and nothing else was committed
    assert git(w["forge"] / "target.git", "log", "-1", "--format=%s", "main").strip() == "lattice: unit p450-e010-pipeline"
    calls = [json.loads(line) for line in w["log"].read_text(encoding="utf-8").splitlines()]
    plan, wrap = (c for c in calls if c[2:3] in (["plan"], ["wrap"]))
    assert plan[3:] == wrap[3:], "plan and wrap must be given the same unit"
    assert plan[plan.index("--") + 1:][:2] == ["bash", "-c"]


@needs_shell
def test_route_fails_when_a_control_is_as_good(world, capsys):
    w = world("0.15")
    code, record = route(w, capsys)
    assert code == cli.PROBLEMS and record["overall"] == "failed"
    compare = record["checks"][-1]
    assert compare["verdict"] == "fail" and "does not beat" in compare["detail"]
    assert all(c["verdict"] == "pass" for c in record["checks"][:-1])


def test_wrap_is_refused_before_lattice_when_a_required_commit_is_not_on_the_forge(world, capsys):
    w = world("0.3")
    (w["dep"] / "base.txt").write_text("1.0\n", encoding="utf-8")
    git(w["dep"], "commit", "-qam", "unpushed")
    assert cli.main(["wrap", EXP_ID, "--yes"]) == cli.BLOCKED
    assert "is not on uni" in capsys.readouterr().out
    assert not w["log"].exists() or "wrap" not in w["log"].read_text(encoding="utf-8")


def test_console_script_lists_and_checks_the_committed_experiments(monkeypatch):
    """The installed entry point against this repository's own experiments."""
    exe = shutil.which("dcat-ap-plus-p450")
    if exe is None:
        pytest.skip("console script not installed (uv sync installs it)")
    env = {k: v for k, v in os.environ.items() if k not in {"P450_ROOT", "P450_LATTICE"}}
    out = subprocess.run([exe, "experiment", "list", "--json"], capture_output=True, text=True, env=env, check=True).stdout
    listed = {r["id"] for r in json.loads(out)}
    on_disk = {d.name for d in spec.discover(paths.experiments_dir())}
    assert listed == on_disk and listed
    for exp_id in sorted(listed):
        r = subprocess.run([exe, "experiment", "check", exp_id], capture_output=True, text=True, env=env, check=False)
        # an unreachable repository or a missing lattice blocks (2); a broken specification is a problem (1)
        assert r.returncode in {cli.OK, cli.BLOCKED}, r.stdout
        assert "problem" not in r.stdout, r.stdout


def test_real_lattice_accepts_what_the_driver_generates(tmp_path):
    """Only where lattice is installed: `lattice plan` on every committed experiment's command line."""
    if os.environ.get("P450_LATTICE") or shutil.which("lattice") is None:
        pytest.skip("lattice is not on PATH")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "init")
    resolved = [lattice.Resolved(spec.Require("dep", "HEAD", "../dep"), "https://example.org/dep.git", "a" * 40)]
    for d in spec.discover(paths.experiments_dir()):
        exp, _ = spec.load(d)
        if exp.task:
            continue
        r = lattice.run(lattice.argv("plan", exp, str(repo), resolved if exp.requires else []))
        assert r.returncode == 0, f"{exp.id}: {r.stderr}"
        assert "shard(s)" in r.stdout


def test_stub_shard_ids_match_lattice():
    """The stand-in names shards as lattice does (lattice/src/unit.rs test of the same name)."""
    sys.path.insert(0, str(STUB.parent))
    try:
        import stub_lattice
    finally:
        sys.path.pop(0)
    ids = [s["id"] for s in stub_lattice.shards({"seed": ["1", "2"], "model": ["llama3.2", "qwen 14b"]})]
    assert ids == ["model-llama3.2_seed-1", "model-llama3.2_seed-2", "model-qwen-14b_seed-1", "model-qwen-14b_seed-2"]
    assert stub_lattice.sanitize("a//b") == "a-b"
