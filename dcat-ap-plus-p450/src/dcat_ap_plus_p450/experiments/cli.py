"""`dcat-ap-plus-p450 experiment ...`: plan here, run on AppHub through lattice, judge here.

    doctor              lattice, the forge, and which registered repositories lattice can reach
    list                every experiment and where it stands
    new ID --repo KEY   a new experiment from the template
    check ID            validate the specification and everything a wrap would need
    plan ID             `lattice plan`: what the unit needs on AppHub; writes nothing anywhere
    wrap ID             `lattice wrap`: commit the unit in the target repository and push it
    status ID           shard states from the unit's results branch
    collect ID          copy results back here, with digests, and judge the checks

Exit codes: 0 fine / concluded; 1 problems / a check failed; 2 blocked (unreachable,
lattice missing, or a lattice error); 3 incomplete (a check was skipped).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from . import checks, lattice, ledger, repos, results, spec

OK, PROBLEMS, BLOCKED, INCOMPLETE = 0, 1, 2, 3


@dataclass
class Preflight:
    exp: spec.Experiment
    repo: repos.Repo | None = None
    reach: repos.Reach | None = None
    problems: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved: list[lattice.Resolved] = field(default_factory=list)


def _registry() -> dict[str, repos.Repo]:
    return repos.load_registry(paths.experiments_dir() / "repos.toml")


def _exp_dir(exp_id: str) -> Path:
    d = paths.experiments_dir() / exp_id
    if not (d / "experiment.toml").exists():
        sys.exit(f"no experiment {exp_id!r} (looked for {d / 'experiment.toml'})")
    return d


def preflight(exp_id: str) -> Preflight:
    registry, forge = _registry(), repos.forge_host()
    exp, problems = spec.load(_exp_dir(exp_id))
    pf = Preflight(exp=exp, problems=problems)
    pf.problems += spec.validate(exp, spec.component_ids(paths.components_json()), set(registry))
    if not lattice.available():
        pf.blockers.append("lattice is not on PATH (cargo install --path <pylon>/lattice)")
    pf.repo = registry.get(exp.repo)
    if pf.repo is None:
        return pf
    pf.reach = repos.reach(pf.repo, forge)
    if pf.reach.route == "unreachable":
        pf.blockers.append(f"{exp.repo}: {pf.reach.detail}")
    elif (n := repos.dirty(pf.repo.path)):
        pf.warnings.append(f"{exp.repo} has {n} uncommitted change(s); only committed files reach AppHub")
    for req in exp.requires:
        other = registry.get(req.repo)
        if other is None:
            continue
        url = repos.clone_url(other, forge)
        sha = repos.resolve(other.path, req.ref)
        if not url:
            pf.blockers.append(f"requires {req.repo}: no clone URL on {forge} (add a forge remote or clone_url in repos.toml)")
        if not sha:
            pf.blockers.append(f"requires {req.repo}: ref {req.ref!r} does not resolve in {other.path}")
        elif other.clone_url:
            pf.warnings.append(f"requires {req.repo}: clone_url is set by hand, so whether {sha[:12]} is pushed cannot be checked")
        elif url:
            remote = repos.reach(other, forge).remote
            if not repos.on_remote(other.path, remote, sha):
                pf.blockers.append(f"requires {req.repo}: {sha[:12]} is not on {remote} as last fetched (push it, or AppHub cannot fetch it)")
        if url and sha:
            pf.resolved.append(lattice.Resolved(req, url, sha))
    return pf


def _print_preflight(pf: Preflight) -> int:
    for p in pf.problems:
        print(f"  problem  {p}")
    for b in pf.blockers:
        print(f"  blocked  {b}")
    for w in pf.warnings:
        print(f"  warning  {w}")
    if pf.problems:
        return PROBLEMS
    if pf.blockers:
        return BLOCKED
    print("  ok")
    return OK


# ---- commands -------------------------------------------------------------------------------

def cmd_doctor(_: argparse.Namespace) -> int:
    forge = repos.forge_host()
    print(f"lattice   {lattice.version() or 'NOT FOUND on PATH'}")
    print(f"forge     {forge}")
    print(f"graph     {paths.components_json()}")
    print()
    print(f"{'repository':22} {'route':12} {'branch':10} {'dirty':>5} {'unpushed':>8}  detail")
    reachable = 0
    for key, repo in sorted(_registry().items()):
        r = repos.reach(repo, forge)
        if r.route == "unreachable" and not (repo.path / ".git").exists():
            print(f"{key:22} {'missing':12} {'':10} {'':>5} {'':>8}  {repo.path}")
            continue
        br = repos.branch(repo.path)
        up = repos.unpushed(repo.path, r.remote, br) if r.route == "direct" else None
        reachable += r.route != "unreachable"
        up_s = "" if r.route != "direct" else ("new" if up is None else str(up))
        print(f"{key:22} {r.route:12} {br:10} {repos.dirty(repo.path):>5} {up_s:>8}  {r.detail}")
    print()
    print(f"{reachable} of {len(_registry())} repositories reachable by lattice.")
    print("An unreachable repository needs a push remote on the forge, or a tracker origin there")
    print("(.sync.toml). Hide private paths such as .claude/ from that origin before its first push.")
    return OK if lattice.available() else BLOCKED


def _state(exp_dir: Path) -> str:
    run = ledger.latest_run(exp_dir)
    if run is None:
        return "planned" if any((exp_dir / "plans").glob("*.txt")) else "specified"
    if (c := ledger.latest_file(run, "collect")):
        return ledger.read_json(c).get("overall", "collected")
    if (s := ledger.latest_file(run, "status")):
        summary = ledger.read_json(s).get("summary", {})
        return "results: " + ", ".join(f"{k} {v}" for k, v in sorted(summary.items())) if summary else "awaiting AppHub"
    wrap = run / "wrap.json"
    return "wrapped" if wrap.exists() and ledger.read_json(wrap).get("returncode") == 0 else "wrap failed"


def cmd_list(a: argparse.Namespace) -> int:
    rows = []
    for d in spec.discover(paths.experiments_dir()):
        exp, _ = spec.load(d)
        rows.append({"id": exp.id, "title": exp.title, "repo": exp.repo, "unit": exp.unit,
                     "components": list(exp.components), "runs": len(ledger.runs(d)), "state": _state(d)})
    if a.json:
        print(json.dumps(rows, indent=2))
        return OK
    for r in rows:
        print(f"{r['id']:34} {r['repo']:18} {r['runs']:>3} run(s)  {r['state']}")
        print(f"{'':34} {r['title']}")
    return OK


def cmd_new(a: argparse.Namespace) -> int:
    if not spec.ID_RE.match(a.id):
        sys.exit(f"id {a.id!r} must look like e001-short-name")
    if a.repo not in _registry():
        sys.exit(f"repo {a.repo!r} is not in experiments/repos.toml")
    root = paths.experiments_dir()
    dest = root / a.id
    if dest.exists():
        sys.exit(f"{dest} already exists")
    text = (root / "_template" / "experiment.toml").read_text(encoding="utf-8")
    text = text.replace("e000-template", a.id).replace('repo = "levinthal"', f'repo = "{a.repo}"')
    if a.title:
        text = text.replace('title = ""', f"title = {json.dumps(a.title)}")
    ledger.write_once(dest / "experiment.toml", text)
    print(f"wrote {dest / 'experiment.toml'}; fill it in, then: dcat-ap-plus-p450 experiment check {a.id}")
    return OK


def cmd_check(a: argparse.Namespace) -> int:
    pf = preflight(a.id)
    print(f"{pf.exp.id}  ->  {pf.exp.repo}  unit {pf.exp.unit}  route {pf.reach.route if pf.reach else '?'}")
    return _print_preflight(pf)


def cmd_plan(a: argparse.Namespace) -> int:
    pf = preflight(a.id)
    code = _print_preflight(pf)
    if pf.problems or pf.repo is None or not lattice.available():
        return code
    args = lattice.argv("plan", pf.exp, str(pf.repo.path), pf.resolved)
    print(f"\n$ {lattice.display(args)}\n")
    r = lattice.run(args)
    out = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "")
    # Record before printing: the record must not depend on the console.
    path = ledger.write_once(pf.exp.dir / "plans" / f"{ledger.stamp()}.txt",
                             f"$ {lattice.display(args)}\n# spec sha256 {pf.exp.digest}\n# exit {r.returncode}\n\n{out}")
    print(out)
    print(f"recorded {path}")
    return OK if r.returncode == 0 else BLOCKED


def cmd_wrap(a: argparse.Namespace) -> int:
    pf = preflight(a.id)
    code = _print_preflight(pf)
    if code != OK:
        print("\nnot wrapping: resolve the problems and blockers above first")
        return code
    exp, repo, reach = pf.exp, pf.repo, pf.reach
    assert repo is not None and reach is not None
    br = repos.branch(repo.path)
    up = repos.unpushed(repo.path, reach.remote, br) if reach.route == "direct" else None
    print(f"\nThis commits .lattice/{exp.unit}/ in {repo.path} on branch {br!r}", end="")
    if a.no_push:
        print(" and does not push.")
    elif reach.route == "sync":
        print(" and hands the push to `tracker sync run`, which pushes each origin its projection.")
    else:
        pushed = "the branch is not on the remote yet, so its whole history" if up is None else f"{up} commit(s) already on the branch"
        print(f" and pushes it to {reach.remote} ({reach.url}).\nAlso pushed: {pushed}.")
    if not a.yes:
        try:
            if input("Proceed? [y/N] ").strip().lower() != "y":
                print("not wrapped")
                return OK
        except EOFError:
            print("\nno answer (not a terminal); pass --yes to wrap without asking")
            return BLOCKED
    run_dir = paths.experiments_dir() / exp.id / "runs" / ledger.stamp()
    extra = (["--force"] if a.force else []) + (["--no-push"] if a.no_push else [])
    args = lattice.argv("wrap", exp, str(repo.path), pf.resolved, extra)
    before = repos.head(repo.path)
    r = lattice.run(args)
    ledger.write_once(run_dir / "wrap.txt", r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else ""))
    ledger.write_json(run_dir / "wrap.json", {
        "experiment": exp.id, "spec_sha256": exp.digest, "unit": exp.unit, "repo": exp.repo,
        "repo_path": str(repo.path), "branch": br, "route": reach.route, "remote": reach.remote,
        "head_before": before, "head_after": repos.head(repo.path), "pushed": not a.no_push,
        "requires": [{"repo": x.require.repo, "dir": x.require.dir, "url": x.url, "sha": x.sha} for x in pf.resolved],
        "lattice": lattice.version(), "argv": args, "returncode": r.returncode,
    })
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        print(f"lattice wrap failed; recorded in {run_dir}")
        return BLOCKED
    print(f"recorded {run_dir}")
    print("\nOn AppHub, in a Code-Server terminal (lattice cannot start the session):")
    print(f"  cd <{repo.path.name}> && git pull     # or git clone {reach.url or '<forge url>'}")
    print(f"  bash .lattice/{exp.unit}/run.sh --detach")
    print(f"\nThen, from here: dcat-ap-plus-p450 experiment status {exp.id}")
    return OK


def _run_for(exp_dir: Path, run_id: str | None) -> Path:
    run = exp_dir / "runs" / run_id if run_id else ledger.latest_run(exp_dir)
    if run is None or not (run / "wrap.json").exists():
        sys.exit("no wrapped run recorded for this experiment (wrap it first)")
    return run


def _fetch_for(run: Path) -> tuple[Path, results.Fetched]:
    w = ledger.read_json(run / "wrap.json")
    repo_path = Path(w["repo_path"])
    if w["route"] != "direct":
        return repo_path, results.Fetched(error=f"route {w['route']}: read results with `lattice -C {repo_path} results {w['unit']}`")
    return repo_path, results.fetch(repo_path, w["remote"], w["unit"])


def cmd_status(a: argparse.Namespace) -> int:
    exp_dir = _exp_dir(a.id)
    run = _run_for(exp_dir, a.run)
    repo_path, fetched = _fetch_for(run)
    if fetched.error:
        print(fetched.error)
        return INCOMPLETE
    states = results.shard_states(repo_path, fetched.rev)
    summary = results.summarise(states)
    ledger.write_json(run / f"status-{ledger.stamp()}.json", {"rev": fetched.rev, "summary": summary, "shards": states})
    print(f"results {fetched.rev[:12]}  " + ", ".join(f"{k} {v}" for k, v in sorted(summary.items())))
    for s in states:
        print(f"  {s.get('shard', '?'):24} {s.get('state', '?'):8} exit {s.get('exit_code')}  {s.get('host', '')}")
    done = states and all(s.get("state") in results.STATES_FINAL for s in states)
    return OK if done else INCOMPLETE


def cmd_collect(a: argparse.Namespace) -> int:
    exp_dir = _exp_dir(a.id)
    exp, _ = spec.load(exp_dir)
    run = _run_for(exp_dir, a.run)
    repo_path, fetched = _fetch_for(run)
    if fetched.error:
        print(fetched.error)
        return INCOMPLETE
    t = ledger.stamp()
    dest = run / f"collect-{t}"
    if dest.exists():
        sys.exit(f"{dest} exists")
    manifest = results.collect(repo_path, fetched.rev, dest)
    states = results.shard_states(repo_path, fetched.rev)
    verdicts = checks.judge_all(exp.checks, states, dest, [m["path"] for m in manifest])
    overall = checks.overall(verdicts)
    ignored = _ignored_by_git(dest, [m["path"] for m in manifest])
    ledger.write_json(run / f"collect-{t}.json", {
        "rev": fetched.rev, "spec_sha256": exp.digest, "files": manifest, "shards": states,
        "checks": [v.as_record() for v in verdicts], "overall": overall, "ignored_by_git": ignored,
    })
    print(f"collected {len(manifest)} file(s) from {fetched.rev[:12]} into {dest}")
    if ignored:
        print(f"  warning  {len(ignored)} collected file(s) are ignored by git and will not be committed "
              f"(levinthal ignores e.g. *.csv, *.log, *.pdf): {', '.join(ignored[:5])}{' …' if len(ignored) > 5 else ''}")
    for v in verdicts:
        print(f"  {v.verdict.upper():8} {v.check.kind:6} {v.check.path or ''} {v.check.key or ''}  {v.detail}")
    print(f"overall: {overall}")
    return {"concluded": OK, "failed": PROBLEMS}.get(overall, INCOMPLETE)


def _ignored_by_git(dest: Path, rel_paths: list[str]) -> list[str]:
    """Collected files the repository's .gitignore rules would keep out of a commit."""
    if not rel_paths:
        return []
    # NUL-separated bytes, not text: on Windows, text-mode stdin turns "\n" into "\r\n",
    # git then sees "table.csv\r", and no pattern matches. Exit status 1 = nothing ignored.
    r = subprocess.run(
        ["git", "-C", str(dest), "check-ignore", "-z", "--stdin"],
        input="\0".join(rel_paths).encode("utf-8") + b"\0", capture_output=True, check=False,
    )
    return sorted(p for p in r.stdout.decode("utf-8").split("\0") if p)


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("experiment", help="plan, run on AppHub via lattice, and judge experiments")
    s = p.add_subparsers(dest="action", required=True)
    s.add_parser("doctor", help="lattice, the forge, and which repositories it can reach").set_defaults(fn=cmd_doctor)
    q = s.add_parser("list", help="every experiment and where it stands")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_list)
    q = s.add_parser("new", help="a new experiment from the template")
    q.add_argument("id")
    q.add_argument("--repo", required=True)
    q.add_argument("--title", default="")
    q.set_defaults(fn=cmd_new)
    for name, fn, text in [("check", cmd_check, "validate everything a wrap needs"),
                           ("plan", cmd_plan, "what the unit needs on AppHub; writes nothing")]:
        q = s.add_parser(name, help=text)
        q.add_argument("id")
        q.set_defaults(fn=fn)
    q = s.add_parser("wrap", help="commit the unit in the target repository and push it")
    q.add_argument("id")
    q.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    q.add_argument("--no-push", action="store_true", help="commit the unit but do not push")
    q.add_argument("--force", action="store_true", help="replace an existing unit of the same name")
    q.set_defaults(fn=cmd_wrap)
    for name, fn, text in [("status", cmd_status, "shard states from the results branch"),
                           ("collect", cmd_collect, "copy results here and judge the checks")]:
        q = s.add_parser(name, help=text)
        q.add_argument("id")
        q.add_argument("--run", help="a run id under runs/ (default: the latest)")
        q.set_defaults(fn=fn)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcat-ap-plus-p450 experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_parser(sub)
    a = parser.parse_args(["experiment", *(argv if argv is not None else sys.argv[1:])])
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
