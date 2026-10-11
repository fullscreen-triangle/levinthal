"""The repositories an experiment can run in, and whether lattice can reach them.

lattice gets a unit to AppHub through the university Gitea, by one of two routes
(lattice/src/publish.rs): a git push remote on the forge (`direct`), or an origin
in the repository's tracker manifest `.sync.toml` (`sync`). A repository with
neither is `unreachable`: lattice refuses to wrap it.

`lattice wrap` pushes the repository's whole current branch, not just the unit
commit, so `unpushed` counts the commits that would go up with it.
"""
from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_FORGE = "git.uni-greifswald.de"


@dataclass(frozen=True)
class Repo:
    key: str
    path: Path
    clone_url: str = ""  # override for [[requires]] clones when detection is not enough


@dataclass(frozen=True)
class Reach:
    route: str  # "direct" | "sync" | "unreachable"
    remote: str = ""
    url: str = ""
    detail: str = ""


def load_registry(path: Path) -> dict[str, Repo]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    root = Path(os.path.expanduser(data.get("root", "~")))
    repos = {}
    for key, entry in data.get("repos", {}).items():
        p = Path(os.path.expanduser(entry["path"]))
        repos[key] = Repo(key=key, path=p if p.is_absolute() else root / p, clone_url=entry.get("clone_url", ""))
    return repos


def forge_host() -> str:
    """The forge from lattice's machine config, as lattice reads it."""
    home = Path(os.environ.get("LATTICE_HOME", Path.home() / ".lattice"))
    cfg = home / "config.toml"
    if cfg.exists():
        try:
            return str(tomllib.loads(cfg.read_text(encoding="utf-8")).get("forge", DEFAULT_FORGE))
        except tomllib.TOMLDecodeError:
            pass
    return DEFAULT_FORGE


def git(repo: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=check,
    )


def _host(url: str) -> str:
    if "://" in url:
        return (urlparse(url).hostname or "").lower()
    if "@" in url and ":" in url:  # scp-like: git@host:owner/repo.git
        return url.split("@", 1)[1].split(":", 1)[0].lower()
    return ""


def reach(repo: Repo, forge: str) -> Reach:
    if not (repo.path / ".git").exists():
        return Reach("unreachable", detail=f"not a git checkout: {repo.path}")
    out = git(repo.path, "remote", "-v").stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "(push)" and _host(parts[1]) == forge.lower():
            return Reach("direct", remote=parts[0], url=parts[1], detail=f"push remote {parts[0]}")
    manifest = repo.path / ".sync.toml"
    if manifest.exists() and forge.lower() in manifest.read_text(encoding="utf-8", errors="replace").lower():
        return Reach("sync", detail="origin on the forge in .sync.toml (tracker pushes)")
    remotes = sorted({line.split()[0] for line in out.splitlines() if line.strip()})
    shown = ", ".join(remotes) if remotes else "none"
    return Reach("unreachable", detail=f"no push remote on {forge} and no .sync.toml origin there (remotes: {shown})")


def clone_url(repo: Repo, forge: str) -> str:
    """The URL AppHub clones a required repository from: the override, else its forge remote."""
    if repo.clone_url:
        return repo.clone_url
    r = reach(repo, forge)
    return r.url if r.route == "direct" else ""


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def branch(repo: Path) -> str:
    return git(repo, "branch", "--show-current").stdout.strip()


def dirty(repo: Path) -> int:
    return len([line for line in git(repo, "status", "--porcelain").stdout.splitlines() if line.strip()])


def resolve(repo: Path, ref: str) -> str:
    """A ref to a full commit id, or "" if it does not resolve locally."""
    r = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return r.stdout.strip() if r.returncode == 0 else ""


def unpushed(repo: Path, remote: str, br: str) -> int | None:
    """Commits on `br` not on `remote/br` as last fetched; None if the remote has no such branch."""
    if not remote or not br:
        return None
    if git(repo, "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{br}").returncode != 0:
        return None
    out = git(repo, "rev-list", "--count", f"{remote}/{br}..{br}").stdout.strip()
    return int(out) if out.isdigit() else None


def on_remote(repo: Path, remote: str, sha: str) -> bool:
    """Whether `sha` is contained in some remote-tracking branch of `remote` (as last fetched)."""
    if not remote or not sha:
        return False
    out = git(repo, "branch", "-r", "--contains", sha).stdout
    return any(line.strip().startswith(f"{remote}/") for line in out.splitlines())
