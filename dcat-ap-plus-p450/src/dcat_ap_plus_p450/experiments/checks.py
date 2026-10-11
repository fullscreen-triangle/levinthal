"""Judging a collected run against the experiment's `[[expect]]` checks.

Three verdicts. `pass` and `fail` are judgements; `skipped` means the check could not
be judged (its input was never produced), and it is never counted as a pass. A run
concludes only when every check passes. Paths may contain `*`, matched against
the collected files; a `json` check then has to hold in every file it matches.
"""
from __future__ import annotations

import fnmatch
import json
import operator
from dataclasses import dataclass
from pathlib import Path

from .spec import Check

OPS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


@dataclass(frozen=True)
class Verdict:
    check: Check
    verdict: str  # "pass" | "fail" | "skipped"
    detail: str

    def as_record(self) -> dict:
        return {"kind": self.check.kind, "path": self.check.path, "key": self.check.key, "op": self.check.op,
                "value": self.check.value, "note": self.check.note, "verdict": self.verdict, "detail": self.detail}


def _matches(collected: list[str], pattern: str) -> list[str]:
    return sorted(p for p in collected if fnmatch.fnmatchcase(p, pattern))


def _lookup(doc: object, key: str) -> tuple[bool, object]:
    cur = doc
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def judge(check: Check, states: list[dict], root: Path, collected: list[str]) -> Verdict:
    if check.kind == "shards":
        if not states:
            return Verdict(check, "skipped", "no shard reported a state")
        bad = [f"{s.get('shard')}={s.get('state')}" for s in states if s.get("state") != "done"]
        return Verdict(check, "fail" if bad else "pass", ", ".join(bad) if bad else f"{len(states)} shards done")

    found = _matches(collected, check.path)
    if check.kind == "file":
        return Verdict(check, "pass" if found else "fail", f"{len(found)} file(s)" if found else "no such file collected")

    if not found:
        return Verdict(check, "skipped", f"no collected file matches {check.path}")
    failures = []
    for rel in found:
        try:
            doc = json.loads((root / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            failures.append(f"{rel}: not JSON ({e})")
            continue
        present, actual = _lookup(doc, check.key)
        if check.op == "exists":
            if not present:
                failures.append(f"{rel}: {check.key} absent")
        elif not present:
            failures.append(f"{rel}: {check.key} absent")
        else:
            try:
                if not OPS[check.op](actual, check.value):
                    failures.append(f"{rel}: {check.key}={actual!r} not {check.op} {check.value!r}")
            except TypeError:
                failures.append(f"{rel}: {check.key}={actual!r} cannot be compared with {check.value!r}")
    if failures:
        return Verdict(check, "fail", "; ".join(failures))
    return Verdict(check, "pass", f"holds in {len(found)} file(s)")


def judge_all(checks: tuple[Check, ...], states: list[dict], root: Path, collected: list[str]) -> list[Verdict]:
    return [judge(c, states, root, collected) for c in checks]


def overall(verdicts: list[Verdict]) -> str:
    """`concluded` only if every check passed; any failure is `failed`; else `incomplete`."""
    if any(v.verdict == "fail" for v in verdicts):
        return "failed"
    if any(v.verdict == "skipped" for v in verdicts):
        return "incomplete"
    return "concluded"
