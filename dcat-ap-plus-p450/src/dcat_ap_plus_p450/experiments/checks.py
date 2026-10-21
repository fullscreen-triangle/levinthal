"""Judging a collected run against the experiment's `[[expect]]` checks.

Three verdicts. `pass` and `fail` are judgements; `skipped` means the check could not
be judged (its input was never produced), and it is never counted as a pass. A run
concludes only when every check passes. Paths may contain `*`, matched against
the collected files; a `json` check then has to hold in every file it matches.

A `compare` check judges each file matching `path` against its alternatives, the files
matching `against` after `$1`, `$2`, … are replaced by what that file's `*`s matched.
It passes only if the observed value beats the best alternative by at least `margin`;
a tie is not a pass. If any alternative is missing or unreadable the pair cannot be
judged (the best option may be the one that is missing), so it is `skipped`.
"""
from __future__ import annotations

import fnmatch
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import CAPTURE_REF_RE, Check

OPS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


@dataclass(frozen=True)
class Verdict:
    check: Check
    verdict: str  # "pass" | "fail" | "skipped"
    detail: str
    pairs: tuple[dict, ...] = ()  # `compare` only: one entry per observed file

    def as_record(self) -> dict:
        record = {"kind": self.check.kind, "path": self.check.path, "key": self.check.key, "op": self.check.op,
                  "value": self.check.value, "note": self.check.note, "verdict": self.verdict, "detail": self.detail}
        if self.check.kind == "compare":
            record.update(against=self.check.against, better=self.check.better, margin=self.check.margin,
                          pairs=list(self.pairs))
        return record


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


def _capturing(pattern: str) -> re.Pattern[str]:
    """`pattern` as a regex in which each `*` is a group; `?` matches one character."""
    out = []
    for ch in pattern:
        out.append("(.*)" if ch == "*" else "." if ch == "?" else re.escape(ch))
    return re.compile("".join(out), re.DOTALL)


def _number(root: Path, rel: str, key: str) -> tuple[float | None, str]:
    """The numeric value of `key` in the JSON file `rel`, or None and the reason."""
    try:
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{rel}: not JSON ({e})"
    present, actual = _lookup(doc, key)
    if not present:
        return None, f"{rel}: {key} absent"
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return None, f"{rel}: {key}={actual!r} is not a number"
    return float(actual), ""


def _judge_compare(check: Check, root: Path, collected: list[str]) -> Verdict:
    observed = _capturing(check.path)
    found = [(rel, m.groups()) for rel in sorted(collected) if (m := observed.fullmatch(rel))]
    if not found:
        return Verdict(check, "skipped", f"no collected file matches {check.path}")
    higher = check.better == "higher"
    pairs: list[dict] = []
    for rel, groups in found:
        pattern = CAPTURE_REF_RE.sub(lambda m, g=groups: g[int(m.group(1)) - 1], check.against)
        alts = [a for a in _matches(collected, pattern) if a != rel]
        pair: dict = {"observed": rel, "against": pattern}
        value, why = _number(root, rel, check.key)
        if value is None:
            pairs.append({**pair, "verdict": "fail", "detail": why})
            continue
        pair["value"] = value
        if not alts:
            pairs.append({**pair, "verdict": "skipped", "detail": f"no collected file matches {pattern}"})
            continue
        values = []
        for a in alts:
            v, why = _number(root, a, check.key)
            if v is None:
                break
            values.append((v, a))
        if len(values) < len(alts):
            pairs.append({**pair, "verdict": "skipped", "detail": f"alternative unreadable: {why}"})
            continue
        best, best_rel = max(values) if higher else min(values)
        lead = value - best if higher else best - value
        pair.update(best=best, best_file=best_rel, alternatives=len(values), lead=lead)
        if lead > 0 and lead >= float(check.margin):
            pairs.append({**pair, "verdict": "pass",
                          "detail": f"{rel}: {value:g} beats {best:g} ({best_rel}) by {lead:g}"})
        elif lead > 0:
            pairs.append({**pair, "verdict": "fail",
                          "detail": f"{rel}: {value:g} leads {best:g} ({best_rel}) by {lead:g}, under the margin {check.margin}"})
        else:
            pairs.append({**pair, "verdict": "fail",
                          "detail": f"{rel}: {value:g} does not beat {best:g} ({best_rel})"})
    verdicts = {p["verdict"] for p in pairs}
    verdict = "fail" if "fail" in verdicts else "skipped" if "skipped" in verdicts else "pass"
    shown = [p["detail"] for p in pairs if p["verdict"] == verdict]
    summary = f"{len(shown)} of {len(pairs)} pair(s) {verdict}: " + "; ".join(shown[:5]) + (" …" if len(shown) > 5 else "")
    return Verdict(check, verdict, summary, tuple(pairs))


def judge(check: Check, states: list[dict], root: Path, collected: list[str]) -> Verdict:
    if check.kind == "shards":
        if not states:
            return Verdict(check, "skipped", "no shard reported a state")
        bad = [f"{s.get('shard')}={s.get('state')}" for s in states if s.get("state") != "done"]
        return Verdict(check, "fail" if bad else "pass", ", ".join(bad) if bad else f"{len(states)} shards done")

    if check.kind == "compare":
        return _judge_compare(check, root, collected)
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
