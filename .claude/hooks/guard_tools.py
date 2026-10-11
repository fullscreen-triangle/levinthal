"""PreToolUse guard for the repo-analysis tools (see CLAUDE.md, "Analysis tools").

spraypaint does not honour .gitignore and indexes .claude/settings.local.json,
which holds a gateway credential. A committed `spraypaint ask` re-reads passages
from disk and prints them, so an unscoped ask can print that file. The only
fence 0.2.0 offers is `--scenes`, so every committed ask must name its scenes
and none of them may be `.claude` or `(root)`. Dry runs print diagnostics only
and are let through.

Also denies `spraypaint serve` on a non-loopback host (no auth, arbitrary file
reads) and routes srn-fleet commands that touch remote machines or spend money
to a prompt.
"""
import json
import re
import shlex
import sys

FORBIDDEN_SCENES = {".claude", "(root)", "root"}


def decide(kind, reason):
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": kind,
        "permissionDecisionReason": reason,
    }}, sys.stdout)
    sys.exit(0)


def segments(command):
    # Split on shell separators so `cd x && spraypaint ask ...` is still seen.
    for part in re.split(r"&&|\|\||;|\|", command):
        try:
            yield shlex.split(part, posix=True)
        except ValueError:
            yield part.split()


def check_spraypaint(argv):
    if "ask" in argv:
        if "--dry-run" in argv:
            return
        scenes = None
        for i, a in enumerate(argv):
            if a == "--scenes" and i + 1 < len(argv):
                scenes = argv[i + 1]
            elif a.startswith("--scenes="):
                scenes = a.split("=", 1)[1]
        if not scenes:
            decide("deny", "spraypaint ask must pass --scenes (e.g. --scenes "
                   "cytochrome,enzymes,models,docs): the index holds "
                   ".claude/settings.local.json and an unscoped ask can print it. "
                   "Use --dry-run to explore without scenes.")
        bad = {s.strip() for s in scenes.split(",")} & FORBIDDEN_SCENES
        if bad:
            decide("deny", f"spraypaint ask may not include scene(s) {sorted(bad)}: "
                   "they contain credential-bearing files.")
    if "serve" in argv:
        host = None
        for i, a in enumerate(argv):
            if a == "--host" and i + 1 < len(argv):
                host = argv[i + 1]
            elif a.startswith("--host="):
                host = a.split("=", 1)[1]
        if host and host not in ("127.0.0.1", "localhost", "::1"):
            decide("deny", "spraypaint serve on a non-loopback host exposes "
                   "arbitrary file reads with no authentication.")
        decide("ask", "spraypaint serve reads arbitrary repo files; confirm.")


def check_srn_fleet(argv):
    touching = {"index", "run", "clear", "serve"}
    if touching & set(argv) or ("burst" in argv and "--live" in argv):
        decide("ask", "srn-fleet: this command reaches remote machines over SSH "
               "or provisions billable Vultr capacity.")


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    command = (payload.get("tool_input") or {}).get("command") or ""
    for argv in segments(command):
        if not argv:
            continue
        exe = re.sub(r"(\.exe)?$", "", argv[0].replace("\\", "/").rsplit("/", 1)[-1])
        if exe == "spraypaint":
            check_spraypaint(argv[1:])
        elif exe == "srn-fleet":
            check_srn_fleet(argv[1:])


if __name__ == "__main__":
    main()
