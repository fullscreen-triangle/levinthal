"""The control arm of e001: the same paper with its paragraphs dealt across sections.

Every heading stays where it is and every section keeps its number of paragraphs; only
which paragraph sits under which heading is shuffled, with a fixed seed. Text before the
first heading and after \\end{document} is left alone. Sections are split as okgg splits
them with `--unit section` at the default level 2 (\\section and \\subsection).

If okgg tells the real sections apart no better than these, the real section boundaries
carry no distinction of their own.

    python3 shuffle_sections.py paper.tex --seed 1 > control.tex

Standard library only: it runs on AppHub with the session's own Python.
"""
from __future__ import annotations

import argparse
import random
import re
import sys

HEADING_RE = re.compile(r"\\(?:section|subsection)\*?\{[^}]*\}")
BLANK_RE = re.compile(r"\n[ \t]*\n")


def shuffle_sections(text: str, seed: int) -> str:
    start = max(text.find("\\begin{document}"), 0)
    end = text.find("\\end{document}", start)
    end = len(text) if end < 0 else end
    body = text[start:end]
    marks = list(HEADING_RE.finditer(body))
    if not marks:
        return text
    # sections[i] = paragraphs between heading i and heading i+1
    sections: list[list[str]] = []
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        sections.append([p for p in BLANK_RE.split(body[m.end():stop]) if p.strip()])
    pool = [p for s in sections for p in s]
    random.Random(seed).shuffle(pool)
    out = [text[:start], body[:marks[0].start()]]
    taken = 0
    for m, s in zip(marks, sections):
        out.append(m.group(0) + "\n\n")
        out.append("\n\n".join(pool[taken:taken + len(s)]) + "\n\n")
        taken += len(s)
    out.append(text[end:])
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tex")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    with open(a.tex, encoding="utf-8") as f:
        sys.stdout.write(shuffle_sections(f.read(), a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
