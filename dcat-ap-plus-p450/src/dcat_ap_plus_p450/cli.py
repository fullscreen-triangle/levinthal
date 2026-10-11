"""`dcat-ap-plus-p450`: the package's command line.

Only `experiment` exists so far; fetch, emit, admit, ground, catalogue, artefacts and
evaluate are added by the work packages that build them (components-graph.md § 10).
"""
import argparse
import sys

from .experiments import cli as experiments_cli


def main(argv: list[str] | None = None) -> int:
    # lattice prints box-drawing characters; a Windows console defaults to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="dcat-ap-plus-p450")
    sub = parser.add_subparsers(dest="cmd", required=True)
    experiments_cli.add_parser(sub)
    a = parser.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
