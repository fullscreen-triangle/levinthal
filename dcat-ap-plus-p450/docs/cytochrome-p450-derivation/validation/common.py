"""Shared paths, figure style and result recording for the validation experiments.

Every experiment writes one JSON file to results/ and its panels to the
figures/ directory of the manuscript it belongs to. Nothing here is specific
to one experiment.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIG_FOUNDATION = ROOT / "positive-occupation-propagation" / "figures"
FIG_P450 = ROOT / "p450-derivation" / "figures"

for d in (DATA / "pdb", DATA / "uniprot", RESULTS, FIG_FOUNDATION, FIG_P450):
    d.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Categorical slots 1-3 validate under all pairs; slots past three are used
# only in line and bar forms, where the adjacent pairlist applies.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
})


def panel_label(ax, letter: str) -> None:
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK, va="bottom", ha="left")


def save_figure(fig, directory: Path, name: str) -> Path:
    path = directory / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def record(name: str, payload: dict) -> Path:
    payload = {"experiment": name,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    raise TypeError(type(x))


def fetch(url: str, dest: Path) -> Path:
    """Download once; later runs read the cached copy. Records the sha256."""
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "p450-validation/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        time.sleep(0.2)
    manifest = DATA / "manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    key = dest.relative_to(DATA).as_posix()
    if key not in entries:
        entries[key] = {"url": url,
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                        "retrieved": time.strftime("%Y-%m-%d")}
        manifest.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return dest
