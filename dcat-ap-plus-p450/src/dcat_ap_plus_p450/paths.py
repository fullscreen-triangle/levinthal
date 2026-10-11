"""Where things are, relative to the package checkout.

The package lives at `levinthal/dcat-ap-plus-p450`. The environment variable
`P450_ROOT` overrides the root, which is how the tests point everything at a
temporary directory.
"""
import os
from pathlib import Path


def package_root() -> Path:
    override = os.environ.get("P450_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def experiments_dir() -> Path:
    return package_root() / "experiments"


def components_json() -> Path:
    return package_root() / "docs" / "specifications" / "components.json"
