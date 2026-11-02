"""Load a model class + preset from one of the Mini packages."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = REPO / "packages"

if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))


def load(pkg: str, attr: str):
    """Return (model_class, preset) for e.g. 'deepseek_flash_v4_mini'."""
    mod = importlib.import_module(f"{pkg}.model")
    return getattr(mod, attr), getattr(mod, "preset")
