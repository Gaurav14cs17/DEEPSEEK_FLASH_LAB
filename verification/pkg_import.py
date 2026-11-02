"""Import helpers for the two Mini packages.

Both packages have distinct top-level names under `packages/`, so there are no
more `model`/`optimization` collisions to work around.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
PACKAGES = REPO / "packages"

if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))


def import_from_pkg(pkg: str, module: str) -> ModuleType:
    """Import e.g. import_from_pkg("deepseek_flash_v4_mini", "model.rope")."""
    return importlib.import_module(f"{pkg}.{module}")
