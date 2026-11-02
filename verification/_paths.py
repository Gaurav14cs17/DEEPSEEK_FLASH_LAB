"""Repo path helpers for verification runners/tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = REPO_ROOT / "packages"
V4_PKG = PACKAGES / "deepseek_flash_v4_mini"
V41_PKG = PACKAGES / "deepseek_flash_v4_1_mini"
VERIFICATION = Path(__file__).resolve().parent
RESULTS = VERIFICATION / "results"


def ensure_pkg_paths() -> None:
    for p in (str(REPO_ROOT), str(PACKAGES), str(VERIFICATION)):
        if p not in sys.path:
            sys.path.insert(0, p)
