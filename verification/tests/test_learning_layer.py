"""Smoke tests for the beginner learning layer (tour, notebooks, docs)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEARN = REPO / "learn"
NOTEBOOKS = LEARN / "notebooks"


def test_tour_runs_end_to_end() -> None:
    r = subprocess.run(
        [sys.executable, str(LEARN / "tour.py"), "--all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr
    assert "Tour complete" in r.stdout
    assert "REGRESSION" in r.stdout  # the honest measurement lesson is present


def test_notebooks_are_valid_and_executed() -> None:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert len(paths) == 5, paths
    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code_cells, f"{path.name} has no code cells"
        errors = [
            out
            for cell in code_cells
            for out in cell.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        assert not errors, f"{path.name} contains execution errors"


def test_learning_docs_exist() -> None:
    for name in (
        "LEARNING_PATH.md",
        "CODE_WALKTHROUGH.md",
        "EXERCISES.md",
        "SOLUTIONS.md",
    ):
        path = REPO / "docs" / name
        assert path.exists(), path
        assert path.stat().st_size > 500, path
