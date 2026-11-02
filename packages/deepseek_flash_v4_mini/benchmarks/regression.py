#!/usr/bin/env python3
"""Thin wrapper: python packages/deepseek_flash_v4_mini/benchmarks/regression.py"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench.regression import main

if __name__ == "__main__":
    main()
