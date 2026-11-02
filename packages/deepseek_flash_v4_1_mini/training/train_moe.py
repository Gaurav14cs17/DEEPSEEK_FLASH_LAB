#!/usr/bin/env python3
"""MoE-focused training entry (same loop, logs expert load)."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES))

from deepseek_flash_v4_1_mini.training.pretrain import main

if __name__ == "__main__":
    main()
