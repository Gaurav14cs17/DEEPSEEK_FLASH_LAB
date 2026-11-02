#!/usr/bin/env python3
"""Parameter counts for qwen3_vl_mini presets."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES))

from qwen3_vl_mini import Qwen3VLMini, preset


def main() -> None:
    for preset_name in ("demo", "pc", "better"):
        cfg = preset(preset_name)
        for label, cls in (('Qwen3VLMini', Qwen3VLMini),):
            model = cls(cfg)
            print(f"{preset_name:8s} {label:16s} params={model.num_parameters():,}")
            break


if __name__ == "__main__":
    main()
