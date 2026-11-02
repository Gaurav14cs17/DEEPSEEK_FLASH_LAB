#!/usr/bin/env python3
"""Parameter counts for every Qwen Mini family at the `pc` preset."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES))

from qwen2_5_mini import Qwen25Model, preset as p25  # noqa: E402
from qwen3_mini import Qwen3Model, preset as p3  # noqa: E402
from qwen3_next_mini import Qwen3NextModel, preset as pn  # noqa: E402
from qwen3_vl_mini import Qwen3VLMini, preset as pvl  # noqa: E402
from qwq_mini import QwQMini, preset as pq  # noqa: E402

FAMILIES = [
    ("Qwen2.5", Qwen25Model, p25),
    ("Qwen3", Qwen3Model, p3),
    ("Qwen3-Next", Qwen3NextModel, pn),
    ("Qwen3-VL", Qwen3VLMini, pvl),
    ("QwQ", QwQMini, pq),
]


def main() -> None:
    print(f"{'family':12s} {'params':>12s} {'active':>12s}")
    print("-" * 40)
    for name, cls, preset in FAMILIES:
        model = cls(preset("pc"))
        print(f"{name:12s} {model.num_parameters():12,d} {model.active_parameters():12,d}")


if __name__ == "__main__":
    main()
