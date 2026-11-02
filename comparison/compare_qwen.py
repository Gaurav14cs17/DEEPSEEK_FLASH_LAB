#!/usr/bin/env python3
"""Compare Qwen Mini families (and V4 for reference) on parameters + latency."""

from __future__ import annotations

import sys

import torch

from _load import REPO

sys.path.insert(0, str(REPO))
from bench.latency import time_ms  # noqa: E402

from qwen2_5_mini import Qwen25Model, preset as p25  # noqa: E402
from qwen3_mini import Qwen3Model, preset as p3  # noqa: E402
from qwen3_next_mini import Qwen3NextModel, preset as pn  # noqa: E402
from qwen3_vl_mini import Qwen3VLMini, preset as pvl  # noqa: E402
from qwq_mini import QwQMini, preset as pq  # noqa: E402
from deepseek_flash_v4_mini import DeepSeekFlashV4Mini, preset as p4  # noqa: E402

FAMILIES = [
    ("V4-Mini", DeepSeekFlashV4Mini, p4),
    ("Qwen2.5", Qwen25Model, p25),
    ("Qwen3", Qwen3Model, p3),
    ("Qwen3-Next", Qwen3NextModel, pn),
    ("Qwen3-VL", Qwen3VLMini, pvl),
    ("QwQ", QwQMini, pq),
]


def main() -> None:
    print(f"{'family':12s} {'params':>12s} {'forward_ms':>12s}")
    print("-" * 40)
    for name, cls, preset in FAMILIES:
        cfg = preset("pc")
        model = cls(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            timing = time_ms(lambda: model(x, trace=False), warmup=2, iters=5)
        print(f"{name:12s} {model.num_parameters():12,d} {timing['mean_ms']:12.3f}")


if __name__ == "__main__":
    main()
