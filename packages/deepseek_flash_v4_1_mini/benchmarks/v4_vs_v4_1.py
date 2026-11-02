#!/usr/bin/env python3
"""Compare V4 Mini vs V4.1 Mini parameter counts and layout."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES))

from deepseek_flash_v4_1_mini import DeepSeekFlashV41Mini, preset as preset41
from deepseek_flash_v4_mini import DeepSeekFlashV4Mini, preset as preset4


def main() -> None:
    m4 = DeepSeekFlashV4Mini(preset4("pc"))
    cfg = preset41("pc")
    cfg.use_vision = False
    m41 = DeepSeekFlashV41Mini(cfg)
    print(f"V4-Mini     params={m4.num_parameters():,} layers={m4.cfg.n_layers}")
    print(
        f"V4.1-Mini   params={m41.num_parameters():,} "
        f"enc={m41.cfg.n_encoder_layers} dec={m41.cfg.n_decoder_layers}"
    )


if __name__ == "__main__":
    main()
