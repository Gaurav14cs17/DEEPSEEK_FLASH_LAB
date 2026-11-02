"""DeepSeekFlashV4-Mini — educational DeepSeek-V4-Flash toy.

Import the public API directly:

    from deepseek_flash_v4_mini import DeepSeekFlashV4Mini, preset
"""

from __future__ import annotations

from .model import (
    DeepSeekFlashV4Mini,
    FlashV4Mini,
    ToyV4Flash,
    V4Config,
    preset,
)

__version__ = "0.1.0"

__all__ = [
    "DeepSeekFlashV4Mini",
    "FlashV4Mini",
    "ToyV4Flash",
    "V4Config",
    "preset",
    "__version__",
]
