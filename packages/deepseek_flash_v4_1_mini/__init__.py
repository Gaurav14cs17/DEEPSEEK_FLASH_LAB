"""DeepSeekFlashV4.1-Mini — educational DeepSeek-V4.1-Flash toy.

Import the public API directly:

    from deepseek_flash_v4_1_mini import DeepSeekFlashV41Mini, preset
"""

from __future__ import annotations

from .model import (
    DeepSeekFlashV41Mini,
    FlashV41Mini,
    ToyConfig,
    ToyDSV41,
    V41Config,
    preset,
)

__version__ = "0.1.0"

__all__ = [
    "DeepSeekFlashV41Mini",
    "FlashV41Mini",
    "ToyDSV41",
    "ToyConfig",
    "V41Config",
    "preset",
    "__version__",
]
