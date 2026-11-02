"""Qwen2.5 Mini — importable educational model."""

from __future__ import annotations

from .config import Qwen25Config, preset
from .model import Qwen25MoE, Qwen25Model, Qwen2_5Mini, Qwen2_5MoE

__version__ = "0.1.0"

__all__ = [
    "Qwen25Config",
    "preset",
    "Qwen25Model",
    "Qwen25MoE",
    "Qwen2_5Mini",
    "Qwen2_5MoE",
    "__version__",
]
