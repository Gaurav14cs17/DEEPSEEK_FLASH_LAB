"""Qwen3 Mini — importable educational model."""

from __future__ import annotations

from .config import Qwen3Config, preset
from .model import THINK_CLOSE, THINK_OPEN, Qwen3Dense, Qwen3Model, Qwen3MoE

__version__ = "0.1.0"

__all__ = [
    "Qwen3Config",
    "preset",
    "Qwen3Model",
    "Qwen3Dense",
    "Qwen3MoE",
    "THINK_OPEN",
    "THINK_CLOSE",
    "__version__",
]
