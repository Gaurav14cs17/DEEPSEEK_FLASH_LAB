"""Shared primitives for the Qwen Mini family."""

from __future__ import annotations

from .attention import GQAttention
from .config import QwenConfig, tiny_common
from .mlp import SwiGLU
from .moe import MoE, Router
from .model import QwenBlock, QwenModel, loss_next_token
from .normalization import RMSNorm
from .rope import apply_rope, rope_freqs
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

__all__ = [
    "QwenConfig",
    "tiny_common",
    "RMSNorm",
    "rope_freqs",
    "apply_rope",
    "GQAttention",
    "SwiGLU",
    "Router",
    "MoE",
    "QwenBlock",
    "QwenModel",
    "loss_next_token",
    "CharTokenizer",
]
