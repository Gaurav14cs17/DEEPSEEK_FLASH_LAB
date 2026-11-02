"""Qwen2.5 Mini — dense and MoE educational variants."""

from __future__ import annotations

from typing import Optional

from qwen_common import QwenConfig, QwenModel

from .config import Qwen25Config, preset


class Qwen25Model(QwenModel):
    """Dense Qwen2.5-style model (GQA, RoPE, SwiGLU, RMSNorm)."""

    def __init__(self, cfg: Optional[QwenConfig] = None):
        super().__init__(cfg or preset("pc"))


class Qwen25MoE(QwenModel):
    """Qwen2.5-MoE-style model: shared + top-k routed experts."""

    def __init__(self, cfg: Optional[QwenConfig] = None):
        cfg = cfg or preset("moe")
        if not cfg.use_moe:
            cfg = cfg.copy(use_moe=True)
        super().__init__(cfg)


# Friendly aliases
Qwen2_5Mini = Qwen25Model
Qwen2_5MoE = Qwen25MoE
