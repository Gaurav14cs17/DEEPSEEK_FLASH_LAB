"""Qwen3 Mini — QK-Norm, dense/MoE, thinking-mode flag.

The architecture is a standard decoder with QK-Norm. "Thinking" is a chat
template / decoding policy, not a different weight layout, so it is exposed as
a flag and a helper rather than a separate model.
"""

from __future__ import annotations

from typing import Optional

from qwen_common import QwenConfig, QwenModel

from .config import Qwen3Config, preset

THINK_OPEN = " thinking"
THINK_CLOSE = "</think>"


class Qwen3Model(QwenModel):
    def __init__(self, cfg: Optional[QwenConfig] = None, thinking: bool = False):
        cfg = cfg or preset("pc", thinking=thinking)
        cfg.thinking = thinking
        super().__init__(cfg)

    def set_thinking(self, enabled: bool) -> None:
        self.cfg.thinking = enabled

    def thinking_prompt(self, question: str) -> str:
        return f"{THINK_OPEN}\n{question}" if self.cfg.thinking else question


class Qwen3MoE(QwenModel):
    def __init__(self, cfg: Optional[QwenConfig] = None, thinking: bool = False):
        cfg = cfg or preset("moe", thinking=thinking)
        if not cfg.use_moe:
            cfg = cfg.copy(use_moe=True)
        cfg.thinking = thinking
        super().__init__(cfg)


Qwen3Dense = Qwen3Model
