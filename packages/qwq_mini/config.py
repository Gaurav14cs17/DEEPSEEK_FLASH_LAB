"""QwQ Mini configuration: long-context reasoning variant of Qwen3."""

from __future__ import annotations

from qwen3_mini import Qwen3Config, preset as qwen3_preset


class QwQConfig(Qwen3Config):
    """QwQ is a reasoning-tuned model; the architecture stays Qwen3-like."""


def preset(name: str = "pc", vocab_size: int = 64) -> QwQConfig:
    base = qwen3_preset(name, vocab_size=vocab_size, thinking=True)
    return QwQConfig(
        **{
            **base.__dict__,
            "max_seq_len": max(base.max_seq_len, 256),
            "thinking": True,
        }
    )
