"""Qwen3 Mini configuration presets (QK-Norm enabled)."""

from __future__ import annotations

from qwen_common import QwenConfig, tiny_common


class Qwen3Config(QwenConfig):
    """Qwen3 adds QK-Norm and a thinking/non-thinking mode flag."""


def _base(vocab_size: int, **changes) -> Qwen3Config:
    base = tiny_common(vocab_size)
    values = {**{k: v for k, v in base.__dict__.items() if k != "layer_pattern"}, "use_qk_norm": True, **changes}
    return Qwen3Config(**values)


def preset(name: str = "pc", vocab_size: int = 64, thinking: bool = False) -> Qwen3Config:
    name = name.lower().strip()
    if name in ("demo", "tiny"):
        return _base(vocab_size, n_layers=2, max_seq_len=96, thinking=thinking)
    if name == "pc":
        return _base(vocab_size, max_seq_len=96, dropout=0.05, thinking=thinking)
    if name == "moe":
        return _base(
            vocab_size,
            max_seq_len=96,
            dropout=0.05,
            thinking=thinking,
            use_moe=True,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
        )
    if name in ("better", "better_pc", "large"):
        return _base(
            vocab_size,
            d_model=96,
            n_heads=4,
            d_head=24,
            n_layers=6,
            max_seq_len=128,
            dropout=0.08,
            thinking=thinking,
        )
    raise ValueError(f"unknown preset {name!r}; use demo|pc|moe|better")
