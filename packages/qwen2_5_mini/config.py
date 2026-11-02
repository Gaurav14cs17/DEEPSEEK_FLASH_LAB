"""Qwen2.5 Mini configuration presets."""

from __future__ import annotations

from qwen_common import QwenConfig, tiny_common


class Qwen25Config(QwenConfig):
    """Qwen2.5 uses standard GQA + RoPE + SwiGLU + RMSNorm."""


def preset(name: str = "pc", vocab_size: int = 64) -> Qwen25Config:
    name = name.lower().strip()
    base = tiny_common(vocab_size)
    if name in ("demo", "tiny"):
        return Qwen25Config(**{**{k: v for k, v in base.__dict__.items() if k != "layer_pattern"}, "n_layers": 2, "max_seq_len": 96})
    if name == "pc":
        return Qwen25Config(**{**{k: v for k, v in base.__dict__.items() if k != "layer_pattern"}, "max_seq_len": 96, "dropout": 0.05})
    if name == "moe":
        return Qwen25Config(
            **{
                **{k: v for k, v in base.__dict__.items() if k != "layer_pattern"},
                "max_seq_len": 96,
                "dropout": 0.05,
                "use_moe": True,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "n_activated_experts": 2,
            }
        )
    if name in ("better", "better_pc", "large"):
        return Qwen25Config(
            **{
                **{k: v for k, v in base.__dict__.items() if k != "layer_pattern"},
                "d_model": 96,
                "n_heads": 4,
                "d_head": 24,
                "n_layers": 6,
                "max_seq_len": 128,
                "dropout": 0.08,
            }
        )
    raise ValueError(f"unknown preset {name!r}; use demo|pc|moe|better")
