"""Qwen3-Next Mini configuration (hybrid linear/full attention + sparse MoE)."""

from __future__ import annotations

from qwen_common import QwenConfig, tiny_common


class Qwen3NextConfig(QwenConfig):
    """Qwen3-Next mixes gated linear attention with periodic full attention."""


def _hybrid_pattern(n_layers: int) -> list[str]:
    """3 linear-attention layers followed by 1 full-attention layer.

    The last layer is always full attention so every config has a global path.
    """
    return [
        "full" if ((i + 1) % 4 == 0 or i == n_layers - 1) else "linear"
        for i in range(n_layers)
    ]


def preset(name: str = "pc", vocab_size: int = 64) -> Qwen3NextConfig:
    name = name.lower().strip()
    base = tiny_common(vocab_size)
    values = {
        **base.__dict__,
        "use_qk_norm": True,
        "use_moe": True,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
    }
    if name in ("demo", "tiny"):
        n = 2
        return Qwen3NextConfig(**{**values, "n_layers": n, "layer_pattern": _hybrid_pattern(n), "use_moe": False})
    if name == "pc":
        n = 4
        return Qwen3NextConfig(
            **{**values, "n_layers": n, "layer_pattern": _hybrid_pattern(n), "max_seq_len": 128}
        )
    if name == "moe":
        n = 8
        return Qwen3NextConfig(
            **{**values, "n_layers": n, "layer_pattern": _hybrid_pattern(n), "max_seq_len": 128}
        )
    if name in ("better", "better_pc", "large"):
        n = 8
        return Qwen3NextConfig(
            **{
                **values,
                "d_model": 96,
                "n_heads": 4,
                "d_head": 24,
                "n_layers": n,
                "layer_pattern": _hybrid_pattern(n),
                "max_seq_len": 192,
            }
        )
    raise ValueError(f"unknown preset {name!r}; use demo|pc|moe|better")
