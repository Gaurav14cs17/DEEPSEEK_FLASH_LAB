"""Shared configuration for the Qwen Mini family.

Every Qwen Mini package uses this dataclass and only changes defaults or adds
family-specific flags. Keeping one schema makes cross-family comparison honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class QwenConfig:
    # Core transformer
    vocab_size: int = 64
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_head: int = 16
    max_seq_len: int = 128
    mlp_hidden: int = 0  # 0 -> 4 * d_model

    # Normalization / position
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_qk_norm: bool = False
    attention_bias: bool = False
    sliding_window: Optional[int] = None
    dropout: float = 0.0

    # MLP / MoE
    use_moe: bool = False
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    expert_hidden: int = 0  # 0 -> 2 * d_model
    norm_topk_prob: bool = True

    # Embedding
    tie_embeddings: bool = True

    # Family flags (architecture-neutral)
    thinking: bool = False
    layer_pattern: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not self.mlp_hidden:
            self.mlp_hidden = 4 * self.d_model
        if not self.expert_hidden:
            self.expert_hidden = 2 * self.d_model
        if self.use_moe and self.n_activated_experts > self.n_routed_experts:
            raise ValueError("n_activated_experts cannot exceed n_routed_experts")
        if not self.layer_pattern:
            self.layer_pattern = ["full"] * self.n_layers
        if len(self.layer_pattern) != self.n_layers:
            raise ValueError("layer_pattern length must equal n_layers")

    def copy(self, **changes) -> "QwenConfig":
        return replace(self, **changes)


def tiny_common(vocab_size: int = 64) -> QwenConfig:
    """A small but non-trivial default used by every family preset."""
    return QwenConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        max_seq_len=128,
    )
