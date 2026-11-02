"""Tests for shared Qwen primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen_common import (
    GQAttention,
    MoE,
    QwenConfig,
    QwenModel,
    RMSNorm,
    apply_rope,
    rope_freqs,
)


def test_rmsnorm_scale() -> None:
    x = torch.randn(3, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(dim=-1).sqrt(), torch.ones(3), atol=1e-4)


def test_rope_preserves_norm() -> None:
    x = torch.randn(1, 2, 6, 8)
    cos, sin = rope_freqs(8, 6, x.device)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_gqa_causality() -> None:
    cfg = QwenConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_head=8)
    attn = GQAttention(cfg).eval()
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, -1] += 5.0
    y, y2 = attn(x), attn(x2)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)
    assert not torch.allclose(y[:, -1], y2[:, -1])


def test_moe_shape_and_log() -> None:
    cfg = QwenConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_head=8,
        use_moe=True,
        n_routed_experts=4,
        n_activated_experts=2,
    )
    model = QwenModel(cfg)
    log: list[str] = []
    out, _ = model(torch.randint(0, 32, (2, 7)), log=log)
    assert out.shape == (2, 7, 32)
    assert any("MoE" in line for line in log)


if __name__ == "__main__":
    test_rmsnorm_scale()
    test_rope_preserves_norm()
    test_gqa_causality()
    test_moe_shape_and_log()
    print("ok")
