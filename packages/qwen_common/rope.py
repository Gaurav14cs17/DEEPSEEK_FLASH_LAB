"""Rotary position embedding (rotate-half form used by Qwen)."""

from __future__ import annotations

import torch


def rope_freqs(
    d_head: int,
    seq_len: int,
    device: torch.device,
    theta: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each shaped [seq_len, d_head/2]."""
    if d_head % 2:
        raise ValueError("RoPE requires an even head dimension")
    half = d_head // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """Apply RoPE to [B, H, T, d_head] using cos/sin [T, d_head/2]."""
    T = x.shape[2]
    c = cos[position_offset : position_offset + T].unsqueeze(0).unsqueeze(0)
    s = sin[position_offset : position_offset + T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.stack((y1, y2), dim=-1).flatten(-2).type_as(x)
