"""SwiGLU feed-forward, the MLP used by Qwen models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
