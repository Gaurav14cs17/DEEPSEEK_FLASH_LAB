"""Grouped-query causal attention used by Qwen2.5 / Qwen3 / QwQ.

Qwen-specific options:
  - GQA (n_kv_heads < n_heads)
  - optional QK-Norm (Qwen3)
  - optional sliding-window mask
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QwenConfig
from .normalization import RMSNorm
from .rope import apply_rope, rope_freqs


class GQAttention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=bias)
        self.q_norm = RMSNorm(cfg.d_head, cfg.rms_norm_eps) if cfg.use_qk_norm else None
        self.k_norm = RMSNorm(cfg.d_head, cfg.rms_norm_eps) if cfg.use_qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        log: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = rope_freqs(
            self.d_head,
            position_offset + T,
            x.device,
            theta=self.cfg.rope_theta,
        )
        q = apply_rope(q, cos, sin, position_offset)
        k = apply_rope(k, cos, sin, position_offset)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        idx = torch.arange(T, device=x.device)
        allow = idx[None, :] <= idx[:, None]
        if self.cfg.sliding_window is not None:
            allow = allow & (idx[:, None] - idx[None, :] < self.cfg.sliding_window)
        scores = scores.masked_fill(~allow[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if log is not None:
            log.append(
                f"GQA heads={self.n_heads} kv_heads={self.n_kv_heads} "
                f"qk_norm={self.cfg.use_qk_norm}"
            )
        return self.o_proj(out)
