"""Qwen3-Next Mini — hybrid gated linear attention + full attention.

The linear block is a [SIMPLIFIED] gated linear attention recurrence, not the
exact Gated DeltaNet kernel used by the real model. It preserves the important
educational properties: causal, O(T) state update, and a gated memory.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from qwen_common import GQAttention, MoE, QwenModel, RMSNorm, SwiGLU

from .config import Qwen3NextConfig, preset


class GatedLinearAttention(nn.Module):
    """Causal gated linear attention.

    State update per head:

        S_t = decay_t * S_{t-1} + k_t v_t^T
        y_t = q_t S_t

    with decay_t in (0, 1) produced from the input. Cost is O(T) instead of the
    O(T^2) score matrix of full attention.
    """

    def __init__(self, cfg: Qwen3NextConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, *, log: Optional[List[str]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        decay = torch.sigmoid(self.gate_proj(x)).transpose(1, 2)  # [B,H,T]

        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            d = decay[:, :, t].unsqueeze(-1).unsqueeze(-1)
            state = d * state + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
            y = torch.einsum("bhd,bhde->bhe", q[:, :, t], state)
            outputs.append(y)
        y = torch.stack(outputs, dim=2)  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        if log is not None:
            log.append(f"GatedLinearAttention heads={H} O(T) state={D}x{D}")
        return self.o_proj(y)


class HybridBlock(nn.Module):
    def __init__(self, cfg: Qwen3NextConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAttention(cfg) if kind == "full" else GatedLinearAttention(cfg)
        self.ffn = MoE(cfg) if cfg.use_moe else SwiGLU(cfg.d_model, cfg.mlp_hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, *, log: Optional[List[str]] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), log=log))
        h = self.norm2(x)
        ffn_out = self.ffn(h, log=log) if isinstance(self.ffn, MoE) else self.ffn(h)
        return x + self.drop(ffn_out)


class Qwen3NextModel(QwenModel):
    def __init__(self, cfg: Optional[Qwen3NextConfig] = None):
        cfg = cfg or preset("pc")
        super().__init__(cfg)
        self.layers = nn.ModuleList(
            [HybridBlock(cfg, kind) for kind in cfg.layer_pattern]
        )
        self.apply(self._init_weights)


Qwen3NextMoE = Qwen3NextModel
