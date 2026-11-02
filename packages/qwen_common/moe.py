"""Sparse Mixture of Experts used by Qwen MoE variants.

Shared experts always run. A softmax router selects top-k routed experts per
token. This mirrors the Qwen2.5-MoE / Qwen3-MoE idea at tiny scale.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QwenConfig
from .mlp import SwiGLU


class Router(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.n_activated = cfg.n_activated_experts
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.softmax(self.gate(x), dim=-1)
        topv, topi = scores.topk(self.n_activated, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topv, topi


class MoE(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_shared_experts)]
        )
        self.routed = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_routed_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        log: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = torch.zeros_like(flat)
        for expert in self.shared:
            out = out + expert(flat)

        topv, topi = self.router(flat)
        if log is not None:
            hist = torch.bincount(topi.reshape(-1), minlength=self.cfg.n_routed_experts)
            log.append(
                f"MoE routed={self.cfg.n_routed_experts} "
                f"active={self.cfg.n_activated_experts} hist={hist.tolist()}"
            )

        for slot in range(self.cfg.n_activated_experts):
            idx = topi[:, slot]
            weight = topv[:, slot].unsqueeze(-1)
            for expert_id in idx.unique().tolist():
                mask = idx == int(expert_id)
                out[mask] = out[mask] + weight[mask] * self.routed[int(expert_id)](flat[mask])
        return out.view(B, T, D)
