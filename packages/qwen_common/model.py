"""Shared Qwen Mini transformer.

Dense and MoE variants share this stack. Family packages add flags (QK-Norm,
thinking mode, hybrid attention) without duplicating the basic math.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GQAttention
from .config import QwenConfig
from .mlp import SwiGLU
from .moe import MoE
from .normalization import RMSNorm


class QwenBlock(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = (
            MoE(cfg) if cfg.use_moe else SwiGLU(cfg.d_model, cfg.mlp_hidden)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        log: Optional[List[str]] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), log=log))
        h = self.norm2(x)
        ffn_out = self.ffn(h, log=log) if isinstance(self.ffn, MoE) else self.ffn(h)
        return x + self.drop(ffn_out)


class QwenModel(nn.Module):
    """Base dense/MoE Qwen Mini.

    Family packages subclass or alias this. `inputs_embeds` is used by the
    vision-language Mini to prepend image tokens.
    """

    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([QwenBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameters(self) -> int:
        """Parameters that actually run for one token.

        Dense: all parameters. MoE: shared experts plus the top-k fraction of
        routed experts. This is an estimate, not a kernel-level FLOP count.
        """
        total = self.num_parameters()
        if not self.cfg.use_moe:
            return total
        routed = 0
        for block in self.layers:
            ffn = getattr(block, "ffn", None)
            if isinstance(ffn, MoE):
                routed += sum(p.numel() for e in ffn.routed for p in e.parameters())
        if not routed:
            return total
        fraction = self.cfg.n_activated_experts / self.cfg.n_routed_experts
        return int(total - routed + routed * fraction)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        log: Optional[List[str]] = None,
        trace: bool = True,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of tokens or inputs_embeds")
        notes: Optional[List[str]] = log if log is not None else ([] if trace else None)
        x = inputs_embeds if inputs_embeds is not None else self.embed(tokens)
        if notes is not None:
            notes.append(
                f"QwenMini T={x.shape[1]} layers={self.cfg.n_layers} "
                f"moe={self.cfg.use_moe} qk_norm={self.cfg.use_qk_norm}"
            )
        for i, layer in enumerate(self.layers):
            if notes is not None:
                notes.append(f"— layer {i} —")
            x = layer(x, log=notes)
        h = self.norm(x)
        logits = self.lm_head(h)
        info = {"log": notes or [], "hidden": h if return_hidden else None}
        return logits, info

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        tokens = prompt
        for _ in range(max_new):
            ctx = tokens[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(ctx, trace=False)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens


def loss_next_token(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
