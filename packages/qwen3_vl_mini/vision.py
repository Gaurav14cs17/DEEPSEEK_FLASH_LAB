"""Tiny ViT tower for the Qwen3-VL Mini.

[FAITHFUL] idea: patchify -> positional embedding -> transformer -> project into
the language model's embedding space.
[SIMPLIFIED] scale and fusion: image tokens are prepended, not interleaved with
special image markers as in the full Qwen3-VL chat template.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3VLConfig


class TinyViT(nn.Module):
    def __init__(self, cfg: Qwen3VLConfig):
        super().__init__()
        if cfg.image_size % cfg.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.cfg = cfg
        self.patch = nn.Conv2d(
            3,
            cfg.vision_d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.pos = nn.Parameter(torch.randn(1, n_patches, cfg.vision_d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_d_model,
            nhead=cfg.vision_n_heads,
            dim_feedforward=4 * cfg.vision_d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.vision_n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.vision_d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, 3, H, W] -> patches [B, n_patches, vision_d]."""
        x = self.patch(images).flatten(2).transpose(1, 2)
        x = x + self.pos[:, : x.shape[1]]
        return self.norm(self.encoder(x))
