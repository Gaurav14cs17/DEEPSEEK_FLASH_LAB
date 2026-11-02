"""Qwen3-VL Mini — text decoder + tiny vision tower + projector."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from qwen3_mini import Qwen3Model

from .config import Qwen3VLConfig, preset
from .vision import TinyViT


class Qwen3VLMini(Qwen3Model):
    def __init__(self, cfg: Optional[Qwen3VLConfig] = None):
        cfg = cfg or preset("pc")
        super().__init__(cfg)
        self.vision = TinyViT(cfg)
        self.projector = nn.Linear(cfg.vision_d_model, cfg.d_model)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        *,
        images: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        log: Optional[List[str]] = None,
        trace: bool = True,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        if inputs_embeds is None:
            if tokens is None:
                raise ValueError("pass tokens or inputs_embeds")
            notes = log if log is not None else ([] if trace else None)
            inputs_embeds = self.embed(tokens)
            if images is not None:
                image_tokens = self.projector(self.vision(images))
                inputs_embeds = torch.cat([image_tokens, inputs_embeds], dim=1)
                if notes is not None:
                    notes.append(
                        f"VL: {image_tokens.shape[1]} image tokens + "
                        f"{tokens.shape[1]} text tokens"
                    )
        else:
            notes = log
        return super().forward(
            inputs_embeds=inputs_embeds,
            log=notes,
            trace=trace,
            return_hidden=return_hidden,
        )


Qwen3VLMoE = Qwen3VLMini
