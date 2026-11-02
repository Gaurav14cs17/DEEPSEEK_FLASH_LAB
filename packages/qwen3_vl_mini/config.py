"""Qwen3-VL Mini configuration: text stack + tiny vision tower."""

from __future__ import annotations

from dataclasses import dataclass

from qwen3_mini import Qwen3Config, preset as text_preset


@dataclass
class Qwen3VLConfig(Qwen3Config):
    vision_d_model: int = 64
    vision_n_heads: int = 4
    vision_n_layers: int = 2
    image_size: int = 32
    patch_size: int = 8


def preset(name: str = "pc", vocab_size: int = 64) -> Qwen3VLConfig:
    text = text_preset(name, vocab_size=vocab_size)
    return Qwen3VLConfig(
        **text.__dict__,
        vision_d_model=64,
        vision_n_heads=4,
        vision_n_layers=2,
        image_size=32,
        patch_size=8,
    )
