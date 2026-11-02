"""Qwen3-VL Mini smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen3_vl_mini import Qwen3VLMini, preset


def test_text_only() -> None:
    cfg = preset("demo")
    model = Qwen3VLMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    assert model(x)[0].shape == (2, 6, cfg.vocab_size)


def test_image_tokens_prepended() -> None:
    cfg = preset("demo")
    model = Qwen3VLMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    images = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    log: list[str] = []
    logits, _ = model(x, images=images, log=log)
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    assert logits.shape == (2, 6 + n_patches, cfg.vocab_size)
    assert any("image tokens" in line for line in log)


if __name__ == "__main__":
    test_text_only()
    test_image_tokens_prepended()
    print("ok")
