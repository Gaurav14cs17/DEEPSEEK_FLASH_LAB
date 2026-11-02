"""Qwen3-Next Mini smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen3_next_mini import Qwen3NextModel, preset


def test_hybrid_pattern() -> None:
    cfg = preset("pc")
    assert "linear" in cfg.layer_pattern
    assert cfg.layer_pattern[-1] == "full"


def test_forward_and_causality() -> None:
    cfg = preset("demo")
    model = Qwen3NextModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y, _ = model(x)
    assert y.shape == (1, 8, cfg.vocab_size)

    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
    y2, _ = model(x2)
    assert torch.allclose(y[0, :-1], y2[0, :-1], atol=1e-5)


if __name__ == "__main__":
    test_hybrid_pattern()
    test_forward_and_causality()
    print("ok")
