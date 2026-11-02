"""Qwen2.5 Mini smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen2_5_mini import Qwen25Model, Qwen25MoE, preset


def test_dense_forward() -> None:
    cfg = preset("demo")
    model = Qwen25Model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert model.num_parameters() > 0


def test_moe_forward() -> None:
    cfg = preset("moe")
    model = Qwen25MoE(cfg)
    log: list[str] = []
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x, log=log)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert any("MoE" in line for line in log)


if __name__ == "__main__":
    test_dense_forward()
    test_moe_forward()
    print("ok")
