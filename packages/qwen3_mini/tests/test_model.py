"""Qwen3 Mini smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen3_mini import Qwen3Model, Qwen3MoE, preset


def test_qk_norm_enabled() -> None:
    cfg = preset("demo")
    assert cfg.use_qk_norm is True
    model = Qwen3Model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_thinking_prompt() -> None:
    model = Qwen3Model(preset("demo"), thinking=True)
    prompt = model.thinking_prompt("What is 2+2?")
    assert prompt.startswith(" thinking")
    model.set_thinking(False)
    assert model.thinking_prompt("What is 2+2?") == "What is 2+2?"


def test_moe_variant() -> None:
    cfg = preset("moe")
    model = Qwen3MoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    assert model(x)[0].shape == (1, 6, cfg.vocab_size)


if __name__ == "__main__":
    test_qk_norm_enabled()
    test_thinking_prompt()
    test_moe_variant()
    print("ok")
