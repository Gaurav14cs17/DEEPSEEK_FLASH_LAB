"""QwQ Mini smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwq_mini import QwQMini, preset


def test_reasoning_format() -> None:
    prompt = QwQMini.format_reasoning("What is 2+2?")
    assert " thinking" in prompt and "</think>" in prompt
    reasoning, answer = QwQMini.parse_reasoning(" thinking\nbecause\n</think> four")
    assert reasoning == "because"
    assert answer == "four"


def test_forward() -> None:
    cfg = preset("demo")
    model = QwQMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


if __name__ == "__main__":
    test_reasoning_format()
    test_forward()
    print("ok")
