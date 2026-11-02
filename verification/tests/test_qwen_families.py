"""Cross-family smoke test for the Qwen Mini packages."""

from __future__ import annotations

import torch

from qwen2_5_mini import Qwen25Model, preset as p25
from qwen3_mini import Qwen3Model, preset as p3
from qwen3_next_mini import Qwen3NextModel, preset as pn
from qwen3_vl_mini import Qwen3VLMini, preset as pvl
from qwq_mini import QwQMini, preset as pq

CASES = [
    (Qwen25Model, p25),
    (Qwen3Model, p3),
    (Qwen3NextModel, pn),
    (QwQMini, pq),
]


def test_qwen_text_families_forward() -> None:
    for cls, preset in CASES:
        cfg = preset("demo")
        model = cls(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 6))
        logits, _ = model(tokens)
        assert logits.shape == (2, 6, cfg.vocab_size), cls.__name__
        assert model.num_parameters() > 0


def test_qwen3_vl_forward_with_image() -> None:
    cfg = pvl("demo")
    model = Qwen3VLMini(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 5))
    images = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    logits, _ = model(tokens, images=images)
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    assert logits.shape == (1, 5 + n_patches, cfg.vocab_size)


def test_qwen3_next_is_causal() -> None:
    cfg = pn("demo")
    model = Qwen3NextModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 7))
    a, _ = model(tokens)
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
    b, _ = model(changed)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_qwq_reasoning_markers() -> None:
    prompt = QwQMini.format_reasoning("question")
    assert " thinking" in prompt and "</think>" in prompt
    reasoning, answer = QwQMini.parse_reasoning(" thinking\nsteps\n</think> answer")
    assert reasoning == "steps"
    assert answer == "answer"
