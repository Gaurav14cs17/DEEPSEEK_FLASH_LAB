# Qwen3 Mini — paper → code

## Family

Qwen3 keeps the Qwen2.5 decoder recipe and adds:

- QK-Norm on queries and keys (stabilises attention at scale)
- a thinking / non-thinking mode switch
- dense and MoE variants (the MoE family activates a small subset of experts)

## Code map

| Idea | File |
|------|------|
| config / presets | `config.py` |
| model + thinking helpers | `model.py` |
| shared primitives | `packages/qwen_common/` |
| tests | `tests/test_model.py` |
| notebook | `experiments/core/01_qwen3.ipynb` |

## Fidelity

- `[FAITHFUL]` QK-Norm, GQA, RoPE, RMSNorm, SwiGLU, MoE routing.
- `[SIMPLIFIED]` thinking mode is a prompt/decoding flag, not a separate
  post-trained checkpoint. No RLHF or reasoning data is included.

## What to inspect

1. `use_qk_norm=True` adds one RMSNorm per head before RoPE.
2. `Qwen3Model.thinking_prompt(...)` wraps a question with ` thinking`.
3. `Qwen3MoE` uses the same router as Qwen2.5-MoE but with QK-Norm enabled.

The real Qwen3 family spans dense and 30B-A3B-style sparse MoE models; this Mini
is a scale model, not a quality reproduction.
