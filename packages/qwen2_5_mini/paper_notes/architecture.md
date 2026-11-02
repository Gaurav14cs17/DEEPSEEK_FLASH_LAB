# Qwen2.5 Mini — paper → code

## Family

Qwen2.5 is a decoder-only transformer family (dense and MoE variants) with:

- RMSNorm (pre-norm)
- Rotary position embedding (RoPE)
- Grouped-query attention (GQA)
- SwiGLU feed-forward
- Tied input/output embeddings in the small variants

## Code map

| Idea | File |
|------|------|
| config / presets | `config.py` |
| dense and MoE models | `model.py` |
| shared primitives | `packages/qwen_common/` |
| tests | `tests/test_model.py` |
| notebook | `experiments/core/01_qwen2_5.ipynb` |

## Fidelity

| Tag | Meaning here |
|-----|--------------|
| `[FAITHFUL]` | GQA, RoPE, RMSNorm, SwiGLU, pre-norm residual layout |
| `[SIMPLIFIED]` | character tokenizer, tiny dimensions, top-k MoE loop |
| `[SIMULATED]` | no production kernel, no distributed serving |

## What to inspect

1. `n_kv_heads < n_heads` — K/V tensors are smaller than Q; `repeat_interleave`
   expands them for the score computation.
2. `use_moe=True` — shared experts always run, routed experts are top-k.
3. `tie_embeddings=True` — the LM head shares the embedding matrix.

The real Qwen2.5 models are 0.5B–72B+; this Mini is ~0.1–0.6M parameters.
