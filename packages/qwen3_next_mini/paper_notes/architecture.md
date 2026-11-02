# Qwen3-Next Mini — paper → code

## Family

Qwen3-Next is a hybrid architecture:

- most layers use a **gated linear attention** / DeltaNet-style recurrence
- periodic layers use **full attention** for global mixing
- an **ultra-sparse MoE** supplies capacity with low per-token compute

## Code map

| Idea | File |
|------|------|
| config + hybrid layer pattern | `config.py` |
| gated linear attention + hybrid block | `model.py` |
| shared attention / MoE primitives | `packages/qwen_common/` |
| tests | `tests/test_model.py` |
| notebook | `experiments/core/01_qwen3_next.ipynb` |

## Fidelity

- `[FAITHFUL]` hybrid layer concept, causal recurrence, gated memory, sparse MoE.
- `[SIMPLIFIED]` the linear block is a gated linear attention recurrence, **not**
  the exact Gated DeltaNet kernel. There is no chunked parallel kernel.
- `[SIMULATED]` long-context efficiency is demonstrated by complexity, not by a
  production-scale throughput claim.

## The recurrence

For each head:

```
S_t = decay_t * S_{t-1} + k_t v_t^T
y_t = q_t S_t
```

`decay_t` is a sigmoid gate from the input. State size is `d_head × d_head`, so
cost is `O(T)` instead of the `O(T²)` score matrix of full attention.

## What to inspect

1. `cfg.layer_pattern` — `linear` vs `full` per layer.
2. `GatedLinearAttention` — the explicit `for t in range(T)` loop.
3. `HybridBlock` — same residual/normalization shape as the dense block.
