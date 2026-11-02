# Qwen shared primitives

`qwen_common` is the shared core for every Qwen Mini package. It exists so the
five families do not drift into five incompatible versions of RMSNorm or RoPE.

| File | Responsibility |
|------|----------------|
| `config.py` | `QwenConfig` schema, validation, tiny defaults |
| `normalization.py` | RMSNorm |
| `rope.py` | RoPE frequencies + rotate-half application |
| `attention.py` | GQA causal attention, optional QK-Norm, sliding window |
| `mlp.py` | SwiGLU |
| `moe.py` | softmax router + shared/routed experts |
| `model.py` | `QwenBlock`, `QwenModel`, generation, next-token loss |
| `tokenizer.py` | character tokenizer |

## Family deltas

| Family | Delta on top of common |
|--------|------------------------|
| Qwen2.5 | defaults only |
| Qwen3 | `use_qk_norm=True`, thinking flag |
| Qwen3-Next | hybrid linear/full attention blocks, sparse MoE |
| Qwen3-VL | tiny ViT + projector + prepended image tokens |
| QwQ | Qwen3 + long context + reasoning format helpers |

## Rule

If a mechanism is shared by two or more families, it belongs here. If it is
specific to one family, it belongs in that package.
