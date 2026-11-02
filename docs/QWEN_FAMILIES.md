# Qwen Mini families

Five educational Qwen packages live under `packages/`, built on a shared
`qwen_common` core. They are **tiny architecture labs**, not reproductions of
Qwen's trained quality.

| Package | Import | What it teaches |
|---------|--------|-----------------|
| `qwen2_5_mini` | `from qwen2_5_mini import Qwen25Model, Qwen25MoE` | GQA, RoPE, SwiGLU, RMSNorm, dense + MoE |
| `qwen3_mini` | `from qwen3_mini import Qwen3Model, Qwen3MoE` | QK-Norm, thinking/non-thinking mode |
| `qwen3_next_mini` | `from qwen3_next_mini import Qwen3NextModel` | hybrid gated linear + full attention, sparse MoE |
| `qwen3_vl_mini` | `from qwen3_vl_mini import Qwen3VLMini` | ViT patch tower + projector + text decoder |
| `qwq_mini` | `from qwq_mini import QwQMini` | long-CoT reasoning format on a Qwen3-style decoder |

Shared primitives: `packages/qwen_common/` (RMSNorm, RoPE, GQA, SwiGLU, MoE,
`QwenModel`, char tokenizer).

---

## Quick start

```bash
pip install -e .          # optional, makes imports work everywhere

python -c "from qwen2_5_mini import Qwen25Model, preset; \
m = Qwen25Model(preset('pc')); print(m.num_parameters())"

python packages/qwen_common/benchmarks/parameter_count.py
python scripts/record_qwen_baselines.py
python comparison/compare_qwen.py
```

Notebooks (already executed):

```bash
jupyter notebook packages/qwen2_5_mini/experiments/core
jupyter notebook packages/qwen3_mini/experiments/core
jupyter notebook packages/qwen3_next_mini/experiments/core
jupyter notebook packages/qwen3_vl_mini/experiments/core
jupyter notebook packages/qwq_mini/experiments/core
```

---

## Family notes

### Qwen2.5

Baseline decoder: RMSNorm → GQA → residual, then SwiGLU. `Qwen25MoE` adds
shared + top-k routed experts. `n_kv_heads < n_heads` is the GQA saving.

### Qwen3

Same decoder with QK-Norm (`use_qk_norm=True`). The thinking mode is a
prompt/decoding policy, exposed through `thinking_prompt()`; it does not change
the parameter layout.

### Qwen3-Next

Most layers use a gated linear-attention recurrence:

```
S_t = decay_t * S_{t-1} + k_t v_t^T
y_t = q_t S_t
```

Every fourth layer (and the last layer) uses full attention. This is
`[SIMPLIFIED]`: it captures the hybrid idea and O(T) state cost, not the exact
Gated DeltaNet kernel.

### Qwen3-VL

`TinyViT` patchifies an image, runs a small transformer, projects the patches
into the language embedding space, and prepends them to text tokens. Real
Qwen3-VL uses image placeholders inside a chat template; this Mini concatenates
tokens for clarity.

### QwQ

A Qwen3-style decoder with `thinking=True`, longer context, and helpers:

```python
from qwq_mini import QwQMini
prompt = QwQMini.format_reasoning("What is 2 + 2?")
reasoning, answer = QwQMini.parse_reasoning(completion)
```

The honest lesson: reasoning models are usually **post-training + decoding**,
not a new attention block.

---

## Fidelity labels

| Tag | Meaning |
|-----|---------|
| `[FAITHFUL]` | documented mechanism, miniaturised |
| `[SIMPLIFIED]` | scaled or approximated for CPU |
| `[SIMULATED]` | systems behaviour without production infrastructure |
| `[EXPERIMENTAL]` | local probe, not a production claim |

Every family's `paper_notes/architecture.md` states which parts are which.

---

## Measured baselines

`scripts/record_qwen_baselines.py` writes the same three-file bundle as the
DeepSeek experiments:

```
results/qwen/<family>/baseline/{config,metrics,memory}.json
results/qwen/summary.json
```

Protocol: `pc` preset, batch 2, seq 16, fp32, CPU, seed 42. Read
`results/qwen/summary.json` for parameters, active parameters, latency, and
throughput.

---

## Qwen vs DeepSeek

| Aspect | DeepSeek Minis | Qwen Minis |
|--------|----------------|------------|
| Attention | SWA / CSA / HCA / MLA | GQA; hybrid linear in Qwen3-Next |
| MoE | shared + routed, hash/learned | shared + routed, softmax top-k |
| Position | complex RoPE | rotate-half RoPE |
| Extras | mHC, MTP, CED | QK-Norm, ViT, thinking mode |
| Structure | `packages/deepseek_flash_*` | `packages/qwen_*` |

Use `comparison/compare_qwen.py` for a side-by-side parameter/latency table.
