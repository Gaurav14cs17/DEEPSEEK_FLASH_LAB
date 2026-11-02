# Learning path (start here)

A guided, beginner-friendly route through `deepseek_flash_lab`.

You do **not** need a GPU. You do **not** need prior transformer experience.
You need basic Python and curiosity.

> **Big idea:** every session follows the same loop:
> **idea → math/shape → code → run → observe → measure**.

---

## Before you start

```bash
cd deepseek_flash_lab
pip install -r requirements.txt
python learn/tour.py          # 10-minute guided tour
```

If the tour runs, your environment is ready. If not, fix that first — everything
else assumes the tour works.

| Resource | What it is |
|----------|------------|
| `learn/tour.py` | Interactive CLI tour of every core component |
| `learn/notebooks/` | Hands-on notebooks with live outputs |
| `docs/EXERCISES.md` | Practice tasks for each session |
| `docs/SOLUTIONS.md` | Worked answers (open after trying) |
| `docs/CODE_WALKTHROUGH.md` | File-by-file map of the source |
| `docs/optimization_protocol.md` | The formal measurement rules |

---

## The 7-session path

### Session 0 — Orientation (20 min)

**Goal:** know what the lab is and is not.

Read: the top of [`README.md`](../README.md), especially the table
"This lab is / is not".

Run:
```bash
python learn/tour.py -s 0
```

Checkpoint: explain in one sentence why a 1M-parameter model is useful here
even though the real model has hundreds of billions of parameters.

---

### Session 1 — Tokens and embeddings (45 min)

**Goal:** turn text into integers, then integers into vectors.

Read: `packages/deepseek_flash_v4_mini/model/embedding.py`

Run:
```bash
python learn/tour.py -s 1
python learn/tour.py -s 2
jupyter notebook learn/notebooks/01_tokens_embeddings.ipynb
```

What to notice:
- One character = one token in this lab.
- Embedding is a **row lookup**, not a matrix multiply.
- Shape goes `[B, T]` → `[B, T, d_model]`.
- Parameter count is `vocab_size × d_model`.

Checkpoint: if `vocab_size=11` and `d_model=8`, how many embedding parameters?
(Answer: 88.)

Exercises: 1 and 2 in [`EXERCISES.md`](EXERCISES.md).

---

### Session 2 — Normalization and position (60 min)

**Goal:** understand RMSNorm and RoPE, the two operations applied before
attention.

Read: `model/normalization.py`, `model/rope.py`

Run:
```bash
python learn/tour.py -s 3
python learn/tour.py -s 4
jupyter notebook learn/notebooks/02_rmsnorm_rope.ipynb
```

What to notice:
- RMSNorm divides by the root-mean-square and multiplies by a learned gain.
- After RMSNorm, each vector's RMS is about 1.
- RoPE rotates vectors; it does **not** change their length.
- The attention dot product depends on **relative** position.

Checkpoint: why can RoPE be applied to queries and keys but not values?
(Answer: attention scores use q·k; values are mixed after the softmax and do
not need position.)

Exercises: 3 and 4 in [`EXERCISES.md`](EXERCISES.md).

---

### Session 3 — Attention (60 min)

**Goal:** compute attention by hand, then read the module that does it.

Read: `model/attention.py`

Run:
```bash
python learn/tour.py -s 5
jupyter notebook learn/notebooks/03_attention.ipynb
```

What to notice:
- `scores = q @ k.T / sqrt(d_head)`.
- The causal mask sets future scores to `-inf`.
- Softmax rows sum to 1.
- Sliding-window attention blocks tokens older than `window`.
- Changing the last token must not change earlier outputs.

Checkpoint: what happens to the softmax if every score in a row is `-inf`?
(Answer: it is undefined/NaN; real implementations guard against this.)

Exercises: 5 and 6 in [`EXERCISES.md`](EXERCISES.md).

---

### Session 4 — Mixture of Experts (45 min)

**Goal:** see how a model can have many parameters but use few per token.

Read: `model/moe.py`, `model/router.py`, `model/experts.py`

Run:
```bash
python learn/tour.py -s 6
jupyter notebook learn/notebooks/04_moe.ipynb
```

What to notice:
- The router picks top-`k` experts per token.
- Shared experts always run; routed experts run only when selected.
- Expert usage can be imbalanced.
- Hash routing depends on token ID; learned routing depends on the vector.

Checkpoint: if `n_routed=8` and `n_activated=2`, how many routed experts run
per token? (Answer: 2, plus the shared experts.)

Exercises: 7 and 8 in [`EXERCISES.md`](EXERCISES.md).

---

### Session 5 — Train the tiny model (45 min)

**Goal:** watch a loss go down and understand what the numbers mean.

Run:
```bash
python scripts/verify_stage01.py
cd packages/deepseek_flash_v4_mini
python training/stage01_train.py
```

What to notice:
- `train_loss` is cross-entropy on the next token.
- `validation_loss` is measured on held-out windows.
- Perplexity is `exp(loss)`; lower is better.
- The model is tiny, so the loss will not reach production quality.

Checkpoint: if validation loss is much higher than training loss, what is the
model doing? (Answer: overfitting.)

---

### Session 6 — Measure BEFORE vs AFTER (60 min)

**Goal:** learn the lab's most important habit.

Read: [`optimization_protocol.md`](optimization_protocol.md)

Run:
```bash
python scripts/run_stage01_baseline.py
python scripts/run_rope_delta_demo.py
jupyter notebook learn/notebooks/05_measure_before_after.ipynb
```

What to notice:
- The RoPE candidate is mathematically equivalent but slower on CPU.
- The verdict is `REGRESSION`.
- Keeping the baseline is the correct outcome.
- A measurement is only valid if the protocol matches.

Checkpoint: name three things that must match between baseline and candidate.
(Answer: hardware, batch/sequence shape, seed; also precision and warmup.)

Exercises: 9 and 10 in [`EXERCISES.md`](EXERCISES.md).

---

### Session 7 (optional) — V4.1 ideas (60 min)

**Goal:** see what changes in the newer model family.

Read:
- `packages/deepseek_flash_v4_1_mini/paper_notes/ced.md`
- `packages/deepseek_flash_v4_1_mini/paper_notes/csa2.md`
- `packages/deepseek_flash_v4_1_mini/paper_notes/fp4_kv.md`
- `packages/deepseek_flash_v4_1_mini/paper_notes/swa_replay.md`

Run:
```bash
cd packages/deepseek_flash_v4_1_mini
python benchmarks/v4_vs_v4_1.py
```

What to notice:
- CED splits the stack into an encoder and a decoder.
- CSA2 reuses compressed KV across layers.
- FP4 KV stores the cache in 4-bit.
- SWA Bounded Replay limits prefill work.

Important: these are **educational** implementations. The stage gates in
[`progress.md`](progress.md) decide what is considered complete.

---

### Session 8 (optional) — Qwen Mini families (90 min)

**Goal:** compare five Qwen-family mechanisms at the same tiny scale.

Read: [`QWEN_FAMILIES.md`](QWEN_FAMILIES.md)

Run:
```bash
python packages/qwen_common/benchmarks/parameter_count.py
python scripts/record_qwen_baselines.py
python comparison/compare_qwen.py
jupyter notebook packages/qwen3_next_mini/experiments/core
```

What to notice:
- Qwen2.5 and Qwen3 are standard decoders; Qwen3 adds QK-Norm.
- Qwen3-Next mixes O(T) gated linear attention with periodic full attention.
- Qwen3-VL prepends projected image patches to text tokens.
- QwQ is a Qwen3-style decoder plus a reasoning format, not a new block.
- The Qwen packages share `qwen_common`, so differences are visible.

Checkpoint: name one thing Qwen3-Next saves compared with full attention, and
one thing it gives up. (Answer: linear-time state instead of an O(T²) score
matrix; exact token-to-token recall is weaker.)

---

## Suggested pace

| Week | Sessions | Outcome |
|-----:|----------|---------|
| 1 | 0, 1, 2 | Understand the input stack |
| 2 | 3, 4 | Understand attention and MoE |
| 3 | 5, 6 | Train and measure honestly |
| 4 | 7 | Explore V4.1 ideas |
| 5 | 8 | Compare the Qwen families |

## How to know you learned it

You can answer these without notes:

1. Why is an embedding a lookup and not a matrix multiply?
2. What does RMSNorm remove, and what does it keep?
3. Why does RoPE preserve vector length?
4. Why is a causal mask necessary?
5. What is the difference between total and active parameters in MoE?
6. Why is a negative `Improve %` still a useful result?

If you can answer all six, you understand the lab's core.
