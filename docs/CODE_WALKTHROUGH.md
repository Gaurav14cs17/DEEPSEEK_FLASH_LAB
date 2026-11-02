# Code walkthrough

A plain-language map of the source. Read this alongside the code; every section
tells you **what the file is for**, **what shapes flow through it**, and **why
it exists**.

If you only have 30 minutes, read the "Core reading order" table and then the
first four files.

---

## Core reading order

| Order | File | One-line purpose |
|------:|------|------------------|
| 1 | `DeepSeekFlashV4-Mini/tokenizer/tokenizer.py` | text ↔ integer IDs |
| 2 | `model/embedding.py` | IDs → vectors |
| 3 | `model/normalization.py` | keep vector scale stable |
| 4 | `model/rope.py` | encode position by rotation |
| 5 | `model/attention.py` | mix information across tokens |
| 6 | `model/moe.py` + `router.py` | choose experts per token |
| 7 | `model/decoder.py` | one full transformer block |
| 8 | `model/model.py` | stack blocks into a model |
| 9 | `training/stage01_train.py` | train and measure |
| 10 | `bench/delta.py` | compare BEFORE vs AFTER |

---

## 1. Tokenizer

**File:** `DeepSeekFlashV4-Mini/tokenizer/tokenizer.py`

```python
tok = CharTokenizer.from_text("the cat sat")
ids = tok.encode("cat")     # [3, 2, 10] (example)
text = tok.decode(ids)      # "cat"
```

- `stoi`: string → integer.
- `itos`: integer → string.
- `vocab_size`: number of distinct characters.
- Unknown characters map to the space token (a deliberate simplification).

**Why it matters:** the model only ever sees integers. Everything downstream
depends on this vocabulary.

**Fidelity:** `[SIMPLIFIED]` — production DeepSeek uses BPE, not characters.

---

## 2. Embedding

**File:** `model/embedding.py`

```python
emb = TokenEmbedding(vocab_size=11, d_model=8)
x = emb(tokens)   # [B, T] -> [B, T, 8]
```

- The parameter matrix is `[vocab_size, d_model]`.
- Forward is a row lookup: `E[token_id]`.
- No bias, no nonlinearity.

**Beginner mistake to avoid:** thinking the embedding multiplies by a matrix.
It does not. It selects rows.

**Memory:** `vocab_size × d_model` parameters.

---

## 3. RMSNorm

**File:** `model/normalization.py`

```
RMS(x) = sqrt(mean(x²) + eps)
y      = x / RMS(x) * gamma
```

- Operates on the last dimension.
- No mean subtraction.
- No bias.
- `gamma` is a learned vector of length `d_model`, initialised to ones.

**Why it exists:** deep stacks amplify or shrink activations. Normalizing keeps
the scale in a usable range so gradients flow.

**Shape:** `[..., d] → [..., d]` (unchanged).

---

## 4. RoPE

**File:** `model/rope.py`

Two functions matter:

1. `rope_freqs(d_head, seq_len, device)` builds a complex tensor `[T, d_head/2]`.
   Each entry is `cos(angle) + i·sin(angle)` for a position and frequency.
2. `apply_rope(x, freqs)` rotates pairs of dimensions.

```python
q = torch.randn(B, H, T, d_head)
q_rot = apply_rope(q, freqs=rope_freqs(d_head, T, q.device))
```

Key facts:
- `d_head` must be even.
- The norm of each vector is unchanged.
- `dot(q_i, k_j)` depends on `i - j`, not on the absolute positions.
- `rope_reference_rotate_half` is a slower, equivalent formulation used by tests
  and by the measurement demo.

---

## 5. Attention

**File:** `model/attention.py`

Two classes:

| Class | Mask | Use |
|-------|------|-----|
| `CausalSelfAttention` | future tokens blocked | baseline |
| `SlidingWindowAttention` | future + old tokens blocked | cheap local context |

The forward pass in four lines:

```python
q, k, v = split(qkv(x))
scores = q @ k.transpose(-2, -1) / sqrt(d_head)
scores = scores.masked_fill(~allowed, -inf)
out = softmax(scores) @ v
```

Shapes:
- input `[B, T, d_model]`
- q/k/v `[B, H, T, d_head]`
- scores `[B, H, T, T]`
- output `[B, T, d_model]`

**Why the mask:** without it, token 0 could read token 99, which would leak the
answer during training.

**Cost:** score matrix grows as `T²`. The sliding window caps the number of
allowed keys, which is the main idea behind cheap long-context attention.

---

## 6. Compressed attention (V4-specific)

**File:** `model/mla.py`

Three educational variants:

| Class | Idea | Compression |
|-------|------|-------------|
| `CSA` | compressed sparse attention | overlap-compress K/V, then top-k |
| `HCA` | heavily compressed attention | non-overlap-compress K/V |
| `MLA` | multi-head latent attention | low-rank KV bottleneck |

`compress_overlap` averages windows that overlap by half, so information at
window boundaries is not lost. `compress_nonoverlap` averages disjoint blocks.

`CSA` also adds a sliding-window branch, so each token keeps precise local
context plus a cheap compressed global view.

**Fidelity:** `[FAITHFUL]` idea, `[SIMPLIFIED]` scale.

---

## 7. MoE

**Files:** `model/moe.py`, `model/router.py`, `model/experts.py`,
`model/shared_experts.py`

Flow:

```
flat tokens
  ├─ shared experts (always on)
  └─ router → top-k expert IDs → run only those experts → weighted sum
```

- `Router.scores` uses `sqrt(softplus(gate(x)))`.
- `topk` chooses experts; weights are normalized and scaled.
- During training, a small bias update nudges usage toward balance.
- `HashGate` is a fixed token-ID → expert table for the first layers.

**Why it exists:** total parameters can be large while compute per token stays
small. This is the central efficiency trick of modern MoE models.

**Watch for:** expert collapse — all tokens choosing the same expert. The
histogram in the training log is the diagnostic.

---

## 8. Decoder block

**File:** `model/decoder.py`

One block does:

```python
h = norm1(x)
y = attention(h)
y2 = moe(norm2(x + y))
x = x + y + y2
```

This is a pre-norm residual block:
- normalize before the sublayer,
- add the sublayer output back to the input.

If `use_mhc` is on, a multi-stream residual path wraps the block.

**Why residuals:** they give gradients a short path to earlier layers, which
makes deep stacks trainable.

---

## 9. Full model

**File:** `model/model.py`

`DeepSeekFlashV4Mini` does:

```
tokens
  → embedding
  → N decoder blocks (SWA / CSA / HCA pattern)
  → final RMSNorm
  → LM head (tied to the embedding)
  → optional MTP heads
```

The LM head produces `[B, T, vocab_size]` logits. Cross-entropy against the
next token is the training loss.

**Weight tying:** the LM head reuses the embedding matrix. This saves
parameters and is common in small language models.

---

## 10. Training

**File:** `training/stage01_train.py`

The training loop is deliberately plain:

```python
logits = model(x)
loss = cross_entropy(logits, y)
opt.zero_grad()
loss.backward()
opt.step()
```

It also:
- estimates memory before training and aborts if over budget,
- records peak CPU/GPU memory,
- measures tokens/second,
- writes `results/stage01/metrics.json`.

**Stage-1 note:** the attention inside this script is a temporary inline block
marked `[EXPERIMENTAL]`. The real attention module arrives at Stage 2.

---

## 11. Measurement harness

**Files:** `bench/protocol.py`, `bench/delta.py`, `bench/latency.py`,
`bench/memory.py`, `bench/compare.py`, `bench/regression.py`

The important module for beginners is `bench/delta.py`:

- `compare_metrics(baseline, candidate)` builds rows.
- `format_table(rows)` renders the Markdown table.
- `verdict(rows)` returns IMPROVEMENT / REGRESSION / TRADE-OFF / NEUTRAL.
- `assert_same_protocol(...)` refuses unfair comparisons.

Metric direction is encoded explicitly:
- latency, memory, loss → lower is better,
- throughput, similarity → higher is better,
- parameter counts → neutral.

**This is the heart of the lab.** The model is small; the discipline is the
point.

---

## 12. V4.1 differences

The `DeepSeekFlashV4.1-Mini/` package reuses the same primitives and adds:

| File | Idea |
|------|------|
| `model/ced.py` | causal encoder–decoder split |
| `model/csa2.py` | compressed sparse attention v2 |
| `model/_swa.py` | sliding-window replay |
| `optimization/fp4_kv_cache.py` | simulated FP4 KV cache |
| `model/vision_encoder.py` | multimodal entry |
| `model/_engram.py`, `_dspark.py` | extra heads / memory ideas |

Read these **after** the V4 path. They assume you already understand attention,
RoPE, and KV caching.

---

## How to read unfamiliar code

Use this checklist:

1. What are the input and output shapes? (Look for comments like `[B, T, d]`.)
2. What parameters are learned? (`nn.Linear`, `nn.Embedding`, `nn.Parameter`.)
3. Where is the nonlinearity? (`softmax`, `GELU`, `SiLU`.)
4. Where is the residual add?
5. What is the mask, and what does it forbid?
6. What would break if you removed this line? (Then try it in a notebook.)

If you can answer those six for a file, you can read the file.
