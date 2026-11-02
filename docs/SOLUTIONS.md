# Solutions

Worked answers for [`EXERCISES.md`](EXERCISES.md). Try the exercise first; then
compare your reasoning, not just your output.

---

## Exercise 1 — Parameter count

```python
from model.embedding import TokenEmbedding

for d in (8, 16, 32):
    emb = TokenEmbedding(11, d)
    print(d, emb.weight.numel())
```

Output: `8 88`, `16 176`, `32 352`.

**Formula:** `vocab_size × d_model`. Doubling `vocab_size` doubles the parameter
count (the embedding table gets twice as many rows).

---

## Exercise 2 — Unknown characters

```python
tok = CharTokenizer.from_text("the cat sat on the mat.")
ids = tok.encode("dog")
print(ids)                    # [0, 8, 0]
print(repr(tok.decode(ids)))  # ' o '
```

`d` and `g` never appeared in the training text, so they are not in `stoi` and
become the space token (`0`). `o` *did* appear (in "on"), so it survives as ID 8.

A production BPE tokenizer would fall back to smaller subword units (or a byte
fallback), so arbitrary text can still be represented without becoming spaces.

**Lesson:** a tiny vocabulary has blind spots. Always test round-trips on text
outside the training corpus.

---

## Exercise 3 — Scale invariance and gamma

```python
from model.normalization import RMSNorm

x = torch.randn(3, 8) * 4
norm = RMSNorm(8)

y1 = norm(x)
y2 = norm(x * 1000)
print((y1 - y2).abs().max())   # ~1e-7 or smaller

norm.weight.data.fill_(2.0)
y3 = norm(x)
print(y3.abs().mean(), y1.abs().mean())  # roughly 2x
```

1. The difference is tiny because RMSNorm divides by the input's own RMS. A
   global scale cancels out (up to the `eps` term).
2. `gamma = 2` doubles the output magnitude. `gamma` is a per-feature learned
   scale, not a normalization constant.
3. With `x = 0`, the RMS is `sqrt(0 + eps)`, so the result is `0`, not `NaN`.
   Without `eps`, you would divide by zero.

---

## Exercise 4 — Changing the RoPE base

```python
from model.rope import apply_rope, rope_freqs

d_head, T = 8, 8
q = torch.randn(1, 1, 1, d_head)
k = torch.randn(1, 1, 1, d_head)

def dots(base):
    f = rope_freqs(d_head, T, q.device, base=base)
    out = []
    for offset in (0, 1, 2, 4):
        qi = apply_rope(q, freqs=f[4:5])
        kj = apply_rope(k, freqs=f[4 - offset:5 - offset])
        out.append(float((qi * kj).sum()))
    return out

print("base=10000:", [round(v, 4) for v in dots(10000.0)])
print("base=100  :", [round(v, 4) for v in dots(100.0)])
```

Observations:

1. Offset 0 is unchanged (no rotation difference). Larger offsets change more,
   because a smaller base makes the higher-frequency dimensions rotate faster.
2. A smaller base → faster rotation with position → nearby positions become
   more distinguishable, but long-range relationships decorrelate sooner.
3. Rotation is orthogonal, so it preserves length. That keeps q·k magnitudes
   stable and avoids blowing up attention scores.

---

## Exercise 5 — Break causality

```python
mask = torch.ones(T, T, dtype=torch.bool)   # all allowed
masked = scores.masked_fill(~mask, float("-inf"))
attn = torch.softmax(masked, dim=-1)
```

Now modify only the last value vector:

```python
v2 = v.clone()
v2[-1] += 10.0
out2 = attn @ v2
print((out2[0] - out[0]).abs().item())   # > 0
```

1. Row 0 now puts weight on every token, including the last one, so changing
   `v[-1]` changes output 0.
2. During training this leaks the target token into earlier predictions. The
   model could "cheat" and the loss would be misleadingly low.
3. Restore `mask = torch.tril(torch.ones(T, T, dtype=torch.bool))` and the first
   output stops changing.

---

## Exercise 6 — Window size

```python
import torch
from model.attention import CausalSelfAttention, SlidingWindowAttention

T, d_model, H, dh = 5, 16, 2, 8
x = torch.randn(1, T, d_model)
full_mod = CausalSelfAttention(d_model, H, dh, use_rope=False)
full = full_mod(x)

for w in (1, 2, 5):
    win_mod = SlidingWindowAttention(d_model, H, dh, window=w, use_rope=False)
    win_mod.load_state_dict(full_mod.state_dict())  # only the mask differs
    win = win_mod(x)
    print(w, (full - win).abs().mean().item())
```

1. `W = 5` matches full attention for `T = 5`, because every token is within
   the window of every earlier token.
2. `W = 1` is cheapest: each token sees only itself. It loses all context.
3. `W = 0` makes the mask all-false, so every score is `-inf`, the softmax is
   `NaN`, and the output is `NaN`. There is no guard in this educational code.
   Production kernels validate the window.

---

## Exercise 7 — Active experts

```python
from model.moe import MoE
import torch

x = torch.randn(4, 6, 16)
for k in (1, 2, 4):
    moe = MoE(d_model=16, n_routed=8, n_shared=1, n_activated=k,
              hidden=32, vocab_size=16)
    total = sum(p.numel() for p in moe.parameters())
    print("k =", k, "params =", total)
```

1. Total parameters are **the same** for every `k`. All 8 routed experts always
   exist; `k` only changes how many are selected.
2. Routed expert calls per token = `k`. Shared experts add a constant amount.
3. `k = 1` is cheapest per token, but less expressive than `k = 2` or `4`.

This is the key MoE distinction: **total parameters ≠ active compute**.

---

## Exercise 8 — Expert imbalance

```python
import torch
from model.moe import MoE

for seed in range(5):
    torch.manual_seed(seed)
    moe = MoE(d_model=16, n_routed=8, n_shared=1, n_activated=2,
              hidden=32, vocab_size=16)
    x = torch.randn(4, 16, 16)
    topv, topi = moe.router(x.reshape(-1, 16))
    hist = torch.bincount(topi.reshape(-1), minlength=8)
    print(seed, hist.tolist(), "ratio", round((hist.max() / hist.float().mean()).item(), 2))
```

1. Routing is rarely perfectly balanced; some experts get many more tokens.
2. Extreme imbalance means some experts rarely train, so they stay random. The
   model effectively loses capacity, and gradients become noisy.
3. `Router.forward` updates a `bias` buffer during training:

   ```python
   self.bias.add_(self.balance_lr * (1.0 / self.n_routed - load))
   ```

   Experts that are over-used get a negative nudge; under-used experts get a
   positive nudge. This is a simple load-balancing heuristic, not a full
   auxiliary loss.

---

## Exercise 9 — Loss and perplexity

```python
import json, math

m = json.load(open("results/stage01/metrics.json"))
print(m["train_loss"], m["validation_loss"])
print("exp(val) =", math.exp(m["validation_loss"]))
print("stored   =", m["perplexity_val"])
print("gap      =", m["validation_loss"] - m["train_loss"])
```

1. Perplexity is the effective number of equally likely next tokens. Lower is
   better. `perplexity = exp(cross_entropy_loss)`.
2. A large positive gap suggests overfitting: the model fits training windows
   better than held-out windows.
3. The loss is not near zero because the model is tiny, the corpus is tiny, and
   the task (predict the next character) is genuinely uncertain. The best
   possible loss is the entropy of the data, not zero.

---

## Exercise 10 — Interpret and design a comparison

```python
b = 0.009511859389021993
c = 0.06374
print((b - c) / b * 100)   # about -570%
```

1. The negative value means the candidate is slower: `Improve % < 0`.
2. Changing batch size from 4 to 8 is **not** an optimization comparison. A
   larger batch changes latency, memory, and tokens/second all at once, so you
   cannot attribute the difference to a code change.
3. To make it fair, hold batch size, sequence length, dtype, hardware, warmup,
   and seeds fixed. If you must study batch size, treat it as a separate scaling
   study and report per-token metrics alongside per-batch metrics.
4. Trade-off example: a candidate uses 30% less KV memory but is 10% slower.
   Memory is better, latency is worse → `TRADE-OFF`, and the decision depends on
   which constraint matters.

---

## Stretch exercise

There is no single answer. A good submission:

- states input/output shapes,
- names the learned parameters,
- identifies the nonlinearity and residual path,
- explains what the component would do differently at production scale,
- includes a runnable 5-line cell with a printed shape.

If your explanation matches the six questions at the end of
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md), you are done.
