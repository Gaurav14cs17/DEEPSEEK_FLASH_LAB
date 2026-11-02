# Exercises

Beginner practice for the 7-session path. Do these **before** opening
[`SOLUTIONS.md`](SOLUTIONS.md).

Each exercise is small: a few lines in a notebook or a Python shell. The point
is to predict first, then run, then explain the difference.

> Setup for every exercise: run the setup cell from the matching notebook, or
> start from `learn/tour.py` imports.

---

## Session 1 — Tokens and embeddings

### Exercise 1 — Predict the parameter count

`TokenEmbedding(vocab_size=11, d_model=8)` has how many parameters?

Before running: write your prediction.
Then create embeddings with `d_model = 8`, `16`, `32` and print
`emb.weight.numel()`.

**Question:** what is the formula? What happens if `vocab_size` doubles?

---

### Exercise 2 — The unknown-character blind spot

Build a tokenizer from the text `"the cat sat on the mat."`.
Encode `"dog"` and decode the result.

**Questions:**
1. Which IDs do you get?
2. Why does `"dog"` not round-trip?
3. What would a production BPE tokenizer do differently?

---

## Session 2 — Normalization and position

### Exercise 3 — Scale invariance and gamma

Start with `x = torch.randn(3, 8) * 4` and `norm = RMSNorm(8)`.

1. Compute `norm(x)` and `norm(x * 1000)`.
2. Print the maximum absolute difference.
3. Set `norm.weight.data.fill_(2.0)` and run again.

**Questions:**
1. Why is the difference so small for `x * 1000`?
2. What does `gamma = 2` do to the output scale?
3. Why is `eps` needed when `x` is exactly zero?

---

### Exercise 4 — Change the RoPE base

`rope_freqs(d_head, seq_len, device, base=10000.0)`.

Build frequencies with `base=10000` and `base=100`. Apply each to the same
`q` tensor and compare:

1. the norm before/after (should still match),
2. the dot products at relative offsets 0, 1, 2, 4.

**Questions:**
1. Which offsets change the most when the base changes?
2. In plain English, what does a smaller base do to the rotation speed?
3. Why must RoPE keep the norm unchanged?

---

## Session 3 — Attention

### Exercise 5 — Break causality

In notebook 3, find the manual attention cell and replace the causal mask with
`mask = torch.ones(T, T, dtype=torch.bool)` (all allowed).

1. Print the attention weights.
2. Change the last token's `v` vector only.
3. Check whether the first output changes.

**Questions:**
1. Why does the first output now change?
2. What would this mean during training?
3. Restore the mask and confirm the leak disappears.

---

### Exercise 6 — Window size

Use `SlidingWindowAttention(d_model=16, n_heads=2, d_head=8, window=W)` with
`T = 5` and `W = 1, 2, 5`.

1. Compare each output against full causal attention.
2. Print the mean absolute difference.

**Questions:**
1. Which `W` matches full attention exactly? Why?
2. Which `W` is cheapest? What context does it lose?
3. What happens to the mask when `W = 0`? (Try it and read the error or output.)

---

## Session 4 — Mixture of Experts

### Exercise 7 — Change the number of active experts

Build a MoE with `n_routed=8`, `n_shared=1`, `hidden=32`, `d_model=16`.

Run it with `n_activated = 1, 2, 4`.

**Questions:**
1. How does total parameter count change?
2. How does the number of routed expert calls per token change?
3. Which setting is cheapest per token?

---

### Exercise 8 — Expert imbalance

Use the same MoE with a fixed input batch. Run the router 5 times with different
random inputs (change `torch.manual_seed`).

1. Print the histogram each time.
2. Compute `max(hist) / mean(hist)`.

**Questions:**
1. Is the routing balanced?
2. Why is extreme imbalance a problem during training?
3. What mechanism in `router.py` tries to fix it?

---

## Session 5 — Training

### Exercise 9 — Loss and perplexity

Run `python training/stage01_train.py` from `DeepSeekFlashV4-Mini/`.
Read `results/stage01/metrics.json`.

1. Compute `exp(validation_loss)` yourself.
2. Compare it with the stored `perplexity_val`.
3. Compute the gap `validation_loss - train_loss`.

**Questions:**
1. What does perplexity mean in plain English?
2. Is the gap a sign of overfitting or underfitting?
3. Why is the loss not close to zero?

---

## Session 6 — Measurement

### Exercise 10 — Interpret and design a comparison

Use the stored RoPE experiment
(`results/v4/rope_formulation/A0_complex` and `A1_rotate_half`).

1. Compute the `Improve %` for `forward_ms` by hand.
2. Confirm it matches the table.
3. Design a comparison where you change the batch size from 4 to 8.

**Questions:**
1. Why is the batch-size comparison invalid as an "optimization"?
2. What would you have to hold fixed to make it fair?
3. Give an example of a TRADE-OFF verdict: one metric better, one worse.

---

## Stretch exercise (optional)

Pick one file from `DeepSeekFlashV4-Mini/model/` that we did not cover
(`mhc.py`, `mtp.py`, `experts.py`, `shared_experts.py`).

Answer the six reading questions from the end of
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md), then write a 5-line notebook cell
that runs it on a tiny tensor and prints the output shape.
