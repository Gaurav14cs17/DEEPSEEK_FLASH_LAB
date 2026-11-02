# learn/ — beginner layer

Hands-on materials that sit on top of the research lab. Nothing here changes
the measured results; it only explains and demonstrates them.

## Quick start

```bash
cd deepseek_flash_lab           # repo root
pip install -r requirements.txt

python learn/tour.py            # guided, press Enter between sections
python learn/tour.py --all      # run straight through
python learn/tour.py -s 5       # just the attention section
```

Then open the notebooks:

```bash
jupyter notebook learn/notebooks
```

Qwen family notebooks live with their packages:

```bash
jupyter notebook packages/qwen2_5_mini/experiments/core
jupyter notebook packages/qwen3_next_mini/experiments/core
```

See [`docs/QWEN_FAMILIES.md`](../docs/QWEN_FAMILIES.md).

## Notebooks

| Notebook | Topic | Session |
|----------|-------|--------:|
| `01_tokens_embeddings.ipynb` | text → IDs → vectors | 1 |
| `02_rmsnorm_rope.ipynb` | normalization and position | 2 |
| `03_attention.ipynb` | attention and masks | 3 |
| `04_moe.ipynb` | routing and experts | 4 |
| `05_measure_before_after.ipynb` | baseline → delta → verdict | 6 |

All notebooks are already executed, so you can read the outputs without running
anything. When you run them yourself, results may differ slightly (especially
timings) — that is expected.

## Suggested order

1. `docs/LEARNING_PATH.md` — the 7-session route.
2. `learn/tour.py` — 10-minute tour.
3. Notebooks 1 → 5.
4. `docs/EXERCISES.md` — one set per session.
5. `docs/SOLUTIONS.md` — only after trying.
6. `docs/CODE_WALKTHROUGH.md` — read the source.

## Rules of the lab

- **Predict before you run.** Write down the shape or number first.
- **Measure before you claim.** No optimization without a baseline.
- **A regression is a result.** If the candidate is slower, say so and keep the
  baseline.
- **Tiny does not mean fake.** The mechanisms are real; only the scale is small.

## Requirements

Only `torch` is needed for the tour and notebooks 1–4. Notebook 5 also uses the
local `bench/` package (standard library only). No GPU, no downloads.
