# deepseek_flash_lab

Learn **DeepSeek-V4-Flash** and **DeepSeek-V4.1-Flash** by building small, trainable Minis on a laptop (~**2 GB** GPU or **CPU**).

| This lab **is** | This lab **is not** |
|-----------------|---------------------|
| Paper → math → code → **measure Δ** → analyze | The real 284B / 552B Flash models |
| Two sequential Minis: **V4 first**, then V4.1 | A chat API or production cluster |
| Tiny configs that must fit ~2 GB VRAM | Multi-GPU serving |

```text
  docs/ (paper)
       ↓
  deepseek_flash_v4_mini   →  V4-Flash ideas (Stage 1 ✅ … Stage 3)
       ↓
  deepseek_flash_v4_1_mini →  CED / CSA2 / FP4 KV / SWA replay
       ↓
  bench/ + results/      →  BEFORE vs AFTER → Improve % → VERDICT
```

---

## New here? Start with the beginner path

Everything below is the research lab. If you are new to transformers, begin
with the learning layer instead:

```bash
python learn/tour.py                 # 10-minute guided tour (CPU only)
jupyter notebook learn/notebooks     # 5 hands-on notebooks
```

| Resource | What it gives you |
|----------|-------------------|
| [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) | 7-session curriculum, beginner → measurement |
| [`learn/tour.py`](learn/tour.py) | Interactive CLI tour of every core component |
| [`learn/notebooks/`](learn/notebooks) | Tokens, RMSNorm/RoPE, attention, MoE, measurement |
| [`docs/EXERCISES.md`](docs/EXERCISES.md) | Practice tasks for each session |
| [`docs/SOLUTIONS.md`](docs/SOLUTIONS.md) | Worked answers |
| [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) | File-by-file source map |
| [`STRUCTURE.md`](STRUCTURE.md) | Repository layout and import conventions |

---

## Quick start

```bash
cd deepseek_flash_lab
pip install -r requirements.txt

# optional: make the two model packages importable everywhere
pip install -e .

python scripts/verify_stage01.py          # Stage-1 gate
python scripts/run_stage01_baseline.py    # formal B0 (required before any opt)
python scripts/run_rope_delta_demo.py     # example BEFORE→AFTER→Δ (see verdict)
```

| Doc | Link |
|-----|------|
| Progress | [`docs/progress.md`](docs/progress.md) — Stage 1 passed; Stage 2+ locked |
| Paper | [`docs/DeepSeek_V41_Tech_Report.pdf`](docs/DeepSeek_V41_Tech_Report.pdf) |
| Bench protocol | [`docs/optimization_protocol.md`](docs/optimization_protocol.md) |

Deferred stubs (`compression`, `dsa`, V4.1 CED/CSA2, …) stay locked until their checklist unlocks.

---

## What is V4 / V4-Flash / V4.1-Flash?

| Name | From the reports | Local package |
|------|------------------|---------------|
| **DeepSeek-V4** | MoE family; Flash vs Pro scale | concepts |
| **DeepSeek-V4-Flash** | 284B MoE, ~13B active; CSA–HCA; exact SWA replay ~\(L\times n_{\mathrm{win}}\) | `deepseek_flash_v4_mini` |
| **DeepSeek-V4.1-Flash** | 552B multimodal MoE; CED; CSA2; FP4 KV; SWA Bounded Replay | `deepseek_flash_v4_1_mini` |

V4.1 must keep prefill vs decode visible:

```text
PREFILL:  input → encoder → compressed state → decoder
DECODE:   token  → decoder → global/compressed KV → next token
```

---

## Qwen Mini families (optional extension)

Five Qwen-family Minis share one core (`packages/qwen_common/`) and the same
package layout, tests, notebooks, and B0 protocol:

| Package | Mechanism | Notebook |
|---------|-----------|----------|
| `qwen2_5_mini` | GQA + RoPE + SwiGLU, dense + MoE | `experiments/core/01_qwen2_5_mini.ipynb` |
| `qwen3_mini` | QK-Norm + thinking/non-thinking mode | `experiments/core/01_qwen3_mini.ipynb` |
| `qwen3_next_mini` | hybrid gated linear + full attention | `experiments/core/01_qwen3_next_mini.ipynb` |
| `qwen3_vl_mini` | tiny ViT + projector + text decoder | `experiments/core/01_qwen3_vl_mini.ipynb` |
| `qwq_mini` | long-CoT reasoning format | `experiments/core/01_qwq_mini.ipynb` |

```bash
python scripts/record_qwen_baselines.py   # B0 for every Qwen family
python comparison/compare_qwen.py         # params + latency table
```

Full guide: [`docs/QWEN_FAMILIES.md`](docs/QWEN_FAMILIES.md). These are
architecture labs, not reproductions of Qwen's trained quality.

---

## Repo map

```text
deepseek_flash_lab/
├── README.md · STRUCTURE.md · pyproject.toml · requirements.txt
├── packages/
│   ├── deepseek_flash_v4_mini/       # import: deepseek_flash_v4_mini
│   │   ├── model/ tokenizer/ training/ inference/ optimization/
│   │   ├── benchmarks/ tests/ paper_notes/
│   │   └── experiments/core/         # 01…11, numbered by topic
│   ├── deepseek_flash_v4_1_mini/     # import: deepseek_flash_v4_1_mini
│   │   ├── model/ tokenizer/ training/ inference/ optimization/
│   │   ├── benchmarks/ tests/ paper_notes/
│   │   └── experiments/
│   │       ├── core/                 # shared V4 primitives
│   │       └── v4_1/                 # CED / CSA2 / FP4 KV / SWA replay
│   ├── qwen_common/                  # shared Qwen primitives
│   ├── qwen2_5_mini/                 # GQA + MoE
│   ├── qwen3_mini/                   # QK-Norm + thinking mode
│   ├── qwen3_next_mini/              # hybrid linear/full attention
│   ├── qwen3_vl_mini/                # tiny ViT + text decoder
│   └── qwq_mini/                     # long-CoT reasoning format
├── configs/                 # v4_tiny / v4_small / v4_1_*
├── datasets/                # Level 0–5 (capped downloads)
├── docs/                    # paper, progress, protocol, learning path
├── learn/                   # beginner tour + notebooks
├── bench/                   # BEFORE→AFTER→Δ harness
├── scripts/                 # verify + baseline + delta demos
├── results/                 # v4/… and qwen/<family>/baseline/ bundles
├── comparison/              # optimization_history.csv + A/B scripts
├── verification/            # independent audit suite + reports
└── archive/oldDATA/         # ignored legacy experiments
```

---

## Hardware (2 GB rule)

- Prefer `configs/*_tiny.yaml`
- Print memory estimates before train; **abort** if over budget
- Micro-batch, grad accum, short seq, checkpointing when needed
- **CPU always supported** (current Stage-1 runs are CPU)

---

## Datasets

See [`datasets/README.md`](datasets/README.md). Never download huge corpora by default.

| Level | Data | Use |
|------:|------|-----|
| 0 | `datasets/shards/level0_char.txt` | Stage 1 / unit tests |
| 1–3 | TinyStories / FineWeb (capped) | later training |
| 4–5 | multimodal / long-context synthetic | V4.1 / KV labs |

```bash
python datasets/download.py --dry-run
python datasets/download.py --num-samples 256 --max-tokens 50000 --seed 42
```

---

## Fidelity labels

| Tag | Meaning |
|-----|---------|
| **[FAITHFUL]** | Documented paper mechanism, miniaturized |
| **[SIMPLIFIED]** | Scaled for 2 GB |
| **[SIMULATED]** | Systems behavior without cluster infra |
| **[EXPERIMENTAL]** | Local probe — not a production claim |

---

## Where we are

| Stage | Status |
|------:|--------|
| 0 Scaffold | done |
| 1 Tokenizer, embedding, RMSNorm, RoPE | **done** |
| 2 Attention, decoder, compression, CSA/HCA | next |
| 3 MoE, KV, inference, profiling | later |
| 4 V4.1-Mini + full comparison | after V4 |

### Stage-1 measured B0 (`results/v4/stage01/baseline/`)

| Metric | Value |
|--------|------:|
| Parameters | 100,480 |
| Dataset tokens | 401 (Level 0) |
| Train loss | ~1.76 |
| Val loss | ~1.60 |
| Val perplexity | ~5.0 |
| Device | CPU (peak GPU 0 MB) |
| Peak CPU RAM | ~789 MB |
| Training tokens/sec | ~53k |
| Step time (train loop) | ~3.6 ms |
| Train forward / backward / optimizer | ~1.25 / ~1.69 / ~0.66 ms |

Also: `results/stage01/metrics.json` from `training/stage01_train.py`.

---

## Optimization → Benchmark → Delta (mandatory)

**Never call something an optimization without a measured baseline.**

```text
BEFORE (B0) → MEASURE → IMPLEMENT candidate → AFTER (B1)
    → SAME protocol → Improve % table → VERDICT → SAVE
```

Full protocol (mandatory metrics, inference/training/KV grids, ladder L0–L5, report template): [`docs/optimization_protocol.md`](docs/optimization_protocol.md)

| Term | Meaning |
|------|---------|
| **Baseline (A0)** | Recorded reference |
| **Candidate (A1)** | Challenger — not assumed better |
| **Improve %** | `>0` = better for that metric; `<0` = worse |
| **VERDICT** | `IMPROVEMENT` / `REGRESSION` / `TRADE-OFF` / `NEUTRAL` |

Unfair compares (different batch, seq, seed, warmup, precision, hardware) are **rejected**.

```bash
# Record / compare
python scripts/run_stage01_baseline.py
python scripts/run_rope_delta_demo.py

python -m bench.compare \
  --baseline results/v4/rope_formulation/A0_complex \
  --optimized results/v4/rope_formulation/A1_rotate_half

python -m bench.regression \
  --previous results/v4/stage01/baseline \
  --current results/v4/<your_candidate>
```

History: [`comparison/optimization_history.csv`](comparison/optimization_history.csv)

`bench/` modules: `protocol` · `memory` · `latency` · `inference` · `training_metrics` · `kv` · `accuracy` · `ladder` · `delta` · `compare` · `regression` · `report` · `history`

### How to read a delta table

If you see **Improve % ≈ −500%** and **Better? = NO** on `forward_ms`, the candidate is **slower**. That is a successful measurement, not a broken script.

### V4 optimization dashboard

Only **measured** rows appear here. Missing experiments are not unfinished docs — they are **not unlocked yet** (`docs/progress.md`).

| Optimization | Memory Δ | Prefill Δ | Decode Δ | Throughput Δ | KV Δ | Quality Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (Stage-1 B0) | — | — | — | — | — | — |
| RoPE A0 complex → A1 rotate-half | ~0 | — | — | A1 slower (`forward_ms` Improve % ≪ 0) | — | ≈0 abs error |

**Verdict (RoPE A1):** REGRESSION — keep A0. Full write-up: `results/v4/rope_formulation/optimization_report.md`

**Upcoming (no Δ yet — Stage 2+ locked):** attention, compression, DSA/CSA–HCA, MoE, KV cache, quantization.

### V4.1 dashboard

Locked until V4 Stages 1–3 complete. Rows appear only after B0+B1 JSON exists under `results/v4_1/`.

| Optimization | Memory Δ | Prefill Δ | Decode Δ | Throughput Δ | KV Δ | Quality Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 baseline | — | — | — | — | — | — |
| CED | *(locked)* | | | | | |
| CSA2 | *(locked)* | | | | | |
| FP4 KV | *(locked)* | | | | | |
| SWA Replay | *(locked)* | | | | | |

---

## Commands cheat sheet

```bash
pip install -r requirements.txt

# Learn (beginner)
python learn/tour.py
jupyter notebook learn/notebooks

# Stage 1
python scripts/verify_stage01.py
python scripts/run_stage01_baseline.py
python scripts/run_rope_delta_demo.py

cd packages/deepseek_flash_v4_mini
python tests/test_rope.py
python training/stage01_train.py
```

Notebook: `packages/deepseek_flash_v4_mini/experiments/core/01_transformer.ipynb`

---

## What can / cannot be reproduced

| Can | Cannot |
|-----|--------|
| Shapes, math, unit tests, Δ tables | 284B / 552B quality |
| CPU + ≤2 GB training | Multi-GPU / EP / TP |
| Tiny CSA/CSA2 / MoE / KV *ideas* | Production HBM FP4 kernels |

---

## Principle

**Correctness > speed · Understanding > copying · Measurement > assumptions**

**BEFORE → MEASURE → AFTER → Improve % → VERDICT → SAVE** — every time.

A slower candidate with **Better? = NO** means: **keep the baseline**. That is science, not failure.
