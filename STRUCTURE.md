# Project structure

This repo is a lab, not a single application. The layout separates **models**,
**measurement**, **teaching**, and **history** so each can evolve without
breaking the others.

```
deepseek_flash_lab/
├── packages/                 # importable model libraries
│   ├── deepseek_flash_v4_mini/
│   └── deepseek_flash_v4_1_mini/
├── bench/                    # measurement harness (B0 → B1 → Δ)
├── comparison/               # A/B scripts for the two Minis
├── configs/                  # YAML presets
├── datasets/                 # capped data loaders + Level-0 shard
├── docs/                     # paper, protocol, learning path, exercises
├── learn/                    # beginner tour + executed notebooks
├── results/                  # measured JSON bundles (tracked evidence)
├── scripts/                  # user-facing entry points
├── verification/             # independent audit suite + reports
├── archive/                  # ignored legacy experiments
├── pyproject.toml            # install + pytest + ruff config
└── requirements.txt          # full dev/learning environment
```

---

## 1. Packages are importable

Both model families are real Python packages under `packages/`:

```python
from deepseek_flash_v4_mini import DeepSeekFlashV4Mini, preset
from deepseek_flash_v4_1_mini import DeepSeekFlashV41Mini, preset
```

No more hyphenated folders and no more `sys.path` collision between two
top-level modules both named `model`.

### Optional editable install

```bash
pip install -e .
```

Then imports work from any directory. Running from the repo root also works
without installing because scripts and tests add `packages/` to `sys.path`
once, at their entry point.

---

## 2. One package = one model family

Each package has the same internal layout:

| Directory | Responsibility |
|-----------|----------------|
| `model/` | architecture: embeddings, attention, MoE, decoder, config |
| `tokenizer/` | text ↔ IDs |
| `training/` | loss, scheduler, checkpoint, pretrain loop |
| `inference/` | sampling / generation helpers |
| `optimization/` | KV accounting, quantization, memory, profiling |
| `benchmarks/` | small runnable benchmark scripts |
| `tests/` | package-level smoke tests |
| `paper_notes/` | paper → code mapping |
| `experiments/` | notebooks |

Notebooks are grouped by intent:

- `experiments/core/` — primitives shared by both families
  (transformer, RoPE, attention, MLA, MoE, router, training, KV, quantization)
- `experiments/v4_1/` — mechanisms specific to V4.1
  (CED, CSA2, FP4 KV, SWA replay, multimodal)

Numbering is unique **within** each folder. There is no duplicate `07_moe.ipynb`
in the same directory.

---

## 3. Import rules

Inside a package, use absolute package imports:

```python
# packages/deepseek_flash_v4_mini/training/checkpoint.py
from deepseek_flash_v4_mini.model.config import V4Config
from deepseek_flash_v4_mini.tokenizer.tokenizer import CharTokenizer
```

Why absolute? It is unambiguous, works the same when installed or run from the
repo root, and makes the dependency direction visible.

Entry points (`training/pretrain.py`, `benchmarks/*.py`, `tests/*.py`,
`experiments/*.ipynb`) add `packages/` to `sys.path` once and then import the
package by name.

Do **not** add a second `sys.path` hack in a library module.

---

## 4. Where does a new file go?

| You are adding… | Put it in… |
|-----------------|------------|
| a new model component | `packages/<pkg>/model/` + a test in `tests/` |
| a training feature | `packages/<pkg>/training/` |
| a KV/quantization idea | `packages/<pkg>/optimization/` |
| a micro-benchmark | `packages/<pkg>/benchmarks/` |
| a teaching notebook | `packages/<pkg>/experiments/core/` or `v4_1/` |
| a formal A/B experiment | `scripts/` + output under `results/` |
| a cross-model comparison | `comparison/` |
| an audit/check | `verification/tests/` |
| an explanation | `docs/` |
| a beginner lesson | `learn/` |

Measured evidence always goes under `results/<family>/<experiment>/<version>/`
as `config.json`, `metrics.json`, and `memory.json`.

---

## 5. Commands that follow the structure

```bash
# Learn
python learn/tour.py
jupyter notebook learn/notebooks

# Verify
python -m pytest -q verification/tests
python verification/run_full_verification.py

# Stage-1 gate
python scripts/verify_stage01.py

# Compare the two families
cd comparison && python compare_parameters.py

# Package smoke tests
python packages/deepseek_flash_v4_mini/tests/test_rope.py

# Package notebooks
jupyter notebook packages/deepseek_flash_v4_mini/experiments/core
```

---

## 6. Naming conventions

| Kind | Convention | Example |
|------|------------|---------|
| Python package | lowercase + underscores | `deepseek_flash_v4_mini` |
| Module | lowercase | `causal_encoder.py` |
| Class | PascalCase | `DeepSeekFlashV41Mini` |
| Config | PascalCase + `Config` | `V4Config` |
| Result bundle | `<model>/<experiment>/<version>/` | `v4/rope_formulation/A1_rotate_half/` |
| Notebook | `NN_topic.ipynb`, unique per folder | `core/03_attention.ipynb` |
| Report | `lower_snake.md` | `optimization_report.md` |

---

## 7. Legacy

`archive/oldDATA/` contains the pre-Mini toy experiments. It is ignored by git,
kept only for reference, and is **not** part of the active layout. Nothing in
`packages/`, `bench/`, or the verification suite imports it.
