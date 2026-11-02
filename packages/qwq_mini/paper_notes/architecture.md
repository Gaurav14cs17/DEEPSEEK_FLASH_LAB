# QwQ Mini — paper → code

## Family

QwQ is a reasoning-focused model. Architecturally it is a Qwen-family decoder;
the distinctive part is long chain-of-thought **post-training** and a decoding
policy that keeps the reasoning trace before the final answer.

## Code map

| Idea | File |
|------|------|
| config (longer context, thinking on) | `config.py` |
| model + format/parse helpers | `model.py` |
| tests | `tests/test_model.py` |
| notebook | `experiments/core/01_qwq.ipynb` |

## Fidelity

- `[FAITHFUL]` thinking-token format and separation of reasoning vs answer.
- `[SIMPLIFIED]` weights are a randomly initialised Qwen3-style decoder. No
  reasoning fine-tuning data is included, so generations are not meaningful.
- `[SIMULATED]` long-CoT behaviour is represented by the prompt/parse contract,
  not by trained reasoning quality.

## What to inspect

1. `QwQMini.format_reasoning` wraps a question in ` thinking ... </think>`.
2. `QwQMini.parse_reasoning` splits a completion into (reasoning, answer).
3. `preset()` raises `max_seq_len` so long traces have room.

## Why include it?

It teaches an important lesson: a "reasoning model" is usually a **training and
decoding** change on top of a normal decoder, not a secret new attention block.
