# Qwen3-VL Mini — paper → code

## Family

Qwen3-VL couples a vision tower to a Qwen3 language decoder:

```
image -> patchify -> ViT blocks -> projector -> image tokens
text  -> embedding ---------------------------> concatenate -> decoder
```

## Code map

| Idea | File |
|------|------|
| text config + vision config | `config.py` |
| tiny ViT tower | `vision.py` |
| fusion + text decoder | `model.py` |
| tests | `tests/test_model.py` |
| notebook | `experiments/core/01_qwen3_vl.ipynb` |

## Fidelity

- `[FAITHFUL]` patchify → transformer → project into the LM embedding space.
- `[SIMPLIFIED]` image tokens are simply prepended. The real model interleaves
  image placeholders in the chat template and uses a much larger ViT.
- `[SIMULATED]` no real image dataset or instruction tuning.

## What to inspect

1. `TinyViT.patch` — a convolution with `kernel_size = stride = patch_size`.
2. `Qwen3VLMini.projector` — maps vision width to language width.
3. The output length: `n_patches + n_text_tokens`.

## Try it

```python
import torch
from qwen3_vl_mini import Qwen3VLMini, preset

cfg = preset("demo")
model = Qwen3VLMini(cfg)
tokens = torch.randint(0, cfg.vocab_size, (1, 5))
images = torch.randn(1, 3, cfg.image_size, cfg.image_size)
logits, _ = model(tokens, images=images)
print(logits.shape)  # [1, 5 + n_patches, vocab]
```
