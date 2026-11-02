"""Token embedding.

[FAITHFUL] Standard learned token embedding E ∈ R^{V×d}.
Paper path: input embedding before the Transformer / CED stack
(DeepSeek-V4.1-Flash Fig. 3; same role in V4-Flash).

In plain English
----------------
The model cannot read text. A tokenizer first turns each token into an integer
ID. This module owns one trainable vector per ID and looks it up:

    vector = E[token_id]

That is the whole operation — a row lookup, not a matrix multiplication.
Training changes E; inference only reads it.

Shapes
------
tokens:  [B, T] long
output:  [B, T, d_model]

Memory: O(V · d_model) parameters; activations O(B · T · d_model).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # [V, d]
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B, T] → [B, T, d]
        return self.emb(tokens)

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight
