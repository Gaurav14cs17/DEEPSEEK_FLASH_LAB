"""QwQ Mini — long chain-of-thought educational variant.

The weights are a Qwen3-style decoder. QwQ's difference in the real world is
post-training and a long-CoT decoding policy, so this Mini demonstrates the
prompt/parse format rather than inventing a new attention mechanism.
"""

from __future__ import annotations

from typing import Optional, Tuple

from qwen3_mini import Qwen3Model, THINK_CLOSE, THINK_OPEN

from .config import QwQConfig, preset


class QwQMini(Qwen3Model):
    def __init__(self, cfg: Optional[QwQConfig] = None):
        super().__init__(cfg or preset("pc"), thinking=True)

    @staticmethod
    def format_reasoning(question: str) -> str:
        return f"{THINK_OPEN}\n{question}\n{THINK_CLOSE}\n"

    @staticmethod
    def parse_reasoning(text: str) -> Tuple[str, str]:
        """Split a completion into (reasoning, final answer)."""
        if THINK_CLOSE in text:
            head, tail = text.split(THINK_CLOSE, 1)
            reasoning = head.replace(THINK_OPEN, "", 1).strip()
            return reasoning, tail.strip()
        return "", text.strip()


QwQModel = QwQMini
