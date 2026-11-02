"""DeepSeekFlashV4-Mini public exports.

Stage 1 (tokenizer, embedding, RMSNorm, RoPE) is the verified gate.
Attention/MoE/system modules exist as provisional implementations; they are
exposed for tests and experiments but are **not** claimed complete until the
checklist and delta reports in ``docs/progress.md`` are satisfied.
"""

from .attention import CausalSelfAttention, SlidingWindowAttention
from .config import V4Config, preset
from .decoder import DecoderBlock
from .embedding import TokenEmbedding
from .experts import Expert
from .lm_head import LMHead
from .mhc import MultiPassMHC
from .mla import CSA, HCA, MLA, compress_nonoverlap, compress_overlap
from .model import DeepSeekFlashV4Mini, FlashV4Mini, ToyV4Flash
from .moe import MoE
from .mtp import MTP
from .normalization import RMSNorm
from .rope import apply_rope, rope_freqs, rope_reference_rotate_half
from .router import HashGate, Router
from .shared_experts import SharedExperts

__all__ = [
    "V4Config",
    "preset",
    "DeepSeekFlashV4Mini",
    "FlashV4Mini",
    "ToyV4Flash",
    "TokenEmbedding",
    "RMSNorm",
    "apply_rope",
    "rope_freqs",
    "rope_reference_rotate_half",
    "SlidingWindowAttention",
    "CausalSelfAttention",
    "CSA",
    "HCA",
    "MLA",
    "compress_overlap",
    "compress_nonoverlap",
    "MoE",
    "Router",
    "HashGate",
    "Expert",
    "SharedExperts",
    "DecoderBlock",
    "LMHead",
    "MultiPassMHC",
    "MTP",
]
