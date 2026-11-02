#!/usr/bin/env python3
"""Beginner tour of deepseek_flash_lab.

Run this from anywhere:

    python learn/tour.py            # guided, one section at a time
    python learn/tour.py --all      # run every section without stopping
    python learn/tour.py -s 3       # run only section 3

The tour is intentionally small and CPU-only. It explains the shape of every
tensor and the *reason* each step exists, then points to the matching notebook
and exercise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PACKAGES = REPO / "packages"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACKAGES))

from deepseek_flash_v4_mini.model.attention import CausalSelfAttention, SlidingWindowAttention  # noqa: E402
from deepseek_flash_v4_mini.model.embedding import TokenEmbedding  # noqa: E402
from deepseek_flash_v4_mini.model.moe import MoE  # noqa: E402
from deepseek_flash_v4_mini.model.normalization import RMSNorm  # noqa: E402
from deepseek_flash_v4_mini.model.rope import apply_rope, rope_freqs  # noqa: E402
from deepseek_flash_v4_mini.tokenizer import CharTokenizer  # noqa: E402

BANNER = "=" * 68


def title(text: str) -> None:
    print(f"\n{BANNER}\n{text}\n{BANNER}")


def note(text: str) -> None:
    print(f"\n  [plain English] {text}")


def observe(text: str) -> None:
    print(f"  [notice]        {text}")


def shapes(name: str, tensor: torch.Tensor) -> None:
    print(f"  {name:<22} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def pause(auto: bool) -> None:
    if auto or not sys.stdin.isatty():
        return
    try:
        input("\n  Press Enter for the next section…")
    except EOFError:
        pass


# --------------------------------------------------------------------------
# 0 — orientation
# --------------------------------------------------------------------------
def section0(auto: bool) -> None:
    title("0. Orientation — what is this lab?")
    print(
        "  This repo builds tiny, readable versions of two model families:\n"
        "    deepseek_flash_v4_mini    -> V4-Flash ideas\n"
        "    deepseek_flash_v4_1_mini  -> V4.1-Flash ideas (CED, CSA2, FP4 KV)\n\n"
        "  The real models have hundreds of billions of parameters.\n"
        "  Here everything is ~1-3 million parameters so it runs on a laptop."
    )
    note(
        "We are not trying to match the real model's quality. We are trying to "
        "make each mechanism small enough to inspect, run, and measure."
    )
    print("\n  The four habits this lab teaches:")
    for line in (
        "    1. Read the paper idea.",
        "    2. Write the math (usually a shape + a formula).",
        "    3. Implement it in a few lines of PyTorch.",
        "    4. Measure BEFORE vs AFTER and report the delta honestly.",
    ):
        print(line)
    observe("A slower candidate is a valid result. The verdict can be REGRESSION.")
    print("\n  Next sections: tokens -> embeddings -> norm -> positions -> attention -> MoE -> measurement.")


# --------------------------------------------------------------------------
# 1 — tokenizer
# --------------------------------------------------------------------------
def section1(auto: bool) -> None:
    title("1. Tokens — turning text into integers")
    text = "the cat sat on the mat."
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    print(f"  text      : {text!r}")
    print(f"  vocab size: {tok.vocab_size}")
    print(f"  ids       : {ids}")
    print(f"  decoded   : {tok.decode(ids)!r}")
    note(
        "A tokenizer is a lookup table. The model never sees letters; it sees "
        "integer IDs. This lab uses characters, so the vocabulary is tiny. "
        "Production DeepSeek uses a large BPE vocabulary."
    )
    observe(f"Each character is one token here, so T={len(ids)}.")
    pause(auto)


# --------------------------------------------------------------------------
# 2 — embeddings
# --------------------------------------------------------------------------
def section2(auto: bool) -> None:
    title("2. Embeddings — integers to vectors")
    torch.manual_seed(0)
    tok = CharTokenizer.from_text("the cat sat on the mat.")
    emb = TokenEmbedding(tok.vocab_size, d_model=8)
    ids = torch.tensor([tok.encode("cat")])
    x = emb(ids)
    print(f"  token ids  : {ids.tolist()}")
    shapes("ids [B, T]", ids)
    shapes("embeddings [B, T, d]", x)
    print(f"  first vector: {x[0, 0].tolist()}")
    note(
        "nn.Embedding is a trainable matrix E of shape [vocab, d_model]. "
        "Forward is just a row lookup: E[id]. No multiplication happens."
    )
    print(f"\n  parameter count: {emb.weight.numel()} = vocab_size * d_model")
    observe("The same token ID always starts from the same vector, but training changes E.")
    pause(auto)


# --------------------------------------------------------------------------
# 3 — RMSNorm
# --------------------------------------------------------------------------
def section3(auto: bool) -> None:
    title("3. RMSNorm — keep activations at a stable scale")
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16) * 5.0
    norm = RMSNorm(16)
    y = norm(x)

    rms_before = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    rms_after = y.pow(2).mean(dim=-1, keepdim=True).sqrt()
    print(f"  input shape       : {tuple(x.shape)}")
    print(f"  RMS before (first): {rms_before.flatten()[0].item():.4f}")
    print(f"  RMS after  (first): {rms_after.flatten()[0].item():.4f}")
    note(
        "RMSNorm divides each vector by its root-mean-square, then multiplies "
        "by a learned gain gamma. It has no bias and no mean subtraction."
    )
    print("  formula: y = x / sqrt(mean(x^2) + eps) * gamma")
    observe("After normalization the RMS is close to 1 (the learned gamma starts at 1).")
    print(f"  learned parameters: {norm.weight.numel()} (one gamma per feature)")
    pause(auto)


# --------------------------------------------------------------------------
# 4 — RoPE
# --------------------------------------------------------------------------
def section4(auto: bool) -> None:
    title("4. RoPE — encode position by rotating vectors")
    torch.manual_seed(0)
    d_head, T = 8, 6
    x = torch.randn(1, 1, T, d_head)
    freqs = rope_freqs(d_head, T, x.device)
    y = apply_rope(x, freqs=freqs)

    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    print(f"  q shape           : {tuple(x.shape)}  -> [B, H, T, d_head]")
    print(f"  freqs shape       : {tuple(freqs.shape)} -> [T, d_head/2] complex")
    print(f"  norm before pos 0 : {norm_x[0, 0, 0].item():.6f}")
    print(f"  norm after  pos 0 : {norm_y[0, 0, 0].item():.6f}")

    # Relative-position property: dot(R_i q, R_j k) depends only on (i - j).
    # Use the SAME content at every position, then shift both positions together.
    q_base = torch.randn(1, 1, 1, d_head)
    k_base = torch.randn(1, 1, 1, d_head)
    freqs_long = rope_freqs(d_head, 8, q_base.device)

    def dot_at(i: int, j: int) -> float:
        qi = apply_rope(q_base, freqs=freqs_long[i : i + 1])
        kj = apply_rope(k_base, freqs=freqs_long[j : j + 1])
        return float((qi * kj).sum())

    print("\n  dot(rotated q at i, rotated k at j) — shift both positions together:")
    for offset in (0, 1, 2):
        values = [dot_at(i, i - offset) for i in (2, 3, 4)]
        print(
            f"    offset i-j={offset}: "
            + ", ".join(f"({i},{i - offset})={v:.4f}" for i, v in zip((2, 3, 4), values))
        )
    note(
        "RoPE rotates the query and key by an angle proportional to position. "
        "The rotation preserves vector length, and the dot product between two "
        "rotated vectors depends on their relative distance, not absolute position."
    )
    observe(
        "For a fixed offset, shifting both positions (2,3,4) gives the same dot. "
        "That is the property attention relies on."
    )
    pause(auto)


# --------------------------------------------------------------------------
# 5 — attention
# --------------------------------------------------------------------------
def section5(auto: bool) -> None:
    title("5. Attention — let tokens look at earlier tokens")
    torch.manual_seed(0)
    B, T, d, H, dh = 1, 5, 16, 2, 8
    x = torch.randn(B, T, d)

    full = CausalSelfAttention(d, H, dh, use_rope=False)
    window = SlidingWindowAttention(d, H, dh, window=2, use_rope=False)
    # Copy weights so the ONLY difference is the mask.
    window.load_state_dict(full.state_dict())
    y_full = full(x)
    y_win = window(x)
    shapes("attention input", x)
    shapes("attention output", y_full)

    note(
        "Attention computes q, k, v. Scores = q·k / sqrt(d_head). A causal mask "
        "sets future positions to -infinity, so token t can only read tokens <= t. "
        "Sliding-window attention also blocks tokens older than `window`."
    )
    print("\n  full causal mask (1 = allowed):")
    print("    " + str(torch.tril(torch.ones(T, T, dtype=torch.int)).tolist()))
    print("\n  sliding-window mask, window=2 (1 = allowed):")
    idx = torch.arange(T)
    allow = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < 2)
    print("    " + str(allow.int().tolist()))
    observe("The outputs differ because the two masks allow different past tokens.")
    print(f"  difference (mean abs): {(y_full - y_win).abs().mean().item():.4f}")
    pause(auto)


# --------------------------------------------------------------------------
# 6 — MoE
# --------------------------------------------------------------------------
def section6(auto: bool) -> None:
    title("6. Mixture of Experts — route each token to a few experts")
    torch.manual_seed(0)
    d, vocab = 16, 16
    moe = MoE(
        d_model=d,
        n_routed=8,
        n_shared=1,
        n_activated=2,
        hidden=32,
        vocab_size=vocab,
        use_hash=False,
    )
    x = torch.randn(4, 6, d)
    log: list[str] = []
    y = moe(x, log=log)
    shapes("moe input", x)
    shapes("moe output", y)
    print("  routing log:")
    for line in log:
        print("   ", line)

    topv, topi = moe.router(x.reshape(-1, d))
    print(f"\n  top-1 expert per token: {topi[:, 0].tolist()}")
    print(f"  gate weights (first 3): {[round(v, 3) for v in topv[:3, 0].tolist()]}")
    note(
        "Every token goes through the shared expert(s). A router then picks the "
        "top-k routed experts for that token. Only those experts run, so total "
        "parameters can be large while compute per token stays small."
    )
    observe("Different tokens pick different experts. A balanced histogram is healthy.")
    pause(auto)


# --------------------------------------------------------------------------
# 7 — measurement loop
# --------------------------------------------------------------------------
def section7(auto: bool) -> None:
    title("7. Measurement — BEFORE -> AFTER -> delta")
    print(
        "  The lab rule is: never call something an optimization without a\n"
        "  measured baseline using the same hardware, batch, sequence, and seed."
    )
    print("\n  Protocol:")
    for line in (
        "    B0  record baseline metrics",
        "    B1  implement one candidate change",
        "    Δ   same protocol, compare JSON",
        "    verdict  IMPROVEMENT / REGRESSION / TRADE-OFF / NEUTRAL",
        "    save the result bundle",
    ):
        print(line)

    baseline = REPO / "results" / "v4" / "rope_formulation" / "A0_complex" / "metrics.json"
    candidate = REPO / "results" / "v4" / "rope_formulation" / "A1_rotate_half" / "metrics.json"
    if baseline.exists() and candidate.exists():
        import json

        b = json.loads(baseline.read_text())
        c = json.loads(candidate.read_text())
        print("\n  stored RoPE experiment:")
        print(f"    forward_ms  B0={b['forward_ms']:.5f}  B1={c['forward_ms']:.5f}")
        change = (b["forward_ms"] - c["forward_ms"]) / b["forward_ms"] * 100
        print(f"    improvement = {change:+.1f}%  (positive means B1 is faster)")
        note(
            "Here the candidate is slower, so the honest verdict is REGRESSION. "
            "Keeping the baseline is the correct scientific outcome."
        )
    else:
        print("\n  (Run scripts/run_rope_delta_demo.py to create the result bundles.)")

    print("\n  Try it yourself:")
    print("    python scripts/run_stage01_baseline.py")
    print("    python scripts/run_rope_delta_demo.py")


SECTIONS = [section0, section1, section2, section3, section4, section5, section6, section7]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="run every section without pausing"
    )
    parser.add_argument(
        "-s",
        "--section",
        type=int,
        default=None,
        help=f"run one section (0-{len(SECTIONS) - 1})",
    )
    args = parser.parse_args()

    if args.section is not None:
        if not 0 <= args.section < len(SECTIONS):
            parser.error(f"section must be 0..{len(SECTIONS) - 1}")
        SECTIONS[args.section](True)
        return

    print("Welcome to the deepseek_flash_lab beginner tour.")
    print("Everything here is tiny and runs on CPU in a few seconds.")
    for i, fn in enumerate(SECTIONS):
        fn(args.all)
        if i < len(SECTIONS) - 1:
            pause(args.all)

    title("Tour complete")
    print("  Next steps:")
    print("    docs/LEARNING_PATH.md       guided curriculum")
    print("    learn/notebooks/            hands-on notebooks")
    print("    docs/EXERCISES.md           practice tasks")
    print("    docs/SOLUTIONS.md           worked answers")


if __name__ == "__main__":
    main()
