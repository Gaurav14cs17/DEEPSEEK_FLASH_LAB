#!/usr/bin/env python3
"""Record a B0 baseline for every Qwen Mini family.

Writes results/qwen/<family>/baseline/{config,metrics,memory}.json plus a
summary table. Same protocol for every family: pc preset, batch 2, seq 16,
fp32, CPU, seed 42.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages"))

from bench.latency import time_ms  # noqa: E402
from bench.memory import collect_memory  # noqa: E402
from bench.protocol import ExperimentMeta, save_result_bundle  # noqa: E402
from qwen2_5_mini import Qwen25Model, preset as p25  # noqa: E402
from qwen3_mini import Qwen3Model, preset as p3  # noqa: E402
from qwen3_next_mini import Qwen3NextModel, preset as pn  # noqa: E402
from qwen3_vl_mini import Qwen3VLMini, preset as pvl  # noqa: E402
from qwq_mini import QwQMini, preset as pq  # noqa: E402

FAMILIES = [
    ("qwen2_5", "Qwen2.5-Mini", Qwen25Model, p25),
    ("qwen3", "Qwen3-Mini", Qwen3Model, p3),
    ("qwen3_next", "Qwen3-Next-Mini", Qwen3NextModel, pn),
    ("qwen3_vl", "Qwen3-VL-Mini", Qwen3VLMini, pvl),
    ("qwq", "QwQ-Mini", QwQMini, pq),
]

BATCH, SEQ = 2, 16


def main() -> None:
    torch.manual_seed(42)
    summary = {}
    print(f"{'family':14s} {'params':>12s} {'active':>12s} {'forward_ms':>12s}")
    print("-" * 56)
    for key, display, cls, preset in FAMILIES:
        cfg = preset("pc")
        model = cls(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))
        with torch.no_grad():
            timing = time_ms(lambda: model(tokens, trace=False), warmup=3, iters=10)
        memory = collect_memory(
            model,
            batch=BATCH,
            seq=SEQ,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
        )
        metrics = {
            "parameters": model.num_parameters(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "active_parameters": model.active_parameters(),
            "peak_gpu_memory_mb": memory.peak_gpu_memory_mb,
            "peak_cpu_memory_mb": memory.peak_cpu_memory_mb,
            "forward_ms": timing["mean_ms"],
            "p50_latency_ms": timing["p50_latency_ms"],
            "p95_latency_ms": timing["p95_latency_ms"],
            "tokens_per_second": (BATCH * SEQ) / (timing["mean_ms"] / 1000.0),
        }
        meta = ExperimentMeta(
            model=display,
            experiment="stage01",
            component="qwen_family",
            version="baseline",
            precision="fp32",
            batch_size=BATCH,
            sequence_length=SEQ,
            warmup_iters=3,
            measure_iters=10,
            seed=42,
            config_name="qwen_pc",
            notes="Qwen Mini family B0; text-only forward on CPU.",
        )
        out = save_result_bundle("qwen", key, "baseline", meta, metrics, memory.to_dict())
        summary[key] = {"model": display, "dir": str(out.relative_to(REPO)), **metrics}
        print(
            f"{key:14s} {metrics['parameters']:12,d} "
            f"{metrics['active_parameters']:12,d} {metrics['forward_ms']:12.3f}"
        )

    summary_path = REPO / "results" / "qwen" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
