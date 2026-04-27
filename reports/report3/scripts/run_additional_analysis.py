#!/usr/bin/env python3
"""Run additional analyses for report3 and export CSV/figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from experiments.loss_landscape_additional import (  # noqa: E402
    AnalysisConfig,
    run_additional_analyses,
)
from surface import get_val_loader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2"])
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--grid-steps", type=int, default=11)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    loader = get_val_loader()

    report_root = ROOT / "reports" / "report3"
    cfg = AnalysisConfig(
        model_name=args.model,
        radius=args.radius,
        grid_steps=args.grid_steps,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    run_additional_analyses(
        cfg=cfg,
        loader=loader,
        device=device,
        checkpoint_dir=ROOT / config.CHECKPOINT_DIR,
        figures_dir=report_root / "figures",
        tables_dir=report_root / "tables",
    )


if __name__ == "__main__":
    main()
