#!/usr/bin/env python3
"""Generate report3 figure pack from local artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
RESULT_FIG_DIR = ROOT / "data" / "processed" / "results" / "figures"
REPORT_FIG_DIR = ROOT / "reports" / "report3" / "figures"


def _ensure_dirs() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_class_distribution() -> None:
    train_set = datasets.CIFAR10(root=str(RAW_DIR), train=True, download=False)
    counts = pd.Series(train_set.targets).value_counts().sort_index()
    class_names = train_set.classes

    plt.figure(figsize=(9, 4))
    plt.bar(class_names, counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("CIFAR-10 class distribution (train)")
    plt.tight_layout()
    plt.savefig(REPORT_FIG_DIR / "eda_class_distribution.png", dpi=160)
    plt.close()


def build_sample_grid() -> None:
    train_set = datasets.CIFAR10(root=str(RAW_DIR), train=True, download=False)
    classes = train_set.classes
    chosen = {idx: [] for idx in range(len(classes))}
    for image, target in train_set:
        if len(chosen[target]) < 3:
            chosen[target].append(image)
        if all(len(v) >= 3 for v in chosen.values()):
            break

    fig, axes = plt.subplots(len(classes), 3, figsize=(7, 14))
    for cls_idx, cls_name in enumerate(classes):
        for col in range(3):
            axes[cls_idx, col].imshow(chosen[cls_idx][col])
            axes[cls_idx, col].axis("off")
            if col == 0:
                axes[cls_idx, col].set_title(cls_name, loc="left", fontsize=9)
    fig.suptitle("CIFAR-10 samples by class", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "eda_sample_grid.png", dpi=160)
    plt.close(fig)


def copy_core_landscape_figures() -> None:
    preferred = [
        "surface_contour_model1_r0.1.png",
        "comparison_model1_r0.1.png",
        "metrics_vs_radius_model1.png",
        "profile_1d_model1_r0.1.png",
    ]
    for name in preferred:
        src = RESULT_FIG_DIR / name
        if src.exists():
            shutil.copy2(src, REPORT_FIG_DIR / name)


def main() -> None:
    _ensure_dirs()
    build_class_distribution()
    build_sample_grid()
    copy_core_landscape_figures()
    print(f"Report figures are ready in: {REPORT_FIG_DIR}")


if __name__ == "__main__":
    main()
