#!/usr/bin/env python3
"""Полный пайплайн: обучение, поверхность, аппроксимации, метрики, графики."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from metrics import compute_metrics, metrics_vs_radius  # noqa: E402
from model import build_model  # noqa: E402
from plotting import (  # noqa: E402
    plot_comparison,
    plot_error_heatmap,
    plot_metrics_vs_radius,
    plot_surface_3d,
    plot_surface_contour,
)
from quadratic_fit import (  # noqa: E402
    fit_diagonal_quadratic,
    fit_full_quadratic,
    fit_hessian_quadratic,
)
from surface import compute_surface, get_val_loader  # noqa: E402
from train import train_model  # noqa: E402


def _set_seeds() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def _ensure_dirs() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)


def _load_checkpoint(model: torch.nn.Module, model_name: str, device: torch.device) -> None:
    path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_final.pth")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def main() -> None:
    _set_seeds()
    _ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["model1", "model2"])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    device = torch.device(config.DEVICE)
    model_name = args.model

    if not args.skip_train:
        model = build_model(model_name)
        train_model(model, model_name, config)

    model = build_model(model_name).to(device)
    _load_checkpoint(model, model_name, device)
    loader = get_val_loader()

    compute_surface(
        model,
        loader,
        device,
        radius=config.RADIUS_DEFAULT,
        steps=config.GRID_STEPS,
        seed=config.SEED,
        model_name=model_name,
        cfg=config,
    )

    npz_path = os.path.join(
        config.RESULTS_DIR,
        f"surface_{model_name}_r{config.RADIUS_DEFAULT}.npz",
    )
    data = np.load(npz_path)
    alpha = data["alpha_grid"]
    beta = data["beta_grid"]
    f_true = data["f"]

    p1, f_q1 = fit_full_quadratic(alpha, beta, f_true)
    p2, f_q2 = fit_diagonal_quadratic(alpha, beta, f_true)
    p3, f_q3 = fit_hessian_quadratic(
        model,
        loader,
        device,
        alpha,
        beta,
        data["theta_flat"],
        data["d1_flat"],
        data["d2_flat"],
    )

    rows_default = []
    for name, f_hat in (
        ("full_quadratic", f_q1),
        ("diagonal_quadratic", f_q2),
        ("hessian_quadratic", f_q3),
    ):
        m = compute_metrics(f_true, f_hat)
        rows_default.append(
            {
                "radius": config.RADIUS_DEFAULT,
                "model_name": model_name,
                "approx_type": name,
                **m,
            }
        )
    df_default = pd.DataFrame(rows_default)

    base = f"{model_name}_r{config.RADIUS_DEFAULT}"
    plot_surface_3d(alpha, beta, f_true, f"Loss surface ({model_name})", f"surface_3d_{base}.png")
    plot_surface_contour(
        alpha, beta, f_true, f"Loss contour ({model_name})", f"surface_contour_{base}.png"
    )
    plot_comparison(alpha, beta, f_true, f_q1, f_q2, f_q3, f"comparison_{base}.png")
    plot_error_heatmap(
        alpha,
        beta,
        f_true,
        f_q1,
        "Абсолютная ошибка |f - q1|",
        f"error_heatmap_full_{base}.png",
    )
    plot_error_heatmap(
        alpha,
        beta,
        f_true,
        f_q2,
        "Абсолютная ошибка |f - q2|",
        f"error_heatmap_diag_{base}.png",
    )
    plot_error_heatmap(
        alpha,
        beta,
        f_true,
        f_q3,
        "Абсолютная ошибка |f - q3|",
        f"error_heatmap_hess_{base}.png",
    )

    df_rad = metrics_vs_radius(model_name, model, loader, device, config)
    plot_metrics_vs_radius(df_rad, f"metrics_vs_radius_{model_name}.png")

    print("\n=== Параметры аппроксимаций (default radius) ===")
    print("full_quadratic:", p1)
    print("diagonal_quadratic:", p2)
    print("hessian_quadratic:", p3)

    print("\n=== Метрики при RADIUS_DEFAULT ===")
    print(df_default.to_string(index=False))

    print("\n=== Метрики vs radius (все радиусы) ===")
    print(df_rad.to_string(index=False))


if __name__ == "__main__":
    main()
