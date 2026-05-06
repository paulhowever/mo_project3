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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from metrics import (  # noqa: E402
    compute_metrics,
    extract_1d_profiles,
    metrics_1d_vs_radius,
    metrics_vs_radius,
    profiles_to_dataframe,
    profile_metrics_rows,
)
from model import build_model  # noqa: E402
from plotting import (  # noqa: E402
    plot_1d_metrics_vs_radius,
    plot_1d_profiles,
    plot_comparison,
    plot_error_heatmap,
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
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Чекпоинт не найден: {path}. "
            "Запустите без --skip-train или предварительно обучите модель через src/train.py."
        )
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def _try_load_checkpoint(model: torch.nn.Module, model_name: str, device: torch.device) -> bool:
    path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_final.pth")
    if not os.path.isfile(path):
        return False
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return True


def _build_synthetic_loader(batch_size: int, total_samples: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(total_samples, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (total_samples,), generator=generator)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _apply_runtime_overrides(args: argparse.Namespace) -> None:
    if args.grid_steps is not None:
        config.GRID_STEPS = int(args.grid_steps)
    if args.radius_default is not None:
        config.RADIUS_DEFAULT = float(args.radius_default)
    if args.radius_list:
        config.RADIUS_LIST = [float(x.strip()) for x in args.radius_list.split(",") if x.strip()]
    if args.device is not None:
        config.DEVICE = args.device


def main() -> None:
    _set_seeds()
    _ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["model1", "model2"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Запускать пайплайн на синтетическом DataLoader без CIFAR-10.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=256,
        help="Размер синтетической выборки для --synthetic-smoke.",
    )
    parser.add_argument("--grid-steps", type=int, default=None)
    parser.add_argument("--radius-default", type=float, default=None)
    parser.add_argument(
        "--radius-list",
        type=str,
        default="",
        help="Список радиусов через запятую, например: 0.1,0.25,0.5",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Число батчей для оценки loss в surface и Hessian-fit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Форсировать устройство: cpu, cuda, mps. По умолчанию — из config.py.",
    )
    args = parser.parse_args()
    _apply_runtime_overrides(args)
    device = torch.device(config.DEVICE)
    model_name = args.model

    if not args.skip_train:
        model = build_model(model_name)
        train_model(model, model_name, config)

    model = build_model(model_name).to(device)
    if args.synthetic_smoke:
        loaded = _try_load_checkpoint(model, model_name, device)
        if not loaded:
            print(
                "WARNING: чекпоинт не найден, synthetic-smoke продолжится с случайной инициализацией."
            )
        loader = _build_synthetic_loader(
            batch_size=config.BATCH_SIZE,
            total_samples=args.synthetic_samples,
            seed=config.SEED,
        )
    else:
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
        max_batches=args.max_batches,
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
        eps=config.EPS_HESSIAN,
        max_batches=args.max_batches,
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
    profiles_1d = extract_1d_profiles(alpha, beta, f_true, f_q1, f_q2, f_q3)
    df_1d_default_metrics = pd.DataFrame(
        profile_metrics_rows(
            profiles_1d, model_name=model_name, radius=float(config.RADIUS_DEFAULT)
        )
    )
    df_1d_default_profile = profiles_to_dataframe(
        profiles_1d, model_name=model_name, radius=float(config.RADIUS_DEFAULT)
    )
    out_1d_profile = os.path.join(
        config.RESULTS_DIR, f"profile_1d_{model_name}_r{config.RADIUS_DEFAULT}.csv"
    )
    out_1d_metrics_default = os.path.join(
        config.RESULTS_DIR, f"metrics_1d_default_{model_name}.csv"
    )
    df_1d_default_profile.to_csv(out_1d_profile, index=False)
    df_1d_default_metrics.to_csv(out_1d_metrics_default, index=False)

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
    plot_1d_profiles(
        profiles_1d,
        filename=f"profile_1d_{base}.png",
        title=f"1D профили ({model_name}, r={config.RADIUS_DEFAULT})",
    )

    df_rad = metrics_vs_radius(
        model_name, model, loader, device, config, max_batches=args.max_batches
    )
    df_rad_1d = metrics_1d_vs_radius(
        model_name, model, loader, device, config, max_batches=args.max_batches
    )
    # Удалено: однопиксельный график "RMSE и L_inf vs радиус" (задача 2).
    # plot_metrics_vs_radius(df_rad, f"metrics_vs_radius_{model_name}.png")
    plot_1d_metrics_vs_radius(df_rad_1d, f"metrics_1d_vs_radius_{model_name}.png")

    print("\n=== Параметры аппроксимаций (default radius) ===")
    print("full_quadratic:", p1)
    print("diagonal_quadratic:", p2)
    print("hessian_quadratic:", p3)

    print("\n=== Метрики при RADIUS_DEFAULT ===")
    print(df_default.to_string(index=False))

    print("\n=== Метрики vs radius (все радиусы) ===")
    print(df_rad.to_string(index=False))
    print("\n=== 1D метрики при RADIUS_DEFAULT ===")
    print(df_1d_default_metrics.to_string(index=False))
    print("\n=== 1D метрики vs radius (все радиусы) ===")
    print(df_rad_1d.to_string(index=False))


if __name__ == "__main__":
    main()
