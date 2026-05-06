#!/usr/bin/env python3
"""
Финальные артефакты проекта Loss Landscape Analysis.

Что делает:
1. Сводная таблица RMSE / L_inf / RelRMSE по обеим моделям и всем трём
   аппроксимациям при r=SUMMARY_RADIUS (по умолчанию 0.1) → results/final_summary.csv
   и LaTeX-готовая таблица в stdout.
2. Отцентрированный контур loss surface для Model1: поверхность строится
   по тем же random-направлениям (seed=42, filter-wise), но повёрнутым в
   собственный базис 2×2-проекции Гессиана — минимум попадает в (0, 0).
   → results/figures/loss_surface_centered.png
3. Большой масштаб для Model1 (r=LARGE_RADIUS, сетка LARGE_GRID_STEPS²,
   max_batches=10): contour в лог-шкале + 3D с подобранным view_init.
   → results/figures/loss_surface_large_scale.png
   → results/figures/loss_surface_3d_large.png

Все .npz и фигуры сохраняются в config.RESULTS_DIR. Скрипт идемпотентен:
если кэш уже посчитан, то заново не считает.

Запуск:
    python3 run_summary.py
    python3 run_summary.py --max-batches 10
    python3 run_summary.py --device mps   # если есть MPS-ускорение
"""

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
from hessian_directions import (  # noqa: E402
    project_hessian_2x2_fd,
    rotate_directions_to_eigenbasis,
)
from metrics import compute_metrics  # noqa: E402
from model import build_model  # noqa: E402
from plotting import (  # noqa: E402
    plot_surface_3d_view,
    plot_surface_contour,
    plot_surface_contour_log,
)
from quadratic_fit import (  # noqa: E402
    fit_diagonal_quadratic,
    fit_full_quadratic,
    fit_hessian_quadratic,
)
from surface import (  # noqa: E402
    compute_surface,
    compute_surface_along_directions,
    get_val_loader,
    make_random_filterwise_directions,
)
from train import train_model  # noqa: E402


APPROX_LABELS = {
    "full_quadratic": "q1",
    "diagonal_quadratic": "q2",
    "hessian_quadratic": "q3",
}


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


def _load_or_train(model_name: str, device: torch.device) -> torch.nn.Module:
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_final.pth")
    if not os.path.isfile(ckpt_path):
        print(f"[{model_name}] checkpoint missing → обучаем с нуля "
              f"(SGD mom=0.9 wd=5e-4 LR={config.LR} {config.EPOCHS} epoch).")
        m = build_model(model_name)
        train_model(m, model_name, config)
    model = build_model(model_name).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def _ensure_surface(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    model_name: str,
    radius: float,
    steps: int,
    max_batches: int,
):
    npz = os.path.join(config.RESULTS_DIR, f"surface_{model_name}_r{radius}.npz")
    if not os.path.isfile(npz):
        compute_surface(
            model, loader, device,
            radius=radius, steps=steps, seed=config.SEED,
            model_name=model_name, cfg=config, max_batches=max_batches,
        )
    return np.load(npz)


def summary_rows_for_model(
    model_name: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> list[dict]:
    """Вернуть три строки (q1, q2, q3) для сводной таблицы при r=SUMMARY_RADIUS."""
    radius = float(config.SUMMARY_RADIUS)
    data = _ensure_surface(
        model, loader, device, model_name,
        radius=radius, steps=config.GRID_STEPS, max_batches=max_batches,
    )
    alpha = data["alpha_grid"]
    beta = data["beta_grid"]
    f_true = data["f"]
    _, f_q1 = fit_full_quadratic(alpha, beta, f_true)
    _, f_q2 = fit_diagonal_quadratic(alpha, beta, f_true)
    _, f_q3 = fit_hessian_quadratic(
        model, loader, device, alpha, beta,
        theta_flat=data["theta_flat"],
        d1_flat=data["d1_flat"], d2_flat=data["d2_flat"],
        eps=config.EPS_HESSIAN, max_batches=max_batches,
    )
    rows: list[dict] = []
    for name, f_hat in (
        ("full_quadratic", f_q1),
        ("diagonal_quadratic", f_q2),
        ("hessian_quadratic", f_q3),
    ):
        m = compute_metrics(f_true, f_hat)
        rows.append({
            "model": model_name,
            "approx": APPROX_LABELS[name],
            "rmse": m["RMSE"],
            "linf": m["L_inf"],
            "relrmse": m["RelRMSE"],
        })
    return rows


def build_centered_surface_for_model1(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> None:
    """Centered loss surface (Hessian-eigvec axes) для Model1 при r=SUMMARY_RADIUS."""
    model_name = "model1"
    radius = float(config.SUMMARY_RADIUS)
    data = _ensure_surface(
        model, loader, device, model_name,
        radius=radius, steps=config.GRID_STEPS, max_batches=max_batches,
    )
    H = project_hessian_2x2_fd(
        model, loader, device,
        theta_flat=data["theta_flat"],
        d1_flat=data["d1_flat"],
        d2_flat=data["d2_flat"],
        eps=config.EPS_HESSIAN,
        max_batches=max_batches,
    )
    print(f"[centered] H_2x2 = \n{H}")

    d1, d2, names = make_random_filterwise_directions(model, seed=config.SEED)
    e1, e2 = rotate_directions_to_eigenbasis(d1, d2, names, H)

    out_filename = f"surface_centered_{model_name}_r{radius}.npz"
    cache = os.path.join(config.RESULTS_DIR, out_filename)
    if os.path.isfile(cache):
        cached = np.load(cache)
        alpha, beta, f = cached["alpha_grid"], cached["beta_grid"], cached["f"]
    else:
        f, alpha, beta = compute_surface_along_directions(
            model, loader, device,
            radius=radius, steps=config.GRID_STEPS,
            model_name=model_name, cfg=config,
            d1=e1, d2=e2,
            out_filename=out_filename,
            max_batches=max_batches,
        )

    plot_surface_contour(
        alpha, beta, f,
        f"Loss surface centered ({model_name}, r={radius}, Hessian-eigvec axes)",
        "loss_surface_centered.png",
    )
    i_min, j_min = np.unravel_index(int(np.argmin(f)), f.shape)
    print(f"[centered] argmin at (α={alpha[i_min, j_min]:.4f}, "
          f"β={beta[i_min, j_min]:.4f}); f_min={f.min():.4f}")


def build_large_scale_for_model1(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> None:
    """Большой масштаб r=LARGE_RADIUS, сетка LARGE_GRID_STEPS²."""
    model_name = "model1"
    radius = float(config.LARGE_RADIUS)
    npz = os.path.join(config.RESULTS_DIR, f"surface_{model_name}_r{radius}.npz")
    if not os.path.isfile(npz):
        compute_surface(
            model, loader, device,
            radius=radius, steps=config.LARGE_GRID_STEPS, seed=config.SEED,
            model_name=model_name, cfg=config, max_batches=max_batches,
        )
    data = np.load(npz)
    alpha = data["alpha_grid"]
    beta = data["beta_grid"]
    f = data["f"]

    plot_surface_contour_log(
        alpha, beta, f,
        f"Loss surface (large scale, {model_name}, r={radius})",
        "loss_surface_large_scale.png",
    )
    plot_surface_3d_view(
        alpha, beta, f,
        f"Loss surface 3D (large scale, {model_name}, r={radius})",
        "loss_surface_3d_large.png",
        elev=35.0, azim=-55.0,
    )
    print(f"[large-scale] f range: [{f.min():.3f}, {f.max():.3f}]; "
          f"local minima rough count: {_count_local_minima(f)}")


def _count_local_minima(f: np.ndarray) -> int:
    """Грубый счётчик локальных минимумов: f(i,j) строго меньше всех 8 соседей."""
    f_pad = np.pad(f, 1, mode="edge")
    mask = np.ones_like(f, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = f_pad[1 + di:1 + di + f.shape[0], 1 + dj:1 + dj + f.shape[1]]
            mask &= f < shifted
    return int(mask.sum())


def print_latex_table(df: pd.DataFrame) -> None:
    print("\n=== LaTeX-готовая таблица (вставить в \\begin{table} ... \\end{table}) ===")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"Model & Approx & RMSE & $L_{\infty}$ & RelRMSE \\")
    print(r"\midrule")
    for _, r in df.iterrows():
        print(
            f"{r['model']} & {r['approx']} & "
            f"{r['rmse']:.4e} & {r['linf']:.4e} & {r['relrmse']:.4e} "
            r"\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-batches", type=int, default=10,
                   help="Число батчей для оценки лосса (default 10).")
    p.add_argument("--device", type=str, default=None,
                   help="Форсировать устройство (cpu/cuda/mps); по умолчанию из config.")
    p.add_argument("--skip-centered", action="store_true",
                   help="Пропустить отцентрованный график (Hessian-eigvec).")
    p.add_argument("--skip-large", action="store_true",
                   help="Пропустить большой масштаб r=LARGE_RADIUS.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is not None:
        config.DEVICE = args.device
    _set_seeds()
    _ensure_dirs()
    device = torch.device(config.DEVICE)
    max_batches = int(args.max_batches)

    rows: list[dict] = []
    models: dict[str, torch.nn.Module] = {}
    for name in ("model1", "model2"):
        m = _load_or_train(name, device)
        models[name] = m
        loader = get_val_loader()
        rows.extend(summary_rows_for_model(name, m, loader, device, max_batches))

    df = pd.DataFrame(rows, columns=["model", "approx", "rmse", "linf", "relrmse"])
    out_csv = os.path.join(config.RESULTS_DIR, "final_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print("\n=== Сводная таблица (r={}) ===".format(config.SUMMARY_RADIUS))
    print(df.to_string(index=False))
    print_latex_table(df)

    loader = get_val_loader()
    if not args.skip_centered:
        build_centered_surface_for_model1(models["model1"], loader, device, max_batches)
    if not args.skip_large:
        build_large_scale_for_model1(models["model1"], loader, device, max_batches)


if __name__ == "__main__":
    main()
