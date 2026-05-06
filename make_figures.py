#!/usr/bin/env python3
"""
Дорисовывает оставшиеся 10 PNG для отчёта (плюс 3 уже сделанных run_summary.py
дают итого 13). Все файлы кладутся в config.RESULTS_DIR/figures.

Что строится здесь:
    1.  class_distribution.png   — баланс классов в CIFAR-10 train.
    2.  class_samples.png        — примеры по 8 картинок на класс.
    3.  approx_comparison.png    — f_true vs q1, q2, q3 (Model1, r=0.1).
    4.  profiles_1d.png          — 1D-срезы f(α,0) и f(0,β) с тремя аппроксимациями.
    5.  dirs_random.png          — гистограмма per-filter норм чисто случайного направления.
    6.  dirs_layer.png           — то же для filter-wise нормированного направления.
    7.  dirs_grad.png            — то же для gradient-aligned направления.
    8.  error_vs_radius.png      — MAE и RMSE в зависимости от радиуса (linear).
    9.  error_loglog.png         — то же в log-log с reference кривыми r^k.
    10. sharpness_dynamics.png   — λ_max(H_2x2), λ_min, trace по чекпоинтам
                                    эпох 10/20/30 (sharpness растёт при сходимости).

Запуск:
    python3 make_figures.py [--device mps]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from metrics import compute_metrics, extract_1d_profiles  # noqa: E402
from model import build_model  # noqa: E402
from plotting import _fig_dir  # noqa: E402
from quadratic_fit import (  # noqa: E402
    fit_diagonal_quadratic,
    fit_full_quadratic,
    fit_hessian_quadratic,
)
from surface import (  # noqa: E402
    _filterwise_normalize_direction,
    _random_direction_like_state,
    _set_seeds,
    compute_surface,
    get_val_loader,
)

# Имена 13 PNG, которые мы ожидаем увидеть в figures/.
EXPECTED_PNGS = [
    "loss_surface_centered.png",
    "loss_surface_large_scale.png",
    "loss_surface_3d_large.png",
    "class_distribution.png",
    "class_samples.png",
    "approx_comparison.png",
    "profiles_1d.png",
    "dirs_random.png",
    "dirs_layer.png",
    "dirs_grad.png",
    "error_vs_radius.png",
    "error_loglog.png",
    "sharpness_dynamics.png",
]

CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Числа из отчёта — чтобы не пересчитывать поверхности на 5 радиусах (~50 мин).
# Источник: ручной замер на финальной модели Model1, r ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.
ERROR_VS_RADIUS_INLINE = pd.DataFrame([
    {"radius": 0.05, "MAE": 3.485e-5, "RMSE": 4.520e-5},
    {"radius": 0.10, "MAE": 1.660e-4, "RMSE": 2.131e-4},
    {"radius": 0.20, "MAE": 8.250e-4, "RMSE": 1.039e-3},
    {"radius": 0.30, "MAE": 1.847e-3, "RMSE": 2.287e-3},
    {"radius": 0.50, "MAE": 3.184e-3, "RMSE": 4.332e-3},
])

# Sharpness и condition number из отчёта (Model1, чекпоинты эпох 10/20/30).
# sharpness ≡ λ_max(H_2x2); condition_number ≡ λ_max / λ_min.
SHARPNESS_BY_EPOCH_INLINE = pd.DataFrame([
    {"epoch": 10, "sharpness": 3.20e-2, "condition_number": 45.3},
    {"epoch": 20, "sharpness": 8.00e-3, "condition_number": 12.7},
    {"epoch": 30, "sharpness": 1.78e-3, "condition_number": 1.8},
])


# ---------- инфраструктура ----------

def _set_seeds_global() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def _load_model1(device: torch.device) -> nn.Module:
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model1_final.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} не найден. Сначала запустите run_summary.py."
        )
    model = build_model("model1").to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def _ensure_surface(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    radius: float,
    steps: int,
    max_batches: int,
):
    npz = os.path.join(config.RESULTS_DIR, f"surface_model1_r{radius}.npz")
    if not os.path.isfile(npz):
        compute_surface(
            model, loader, device,
            radius=radius, steps=steps, seed=config.SEED,
            model_name="model1", cfg=config, max_batches=max_batches,
        )
    return np.load(npz)


# ---------- 1, 2: датасет ----------

def fig_class_distribution(out_dir: str) -> str:
    train_set = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=False,
        transform=transforms.ToTensor(),
    )
    targets = np.asarray(train_set.targets)
    counts = np.bincount(targets, minlength=10)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Количество примеров (train)")
    ax.set_title("CIFAR-10 — баланс классов в train выборке")
    ax.grid(axis="y", alpha=0.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 50, str(int(c)),
                ha="center", fontsize=9)
    path = os.path.join(out_dir, "class_distribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_class_samples(out_dir: str, per_class: int = 8) -> str:
    raw_set = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=False, transform=None,
    )
    targets = np.asarray(raw_set.targets)
    rng = np.random.default_rng(config.SEED)
    indices_by_class = [
        rng.choice(np.where(targets == c)[0], size=per_class, replace=False)
        for c in range(10)
    ]
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 1.2, 10 * 1.3))
    for c in range(10):
        for k, idx in enumerate(indices_by_class[c]):
            img, _ = raw_set[int(idx)]
            ax = axes[c, k]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(CIFAR_CLASSES[c], rotation=0, ha="right",
                              va="center", fontsize=10)
    fig.suptitle("CIFAR-10 — примеры по классам", fontsize=12)
    path = os.path.join(out_dir, "class_samples.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- 3, 4: аппроксимации в главной плоскости ----------

def fig_approx_comparison_and_profiles(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_batches: int,
) -> tuple[str, str]:
    radius = float(config.SUMMARY_RADIUS)
    data = _ensure_surface(
        model, loader, device, radius=radius,
        steps=config.GRID_STEPS, max_batches=max_batches,
    )
    alpha = data["alpha_grid"]
    beta = data["beta_grid"]
    f_true = data["f"]
    _, f_q1 = fit_full_quadratic(alpha, beta, f_true)
    _, f_q2 = fit_diagonal_quadratic(alpha, beta, f_true)
    _, f_q3 = fit_hessian_quadratic(
        model, loader, device, alpha, beta,
        theta_flat=data["theta_flat"], d1_flat=data["d1_flat"],
        d2_flat=data["d2_flat"], eps=config.EPS_HESSIAN,
        max_batches=max_batches,
    )

    # 3) approx_comparison.png — 2×2 сетка контуров
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    pairs = [
        (f_true, "Исходная поверхность"),
        (f_q1, "q1: полная квадратика"),
        (f_q2, "q2: диагональная квадратика"),
        (f_q3, "q3: Гессиан (FD)"),
    ]
    vmin = min(p[0].min() for p in pairs)
    vmax = max(p[0].max() for p in pairs)
    levels = np.linspace(vmin, vmax, 25)
    for ax, (vals, ttl) in zip(axes.flat, pairs):
        cs = ax.contourf(alpha, beta, vals, levels=levels, cmap="viridis")
        rmse = compute_metrics(f_true, vals)["RMSE"] if vals is not f_true else 0.0
        ax.set_title(f"{ttl}\nRMSE = {rmse:.3e}")
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Сравнение аппроксимаций (Model1, r={radius})", fontsize=13)
    path_compare = os.path.join(out_dir, "approx_comparison.png")
    fig.tight_layout()
    fig.savefig(path_compare, dpi=150)
    plt.close(fig)

    # 4) profiles_1d.png — два 1D-среза
    profiles = extract_1d_profiles(alpha, beta, f_true, f_q1, f_q2, f_q3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, axis_name in zip(axes, ("alpha_axis", "beta_axis")):
        p = profiles[axis_name]
        x = p["x"]
        ax.plot(x, p["true"], label="f_true", linewidth=2, color="black")
        ax.plot(x, p["full_quadratic"], label="q1", linestyle="--")
        ax.plot(x, p["diagonal_quadratic"], label="q2", linestyle="--")
        ax.plot(x, p["hessian_quadratic"], label="q3", linestyle="--")
        ax.set_xlabel("α" if axis_name == "alpha_axis" else "β")
        ax.set_ylabel("Loss")
        ax.set_title("f(α, 0)" if axis_name == "alpha_axis" else "f(0, β)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"1D профили loss landscape (Model1, r={radius})", fontsize=12)
    path_profiles = os.path.join(out_dir, "profiles_1d.png")
    fig.tight_layout()
    fig.savefig(path_profiles, dpi=150)
    plt.close(fig)
    return path_compare, path_profiles


# ---------- 5, 6, 7: визуализации трёх стратегий направлений ----------

def _per_filter_norms(
    direction: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-filter ||d_i|| для conv-весов и ||d|| для прочих параметров,
    плюс соответствующие ||θ_i||. Возвращает (norms_d, norms_theta).
    """
    nd, nt = [], []
    for n, t in state.items():
        d = direction[n]
        if t.dim() == 4:
            for i in range(t.shape[0]):
                nd.append(d[i].flatten().norm().item())
                nt.append(t[i].flatten().norm().item())
        else:
            nd.append(d.flatten().norm().item())
            nt.append(t.flatten().norm().item())
    return np.array(nd), np.array(nt)


def _compute_loss_grad(
    model: nn.Module, loader: DataLoader, device: torch.device,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    """∇L(θ) на тех же max_batches батчах, в виде state-dict."""
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    model.zero_grad(set_to_none=True)
    images_all, targets_all = [], []
    for i, (im, tg) in enumerate(loader):
        if i >= max_batches:
            break
        images_all.append(im.to(device))
        targets_all.append(tg.to(device))
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    n_sum = 0
    for im, tg in zip(images_all, targets_all):
        logits = model(im)
        loss_sum = loss_sum + criterion(logits, tg) * tg.numel()
        n_sum += tg.numel()
    (loss_sum / max(n_sum, 1)).backward()
    grads: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        grads[n] = p.grad.detach().cpu().clone()
    for p in model.parameters():
        p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return grads


def _plot_dirs_histogram(
    direction: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    title: str,
    filename: str,
    out_dir: str,
) -> str:
    nd, nt = _per_filter_norms(direction, state)
    ratio = nd / np.maximum(nt, 1e-12)

    fig = plt.figure(figsize=(11, 4))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(nd.max(), nt.max()), 40)
    ax1.hist(nd, bins=bins, alpha=0.6, label="‖d_i‖", color="tab:orange")
    ax1.hist(nt, bins=bins, alpha=0.4, label="‖θ_i‖", color="tab:blue")
    ax1.set_xlabel("Per-filter L2-норма")
    ax1.set_ylabel("Кол-во фильтров")
    ax1.set_title("Распределение норм")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ratio, bins=40, color="tab:green", edgecolor="black", alpha=0.7)
    ax2.axvline(1.0, color="red", linestyle="--", label="‖d_i‖=‖θ_i‖")
    ax2.set_xlabel("‖d_i‖ / ‖θ_i‖")
    ax2.set_ylabel("Кол-во фильтров")
    ax2.set_title(f"Отношение норм; mean={ratio.mean():.3f}, std={ratio.std():.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_directions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_batches: int,
) -> tuple[str, str, str]:
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    _set_seeds(config.SEED)
    raw = _random_direction_like_state(state)

    # 5) random — без какой-либо нормировки
    p_random = _plot_dirs_histogram(
        raw, state,
        "Случайные направления (без нормировки): ‖d_i‖ ≠ ‖θ_i‖",
        "dirs_random.png", out_dir,
    )

    # 6) layer / filter-wise (Li et al. 2018) — нормировка под θ_i
    layer = {n: _filterwise_normalize_direction(state[n], raw[n]) for n in state}
    p_layer = _plot_dirs_histogram(
        layer, state,
        "Filter-wise нормированные направления: ‖d_i‖ = ‖θ_i‖",
        "dirs_layer.png", out_dir,
    )

    # 7) gradient-aligned: d ∝ −∇L(θ), нормировка filter-wise чтобы шкалы совпадали
    grads = _compute_loss_grad(model, loader, device, max_batches=max_batches)
    grad_norm = {n: _filterwise_normalize_direction(state[n], -grads[n]) for n in state}
    p_grad = _plot_dirs_histogram(
        grad_norm, state,
        "Gradient-aligned направления (−∇L(θ), filter-wise нормированные)",
        "dirs_grad.png", out_dir,
    )
    return p_random, p_layer, p_grad


# ---------- 8, 9, 10: метрики и sharpness vs radius ----------

def fig_error_vs_radius(out_dir: str) -> tuple[str, str]:
    """error_vs_radius.png (linear) + error_loglog.png — из ERROR_VS_RADIUS_INLINE."""
    df = ERROR_VS_RADIUS_INLINE.sort_values("radius").reset_index(drop=True)
    radii = df["radius"].to_numpy()

    # 8) linear axes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, df["MAE"], marker="o", linewidth=2, label="MAE")
    ax.plot(radii, df["RMSE"], marker="s", linewidth=2, label="RMSE")
    ax.set_xlabel("radius r")
    ax.set_ylabel("Ошибка аппроксимации")
    ax.set_title("Ошибка vs радиус (Model1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path_lin = os.path.join(out_dir, "error_vs_radius.png")
    fig.tight_layout()
    fig.savefig(path_lin, dpi=150)
    plt.close(fig)

    # 9) log-log + reference power laws ∝ r^2 и ∝ r^3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(radii, df["MAE"], marker="o", linewidth=2, label="MAE")
    ax.loglog(radii, df["RMSE"], marker="s", linewidth=2, label="RMSE")
    base_r = radii[0]
    base_rmse = float(df["RMSE"].iloc[0])
    for k, ls in ((2, "--"), (3, ":")):
        ax.loglog(radii, base_rmse * (radii / base_r) ** k,
                  color="grey", linestyle=ls, alpha=0.7, label=f"∝ r^{k}")
    # численная оценка степени по log-log регрессии
    slope_rmse = float(
        np.polyfit(np.log(radii), np.log(df["RMSE"].to_numpy()), 1)[0]
    )
    ax.set_xlabel("radius r (log)")
    ax.set_ylabel("Ошибка (log)")
    ax.set_title(f"Ошибка vs радиус, log-log (Model1); slope(RMSE) ≈ r^{slope_rmse:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path_log = os.path.join(out_dir, "error_loglog.png")
    fig.tight_layout()
    fig.savefig(path_log, dpi=150)
    plt.close(fig)
    return path_lin, path_log


def fig_sharpness_dynamics(out_dir: str) -> str:
    """λ_max и condition number по чекпоинтам эпох — числа из отчёта (inline)."""
    df = SHARPNESS_BY_EPOCH_INLINE.sort_values("epoch").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(df["epoch"], df["sharpness"], marker="o", linewidth=2,
             color="tab:red", label="λ_max(H_2×2)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Sharpness ≡ λ_max(H_2×2)")
    ax1.set_yscale("log")
    ax1.set_title("Sharpness по ходу обучения (log Y)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    for x, y in zip(df["epoch"], df["sharpness"]):
        ax1.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)

    ax2.plot(df["epoch"], df["condition_number"], marker="s", linewidth=2,
             color="tab:purple", label="cond(H_2×2) = λ_max/λ_min")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Condition number")
    ax2.set_title("Изотропность минимума (cond → 1 при сходимости)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    for x, y in zip(df["epoch"], df["condition_number"]):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)

    fig.suptitle(
        "Sharpness Dynamics — Model1 (числа из отчёта)", fontsize=12
    )
    df.to_csv(
        os.path.join(config.RESULTS_DIR, "sharpness_by_epoch.csv"),
        index=False,
    )
    path = os.path.join(out_dir, "sharpness_dynamics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=None,
                   help="Форсировать устройство (cpu/cuda/mps).")
    p.add_argument("--max-batches", type=int, default=10,
                   help="Число батчей для оценки лосса.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is not None:
        config.DEVICE = args.device
    _set_seeds_global()
    out_dir = _fig_dir()
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(config.DEVICE)
    model = _load_model1(device)
    loader = get_val_loader()

    print("[1/10] class_distribution.png")
    fig_class_distribution(out_dir)
    print("[2/10] class_samples.png")
    fig_class_samples(out_dir)
    print("[3/10, 4/10] approx_comparison.png + profiles_1d.png")
    fig_approx_comparison_and_profiles(model, loader, device, out_dir,
                                        max_batches=args.max_batches)
    print("[5/10, 6/10, 7/10] dirs_random / dirs_layer / dirs_grad")
    fig_directions(model, loader, device, out_dir, max_batches=args.max_batches)

    print("[8/10, 9/10] error_vs_radius.png + error_loglog.png (inline-data)")
    fig_error_vs_radius(out_dir)
    print("[10/10] sharpness_dynamics.png (inline-data)")
    fig_sharpness_dynamics(out_dir)

    print("\n=== Все 13 PNG ===")
    print(f"{'имя файла':<32} {'размер':>10}")
    print("-" * 44)
    for name in EXPECTED_PNGS:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            human = (
                f"{size/1024:.1f} KB" if size < 1024 * 1024
                else f"{size/1024/1024:.2f} MB"
            )
        else:
            human = "MISSING"
        print(f"{name:<32} {human:>10}")
    print(f"\nfigures/ → {out_dir}")


if __name__ == "__main__":
    main()
