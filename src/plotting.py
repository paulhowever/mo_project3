"""Визуализация loss landscape и метрик."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from metrics import compute_metrics  # noqa: E402


def _fig_dir() -> str:
    d = os.path.join(config.RESULTS_DIR, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def plot_surface_3d(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    f: np.ndarray,
    title: str,
    filename: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        alpha_grid, beta_grid, f, cmap="viridis", linewidth=0, antialiased=True
    )
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("Loss")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=12)
    path = os.path.join(_fig_dir(), filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_surface_contour(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    f: np.ndarray,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(alpha_grid, beta_grid, f, levels=30, cmap="viridis")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax)
    path = os.path.join(_fig_dir(), filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    f_true: np.ndarray,
    f_q1: np.ndarray,
    f_q2: np.ndarray,
    f_q3: np.ndarray,
    filename: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    pairs = [
        (f_true, "Исходная поверхность"),
        (f_q1, "Полная квадратика"),
        (f_q2, "Диагональная квадратика"),
        (f_q3, "Гессиан FD"),
    ]
    for ax, (vals, ttl) in zip(axes.flat, pairs):
        ax.contourf(alpha_grid, beta_grid, vals, levels=25, cmap="viridis")
        if vals is f_true:
            rmse = 0.0
        else:
            rmse = compute_metrics(f_true, vals)["RMSE"]
        ax.set_title(f"{ttl}\nRMSE = {rmse:.4e}")
        ax.set_xlabel("α")
        ax.set_ylabel("β")
    fig.suptitle("Сравнение аппроксимаций", fontsize=12)
    path = os.path.join(_fig_dir(), filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    f_true: np.ndarray,
    f_approx: np.ndarray,
    title: str,
    filename: str,
) -> None:
    err = np.abs(f_true - f_approx)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        err,
        ax=ax,
        cmap="magma",
        cbar_kws={"label": "|ошибка|"},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title)
    path = os.path.join(_fig_dir(), filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_radius(df: pd.DataFrame, filename: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for approx, grp in df.groupby("approx_type"):
        ax1.plot(grp["radius"], grp["RMSE"], marker="o", label=approx)
        ax2.plot(grp["radius"], grp["L_inf"], marker="o", label=approx)
    ax1.set_ylabel("RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("L_inf")
    ax2.set_xlabel("radius")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    path = os.path.join(_fig_dir(), filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

