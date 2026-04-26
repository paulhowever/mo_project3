"""Метрики качества аппроксимаций поверхности потерь."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from quadratic_fit import (  # noqa: E402
    fit_diagonal_quadratic,
    fit_full_quadratic,
    fit_hessian_quadratic,
)
from surface import compute_surface  # noqa: E402


def compute_metrics(f_true: np.ndarray, f_approx: np.ndarray) -> dict[str, float]:
    """RMSE, L_inf, относительный RMSE к размаху f_true."""
    diff = f_true - f_approx
    rmse = float(np.sqrt(np.mean(diff**2)))
    linf = float(np.max(np.abs(diff)))
    span = float(np.max(f_true) - np.min(f_true))
    rel = rmse / span if span > 1e-12 else float("nan")
    return {"RMSE": rmse, "L_inf": linf, "RelRMSE": rel}


def extract_1d_profiles(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    f_true: np.ndarray,
    f_q1: np.ndarray,
    f_q2: np.ndarray,
    f_q3: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """
    1D срезы из 2D-плоскости:
    - alpha_axis: f(alpha, 0)
    - beta_axis:  f(0, beta)
    """
    i0 = int(np.argmin(np.abs(alpha_grid[:, 0])))
    j0 = int(np.argmin(np.abs(beta_grid[0, :])))
    profiles = {
        "alpha_axis": {
            "x": alpha_grid[:, j0],
            "true": f_true[:, j0],
            "full_quadratic": f_q1[:, j0],
            "diagonal_quadratic": f_q2[:, j0],
            "hessian_quadratic": f_q3[:, j0],
        },
        "beta_axis": {
            "x": beta_grid[i0, :],
            "true": f_true[i0, :],
            "full_quadratic": f_q1[i0, :],
            "diagonal_quadratic": f_q2[i0, :],
            "hessian_quadratic": f_q3[i0, :],
        },
    }
    return profiles


def profile_metrics_rows(
    profiles: dict[str, dict[str, np.ndarray]], model_name: str, radius: float
) -> list[dict]:
    rows: list[dict] = []
    for axis_name, payload in profiles.items():
        f_true = payload["true"]
        for approx_type in ("full_quadratic", "diagonal_quadratic", "hessian_quadratic"):
            m = compute_metrics(f_true, payload[approx_type])
            rows.append(
                {
                    "radius": radius,
                    "model_name": model_name,
                    "axis": axis_name,
                    "approx_type": approx_type,
                    "RMSE": m["RMSE"],
                    "L_inf": m["L_inf"],
                    "RelRMSE": m["RelRMSE"],
                }
            )
    return rows


def profiles_to_dataframe(
    profiles: dict[str, dict[str, np.ndarray]], model_name: str, radius: float
) -> pd.DataFrame:
    rows: list[dict] = []
    for axis_name, payload in profiles.items():
        x_name = "alpha" if axis_name == "alpha_axis" else "beta"
        for idx, x_val in enumerate(payload["x"]):
            rows.append(
                {
                    "model_name": model_name,
                    "radius": radius,
                    "axis": axis_name,
                    x_name: float(x_val),
                    "f_true": float(payload["true"][idx]),
                    "f_q1": float(payload["full_quadratic"][idx]),
                    "f_q2": float(payload["diagonal_quadratic"][idx]),
                    "f_q3": float(payload["hessian_quadratic"][idx]),
                }
            )
    return pd.DataFrame(rows)


def metrics_vs_radius(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
    max_batches: int = 10,
) -> pd.DataFrame:
    """
    Для каждого радиуса: загрузить surface из .npz или пересчитать, три аппроксимации, метрики.
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []
    for r in cfg.RADIUS_LIST:
        npz_path = os.path.join(cfg.RESULTS_DIR, f"surface_{model_name}_r{r}.npz")
        if not os.path.isfile(npz_path):
            compute_surface(
                model,
                loader,
                device,
                radius=float(r),
                steps=cfg.GRID_STEPS,
                seed=cfg.SEED,
                model_name=model_name,
                cfg=cfg,
                max_batches=max_batches,
            )
        data = np.load(npz_path)
        alpha = data["alpha_grid"]
        beta = data["beta_grid"]
        f_true = data["f"]
        d1_flat = data["d1_flat"]
        d2_flat = data["d2_flat"]

        _, f1 = fit_full_quadratic(alpha, beta, f_true)
        _, f2 = fit_diagonal_quadratic(alpha, beta, f_true)
        _, f3 = fit_hessian_quadratic(
            model,
            loader,
            device,
            alpha,
            beta,
            theta_flat=data["theta_flat"],
            d1_flat=d1_flat,
            d2_flat=d2_flat,
            max_batches=max_batches,
        )

        for approx_type, f_hat in (
            ("full_quadratic", f1),
            ("diagonal_quadratic", f2),
            ("hessian_quadratic", f3),
        ):
            m = compute_metrics(f_true, f_hat)
            rows.append(
                {
                    "radius": r,
                    "model_name": model_name,
                    "approx_type": approx_type,
                    "RMSE": m["RMSE"],
                    "L_inf": m["L_inf"],
                    "RelRMSE": m["RelRMSE"],
                }
            )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.RESULTS_DIR, f"metrics_vs_radius_{model_name}.csv")
    df.to_csv(out_csv, index=False)
    return df


def metrics_1d_vs_radius(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
    max_batches: int = 10,
) -> pd.DataFrame:
    """
    Метрики 1D-срезов f(alpha,0) и f(0,beta) по радиусам.
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []
    for r in cfg.RADIUS_LIST:
        npz_path = os.path.join(cfg.RESULTS_DIR, f"surface_{model_name}_r{r}.npz")
        if not os.path.isfile(npz_path):
            compute_surface(
                model,
                loader,
                device,
                radius=float(r),
                steps=cfg.GRID_STEPS,
                seed=cfg.SEED,
                model_name=model_name,
                cfg=cfg,
                max_batches=max_batches,
            )
        data = np.load(npz_path)
        alpha = data["alpha_grid"]
        beta = data["beta_grid"]
        f_true = data["f"]
        _, f1 = fit_full_quadratic(alpha, beta, f_true)
        _, f2 = fit_diagonal_quadratic(alpha, beta, f_true)
        _, f3 = fit_hessian_quadratic(
            model,
            loader,
            device,
            alpha,
            beta,
            theta_flat=data["theta_flat"],
            d1_flat=data["d1_flat"],
            d2_flat=data["d2_flat"],
            max_batches=max_batches,
        )
        profiles = extract_1d_profiles(alpha, beta, f_true, f1, f2, f3)
        rows.extend(profile_metrics_rows(profiles, model_name=model_name, radius=float(r)))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.RESULTS_DIR, f"metrics_1d_vs_radius_{model_name}.csv")
    df.to_csv(out_csv, index=False)
    return df

