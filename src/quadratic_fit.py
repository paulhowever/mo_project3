"""Квадратичные аппроксимации поверхности потерь на сетке (α, β)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _grid_points(alpha_grid: np.ndarray, beta_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = alpha_grid.reshape(-1)
    b = beta_grid.reshape(-1)
    return a, b


def _project_pd(a: float, b: float, d: float) -> tuple[float, float, float]:
    """Проекция симметричной матрицы [[a,b],[b,d]] на конус PD."""
    H = np.array([[a, b], [b, d]], dtype=np.float64)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 1e-6)
    Hp = (V * w) @ V.T
    return float(Hp[0, 0]), float(Hp[0, 1]), float(Hp[1, 1])


def fit_full_quadratic(
    alpha_grid: np.ndarray, beta_grid: np.ndarray, f_values: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """
    q1(α,β) = c + u·α + v·β + 0.5·(a·α² + 2b·αβ + d·β²).
    Признаки: [1, α, β, α²/2, αβ, β²/2]; после LS — проекция [[a,b],[b,d]] на PD.
    """
    alpha, beta = _grid_points(alpha_grid, beta_grid)
    y = f_values.reshape(-1)
    X = np.column_stack(
        [
            np.ones_like(alpha),
            alpha,
            beta,
            0.5 * alpha**2,
            alpha * beta,
            0.5 * beta**2,
        ]
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, u, v, a, b, d = (float(x) for x in coef[:6])
    a, b, d = _project_pd(a, b, d)
    f_approx = (
        c
        + u * alpha_grid
        + v * beta_grid
        + 0.5 * (a * alpha_grid**2 + 2 * b * alpha_grid * beta_grid + d * beta_grid**2)
    )
    params = {"c": c, "u": u, "v": v, "a": a, "b": b, "d": d}
    return params, f_approx.astype(np.float64)


def fit_diagonal_quadratic(
    alpha_grid: np.ndarray, beta_grid: np.ndarray, f_values: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """q2(α,β) = c + u·α + v·β + 0.5·(a·α² + d·β²), b=0; PD — положительные a,d."""
    alpha, beta = _grid_points(alpha_grid, beta_grid)
    y = f_values.reshape(-1)
    X = np.column_stack(
        [
            np.ones_like(alpha),
            alpha,
            beta,
            0.5 * alpha**2,
            0.5 * beta**2,
        ]
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, u, v, a, d = (float(x) for x in coef[:5])
    a = max(a, 1e-6)
    d = max(d, 1e-6)
    f_approx = (
        c
        + u * alpha_grid
        + v * beta_grid
        + 0.5 * (a * alpha_grid**2 + d * beta_grid**2)
    )
    params = {"c": c, "u": u, "v": v, "a": a, "b": 0.0, "d": d}
    return params, f_approx.astype(np.float64)


def _set_model_flat(model: nn.Module, flat: torch.Tensor, device: torch.device) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p.data).to(device))
        idx += n


def _eval_loss_mean(
    model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 10
) -> float:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            total += criterion(logits, targets).item()
            count += targets.numel()
    return total / max(count, 1)


def fit_hessian_quadratic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    theta_flat: np.ndarray,
    d1_flat: np.ndarray,
    d2_flat: np.ndarray,
    eps: float = 0.01,
    max_batches: int = 10,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    q3(α,β) = L(θ) + g1·α + g2·β + 0.5·(H11·α² + 2H12·αβ + H22·β²)
    с конечными разностями для проекций Гессиана.

    Параметр ``eps`` — шаг для FD-оценки; при малом ``eps`` доминирует численный шум,
    при большом — нелинейность. Рекомендуемый диапазон: 0.005–0.05.
    """
    theta = torch.tensor(theta_flat, dtype=torch.float32, device=device)
    d1 = torch.tensor(d1_flat, dtype=torch.float32, device=device)
    d2 = torch.tensor(d2_flat, dtype=torch.float32, device=device)

    model = model.to(device)
    model.eval()
    _set_model_flat(model, theta, device)
    with torch.no_grad():
        L0 = _eval_loss_mean(model, loader, device, max_batches=max_batches)

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
        loss_sum += criterion(logits, tg) * tg.numel()
        n_sum += tg.numel()
    loss_mean = loss_sum / max(n_sum, 1)
    loss_mean.backward()
    grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    for p in model.parameters():
        p.requires_grad_(False)
    model.zero_grad(set_to_none=True)

    g1 = float((grads * d1).sum().item())
    g2 = float((grads * d2).sum().item())

    def L_shift(a: float, b: float) -> float:
        flat = theta + a * d1 + b * d2
        _set_model_flat(model, flat, device)
        with torch.no_grad():
            return _eval_loss_mean(model, loader, device, max_batches=max_batches)

    inv_eps2 = 1.0 / (eps**2)
    Lp1 = L_shift(eps, 0.0)
    Lm1 = L_shift(-eps, 0.0)
    Lp2 = L_shift(0.0, eps)
    Lm2 = L_shift(0.0, -eps)
    Lpp = L_shift(eps, eps)
    H11 = (Lp1 - 2 * L0 + Lm1) * inv_eps2
    H22 = (Lp2 - 2 * L0 + Lm2) * inv_eps2
    H12 = (Lpp - Lp1 - Lp2 + L0) * inv_eps2
    H11, H12, H22 = _project_pd(float(H11), float(H12), float(H22))

    f_approx = (
        L0
        + g1 * alpha_grid
        + g2 * beta_grid
        + 0.5
        * (
            H11 * alpha_grid**2
            + 2 * H12 * alpha_grid * beta_grid
            + H22 * beta_grid**2
        )
    )
    params = {
        "L0": float(L0),
        "g1": g1,
        "g2": g2,
        "H11": float(H11),
        "H12": float(H12),
        "H22": float(H22),
        "eps": eps,
    }
    _set_model_flat(model, theta, device)
    return params, f_approx.astype(np.float64)

