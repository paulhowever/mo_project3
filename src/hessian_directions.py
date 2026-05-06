"""
Построение пары направлений (e1, e2), выровненных с собственными векторами
2×2-проекции Гессиана внутри уже выбранной плоскости (d1, d2).

Зачем: на необучённой плоскости (случайные filter-wise нормированные d1, d2)
эмпирический минимум поверхности может быть смещён относительно центра сетки —
из-за шума оценки лосса по подвыборке батчей и из-за того, что d1, d2 не
выровнены с осями кривизны. Поворот в собственный базис проектированной 2×2-
матрицы Гессиана делает квадратичную аппроксимацию диагональной, и для
радиусов в линейной зоне (r ~ 0.1) это притягивает реальный минимум к (0, 0).

Стоимость: 5 дополнительных оценок лосса (5 вызовов ``_eval_loss_mean``)
для конечных разностей H11, H22, H12 — на порядки дешевле полного Лагранжа.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


def project_hessian_2x2_fd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    theta_flat: np.ndarray,
    d1_flat: np.ndarray,
    d2_flat: np.ndarray,
    eps: float = 0.01,
    max_batches: int = 10,
) -> np.ndarray:
    """
    2×2 проекция Гессиана на span(d1, d2), посчитанная конечными разностями.
    H11 = (L(θ+εd1) − 2L(θ) + L(θ−εd1)) / ε²,
    H22 — аналогично, H12 = (L(θ+εd1+εd2) − L(θ+εd1) − L(θ+εd2) + L(θ)) / ε².
    Возвращает симметричную матрицу 2×2.
    """
    theta = torch.tensor(theta_flat, dtype=torch.float32, device=device)
    d1 = torch.tensor(d1_flat, dtype=torch.float32, device=device)
    d2 = torch.tensor(d2_flat, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()

    _set_model_flat(model, theta, device)
    L0 = _eval_loss_mean(model, loader, device, max_batches=max_batches)

    def L_shift(a: float, b: float) -> float:
        flat = theta + a * d1 + b * d2
        _set_model_flat(model, flat, device)
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

    _set_model_flat(model, theta, device)
    return np.array([[H11, H12], [H12, H22]], dtype=np.float64)


def rotate_directions_to_eigenbasis(
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    names: list[str],
    H_2x2: np.ndarray,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Поворот пары (d1, d2) в собственный базис H_2x2.
    Если V — ортонормированная матрица собственных векторов H_2x2,
    то новые направления — линейные комбинации (d1, d2) с коэффициентами
    из столбцов V. По модулю и по геометрии плоскости результирующая пара
    остаётся в span(d1, d2), но проекция Гессиана становится диагональной.

    Note: фильтр-wise нормировка не сохраняется ровно, но изменение масштабов
    предсказуемо (поворот в 2D подпространстве); общий «бюджет» нормы плоскости
    не меняется.
    """
    w, V = np.linalg.eigh(H_2x2)
    e1: dict[str, torch.Tensor] = {}
    e2: dict[str, torch.Tensor] = {}
    for n in names:
        e1[n] = float(V[0, 0]) * d1[n] + float(V[1, 0]) * d2[n]
        e2[n] = float(V[0, 1]) * d1[n] + float(V[1, 1]) * d2[n]
    return e1, e2
