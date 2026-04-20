"""
Бонус: вспомогательные операции для двумерной Гессиан-аппроксимации в плоскости (d1, d2).

Основная реализация квадратичной модели по конечным разностям находится в
``quadratic_fit.fit_hessian_quadratic``. Здесь — заглушки/утилиты для расширений.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def finite_difference_second(
    f: Callable[[float, float], float],
    alpha: float,
    beta: float,
    eps: float,
    mode: str,
) -> float:
    """
    Заглушка: оценка второй производной / смешанной по шаблону FD в точке (α,β).

    Parameters
    ----------
    f :
        Скалярная функция f(α, β).
    mode :
        Один из ``"d11"``, ``"d22"``, ``"d12"`` — какой элемент матрицы Гессиана
        аппроксимировать (логика подключается при необходимости).
    """
    raise NotImplementedError("Используйте quadratic_fit.fit_hessian_quadratic для пайплайна.")


def project_to_directions(
    hessian_full: np.ndarray, d1: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    """
    Заглушка: проекция полного Гессиана H на подпространство span{d1, d2}.

    Возвращает матрицу 2×2 с коэффициентами [ [d1^T H d1, d1^T H d2], [...] ].
    """
    raise NotImplementedError("Не реализовано в базовом пайплайне.")
