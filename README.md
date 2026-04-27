# Loss Landscape Research Pipeline (CIFAR-10)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research%20mini--project-success.svg)](#)

Мини-исследовательский проект по визуализации локального ландшафта функции потерь нейросетей на CIFAR-10.  
В репозитории реализован полный pipeline: обучение, построение 2D-сечений loss surface, 1D-профилей, три типа квадратичных аппроксимаций, метрики и визуализации.  
Проект ориентирован на академическую воспроизводимость: единая точка входа, документированные ограничения, проверяемые артефакты.

## 1. Название проекта

`Loss Landscape Research Pipeline (CIFAR-10)`

## 2. Краткое описание

Проект исследует локальную геометрию функции потерь обученной модели в окрестности параметров `theta` через двумерное сечение в случайной нормированной плоскости.  
Сравниваются три квадратичные аппроксимации поверхности (`q1`, `q2`, `q3`) по количественным метрикам (`RMSE`, `L_inf`, `RelRMSE`) и по визуальным характеристикам.

## 3. Research Motivation

В задачах оптимизации нейросетей важно не только минимальное значение loss, но и форма окрестности минимума: "плоская" или "острая", устойчивая или чувствительная к возмущениям.  
Исследование loss landscape помогает:
- интерпретировать поведение разных архитектур;
- сравнивать качество локальных аппроксимаций;
- объяснять устойчивость результатов к шуму/вариациям инициализации.

## 4. Problem Statement

Для обученной модели с параметрами `theta` рассматривается функция:

$$
f(\\alpha, \\beta) = L\\left(\\theta + \\alpha d_1 + \\beta d_2\\right),
$$

где `d1`, `d2` — случайные направления в пространстве параметров после filter-wise normalization.  
Требуется:
- вычислить `f(alpha, beta)` на сетке `[-r, r] x [-r, r]`;
- построить 1D-срезы `f(alpha, 0)` и `f(0, beta)`;
- сравнить аппроксимации `q1/q2/q3` по метрикам и трендам по радиусу `r`.

## 5. Methodology

### Модели
- `model1`: residual ResNet-like (с shortcut-связями).
- `model2`: plain-вариант той же структуры (без residual-суммирования).

### Pipeline
1. Обучение на CIFAR-10 (`src/train.py`) или загрузка чекпоинта.
2. Расчёт 2D-surface (`src/surface.py`).
3. Квадратичные аппроксимации (`src/quadratic_fit.py`):
   - `q1`: full quadratic + PD-проекция,
   - `q2`: diagonal quadratic,
   - `q3`: Hessian-based FD approximation.
4. Метрики и sweep по радиусам (`src/metrics.py`).
5. Визуализации (`src/plotting.py`).

### Важное свойство воспроизводимости
При фиксированном `SEED` направления `d1`, `d2` воспроизводимы. Для разных радиусов в одном запуске используется одна и та же плоскость сравнения.

## 6. Dataset / Input Data

- Датасет: `CIFAR-10` через `torchvision.datasets.CIFAR10`.
- train transforms: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `Normalize`.
- val transforms: `ToTensor`, `Normalize`.
- Локальное размещение: `data/raw/`.

## 7. Experiments

Покрытые сценарии:
- 2D surface + `q1/q2/q3` (базовый эксперимент),
- 1D-срезы из той же плоскости,
- sweep по радиусу `RADIUS_LIST = [0.1, 0.25, 0.5, 1.0, 2.0]`.

Быстрый режим для оперативного обновления отчётных таблиц:
- `--grid-steps 21 --max-batches 1`.

## 8. Results

Текущие зафиксированные метрики (model1, `r=0.1`) из артефактов, полученных на обучении/расчёте на `RTX 3060` с полным режимом батчей:

| Approximation | RMSE | L_inf | RelRMSE |
|---|---:|---:|---:|
| full_quadratic | 1.373e-05 | 2.814e-05 | 1.372e-02 |
| diagonal_quadratic | 1.539e-05 | 3.401e-05 | 1.538e-02 |
| hessian_quadratic | 1.194e-04 | 3.355e-04 | 1.194e-01 |

Интерпретация:
- `q1` и `q2` дают близкое и существенно лучшее приближение поверхности, чем `q3`.
- FD-Hessian аппроксимация чувствительнее к выбору `eps` и численному шуму.

Полная сводка: `docs/results_summary.md`.

## 9. Project Structure

```text
project3/
├── README.md
├── config.py
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── surface.py
│   ├── quadratic_fit.py
│   ├── metrics.py
│   ├── plotting.py
│   └── hessian_2d.py
├── notebooks/
│   └── main_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── docs/
    ├── CHANGELOG.md
    ├── CONTRIBUTING.md
    ├── abstract.md
    ├── reproducibility.md
    ├── results_summary.md
    ├── implementation_report.md
    ├── report_template.md
    ├── final_gate.md
    └── roadmap.md
```

## 10. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## 11. How to Run

### Full run (train + analysis)
```bash
python3 run_pipeline.py --model model1
```

### Analysis only (if checkpoint exists)
```bash
python3 run_pipeline.py --model model1 --skip-train
```

### Fast run (for quick research iteration)
```bash
python3 run_pipeline.py --model model1 --skip-train --grid-steps 21 --max-batches 1
```

### Smoke run without CIFAR-10
```bash
MPLBACKEND=Agg python3 run_pipeline.py \
  --model model1 \
  --skip-train \
  --synthetic-smoke \
  --synthetic-samples 128 \
  --grid-steps 5 \
  --radius-default 0.1 \
  --radius-list 0.1 \
  --max-batches 1
```

## 12. Reproducibility

Кратко:
- фиксирован `SEED=42`;
- `d1/d2` сохраняются в `.npz` вместе с `theta_flat`;
- pipeline детерминирован на уровне протокола, но численно может отличаться между CPU/GPU.

Подробный protocol: `docs/reproducibility.md`.

## 13. Key Findings

- Локальная квадратичная аппроксимация (`q1`) хорошо описывает поверхность вблизи центра.
- Упрощённая диагональная модель (`q2`) даёт близкий к `q1` уровень ошибки на малом радиусе.
- Hessian FD-модель (`q3`) заметно слабее по точности в текущей конфигурации `eps`.
- Residual-архитектура ожидаемо формирует более стабильный landscape, чем plain-вариант (для полного подтверждения по `model2` требуется отдельный полный прогон).

## 14. Limitations

- Расчёт surface дорогой по времени (особенно при `51x51` и `max_batches=10`).
- Значения зависят от платформы (CPU/GPU, версии BLAS/CUDA), хотя качественные тренды сохраняются.
- В текущей политике репозитория heavy-артефакты (`.npz`, `.pth`, large PNG sets) не коммитятся.

## 15. Future Work

- Sweep по `eps` для `q3` и анализ устойчивости.
- Несколько случайных плоскостей (несколько `seed`) с оценкой `mean±std`.
- Сравнение `train-loss surface` vs `val-loss surface`.
- Расширение сравнений на `model2` в полном режиме.

## 16. Authors / Course / Context

- Автор: Павел Тищенко.
- Контекст: проектная лабораторная работа по анализу loss landscape.
- Статья-ориентир: Li et al., *Visualizing the Loss Landscape of Neural Nets* (NeurIPS 2018).

## 17. Repository Policy

- В текущей версии проекта отдельный файл лицензии не публикуется.
- Для публичного OSS-релиза вопрос лицензирования должен быть согласован отдельно.

---

## Дополнительно

- Техническая детализация: `docs/implementation_report.md`
- Сводка результатов: `docs/results_summary.md`
- Финальный gate перед merge: `docs/final_gate.md`
- Правила contribution: `docs/CONTRIBUTING.md`
