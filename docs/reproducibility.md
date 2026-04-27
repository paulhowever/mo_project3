# Reproducibility Protocol

Документ описывает, как воспроизвести эксперименты и получить сравнимые артефакты.

## 1. Environment

- Python: `3.11+`
- OS: macOS/Linux (GPU опционально)
- Установка:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## 2. Determinism Policy

- Базовый сид: `SEED = 42` (`config.py`).
- Для `compute_surface` используется фиксированный `seed`, поэтому направления `d1`, `d2` воспроизводимы.
- При sweep по радиусам с одинаковым `seed` используется одна и та же плоскость, что корректно для сравнения радиусов.
- Межплатформенная bit-exact воспроизводимость не гарантируется (CPU vs GPU), но качественные тренды должны совпадать.

## 3. Data Access

Ожидается CIFAR-10 в `data/raw/` через `torchvision`.

Если автоскачивание недоступно (SSL):
1. вручную положить `cifar-10-batches-py` в `data/raw/`,
2. либо создать symlink в `data/raw/cifar-10-batches-py`.

## 4. Canonical Commands

### 4.1 Smoke (без внешних зависимостей)
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

### 4.2 Fast research run
```bash
python3 run_pipeline.py --model model1 --skip-train --grid-steps 21 --max-batches 1
```

### 4.3 Full quality run
```bash
python3 run_pipeline.py --model model1 --skip-train
```

## 5. Expected Artifacts (local)

В `data/processed/results/`:
- `surface_*.npz`
- `metrics_vs_radius_*.csv`
- `metrics_1d_vs_radius_*.csv`
- `profile_1d_*.csv`
- `figures/*.png`

В `data/processed/checkpoints/`:
- `model*_final.pth`

## 6. Artifact Publishing Policy

Репозиторий хранит код и документацию.  
Тяжёлые артефакты (`data/processed/**`, dataset в `data/raw/`) остаются локальными и игнорируются `.gitignore`.

Для ревью и отчёта публикуются:
- агрегированные таблицы и интерпретация (`docs/results_summary.md`),
- контрольные хэши (SHA256) ключевых локальных файлов (при необходимости).

## 7. Runtime Budget (ориентир)

- Full run (`51x51`, `max_batches=10`) на CPU может занимать часы.
- Fast run (`21x21`, `max_batches=1`) рассчитан на быструю итерацию и обновление отчёта.

## 8. Known Failure Modes

- SSL при загрузке CIFAR-10.
- Различия значений метрик между платформами.
- Длительное время расчёта при full-grid.
