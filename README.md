# Project 3: Loss Landscape Pipeline

Репозиторий содержит **рабочий воспроизводимый пайплайн** лабораторной работы по визуализации двумерных сечений функции потерь нейросетей (по мотивам Li et al., NeurIPS 2018), а не только каркас.

## Что уже реализовано

- Обучение моделей `model1` (residual) и `model2` (plain) на CIFAR-10.
- Расчёт 2D-поверхности `f(alpha, beta)` в случайной нормированной плоскости направлений.
- Расчёт 1D-срезов из той же плоскости: `f(alpha, 0)` и `f(0, beta)`.
- Три аппроксимации поверхности: full quadratic, diagonal quadratic, hessian quadratic (FD).
- Метрики ошибок (RMSE, L_inf, RelRMSE) и графики.
- Единый запуск через `run_pipeline.py` и отдельный notebook для анализа.

Подробная техническая документация: `docs/implementation_report.md`.

## Структура проекта

```text
project3/
├── README.md
├── CONTRIBUTING.md
├── config.py
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── surface.py
│   ├── quadratic_fit.py
│   ├── metrics.py
│   └── plotting.py
├── notebooks/
│   └── main_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── docs/
    ├── task_sources.md
    ├── report_template.md
    ├── roadmap.md
    └── implementation_report.md
```

## Быстрый старт

```bash
python3 -m pip install -r requirements.txt
```

## Режимы запуска

### 1) Полный прогон (с обучением)

```bash
python3 run_pipeline.py --model model1
```

### 2) Прогон без обучения (если уже есть checkpoint)

```bash
python3 run_pipeline.py --model model1 --skip-train
```

### 3) Быстрый smoke без CIFAR-10 (синтетический loader)

Режим полезен для проверки целостности пайплайна в окружениях без доступа к датасету.

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

## Runbook Павла перед merge

1. Проверить smoke-команду из раздела выше (`--synthetic-smoke`).
2. При доступности сети/сертификатов проверить реальный запуск на CIFAR-10:
   - `python3 run_pipeline.py --model model1 --skip-train` (при наличии checkpoint),
   - или полный `python3 run_pipeline.py --model model1`.
3. Проверить, что появились/обновились артефакты:
   - `data/processed/results/surface_*.npz`,
   - `data/processed/results/metrics_vs_radius_*.csv`,
   - `data/processed/results/metrics_1d_vs_radius_*.csv`,
   - `data/processed/results/profile_1d_*.csv`,
   - `data/processed/results/figures/*.png`.
4. Сверить документацию:
   - `docs/roadmap.md` (актуальные статусы),
   - `docs/implementation_report.md` (актуальный CLI и ограничения окружения),
   - `README.md` (команды запуска совпадают с кодом).
5. Пройти чеклист из `CONTRIBUTING.md` перед PR в `main`.

## Воспроизводимость и правила

- Не коммитить временные файлы и локальные артефакты окружения.
- Фиксировать отклонения от исходной постановки в документации.
- Для правил структуры и оформления использовать `CONTRIBUTING.md`.

## Связанные документы

- Гайд по вкладу и оформлению: `CONTRIBUTING.md`
- План этапов и статусы: `docs/roadmap.md`
- Реестр исходных материалов: `docs/task_sources.md`
- Шаблон итогового отчёта: `docs/report_template.md`
- Технический отчёт по реализации: `docs/implementation_report.md`
- Финальный merge-gate: `docs/final_gate.md`
- Checklist по добровольному ТЗ: `docs/voluntary_requirements_checklist.md`
