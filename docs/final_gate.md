# Final Gate перед merge в `main`

Документ фиксирует минимальный набор проверок перед финальным PR/merge.

## 1. Код и запуск

- [x] Проходит быстрый sanity-check без внешних зависимостей:
  - `MPLBACKEND=Agg python3 run_pipeline.py --model model1 --skip-train --synthetic-smoke --synthetic-samples 128 --grid-steps 5 --radius-default 0.1 --radius-list 0.1 --max-batches 1`
- [ ] Проходит быстрый research-check (локальный CIFAR-10, ускоренный режим):
  - `python3 run_pipeline.py --model model1 --skip-train --grid-steps 21 --max-batches 1`
- [ ] При доступности окружения проходит реальный запуск на CIFAR-10 (heavy-check):
  - `python3 run_pipeline.py --model model1 --skip-train` (или полный запуск без `--skip-train`).
- [ ] Нет блокирующих ошибок по состоянию ветки и merge-конфликтов с `main`.

## 2. Артефакты

- [ ] Локально в `data/processed/results/` присутствуют актуальные:
  - `surface_*.npz`
  - `metrics_vs_radius_*.csv`
  - `metrics_1d_vs_radius_*.csv`
  - `profile_1d_*.csv`
  - `figures/*.png`
- [x] В репозитории отражена сводка результатов:
  - `docs/results_summary.md`
- [ ] Чекпоинты в `data/processed/checkpoints/` согласованы с выбранным сценарием запуска.

## 3. Документация

- [x] `README.md` отражает фактические команды запуска (full + smoke).
- [x] `docs/roadmap.md` содержит актуальные статусы этапов.
- [x] `docs/implementation_report.md` соответствует текущему CLI и инженерным доработкам.
- [x] Источники и постановка остаются согласованы с `docs/task_sources.md`.
- [x] Актуален `docs/voluntary_requirements_checklist.md` (покрытие добровольного ТЗ).

## 4. Чеклист из `docs/CONTRIBUTING.md`

- [x] При необходимости обновлён `README.md` (структура/процесс).
- [x] Актуализированы `docs/roadmap.md` и `docs/report.md` (если нужно).
- [ ] В diff нет временных файлов и локальных артефактов.
- [ ] Изменения воспроизводимы без устных пояснений.

## 5. Definition of Done

Merge в `main` выполняется, когда:

1. Пройден хотя бы smoke-check, и зафиксирован статус heavy-check.
2. Документация не противоречит коду.
3. Чеклист из `docs/CONTRIBUTING.md` закрыт.
4. PR содержит понятное описание «что сделано» и «как проверить».

## Дата последнего прохода gate

Smoke-check: пройден (дата: 2026-04-28)  
Heavy-check: требует локального запуска с CIFAR-10
