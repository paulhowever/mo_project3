# Final Gate перед merge в `main`

Документ фиксирует минимальный набор проверок перед финальным PR/merge.

## 1. Код и запуск

- [ ] Проходит быстрый sanity-check без внешних зависимостей:
  - `MPLBACKEND=Agg python3 run_pipeline.py --model model1 --skip-train --synthetic-smoke --synthetic-samples 128 --grid-steps 5 --radius-default 0.1 --radius-list 0.1 --max-batches 1`
- [ ] При доступности окружения проходит реальный запуск на CIFAR-10 (heavy-check):
  - `python3 run_pipeline.py --model model1 --skip-train` (или полный запуск без `--skip-train`).
- [ ] Нет блокирующих ошибок по состоянию ветки и merge-конфликтов с `main`.

## 2. Артефакты

- [ ] В `data/processed/results/` присутствуют актуальные:
  - `surface_*.npz`
  - `metrics_vs_radius_*.csv`
  - `metrics_1d_vs_radius_*.csv`
  - `profile_1d_*.csv`
  - `figures/*.png`
- [ ] Чекпоинты в `data/processed/checkpoints/` согласованы с выбранным сценарием запуска.

## 3. Документация

- [ ] `README.md` отражает фактические команды запуска (full + smoke).
- [ ] `docs/roadmap.md` содержит актуальные статусы этапов.
- [ ] `docs/implementation_report.md` соответствует текущему CLI и инженерным доработкам.
- [ ] Источники и постановка остаются согласованы с `docs/task_sources.md`.
- [ ] Актуален `docs/voluntary_requirements_checklist.md` (покрытие добровольного ТЗ).

## 4. Чеклист из `CONTRIBUTING.md`

- [ ] При необходимости обновлён `README.md` (структура/процесс).
- [ ] Актуализированы `docs/roadmap.md` и `docs/report_template.md` (если нужно).
- [ ] В diff нет временных файлов и локальных артефактов.
- [ ] Изменения воспроизводимы без устных пояснений.

## 5. Definition of Done

Merge в `main` выполняется, когда:

1. Пройден хотя бы smoke-check, и зафиксирован статус heavy-check.
2. Документация не противоречит коду.
3. Чеклист из `CONTRIBUTING.md` закрыт.
4. PR содержит понятное описание «что сделано» и «как проверить».
