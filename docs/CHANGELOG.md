# Changelog

Все заметные изменения проекта фиксируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
проект следует принципу [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added
- Полностью переоформленный research-README с научной постановкой, методологией и результатами.
- `docs/reproducibility.md` с протоколом воспроизводимости.
- `docs/results_summary.md` со сводкой результатов и TODO по недостающим экспериментам.
- `docs/abstract.md` с кратким научным abstract.
- `docs/CONTRIBUTING.md` и перенос markdown-процесса в папку `docs/`.
- `Makefile` с целями `smoke`, `fast-run`, `lint`.
- `.github/workflows/smoke.yml` — CI smoke на Ubuntu + CPU PyTorch.
- `config.EPS_HESSIAN` и явная передача `eps` в `fit_hessian_quadratic` из пайплайна и метрик.
- Ортогонализация Грам-Шмидта для направлений плоскости в `src/surface.py`.
- `plot_two_surfaces_side_by_side` в `src/plotting.py`.
- Флаг `--device` в `run_pipeline.py`.
- Секция «2.1 Геометрия плоскости» в `docs/reproducibility.md`.
- Расширения отчёта и сводки (sweep по радиусам, RelRMSE в 1D-таблице).

### Changed
- `src/train.py`: добавлен fallback `num_workers=0` на macOS и подробный лог по эпохам.
- `src/surface.py`: добавлен fallback `num_workers=0` на macOS, ортогонализация направлений и явная документация поведения `seed`/плоскости.
- `src/quadratic_fit.py`: добавлена PD-проекция для Hessian-квадратики (`q3`); уточнён docstring про `eps`.
- `docs/report.md` (переименован из `docs/report_template.md`): итоговый отчёт, дополнена секция 5 по sweep радиусов.
- `docs/roadmap.md`, `docs/final_gate.md`, `docs/implementation_report.md`, `docs/voluntary_requirements_checklist.md`, `README.md` — актуализация под финальную сдачу.
- `.gitignore`: явные правила для `checkpoints/`, `results/`, датасета CIFAR-10.

### Removed
- `src/hessian_2d.py` (заглушки, не использовались в пайплайне).
- `docs/repo_audit.md` (внутренний рабочий документ).

### Notes
- Политика артефактов: тяжёлые бинарные файлы (`.pth`, `.npz`, массовые `.png`) остаются локальными и не публикуются в Git.
