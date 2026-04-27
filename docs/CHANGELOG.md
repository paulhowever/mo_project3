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
- `docs/repo_audit.md` с формальным аудитом репозитория.
- `docs/CONTRIBUTING.md` и перенос markdown-процесса в папку `docs/`.

### Changed
- `src/train.py`: добавлен fallback `num_workers=0` на macOS и подробный лог по эпохам.
- `src/surface.py`: добавлен fallback `num_workers=0` на macOS и явная документация поведения `seed`/плоскости.
- `src/quadratic_fit.py`: добавлена PD-проекция для Hessian-квадратики (`q3`).
- `docs/report_template.md`: шаблон заменён на заполненную отчётную версию.
- `docs/roadmap.md`: актуализирован статус этапа экспериментов.
- `README.md`: обновлено научное оформление, структура ссылок и описание результатов.
- `.gitignore`: добавлен `data/raw/cifar-10-batches-py` для защиты от случайного коммита датасета.

### Notes
- Политика артефактов: тяжёлые бинарные файлы (`.pth`, `.npz`, массовые `.png`) остаются локальными и не публикуются в Git.
