# Report 3 Kit

Папка содержит готовый каркас отчёта `report3` в формате LaTeX для ЛР по loss landscape.

## Что внутри

- `main.tex` — основной файл отчёта.
- `sections/` — секции отчёта (постановка, EDA, теория, эксперименты, выводы).
- `bibliography.bib` — минимальный bib-файл со ссылками.
- `scripts/generate_report_figures.py` — генератор figure-пакета для отчёта.
- `figures/` — графики для вставки в `.tex` (создаются скриптом).

## Быстрый старт

1. Сгенерировать figure-пакет:

```bash
python3 reports/report3/scripts/generate_report_figures.py
```

2. Собрать LaTeX:

```bash
cd reports/report3
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Замечания по предметке

- Структура отчёта адаптирована под текущую ЛР (loss landscape), а не под общие шаблоны классификации.
- Блоки типа confusion matrix не добавлялись, так как они не являются центральными для этой постановки.
- В секциях прямо встроена логика `гипотеза -> проверка -> интерпретация`.
