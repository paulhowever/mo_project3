# Report 3 Kit

Папка содержит готовый каркас отчёта `report3` в формате LaTeX для ЛР по loss landscape.

## Что внутри

- `main.tex` — основной файл отчёта.
- `sections/` — секции отчёта (постановка, EDA, теория, эксперименты, выводы).
- `bibliography.bib` — минимальный bib-файл со ссылками.
- `scripts/generate_report_figures.py` — генератор figure-пакета для отчёта.
- `scripts/run_additional_analysis.py` — запуск дополнительных экспериментов (direction sensitivity, error scaling, training dynamics).
- `figures/` — графики для вставки в `.tex` (создаются скриптом).
- `tables/` — CSV-таблицы для дополнительных анализов.

## Быстрый старт

1. Сгенерировать figure-пакет:

```bash
python3 reports/report3/scripts/generate_report_figures.py
python3 reports/report3/scripts/run_additional_analysis.py --model model1 --radius 0.1 --grid-steps 7 --max-batches 1
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
