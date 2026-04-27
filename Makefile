.PHONY: smoke fast-run lint help

help:
	@echo "Доступные команды:"
	@echo "  make smoke      — быстрый sanity-check без CIFAR-10"
	@echo "  make fast-run   — ускоренный прогон (grid=21, batches=1), нужен checkpoint"
	@echo "  make lint       — проверка стиля (ruff или flake8)"

smoke:
	MPLBACKEND=Agg python3 run_pipeline.py \
	  --model model1 \
	  --skip-train \
	  --synthetic-smoke \
	  --synthetic-samples 128 \
	  --grid-steps 5 \
	  --radius-default 0.1 \
	  --radius-list 0.1 \
	  --max-batches 1

fast-run:
	python3 run_pipeline.py \
	  --model model1 \
	  --skip-train \
	  --grid-steps 21 \
	  --max-batches 1

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check src/ run_pipeline.py config.py || \
	 command -v flake8 >/dev/null 2>&1 && flake8 src/ run_pipeline.py config.py --max-line-length 100 || \
	 echo "Установи ruff: pip install ruff"
