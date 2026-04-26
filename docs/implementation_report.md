# Отчёт о реализации: пайплайн loss landscape (mo_project3)

Документ описывает **что**, **где** и **как** реализовано в репозитории лабораторного проекта по визуализации и аппроксимации двумерных сечений функции потерь обученной нейросети (в духе Li et al., NeurIPS 2018). Цель отчёта — воспроизводимость: по тексту можно найти файл, функцию, формат данных и порядок запуска без чтения всего кода.

---

## 1. Назначение проекта (кратко)

- **Вход:** обученные веса \(\theta\) модели для CIFAR-10, два случайных направления \(d_1, d_2\) в пространстве параметров с **filter-wise** нормировкой направлений для свёрточных весов.
- **Поверхность:** \(f(\alpha, \beta) = L(\theta + \alpha d_1 + \beta d_2)\) на сетке \((\alpha, \beta) \in [-r, r]^2\), где \(L\) — средний cross-entropy loss на фиксированном подмножестве батчей из `DataLoader`.
- **Аппроксимации:** три модели — полная квадратика (LS + проекция на PD), диагональная квадратика (\(b=0\)), квадратика по градиенту и конечно-разностной оценке проекций Гессиана на \((d_1, d_2)\).
- **Метрики:** RMSE, \(L_\infty\), относительный RMSE к размаху \(f\); зависимость от радиуса \(r\).
- **Выход:** чекпоинты, `.npz` с поверхностью и плоскими направлениями, `.csv` с метриками, графики в `figures/`, ноутбук для повторного анализа.

---

## 2. Структура каталогов и файлов

Корень проекта: **`mo_project3/`** (рабочая директория для команд).

| Путь | Назначение |
|------|------------|
| `config.py` | Глобальные константы: сид, устройство, пути, сетка, эпохи, LR и т.д. |
| `requirements.txt` | Зависимости Python; комментарий про установку PyTorch с CUDA. |
| `run_pipeline.py` | Единый сценарий полного прогона (обучение опционально). |
| `src/model.py` | Архитектуры **Model1** (ResNet-like) и **Model2** (plain), `build_model`. |
| `src/train.py` | Обучение на CIFAR-10, чекпоинты, CLI. |
| `src/surface.py` | Расчёт сечения \(f(\alpha,\beta)\), сохранение `.npz`, CLI. |
| `src/quadratic_fit.py` | Три функции фитирования / q3 по FD. |
| `src/metrics.py` | Метрики и `metrics_vs_radius` → CSV. |
| `src/plotting.py` | Сохранение всех фигур в `data/processed/results/figures/`. |
| `src/hessian_2d.py` | Заглушки под расширения (основной q3 в `quadratic_fit.py`). |
| `src/__init__.py` | Пакет `src` (пустое описание). |
| `notebooks/main_analysis.ipynb` | Повторение ключевых шагов анализа в Jupyter. |
| `data/raw/` | Скачивание CIFAR-10 (`torchvision`, `root=config.DATA_DIR`). |
| `data/processed/checkpoints/` | Файлы `*.pth` с весами и метаданными эпохи. |
| `data/processed/results/` | `.npz`, `.csv`, подкаталог `figures/` с PNG. |
| `docs/` | Документация задания, шаблон отчёта, **этот файл**. |

---

## 3. Конфигурация: `config.py`

Файл: **`mo_project3/config.py`**.

| Константа | Значение / смысл |
|-----------|------------------|
| `SEED` | `42` — базовый сид для NumPy и PyTorch в скриптах. |
| `DEVICE` | `"cuda"` если `torch.cuda.is_available()` иначе `"cpu"` — **нигде в логике не хардкодится cuda строкой** вне этого места (скрипты берут `cfg.DEVICE` или `config.DEVICE`). |
| `DATA_DIR` | `"data/raw"` — корень для CIFAR-10. |
| `CHECKPOINT_DIR` | `"data/processed/checkpoints"`. |
| `RESULTS_DIR` | `"data/processed/results"`. |
| `GRID_STEPS` | `51` — число узлов по \(\alpha\) и по \(\beta\) (сетка \(51 \times 51\)). |
| `RADIUS_DEFAULT` | `1.0` — радиус по умолчанию для основного сечения в пайплайне. |
| `RADIUS_LIST` | `[0.1, 0.25, 0.5, 1.0, 2.0]` — радиусы для `metrics_vs_radius`. |
| `EPOCHS` | `30`. |
| `BATCH_SIZE` | `128`. |
| `LR` | `0.1` — начальный шаг SGD. |

**Замечание:** при импорте `config.py` выполняется `import torch`, поэтому в среде без PyTorch модуль не загрузится.

---

## 4. Зависимости: `requirements.txt`

Файл: **`mo_project3/requirements.txt`**.

Пакеты: `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `pandas` (с нижними границами версий).

В начале файла дан **комментарий**: для GPU обычно ставят PyTorch с официального индекса колёс (пример команды для CUDA 12.4). Это важно, потому что обычный `pip install torch` из PyPI часто даёт только CPU-сборку.

**Использование в коде:** `scipy` в текущей реализации пайплайна для обязательных вычислений не требуется (достаточно NumPy), но пакет указан в ТЗ и оставлен для совместимости / возможных расширений.

---

## 5. Общий приём: пути импорта (`sys.path`)

Скрипты в **`src/`** и **`run_pipeline.py`** в корне должны находить:

1. Корень проекта (где лежит `config.py`).
2. Каталог `src/` (чтобы писать `import model`, `import surface`, …).

**Как сделано:** вычисляется `ROOT = Path(__file__).resolve().parent.parent` для файлов в `src/` (родитель `src` = корень проекта) или `parent` для `run_pipeline.py`. В `sys.path` добавляются `str(ROOT)` и `str(ROOT / "src")`.

**Запуск:**

- Из корня `mo_project3`: `python src/train.py`, `python src/surface.py`, `python run_pipeline.py` — ожидаемый сценарий.
- Импорт `config` возможен после добавления корня в `sys.path`.

---

## 6. Модель: `src/model.py`

### 6.1. Класс `BasicBlock`

- Два слоя `Conv2d 3×3` с `BatchNorm2d` и `ReLU` после первого свёрточного блока; второй блок заканчивается BN.
- Если **`use_shortcut=True`** (ResNet-like) и нужен приведение размерности/каналов (`stride != 1` или `in_planes != planes`), shortcut — последовательность `Conv2d 1×1` (stride как у основной ветки) + BN.
- **Forward:** `out = ReLU(BN(conv2(ReLU(BN(conv1(x))))))`; при shortcut: к `out` **прибавляется** `shortcut(x)`, затем `ReLU`.

### 6.2. Класс `CifarResNetLike`

- Вход: \(3 \times 32 \times 32\) (CIFAR-10).
- Слой входа: `Conv2d(3→64, k=3, s=1, p=1)`, BN, ReLU.
- Три группы **`_make_layer(planes, num_blocks=2, stride)`**:
  - `layer1`: 64 канала, первый блок с `stride=1`.
  - `layer2`: 128 каналов, **первый** блок стадии с `stride=2` (даунсэмплинг).
  - `layer3`: 256 каналов, первый блок с `stride=2`.
- В каждой группе ровно **2** `BasicBlock` (итого 6 блоков).
- Завершение: `adaptive_avg_pool2d(..., 1)`, выравнивание в вектор, `Linear(256 → 10)`.

### 6.3. Model1 и Model2

| Функция | Параметр `use_shortcut` | Смысл |
|---------|-------------------------|--------|
| `Model1()` | `True` | ResNet-like: остаточные связи включены. |
| `Model2()` | `False` | Plain: **нет суммирования** с `shortcut(x)`; ветка только через два conv (shortcut-последовательность в блоке остаётся пустой, т.к. условие `use_shortcut and (...)` ложно). |

Топология числа слоёв и изменения разрешения совпадают; отличается только наличие skip-connection в сумме.

### 6.4. `build_model(name: str)`

- Принимает `"model1"` или `"model2"` (регистронезависимо, с `strip()`).
- Иначе `ValueError`.

---

## 7. Обучение: `src/train.py`

### 7.1. Сиды

`_set_seeds()`: `torch.manual_seed`, `np.random.seed`, при CUDA — `torch.cuda.manual_seed_all`.

Вызывается в `main()` перед обучением.

### 7.2. Данные

`get_train_val_loaders()`:

- **Train:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `ToTensor`, `Normalize(mean, std)` с константами CIFAR-10 из ТЗ.
- **Val:** только `ToTensor` + тот же `Normalize`.
- `datasets.CIFAR10(..., root=config.DATA_DIR, download=True)`.
- `DataLoader`: `batch_size=config.BATCH_SIZE`, `num_workers=2`, `pin_memory` если CUDA.

### 7.3. Цикл обучения: `train_model(model, model_name, cfg)`

- Модель переносится на `torch.device(cfg.DEVICE)`.
- **Loss:** `CrossEntropyLoss`.
- **Оптимизатор:** `SGD(lr=cfg.LR, momentum=0.9, weight_decay=5e-4)`.
- **Планировщик:** `CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)`; `scheduler.step()` **после каждой эпохи** (в конце эпохи).
- **Эпоха:** `model.train()`, проход по `train_loader` с **tqdm**, усреднение `train_loss` по числу батчей.
- После эпохи: `evaluate` на всём val — доля верных (`val_accuracy`).

### 7.4. Чекпоинты

Сохраняется словарь с ключами:

- `state_dict`
- `epoch`
- `train_loss` (среднее за последнюю эпоху)
- `val_accuracy`

**Имена файлов:**

- Каждые **10** эпох и на **последней** эпохе: `{model_name}_epoch{N}.pth` (фактически при `epoch % 10 == 0` или `epoch == EPOCHS` внутри цикла; при `EPOCHS=30` это 10, 20, 30).
- В конце дополнительно: **`{model_name}_final.pth`** с теми же ключами (финальное состояние).

Директории: `_ensure_dirs()` создаёт `DATA_DIR` и `CHECKPOINT_DIR`.

### 7.5. CLI

```bash
python src/train.py --model model1
python src/train.py --model model2
```

В консоль выводится строка с финальной val accuracy.

---

## 8. Loss surface: `src/surface.py`

### 8.1. Вспомогательные функции

| Функция | Назначение |
|---------|------------|
| `get_val_loader()` | Только нормализованный CIFAR-10 test без аугментаций; тот же `BATCH_SIZE`, `num_workers`, `pin_memory`. |
| `_random_direction_like_state` | Для каждого параметра `torch.randn_like`. |
| `_filterwise_normalize_direction(theta, direction)` | Если `theta.dim() == 4` — для каждого выходного индекса \(i\) масштабировать срез `direction[i]` так, чтобы \(\|direction[i]\|_2 = \|\theta[i]\|_2\). Иначе — масштабировать **весь** тензор на \(\|\theta\|/\|d\|\) (tensor-wise для bias, BN, FC и т.д.). |
| `_normalize_directions` | Применяет нормировку к `d1` и `d2` по каждому имени в `state_dict`. |
| `_state_from_base_and_directions` | \(\theta_{new} = \theta + \alpha d_1 + \beta d_2\) покомпонентно, затем перенос на `device`. |
| `_eval_loss_batches` | В `torch.no_grad()`, `model.eval()`, **первые `max_batches` батчей** (по умолчанию 10): `CrossEntropyLoss(reduction="sum")`, возврат **среднего** loss на всех учтённых объектах (сумма / число примеров). |
| `_flatten_state_dict` | Конкатенация векторов в порядке `names` из `named_parameters()`. |

### 8.2. Ядро: `compute_surface(model, loader, device, radius, steps, seed, model_name, cfg, max_batches=10)`

**Порядок работы:**

1. Создаётся `RESULTS_DIR` при необходимости.
2. Выставляются сиды через `_set_seeds(seed)` (для воспроизводимости направлений).
3. Фиксируется порядок имён параметров `names`.
4. **`base_state`:** копии всех параметров на **CPU** (`detach().cpu().clone()`).
5. Генерируются `raw1`, `raw2`, затем **`d1`, `d2`** после `_normalize_directions` (filter-wise + tensor-wise правило выше).
6. Сохраняются **`theta_flat`, `d1_flat`, `d2_flat`** (NumPy после `.numpy()`), чтобы **тот же** план сечения использовался в `fit_hessian_quadratic` и при загрузке с диска без повторной генерации СВ.
7. Сетка: `np.linspace(-radius, radius, steps)` по \(\alpha\) и \(\beta\); `np.meshgrid(..., indexing="ij")` → `alpha_grid[i,j]`, `beta_grid[i,j]` согласованы с `f[i,j]`.
8. Цикл по **плоскому** индексу `0 .. steps*steps-1` с **tqdm**: для каждой пары \((i,j)\) загружается `state_dict` в модель, внутри цикла вызов `_eval_loss_batches` обёрнут в **`with torch.no_grad()`** (двойная защита).
9. Сохранение **`{RESULTS_DIR}/surface_{model_name}_r{radius}.npz`** с полями:
   - `alpha_grid`, `beta_grid`, `f`
   - `theta_flat`, `d1_flat`, `d2_flat`
   - `seed` (скаляр в массиве NumPy)
10. **Восстановление весов модели** в `base_state` на `device` (чтобы после длительного сканирования сетки модель снова была в точке \(\theta\)).

**Возврат:** `f`, `alpha_grid`, `beta_grid` (все `numpy.ndarray`).

### 8.3. CLI `surface.py`

- Сиды в начале: `config.SEED` + `np.random.seed`.
- Создаются `RESULTS_DIR`, `DATA_DIR`.
- Аргументы: `--model model1|model2`, `--radius` (по умолчанию `RADIUS_DEFAULT`).
- Загрузка **`{CHECKPOINT_DIR}/{model}_final.pth`**: `torch.load` с `weights_only=False` при поддержке, иначе без этого аргумента.
- Модель `.to(device)`, затем `compute_surface(...)`.

---

## 9. Аппроксимации: `src/quadratic_fit.py`

Общее: сеточные значения выравниваются в длинные векторы для регрессии; аппроксимация на всей сетке восстанавливается в форме `(steps, steps)`.

### 9.1. `fit_full_quadratic` — модель q1

**Модель:**

\[
q_1(\alpha,\beta) = c + u\alpha + v\beta + \tfrac{1}{2}(a\alpha^2 + 2b\alpha\beta + d\beta^2).
\]

**Матрица признаков** (строка на каждую точку сетки):

\[
[1,\ \alpha,\ \beta,\ \alpha^2/2,\ \alpha\beta,\ \beta^2/2].
\]

Решение: **`np.linalg.lstsq(X, y, rcond=None)`**, коэффициенты интерпретируются как `(c, u, v, a, b, d)`.

**PD-проекция:** для симметричной \(H = \begin{pmatrix} a & b \\ b & d \end{pmatrix}\) выполняется спектральное разложение `np.linalg.eigh`, собственные значения поднимаются до минимум **`1e-6`**, матрица пересобирается; обновлённые `a, b, d` используются в формуле `f_approx`.

### 9.2. `fit_diagonal_quadratic` — модель q2

Признаки: \([1, \alpha, \beta, \alpha^2/2, \beta^2/2]\) — **без** члена \(\alpha\beta\) ⇒ в квадратичной части **`b = 0`**.

После LS: **`a = max(a, 1e-6)`**, **`d = max(d, 1e-6)`** (простая гарантия положительности вторых производных по осям).

### 9.3. Вспомогательные для q3

- **`_set_model_flat(model, flat, device)`** — записывает вектор `flat` в `p.data` всех `model.parameters()` последовательно (тот же порядок, что при конкатенации градиента).
- **`_eval_loss_mean`** — как в surface: средний CE на первых `max_batches` батчах с `reduction="sum"`.

### 9.4. `fit_hessian_quadratic` — модель q3

**Вход:** `theta_flat`, `d1_flat`, `d2_flat` (обычно из `.npz`); сетки `alpha_grid`, `beta_grid`; модель и `loader`.

**Шаги:**

1. \(L_0 = L(\theta)\) при `no_grad`, модель в `eval`.
2. Включение `requires_grad_(True)` у параметров; снова `model.eval()` (стабильность BN).
3. Предзагрузка первых `max_batches` батчей в списки на `device`.
4. Скалярный loss = **средний** CE по всем примерам в этих батчах: для каждого батча `mean_CE * batch_size` суммируется, делится на общее число объектов — согласовано с `CrossEntropyLoss(reduction='mean')` по полной подвыборке.
5. **`backward()`** → конкатенация градиентов `grads`.
6. **`g1 = <grads, d1>`**, **`g2 = <grads, d2>`** (поэлементно, сумма произведений).
7. Конечные разности при **`eps=0.01`** (по умолчанию):
   - \(H_{11} \approx (L(\theta+\varepsilon d_1) - 2L_0 + L(\theta-\varepsilon d_1))/\varepsilon^2\)
   - \(H_{22} \approx (L(\theta+\varepsilon d_2) - 2L_0 + L(\theta-\varepsilon d_2))/\varepsilon^2\)
   - \(H_{12} \approx (L(\theta+\varepsilon d_1+\varepsilon d_2) - L(\theta+\varepsilon d_1) - L(\theta+\varepsilon d_2) + L_0)/\varepsilon^2\)
8. Восстановление поверхности:

\[
q_3(\alpha,\beta) = L_0 + g_1\alpha + g_2\beta + \tfrac{1}{2}(H_{11}\alpha^2 + 2H_{12}\alpha\beta + H_{22}\beta^2).
\]

9. Параметры `requires_grad` выключаются; модель возвращается на **`theta`**.

**Возврат:** словарь параметров (`L0`, `g1`, `g2`, `H11`, `H12`, `H22`, `eps`) и `f_approx` на сетке.

---

## 10. Метрики: `src/metrics.py`

### 10.1. `compute_metrics(f_true, f_approx)`

- **RMSE:** \(\sqrt{\mathrm{mean}((f_{true}-f_{approx})^2)}\).
- **L_inf:** \(\max |f_{true}-f_{approx}|\).
- **RelRMSE:** RMSE / \((\max f_{true} - \min f_{true})\); если размах \(\le 10^{-12}\), возвращается **`float("nan")`**.

### 10.2. `metrics_vs_radius(model_name, model, loader, device, cfg)`

Для каждого **`r` из `cfg.RADIUS_LIST`**:

1. Если нет файла **`surface_{model_name}_r{r}.npz`**, вызывается **`compute_surface`** с `seed=cfg.SEED`, `steps=cfg.GRID_STEPS`.
2. Загрузка `npz`.
3. Вычисляются **`f1`, `f2`, `f3`** через `fit_full_quadratic`, `fit_diagonal_quadratic`, `fit_hessian_quadratic` (для q3 передаются `theta_flat`, `d1_flat`, `d2_flat` из файла).
4. Для каждой аппроксимации добавляется строка в таблицу с полями: **`radius`**, **`model_name`**, **`approx_type`**, **`RMSE`**, **`L_inf`**, **`RelRMSE`**.

Имена `approx_type`: `full_quadratic`, `diagonal_quadratic`, `hessian_quadratic`.

**Сохранение:** `metrics_vs_radius_{model_name}.csv` в `RESULTS_DIR`.

---

## 11. Визуализация: `src/plotting.py`

Каталог для всех PNG: **`os.path.join(config.RESULTS_DIR, "figures")`** (создаётся в `_fig_dir()`).

| Функция | Что строит | Файл |
|---------|------------|------|
| `plot_surface_3d` | `plot_surface`, ось Z = loss, `cmap="viridis"`, colorbar | переданный `filename` внутри `figures/` |
| `plot_surface_contour` | `contourf`, 30 уровней, colorbar | то же |
| `plot_comparison` | Сетка 2×2: истина, q1, q2, q3; в заголовке каждого субплота **RMSE** относительно истины (для истины RMSE = 0) | то же |
| `plot_error_heatmap` | `seaborn.heatmap` от \(\|f_{true}-f_{approx}\|\), без подписей тиков сетки | то же |
| `plot_metrics_vs_radius` | Два вертикальных субплота: RMSE(r) и L_inf(r), линии по `groupby("approx_type")`, легенда | то же |

Импорт `Axes3D` сделан для регистрации 3D-проекции в matplotlib (`# noqa: F401`).

---

## 12. Полный сценарий: `run_pipeline.py`

**Путь:** корень `mo_project3/run_pipeline.py`.

**Аргументы:**

- `--model model1|model2` (обязательный).
- `--skip-train` — пропустить обучение (ожидается готовый `{model}_final.pth`).

**Последовательность:**

1. Сиды и создание `DATA_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR`, `figures`.
2. Если нет `--skip-train`: `build_model` → **`train_model`**.
3. Новый экземпляр модели **`build_model(...).to(device)`**, загрузка **`_load_checkpoint`**.
4. **`get_val_loader()`**.
5. **`compute_surface`** с `radius=RADIUS_DEFAULT`, `steps=GRID_STEPS`, `seed=SEED`.
6. Загрузка соответствующего `.npz`.
7. Три фита: `p1,f_q1`, `p2,f_q2`, `p3,f_q3`.
8. Таблица метрик для default radius (`df_default`).
9. Графики: 3D, contour, comparison, три error heatmap (q1, q2, q3).
10. **`metrics_vs_radius`** → CSV.
11. **`plot_metrics_vs_radius`**.
12. Печать в консоль словарей параметров и двух таблиц (`df_default`, `df_rad`).

**Примеры команд:**

```bash
cd mo_project3
python run_pipeline.py --model model1
python run_pipeline.py --model model1 --skip-train
```

---

## 13. Ноутбук: `notebooks/main_analysis.ipynb`

Назначение: **офлайн-анализ** уже посчитанных артефактов без обязательного повторного обучения.

Логика путей: **`ROOT = Path.cwd()`**, если в текущей директории нет `config.py`, берётся **`ROOT.parent`** (удобно, если ядро Jupyter открыто в `notebooks/`).

Далее в `sys.path` добавляются корень и `src`, импортируются `config`, модули из `src`, выставляются сиды.

**Ячейки по смыслу:**

1. Вводный markdown.
2. Импорты, пути, `MODEL_NAME`, `R = RADIUS_DEFAULT`.
3. Загрузка `surface_{MODEL_NAME}_r{R}.npz`.
4. Вызовы `plot_surface_3d` и `plot_surface_contour` (файлы с префиксом `nb_` в `figures/`).
5. Три фита + загрузка чекпоинта для `fit_hessian_quadratic` (с `try/except` для `torch.load` и `weights_only`).
6. Таблица метрик через `compute_metrics` и `pandas`; при наличии IPython — стилизованный `display`, иначе `print`.
7. `plot_comparison`.
8. Чтение `metrics_vs_radius_{MODEL_NAME}.csv`, `display`, `plot_metrics_vs_radius`.

---

## 14. Заглушки бонуса: `src/hessian_2d.py`

Содержит **две функции-заглушки** с docstring:

- `finite_difference_second(...)` — при вызове поднимает `NotImplementedError` с текстом, что для пайплайна нужно использовать `quadratic_fit.fit_hessian_quadratic`.
- `project_to_directions(...)` — аналогично не реализовано.

Это сознательное разделение: **рабочая q3** сосредоточена в **`quadratic_fit.py`**, файл `hessian_2d.py` зарезервирован под расширения (другие схемы FD, проекция полного Гессиана и т.п.).

---

## 15. Сводная таблица артефактов

| Артефакт | Путь / шаблон имени |
|----------|---------------------|
| Промежуточные чекпоинты | `data/processed/checkpoints/{model}_epoch{10,20,30}.pth` |
| Финальный чекпоинт | `data/processed/checkpoints/{model}_final.pth` |
| Поверхность | `data/processed/results/surface_{model}_r{radius}.npz` |
| Метрики по радиусам | `data/processed/results/metrics_vs_radius_{model}.csv` |
| Рисунки пайплайна | `data/processed/results/figures/*.png` (имена задаёт `run_pipeline.py`) |
| Рисунки из ноутбука | те же `figures/`, имена с префиксом `nb_` где указано в ячейках |

---

## 16. Сложность и практические замечания

- **Размер сетки:** при `GRID_STEPS=51` выполняется **2601** оценка loss; каждая оценка — до **10** полных проходов по батчам val. Полный прогон по всем радиусам из `RADIUS_LIST` умножает стоимость ещё на число отсутствующих `.npz` файлов.
- **Кеширование:** повторный запуск не пересчитывает существующие `surface_*.npz` и может опираться на уже сохранённые данные в `metrics_vs_radius`.
- **Согласованность q3:** сохранение **`d1_flat`, `d2_flat`, `theta_flat`** в `.npz` критично: иначе при перезапуске нельзя было бы гарантировать те же направления и ту же плоскость в параметрах.
- **Восстановление \(\theta\)` после `compute_surface`** и в конце **`fit_hessian_quadratic`** снижает риск ошибочного использования «сдвинутой» модели в последующих шагах пайплайна.

---

## 17. Соответствие исходному техническому заданию (чеклист)

| Требование | Где реализовано |
|------------|-----------------|
| `requirements.txt` с перечисленными пакетами + CUDA-комментарий | `requirements.txt` |
| `config.py` с перечисленными константами | `config.py` |
| Модули `src/*.py` | все перечисленные файлы |
| ResNet-like + plain, 3×2 блока, каналы 64/128/256, GAP+FC | `src/model.py` |
| Обучение CIFAR-10, аугментации, нормализация, SGD, Cosine, чекпоинты, tqdm, CLI | `src/train.py` |
| Loss landscape, filter-wise, сетка, no_grad, tqdm, сохранение, CLI | `src/surface.py` |
| Три аппроксимации (LS+PD, диагональная, Hessian FD) | `src/quadratic_fit.py` |
| Метрики и `metrics_vs_radius` + CSV | `src/metrics.py` |
| Пять типов графиков | `src/plotting.py` |
| `run_pipeline.py` с флагами и порядком шагов | `run_pipeline.py` |
| Ноутбук с этапами анализа | `notebooks/main_analysis.ipynb` |
| Сиды в скриптах | `train.py`, `surface.py` (main + внутри compute с `seed`), `run_pipeline.py` |
| `device` из конфига | везде через `cfg.DEVICE` / `config.DEVICE` |
| `os.makedirs(..., exist_ok=True)` | в модулях при сохранении результатов и в пайплайне |

---

*Документ сгенерирован для репозитория `mo_project3` и отражает состояние кода на момент составления отчёта.*
