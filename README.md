# SoAP — SOLPS-ITER Analysis Package

Python-пакет для анализа моделирования SOLPS-ITER. Порт MATLAB-цепочки
`read_data_3x.m` + `calc_additional.m` + `energy_balance_Extended.m` + plot-функций.

> **Разработчикам:** как всё устроено, почему именно так и что ещё пусто —
> см. [DEVELOPMENT.md](DEVELOPMENT.md) (точка входа; архитектура, единая
> схема геометрии STR/UNSTR, статус, TODO, тестирование).

## Быстрый старт

```bash
pip install -e .            # установка пакета (editable)
jupyter notebook notebooks/quickstart.ipynb   # рабочая среда с примерами
```

Минимальный пример:

```python
from solps_analysis.core.dataset import SolpsWatch

watch = SolpsWatch.from_directory("/путь/к/watch")
watch.compute_regions()              # таргеты, midplane, сепаратриса (один раз)

# Производные величины (229 шт.) — с метаданными
te_sep = watch.construct("te_sep")   # SolpsVariable(data, meta)
print(te_sep.data, te_sep.meta.unit) # [44.39] eV

# Графики
watch.plot("te", type="1d", along="omp")
watch.plot("te", type="2d")

# Низкоуровневый экстрактор
from solps_analysis.extract import extract_profile
x, y, xl, yl = extract_profile(watch, "te", along="omp")
```

## Два уровня данных

| Уровень | Что это | Как получить |
|---|---|---|
| Сырые .dat | 1100+ переменных из output/ | `watch.get("b2nph9_te")` |
| Workspace | 178 MATLAB-переменных (как read_data_3x.m) | `build_workspace(watch)` |
| Quantity | 229 производных (порт calc_additional.m) | `watch.construct("te_sep")` |

Каждая Quantity несёт метаданные: `unit`, `description`, `location`,
`matlab_lines` (номера строк исходника MATLAB — прослеживаемость порта).
Любую величину можно дополнить: `watch.construct("E_r").meta.extra["note"] = "..."`.

## Структура

```
src/solps_analysis/
  core/        dataset (SolpsWatch), grid, operators, variable (frozen)
  io/          ридеры: geometry, .dat, b2fstati, eirene, matlab_vars (workspace)
  construct/   registry (@quantity) + builtin (basic/advanced/eirene/calc_additional/energy_balance)
  extract/     profile.py — 1D экстракторы (omp/imp/ft/target/wall/ns)
  plot/        Plot1D / Plot2D (style presets: screen/journal/presentation)
notebooks/     quickstart.ipynb — рабочая среда
```

## Поддерживаемые watch'и (проверено)

- Structured: `Globus_44644_ng_new_equ/watch_15.05.2026_11_31` (98×32, ns=9)
- Unstructured WG: `Globus-3_WG/sonya_refined_no_redef_v2/watch_02.06.2026_15_30`

Регрессия 160/160 величин на обоих типах сеток:
`python3 /tmp/test_all_calc_additional.py` (см. scripts/regression_calc_additional.py).
