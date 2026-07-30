# Восстановление контекста SoAP в Hermes Agent

Если этот компьютер сменился или Hermes "забыл" проект,
сделай следующее:

## 1. Восстановить skill

```bash
mkdir -p ~/.hermes/skills/software-development/solps-analysis-project/references/
cp .hermes-backup/skill/SKILL.md ~/.hermes/skills/software-development/solps-analysis-project/
cp .hermes-backup/skill/references/* ~/.hermes/skills/software-development/solps-analysis-project/references/
```

## 2. Восстановить memory

Скажи агенту Hermes:

> Сохрани в память: SOLPS analysis Python package (SoAP) at /home/kirill/projects/solps_analysis/. EIRENE branch done: eirene_na (fort.44 dab2+dmb2, 100% MATLAB match), eirene_tot_flux_th/r, tdena, P_neut. All 12 B2 quantities auto-use eirene_na via _resolve_na(). ni now matches MATLAB. Plot1D/Plot2D done (screen/journal/presentation presets). Extractors: OMP, IMP, ft:N, wall, target, ns. Pending: calc_additional.m lines 171-1762 (recycling, P_pfr), PlotList YAML, Multi-watch, TimeDep, target indices for structured.

## 3. Загрузить проект

```bash
pip install -e .
```

## 4. Проверить загрузку

```python
from solps_analysis.core.dataset import SolpsWatch
watch = SolpsWatch.from_directory("/path/to/watch")
print(watch)
```

## Структура проекта

```
solps_analysis/
├── core/          — GridTopology, SolpsWatch, regions, variable
├── io/            — geometry_reader, data_readers, eirene_reader и др.
├── construct/     — @quantity decorator, composition, builtin quantities
├── extract/       — profile extractors (OMP, IMP, ft:N, wall, target, ns)
├── plot/          — Plot1D, Plot2D с пресетами стилей
├── tests/         — тесты
└── notebooks/     — Jupyter notebooks
```

## Полезные команды

```bash
# Быстрая проверка
cd /home/kirill/projects/solps_analysis && python3 -c "
from solps_analysis.core.dataset import SolpsWatch
from solps_analysis.core.regions import compute_all_regions
watch = SolpsWatch.from_directory('/path/to/watch')
compute_all_regions(watch.grid)
watch.construct_all()
print('OK:', watch.list_variables()[:5])
"
```
