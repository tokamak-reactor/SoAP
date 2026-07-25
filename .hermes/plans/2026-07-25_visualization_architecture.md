# SoAP Visualization — Архитектура (v2)

## 1. Иерархия классов

```
Plot (ABC)
 ├── Plot1D       — профили вдоль линии (OMP, IMP, таргеты, flux tube)
 ├── Plot2D       — 2D цветовые карты на сетке
 ├── PlotWall     — геометрия стенки
 ├── PlotMesh     — расчётная сетка
 └── PlotMulti1D  — несколько переменных на одном графике
```

**Plot (базовый класс):**
- `__init__(watch: SolpsWatch, config: PlotConfig | None = None)`
- `render(ax=None) -> tuple[Figure, Axes]` — отрисовка
- `check_compatibility(watch)` — проверка перед отрисовкой
- После render() можно править fig/ax как обычный matplotlib

## 2. PlotConfig

```python
@dataclass
class PlotConfig:
    type: str                    # "1d" | "2d" | "wall" | "mesh"
    variable: str | list[str] | None = None
    along: str | None = None     # "omp", "imp", "target_in", "target_out", "ft_N"
    region: str = "all"
    log: bool = False
    cmap: str | None = None      # None = из пресета
    title: str | None = None     # auto from metadata
    style: str = "screen"        # "screen" | "journal" | "presentation" | "custom"
    # per-plot style overrides (пропускаются в matplotlib.rcParams)
    style_overrides: dict = field(default_factory=dict)
```

## 3. Пресеты стилей (style.py)

```python
PRESETS = {
    "screen": {
        "figure.figsize": (10, 8),
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.size": 11,
        "lines.linewidth": 1.5,
        "contour.cmap": "viridis",  # кастомный ключ для 2D
    },
    "journal": {
        "figure.figsize": (17/2.54, 11/2.54),  # 17×11 см
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "contour.cmap": "viridis",
    },
    "presentation": {
        "figure.figsize": (16, 9),
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 18,
        "lines.linewidth": 2.5,
        "contour.cmap": "viridis",
    },
}
```

## 4. PlotList (список графиков)

```python
@dataclass
class PlotList:
    name: str
    description: str
    plots: list[PlotConfig]
    style: str = "screen"

    def run(watch, tweaks=None) -> list[PlotResult]
    def save(path)
    @classmethod def load(path) -> PlotList
    @classmethod def builtin(name) -> PlotList
```

**Где лежат:**
- Встроенные: `src/solps_analysis/plot/lists/*.yaml` (под git)
- Пользовательские: `plot_lists/*.yaml` (в .gitignore)

**Дефолтные (4-5 штук):**
- `student_overview` — Te, Ti, ne, ne_2d, стенка
- `divertor_analysis` — профили на таргетах, power loads
- `advanced_physics` — дивергенции и балансы
- `quick_look` — 6 subplots на одном figure

## 5. PlotResult

```python
@dataclass
class PlotResult:
    config: PlotConfig
    fig: Figure
    ax: Axes
    # После render() пользователь может:
    #   result.ax.legend(loc="upper left")
    #   result.fig.savefig("plot.pdf")
```

## 6. Tweaks (опциональная доводка)

```python
def fix_legend(fig, ax):
    ax.legend(loc="upper left", fontsize=7)

list.run(watch, tweaks=fix_legend)
```

## 7. Использование

```python
# Ad-hoc
watch.plot("te_eV", type="2d")
watch.plot("te_eV", type="2d", style="journal")

# Конструктор
p = Plot2D(watch, config={"style": "journal", "log": True})
fig, ax = p.render()
ax.legend(loc="upper left")  # правка после render()

# PlotList одной командой
PlotList.builtin("student_overview").run(watch)
PlotList.load("plot_lists/my_list.yaml").run(watch, tweaks=fix_legend)
```

## 8. Структура файлов

```
src/solps_analysis/plot/
    __init__.py         # экспорт Plot, PlotConfig, PlotList, PlotResult
    base.py             # Plot (ABC)
    plot1d.py           # Plot1D, PlotMulti1D
    plot2d.py           # Plot2D
    plot_wall.py        # PlotWall
    plot_mesh.py        # PlotMesh
    style.py            # PRESETS, PlotStyle
    lists/              # дефолтные YAML
        student_overview.yaml
        divertor_analysis.yaml
        quick_look.yaml
```

## 9. Проверка совместимости

- location == "cell" → можно 2D и 1D
- location == "face_r" / "face_th" → только 1D
- along="omp" → требует grid.inner_midplane_cells / outer_midplane_cells
- along="target_in" → требует inner_target_cells

## 10. План реализации (очередность)

1. `style.py` — пресеты, `apply_preset()`
2. `base.py` — базовый класс Plot
3. `plot2d.py` — Plot2D (самый востребованный)
4. `plot1d.py` — Plot1D, PlotMulti1D
5. `plot_wall.py`, `plot_mesh.py`
6. `__init__.py` — метод `SolpsWatch.plot()`
7. `lists/` — YAML-файлы с дефолтными PlotList'ами
8. Загрузка/сохранение PlotList (YAML)
