# SoAP Visualization v2 — архитектура (черновик, 05.08.2026)

Согласовано с пользователем в сессии 05.08.2026. Цель: гибкая система
графиков, покрывающая ТРИ режима работы и ТРИ группы пользователей.

## 0. Три режима работы пакета

| Режим | Что | Число watch'ей | Тип графиков |
|-------|-----|----------------|--------------|
| 1 | Один watch, подробный анализ | 1 | Парные/групповые графики на одном поле (te+ti на OMP) |
| 2 | Сравнение watch'ей (разные конфигурации, разные токамаки, разные сетки) | до 10–12 | Сравнение всех watch одновременно; первично удобство чтения и сравниваемость |
| 3 | Time-dependent динамика (каждый watch = временная точка) | до 300 | Интерактивные (листай/закрепляй) и интегральные (динамика во времени) |

MATLAB делался только под режим 1 (частично 2) — третий не работает.
Наш дизайн закладывает все три с самого начала.

## 1. Три уровня системы (требования пользователя)

1. **Уровень графика**: конвейер `величина × регион × экстрактор × плоттер`.
   - Экстрактор извлекает величину в регионе, делает преобразования,
     возвращает плоттеру готовый контракт (x, y, подписи, заголовок).
   - Плоттер НЕ знает о величине — рисует контракт в заданном стиле.
   - Комбинаторика покрывает все plotting_functions и plotting_scripts MATLAB
     без написания функции на каждый случай.
2. **Уровень списков**: PlotList — базовые (студенты), свои (копирование/
   редактирование/с нуля), чужие (от коллег). YAML: встроенные в пакете,
   пользовательские в plot_lists/.
3. **Уровень стилей**: редактируемые пресеты — режим просмотра (наглядный)
   и режим статьи (размеры, шрифты, 300 dpi). Сделано: plot/style.py.

## 2. Ключевые архитектурные решения

### 2.1 PlotData — мультикривой контракт (критично для режима 2)

```python
@dataclass
class Curve:
    x: np.ndarray          # может быть РАЗНЫМ для разных кривых (разные сетки!)
    y: np.ndarray
    label: str

@dataclass
class PlotData:
    curves: list[Curve]
    xlabel: str; ylabel: str; title: str
    location: str          # "cell" | "face" — для проверки совместимости
    unit: str; log_ok: bool
```

- Режим 1: 1 кривая (или несколько = разные переменные на одном поле).
- Режим 2: N кривых, у каждой свой x (разные сетки!) — как MATLAB
  plot1D_simple_Nresults принимал `argument` матрицей (колонка на случай).
- Режим 3 (интегральные): x = время, y = скаляр/профиль по времени.

### 2.2 Источник данных (Source) — режим НЕ аргумент экстрактора

```python
Source
 ├── SolpsWatch          — режим 1 (текущий API не ломается)
 ├── WatchCollection     — режим 2: метки + watch'и (10–12)
 └── TimeSeries          — режим 3: WatchCollection + временная ось (300)
```

Экстрактор пишется ОДИН раз и декларирует, с какими источниками работает.
Режим выбирается тем, ЧТО передали, а не аргументом.

### 2.3 Реестр экстракторов @extractor (не монстры!)

```python
@extractor(
    name="heat_flux_target",
    description="Поток тепла на пластину дивертора",
    location="face",
    sources={"watch", "collection", "timeseries"},
    params={"target": "inner|outer|lower|upper"},   # доп. параметры экстрактора
    returns="1d_profile",                            # что возвращает
)
```

- **Примитивы + композиция**: тяжёлая логика в маленьких функциях
  (_extract_cell_profile, _extract_face_profile, _interpolate_chord,
  _aggregate_time, _integrate_target...), экстракторы — тонкие обёртки 10–40
  строк. Если разрастается — выносим кусок в примитив.
- **Уникальные экстракторы — отдельные, в реестре** (находятся поиском,
  как list_quantities()). Пример: интегральный поток тепла на пластину от
  времени = отдельный экстрактор `sources={"timeseries"}`, returns="time_series".
  НЕ доп. аргумент режима 3 (это вносит путаницу).
- Регистрация автоматическая (декоратор), как @quantity.

### 2.4 Конструктор — 4 поля + контекстные параметры

GUI/API: поля (плоттер, экстрактор, величина, регион) всегда на виду.
Доп. параметры появляются ТОЛЬКО для выбранного экстрактора (у
heat_flux_target — одно поле target). В YAML — просто ключи:

```yaml
- type: 1d
  extractor: heat_flux_target
  variable: fh_pls
  region: target_out
  target: outer
```

Совместимость фильтрует поля: плоттер → экстракторы с подходящим returns;
экстрактор → величины с подходящим location и источники с подходящим режимом.

### 2.5 Совместимость — жёсткая + предложение альтернативы

Несовместимо → ошибка с объяснением и предложением ближайшего варианта:
«face-величина fna_mdf_th не рисуется плоттером 2d; попробуйте plotter=1d
или величину te». Реализуемо: реестры с декларациями.

### 2.6 Плоттеры

- Plot1D — универсальный, рисует list[Curve]. Все режимы.
- Plot2D — режим 1; режим 2 — сетка сабплотов (аналог
  subplots_2D_Nresults_execute.m); режим 3 — анимация/слайдер.
- PlotTimeSeries — режим 3, интегральные: x=время, y=скаляр; профили —
  waterfall/heatmap (time × radius).
- PlotTimeSlider — режим 3, интерактивный: один watch на поле, ползунок
  листает (matplotlib.widgets).
- 2D: экстрактор extract_2d_b2 (поле+геометрия сетки); EIRENE-треугольники —
  ОТДЕЛЬНЫЙ extract_2d_eirene (другая сетка = другой источник, не регион).
  Регион для 2D — окно просмотра (вся сетка / дивертор верх-низ / xlim-ylim /
  маска по cv_reg), живёт в конфигурации плоттера.

### 2.7 Принцип «экстрактор = глагол, регион = существительное»

Глаголов мало (5–8), существительных много (любые). Предварительный список
экстракторов:
- extract_cell_profile — скаляр в центрах ячеек региона (omp/imp/target/ft/chord/wall)
- extract_face_profile — значения на гранях региона (потоки)
- extract_ns_profile — по зарядным состояниям
- extract_int_poloidal — полоидальный интеграл (радиальные профили)
- extract_2d_b2 / extract_2d_eirene — 2D поля
- extract_wall — профиль вдоль стенки (склейка сегментов)
- extract_heat_flux_target — поток тепла на пластину (face, интегрирование)
- extract_heat_flux_target_time — интегральный поток от времени (timeseries)
- (позже) extract_vector — 2D векторные поля (quiver)

Хорда — НЕ экстрактор, а регион: точки (R,Z) → интерполяция на них.
Референс MATLAB: feature/plot-chord ветка (interpolate_chord.m —
scatteredInterpolant linear + nearest). В Python: matplotlib.tri.
LinearTriInterpolator (без scipy).

## 3. Координаты x_coord (для сравнения, режим 2)

Традиция группы:
- экваторы (OMP/IMP): r − r_sep (расстояние от сепаратрисы)
- пластины: расстояние от X-точки по поверхности пластины
- полоидальные графики: нормированная полоидальная координата θ_norm (0→1)
- желательно: psi-нормированное (физически осмысленное; есть в геометрии)

Варианты x_coord: "r" | "r_sep" | "s_target" | "theta" | "theta_norm" | "psi_n".

Что уже есть в GridTopology (145 атрибутов, проверено 05.08.2026):
- cv_fpsi / fc_fpsi / fs_psi / vx_fpsi — psi ЕСТЬ (b2fgmtry fpsi)
- cv_r, cv_theta, cv_theta_n, cv_lbl_len — ЕСТЬ, НО на structured-сетке
  после compute_regions() = None (надо доделать structured-ветку regions!)
- wall_cells / wall_cells_len / wall_cells_vol / wall_faces — стенка есть
- cv_vol — объёмы ячеек (для интегралов)
- fc_lbl_group / fc_lbl_group_len — сегменты границы

psi_n = (psi − psi_sep)/(psi_wall − psi_sep) — производная, считать в
экстракторе/квантити.

### 3.1 Structured-координаты — по формулам Matlab_SPb (важно! проверено 05.08.2026)

В ДВУХ версиях MATLAB координаты считаются ПО-РАЗНОМУ:

**Matlab_SPb (structured, прямоугольная матрица)** — /home/kirill/projects/
Matlab_SPb/IO/read_geometry.m:
- **y2** — «расстояние от внутренней сепаратрисы»:
  yc — накопленные размеры ячеек от южной границы:
  `yc(1,j)=hy(1,j)/2;  yc(i,j)=yc(i-1,j)+(hy(i,j)+hy(i-1,j))/2`
  `y2(i,j) = yc(i,j) - yc(nsep+2,j) - hy(nsep+2,j)/2`
  Отрицательные — core/PFR, положительные — SOL.
- **x2** — полоидальная координата: накопленные hx с учётом разрезов
  (nc1..nc4, ntt — топология DND/SND), `x2 = xc - xzero`, где xzero:
  CORE/внутр. SOL → xc(i,nout+2) (от ВНЕШНЕГО ЭКВАТОРА/OMP); bottom PFR →
  0 (от внутренней нижней пластины); top PFR → 0 (от внешней верхней);
  есть и x2_prll (с весом B/Bx — параллельная длина).
- На пластинах используется ТА ЖЕ y2: из-за прямоугольности матрицы y2
  вдоль строки пластины = расстояние от strike point по поверхности
  (не может быть искажённых ячеек/коротких трубок). Te_mout.m →
  R_plot(y2,...), ne_tar.m → R_plot_targets(y2,...). Подпись: «Distance
  from separatrix, m».
- **В matlab_wg** y2/x2 соответствуют **gmtry.cvR / gmtry.cvTheta**
  (и аналоги для фейсов), а cvX/cvY — это R и Z в цилиндрической системе
  (центр — ось тора), как на чертежах сечений токамаков.

**Следствия для SoAP:**
- cv_x/cv_y в нашем grid = R, Z (средние crx/cry) — это НЕ cv_r!
- Текущий fallback в extract/profile.py: `sqrt(cv_x²+cv_y²)` даёт R
  (мажорный радиус), а НЕ расстояние от сепаратрисы. Традиция группы —
  r−r_sep (y2) для экваторов → дефолтная ось для omp/imp должна быть
  y2-координата (r_sep), а не R.
- Для structured: cv_r (y2) и cv_theta (x2) считать НАКОПЛЕНИЕМ размеров
  ячеек hy/hx от сепаратрисы/разрезов (просто, по матрице), НЕ ходьбой
  по флакс-трубкам как в unstructured. Нужны hy/hx (размеры ячеек) —
  взять из fcHc/углов crx/cry, и топологические индексы nsep, nout,
  nc1..nc4, ntt (у нас есть leftcut/rightcut/topcut/bottomcut).
- cv_theta_n (0→1): для structured — нормировка по столбцам матрицы.

**Что уже есть в SoAP для structured (проверено 05.08.2026):**
- geometry_reader читает из b2fgmtry: hx, hy (→ grid._hx/_hy), bb
  (→ cv_bb, но с np.abs), wbbl, crx/cry, region. Значит y2/x2 можно
  считать накоплением _hx/_hy без чтения hx.dat/hy.dat из output/
  (они там тоже есть, но b2fgmtry достаточно).
- Топология: leftcut/rightcut/topcut/bottomcut уже есть; outer_midplane_cells
  вычисляется (nout можно взять оттуда); nsep — найти (строка сепаратрисы
  в матрице: по region/cv_reg или по cuts).
- Matlab_SPb: nout = мин |Bz| в SOL (LFS), nin = макс |Bz| (HFS) — Bz
  из bb(:,:,3); с нашим abs-вариантом cv_bb это согласуется.
- НЕ хватает: nsep/nc1..nc4/ntt-индексы в явном виде + функция
  compute_structured_coords() в regions_structured.py.

## 4. Время (режим 3) — проверено 05.08.2026

- В b2fstati есть поле `time` (real, 1 значение; пример: 8.757E-01 с).
- Наш b2fstate_reader его НЕ читает → ДОБАВИТЬ.
- Соглашение (как в MATLAB): t = time[i] − time[0]; первый watch = t0.
- В MATLAB workspace есть переменная time/times — добавить в build_workspace
  для паритета.

## 5. Эффективность режима 3 (300 watch'ей)

- Сейчас from_directory читает все 1138 файлов (~1.6 с × 300 = 8 мин — НЕТ).
- Нужен ленивый режим: сканирование + чтение переменных по требованию
  (watch.get() читает файл когда попросят). Затронет систему чтения
  (параллельно с визуализацией).
- TimeSeries грузит только нужные переменные.

## 6. Наработки (НЕ начинать с нуля)

- Каталог MATLAB: /home/kirill/solps_matlab_plotting_catalog.md (42 функции +
  50 скриптов, полный разбор).
- references/skipped-matlab-functions.md — 9 функций с пометками, чего не хватает.
- references/matlab-plotting-catalog.md — краткая версия в скилле.
- План v1: .hermes/plans/2026-07-25_visualization_architecture.md.
- Уже реализовано: extract/profile.py (экстракторы omp/imp/target/wall/ft/ns +
  extract_profile_ns + find_flux_tube_by_r), Plot1D (base.py), Plot2D (plot2d.py),
  style.py (пресеты, сделано сегодня), PlotConfig расширенный (base.py).
- MATLAB-ветки-референсы: feature/plot-chord, feature/integration-paths-chords
  (interpolate_chord.m, plot1D_chord.m, PathManager), feature/interactive-plots.

## 7. План реализации (очередность)

0. Доделать structured-ветку regions: cv_r, cv_theta, cv_theta_n, cv_lbl_len
   (сейчас None на structured). + чтение time из b2fstati.
1. PlotData (curves) + Curve — новый контракт.
2. Примитивы: _extract_cell_profile, _extract_face_profile, _interpolate_chord.
3. Реестр @extractor + декларации (name/description/location/sources/params/returns).
4. Экстракторы v1: cell_profile, face_profile, ns_profile, int_poloidal, wall,
   2d_b2. Регионы: named (omp/imp/target_*), ft:N, chord, wall.
5. x_coord: r, r_sep, s_target, theta, theta_norm, psi_n.
6. Плоттеры: Plot1D (curves), Plot2D (viewport-регионы).
7. Конструктор build_plot(watch|collection|ts, variable, region, extractor,
   plotter, config) + проверка совместимости.
8. WatchCollection + TimeSeries (леновая загрузка; время из b2fstati).
9. PlotTimeSeries (интегральные) + PlotTimeSlider (интерактивные).
10. PlotList + YAML (встроенные: student_overview, divertor_analysis,
    advanced_physics, quick_look) + пользовательские.
11. Стили: редактируемые (просмотр/статья), возможно YAML.
12. Проход по каталогу MATLAB: функция за функцией → комбинация
    (экстрактор×плоттер×регион×величина) → чего не хватает → реализовать.
13. Тесты, README, quickstart.ipynb (все три режима).

## 8. Открытые вопросы

- WatchCollection: метки watch'ей (label) — откуда: имя каталога, user_params?
- TimeSeries: где брать метки времени, если b2fstati time одинаков (повтор)?
- psi_n: psi_sep/psi_wall брать из fs_psi (флакс-поверхности)?
- Регион «дивертор верх/низ» для 2D: по координатам Z или по ft_reg?
