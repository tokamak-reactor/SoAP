# SoAP — руководство для разработчика

SoAP (SOLPS-ITER Analysis Package) — Python-пакет для обработки и
визуализации выходных данных кода SOLPS-ITER (плазменное моделирование
дивертора токамака). Работает с двумя типами расчётных сеток:

- **STR** (structured, B2 v3.0.x) — ортогональная матричная сетка,
  классические расчёты (старый пакет: `Matlab_SPb`).
- **UNSTR** (unstructured, B2.5 v3.2.x) — неструктурированная сетка,
  «широкие» расчёты до стенки камеры (новый пакет: `matlab_wg`).

**Цель SoAP** — воспроизводить полный набор переменных, которые MATLAB-
скрипты группы считают при обработке watch'а (диагностического расчёта),
и предоставить единую систему графиков для трёх групп пользователей:
студенты / аналитики / продвинутые, в трёх режимах: 1 watch (подробно),
до 10–12 watch'ей (сравнение, разные сетки/токамаки), до ~300 watch'ей
(time-dependent).

Этот документ — точка входа: что сделано, как устроено, почему именно
так, и что ещё пусто. Подробности — в файлах, на которые даны ссылки.

---

## 1. Быстрый старт

```python
from solps_analysis.core.dataset import SolpsWatch

# STR watch
w = SolpsWatch.from_directory(
    "/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/"
    "Globus_44644_ng_new_equ/watch_15.05.2026_11_31")
# UNSTR watch
# w = SolpsWatch.from_directory(".../Globus-3_WG/sonya_refined_no_redef_v2/watch_02.06.2026_15_30")

w.compute_regions()            # геометрия: координаты, таргеты, сепаратриса
w.construct("calc_additional") # все 160 дополнительных величин
te = w.ws_var("te")            # массив (n_cells,)

# Графики
from solps_analysis.plot import Plot1D, PlotConfig
Plot1D(w, PlotConfig(type="1d", variable=["te", "ti"], along="omp")).render()
```

Данные watch'ей лежат на внешнем диске:
`/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/`.

---

## 2. Архитектура (модули)

```
src/solps_analysis/
├── io/                  # чтение файлов SOLPS
│   ├── b2fgmtry_parser.py   # b2fgmtry (геометрия), b2fstati (состояние)
│   ├── geometry_reader.py   # GridTopology из b2fgmtry (STR и UNSTR)
│   ├── b2fstate_reader.py   # .dat файлы состояния (параллельное чтение)
│   └── stdstream.py         # log_msg и пр.
├── core/
│   ├── grid.py              # GridTopology — dataclass всех полей сетки
│   ├── dataset.py           # SolpsWatch: from_directory, compute_regions,
│   │                        #   construct, ws_var, plot (якорь API)
│   ├── regions.py           # геометрия UNSTR: координаты, таргеты,
│   │                        #   сепаратриса, стенка, сдвиги
│   └── regions_structured.py# геометрия STR: матричная логика (разрезы,
│                            #   строки пластин) + вызов общих функций
├── construct/
│   ├── base.py              # реестр конструкторов (@registry)
│   └── builtin/calc_additional.py  # 160 величин (порт calc_additional.m)
├── extract/
│   └── profile.py           # экстракторы профилей: omp/imp/target_in/out/
│                            #   ft/ns/wall + EXTRACTOR_REGISTRY
└── plot/
    ├── base.py              # Plot, PlotConfig (4 поля + контекстные),
    │                        #   PlotResult
    ├── style.py             # PRESETS: screen/journal/presentation,
    │                        #   apply_preset(), merge_style()
    ├── plot1d.py            # Plot1D, PlotMulti1D
    └── plot2d.py            # Plot2D
```

**Три уровня графиков (архитектура v2, согласована с пользователем):**
1. уровень графика: величина × регион × экстрактор × плоттер
2. уровень списков: PlotList / YAML (встроенные наборы для групп)
3. уровень стилей: редактируемые пресеты

Полный дизайн: `.hermes/plans/visualization-v2.md`.

---

## 3. Единая схема геометрии (главное решение)

**Принцип:** STR и UNSTR дают ОДНИ И ТЕ ЖЕ переменные в SoAP, но каждая
версия считает их «натуральными» для себя методами. Пользователь не видит
разницы в вызовах.

| Сущность | STR | UNSTR |
|---|---|---|
| Вершины vx_x/vx_y | из crx/cry (4 угла на ячейку, дедупликация) | из файла vxX/vxY |
| Центры ячеек cv_x/cv_y | mean(4 углов) — точно | **читать из файла cvX/cvY** (b2ag считает иначе, до 0.53 м расхождения) |
| Середины граней fc_x/fc_y | mean(2 вершин) | mean(2 вершин) |
| Потоки (fna и пр.) | матрицы на ячейках со сдвигом → грани через imap_fcx/fcy | на гранях напрямую (1:nFc) |
| cv_r (радиальная) | физическая ходьба по центрам + сдвиг на сепаратрису | то же |
| cv_theta (полоидальная) | ходьба по флакс-трубкам + сдвиг на OMP | то же |
| Таргеты | строки матрицы (iy=2, iy=ny+1) по cv_reg | fcLbl 1–4 + fallback по координатам |
| Сепаратриса | граница core по cv_reg (все ветви) | ходьба от X-точки по fs |
| Стенка | **нет** — честный отказ (None) | fcLbl 5–8 |
| psi | fpsi в b2fgmtry обычно **нули** → psi из .equ (опция) | vxFpsi есть (сепаратриса ≈ 0) |

**Ключевые решения и почему:**

1. **Центры ячеек не универсальны.** STR: mean(углов) даёт точные центры
   (ортогональность). UNSTR: b2ag считает центры иначе (площадной
   центроид), есть треугольники и вырожденные ячейки — поэтому читаем
   cvX/cvY из файла.
2. **Середины граней универсальны** (грань — отрезок, mean 2 вершин).
3. **Потоки приводятся к граням (1D, nFc)** в обоих типах — это единая
   схема хранения (variable_catalog).
4. **cv_r: 0 на сепаратрисе, минус в core, плюс в SOL** (как y2 в
   Matlab_SPb). Сдвиг на сепаратрису — по радиальным колонкам, где
   сепаратриса = граница core (robust для DND с двумя X-точками).
   Накопление hy (y2) — будущая опция `x_coord="y2"` для совместимости
   со старыми графиками.
5. **cv_theta: 0 на OMP** (по флакс-трубкам, каждая на свою OMP-ячейку).
6. **psi нормировка (стандарт сообщества):**
   `psi_N = (psi − psi_axis)/(psib − psi_axis)` — 0 на оси, 1 на
   сепаратрисе, > 1 в SOL. Источник: b2fgmtry (если ненулевая) или .equ
   файл равновесия (fallback, для STR обязателен). Если fpsi нули —
   отключаемое предупреждение + psi_source=None (БЕЗ подмены на номер
   трубки — это была тихая порча данных!).
7. **«Честный отказ» вместо имитации:** где сущности физически нет
   (стенка в STR, psi в STR без equ) — атрибут = None + понятная ошибка
   при попытке использовать, а не подделка.
8. **is_structured врёт на WG-сетках** (в файле может стоять true при
   UNSTR-структуре). Везде проверять `grid.imap_cv.ndim == 2`, не
   `grid.is_structured`.

Детали: `docs/geometry-unified-scheme.md` (полная таблица + статус всех
8 пунктов), `docs/solps-b2cdcv-reference.md` и
`docs/solps-b2cdcv-unstructured-reference.md` (официальная документация
SOLPS, конспекты), `.hermes/plans/geometry-structured-map.md`
(read_geometry.m → grid, разделы A–F).

---

## 4. Статус: что сделано

**Ядро (готово, регрессия 160/160 на обоих watch'ах):**
- Чтение b2fgmtry / .dat (параллельное, кэши), GridTopology
- construct("calc_additional"): все 160 величин из calc_additional.m
- workspace-совместимый API: ws_var, provenance (matlab_lines)

**Единая геометрия (все 8 пунктов схемы закрыты 06.08.2026):**
1. fc_x/fc_y — проверены (diff 0.0; «баг сдвига» был ошибкой теста)
2. bb — уже неотрицательные в b2fgmtry (abs безвреден)
3. fc_s == gsx/gsy (diff 0.0)
4. честный psi_source (убрана подмена cv_fpsi=cv_ft)
5. единые cv_r/cv_theta (ходьба + сдвиги на сепаратрису/OMP) — STR и UNSTR
6. таргеты/сепаратриса едиными именами (inner/outer/top/active/inactive,
   faces; STR из матрицы, UNSTR из fcLbl + fallback)
7. единые psi-поля: vx_fpsi/cv_fpsi/fc_fpsi/fs_psi (UNSTR из b2fgmtry;
   STR — None без .equ)
8. стенка: UNSTR wall_cells/wall_faces/wall_cells_len (fcLbl 5–8);
   STR — честный None

**Графики (начало):**
- style.py (PRESETS, apply_preset), PlotConfig расширен
- Plot1D / PlotMulti1D (species, boundary, x_axis_unit)
- Plot2D (существовал)
- экстракторы: omp/imp/target/ns/ft/wall

---

## 5. Статус: что пусто / TODO

**Геометрия:**
- [ ] sep2_fc / sep2_vx (вторая сепаратриса для DND) — не реализовано
- [ ] .equ чтение и psi из равновесия (порт `psi_norm.m`; API
      `watch.compute_psi(equ_path=...)`; надстройка авто-поиска .equ на
      уровень выше watch / baserun)
- [ ] x_coord="y2"/"x2" (накопление hy/hx) — опция совместимости
- [ ] wall-сегменты в экстракторе (boundary>1)
- [ ] остаточные артефакты cv_r в углах DND (X-точки): ~14 core-ячеек
      с cv_r>0.01 — связано с сепаратрисой DND
- [ ] fs_psi для STR (появится вместе с .equ)

**Визуализация (по `.hermes/plans/visualization-v2.md`):**
- [ ] PlotData мультикривой (curves с независимыми x — критично для
      сравнения watch'ей на разных сетках)
- [ ] Source-абстракция: SolpsWatch | WatchCollection | TimeSeries
      (режим — свойство источника, не аргумент экстрактора)
- [ ] реестр экстракторов @extractor (имя/описание/location/sources/
      params/returns; примитивы + композиция, 10–40 строк)
- [ ] плоттеры: PlotWall (геометрия стенки), PlotMesh (сетка +
      сепаратриса), PlotTimeSeries/PlotTimeSlider (режим 3)
- [ ] PlotList + YAML load/save + встроенные наборы: student_overview,
      divertor_analysis, advanced_physics, quick_look
- [ ] watch.plot() dispatch по типу + watch.plot_list()
- [ ] экстрактор полоидального интеграла (plot1D_Int_simple) + region filter
- [ ] хорды (регион-хорда: точки (R,Z) → интерполяция; MATLAB-референс
      interpolate_chord.m; Python: matplotlib.tri.LinearTriInterpolator)
- [ ] конструктор графика: 4 поля (плоттер, экстрактор, величина, регион)
      + контекстные параметры только выбранного экстрактора
- [ ] совместимость: жёсткая + предложение альтернативы
- [ ] производительность режима 3 (~300 watch'ей): ленивое чтение
      (сейчас ~1.6 c/watch)

**Прочее:**
- [ ] `docs/` — ВРЕМЕННЫЙ каталог, удалить из репозитория перед релизом
- [ ] README + quickstart.ipynb для трёх групп пользователей
- [ ] тесты (pytest) — сейчас 6 шт., расширить
- [ ] команда из 42 функций matlab_wg разобрана на
      (экстрактор × плоттер × регион × величина) — частично в
      `.hermes/plans/skipped-matlab-functions.md`

---

## 6. Тестирование

```bash
# Быстрая регрессия всех 160 величин (скрипт в скилле):
python3 ~/.hermes/skills/software-development/solps-analysis-project/scripts/regression_calc_additional.py
#  → OK: 160/160 на STR watch (~4 с)
python3 ~/.hermes/skills/software-development/solps-analysis-project/scripts/regression_calc_additional.py \
    /run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus-3_WG/\
sonya_refined_no_redef_v2/watch_02.06.2026_15_30/
#  → OK: 160/160 на UNSTR watch (~10 с)

python3 -m pytest tests/ -q   # юнит-тесты
```

**Регрессия обязательна** после изменений в geometry/regions/construct:
квантити зависят от геометрии (таргеты, сепаратриса, флакс-трубки), и
тихая поломка возможна.

---

## 7. Ссылки (все документы)

| Файл | Что это |
|---|---|
| `docs/geometry-unified-scheme.md` | **единая схема STR/UNSTR** + статус 8 пунктов (актуальный) |
| `docs/solps-b2cdcv-reference.md` | конспект офиц. документации (structured) |
| `docs/solps-b2cdcv-unstructured-reference.md` | конспект офиц. документации (unstructured) |
| `.hermes/plans/visualization-v2.md` | архитектура визуализации v2 (дизайн графиков) |
| `.hermes/plans/geometry-structured-map.md` | read_geometry.m → grid (разделы A–F) |
| `.hermes/plans/2026-07-25_visualization_architecture.md` | ранний план визуализации |
| `.hermes/plans/skipped-matlab-functions.md` | 9 функций MATLAB, не портированных |
| `/home/kirill/solps_matlab_plotting_catalog.md` | каталог 42 MATLAB-функций группы |
| MATLAB-референсы | `/run/media/kirill/Fusion/Science/SOLPS_ITER/MATLAB/matlab_wg/` (новый), `/home/kirill/projects/Matlab_SPb/` (старый) |
| Офиц. документация SOLPS | `/run/media/kirill/MAIN/00_Active_Science/Papers_Books_Manuals/{structured,unstructured} info/` (b2cdca.F, b2cdcv.F) |

---

## 8. Словарь терминов

- **watch** — каталог диагностического расчёта SOLPS-ITER (набор .dat
  файлов состояния + b2fgmtry + b2fstati).
- **cv / fc / vx** — control volume (ячейка) / face (грань) / vertex
  (вершина). Префиксы имён массивов.
- **fs / ft** — flux surface (флакс-поверхность) / flux tube (флакс-трубка).
- **cv_r, cv_theta** — радиальная (от сепаратрисы) и полоидальная (от OMP)
  координаты в центрах ячеек.
- **OMP / IMP** — outer / inner midplane (внешний / внутренний экватор).
- **PFR** — private flux region (частная область между сепаратрисами).
- **DND / SN / DDN** — double null / single null / double-null
  диверторные конфигурации (по числу X-точек).
- **y2, x2** — классические координаты Matlab_SPb: y2 = расстояние от
  внутренней сепаратрисы (накопление hy), x2 = полоидальная (накопление
  hx с обходом разрезов).
- **psi_N** — нормированный полоидальный поток: 0 на оси, 1 на
  сепаратрисе, >1 в SOL.
