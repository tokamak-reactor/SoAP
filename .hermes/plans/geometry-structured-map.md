# Карта соответствия: Matlab_SPb read_geometry.m → SoAP grid (structured)

Составлено 05.08.2026. Цель: полный набор геометрических величин для
structured-сетки, «натуральных» для матричной структуры (в отличие от
unstructured, где всё из топологии). Ничего не пропустить, не дублировать.

## A. Топология (из b2fgmtry + разрезов)

Источник формул: Matlab_SPb/IO/read_b2fgmtry.m строки 88-115.

| MATLAB_SPb | Формула (nncut=2 / DDN) | Формула (nncut=1 / SN) | Наш grid | Статус |
|---|---|---|---|---|
| nx, ny | — | — | nx, ny | ✅ есть |
| nncut | — | — | nncut | ✅ есть |
| leftcut/rightcut/topcut | — | — | _cut_leftcut/_cut_rightcut/_cut_topcut | ✅ есть |
| nc1 | leftcut(1) | leftcut(1) | — | ❌ вычислить |
| nc2 | leftcut(2) − 1 | nc3 − 1 | — | ❌ вычислить |
| nc3 | rightcut(2) | round(lc1+rc1)/2 | — | ❌ вычислить |
| nc4 | rightcut(1) − 1 | rightcut(1) − 1 | — | ❌ вычислить |
| ntt | last(leftix==−2) − 3 | nc2 | — | ❌ (нужен leftix!) |
| nsep | topcut(1) − 1 | topcut(1) − 1 | — | ❌ вычислить |
| nsep2 | topcut(2) − 1 | nsep | — | ❌ вычислить |
| nout, nin | экстремумы Bz в SOL | то же | outer/inner_midplane_cells | ✅ есть (др. путём) |
| leftix/rightix/topiy/bottomyi | соседи | — | bp_dir/bt_dir? (проверить!) | ❓ проверить |

## B. Размеры, объёмы, B-поля, углы

| MATLAB_SPb (источник) | Наш grid | Статус |
|---|---|---|
| hx (hx.dat / b2fgmtry.hx) | _hx | ✅ есть |
| hy (hy.dat / b2fgmtry.hy) | _hy | ✅ есть |
| hy1, hz, pbsx (hy1.dat, hz.dat, pbsx.dat) | cv_hz? | ❓ частично (cv_hz есть) |
| vol (vol.dat) | cv_vol | ✅ есть |
| bb → Bx, Bz, B (bbx/bbz/bb.dat) | cv_bb (с np.abs!) | ⚠️ есть, проверить знак |
| R0..R3, Z0..Z3 (crx0-3/cry0-3.dat) | cv_crn_r, cv_crn_z | ✅ есть |
| gsx = π(R0+R2)·√((R0−R2)²+(Z0−Z2)²) (площадь x-грани) | fc_s? | ❓ проверить соответствие |
| gsy = π(R0+R1)·√((R0−R1)²+(Z0−Z1)²) (площадь y-грани) | fc_s? | ❓ проверить |
| qc (qc.dat) | — | ❓ |

## C. Координаты (главное!)

| MATLAB_SPb | Формула | Наш grid | Статус |
|---|---|---|---|
| yc | накопление hy от южной границы: yc(1)=hy(1)/2; yc(i)=yc(i−1)+(hy(i)+hy(i−1))/2 | — | промежуточная |
| y1, y1_shift | накопление hy (суммы) | — | промежуточная |
| **y2** | yc(i,j) − yc(nsep+2,j) − hy(nsep+2,j)/2 — расстояние от ВНУТРЕННЕЙ сепаратрисы (минус = core/PFR, плюс = SOL) | **cv_r** | ❌ вычислить |
| xc | накопление hx с разрезами (nc1..nc4, ntt; сложная логика 236-337) | — | промежуточная |
| x1, x1_shift | накопление hx | — | промежуточная |
| **x2** | xc − xzero; xzero: CORE/внутр.SOL → xc(i,nout+2) (от OMP); bottom PFR → 0; top PFR → 0; outer SOL → xc(i,nout+2) | **cv_theta** | ❌ вычислить |
| x2_prll | вариант с B/Bx (параллельная длина) | — | опционально |
| θ_norm (0→1) | в Matlab_SPb нет — в matlab_wg есть (для unstr) | cv_theta_n | ❌ вычислить (нормировка по столбцам) |
| cvLbl_len | в Matlab_SPb НЕТ (y2 покрывает пластины) | cv_lbl_len/fc_lbl_len | ❌ вычислить (границы матрицы) |

## D. Чего НЕТ в Matlab_SPb read_geometry, но нужно для единого API

**Принцип (уточнён 05.08.2026, слова пользователя):** в structured НЕТ
стенки, фейсов и cv_lbl_len как самостоятельных сущностей. Есть:
- последние магнитные поверхности в SOL/PFR — они НЕ совпадают со стенкой
  (просто последние поверхности, проходящие от пластины к пластине, не
  пересекая стенку);
- пластины, примыкающие полоидально к матрице как её границы;
- матрицы потоков, где каждой ЯЧЕЙКЕ соответствует поток (у граничных
  ячеек в сторону границы потоков нет → потоки «сдвинуты» в зависимости
  от того, какая из 4 граней).
Поэтому «натурально» для STR:

| Величина | Для STR (натурально) |
|---|---|
| inner/outer_target_cells/faces, top_targets | строки матрицы: ntt/nc2/nc3 (R_plot_targets.m: npl_ib/npl_ob/npl_it/npl_ot) |
| sep_fc, separatrix_cells, core_sep_faces | строка nsep матрицы |
| **стенка (wall_cells, cvs_boundary, fcs_boundary)** | **НЕ ИМЕЕТ СМЫСЛА для STR** — честный отказ (последняя поверхность ≠ стенка). Возможность: внешний .ogr файл стенки (TRT) как отдельный слой, не связанный с сеткой |
| **cv_lbl_len / fc_lbl_len** | **НЕ ИМЕЕТ СМЫСЛА для STR** — это метки участков стенки WG. Для STR вместо них: y2 на пластинах (от strike point) |
| fc_lbl_group, fc_lbl_group_len | не нужны для STR |
| **fc_theta, fc_r, fc_theta_n (фейсовые координаты)** | **фейсов как объектов НЕТ** — для STR «фейсовые» величины (потоки) рисуются на ТЕХ ЖЕ координатах y2/x2, что и скалярные, но со СДВИГОМ пределов по матрице (Flux_flag, см. раздел G) |
| cv_fs, fc_ft, fc_fs, fs_fs, fs_sep, fs_sep2 | из ft/fs-структур (уже строятся для STR — дополнить имена) |
| is_cross_outmidpl/inmidpl | какие флакс-поверхности пересекают OMP/IMP |
| ft_conn | связность флакс-трубок (для STR) |
| cv_sz | размеры ячеек |
| cv_or | ориентация ячеек |
| fs_psi, vx_fpsi | psi на поверхностях/вершинах (из cv_fpsi — есть) |
| imap_vx | (есть на STR, нет на UNSTR — обратная задача) |

## G. Потоки на structured — «матрица со сдвигом» (Flux_flag)

В Matlab_SPb фейсовые величины (потоки, E-поля) рисуются на тех же y2,
что и скалярные; меняются только пределы по матрице (аргумент Flux_flag
в R_plot/R_plot_core_1Darray):

```matlab
if Flux_flag == 0      % скаляры: ibeg=1, iend=nsep+2
elseif Flux_flag == 1  % полоидально-интегрированные потоки: ibeg=2, iend=nsep+3
elseif Flux_flag == 2  % величины с мусором в guard cell (источники): ibeg=2, iend=nsep+2
```

Пример: `R_plot_core_1Darray(y2, 1, ny-1, nout, ..., Ey, 'E_{rad}', ...)`.
Значит для STR экстрактор потока = та же строка матрицы, сдвинутая на 1
в нужную сторону; координата НЕ меняется (y2). Это «натурально».

## E. EIRENE (fort.33/34/35) — отдельная тема

R_apex/Z_apex, N_tria, i_apex1-3, i_Neighb_tria_side1-3, b2_cell_x/y,
RC_tria/ZC_tria (центры треугольников), neighb_apex/tria.
В SoAP: EIRENE-сетка ещё не читается (вне текущей задачи, но
extract_2d_eirene из плана визуализации потребует этого).

## F. Итог: что реально добавить в regions_structured.py

1. Топология: nc1..nc4, nsep, nsep2, ntt (формулы из раздела A; для ntt
   нужен leftix — проверить bp_dir/bt_dir или прочитать leftix).
2. cv_r = y2 (накопление _hy от nsep).
3. cv_theta = x2 (накопление _hx с разрезами, отсчёт от nout/пластин).
4. cv_theta_n (нормировка 0→1 по столбцам).
5. Таргеты под ЕДИНЫМИ именами (inner_target_cells и т.д. — сейчас
   regions_structured выдаёт старые cv_inner_tar/cv_outer_tar, а API
   экстракторов/calc_additional ждёт новые имена — это и есть причина
   «таргеты отсутствуют на STR»). Для faces — СДВИГ по матрице (раздел G),
   отдельных fc-объектов не создаём.
6. sep_fc/separatrix_cells/core_sep_faces (строка nsep).
7. fc_ft/fc_fs/cv_fs/fs_fs/fs_sep/fs_sep2 (дополнить имена).
8. is_cross_outmidpl/inmidpl, ft_conn, cv_sz, cv_or.
9. fs_psi/vx_fpsi (из cv_fpsi).
10. НЕ ДЕЛАЕМ для STR: стенка, cv_lbl_len/fc_lbl_len, fc_lbl_group,
    fc_theta/fc_r/fc_theta_n как объекты — честный отказ в API
    (стенка: опционально внешний .ogr как отдельный слой).

Проверка после реализации: инвентаризация (скрипт из сессии 05.08.2026)
должна показать «OK» для всех величин на STR; численные значения y2/x2
сверить с MATLAB (R_plot(y2,...) на Te_mout/ne_tar эталонах).
