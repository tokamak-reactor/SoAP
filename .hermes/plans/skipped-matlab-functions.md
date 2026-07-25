# Список пропущенных функций MATLAB при обзоре

При обходе plotting_functions/plot1D*.m помечены как «пропущено»:

1. **plot1D_simple_Nresults.m** — multi-case 1D (сравнение нескольких watch'ей).
   Нужен: ComparisonPlot + tiledlayout + average_ft.

2. **plot1D_Int_simple.m** — poloidally-integrated 1D (интеграл по poloidal direction).
   Нужен: extract_int_simple экстрактор.

3. **plot1D_Int_simple_Nresults.m** — multi-case poloidally-integrated (комбинация #1 + #2).

4. **plot1D_multiline.m** — несколько flux tube'ов на одном 1D графике.
   Нужен: поддержка нескольких `along=` значений.

5. **plot1D_simple_multiline.m** — то же, но с gmtry-интерфейсом и textbox.

6. **plot1D_simple_ns.m** — по одной линии на charge state (ns видов).
   Нужен: support species parameter в Plot1D.

7. **plot1D_Nresults_along_Ft.m** — multi-case вдоль flux tube.
   Комбинация #1 + ft:N.

8. **plot1D_wall_part_Nresults.m** — multi-case wall profile (concatenated segments).
   Нужен: multi-case + wall extractor.

9. **poloidal_plot_multiline.m** — poloidal profiles на нескольких ft с фильтром по региону.
   Нужен: region filter в экстракторе.
