"""Plot1D and PlotMulti1D — 1D profile plots."""
from __future__ import annotations

import matplotlib.pyplot as plt

from solps_analysis.plot.base import Plot, PlotResult


class Plot1D(Plot):
    """1D profile plot along OMP/IMP/target/ft/wall.

    ``config.variable`` may be a single name or a list (multiple curves on
    one axes); ``config.along`` likewise. ``config.species`` selects a
    species column for multi-species variables (na, fna, ...).
    """

    def render(self, ax=None) -> PlotResult:
        from solps_analysis.extract import extract_profile

        self._apply_style()
        fig, ax = self._figure(ax)

        variables = self.config.variable
        if variables is None:
            raise ValueError("Plot1D requires config.variable")
        if isinstance(variables, str):
            variables = [variables]

        alongs = self.config.along
        if alongs is None:
            alongs = ["omp"]
        elif isinstance(alongs, str):
            alongs = [alongs]

        pairs = [(v, a) for v in variables for a in alongs]

        xl, yl = "", ""
        for var_name, a in pairs:
            try:
                x, y, xl, yl = extract_profile(
                    self.watch, var_name,
                    along=a,
                    species=self.config.species,
                    include_guards=self.config.include_guards,
                    boundary=self.config.boundary,
                    x_axis_unit=self.config.x_axis_unit,
                )
                label = f"{var_name} ({a})" if len(pairs) > 1 else var_name
                ax.plot(x, y, label=label,
                        linewidth=self._style.get("lines.linewidth", 1.5))
            except Exception as e:
                ax.text(0.5, 0.5, f"Error ({var_name}, {a}): {e}",
                        transform=ax.transAxes, ha="center")

        ax.set_xlabel(xl)
        ax.set_ylabel(yl if len(variables) == 1 else "")
        if self.config.title:
            ax.set_title(self.config.title)
        if self.config.log:
            ax.set_yscale("log")
        if self.config.xlim:
            ax.set_xlim(self.config.xlim)
        if self.config.ylim:
            ax.set_ylim(self.config.ylim)
        ax.grid(True, alpha=self._style.get("grid.alpha", 0.3))
        if len(pairs) > 1:
            ax.legend()
        fig.tight_layout()
        return PlotResult(config=self.config, fig=fig, ax=ax)


class PlotMulti1D(Plot):
    """Multiple 1D profiles (different variables) on one axes.

    Unlike :class:`Plot1D` (which takes ``variable`` as str or list), this
    is an explicit multi-curve container: one curve per variable, always
    with a legend. Convenient for paired plots (Te+Ti on OMP).
    """

    def render(self, ax=None) -> PlotResult:
        self._apply_style()
        fig, ax = self._figure(ax)

        variables = self.config.variable
        if variables is None:
            raise ValueError("PlotMulti1D requires config.variable")
        if isinstance(variables, str):
            variables = [variables]
        along = self.config.along if isinstance(self.config.along, str) else "omp"

        xl, yl = "", ""
        for var_name in variables:
            try:
                from solps_analysis.extract import extract_profile
                x, y, xl, yl = extract_profile(
                    self.watch, var_name, along=along,
                    species=self.config.species,
                    include_guards=self.config.include_guards,
                )
                ax.plot(x, y, label=var_name,
                        linewidth=self._style.get("lines.linewidth", 1.5))
            except Exception as e:
                ax.text(0.5, 0.5, f"Error ({var_name}): {e}",
                        transform=ax.transAxes, ha="center")

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if self.config.title:
            ax.set_title(self.config.title)
        if self.config.log:
            ax.set_yscale("log")
        if self.config.xlim:
            ax.set_xlim(self.config.xlim)
        if self.config.ylim:
            ax.set_ylim(self.config.ylim)
        ax.grid(True, alpha=self._style.get("grid.alpha", 0.3))
        if len(variables) > 1:
            ax.legend()
        fig.tight_layout()
        return PlotResult(config=self.config, fig=fig, ax=ax)
