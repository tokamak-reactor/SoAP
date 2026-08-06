"""Base plot classes: Plot, PlotConfig, PlotResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from solps_analysis.core.dataset import SolpsWatch
from solps_analysis.plot.style import PRESETS, VALID_RCPARAMS, merge_style


@dataclass
class PlotConfig:
    """Configuration for a single plot.

    Fields are deliberately YAML-friendly (plain types) so a PlotConfig
    can be serialized to/from a PlotList YAML file.
    """

    type: str = "1d"  # "1d" | "2d" | "wall" | "mesh"
    variable: str | list[str] | None = None
    along: str | list[str] | None = None
    region: str | None = None      # filter by region label ("core", "sol", ...)
    species: int | None = None     # species column for multi-species variables
    element: str | None = None     # element symbol ("C", "D") for ns plots
    log: bool = False
    include_guards: bool = False
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    x_axis_unit: str = "m"         # "m" | "cm" (wall plots)
    boundary: int = 1              # wall boundary segment index
    title: str | None = None
    style: str = "screen"          # "screen" | "journal" | "presentation" | "custom"
    figsize: tuple[float, float] | None = None
    style_overrides: dict = field(default_factory=dict)
    save: str | None = None        # output path (PlotList.run writes here)

    # ── helpers ────────────────────────────────────────────────────
    def resolved_style(self) -> dict[str, Any]:
        style = merge_style(self.style, self.style_overrides)
        if self.figsize is not None:
            style["figure.figsize"] = self.figsize
        return style

    def as_list(self, name: str) -> "PlotConfig":
        """Return a copy with ``variable``/``along`` wrapped as lists."""
        return PlotConfig(
            type=self.type,
            variable=[self.variable] if isinstance(self.variable, str) else self.variable,
            along=[self.along] if isinstance(self.along, str) else self.along,
            region=self.region, species=self.species, element=self.element,
            log=self.log, include_guards=self.include_guards, cmap=self.cmap,
            vmin=self.vmin, vmax=self.vmax, xlim=self.xlim, ylim=self.ylim,
            x_axis_unit=self.x_axis_unit, boundary=self.boundary,
            title=self.title, style=self.style, figsize=self.figsize,
            style_overrides=dict(self.style_overrides), save=self.save,
        )


@dataclass
class PlotResult:
    """Result of a plot — allows post-render tweaks.

    After ``render()`` the user can freely edit the figure/axes:

    .. code-block:: python

        result.ax.legend(loc="upper left")
        result.fig.savefig("te_profile.pdf")
    """

    config: PlotConfig
    fig: Figure
    ax: Axes


class Plot:
    """Abstract base for all plot types."""

    def __init__(
        self,
        watch: SolpsWatch,
        config: PlotConfig | None = None,
        **kwargs,
    ):
        self.watch = watch
        if config is None:
            config = PlotConfig(**kwargs)
        self.config = config
        self._style = config.resolved_style()

    def _apply_style(self) -> None:
        """Apply preset style to matplotlib rcParams.

        Only valid rcParams keys are applied — plot-specific keys
        (``cmap``, ``vmin``, ``vmax``) are handled by ``render()``.
        """
        for key, val in self._style.items():
            if key in VALID_RCPARAMS:
                try:
                    plt.rcParams[key] = val
                except KeyError:
                    pass

    def _figure(self, ax: Axes | None = None) -> tuple[Any, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self._style.get("figure.figsize", (10, 6)))
        else:
            fig = ax.figure
        return fig, ax

    def render(self, ax: Axes | None = None) -> PlotResult:
        """Render the plot. Override in subclasses."""
        raise NotImplementedError
