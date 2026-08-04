"""Base plot classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from solps_analysis.core.dataset import SolpsWatch


# ──────────────────────────────────────────────
# Style presets
# ──────────────────────────────────────────────

PRESETS: dict[str, dict[str, Any]] = {
    "screen": {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    },
    "journal": {
        "figure.figsize": (17 / 2.54, 11 / 2.54),
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
    },
    "presentation": {
        "figure.figsize": (16, 9),
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
    },
}


@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    type: str  # "1d" | "2d" | "wall" | "mesh"
    variable: str | list[str] | None = None
    along: str | list[str] | None = None
    log: bool = False
    include_guards: bool = False
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    title: str | None = None
    style: str = "screen"
    style_overrides: dict = field(default_factory=dict)


@dataclass
class PlotResult:
    """Result of a plot — allows post-render tweaks."""
    config: PlotConfig
    fig: Figure
    ax: Axes


# ──────────────────────────────────────────────
# Base Plot class
# ──────────────────────────────────────────────


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
        self._style = PRESETS.get(config.style, PRESETS["screen"]).copy()
        self._style.update(config.style_overrides)

    def _apply_style(self) -> None:
        """Apply preset style to matplotlib rcParams.
        
        Only applies valid rcParams keys — plot-specific keys like
        ``cmap``, ``vmin``, ``vmax`` are handled by the individual
        ``render()`` methods.
        """
        valid_rc = {
            "figure.figsize", "figure.dpi", "font.family", "font.size",
            "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize", "lines.linewidth", "grid.alpha",
            "savefig.bbox", "savefig.pad_inches",
        }
        for key, val in self._style.items():
            if key in valid_rc:
                try:
                    plt.rcParams[key] = val
                except KeyError:
                    pass

    def render(self, ax: Axes | None = None) -> PlotResult:
        """Render the plot. Override in subclasses."""
        raise NotImplementedError


# ──────────────────────────────────────────────
# Plot1D
# ──────────────────────────────────────────────


class Plot1D(Plot):
    """1D profile plot along OMP/IMP/target/ft."""

    def render(self, ax: Axes | None = None) -> PlotResult:
        from solps_analysis.extract import extract_profile

        self._apply_style()

        if ax is None:
            fig, ax = plt.subplots(figsize=self._style.get("figure.figsize", (10, 6)))
        else:
            fig = ax.figure

        variables = self.config.variable
        if isinstance(variables, str):
            variables = [variables]

        alongs = self.config.along
        if isinstance(alongs, str):
            alongs = [alongs]
        elif alongs is None:
            alongs = ["omp"]

        # Build all (var, along) pairs
        pairs: list[tuple[str, str]] = []
        for var_name in variables:
            for a in alongs:
                pairs.append((var_name, a))

        xl = ""
        yl = ""
        for var_name, a in pairs:
            try:
                x, y, xl, yl = extract_profile(
                    self.watch, var_name,
                    along=a,
                    include_guards=self.config.include_guards,
                )
                label = f"{var_name} ({a})" if len(pairs) > 1 else var_name
                ax.plot(x, y, label=label, linewidth=self._style.get("lines.linewidth", 1.5))
            except Exception as e:
                ax.text(0.5, 0.5, f"Error ({var_name}, {a}): {e}",
                        transform=ax.transAxes, ha="center")

        ax.set_xlabel(xl)

        # Combine ylabels for multiple variables
        if len(variables) == 1:
            ax.set_ylabel(yl)
        else:
            ax.set_ylabel("")

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
