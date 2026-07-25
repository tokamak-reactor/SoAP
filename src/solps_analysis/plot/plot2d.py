"""Plot2D — 2D colour maps on SOLPS mesh."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation

from solps_analysis.core.dataset import SolpsWatch
from solps_analysis.plot.base import Plot, PlotConfig, PlotResult, PRESETS


class Plot2D(Plot):
    """2D colour map of a scalar field on the SOLPS mesh.

    Uses :func:`matplotlib.pyplot.tricontourf` for unstructured grids
    and structured-to-unstructured conversion.

    Future: will use proper quadrilateral ``PatchCollection`` for
    precise cell-boundary rendering.
    """

    def render(self, ax: Axes | None = None) -> PlotResult:
        self._apply_style()

        grid = self.watch.grid
        var_name = self.config.variable

        if var_name is None:
            raise ValueError("Plot2D requires a variable name")

        var = self.watch.get(var_name)
        if var is None:
            raise ValueError(f"Variable '{var_name}' not found in watch")

        data = np.asarray(var.data, dtype=np.float64).ravel()
        if data.shape[0] != grid.n_cells:
            # Face data — warn and skip
            raise ValueError(
                f"Variable '{var_name}' has {data.shape[0]} elements "
                f"(expected {grid.n_cells} cells for 2D plot)"
            )

        # --- triangulation from cell centres ---
        x = grid.cv_x
        y = grid.cv_y

        if x is None or y is None:
            raise ValueError("Cell centre coordinates not available")

        # Filter guard cells (where data is zero or extreme)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(data)

        # --- figure / axes ---
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self._style.get("figure.figsize", (10, 8))
            )
        else:
            fig = ax.figure

        # --- triangulation ---
        tri = Triangulation(x[valid], y[valid])

        # --- levels ---
        vmin = self.config.style_overrides.get("vmin")
        vmax = self.config.style_overrides.get("vmax")
        if vmin is None:
            vmin = np.percentile(data[valid], 2)
        if vmax is None:
            vmax = np.percentile(data[valid], 98)

        nlevels = self.config.style_overrides.get("nlevels", 50)

        if self.config.log:
            levels = np.logspace(np.log10(max(vmin, 1e-30)), np.log10(vmax), nlevels)
            norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, nlevels)
            norm = None

        cmap = self.config.style_overrides.get("cmap", "viridis")

        # --- contourf ---
        cf = ax.tricontourf(tri, data[valid], levels=levels, cmap=cmap, norm=norm,
                            extend="both")

        # --- wall overlay (if available) ---
        self._plot_wall(ax, grid)

        # --- separatrix overlay (if available) ---
        self._plot_separatrix(ax, grid)

        # --- styling ---
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        if self.config.title:
            ax.set_title(self.config.title)
        else:
            ax.set_title(
                f"{var_name} [{var.meta.unit}]" if var.meta.unit else var_name
            )

        # --- colorbar ---
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(var_name)

        fig.tight_layout()
        return PlotResult(config=self.config, fig=fig, ax=ax)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _plot_wall(ax: Axes, grid) -> None:
        """Overlay wall geometry if available."""
        wf = getattr(grid, "wall_faces", None)
        if wf is not None and len(wf) > 0 and grid.fc_x is not None:
            idx = np.asarray(wf, dtype=np.intp).ravel()
            valid = (idx > 0) & (idx <= grid.n_faces)
            idx = idx[valid] - 1
            if len(idx) > 0:
                ax.plot(grid.fc_x[idx], grid.fc_y[idx], "k-", linewidth=1, alpha=0.7)

    @staticmethod
    def _plot_separatrix(ax: Axes, grid) -> None:
        """Overlay separatrix if available."""
        sep_fc = getattr(grid, "sep_fc", None)
        if sep_fc is not None and len(sep_fc) > 0 and grid.fc_x is not None:
            idx = np.asarray(sep_fc, dtype=np.intp).ravel()
            valid = (idx > 0) & (idx <= grid.n_faces)
            idx = idx[valid] - 1
            if len(idx) > 0:
                ax.plot(grid.fc_x[idx], grid.fc_y[idx], "w--", linewidth=1.5, alpha=0.7)
