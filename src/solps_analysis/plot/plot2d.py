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

        # --- build cell polygons from vertices ---
        polys = _build_cell_polygons(grid)
        if polys is None:
            raise ValueError(
                "Cannot build cell polygons: vertex data not available. "
                "Use a structured grid or load vertex connectivity."
            )

        # --- filter valid cells ---
        valid_data = np.isfinite(data)
        if not np.any(valid_data):
            raise ValueError("All data values are NaN/Inf")

        # --- figure / axes ---
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self._style.get("figure.figsize", (10, 8))
            )
        else:
            fig = ax.figure

        # --- levels ---
        vmin = self.config.style_overrides.get("vmin")
        vmax = self.config.style_overrides.get("vmax")
        if vmin is None:
            vmin = np.nanpercentile(data[valid_data], 2)
        if vmax is None:
            vmax = np.nanpercentile(data[valid_data], 98)

        cmap_name = self.config.style_overrides.get("cmap", "viridis")
        cmap = plt.colormaps.get(cmap_name)

        if self.config.log:
            norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
        else:
            norm = plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # --- build PatchCollection ---
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        patches = []
        patch_data = []
        for icv in range(grid.n_cells):
            if not valid_data[icv]:
                continue
            verts = polys[icv]
            if verts is None:
                continue
            patches.append(MplPolygon(verts, closed=True))
            patch_data.append(data[icv])

        pc = PatchCollection(patches, array=np.array(patch_data),
                             cmap=cmap, norm=norm,
                             edgecolors="none")
        ax.add_collection(pc)

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
