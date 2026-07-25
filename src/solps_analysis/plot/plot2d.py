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
        polys = _build_cell_vertices(grid)
        if polys is None:
            raise ValueError(
                "Cannot build 2D plot: vertex data not available. "
                "Run watch.compute_regions() first."
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
        cbar = fig.colorbar(pc, ax=ax)
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


def _build_cell_vertices(grid) -> list[np.ndarray] | None:
    """Compute cell vertex coordinates for PatchCollection.

    For structured grids: reconstruct from imap + cell centers.
    For unstructured grids: return None (not yet supported).
    """
    if not grid.is_structured:
        return None

    if grid.imap_cv is None or grid.cv_x is None:
        return None

    nx, ny = grid.nx + 2, grid.ny + 2  # full grid with guards
    imap = grid.imap_cv  # (nx, ny)

    # Build a reverse lookup: cell_index → (ix, iy)
    cell_to_ij: dict[int, tuple[int, int]] = {}
    for ix in range(nx):
        for iy in range(ny):
            c = int(imap[ix, iy])
            if c > 0:
                cell_to_ij[c - 1] = (ix, iy)

    # Build vertex coordinates by averaging cell centers of 4 neighbours
    # Vertex (ix, iy) is at corner between cells (ix-1,..), (ix,..), etc.
    # For each real cell, its 4 vertices are at:
    #   (ix, iy), (ix+1, iy), (ix+1, iy+1), (ix, iy+1) in imap-space
    # But we need (R, Z) coordinates for each vertex.

    # Approach: for each cell, get its 4 face centers from cv_fc
    # Each face center approximates the midpoint of an edge.
    # The 4 corners are reconstructed from adjacent face centers.

    # Simpler: compute vertices as weighted average of surrounding cell centers.
    # For each cell at imap position (ix, iy), its 4 vertices are at
    # half-integer positions.

    # Actually the simplest correct approach for a structured grid:
    # Each vertex (ix, iy) in the imap-space is shared by up to 4 cells.
    # The vertex coordinate is the average of the cell centers of those cells.

    cv_x = np.asarray(grid.cv_x, dtype=np.float64)
    cv_y = np.asarray(grid.cv_y, dtype=np.float64)

    # For each cell, compute its vertices as 4 face-face intersections
    # Cell at (ix, iy) has vertices at corners of imap cells:
    #   v0 = (ix, iy), v1 = (ix+1, iy), v2 = (ix+1, iy+1), v3 = (ix, iy+1)

    polys: list[np.ndarray | None] = [None] * grid.n_cells

    for icv in range(grid.n_cells):
        if icv not in cell_to_ij:
            continue
        ix, iy = cell_to_ij[icv]
        # Get the 4 cells that share each corner vertex
        verts_ij = [
            (ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1),
        ]
        verts_r = np.zeros(4)
        verts_z = np.zeros(4)
        for k, (vix, viy) in enumerate(verts_ij):
            # Average the centers of surrounding cells
            cells_around = []
            for di in [0, -1]:
                for dj in [0, -1]:
                    ci, cj = vix + di, viy + dj
                    if 0 <= ci < nx and 0 <= cj < ny:
                        c = int(imap[ci, cj])
                        if c > 0 and c - 1 < grid.n_cells:
                            cells_around.append(c - 1)
            if cells_around:
                verts_r[k] = np.mean(cv_x[cells_around])
                verts_z[k] = np.mean(cv_y[cells_around])

        # Check for degenerate (all zeros or repeated vertices)
        if np.all(verts_r == 0) or np.all(verts_z == 0):
            continue
        polys[icv] = np.column_stack([verts_r, verts_z])

    return polys

