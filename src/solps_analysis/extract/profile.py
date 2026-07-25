"""1D profile extraction from SolpsWatch."""
from __future__ import annotations
from typing import Any
import numpy as np
from solps_analysis.core.grid import GridTopology
from solps_analysis.core.dataset import SolpsWatch

EXTRACTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def _require_regions(grid: GridTopology, needed: list[str]) -> None:
    missing = [a for a in needed if getattr(grid, a, None) is None]
    if missing:
        raise ValueError(
            f"Region data not computed. Missing grid attributes: {missing}\n"
            f"Run watch.compute_regions() first."
        )


def extract_profile(
    watch: SolpsWatch,
    variable: str,
    along: str = "omp",
    include_guards: bool = False,
    species: int | None = None,
    boundary: int = 1,
    x_axis_unit: str = "m",
) -> tuple[np.ndarray, np.ndarray, str, str]:
    # ── resolve variable data ─────────────────────────────────────
    var = watch.get(variable)
    if var is None:
        raise ValueError(f"Variable '{variable}' not found in watch")
    data = np.asarray(var.data, dtype=np.float64)
    ylabel = f"{variable} [{var.meta.unit}]" if var.meta.unit else variable
    if data.ndim > 1:
        if species is not None:
            data = data[:, species]
        elif data.shape[1] == 1:
            data = data[:, 0]
        else:
            data = data[:, 0]
    grid = watch.grid

    # ── flux tube ─────────────────────────────────────────────────
    if along.startswith("ft:"):
        ft_num = int(along.split(":")[1])
        return _extract_along_ft(grid, data, ft_num, ylabel)

    # ── wall (special: boundary selection, structured rejection) ──
    if along == "wall":
        if grid.is_structured:
            raise ValueError(
                "Wall profile (along='wall') is not meaningful for structured"
                " grids — grid ends at magnetic surfaces, not physical walls."
            )
        return _extract_wall(grid, data, ylabel, boundary=boundary, x_unit=x_axis_unit)

    # ── registry lookup ───────────────────────────────────────────
    if along not in EXTRACTOR_REGISTRY:
        known = ", ".join(sorted(EXTRACTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown extraction path '{along}'. Known: {known}")
    entry = EXTRACTOR_REGISTRY[along]

    needed = [entry["indices_attr"]]
    if include_guards and entry.get("guard_indices_attr"):
        needed = [entry["guard_indices_attr"]]
    _require_regions(grid, needed)

    coord = getattr(grid, entry["coord_attr"], None)
    if coord is None:
        if entry["coord_attr"] in ("cv_r", "cv_theta") and grid.cv_x is not None:
            r = np.sqrt(grid.cv_x ** 2 + grid.cv_y ** 2)
            if entry["coord_attr"] == "cv_r":
                coord = r
                setattr(grid, "cv_r", r)
            else:
                coord = np.arctan2(grid.cv_y, grid.cv_x)
                setattr(grid, "cv_theta", coord)
        else:
            raise ValueError(f"Grid attribute '{entry['coord_attr']}' not available")

    idx_attr = entry["indices_attr"]
    if include_guards and entry.get("guard_indices_attr"):
        idx_attr = entry["guard_indices_attr"]
    indices = getattr(grid, idx_attr, None)
    if indices is None or len(indices) == 0:
        raise ValueError(f"Grid attribute '{idx_attr}' not set or empty.")

    idx_1d = np.asarray(indices, dtype=np.intp).ravel()
    valid = idx_1d > 0
    idx_1d = idx_1d[valid] - 1
    max_idx = grid.n_faces if entry["is_face"] else grid.n_cells
    idx_1d = idx_1d[idx_1d < max_idx]

    x = np.asarray(coord, dtype=np.float64).ravel()[idx_1d]
    y = data.ravel()[idx_1d]
    valid_xy = np.isfinite(x) & np.isfinite(y)
    return x[valid_xy], y[valid_xy], entry["xlabel"], ylabel


def _extract_along_ft(grid, data, ft_num, ylabel):
    if grid.ft_cv_p is None or grid.ft_cv is None:
        raise ValueError("Flux tube data not available in grid")
    if ft_num < 1 or ft_num > grid.n_flux_tubes:
        raise ValueError(f"Flux tube {ft_num} out of range [1, {grid.n_flux_tubes}]")
    start = grid.ft_cv_p[ft_num - 1, 0]
    count = grid.ft_cv_p[ft_num - 1, 1]
    indices = grid.ft_cv[start:start + count].astype(np.intp) - 1
    x = (grid.cv_theta[indices] if grid.cv_theta is not None
         else np.arange(len(indices), dtype=np.float64))
    y = data[indices]
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid], "Poloidal distance [m]", ylabel


def _extract_wall(grid, data, ylabel, boundary=1, x_unit="m"):
    """Extract profile along a wall/limiter boundary segment."""
    closed_cvs = getattr(grid, "closed_stuct_cvs", None)
    if closed_cvs is None or not isinstance(closed_cvs, (list, tuple)):
        raise ValueError(
            "Wall data not available — grid lacks closed_stuct_cvs. "
            "Run watch.compute_regions() first."
        )
    if boundary < 1 or boundary > len(closed_cvs):
        raise ValueError(
            f"Boundary index {boundary} out of range [1, {len(closed_cvs)}]"
        )
    cvs = closed_cvs[boundary - 1]
    cvs = np.asarray(cvs, dtype=np.intp).ravel()
    valid = cvs > 0
    cvs = cvs[valid] - 1
    cvs = cvs[cvs < grid.n_cells]

    coord = grid.cv_lbl_len
    if coord is None:
        raise ValueError("Grid attribute 'cv_lbl_len' not available")

    x = np.asarray(coord, dtype=np.float64).ravel()[cvs]
    y = data.ravel()[cvs]
    if x_unit == "cm":
        x = x * 100.0
        xlabel = "Distance along wall [cm]"
    else:
        xlabel = "Distance along wall [m]"

    valid_xy = np.isfinite(x) & np.isfinite(y)
    return x[valid_xy], y[valid_xy], xlabel, ylabel


# ── Registry entries ─────────────────────────────────────────────
EXTRACTOR_REGISTRY["omp"] = {
    "coord_attr": "cv_r", "indices_attr": "outer_midplane_cells",
    "xlabel": "R [m]", "is_face": False, "description": "Outer midplane",
}
EXTRACTOR_REGISTRY["imp"] = {
    "coord_attr": "cv_r", "indices_attr": "inner_midplane_cells",
    "xlabel": "R [m]", "is_face": False, "description": "Inner midplane",
}
EXTRACTOR_REGISTRY["target_in"] = {
    "coord_attr": "cv_lbl_len", "indices_attr": "inner_target_cells",
    "xlabel": "Distance along target [m]", "is_face": False,
    "description": "Inner divertor target",
}
EXTRACTOR_REGISTRY["target_out"] = {
    "coord_attr": "cv_lbl_len", "indices_attr": "outer_target_cells",
    "xlabel": "Distance along target [m]", "is_face": False,
    "description": "Outer divertor target",
}
EXTRACTOR_REGISTRY["wall"] = {
    "coord_attr": "cv_lbl_len", "indices_attr": "wall_cells",
    "xlabel": "Distance along wall [m]", "is_face": False,
    "description": "Wall boundary",
}
EXTRACTOR_REGISTRY["poloidal"] = {
    "coord_attr": "cv_theta", "indices_attr": "outer_midplane_cells",
    "xlabel": "Poloidal distance [m]", "is_face": False,
    "description": "Poloidal profile",
}


def list_extractors() -> list[str]:
    return sorted(EXTRACTOR_REGISTRY.keys())


def register_extractor(name: str, **kwargs) -> None:
    EXTRACTOR_REGISTRY[name] = kwargs
