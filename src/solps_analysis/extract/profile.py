"""1D profile extraction from SolpsWatch.

Extractors convert mesh data to (x, y) vectors along specified paths
(OMP, IMP, targets, flux tubes, boundary). This separates the geometry
logic from plotting — plotters just draw (x, y) lines.

Usage::

    from solps_analysis.extract import extract_profile

    x, y, xlabel, ylabel = extract_profile(
        watch, "te_eV", along="omp", include_guards=False
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np

from solps_analysis.core.grid import GridTopology
from solps_analysis.core.dataset import SolpsWatch


# ──────────────────────────────────────────────
# Registry of extractors
# ──────────────────────────────────────────────

EXTRACTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def _ensure_regions(grid: GridTopology) -> None:
    """Compute regions on the grid if not already done."""
    if grid.outer_midplane_cells is not None:
        return

    if grid.is_structured:
        from solps_analysis.core.regions_structured import compute_regions_structured
        result = compute_regions_structured(grid)
    else:
        from solps_analysis.core.regions import compute_all_regions
        result = compute_all_regions(grid)

    if result:
        for key, val in result.items():
            setattr(grid, key, val)


def extract_profile(
    watch: SolpsWatch,
    variable: str,
    along: str = "omp",
    include_guards: bool = False,
    species: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Extract a 1D profile from a watch along a specified path.

    Args:
        watch: Loaded SolpsWatch.
        variable: Variable name (e.g. ``"te_eV"``, ``"ne"``).
        along: Extraction path — ``"omp"``, ``"imp"``, ``"target_in"``,
            ``"target_out"``, or ``"ft:<N>`` for flux tube N.
        include_guards: If True, include boundary/guard cells.
        species: Species index for multi-species variables.

    Returns:
        Tuple ``(x, y, xlabel, ylabel)``.
    """
    _ensure_regions(watch.grid)

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

    # ── flux tube ─────────────────────────────────────────────────
    if along.startswith("ft:"):
        ft_num = int(along.split(":")[1])
        return _extract_along_ft(watch.grid, data, ft_num, ylabel)

    # ── registry lookup ───────────────────────────────────────────
    if along not in EXTRACTOR_REGISTRY:
        known = ", ".join(sorted(EXTRACTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown extraction path '{along}'. Known: {known}")

    entry = EXTRACTOR_REGISTRY[along]
    grid = watch.grid

    coord = getattr(grid, entry["coord_attr"], None)
    if coord is None:
        # Fallback: compute cv_r from cv_x, cv_y
        if entry["coord_attr"] in ("cv_r", "cv_theta") and grid.cv_x is not None:
            r = np.sqrt(grid.cv_x ** 2 + grid.cv_y ** 2)
            theta = np.arctan2(grid.cv_y, grid.cv_x)
            if entry["coord_attr"] == "cv_r":
                coord = r
                setattr(grid, "cv_r", r)
            else:
                coord = theta
                setattr(grid, "cv_theta", theta)
        else:
            raise ValueError(
                f"Grid attribute '{entry['coord_attr']}' not available"
            )

    idx_attr = entry["indices_attr"]
    if include_guards and entry.get("guard_indices_attr"):
        idx_attr = entry["guard_indices_attr"]

    indices = getattr(grid, idx_attr, None)
    if indices is None or len(indices) == 0:
        raise ValueError(
            f"Grid attribute '{idx_attr}' not set or empty."
        )

    idx_1d = np.asarray(indices, dtype=np.intp).ravel()
    valid = idx_1d > 0
    idx_1d = idx_1d[valid] - 1  # 1-based → 0-based

    max_idx = grid.n_faces if entry["is_face"] else grid.n_cells
    idx_1d = idx_1d[idx_1d < max_idx]

    x = np.asarray(coord, dtype=np.float64).ravel()[idx_1d]
    y = data.ravel()[idx_1d]

    valid_xy = np.isfinite(x) & np.isfinite(y)
    return x[valid_xy], y[valid_xy], entry["xlabel"], ylabel


def _extract_along_ft(
    grid: GridTopology,
    data: np.ndarray,
    ft_num: int,
    ylabel: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Extract data along a flux tube by number."""
    if grid.ft_cv_p is None or grid.ft_cv is None:
        raise ValueError("Flux tube data not available in grid")
    if ft_num < 1 or ft_num > grid.n_flux_tubes:
        raise ValueError(f"Flux tube {ft_num} out of range [1, {grid.n_flux_tubes}]")

    start = grid.ft_cv_p[ft_num - 1, 0]
    count = grid.ft_cv_p[ft_num - 1, 1]
    indices = grid.ft_cv[start:start + count].astype(np.intp) - 1

    x = grid.cv_theta[indices] if grid.cv_theta is not None else np.arange(len(indices), dtype=np.float64)
    y = data[indices]

    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid], "Poloidal distance [m]", ylabel


# ──────────────────────────────────────────────
# Built-in extractors
# ──────────────────────────────────────────────

# OMP / IMP
EXTRACTOR_REGISTRY["omp"] = {
    "coord_attr": "cv_r", "indices_attr": "outer_midplane_cells",
    "xlabel": "R [m]", "is_face": False,
    "description": "Outer midplane",
}
EXTRACTOR_REGISTRY["imp"] = {
    "coord_attr": "cv_r", "indices_attr": "inner_midplane_cells",
    "xlabel": "R [m]", "is_face": False,
    "description": "Inner midplane",
}
# Targets
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
# Wall
EXTRACTOR_REGISTRY["wall"] = {
    "coord_attr": "cv_lbl_len", "indices_attr": "wall_cells",
    "xlabel": "Distance along wall [m]", "is_face": False,
    "description": "Wall boundary",
}
# Poloidal (along a flux tube) — special, handled by "ft:N" prefix
EXTRACTOR_REGISTRY["poloidal"] = {
    "coord_attr": "cv_theta", "indices_attr": "outer_midplane_cells",
    "xlabel": "Poloidal distance [m]", "is_face": False,
    "description": "Poloidal profile (default OMP cells)",
}


def list_extractors() -> list[str]:
    """List all registered extraction path names."""
    return sorted(EXTRACTOR_REGISTRY.keys())


def register_extractor(name: str, **kwargs) -> None:
    """Register a custom extraction path."""
    EXTRACTOR_REGISTRY[name] = kwargs
