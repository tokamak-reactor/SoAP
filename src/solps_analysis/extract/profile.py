"""1D profile extraction from SolpsWatch."""
from __future__ import annotations
from typing import Any
import numpy as np
from solps_analysis.core.grid import GridTopology
from solps_analysis.core.dataset import SolpsWatch
from solps_analysis.core.variable import SolpsVariable, VariableMeta

EXTRACTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def _require_regions(grid: GridTopology, needed: list[str]) -> None:
    missing = [a for a in needed if getattr(grid, a, None) is None]
    if missing:
        raise ValueError(
            f"Region data not computed. Missing grid attributes: {missing}\n"
            f"Run watch.compute_regions() first."
        )


def _resolve_variable(watch: SolpsWatch, variable: str) -> SolpsVariable:
    """Resolve a variable name to a SolpsVariable.

    Lookup order:
      1. watch.get(name)            — raw .dat variables (b2nph9_te, …)
      2. watch.construct(name)      — derived quantities (te_sep, E_r, …)
      3. workspace (build_workspace) — MATLAB-style names (te, po, …)
    """
    var = watch.get(variable)
    if var is not None:
        return var
    var = watch.construct(variable)
    if var is not None:
        return var
    from solps_analysis.io.matlab_vars import build_workspace
    ws = build_workspace(watch)
    data = ws.get(variable)
    if data is not None:
        return SolpsVariable(
            data=np.asarray(data, dtype=np.float64),
            meta=VariableMeta(name=variable, unit=_guess_unit(variable)),
        )
    raise ValueError(
        f"Variable '{variable}' not found: not a raw .dat variable, "
        "not a registered quantity, and not in the MATLAB workspace. "
        "Use watch.list_variables() / list_quantities() to see names."
    )


_UNITS = {
    "te": "eV", "ti": "eV", "tn": "eV", "ne": "m⁻³", "po": "V", "ue": "m/s",
    "Zeff": "", "she": "W/m³", "shi": "W/m³", "na": "m⁻³", "ua": "m/s",
}


def _guess_unit(name: str) -> str:
    base = name.split("_")[0]
    return _UNITS.get(base, _UNITS.get(name, ""))


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
    var = _resolve_variable(watch, variable)
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
        # is_structured flag LIES for some WG grids (imap_cv.ndim==1) —
        # check the mapping instead.
        if grid.imap_cv is not None and grid.imap_cv.ndim == 2:
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
    xlabel = entry["xlabel"]
    if coord is None:
        if entry["coord_attr"] in ("cv_r", "cv_theta") and grid.cv_x is not None:
            r = np.sqrt(grid.cv_x ** 2 + grid.cv_y ** 2)
            if entry["coord_attr"] == "cv_r":
                coord = r
                setattr(grid, "cv_r", r)
            else:
                coord = np.arctan2(grid.cv_y, grid.cv_x)
                setattr(grid, "cv_theta", coord)
        elif entry["coord_attr"] == "cv_lbl_len" and grid.cv_r is not None:
            # Structured grids have no cv_lbl_len (wall/plate-length is a
            # WG concept). MATLAB_SPb plots target profiles on the same y2
            # (distance from separatrix) — use cv_r as the natural fallback.
            coord = grid.cv_r
            xlabel = "r − r_sep [m]"
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
    return x[valid_xy], y[valid_xy], xlabel, ylabel


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
    """Extract profile along the wall (unstructured grids).

    Uses grid.wall_cells / wall_cells_len computed by regions.compute_wall
    (fcLbl segments 5-8). ``boundary`` selects a wall segment — currently
    only the full wall (boundary=1) is supported.
    """
    wall_cells = getattr(grid, "wall_cells", None)
    wall_len = getattr(grid, "wall_cells_len", None)
    if wall_cells is None or len(wall_cells) == 0:
        raise ValueError(
            "Wall data not available — grid lacks wall_cells. "
            "Run watch.compute_regions() first (unstructured grid required)."
        )
    if boundary != 1:
        raise ValueError(
            f"Wall segment {boundary} not supported yet — only the full "
            "wall (boundary=1) is available."
        )
    cvs = np.asarray(wall_cells, dtype=np.intp).ravel()
    cvs = cvs[cvs < grid.n_cells]

    # Use the FULL cv_lbl_len array (n_cells) for indexing; wall_cells_len
    # is only a sub-selection for reporting.
    if grid.cv_lbl_len is not None:
        coord = np.asarray(grid.cv_lbl_len, dtype=np.float64).ravel()
    else:
        raise ValueError("Grid attribute 'cv_lbl_len' not available")

    x = coord[cvs]
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
    "xlabel": "r − r_sep [m]", "is_face": False, "description": "Outer midplane",
}
EXTRACTOR_REGISTRY["imp"] = {
    "coord_attr": "cv_r", "indices_attr": "inner_midplane_cells",
    "xlabel": "r − r_sep [m]", "is_face": False, "description": "Inner midplane",
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


def find_flux_tube_by_r(watch: SolpsWatch, r_target: float) -> int:
    """Find the flux tube number closest to a given R coordinate on the OMP.

    Args:
        watch: Loaded SolpsWatch (regions must be computed).
        r_target: Target R coordinate on the outer midplane [m].

    Returns:
        Flux tube number (1-indexed) closest to *r_target*.
    """
    _require_regions(watch.grid, ["outer_midplane_cells"])
    omp = watch.grid.outer_midplane_cells
    if omp is None or len(omp) == 0:
        raise ValueError("OMP cells not available — run watch.compute_regions() first.")
    omp = np.asarray(omp, dtype=np.intp).ravel()
    omp = omp[omp > 0] - 1  # 1-based → 0-based

    r_omp = watch.grid.cv_r[omp] if watch.grid.cv_r is not None else _cv_r_fallback(watch.grid)
    i_best = omp[np.argmin(np.abs(r_omp - r_target))]

    ft = watch.grid.cv_ft
    if ft is None:
        raise ValueError("Grid attribute 'cv_ft' not available.")
    return int(ft[i_best])


def _cv_r_fallback(grid) -> np.ndarray:
    r = np.sqrt(grid.cv_x ** 2 + grid.cv_y ** 2)
    setattr(grid, "cv_r", r)
    return r


def extract_profile_ns(
    watch: SolpsWatch,
    variable: str,
    along: str = "omp",
    element: str | None = None,
    include_guards: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Extract 1D profiles for each charge state of an element.

    Args:
        watch: Loaded SolpsWatch.
        variable: Variable name with species dimension (e.g. ``"na"``).
        along: Extraction path.
        element: Element symbol (e.g. ``"C"``, ``"D"``). If None, all species.
        include_guards: Passed through to ``extract_profile``.

    Returns:
        List of ``(x, y, label)`` tuples, one per charge state.
    """
    var = watch.get(variable)
    if var is None:
        raise ValueError(f"Variable '{variable}' not found in watch")
    if var.data.ndim < 2:
        raise ValueError(f"Variable '{variable}' is not multi-species (ndim={var.data.ndim})")

    ns = var.data.shape[1]
    comp = watch.b2_comp

    # Determine which columns to plot
    if element is not None:
        if comp is None:
            raise ValueError("B2 composition not available — needed for element lookup.")
        idx = comp.element_indices(element)
        if not idx:
            raise ValueError(f"Element '{element}' not found in composition.")
        charge_states = [int(comp.zamax[i]) for i in idx]
        col_range = list(enumerate(idx))
        names = [f"{element}+{z}" if z > 0 else str(element) for z in charge_states]
    else:
        col_range = list(enumerate(range(ns)))
        if comp is not None:
            names = []
            for c in range(ns):
                z = int(comp.zamax[c])
                # Find element name
                elem_name = None
                for en in comp.element_names:
                    if c in comp.element_indices(en):
                        elem_name = en
                        break
                names.append(f"{elem_name}+{z}" if elem_name else f"s{c} z={z}")
        else:
            names = [f"species {c}" for c in range(ns)]

    results: list[tuple[np.ndarray, np.ndarray, str]] = []
    for col_idx, spec_idx in col_range:
        x, y, _, _ = extract_profile(
            watch, variable, along=along,
            species=spec_idx, include_guards=include_guards,
        )
        results.append((x, y, names[col_idx]))

    return results
