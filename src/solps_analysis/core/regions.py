"""Region computation for SOLPS-ITER GridTopology.

Replicates functionality of MATLAB's read_geometry.m (region part)
using vectorized numpy operations and graph algorithms.

The key concept: for an unstructured grid, the radial direction follows
the poloidal magnetic field — cells in the same 'radial column' are
connected via faces belonging to different flux tubes.
Poloidal direction follows along flux tubes.

All functions operate on GridTopology in-place and also return a regions dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order

from solps_analysis.core.grid import GridTopology
from solps_analysis.core.regions_structured import compute_regions_structured as _compute_regions_structured


# =========================================================================
# Radial neighbor column via graph traversal
# =========================================================================

def _build_radial_adjacency(grid: GridTopology) -> list[np.ndarray]:
    """Build radial adjacency list for all cells.

    Two cells are radial neighbors if they share a face AND belong to
    different flux tubes (cv_ft differs) AND share no common vertex.
    
    Returns a list of numpy arrays: radial_neighbors[i] = array of radial
    neighbor indices for cell i.
    """
    nCv = grid.n_cells
    cv_ft = grid.cv_ft
    n_core = grid.n_core_cells

    # Pre-compute per-cell vertex sets as python sets for fast intersection
    # cv_vx_p contains (start, count), cv_vx contains the flat vertex list
    cv_vx_sets = []
    for i in range(nCv):
        s = grid.cv_vx_p[i, 0]
        c = grid.cv_vx_p[i, 1]
        cv_vx_sets.append(set(grid.cv_vx[s:s + c].tolist()))

    # Build adjacency: for each cell, find neighbors via shared face
    # that are in a different flux tube
    radial_adj = [np.array([], dtype=np.int32) for _ in range(nCv)]

    for iCv in range(nCv):
        start = grid.cv_fc_p[iCv, 0]
        count = grid.cv_fc_p[iCv, 1]
        faces = grid.cv_fc[start:start + count]

        # Get all neighboring cells via these faces
        all_neighbors = []
        for fc in faces:
            cells = grid.fc_cv[fc]
            other = cells[cells != iCv]
            if len(other) > 0:
                all_neighbors.append(other[0])

        if not all_neighbors:
            continue

        all_neighbors = np.array(all_neighbors, dtype=np.int32)
        ft_icv = cv_ft[iCv]

        # Filter: keep only neighbors in different flux tubes
        diff_mask = cv_ft[all_neighbors] != ft_icv
        diff_ft = all_neighbors[diff_mask]

        if len(diff_ft) < 2:
            if len(diff_ft) == 1:
                radial_adj[iCv] = diff_ft
            else:
                # Fallback: boundary neighbor
                bound = diff_ft[diff_ft >= n_core]
                if len(bound) > 0:
                    radial_adj[iCv] = bound[:1]
            continue

        # Among diff_ft, find pairs with no common vertices (= radial pair)
        found = False
        for k in range(len(diff_ft)):
            icv1 = int(diff_ft[k])
            vx1 = cv_vx_sets[icv1]
            for j in range(k + 1, len(diff_ft)):
                icv2 = int(diff_ft[j])
                # If no shared vertices → radial pair
                if not vx1.intersection(cv_vx_sets[icv2]):
                    radial_adj[iCv] = np.array([icv1, icv2], dtype=np.int32)
                    found = True
                    break
            if found:
                break

        if not found:
            bound = diff_ft[diff_ft >= n_core]
            if len(bound) > 0:
                radial_adj[iCv] = bound[:1]

    return radial_adj


def get_radial_column(cell_index: int, radial_adj: list[np.ndarray],
                      n_core: int) -> np.ndarray:
    """Walk both directions along radial_adj to get the full radial column.
    
    Returns core cells sorted with boundary cell (if any) at position 0.
    """
    visited = {cell_index}
    queue = [cell_index]

    # Walk in both directions via BFS limited to radial adjacency
    while queue:
        current = queue.pop(0)
        for nb in radial_adj[current]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    arr = np.array(list(visited), dtype=np.int32)
    core = arr[arr <= n_core]
    bound = arr[arr > n_core]

    # Sort core by index
    core = np.sort(core)

    if len(bound) > 0:
        result = np.empty(len(core) + 1, dtype=np.int32)
        result[0] = int(bound.min())
        result[1:] = core
        return result
    return core


# =========================================================================
# 1. Face coordinates — vectorized
# =========================================================================

def compute_face_coordinates(grid: GridTopology) -> None:
    """Compute face center coordinates from vertex data — vectorized.
    
    fc_x[i] = 0.5 * (vx_x[fc_vx[i,0]] + vx_x[fc_vx[i,1]])
    """
    grid.fc_x = 0.5 * (grid.vx_x[grid.fc_vx[:, 0]] + grid.vx_x[grid.fc_vx[:, 1]])
    grid.fc_y = 0.5 * (grid.vx_y[grid.fc_vx[:, 0]] + grid.vx_y[grid.fc_vx[:, 1]])


# =========================================================================
# 2. Radial coordinate — walk radial columns, compute distance
# =========================================================================

def compute_radial_coordinate(grid: GridTopology,
                               radial_adj: list[np.ndarray] | None = None) -> None:
    """Compute radial coordinate cv_r for all cells by walking radial neighbors.
    
    If radial_adj is not provided, builds it first.
    """
    if radial_adj is None:
        radial_adj = _build_radial_adjacency(grid)

    nCv = grid.n_cells
    cv_r = np.zeros(nCv, dtype=np.float64)
    visited = np.zeros(nCv, dtype=bool)

    for iCv in range(nCv):
        if visited[iCv] or len(radial_adj[iCv]) == 0:
            continue

        col = get_radial_column(iCv, radial_adj, grid.n_core_cells)
        visited[col] = True

        if len(col) < 2:
            continue

        # Compute cumulative distance along the column
        dist = np.zeros(len(col), dtype=np.float64)
        for k in range(1, len(col)):
            dx = grid.cv_x[col[k]] - grid.cv_x[col[k - 1]]
            dy = grid.cv_y[col[k]] - grid.cv_y[col[k - 1]]
            dist[k] = dist[k - 1] + np.sqrt(dx ** 2 + dy ** 2)

        cv_r[col] = dist

    grid.cv_r = cv_r


# =========================================================================
# 3. Poloidal coordinate — along flux tubes
# =========================================================================

def compute_poloidal_coordinate(grid: GridTopology) -> None:
    """Compute poloidal coordinate cv_theta along flux tubes."""
    if grid.ft_cv_p is None or grid.ft_cv is None:
        return
    nCv = grid.n_cells
    cv_theta = np.zeros(nCv, dtype=np.float64)
    ft_cv: np.ndarray = grid.ft_cv
    ft_cv_p: np.ndarray = grid.ft_cv_p

    for iFt in range(grid.n_flux_tubes):
        start = int(ft_cv_p[iFt, 0])
        count = int(ft_cv_p[iFt, 1])
        # Guard: ft_cv_p may overcount the last tube (structured build bug)
        count = min(count, len(ft_cv) - start)
        if count < 2:
            continue
        cvs = ft_cv[start:start + count]

        dist = np.zeros(count, dtype=np.float64)
        for k in range(1, count):
            dx = grid.cv_x[cvs[k]] - grid.cv_x[cvs[k - 1]]
            dy = grid.cv_y[cvs[k]] - grid.cv_y[cvs[k - 1]]
            dist[k] = dist[k - 1] + np.sqrt(dx ** 2 + dy ** 2)

        cv_theta[cvs] = dist

    grid.cv_theta = cv_theta


# =========================================================================
# 4. Flux tube connection length
# =========================================================================

def compute_flux_tube_connection(grid: GridTopology) -> None:
    """Compute connection length for each flux tube."""
    ft_conn = np.zeros(grid.n_flux_tubes, dtype=np.float64)

    for iFt in range(grid.n_flux_tubes):
        start = grid.ft_fc_p[iFt, 0]
        count = grid.ft_fc_p[iFt, 1]
        if count == 0:
            continue
        fcs = grid.ft_fc[start:start + count]
        # ft_conn = sum over faces of fcHc * fcBb[:,4] / fcBb[:,1]
        total = 0.0
        for iFc in fcs:
            total += np.sum(grid.fc_hc[iFc]) * grid.fc_bb[iFc, 3] / grid.fc_bb[iFc, 0]
        ft_conn[iFt] = total

    grid.ft_conn = ft_conn


# =========================================================================
# 5. Midplane identification
# =========================================================================

def find_midplanes(grid: GridTopology) -> None:
    """Find inner and outer midplane cells via flux tube region 1 (core).
    
    The first core flux tube spans the separatrix.
    Cell with max Bz → outer midplane, min Bz → inner midplane.
    """
    if grid.ft_cv_p is None:
        return

    # Find first flux tube in core region (ftReg == 1)
    iFt = 0
    while iFt < grid.n_flux_tubes:
        if grid.ft_reg is not None and iFt < len(grid.ft_reg):
            if grid.ft_reg[iFt] == 1:
                break
        iFt += 1

    if iFt >= grid.n_flux_tubes:
        return

    # Cells in the first core flux tube
    start = grid.ft_cv_p[iFt, 0]
    count = grid.ft_cv_p[iFt, 1]
    cvs = grid.ft_cv[start:start + count]

    if len(cvs) == 0:
        return

    # Use Bz to determine inner vs outer midplane
    bz = grid.cv_bb[cvs, 2] if grid.cv_bb is not None else np.zeros(len(cvs))

    if np.any(np.abs(bz) > 1e-6):
        nin = cvs[np.argmax(np.abs(bz))]
        nout = cvs[np.argmin(np.abs(bz))]
    else:
        x_coord = grid.cv_x[cvs]
        nin = cvs[np.argmin(x_coord)]
        nout = cvs[np.argmax(x_coord)]

    # Walk radial neighbors to get full midplane columns
    rad_adj = _build_radial_adjacency(grid)
    imp_col = get_radial_column(int(nin), rad_adj, grid.n_core_cells)
    omp_col = get_radial_column(int(nout), rad_adj, grid.n_core_cells)

    grid.inner_midplane_cells = imp_col
    grid.outer_midplane_cells = omp_col
    grid.inner_midplane_cells_sol = imp_col[grid.cv_r[imp_col] > 0] if hasattr(grid, 'cv_r') and grid.cv_r is not None else imp_col
    grid.outer_midplane_cells_sol = omp_col[grid.cv_r[omp_col] > 0] if hasattr(grid, 'cv_r') and grid.cv_r is not None else omp_col

    # Face midplane coordinates
    grid.fc_inner_midplane = _find_midplane_faces(grid, imp_col)
    grid.fc_outer_midplane = _find_midplane_faces(grid, omp_col)


def _find_midplane_faces(grid: GridTopology, cells: np.ndarray) -> np.ndarray:
    """Find faces between adjacent midplane cells."""
    faces = []
    for k in range(1, len(cells)):
        prev = cells[k - 1]
        cur = cells[k]
        s1 = grid.cv_fc_p[prev, 0]
        c1 = grid.cv_fc_p[prev, 1]
        s2 = grid.cv_fc_p[cur, 0]
        c2 = grid.cv_fc_p[cur, 1]
        common = np.intersect1d(grid.cv_fc[s1:s1 + c1], grid.cv_fc[s2:s2 + c2])
        if len(common) > 0:
            faces.append(int(common[0]))
    return np.array(faces, dtype=np.int32) if faces else np.array([], dtype=np.int32)


# =========================================================================
# 6. X-point detection and separatrix
# =========================================================================

def find_xpoints_and_separatrices(grid: GridTopology) -> None:
    """Find X-points (vertices with 4 field-aligned faces) and separatrix flux surfaces.
    
    An X-point is a vertex whose surrounding faces all have fc_qalf ≈ 0
    (field-aligned faces). The separatrix connects to the X-point.
    """
    if grid.fc_qalf is None or grid.vx_fc_p is None:
        return

    nVx = grid.n_vertices
    xp_list = []

    for iVx in range(nVx):
        start = grid.vx_fc_p[iVx, 0]
        count = grid.vx_fc_p[iVx, 1]
        if count == 0:
            continue
        fcs = grid.vx_fc[start:start + count]
        # Field-aligned faces have fc_qalf[:, 0] ≈ 0 (cos(alpha) ≈ 0)
        fa_faces = fcs[np.abs(grid.fc_qalf[fcs, 0]) < 1e-6]
        if len(fa_faces) == 4:
            xp_list.append(iVx)

    grid.xp_vx = np.array(xp_list, dtype=np.int32) if xp_list else np.array([], dtype=np.int32)

    if len(xp_list) == 0:
        return

    # Compute fcFt and fcFs arrays (needed for separatrix search)
    grid.fc_ft = np.zeros(grid.n_faces, dtype=np.int32)
    grid.fc_fs = np.zeros(grid.n_faces, dtype=np.int32)
    for iFt in range(grid.n_flux_tubes):
        start = grid.ft_fc_p[iFt, 0]
        count = grid.ft_fc_p[iFt, 1]
        if count > 0:
            grid.fc_ft[grid.ft_fc[start:start + count]] = iFt + 1
    for iFs in range(grid.n_flux_surfaces):
        start = grid.fs_fc_p[iFs, 0]
        count = grid.fs_fc_p[iFs, 1]
        if count > 0:
            grid.fc_fs[grid.fs_fc[start:start + count]] = iFs + 1

    # Find separatrix flux surfaces from X-point's field-aligned faces
    seps = []
    for iVx in xp_list:
        start = grid.vx_fc_p[iVx, 0]
        count = grid.vx_fc_p[iVx, 1]
        fcs = grid.vx_fc[start:start + count]
        fa_faces = fcs[np.abs(grid.fc_qalf[fcs, 0]) < 1e-6]
        # Get the flux surfaces these faces belong to
        fs = np.unique(grid.fc_fs[fa_faces])
        seps.extend(fs.tolist())

    if seps:
        grid.fs_sep = np.array(sorted(set(seps)), dtype=np.int32)
    if len(xp_list) > 1 and hasattr(grid, 'fs_sep') and grid.fs_sep is not None:
        grid.fs_sep2 = grid.fs_sep.copy()


# =========================================================================
# 7. Target identification
# =========================================================================

def find_target_labels(grid, top: bool = False) -> tuple[int, int]:
    """Determine (inner, outer) target labels by face coordinates.

    Fallback for grids where the separatrix walk cannot classify targets
    (e.g. unstructured wide-grid runs where the separatrix touches only
    two of the four targets). Ported from calc_additional._target_labels
    with a fix: inner/outer are grouped by X (HFS vs LFS) rather than
    taking the two largest-|y| labels (which can both be LFS).

    Target labels = boundary labels with many faces (>=10) and large |y|
    (divertor targets), excluding core-boundary labels (-21/-25).
    top=False → lower targets (y < 0); top=True → upper targets (y > 0).
    Inner = smaller mean x (HFS), outer = larger mean x (LFS).
    Returns (inner_lbl, outer_lbl) in {-1, -1} if not determinable.
    """
    if grid.fc_lbl is None or grid.fc_x is None or grid.fc_y is None:
        return -1, -1
    lbls = np.unique(grid.fc_lbl[grid.fc_lbl != 0])
    stats = []
    for lbl in lbls:
        if lbl in (-21, -25):
            continue
        fcs = np.where(grid.fc_lbl == lbl)[0]
        if fcs.size < 10:
            continue  # drop tiny corner labels (e.g. 5, 7)
        if top:
            fcs = fcs[grid.fc_y[fcs] > 0]
        else:
            fcs = fcs[grid.fc_y[fcs] < 0]
        if fcs.size == 0:
            continue
        my = np.abs(grid.fc_y[fcs]).mean()
        mx = grid.fc_x[fcs].mean()
        stats.append((lbl, mx, my))
    if len(stats) < 2:
        return -1, -1
    stats = np.array(stats)
    # Split into HFS (small x) and LFS (large x) around the median x
    mid = np.median(stats[:, 1])
    hfs = stats[stats[:, 1] < mid]
    lfs = stats[stats[:, 1] > mid]
    if len(hfs) == 0 or len(lfs) == 0:
        return -1, -1
    inner_lbl = int(hfs[np.argmax(hfs[:, 2]), 0])
    outer_lbl = int(lfs[np.argmax(lfs[:, 2]), 0])
    return inner_lbl, outer_lbl


def find_targets(grid: GridTopology) -> None:
    """Identify inner/outer, upper/lower, active/inactive targets.
    
    Starting from the separatrix, walks boundary faces and classifies
    by Y-coordinate (upper/lower) and X-coordinate (inner/outer).
    Falls back to label-geometry classification (find_target_labels)
    when the separatrix walk touches only some of the targets.
    """
    if not hasattr(grid, 'xp_vx') or grid.xp_vx is None or len(grid.xp_vx) == 0:
        return

    from solps_analysis.core.regions import find_sep_fc_vx
    sep_vx_all = []
    sep_fc_all = []
    fs_sep = np.atleast_1d(grid.fs_sep)
    xp_list = np.atleast_1d(grid.xp_vx)
    for xp in xp_list:
        for fs in fs_sep:
            sv, sf = find_sep_fc_vx(grid, np.array([fs]), int(xp))
            sep_vx_all.extend(sv.tolist())
            sep_fc_all.extend(sf.tolist())
    sep_vx = np.unique(np.asarray(sep_vx_all, dtype=np.int32))
    sep_fc = np.unique(np.asarray(sep_fc_all, dtype=np.int32))
    grid.sep_vx = sep_vx
    grid.sep_fc = sep_fc

    # Find target labels from separatrix vertices
    lbl_targets = []
    for ivx in sep_vx:
        start = grid.vx_fc_p[ivx, 0]
        count = grid.vx_fc_p[ivx, 1]
        if count == 0:
            continue
        fcs = grid.vx_fc[start:start + count]
        lbl_fcs = fcs[grid.fc_lbl[fcs] != 0]
        if len(lbl_fcs) > 0:
            lbl = grid.fc_lbl[lbl_fcs[0]]
            lbl_targets.append((lbl, grid.fc_x[lbl_fcs[0]], grid.fc_y[lbl_fcs[0]],
                                grid.fc_bb[lbl_fcs[0], 2]))

    if not lbl_targets:
        return

    lbl_targets = np.array(lbl_targets)

    # Classify targets by Y sign and X min/max
    upper_mask = lbl_targets[:, 2] > 0
    lower_mask = ~upper_mask

    inner_top = 0
    outer_top = 0
    inner_tar = 0
    outer_tar = 0

    if np.any(upper_mask):
        up = lbl_targets[upper_mask]
        inner_top = int(up[np.argmin(up[:, 1]), 0])
        outer_top = int(up[np.argmax(up[:, 1]), 0])

    if np.any(lower_mask):
        low = lbl_targets[lower_mask]
        inner_tar = int(low[np.argmin(low[:, 1]), 0])
        outer_tar = int(low[np.argmax(low[:, 1]), 0])

    # Fallback: label-geometry classification when the separatrix walk
    # produced an invalid/partial (inner, outer) pair (wide-grid runs where
    # the separatrix touches only some targets). Re-derive BOTH when the
    # pair is degenerate (0, or both equal).
    if inner_tar == 0 or outer_tar == 0 or inner_tar == outer_tar:
        fi, fo = find_target_labels(grid, top=False)
        if fi > 0 and fo > 0 and fi != fo:
            inner_tar, outer_tar = fi, fo
    if inner_top == 0 or outer_top == 0 or inner_top == outer_top:
        fi, fo = find_target_labels(grid, top=True)
        if fi > 0 and fo > 0 and fi != fo:
            inner_top, outer_top = fi, fo

    # Active/inactive: same side as X-point
    xp_y = grid.vx_y[grid.xp_vx[0]] if len(grid.xp_vx) > 0 else 0

    _assign_target(grid, inner_tar, outer_tar, inner_top, outer_top, xp_y)


def _assign_target(grid, inner_tar, outer_tar, inner_top, outer_top, xp_y):
    """Assign target cell/face indices to grid attributes.

    cv_*_tar hold the PHYSICAL cells adjacent to the plate (core side,
    cv <= n_core_cells) — same semantics as the structured grid (matrix
    rows). Guard cells are excluded (cv_ft == 0, useless for flux-tube
    queries). Legacy aliases cv_vol_* are kept pointing at the same cells.
    """
    bound_nums = np.unique(grid.fc_lbl[grid.fc_lbl != 0])

    for lbl in bound_nums:
        fc_lbl = np.where(grid.fc_lbl == lbl)[0]
        if len(fc_lbl) == 0:
            continue
        cv_lbl = np.unique(grid.fc_cv[fc_lbl].ravel())
        cv_vol = cv_lbl[cv_lbl <= grid.n_core_cells]

        if lbl == inner_tar:
            grid.cv_inner_tar = cv_vol
            setattr(grid, 'cv_vol_inner_tar', cv_vol)
            grid.fc_inner_tar = fc_lbl
        elif lbl == outer_tar:
            grid.cv_outer_tar = cv_vol
            setattr(grid, 'cv_vol_outer_tar', cv_vol)
            grid.fc_outer_tar = fc_lbl
        elif lbl == inner_top and inner_top != 0 and inner_top != outer_top:
            setattr(grid, 'cv_inner_top_tar', cv_vol)
            setattr(grid, 'cv_vol_inner_top_tar', cv_vol)
            setattr(grid, 'fc_inner_top_tar', fc_lbl)
        elif lbl == outer_top and outer_top != 0 and inner_top != outer_top:
            setattr(grid, 'cv_outer_top_tar', cv_vol)
            setattr(grid, 'cv_vol_outer_top_tar', cv_vol)
            setattr(grid, 'fc_outer_top_tar', fc_lbl)


def find_sep_fc_vx(grid: GridTopology, fs_sep: np.ndarray,
                   xp_vx: int) -> tuple[np.ndarray, np.ndarray]:
    """Walk separatrix from X-point along flux surface fs_sep.
    
    Returns (sep_vertex_list, sep_face_list).
    """
    if grid.fs_fc_p is None:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Get faces on the separatrix flux surface
    sep_fs = fs_sep[0]
    start = grid.fs_fc_p[sep_fs, 0]
    count = grid.fs_fc_p[sep_fs, 1]
    fs_faces = grid.fs_fc[start:start + count]

    # Separate into two branches from X-point
    vx_faces = grid.vx_fc[grid.vx_fc_p[xp_vx, 0]:grid.vx_fc_p[xp_vx, 0] + grid.vx_fc_p[xp_vx, 1]]
    sep_at_vx = np.intersect1d(fs_faces, vx_faces)

    if len(sep_at_vx) < 2:
        return np.array([xp_vx], dtype=np.int32), sep_at_vx

    def _walk(start_fc: int) -> tuple[list, list]:
        """Walk one branch of the separatrix from start_fc."""
        faces = [int(start_fc)]
        vx = [xp_vx]
        curr_fc = int(start_fc)
        prev_vx = xp_vx
        for _ in range(len(fs_faces)):
            v1, v2 = grid.fc_vx[curr_fc]
            next_vx = v1 if v2 == prev_vx else v2
            vx.append(int(next_vx))
            vx_fcs = grid.vx_fc[grid.vx_fc_p[next_vx, 0]:grid.vx_fc_p[next_vx, 0] + grid.vx_fc_p[next_vx, 1]]
            next_fcs = np.setdiff1d(np.intersect1d(vx_fcs, fs_faces), faces)
            if len(next_fcs) == 0:
                break
            curr_fc = int(next_fcs[0])
            faces.append(curr_fc)
            prev_vx = next_vx
        return vx, faces

    all_vx = []
    all_faces = []
    for branch in (sep_at_vx[0], sep_at_vx[1]):
        bv, bf = _walk(int(branch))
        all_vx.extend(bv)
        all_faces.extend(bf)
    all_vx = list(dict.fromkeys(all_vx))  # unique, preserve order
    all_faces = list(dict.fromkeys(all_faces))

    return np.array(all_vx, dtype=np.int32), np.array(all_faces, dtype=np.int32)


# =========================================================================
# 8. Boundary coordinates
# =========================================================================

def compute_boundary_coordinates(grid: GridTopology) -> None:
    """Compute cvLbl_len and fcLbl_len — distance along boundary regions."""
    bound_nums = np.unique(grid.fc_lbl[grid.fc_lbl != 0])

    grid.cv_lbl_len = np.zeros(grid.n_cells, dtype=np.float64)
    grid.fc_lbl_len = np.zeros(grid.n_faces, dtype=np.float64)

    for lbl in bound_nums:
        lbl_fcs = np.where(grid.fc_lbl == lbl)[0]
        if len(lbl_fcs) < 2:
            continue

        # Sort faces by connectivity (walk along boundary)
        sorted_fcs = _sort_boundary_faces(grid, lbl_fcs)
        if len(sorted_fcs) < 2:
            continue

        # Compute cumulative distance
        dx = grid.fc_x[sorted_fcs[1:]] - grid.fc_x[sorted_fcs[:-1]]
        dy = grid.fc_y[sorted_fcs[1:]] - grid.fc_y[sorted_fcs[:-1]]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        cumdist = np.concatenate([[0], np.cumsum(dist)])

        grid.fc_lbl_len[sorted_fcs] = cumdist

        # Map face distances to cells
        cv_in_lbl = np.unique(grid.fc_cv[sorted_fcs].ravel())
        cv_in_lbl = cv_in_lbl[cv_in_lbl > 0]
        for cv in cv_in_lbl:
            cv_fcs = grid.cv_fc[grid.cv_fc_p[cv, 0]:grid.cv_fc_p[cv, 0] + grid.cv_fc_p[cv, 1]]
            common = np.intersect1d(cv_fcs, sorted_fcs)
            if len(common) > 0:
                grid.cv_lbl_len[cv] = np.mean(grid.fc_lbl_len[common])


def _sort_boundary_faces(grid: GridTopology, faces: np.ndarray) -> np.ndarray:
    """Sort boundary faces into a continuous chain by shared vertices."""
    if len(faces) < 2:
        return faces

    remaining = set(faces.tolist())
    sorted_fcs = [faces[0]]
    remaining.remove(faces[0])

    while remaining:
        last_vx = set(grid.fc_vx[sorted_fcs[-1]].tolist())
        best = None
        best_overlap = 0
        for fc in list(remaining):
            overlap = len(last_vx.intersection(grid.fc_vx[fc].tolist()))
            if overlap > best_overlap:
                best_overlap = overlap
                best = fc
                if overlap >= 1:
                    break
        if best is None:
            break
        sorted_fcs.append(best)
        remaining.remove(best)

    return np.array(sorted_fcs, dtype=np.int32)


# =========================================================================
# 9. Normalized poloidal coordinates
# =========================================================================

def compute_normalized_coordinates(grid: GridTopology) -> None:
    """Compute normalized poloidal coordinates (0→1 per flux tube/surface)."""
    if grid.ft_cv_p is None or grid.ft_cv is None:
        return
    grid.cv_theta_n = np.zeros(grid.n_cells, dtype=np.float64)
    grid.fc_theta_n = np.zeros(grid.n_faces, dtype=np.float64)
    ft_cv_p: np.ndarray = grid.ft_cv_p
    ft_fc_p = grid.ft_fc_p

    for iFt in range(grid.n_flux_tubes):
        start = int(ft_cv_p[iFt, 0])
        count = int(ft_cv_p[iFt, 1])
        count = min(count, len(grid.ft_cv) - start)  # guard overcount
        if count < 2:
            continue
        cvs = grid.ft_cv[start:start + count]
        th = grid.cv_theta[cvs]
        th_min, th_max = th.min(), th.max()
        if th_max > th_min:
            grid.cv_theta_n[cvs] = (th - th_min) / (th_max - th_min)

        # Face version
        start_f = int(ft_fc_p[iFt, 0]) if ft_fc_p is not None else 0
        count_f = int(ft_fc_p[iFt, 1]) if ft_fc_p is not None else 0
        count_f = min(count_f, len(grid.ft_fc) - start_f)  # guard overcount
        if count_f > 0 and hasattr(grid, 'fc_theta') and grid.fc_theta is not None:
            fcs = grid.ft_fc[start_f:start_f + count_f]
            th_f = grid.fc_theta[fcs]
            th_min_f, th_max_f = th_f.min(), th_f.max()
            if th_max_f > th_min_f:
                grid.fc_theta_n[fcs] = (th_f - th_min_f) / (th_max_f - th_min_f)

    # HFS/PFR sign marking
    if grid.cv_reg is not None and np.max(grid.cv_reg) > 4:
        for iFt in range(grid.n_flux_tubes):
            start = grid.ft_cv_p[iFt, 0]
            count = grid.ft_cv_p[iFt, 1]
            cvs = grid.ft_cv[start:start + count]
            ureg = np.unique(grid.cv_reg[cvs])
            ureg = ureg[ureg != 0]
            if not np.any(np.isin(ureg, [1, 5, 6, 7, 8])):
                grid.cv_theta_n[cvs] *= -1
            if not np.any(np.isin(ureg, [1, 2, 3, 5, 6, 8])):
                grid.cv_theta_n[cvs] *= -1


# =========================================================================
# 10. Shift coordinates relative to separatrix / OMP
# =========================================================================

def _shift_coordinates(grid: GridTopology) -> None:
    """Shift cv_theta to be 0 at OMP, cv_r to be 0 at separatrix.

    Port of matlab_wg read_geometry.m lines 495-530:
    - cvTheta: per poloidal column (flux tube), subtract the value at the
      OMP cell of that column → 0 at OMP.
    - cvR: per radial column, subtract the mean value at the separatrix
      cells of that column → 0 at separatrix (negative in core, positive
      in SOL).
    """
    if grid.cv_theta is None or grid.cv_r is None:
        return

    # --- cv_theta: shift per flux tube to OMP ---
    omp = getattr(grid, "outer_midplane_cells", None)
    if omp is not None and grid.ft_cv_p is not None:
        omp_set = set(np.asarray(omp, dtype=np.intp).ravel().tolist())  # 1-based
        ft_cv_p: np.ndarray = grid.ft_cv_p
        for iFt in range(grid.n_flux_tubes):
            start, count = int(ft_cv_p[iFt, 0]), int(ft_cv_p[iFt, 1])
            count = min(count, len(grid.ft_cv) - start)  # guard overcount
            if count < 2:
                continue
            cvs = grid.ft_cv[start:start + count]  # 0-based
            common = [int(c) + 1 for c in cvs if int(c) + 1 in omp_set]
            if common:
                grid.cv_theta[cvs] -= np.mean(grid.cv_theta[np.array(common) - 1])

    # --- cv_r: shift per radial column to separatrix ---
    # Separatrix cells = boundary of the core region (cv_reg==1 cells having
    # a non-core neighbour). This is more robust than sep_fc / core_sep_fcs,
    # which may miss branches (e.g. second separatrix in DND).
    if grid.cv_reg is not None and grid.fc_cv is not None:
        regs = grid.cv_reg[grid.fc_cv]  # (nFc, 2) region of the two cells
        is_core = regs == 1
        sep_faces = is_core[:, 0] != is_core[:, 1]  # exactly one side is core
        if np.any(sep_faces):
            sep_cvs = np.unique(grid.fc_cv[sep_faces].ravel())
            sep_cvs = sep_cvs[sep_cvs >= 0]
            # build radial columns from radial adjacency (same as compute_radial_coordinate)
            rad_adj = _build_radial_adjacency(grid)
            visited = np.zeros(grid.n_cells, dtype=bool)
            for iCv in range(grid.n_cells):
                if visited[iCv] or len(rad_adj[iCv]) == 0:
                    continue
                col = get_radial_column(iCv, rad_adj, grid.n_core_cells)
                visited[col] = True
                inter = np.intersect1d(col, sep_cvs)
                if len(inter) > 0:
                    grid.cv_r[col] -= np.mean(grid.cv_r[inter])


# =========================================================================
# 11. Face poloidal/radial field coordinates
# =========================================================================

def _compute_face_field_coordinates(grid: GridTopology) -> None:
    """Compute fcTheta (poloidal coord along flux surfaces) and fcR."""
    if grid.fc_theta is not None:
        return  # Already computed

    grid.fc_theta = np.zeros(grid.n_faces, dtype=np.float64)

    for iFs in range(grid.n_flux_surfaces):
        start = grid.fs_fc_p[iFs, 0]
        count = grid.fs_fc_p[iFs, 1]
        if count < 2:
            continue
        fcs = grid.fs_fc[start:start + count]
        dx = grid.fc_x[fcs[1:]] - grid.fc_x[fcs[:-1]]
        dy = grid.fc_y[fcs[1:]] - grid.fc_y[fcs[:-1]]
        dist = np.concatenate([[0], np.cumsum(np.sqrt(dx ** 2 + dy ** 2))])
        grid.fc_theta[fcs] = dist

        # Shift to OMP
        if hasattr(grid, 'fc_outer_midplane') and grid.fc_outer_midplane is not None:
            common = np.intersect1d(fcs, grid.fc_outer_midplane)
            if len(common) > 0:
                grid.fc_theta[fcs] -= np.mean(grid.fc_theta[common])

    # fcR from cell average
    grid.fc_r = np.zeros(grid.n_faces, dtype=np.float64)
    grid.fc_r = 0.5 * (grid.cv_r[grid.fc_cv[:, 0]] + grid.cv_r[grid.fc_cv[:, 1]])


# =========================================================================
# 12. Core-SEP faces
# =========================================================================

def _find_core_sep_faces(grid: GridTopology) -> None:
    """Find faces between CORE and SOL regions."""
    regs = grid.cv_reg[grid.fc_cv]
    diff_reg = regs[:, 0] != regs[:, 1]
    core_mask = np.any(regs == 1, axis=1) | np.any(regs == 5, axis=1)
    sep_diff = np.abs(regs[:, 0] - regs[:, 1]) == 1

    grid.core_sep_fcs = np.where(diff_reg & core_mask & sep_diff)[0].astype(np.int32)


# =========================================================================
# 13. Orientation (BpDir, BtDir, cvOr, fcOr)
# =========================================================================

def _compute_orientation(grid: GridTopology) -> None:
    """Compute orientation vectors and field directions."""
    nCv = grid.n_cells
    nFc = grid.n_faces

    cv_or = np.zeros(nCv, dtype=np.int32)
    fc_or = np.zeros(nFc, dtype=np.int32)

    for iCv in range(grid.n_core_cells, nCv):
        start = grid.cv_fc_p[iCv, 0]
        count = grid.cv_fc_p[iCv, 1]
        if count == 0:
            continue
        iFc = grid.cv_fc[start]
        cvs = grid.fc_cv[iFc]
        if cvs[1] > grid.n_core_cells and cvs[0] <= grid.n_core_cells:
            cv_or[iCv] = 1
            fc_or[iFc] = 1
        else:
            cv_or[iCv] = -1
            fc_or[iFc] = -1

    grid.cv_or = cv_or
    grid._fc_or_cache = fc_or.astype(np.float64)  # fc_or is a lazy property

    # Field direction from OMP
    if hasattr(grid, 'outer_midplane_cells') and grid.outer_midplane_cells is not None and \
       grid.cv_eb is not None:
        omp = grid.outer_midplane_cells
        idx = np.argmin(np.abs(grid.cv_r[omp])) if hasattr(grid, 'cv_r') and grid.cv_r is not None else 0
        grid.bp_dir = -np.sign(grid.cv_eb[omp[idx], 1])
        grid.bt_dir = np.sign(grid.cv_eb[omp[idx], 2])


# =========================================================================
# Master function
# =========================================================================

def compute_all_regions(grid: GridTopology) -> dict[str, Any]:
    """Run all region computations in order. Returns regions dict."""
    # Structured grids: use fast index-based computation
    is_struct = grid.is_structured and grid.imap_cv is not None and grid.imap_cv.ndim == 2
    if is_struct:
        result = _compute_regions_structured(grid)
        if result:
            return result
        # If structured computation fails (missing data), fall through to topology
    # 1. Face coordinates
    if grid.fc_x is None and grid.vx_x is not None and grid.fc_vx is not None:
        compute_face_coordinates(grid)

    # Pre-compute radial adjacency once
    if grid.cv_ft is not None:
        rad_adj = _build_radial_adjacency(grid)
    else:
        rad_adj = None

    # 2. Radial coordinate
    if grid.cv_r is None and rad_adj is not None:
        compute_radial_coordinate(grid, rad_adj)

    # 3. Poloidal coordinate
    if grid.cv_theta is None and grid.ft_cv_p is not None:
        compute_poloidal_coordinate(grid)

    # 4. Flux tube connection
    if grid.ft_conn is None and grid.ft_fc_p is not None:
        compute_flux_tube_connection(grid)

    # 5. Midplane
    if not hasattr(grid, 'outer_midplane_cells') or grid.outer_midplane_cells is None:
        find_midplanes(grid)

    # 6. X-points
    find_xpoints_and_separatrices(grid)

    # 7. Targets
    if not hasattr(grid, 'cv_outer_tar') or grid.cv_outer_tar is None:
        find_targets(grid)

    # 8. Boundary
    if grid.fc_lbl is not None and np.any(grid.fc_lbl != 0):
        compute_boundary_coordinates(grid)

    # 9. Shift
    _shift_coordinates(grid)

    # 10. Face field coords
    if hasattr(grid, 'fc_theta') and grid.fc_theta is not None:
        _compute_face_field_coordinates(grid)

    # 11. Core-SEP
    if grid.fc_cv is not None and grid.cv_reg is not None:
        _find_core_sep_faces(grid)

    # 12. Orientation
    if grid.cv_reg is not None:
        _compute_orientation(grid)

    # 13. Normalized
    if grid.ft_cv_p is not None:
        compute_normalized_coordinates(grid)

    return _build_regions_dict(grid)


def _build_regions_dict(grid: GridTopology) -> dict[str, Any]:
    """Build a summary dict from grid region attributes."""
    attrs = [
        "cv_r", "cv_theta", "fc_x", "fc_y", "fc_theta", "fc_r",
        "cv_theta_n", "fc_theta_n", "ft_conn",
        "inner_midplane_cells", "outer_midplane_cells",
        "inner_midplane_cells_sol", "outer_midplane_cells_sol",
        "fc_inner_midplane", "fc_outer_midplane",
        "xp_vx", "fs_sep", "fs_sep2",
        "cv_inner_tar", "cv_outer_tar", "fc_inner_tar", "fc_outer_tar",
        "cv_inner_top_tar", "cv_outer_top_tar",
        "sep_vx", "sep_fc", "core_sep_fcs",
        "cv_lbl_len", "fc_lbl_len", "bp_dir", "bt_dir",
    ]
    return {a: getattr(grid, a, None) for a in attrs}
