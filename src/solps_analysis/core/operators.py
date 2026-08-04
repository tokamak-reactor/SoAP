"""Mesh operators for SOLPS-ITER grids.

Ports of the MATLAB Calc_WD operator set:
  - div_us        — divergence of face flows onto cells
  - intface       — cell → face interpolation (vol / hc / halfsum)
  - intcell_us    — face → cell interpolation (weighted)
  - intvertex_us  — cell → vertex interpolation (inverse-volume weighted)
  - calc_vxVol    — vertex volume weights
  - grad_r_us     — radial gradient at faces
  - grad_p_us     — poloidal gradient at faces
  - gradc_r_us    — radial gradient at cell centers (face grad + intcell)
  - gradc_p_us    — poloidal gradient at cell centers

All connectivity arrays are 0-based (fc_cv, fc_vx, cv_fc, ...).
"""

from __future__ import annotations

import numpy as np

from solps_analysis.core.grid import GridTopology


# ──────────────────────────────────────────────────────────────
# div_us
# ──────────────────────────────────────────────────────────────

def div_us(grid: GridTopology, flow: np.ndarray) -> np.ndarray:
    """Divergence of a face-centered flow field.

    flow: (n_faces, 2) for a single field (theta, radial components), or
          (n_faces, 2*ns) for ns fields packed as [th_1..th_ns, r_1..r_ns]
          (MATLAB convention: flow(:,is) and flow(:,is+ns)).

    Returns (n_cells,) or (n_cells, ns).
    """
    n_cv = grid.n_cells
    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim == 1:
        flow = flow[:, None]

    if grid.fc_cv is None:
        raise ValueError("div_us: grid.fc_cv is not available")

    n_cols = flow.shape[1]
    if n_cols == 2:
        ns = 1
    elif n_cols % 2 == 0:
        ns = n_cols // 2
    else:
        raise ValueError("div_us: flow must have 2 or 2*ns columns")

    fc_cv = grid.fc_cv  # (n_faces, 2) 0-based
    div = np.zeros((n_cv, ns), dtype=np.float64)

    # For each species: div[cv1] += f_th + f_r ; div[cv2] -= f_th + f_r
    cv1 = fc_cv[:, 0]
    cv2 = fc_cv[:, 1]
    for is_ in range(ns):
        f = flow[:, is_] + flow[:, is_ + ns]
        np.add.at(div[:, is_], cv1, f)
        np.add.at(div[:, is_], cv2, -f)

    if ns == 1:
        return div[:, 0]
    return div


# ──────────────────────────────────────────────────────────────
# intface
# ──────────────────────────────────────────────────────────────

def intface(grid: GridTopology, field: np.ndarray, direction: int = 1,
            method: str = "vol") -> np.ndarray:
    """Interpolate a cell-centered field to cell faces.

    direction is ignored for unstructured-style grids (kept for API
    compatibility with MATLAB's intface).

    Methods:
      'vol'      — volume weighted: (vol1*f2 + vol2*f1) / (vol1+vol2)
      'hc'       — connector-length weighted: (hc1*f2 + hc2*f1) / (hc1+hc2)
      'halfsum'  — simple average

    Returns (n_faces,) or (n_faces, ns).
    """
    field = np.asarray(field, dtype=np.float64)
    one_dim = field.ndim == 1
    if one_dim:
        field = field[:, None]

    n_fc = grid.n_faces
    n_spec = field.shape[1]
    out = np.zeros((n_fc, n_spec), dtype=np.float64)

    if grid.fc_cv is None:
        raise ValueError("intface: grid.fc_cv is not available")

    cv1 = grid.fc_cv[:, 0]
    cv2 = grid.fc_cv[:, 1]

    if method == "vol":
        vol = grid.cv_vol
        w = vol[cv1] + vol[cv2]
        ok = w > 0
        for is_ in range(n_spec):
            f1 = field[cv1, is_]
            f2 = field[cv2, is_]
            out[:, is_] = np.where(ok, (vol[cv1] * f2 + vol[cv2] * f1) / np.where(ok, w, 1), 0.0)
    elif method == "hc":
        hc = grid.fc_hc  # (n_faces, 2)
        w = hc[:, 0] + hc[:, 1]
        ok = w > 0
        for is_ in range(n_spec):
            f1 = field[cv1, is_]
            f2 = field[cv2, is_]
            out[:, is_] = np.where(ok, (hc[:, 0] * f2 + hc[:, 1] * f1) / np.where(ok, w, 1), 0.0)
    elif method == "halfsum":
        for is_ in range(n_spec):
            out[:, is_] = 0.5 * (field[cv1, is_] + field[cv2, is_])
    else:
        raise ValueError(f"intface: unknown method '{method}'")

    if one_dim:
        return out[:, 0]
    return out


# ──────────────────────────────────────────────────────────────
# intcell_us
# ──────────────────────────────────────────────────────────────

def intcell_us(grid: GridTopology, weights: np.ndarray, face_field: np.ndarray) -> np.ndarray:
    """Interpolate a face-centered field to cell centers.

    weights: flat array aligned with grid.cv_fc (one weight per face-slot).
    MATLAB: intcell_us(nCv, gmtry, gmtry.intcellP, face_field).

    Returns (n_cells,) or (n_cells, ns).
    """
    face_field = np.asarray(face_field, dtype=np.float64)
    one_dim = face_field.ndim == 1
    if one_dim:
        face_field = face_field[:, None]

    n_cv = grid.n_cells
    n_spec = face_field.shape[1]
    centre = np.zeros((n_cv, n_spec), dtype=np.float64)
    wsum = np.zeros(n_cv, dtype=np.float64)

    if grid.cv_fc_p is None or grid.cv_fc is None:
        raise ValueError("intcell_us: grid.cv_fc_p / cv_fc not available")

    cv_fc_p = grid.cv_fc_p  # (n_cells, 2): start, count
    cv_fc = grid.cv_fc      # flat face list (0-based)

    for icv in range(n_cv):
        start = int(cv_fc_p[icv, 0])
        count = int(cv_fc_p[icv, 1])
        if count == 0:
            continue
        faces = cv_fc[start:start + count]
        w = weights[start:start + count]
        ws = w.sum()
        if ws > 0:
            for is_ in range(n_spec):
                centre[icv, is_] = (face_field[faces, is_] * w).sum() / ws

    if one_dim:
        return centre[:, 0]
    return centre


# ──────────────────────────────────────────────────────────────
# calc_vxVol
# ──────────────────────────────────────────────────────────────

def calc_vxVol(grid: GridTopology, mode: int = 0) -> np.ndarray:
    """Vertex volume weights (flat array aligned with vx_cv).

    mode 0: (1/cvVxP(cv,2)) * cvVol(cv)  — inverse volume interpolation
    mode 1: 1/cvVxP(cv,2) for interior cells, volume-scaled for guard cells
    mode 2: 0.25 for interior cells, 0.25e-3 for guard cells
    """
    n_cv = grid.n_cells
    if grid.vx_cv_p is None or grid.vx_cv is None or grid.cv_vx_p is None:
        raise ValueError("calc_vxVol: vertex connectivity not available")
    vx_cv_p = grid.vx_cv_p  # (n_vertices, 2): start, count
    vx_cv = grid.vx_cv      # flat cell list per vertex (0-based)
    n_vx = grid.n_vertices

    # Build inverse map: for each vertex slot, the cell index
    # MATLAB fills vxVol over vxCvP(vx,1)+inCv-1 slots with weight per (vx, cv)
    total_slots = int(vx_cv_p[:, 1].sum()) if vx_cv_p is not None else 0
    vx_vol = np.zeros(total_slots, dtype=np.float64)
    if total_slots == 0:
        return vx_vol

    nci = grid.n_core_cells or grid.n_cells
    for ivx in range(n_vx):
        start = int(vx_cv_p[ivx, 0])
        count = int(vx_cv_p[ivx, 1])
        for k in range(count):
            icv = vx_cv[start + k]
            slot = start + k
            n_cv_of_vx = int(grid.cv_vx_p[icv, 1]) if grid.cv_vx_p is not None else 4
            if mode == 0:
                vx_vol[slot] = (1.0 / n_cv_of_vx) * grid.cv_vol[icv]
            elif mode == 1:
                if icv < nci:
                    vx_vol[slot] = 1.0 / n_cv_of_vx
                else:
                    vx_vol[slot] = (1.0 / n_cv_of_vx) * grid.cv_vol[icv]
            elif mode == 2:
                if icv < nci:
                    vx_vol[slot] = 0.25
                else:
                    vx_vol[slot] = 0.25e-3
            else:
                raise ValueError("calc_vxVol: mode must be 0, 1 or 2")
    return vx_vol


# ──────────────────────────────────────────────────────────────
# intvertex_us
# ──────────────────────────────────────────────────────────────

def intvertex_us(grid: GridTopology, centre: np.ndarray,
                 vx_vol: np.ndarray | None = None) -> np.ndarray:
    """Interpolate a cell-centered field to vertices.

    Uses vxVol weights (inverse-volume style): for vertex vx,
      vertex = sum( centre[cv] / vxVol[slot] ) / sum( 1 / vxVol[slot] )
    """
    centre = np.asarray(centre, dtype=np.float64)
    if vx_vol is None:
        vx_vol = calc_vxVol(grid, 0)

    n_vx = grid.n_vertices
    vertex = np.zeros(n_vx, dtype=np.float64)

    if grid.vx_cv_p is None or grid.vx_cv is None:
        raise ValueError("intvertex_us: grid.vx_cv_p / vx_cv not available")

    vx_cv_p = grid.vx_cv_p
    vx_cv = grid.vx_cv

    for ivx in range(n_vx):
        start = int(vx_cv_p[ivx, 0])
        count = int(vx_cv_p[ivx, 1])
        volsum = 0.0
        acc = 0.0
        for k in range(count):
            w = vx_vol[start + k]
            if w == 0:
                continue
            icv = vx_cv[start + k]
            acc += centre[icv] / w
            volsum += 1.0 / w
        if volsum > 0:
            vertex[ivx] = acc / volsum
    return vertex


# ──────────────────────────────────────────────────────────────
# grad_r_us / grad_p_us
# ──────────────────────────────────────────────────────────────

def _grad_face(grid: GridTopology, fun: np.ndarray, funv: np.ndarray,
               poloidal: bool) -> np.ndarray:
    """Gradient at faces (radial if poloidal=False, poloidal otherwise)."""
    n_fc = grid.n_faces
    gfun = np.zeros(n_fc, dtype=np.float64)

    if grid.fc_cv is None or grid.fc_vx is None or grid.fc_qalf is None \
            or grid.fc_qgam is None or grid.fc_qbet is None \
            or grid.fc_hc is None or grid.fc_ht is None:
        raise ValueError("_grad_face: face geometry not available (fc_qalf, fc_qgam, ...)")

    cv1 = grid.fc_cv[:, 0]
    cv2 = grid.fc_cv[:, 1]
    vx1 = grid.fc_vx[:, 0]
    vx2 = grid.fc_vx[:, 1]

    qalf = grid.fc_qalf   # (n_faces, 2) cos/sin alpha
    qgam = grid.fc_qgam   # (n_faces, 2) cos/sin gamma
    qbet = grid.fc_qbet   # (n_faces, 2) cos/sin beta
    hc_sum = grid.fc_hc[:, 0] + grid.fc_hc[:, 1]
    ht = grid.fc_ht

    if poloidal:
        a1, a2 = qalf[:, 0], qalf[:, 1]
        b1, b2 = qbet[:, 0], qbet[:, 1]
    else:
        a1, a2 = qalf[:, 1], qalf[:, 0]
        b1, b2 = qbet[:, 0], qbet[:, 1]

    # funv is vertex field (already computed by caller for mode=0)
    d_cv = fun[cv2] - fun[cv1]
    d_vx = funv[vx2] - funv[vx1]

    qg = qgam[:, 0]
    ok = (np.abs(qg) > 1e-30) & (hc_sum > 0) & (ht > 0)
    gfun = np.where(ok,
                    d_cv * a1 / (qg * hc_sum) + d_vx * b1 / (qg * ht),
                    0.0)
    return gfun


def grad_r_us(grid: GridTopology, mode: int, fun: np.ndarray,
              funv: np.ndarray | None = None) -> np.ndarray:
    """Radial gradient at faces.

    mode 0: interpolate fun to vertices internally (needs calc_vxVol)
    mode 1: use provided funv (vertex field)
    """
    fun = np.asarray(fun, dtype=np.float64)
    if mode == 0:
        vx_vol = calc_vxVol(grid, 0)
        funv = intvertex_us(grid, fun, vx_vol)
    if funv is None:
        raise ValueError("grad_r_us: funv required for mode 1")
    return _grad_face(grid, fun, np.asarray(funv, dtype=np.float64), poloidal=False)


def grad_p_us(grid: GridTopology, mode: int, fun: np.ndarray,
              funv: np.ndarray | None = None) -> np.ndarray:
    """Poloidal gradient at faces."""
    fun = np.asarray(fun, dtype=np.float64)
    if mode == 0:
        vx_vol = calc_vxVol(grid, 0)
        funv = intvertex_us(grid, fun, vx_vol)
    if funv is None:
        raise ValueError("grad_p_us: funv required for mode 1")
    return _grad_face(grid, fun, np.asarray(funv, dtype=np.float64), poloidal=True)


# ──────────────────────────────────────────────────────────────
# gradc_r_us / gradc_p_us — gradients at cell centers
# ──────────────────────────────────────────────────────────────

def gradc_r_us(grid: GridTopology, mode: int, fun: np.ndarray,
               funv: np.ndarray | None = None) -> np.ndarray:
    """Radial gradient at cell centers: face gradient + intcell (intcellR)."""
    gface = grad_r_us(grid, mode, fun, funv)
    return intcell_us(grid, grid.intcell_r, gface)


def gradc_p_us(grid: GridTopology, mode: int, fun: np.ndarray,
               funv: np.ndarray | None = None) -> np.ndarray:
    """Poloidal gradient at cell centers: face gradient + intcell (intcellP)."""
    gface = grad_p_us(grid, mode, fun, funv)
    return intcell_us(grid, grid.intcell_p, gface)
