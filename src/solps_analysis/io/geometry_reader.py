"""Read SOLPS-ITER geometry from b2fgmtry files.

Handles both structured (3.0.x) and unstructured (3.2.x) formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from solps_analysis.core.grid import GridTopology
from solps_analysis.io.b2fgmtry_parser import read_b2fgmtry, read_tagged_ascii_sections
from solps_analysis.io.stdstream import log_msg


def _parse_version(version_str: str | None) -> tuple[str, int]:
    if version_str is None:
        return "", 0
    if version_str.startswith("VERSION"):
        ver = version_str[7:].split()[0]
    else:
        ver = version_str
    parts = ver.split(".")
    try:
        num = int(parts[0]) * 100 + int(parts[1]) * 10 + int(parts[2])
    except (IndexError, ValueError):
        num = 0
    return ver, num


def _ensure_float(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.ravel() if arr.ndim == 1 else arr
    return np.asarray(arr, dtype=np.float64)


def _ensure_int(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.ravel() if arr.ndim == 1 else arr
    return np.asarray(arr, dtype=np.int32)


def _read_structured_b2fgmtry(path: str | Path) -> dict[str, Any]:
    """Read a structured-format (3.0.x) b2fgmtry file.

    Returns a flat dict with the raw named fields from the file.
    """
    path = Path(path)
    with path.open("r") as f:
        raw = read_tagged_ascii_sections(f)

    result: dict[str, Any] = {}
    result["_version"] = raw.get("_version")

    # Resolve comma-separated field names
    for key, value in raw.items():
        if key == "_version":
            continue
        names = [n.strip() for n in key.split(",")]
        if len(names) == 1:
            result[names[0]] = value
        elif isinstance(value, np.ndarray) and value.size == len(names):
            for i, name in enumerate(names):
                result[name] = value[i]
        else:
            result[key] = value

    return result


def _load_structured_grid(raw: dict[str, Any]) -> GridTopology:
    """Convert a structured b2fgmtry raw dict into a fully-populated GridTopology.

    This is the Python equivalent of MATLAB's read_b2fgmtry_30.m.
    """
    ver, num = _parse_version(raw.get("_version"))

    nx = int(raw["nx"])
    ny = int(raw["ny"])

    # --- Read topology arrays ---
    left = _ensure_int(raw["leftix"]).reshape(nx + 2, ny + 2, order="F")
    right = _ensure_int(raw["rightix"]).reshape(nx + 2, ny + 2, order="F")
    top = _ensure_int(raw["topiy"]).reshape(nx + 2, ny + 2, order="F")
    bottom = _ensure_int(raw["bottomiy"]).reshape(nx + 2, ny + 2, order="F")

    # Fortran stores 2D arrays column-major, so our reshape needs care.
    # The MATLAB code reads [nx+2, ny+2] and indexes (ix, iy) with ix varying fastest.
    # In MATLAB: left(ix, iy) where ix=1:nx+2, iy=1:ny+2.
    # numpy reshapes as (nx+2, ny+2, order='F') for Fortran ordering.
    left = _ensure_int(raw["leftix"]).reshape(nx + 2, ny + 2, order="F")
    right = _ensure_int(raw["rightix"]).reshape(nx + 2, ny + 2, order="F")
    top = _ensure_int(raw["topiy"]).reshape(nx + 2, ny + 2, order="F")
    bottom = _ensure_int(raw["bottomiy"]).reshape(nx + 2, ny + 2, order="F")

    # --- Corner coordinates ---
    crx = _ensure_float(raw["crx"]).reshape(nx + 2, ny + 2, 4, order="F")
    cry = _ensure_float(raw["cry"]).reshape(nx + 2, ny + 2, 4, order="F")

    # --- Cell-centered quantities ---
    bb = _ensure_float(raw["bb"]).reshape(nx + 2, ny + 2, 4, order="F")
    vol = _ensure_float(raw["vol"]).reshape(nx + 2, ny + 2, order="F")
    hx = _ensure_float(raw["hx"]).reshape(nx + 2, ny + 2, order="F")
    hy = _ensure_float(raw["hy"]).reshape(nx + 2, ny + 2, order="F")
    qz = _ensure_float(raw["qz"]).reshape(nx + 2, ny + 2, 2, order="F")
    qc = _ensure_float(raw.get("qc", np.zeros((nx + 2, ny + 2)))).reshape(nx + 2, ny + 2, order="F")
    gs = _ensure_float(raw["gs"]).reshape(nx + 2, ny + 2, 3, order="F")

    # Poloidal flux and fBz at 4 corners
    fpsi_data = _ensure_float(raw.get("fpsi", np.zeros((nx + 2, ny + 2, 4)))).reshape(
        nx + 2, ny + 2, 4, order="F"
    )
    ffbz_data = _ensure_float(raw.get("ffbz", np.zeros((nx + 2, ny + 2, 4)))).reshape(
        nx + 2, ny + 2, 4, order="F"
    )

    # Magnetic contact area (pbs) — 2 components per cell
    pbs = _ensure_float(raw.get("pbs", np.zeros((nx + 2, ny + 2, 2)))).reshape(
        nx + 2, ny + 2, 2, order="F"
    )

    # Region data
    region = _ensure_float(raw.get("region", np.zeros((nx + 2, ny + 2, 3)))).reshape(
        nx + 2, ny + 2, 3, order="F"
    )

    # Cut info
    nncut = int(raw.get("nncut", 1))
    leftcut = _ensure_int(raw.get("leftcut", np.array([0]))).ravel()
    rightcut = _ensure_int(raw.get("rightcut", np.array([0]))).ravel()
    topcut = _ensure_int(raw.get("topcut", np.array([0]))).ravel()

    # --- Compute dimensions ---
    nCi = (nx + 2 - 2 * nncut) * ny
    nCg = (nx + 2 - 2 * nncut) * 2 + ny * nncut * 2
    nCv = nCi + nCg
    nFc = int(2 * nCi + nx + 2 - 2 * nncut + ny * nncut)
    nVx = int(nCi + nx + 2 - 2 * nncut + ny * nncut + nncut)
    nFt = int((ny + 2) * nncut + (topcut[0] + 1))
    nFs = int((ny + 1) * nncut + topcut[0])

    # Compute hz from vol / hx / hy / qz(:,:,2)
    hz = vol / (hx * hy * qz[:, :, 1])
    FFBZ = hz[0, 0] * bb[0, 0, 2]  # Bz component is index 2 (0-based)

    # --- Build grid ---
    grid = GridTopology(
        n_cells=nCv,
        n_faces=nFc,
        n_vertices=nVx,
        n_flux_surfaces=nFs,
        n_flux_tubes=nFt,
        n_core_cells=nCi,
        is_structured=True,
        nx=nx,
        ny=ny,
        nncut=nncut,
        version=ver,
        version_number=num,
    )

    # Store cut data and mesh spacing for region computation
    grid._cut_leftcut = leftcut
    grid._cut_rightcut = rightcut
    grid._cut_topcut = topcut
    grid._cut_bottomcut = _ensure_int(raw.get("bottomcut", np.array([0]))).ravel()
    grid._hx = hx
    grid._hy = hy
    # Store neighbor arrays for radial walking
    grid._leftix = left
    grid._rightix = right
    grid._topiy = top
    grid._bottomiy = bottom

    # --- Build imap arrays ---
    imap_cv = np.zeros((nx + 2, ny + 2), dtype=np.int32)
    imap_fcx = np.zeros((nx + 2, ny + 2), dtype=np.int32)
    imap_fcy = np.zeros((nx + 2, ny + 2), dtype=np.int32)
    imap_vx = np.zeros((nx + 2, ny + 2), dtype=np.int32)

    i_ci = 0
    i_cg = nCi
    i_fc = 0
    i_vx = 0

    # MATLAB indexing: 1-based, ix=1:nx+2, iy=1:ny+2
    # Python: 0-based, ix=0:nx+1, iy=0:ny+1
    # left(ix,iy) == -2 in MATLAB means MATLAB's value is -2 → Python: left[ix, iy] == -2

    for iy in range(ny + 2):
        for ix in range(nx + 2):
            l = left[ix, iy]
            b = bottom[ix, iy]
            r = right[ix, iy]
            t = top[ix, iy]

            is_guard = (l == -2 or b == -2 or r == nx + 1 or t == ny + 1)
            is_corner = (l == -2 and b == -2) or (r == nx + 1 and b == -2) or (l == -2 and t == ny + 1) or (r == nx + 1 and t == ny + 1)

            if is_guard and not is_corner:
                i_cg += 1
                imap_cv[ix, iy] = i_cg
            elif not is_guard and not is_corner:
                i_ci += 1
                imap_cv[ix, iy] = i_ci

            if l != -2 and b != -2 and t != ny + 1:
                i_fc += 1
                imap_fcx[ix, iy] = i_fc

            if l != -2 and b != -2:
                i_vx += 1
                imap_vx[ix, iy] = i_vx

    for ix in range(nx + 2):
        for iy in range(ny + 2):
            if bottom[ix, iy] != -2 and left[ix, iy] != -2 and right[ix, iy] != nx + 1:
                i_fc += 1
                imap_fcy[ix, iy] = i_fc

    grid.imap_cv = imap_cv
    grid.imap_fcx = imap_fcx
    grid.imap_fcy = imap_fcy
    grid.imap_vx = imap_vx

    # --- Build connectivity ---
    cv_fc_p = np.zeros((nCv, 2), dtype=np.int32)
    cv_vx_p = np.zeros((nCv, 2), dtype=np.int32)

    cv_fc_p[:nCi, 1] = 4
    cv_fc_p[nCi:, 1] = 1
    cv_vx_p[:nCi, 1] = 4
    cv_vx_p[nCi:, 1] = 2

    n_cmx_fc = int(4 * nCi + nCg)
    n_cmx_vx = int(4 * nCi + 2 * nCg)

    cv_fc = np.zeros(n_cmx_fc, dtype=np.int32)
    cv_vx = np.zeros(n_cmx_vx, dtype=np.int32)
    intcell_p = np.zeros(n_cmx_fc, dtype=np.float64)
    intcell_r = np.zeros(n_cmx_fc, dtype=np.float64)

    cv_fc_p[0, 0] = 0  # 0-based
    cv_vx_p[0, 0] = 0

    for i in range(1, nCv):
        cv_fc_p[i, 0] = int(cv_fc_p[i - 1, 0] + cv_fc_p[i - 1, 1])
        cv_vx_p[i, 0] = int(cv_vx_p[i - 1, 0] + cv_vx_p[i - 1, 1])

    grid.cv_fc_p = cv_fc_p
    grid.cv_fc = cv_fc
    grid.cv_vx_p = cv_vx_p
    grid.cv_vx = cv_vx
    grid.intcell_p = intcell_p
    grid.intcell_r = intcell_r

    # --- Cell-centered data ---
    cv_bb = np.zeros((nCv, 4), dtype=np.float64)
    cv_eb = np.zeros((nCv, 3), dtype=np.float64)
    cv_x = np.zeros(nCv, dtype=np.float64)
    cv_y = np.zeros(nCv, dtype=np.float64)
    cv_vol_arr = np.zeros(nCv, dtype=np.float64)
    cv_qgam = np.zeros((nCv, 2), dtype=np.float64)
    cv_hz_arr = np.zeros(nCv, dtype=np.float64)

    # BndLbl for guard cells (west -10, east -30, south -20, north -40)
    bnd_lbl = np.zeros(nCg, dtype=np.int32)

    for iy in range(ny + 2):
        for ix in range(nx + 2):
            idx = imap_cv[ix, iy]
            if idx == 0:
                continue
            ci = idx - 1  # 0-based
            cv_bb[ci, :] = np.abs(bb[ix, iy, :])
            cv_x[ci] = np.mean(crx[ix, iy, :])
            cv_y[ci] = np.mean(cry[ix, iy, :])
            cv_vol_arr[ci] = vol[ix, iy]
            cv_hz_arr[ci] = hz[ix, iy]
            cv_qgam[ci, 0] = qz[ix, iy, 1]  # cos(gamma)
            cv_qgam[ci, 1] = -qz[ix, iy, 0]  # sin(gamma)

            # cvEb: unit vector along B (MATLAB read_b2fgmtry_30 lines 517-522)
            e1 = (crx[ix, iy, 3] + crx[ix, iy, 1]) - (crx[ix, iy, 2] + crx[ix, iy, 0])
            e2 = (cry[ix, iy, 3] + cry[ix, iy, 1]) - (cry[ix, iy, 2] + cry[ix, iy, 0])
            t0 = np.sqrt(e1 * e1 + e2 * e2)
            if t0 > 0:
                cv_eb[ci, 0] = abs(bb[ix, iy, 0]) / bb[ix, iy, 3] * e1 / t0
                cv_eb[ci, 1] = abs(bb[ix, iy, 0]) / bb[ix, iy, 3] * e2 / t0
            cv_eb[ci, 2] = bb[ix, iy, 2] / bb[ix, iy, 3]

            # cvFc / cvVx / intcellP / intcellR for interior cells
            if idx <= nCi:
                p = int(cv_fc_p[ci, 0])
                pv = int(cv_vx_p[ci, 0])
                cv_fc[p + 0] = imap_fcx[ix, iy]
                cv_fc[p + 1] = imap_fcx[right[ix, iy] + 1, iy]
                cv_fc[p + 2] = imap_fcy[ix, iy]
                cv_fc[p + 3] = imap_fcy[ix, top[ix, iy] + 1]
                cv_vx[pv + 0] = imap_vx[ix, iy]
                cv_vx[pv + 1] = imap_vx[right[ix, iy] + 1, iy]
                cv_vx[pv + 2] = imap_vx[ix, top[ix, iy] + 1]
                cv_vx[pv + 3] = imap_vx[right[ix, top[ix, iy] + 1] + 1, top[ix, iy] + 1]
                intcell_p[p + 0] = 1.0
                intcell_p[p + 1] = 1.0
                intcell_p[p + 2] = 0.0
                intcell_p[p + 3] = 0.0
                intcell_r[p + 0] = 0.0
                intcell_r[p + 1] = 0.0
                intcell_r[p + 2] = 1.0
                intcell_r[p + 3] = 1.0
            else:
                # guard cell: single face, one or two vertices
                p = int(cv_fc_p[ci, 0])
                pv = int(cv_vx_p[ci, 0])
                gi = idx - nCi - 1  # 0-based guard index
                if left[ix, iy] == -2:  # western boundary
                    cv_fc[p] = imap_fcx[right[ix, iy] + 1, iy]
                    cv_vx[pv] = imap_vx[right[ix, iy] + 1, iy]
                    bnd_lbl[gi] = -10
                elif right[ix, iy] == nx + 1:  # eastern boundary
                    cv_fc[p] = imap_fcx[ix, iy]
                    cv_vx[pv] = imap_vx[ix, iy]
                    bnd_lbl[gi] = -30
                elif bottom[ix, iy] == -2:  # southern boundary
                    cv_fc[p] = imap_fcy[ix, top[ix, iy] + 1]
                    cv_vx[pv] = imap_vx[ix, top[ix, iy] + 1]
                    bnd_lbl[gi] = -20
                elif top[ix, iy] == ny + 1:  # northern boundary
                    cv_fc[p] = imap_fcy[ix, iy]
                    cv_vx[pv] = imap_vx[ix, iy]
                    bnd_lbl[gi] = -40

    grid.cv_bb = cv_bb
    grid.cv_eb = cv_eb
    grid.cv_x = cv_x
    grid.cv_y = cv_y
    grid.cv_vol = cv_vol_arr
    grid.cv_hz = cv_hz_arr
    grid.cv_qgam = cv_qgam
    # Store cell corner coordinates for 2D plotting
    grid.cv_crn_r = crx  # (nx+2, ny+2, 4)
    grid.cv_crn_z = cry

    # --- Face linking arrays (fcCv, fcVx) ---
    fc_cv = np.zeros((nFc, 2), dtype=np.int32)
    fc_vx = np.zeros((nFc, 2), dtype=np.int32)

    for iy in range(ny + 2):
        for ix in range(nx + 2):
            fcx = imap_fcx[ix, iy]
            if fcx != 0:
                # x-face = west face of cell (ix, iy); crossed by x-component of flux
                fc_cv[fcx - 1, 0] = imap_cv[left[ix, iy] + 1, iy]
                fc_cv[fcx - 1, 1] = imap_cv[ix, iy]
                fc_vx[fcx - 1, 0] = imap_vx[ix, iy]
                fc_vx[fcx - 1, 1] = imap_vx[ix, top[ix, iy] + 1]

    for ix in range(nx + 2):
        for iy in range(ny + 2):
            fcy = imap_fcy[ix, iy]
            if fcy != 0:
                # y-face = bottom face of cell (ix, iy); crossed by y-component of flux
                fc_cv[fcy - 1, 0] = imap_cv[ix, bottom[ix, iy] + 1]
                fc_cv[fcy - 1, 1] = imap_cv[ix, iy]
                fc_vx[fcy - 1, 0] = imap_vx[ix, iy]
                fc_vx[fcy - 1, 1] = imap_vx[right[ix, iy] + 1, iy]

    # --- Merge X-point vertices (MATLAB lines 290-304) ---
    for icut in range(nncut):
        ix1 = leftcut[icut] + 1
        iy1 = topcut[icut] + 1
        ix2 = rightcut[icut] + 1
        iy2 = topcut[icut] + 1
        ivx1 = imap_vx[ix1, iy1]
        ivx2 = imap_vx[ix2, iy2]
        if ivx1 != 0 and ivx2 != 0 and ivx1 != ivx2:
            if crx[ix1, iy1, 0] == crx[ix2, iy2, 0] and cry[ix1, iy1, 0] == cry[ix2, iy2, 0]:
                imap_vx[imap_vx == ivx2] = ivx1
                cv_vx[cv_vx == ivx2] = ivx1
                fc_vx[fc_vx == ivx2] = ivx1

    grid.imap_vx = imap_vx

    # --- Vertex linking arrays (vxFcP, vxCvP) ---
    vx_fc_p = np.zeros((nVx, 2), dtype=np.int32)
    vx_cv_p = np.zeros((nVx, 2), dtype=np.int32)

    # Count faces/vertices per vertex (MATLAB: sum(sum(fcVx == ivx)))
    fc_vx_flat = fc_vx.ravel()
    cv_vx_flat = cv_vx.ravel()
    fc_counts = np.bincount(fc_vx_flat[fc_vx_flat > 0], minlength=nVx + 1)[1:]
    cv_counts = np.bincount(cv_vx_flat[cv_vx_flat > 0], minlength=nVx + 1)[1:]
    vx_fc_p[:, 1] = fc_counts
    vx_cv_p[:, 1] = cv_counts
    starts_fc = np.zeros(nVx, dtype=np.int32)
    starts_cv = np.zeros(nVx, dtype=np.int32)
    starts_fc[1:] = np.cumsum(fc_counts)[:-1]
    starts_cv[1:] = np.cumsum(cv_counts)[:-1]
    vx_fc_p[:, 0] = starts_fc
    vx_cv_p[:, 0] = starts_cv

    n_vmx_fc = int(fc_counts.sum())
    n_vmx_cv = int(cv_counts.sum())
    vx_fc = np.zeros(n_vmx_fc, dtype=np.int32)
    vx_cv = np.zeros(n_vmx_cv, dtype=np.int32)
    fill_fc = np.zeros(nVx, dtype=np.int32)
    fill_cv = np.zeros(nVx, dtype=np.int32)

    # Build vxFc: for each vertex, faces containing it (ascending face order)
    order_fc = np.argsort(fc_vx_flat, kind="stable")
    # fc_vx_flat = tile of [fc0 vx0, fc0 vx1, fc1 vx0, ...]; need face index for each slot
    fc_idx_flat = np.repeat(np.arange(1, nFc + 1), 2)
    for pos in order_fc:
        v = int(fc_vx_flat[pos])
        if v > 0:
            slot = int(vx_fc_p[v - 1, 0]) + fill_fc[v - 1]
            vx_fc[slot] = fc_idx_flat[pos]
            fill_fc[v - 1] += 1

    # Build vxCv: for each vertex, cells containing it (ascending cell order)
    order_cv = np.argsort(cv_vx_flat, kind="stable")
    cv_idx_flat = np.repeat(np.arange(1, nCv + 1), 4)
    for pos in order_cv:
        v = int(cv_vx_flat[pos])
        if v > 0:
            slot = int(vx_cv_p[v - 1, 0]) + fill_cv[v - 1]
            vx_cv[slot] = cv_idx_flat[pos]
            fill_cv[v - 1] += 1

    grid.vx_fc_p = vx_fc_p
    grid.vx_fc = vx_fc
    grid.vx_cv_p = vx_cv_p
    grid.vx_cv = vx_cv

    # --- Vertex coordinates (vxX, vxY) ---
    vx_x = np.zeros(nVx, dtype=np.float64)
    vx_y = np.zeros(nVx, dtype=np.float64)
    for ix in range(1, nx + 2):
        for iy in range(1, ny + 2):
            ivx = imap_vx[ix, iy]
            if ivx != 0:
                vx_x[ivx - 1] = crx[ix, iy, 0]
                vx_y[ivx - 1] = cry[ix, iy, 0]

    grid.vx_x = vx_x
    grid.vx_y = vx_y

    # --- Regions ---
    cv_reg = np.zeros(nCv, dtype=np.int32)
    fc_reg = np.zeros(nFc, dtype=np.int32)
    cv_on_closed = np.zeros(nCv, dtype=np.int32)

    for iy in range(ny + 2):
        for ix in range(nx + 2):
            idx = imap_cv[ix, iy]
            if idx != 0:
                ci = idx - 1
                cv_reg[ci] = int(region[ix, iy, 0])
                cv_on_closed[ci] = int(region[ix, iy, 0] % 4 == 1)

    # Face regions for x-faces
    n_fcx_reg = int(np.max(region[:, :, 0]))
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 2):
            idx = imap_fcx[ix, iy]
            if idx != 0:
                fi = idx - 1
                if region[ix, iy, 0] > 0:
                    fc_reg[fi] = int(region[ix, iy, 0])
                else:
                    fc_reg[fi] = int(abs(region[ix, iy, 0])) + n_fcx_reg

    for ix in range(1, nx + 1):
        for iy in range(1, ny + 2):
            idx = imap_fcy[ix, iy]
            if idx != 0:
                fi = idx - 1
                if region[ix, iy, 1] > 0:
                    fc_reg[fi] = int(region[ix, iy, 1]) + n_fcx_reg
                else:
                    fc_reg[fi] = int(abs(region[ix, iy, 1]))

    grid.cv_reg = cv_reg
    grid.fc_reg = fc_reg

    # --- Flux tubes (MATLAB lines 404-489) ---
    ft_cv_p = np.zeros((nFt, 2), dtype=np.int32)
    ft_fc_p = np.zeros((nFt, 2), dtype=np.int32)
    fs_fc_p = np.zeros((nFs, 2), dtype=np.int32)
    ft_cv = np.zeros(0, dtype=np.int32)
    ft_fc = np.zeros(0, dtype=np.int32)
    fs_fc = np.zeros(0, dtype=np.int32)
    cv_ft = np.zeros(nCv, dtype=np.int32)

    ift_cv = 0
    ift_fc = 0
    ift = 0
    ifs_fc = 0
    ifs = 0
    ft_cv_list = []
    ft_fc_list = []
    fs_fc_list = []

    for iy in range(ny + 2):
        tube = np.arange(1, nx + 3)  # 1-based ix
        left_edges = tube[left[:, iy] == -2]
        n_left = len(left_edges)
        for k in range(n_left):
            if imap_cv[left_edges[k] - 1, iy] == 0:
                left_edges[k] += 1
        for ix_beg in range(n_left + 1):
            if ix_beg < n_left:
                ix = left_edges[ix_beg]
            else:
                ix = 1
            # skip already-counted or empty cells
            while imap_cv[ix - 1, iy] == 0 or (ft_cv_list and imap_cv[ix - 1, iy] in ft_cv_list):
                ix += 1
                if ix > nx + 2:
                    break
            if ix > nx + 2:
                break
            # new flux tube
            ift += 1
            ift_cv += 1
            ift_fc += 1
            ft_cv_p[ift - 1, 0] = ift_cv
            ft_cv_p[ift - 1, 1] = 0
            ft_fc_p[ift - 1, 0] = ift_fc
            ft_fc_p[ift - 1, 1] = 0
            if iy > 1:
                ifs += 1
                ifs_fc += 1
                fs_fc_p[ifs - 1, 0] = ifs_fc
                fs_fc_p[ifs - 1, 1] = 0
            # walk along the row until the end of the flux tube
            while imap_cv[ix - 1, iy] > 0:
                while ix < nx + 1 and imap_cv[ix - 1, iy] == 0:
                    ix += 1
                if ft_cv_list and np.any(np.array(ft_cv_list) == imap_cv[ix - 1, iy]):
                    break
                ft_cv_list.append(imap_cv[ix - 1, iy])
                cv_ft[imap_cv[ix - 1, iy] - 1] = ift
                ft_cv_p[ift - 1, 1] += 1
                ift_cv += 1
                if left[ix - 1, iy] > -2 and imap_fcx[ix - 1, iy] != 0:
                    ft_fc_list.append(imap_fcx[ix - 1, iy])
                    ft_fc_p[ift - 1, 1] += 1
                    ift_fc += 1
                if iy > 1 and imap_fcy[ix - 1, iy] != 0:
                    fs_fc_list.append(imap_fcy[ix - 1, iy])
                    fs_fc_p[ifs - 1, 1] += 1
                    ifs_fc += 1
                if right[ix - 1, iy] == nx + 1:
                    break
                else:
                    ix = right[ix - 1, iy] + 2
            ift_cv -= 1
            ift_fc -= 1
            if iy > 1:
                ifs_fc -= 1

    ft_cv = np.array(ft_cv_list, dtype=np.int32)
    ft_fc = np.array(ft_fc_list, dtype=np.int32)
    fs_fc = np.array(fs_fc_list, dtype=np.int32)

    grid.ft_cv_p = ft_cv_p
    grid.ft_cv = ft_cv
    grid.ft_fc_p = ft_fc_p
    grid.ft_fc = ft_fc
    grid.fs_fc_p = fs_fc_p
    grid.fs_fc = fs_fc
    grid.cv_ft = cv_ft

    # --- Face labels (MATLAB lines 492-498) ---
    fc_lbl = np.zeros(nFc, dtype=np.int32)
    for icv in range(nCi, nCv):
        fc_lbl[cv_fc[cv_fc_p[icv, 0]] - 1] = bnd_lbl[icv - nCi] - cv_reg[icv]
    grid.fc_lbl = fc_lbl

    # --- Face geometry (MATLAB lines 555-650) ---
    fc_x = 0.5 * (vx_x[fc_vx[:, 0] - 1] + vx_x[fc_vx[:, 1] - 1])
    fc_y = 0.5 * (vx_y[fc_vx[:, 0] - 1] + vx_y[fc_vx[:, 1] - 1])
    fc_s = np.zeros(nFc, dtype=np.float64)
    fc_hc = np.zeros((nFc, 2), dtype=np.float64)
    fc_ht = np.zeros(nFc, dtype=np.float64)
    fc_bb = np.zeros((nFc, 4), dtype=np.float64)
    fc_qalf = np.zeros((nFc, 2), dtype=np.float64)
    fc_qgam = np.zeros((nFc, 2), dtype=np.float64)
    fc_qbet = np.zeros((nFc, 2), dtype=np.float64)
    fc_pbs = np.zeros(nFc, dtype=np.float64)

    # wbbl: read from file if present, else interpolate from cvBb (vol method)
    wbbl = raw.get("wbbl")
    if wbbl is None:
        wbbl = np.zeros((nx + 2, ny + 2, 4), dtype=np.float64)
        for dim in range(4):
            tmp = (cv_vol_arr[fc_cv[:, 0] - 1] * cv_bb[fc_cv[:, 1] - 1, dim] +
                   cv_vol_arr[fc_cv[:, 1] - 1] * cv_bb[fc_cv[:, 0] - 1, dim]) / \
                  (cv_vol_arr[fc_cv[:, 0] - 1] + cv_vol_arr[fc_cv[:, 1] - 1])
            # tmp[fc] for every fc; store on x-faces via imapFcx
            for iy in range(ny + 2):
                for ix in range(nx + 2):
                    fcx = imap_fcx[ix, iy]
                    if fcx != 0:
                        wbbl[ix, iy, dim] = tmp[fcx - 1]
    else:
        wbbl = _ensure_float(wbbl).reshape(nx + 2, ny + 2, 4, order="F")

    for iy in range(ny + 2):
        for ix in range(nx + 2):
            fcx = imap_fcx[ix, iy]
            if fcx != 0:
                fc = fcx - 1
                # Distance to neighbouring cell centers (approximation, as in b2us)
                fc_hc[fc, 0] = 0.5 * hx[left[ix, iy] + 1, iy]
                fc_hc[fc, 1] = 0.5 * hx[ix, iy]
                # Face area (precise)
                fc_s[fc] = gs[ix, iy, 0]
                # Face length (precise)
                dx = vx_x[fc_vx[fc, 0] - 1] - vx_x[fc_vx[fc, 1] - 1]
                dy = vx_y[fc_vx[fc, 0] - 1] - vx_y[fc_vx[fc, 1] - 1]
                fc_ht[fc] = np.sqrt(dx * dx + dy * dy)
                fc_bb[fc, 0:4] = np.abs(wbbl[ix, iy, 0:4])
                # Cosine of alpha exactly as in b2us
                fc_qalf[fc, 0] = qc[ix, iy] * np.sign(wbbl[ix, iy, 0])
                # Sine of alpha (sign conventions — see MATLAB comments)
                term = ((crx[ix, iy, 1] + crx[ix, iy, 3]) - (crx[ix, iy, 0] + crx[ix, iy, 2])) * (crx[ix, iy, 2] - crx[ix, iy, 0]) + \
                       ((cry[ix, iy, 1] + cry[ix, iy, 3]) - (cry[ix, iy, 0] + cry[ix, iy, 2])) * (cry[ix, iy, 2] - cry[ix, iy, 0])
                fc_qalf[fc, 1] = -np.sqrt(1 - qc[ix, iy] ** 2) * np.sign(wbbl[ix, iy, 0]) * \
                    np.sign(term) * np.sign(wbbl[ix, iy, 2])
                # For poloidally aligned structured meshes gamma = alpha, beta = 0
                fc_qgam[fc, 0] = qc[ix, iy]  # cos(gamma)
                term_g = ((cry[ix, iy, 1] + cry[ix, iy, 3]) - (cry[ix, iy, 0] + cry[ix, iy, 2])) * (crx[ix, iy, 2] - crx[ix, iy, 0]) - \
                         ((crx[ix, iy, 1] + crx[ix, iy, 3]) - (crx[ix, iy, 0] + crx[ix, iy, 2])) * (cry[ix, iy, 2] - cry[ix, iy, 0])
                fc_qgam[fc, 1] = np.sqrt(1 - qc[ix, iy] ** 2) * np.sign(term_g)
                fc_qbet[fc, 0] = fc_qgam[fc, 0] * fc_qalf[fc, 0] + fc_qgam[fc, 1] * fc_qalf[fc, 1]
                fc_qbet[fc, 1] = (fc_qgam[fc, 1] * fc_qalf[fc, 0] + fc_qgam[fc, 0] * fc_qalf[fc, 1]) * np.sign(wbbl[ix, iy, 2])
                fc_pbs[fc] = pbs[ix, iy, 0]

            fcy = imap_fcy[ix, iy]
            if fcy != 0:
                fc = fcy - 1
                fc_hc[fc, 0] = 0.5 * hy[ix, bottom[ix, iy] + 1]
                fc_hc[fc, 1] = 0.5 * hy[ix, iy]
                fc_s[fc] = gs[ix, iy, 1]
                dx = vx_x[fc_vx[fc, 0] - 1] - vx_x[fc_vx[fc, 1] - 1]
                dy = vx_y[fc_vx[fc, 0] - 1] - vx_y[fc_vx[fc, 1] - 1]
                fc_ht[fc] = np.sqrt(dx * dx + dy * dy)
                # Poloidal B: inverse-distance weighted average of neighbours
                cv1 = fc_cv[fc, 0] - 1
                cv2 = fc_cv[fc, 1] - 1
                fc_bb[fc, 0] = abs(fc_hc[fc, 0] * cv_bb[cv2, 0] + fc_hc[fc, 1] * cv_bb[cv1, 0]) / (fc_hc[fc, 0] + fc_hc[fc, 1])
                fc_bb[fc, 1] = 0.0
                fc_bb[fc, 2] = abs(FFBZ) / (fc_s[fc] / fc_ht[fc])
                fc_bb[fc, 3] = np.sqrt(fc_bb[fc, 0] ** 2 + fc_bb[fc, 1] ** 2 + fc_bb[fc, 2] ** 2)
                # Cosine of alpha is exactly zero
                fc_qalf[fc, 0] = 0.0
                fc_qalf[fc, 1] = np.sign(wbbl[ix, iy, 0]) * np.sign(wbbl[ix, iy, 2])
                # Gamma for y-faces
                qc2 = (hy[ix, iy] * qz[ix, iy, 1] + hy[ix, bottom[ix, iy] + 1] * qz[ix, bottom[ix, iy] + 1, 1]) / \
                      (hy[ix, iy] + hy[ix, bottom[ix, iy] + 1])
                fc_qgam[fc, 0] = qc2  # cos(gamma)
                fc_qgam[fc, 1] = np.sqrt(1 - qc2 ** 2) * \
                    np.sign(hy[ix, iy] * qz[ix, iy, 0] + hy[ix, bottom[ix, iy] + 1] * qz[ix, bottom[ix, iy] + 1, 0])
                fc_qbet[fc, 0] = fc_qgam[fc, 0] * fc_qalf[fc, 0] + fc_qgam[fc, 1] * fc_qalf[fc, 1]
                fc_qbet[fc, 1] = (fc_qgam[fc, 1] * fc_qalf[fc, 0] + fc_qgam[fc, 0] * fc_qalf[fc, 1]) * np.sign(wbbl[ix, iy, 2])
                fc_pbs[fc] = pbs[ix, iy, 1]

    grid.fc_cv = fc_cv
    grid.fc_vx = fc_vx
    grid.fc_x = fc_x
    grid.fc_y = fc_y
    grid.fc_s = fc_s
    grid.fc_hc = fc_hc
    grid.fc_ht = fc_ht
    grid.fc_bb = fc_bb
    grid.fc_qalf = fc_qalf
    grid.fc_qgam = fc_qgam
    grid.fc_qbet = fc_qbet
    grid.fc_pbs = fc_pbs

    # --- ftReg (MATLAB lines 685-698) ---
    ft_reg = np.zeros(nFt, dtype=np.int32)
    for i_ft in range(nFt):
        start = int(ft_cv_p[i_ft, 0]) - 1
        count = int(ft_cv_p[i_ft, 1])
        if count == 0:
            continue
        cvs = ft_cv[start:start + count]
        if len(cvs) == 0:
            continue
        c0 = cv_reg[cvs[0] - 1]
        if c0 == 1 or c0 == 5:
            ft_reg[i_ft] = 1
        elif np.any(np.isin(cv_reg[cvs - 1], [2, 6])):
            ft_reg[i_ft] = 2
        else:
            ft_reg[i_ft] = 3
    grid.ft_reg = ft_reg

    # --- Poloidal flux (cvFpsi, fcFpsi; MATLAB lines 700-725) ---
    cv_fpsi = np.zeros(nCv, dtype=np.float64)
    if fpsi_data is not None and np.any(fpsi_data != 0):
        for iy in range(ny + 2):
            for ix in range(nx + 2):
                idx = imap_cv[ix, iy]
                if idx != 0:
                    cv_fpsi[idx - 1] = np.mean(fpsi_data[ix, iy, :])
    else:
        cv_fpsi = cv_ft.astype(np.float64)
    fc_fpsi = np.zeros(nFc, dtype=np.float64)
    for fc in range(nFc):
        fc_fpsi[fc] = 0.5 * (cv_fpsi[fc_cv[fc, 0] - 1] + cv_fpsi[fc_cv[fc, 1] - 1])
    grid.cv_fpsi = cv_fpsi
    grid.fc_fpsi = fc_fpsi

    # --- Convert connectivity arrays to 0-based (same convention as unstructured) ---
    for name in ["fc_cv", "fc_vx", "cv_fc", "cv_vx", "ft_cv", "ft_fc", "fs_fc", "vx_fc", "vx_cv"]:
        arr = getattr(grid, name, None)
        if arr is not None and np.any(arr > 0):
            arr = arr.copy()
            arr[arr > 0] -= 1
            setattr(grid, name, arr)
    if grid.cv_ft is not None and np.any(grid.cv_ft > 0):
        cv_ft_0b = grid.cv_ft.copy()
        cv_ft_0b[cv_ft_0b > 0] -= 1
        grid.cv_ft = cv_ft_0b

    return grid


def read_geometry(path: str | Path) -> GridTopology:
    """Read a b2fgmtry file and return a populated GridTopology.

    Auto-detects structured vs unstructured format.
    """
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        candidates = [path / "b2fgmtry"]
    else:
        candidates = [path]

    for level in range(4):
        base = path if path.is_dir() else path.parent
        for _ in range(level):
            base = base.parent
        candidates.append(base / "b2fgmtry")
        candidates.append(base / "baserun" / "b2fgmtry")

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as f:
                first_line = f.readline().strip()
                content_start = f.read(2000)  # peek at first 2KB

            # Determine format: unstructured b2fgmtry has fields like "nCi,nCg,nCv"
            is_structured = "nCi" not in content_start and "nCv" not in content_start

            if is_structured:
                log_msg(3, 2, "Reading structured b2fgmtry (3.0.x)")
                raw = _read_structured_b2fgmtry(str(candidate))
                return _load_structured_grid(raw)
            else:
                ver_str = first_line[7:].split()[0] if first_line.startswith("VERSION") else ""
                log_msg(3, 2, f"Reading unstructured b2fgmtry (v{ver_str})")
                raw = read_b2fgmtry(str(candidate))
                return _load_into_grid(raw)

    raise FileNotFoundError(f"Could not find b2fgmtry file from {path}")


def _load_into_grid(raw: dict[str, Any]) -> GridTopology:
    """Populate a GridTopology from the flat raw dict for unstructured format."""
    ver, num = _parse_version(raw.get("_version"))
    is_structured = "nx" in raw and "ny" in raw

    grid = GridTopology(
        n_cells=int(raw.get("nCv", 0)),
        n_faces=int(raw.get("nFc", 0)),
        n_vertices=int(raw.get("nVx", 0)),
        n_flux_surfaces=int(raw.get("nFs", 0)),
        n_flux_tubes=int(raw.get("nFt", 0)),
        n_core_cells=int(raw.get("nCi", 0)),
        is_structured=is_structured,
        nx=int(raw.get("nx", 0)),
        ny=int(raw.get("ny", 0)),
        nncut=int(raw.get("nncut", 0)),
        version=ver,
        version_number=num,
    )

    for src_name, dst_name in [
        ("cvX", "cv_x"), ("cvY", "cv_y"), ("cvVol", "cv_vol"),
        ("cvSz", "cv_sz"), ("cvHz", "cv_hz"), ("cvBb", "cv_bb"),
        ("cvEb", "cv_eb"),
        ("fcX", "fc_x"), ("fcY", "fc_y"), ("fcS", "fc_s"),
        ("fcHc", "fc_hc"), ("fcHt", "fc_ht"), ("fcBb", "fc_bb"),
        ("fcLbl", "fc_lbl"), ("fcReg", "fc_reg"),
        ("fcQalf", "fc_qalf"), ("fcQbet", "fc_qbet"), ("fcQgam", "fc_qgam"),
        ("cvQgam", "cv_qgam"), ("fcPbs", "fc_pbs"),
        ("vxX", "vx_x"), ("vxY", "vx_y"), ("vxFpsi", "vx_fpsi"),
        ("cvFcP", "cv_fc_p"), ("cvFc", "cv_fc"),
        ("cvVxP", "cv_vx_p"), ("cvVx", "cv_vx"),
        ("vxFcP", "vx_fc_p"), ("vxFc", "vx_fc"),
        ("vxCvP", "vx_cv_p"), ("vxCv", "vx_cv"),
        ("fcCv", "fc_cv"), ("fcVx", "fc_vx"),
        ("cvFt", "cv_ft"), ("cvReg", "cv_reg"),
        ("ftCvP", "ft_cv_p"), ("ftCv", "ft_cv"),
        ("ftFcP", "ft_fc_p"), ("ftFc", "ft_fc"),
        ("ftReg", "ft_reg"), ("cvConn", "cv_conn"),
        ("fsFcP", "fs_fc_p"), ("fsFc", "fs_fc"),
        ("fsPsi", "fs_psi"),
        ("intcellP", "intcell_p"), ("intcellR", "intcell_r"),
        ("imapCv", "imap_cv"), ("imapFcx", "imap_fcx"), ("imapFcy", "imap_fcy"),
    ]:
        if src_name in raw:
            val = raw[src_name]
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    setattr(grid, dst_name, val)
                else:
                    setattr(grid, dst_name, val)
            else:
                setattr(grid, dst_name, np.asarray(val, dtype=np.float64 if "float" in str(type(val)) else np.int32))

    # Reshape critical 2D arrays that may be stored as 1D in the file
    # These are stored column-major (Fortran order) in the file
    # Note: fcVx, fcCv etc are already mapped and must NOT be reshaped here
    # (they are read as separate _p arrays + flat lists)
    for name, expected_cols in [("fc_cv", 2), ("fc_vx", 2), ("fc_bb", 4), ("cv_bb", 4),
                                 ("fc_hc", 2), ("fc_qalf", 2), ("fc_qbet", 2), ("fc_qgam", 2),
                                 ("cv_eb", 3), ("cv_qgam", 2),
                                 ("cv_fc_p", 2), ("ft_cv_p", 2), ("ft_fc_p", 2),
                                 ("fs_fc_p", 2), ("vx_fc_p", 2), ("vx_cv_p", 2), ("cv_vx_p", 2),
                                 ("vx_bb", 4)]:
        arr = getattr(grid, name, None)
        if arr is not None and arr.ndim == 1 and arr.size > 0:
            n_rows = arr.size // expected_cols
            if n_rows * expected_cols == arr.size:
                setattr(grid, name, arr.reshape(n_rows, expected_cols, order='F'))

    # Convert connectivity arrays from 1-based (Fortran) to 0-based (Python)
    for name in ["fc_cv", "fc_vx", "cv_fc", "cv_ft", "cv_vx", "ft_cv", "ft_fc", "fs_fc",
                 "vx_fc", "vx_cv"]:
        arr = getattr(grid, name, None)
        if arr is not None and np.any(arr > 0):
            setattr(grid, name, np.where(arr > 0, arr - 1, arr))

    # Convert pointer arrays from 1-based to 0-based (subtract 1 from start column)
    for name in ["cv_fc_p", "ft_cv_p", "ft_fc_p", "fs_fc_p", "cv_vx_p", "vx_fc_p", "vx_cv_p"]:
        arr = getattr(grid, name, None)
        if arr is not None and arr.ndim == 2 and np.any(arr[:, 0] > 0):
            arr = arr.copy()
            arr[:, 0] = np.where(arr[:, 0] > 0, arr[:, 0] - 1, arr[:, 0])
            setattr(grid, name, arr)

    return grid


def read_b2fstati(path: str | Path, read_state_data: bool = False) -> GridTopology:
    """Read a b2fstati file, returning species/main information."""
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        candidates = [
            path / "b2fstati", path / "b2fstate",
            path.parent / "b2fstati", path.parent / "b2fstate",
            path / "baserun" / "b2fstate", path / "baserun" / "b2fstati",
        ]
    else:
        candidates = [path]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as f:
                raw = read_tagged_ascii_sections(f)

            ver, num = _parse_version(raw.get("_version"))

            grid = GridTopology(
                n_cells=int(raw.get("nCv", 0)),
                n_faces=int(raw.get("nFc", 0)),
                n_vertices=0, n_flux_surfaces=0, n_flux_tubes=0,
                n_core_cells=0,
                is_structured="nx" in raw,
                nx=int(raw.get("nx", 0)),
                ny=int(raw.get("ny", 0)),
                version=ver, version_number=num,
                n_species=int(raw.get("ns", 0)),
            )

            for key in ("zamin", "zamax", "zn", "am"):
                if key in raw:
                    setattr(grid, f"species_{key if key != 'zn' else 'n'}"
                            if key != 'zamin' and key != 'zamax' and key != 'am'
                            else f"species_{'charges' if key == 'zamin' else 'charge_max' if key == 'zamax' else 'n' if key == 'zn' else 'mass'}",
                            _ensure_float(raw[key]))
            return grid

    raise FileNotFoundError(f"Could not find b2fstati/b2fstate from {path}")
