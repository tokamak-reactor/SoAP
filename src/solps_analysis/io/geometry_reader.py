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

    # --- Cell-centered data ---
    cv_bb = np.zeros((nCv, 4), dtype=np.float64)
    cv_x = np.zeros(nCv, dtype=np.float64)
    cv_y = np.zeros(nCv, dtype=np.float64)
    cv_vol_arr = np.zeros(nCv, dtype=np.float64)
    cv_qgam = np.zeros((nCv, 2), dtype=np.float64)
    cv_hz_arr = np.zeros(nCv, dtype=np.float64)

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

    grid.cv_bb = cv_bb
    grid.cv_x = cv_x
    grid.cv_y = cv_y
    grid.cv_vol = cv_vol_arr
    grid.cv_hz = cv_hz_arr

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

    # --- Face data ---
    fc_cv = np.zeros((nFc, 2), dtype=np.int32)
    fc_vx = np.zeros((nFc, 2), dtype=np.int32)
    fc_x = np.zeros(nFc, dtype=np.float64)
    fc_y = np.zeros(nFc, dtype=np.float64)
    fc_s = np.zeros(nFc, dtype=np.float64)
    fc_hc = np.zeros((nFc, 2), dtype=np.float64)
    fc_ht = np.zeros(nFc, dtype=np.float64)
    fc_bb = np.zeros((nFc, 4), dtype=np.float64)
    fc_qalf = np.zeros((nFc, 2), dtype=np.float64)
    fc_qgam = np.zeros((nFc, 2), dtype=np.float64)
    fc_qbet = np.zeros((nFc, 2), dtype=np.float64)
    fc_pbs = np.zeros(nFc, dtype=np.float64)

    grid.fc_cv = fc_cv
    grid.fc_vx = fc_vx
    grid.fc_x = fc_x
    grid.fc_y = fc_y
    grid.fc_s = fc_s
    grid.fc_hc = fc_hc
    grid.fc_ht = fc_ht
    grid.fc_bb = fc_bb

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
        ("fcX", "fc_x"), ("fcY", "fc_y"), ("fcS", "fc_s"),
        ("fcHc", "fc_hc"), ("fcHt", "fc_ht"), ("fcBb", "fc_bb"),
        ("fcLbl", "fc_lbl"), ("fcReg", "fc_reg"),
        ("vxX", "vx_x"), ("vxY", "vx_y"), ("vxFpsi", "vx_fpsi"),
        ("cvFcP", "cv_fc_p"), ("cvFc", "cv_fc"),
        ("fcCv", "fc_cv"), ("fcVx", "fc_vx"),
        ("cvFt", "cv_ft"), ("cvReg", "cv_reg"),
        ("ftCvP", "ft_cv_p"), ("ftCv", "ft_cv"),
        ("ftFcP", "ft_fc_p"), ("ftFc", "ft_fc"),
        ("ftReg", "ft_reg"), ("cvConn", "cv_conn"),
        ("fsFcP", "fs_fc_p"), ("fsFc", "fs_fc"),
        ("fsPsi", "fs_psi"),
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
