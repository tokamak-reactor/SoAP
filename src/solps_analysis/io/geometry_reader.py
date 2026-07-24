"""Read SOLPS-ITER geometry from b2fgmtry files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from solps_analysis.core.grid import GridTopology
from solps_analysis.io.b2fgmtry_parser import read_b2fgmtry


def _parse_version(version_str: str | None) -> tuple[str, int]:
    """Parse version string like '03.002.001' into (version_str, numeric)."""
    if version_str is None:
        return "", 0
    # Expected format: "VERSION03.002.001 ..."
    if version_str.startswith("VERSION"):
        ver = version_str[7:].split()[0]  # get "03.002.001"
    else:
        ver = version_str
    parts = ver.split(".")
    try:
        num = int(parts[0]) * 100 + int(parts[1]) * 10 + int(parts[2])
    except (IndexError, ValueError):
        num = 0
    return ver, num


def _load_into_grid(raw: dict[str, Any]) -> GridTopology:
    """Populate a GridTopology from the flat raw dict returned by read_b2fgmtry."""
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
        n_species=0,
    )

    # --- Read cell quantities ---
    if "cvX" in raw:
        grid.cv_x = _ensure_float(raw["cvX"])
    if "cvY" in raw:
        grid.cv_y = _ensure_float(raw["cvY"])
    if "cvVol" in raw:
        grid.cv_vol = _ensure_float(raw["cvVol"])
    if "cvSz" in raw:
        grid.cv_sz = _ensure_float(raw["cvSz"])
    if "cvHz" in raw:
        grid.cv_hz = _ensure_float(raw["cvHz"])

    # --- Magnetic field at cells ---
    if "cvBb" in raw:
        arr = _ensure_float(raw["cvBb"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)
        grid.cv_bb = arr

    # --- Face quantities ---
    if "fcX" in raw:
        grid.fc_x = _ensure_float(raw["fcX"])
    if "fcY" in raw:
        grid.fc_y = _ensure_float(raw["fcY"])
    if "fcS" in raw:
        grid.fc_s = _ensure_float(raw["fcS"])
    if "fcHc" in raw:
        arr = _ensure_float(raw["fcHc"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.fc_hc = arr
    if "fcHt" in raw:
        grid.fc_ht = _ensure_float(raw["fcHt"])
    if "fcBb" in raw:
        arr = _ensure_float(raw["fcBb"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)
        grid.fc_bb = arr

    # --- Vertex coordinates ---
    if "vxX" in raw:
        grid.vx_x = _ensure_float(raw["vxX"])
    if "vxY" in raw:
        grid.vx_y = _ensure_float(raw["vxY"])
    if "vxFpsi" in raw:
        grid.vx_fpsi = _ensure_float(raw["vxFpsi"])

    # --- Connectivity ---
    if "cvFcP" in raw:
        arr = _ensure_int(raw["cvFcP"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.cv_fc_p = arr
    if "cvFc" in raw:
        grid.cv_fc = _ensure_int(raw["cvFc"])
    if "fcCv" in raw:
        arr = _ensure_int(raw["fcCv"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.fc_cv = arr
    if "fcVx" in raw:
        arr = _ensure_int(raw["fcVx"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.fc_vx = arr
    if "cvFt" in raw:
        grid.cv_ft = _ensure_int(raw["cvFt"])

    # --- Flux tube / surface ---
    if "ftCvP" in raw:
        arr = _ensure_int(raw["ftCvP"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.ft_cv_p = arr
    if "ftCv" in raw:
        grid.ft_cv = _ensure_int(raw["ftCv"])
    if "ftFcP" in raw:
        arr = _ensure_int(raw["ftFcP"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.ft_fc_p = arr
    if "ftFc" in raw:
        grid.ft_fc = _ensure_int(raw["ftFc"])
    if "ftReg" in raw:
        grid.ft_reg = _ensure_int(raw["ftReg"])
    if "fsFcP" in raw:
        arr = _ensure_int(raw["fsFcP"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        grid.fs_fc_p = arr
    if "fsFc" in raw:
        grid.fs_fc = _ensure_int(raw["fsFc"])
    if "fsPsi" in raw:
        grid.fs_psi = _ensure_float(raw["fsPsi"])

    # --- Region / label data ---
    if "cvReg" in raw:
        grid.cv_reg = _ensure_int(raw["cvReg"])
    if "fcReg" in raw:
        grid.fc_reg = _ensure_int(raw["fcReg"])
    if "fcLbl" in raw:
        grid.fc_lbl = _ensure_int(raw["fcLbl"])
    if "cvConn" in raw:
        grid.cv_conn = _ensure_float(raw["cvConn"])

    # --- Structured grid mapping ---
    if "imapCv" in raw:
        grid.imap_cv = _ensure_int(raw["imapCv"])
    if "imapFcx" in raw:
        grid.imap_fcx = _ensure_int(raw["imapFcx"])
    if "imapFcy" in raw:
        grid.imap_fcy = _ensure_int(raw["imapFcy"])

    # --- Species info (from b2fstati, not b2fgmtry, but stored here) ---
    if "zamin" in raw:
        grid.species_charges = _ensure_float(raw["zamin"])
        grid.n_species = len(grid.species_charges)
    if "zamax" in raw:
        grid.species_charge_max = _ensure_float(raw["zamax"])
    if "zn" in raw:
        grid.species_n = _ensure_float(raw["zn"])
    if "am" in raw:
        grid.species_mass = _ensure_float(raw["am"])

    return grid


def _ensure_float(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.ravel() if arr.ndim == 1 else arr
    return np.asarray(arr, dtype=np.float64)


def _ensure_int(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.ravel() if arr.ndim == 1 else arr
    return np.asarray(arr, dtype=np.int32)


def read_geometry(path: str | Path) -> GridTopology:
    """Read a b2fgmtry file and return a populated GridTopology.

    Searches up to 4 levels up for the file (same behaviour as MATLAB code).
    """
    path = Path(path).expanduser().resolve()

    # If path is a directory, look for b2fgmtry inside it
    if path.is_dir():
        candidates = [path / "b2fgmtry"]
    else:
        candidates = [path]

    # Search up to 4 levels up
    for level in range(4):
        base = path if path.is_dir() else path.parent
        for _ in range(level):
            base = base.parent
        candidates.append(base / "b2fgmtry")
        candidates.append(base / "baserun" / "b2fgmtry")
        candidates.append(base / "b2fgmtry")

    for candidate in candidates:
        if candidate.exists():
            raw = read_b2fgmtry(str(candidate))
            if raw:
                return _load_into_grid(raw)

    raise FileNotFoundError(
        f"Could not find b2fgmtry file searching from {path}"
    )


def read_b2fstati(path: str | Path, read_state_data: bool = False) -> GridTopology:
    """Read a b2fstati file and return species/main information from it.

    When read_state_data is True, also reads the full plasma state arrays
    (na, ne, te, ti, etc.) which are embedded in the b2fstati file.
    For lightweight reading (determining version/species), leave as False.
    """
    from solps_analysis.io.b2fgmtry_parser import read_tagged_ascii_sections

    path = Path(path).expanduser().resolve()

    # Search for b2fstati / b2fstate
    if path.is_dir():
        candidates = [
            path / "b2fstati",
            path / "b2fstate",
            path.parent / "b2fstati",
            path.parent / "b2fstate",
            path / "baserun" / "b2fstate",
            path / "baserun" / "b2fstati",
        ]
    else:
        candidates = [path]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as f:
                raw = read_tagged_ascii_sections(f)

            ver, num = _parse_version(raw.get("_version"))

            # Build a partial GridTopology with the state info
            grid = GridTopology(
                n_cells=int(raw.get("nCv", 0)),
                n_faces=int(raw.get("nFc", 0)),
                n_vertices=0,
                n_flux_surfaces=0,
                n_flux_tubes=0,
                n_core_cells=0,
                is_structured="nx" in raw,
                nx=int(raw.get("nx", 0)),
                ny=int(raw.get("ny", 0)),
                version=ver,
                version_number=num,
                n_species=int(raw.get("ns", 0)),
            )

            if "zamin" in raw:
                grid.species_charges = _ensure_float(raw["zamin"])
            if "zamax" in raw:
                grid.species_charge_max = _ensure_float(raw["zamax"])
            if "zn" in raw:
                grid.species_n = _ensure_float(raw["zn"])
            if "am" in raw:
                grid.species_mass = _ensure_float(raw["am"])

            return grid

    raise FileNotFoundError(f"Could not find b2fstati/b2fstate file from {path}")
