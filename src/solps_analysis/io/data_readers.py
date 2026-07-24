"""Readers for SOLPS-ITER watch .dat files (output/ directory).

Two formats:
  - Structured (3.0.x): 2D matrix with header row + row-indexed lines
  - Unstructured (3.2.x): Linear index+value format
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def read_structured_dat(path: str | Path) -> np.ndarray:
    """Read a structured-format .dat file.

    Format:
      Line 1: header with column indices: '-1 0 1 2 ... nx-1'
      Lines 2+: row index followed by space-delimited values

    Returns a 2D array (n_rows, n_cols) where row 0 is the original last row
    (the matrix is flipped so that index ordering is preserved).
    Column 0 (index -1) is discarded as it contains boundary data.
    """
    path = Path(path)
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    values = data[:, 1:]
    values = np.flipud(values)
    return values


def read_unstructured_dat(path: str | Path) -> np.ndarray:
    """Read an unstructured-format .dat file.

    Format: each line is 'index  value'
    Returns a 1D array of values, indexed 0-based.
    """
    path = Path(path)
    data = np.loadtxt(path)
    if data.ndim == 1:
        return np.array([data[1]])
    return data[:, 1]


def read_watch_file(filepath: str | Path) -> np.ndarray:
    """Auto-detect and read a single .dat file.

    Detection: if 2 columns -> unstructured; if >2 -> structured.
    """
    filepath = Path(filepath)
    raw = np.loadtxt(filepath)
    if raw.ndim == 1:
        return np.array([raw[1]]) if len(raw) == 2 else raw
    if raw.shape[1] == 2:
        return raw[:, 1]
    else:
        return read_structured_dat(filepath)


def _parse_dat_filename(filename: str) -> tuple[str, int | None] | None:
    """Parse a .dat filename into (canonical_name, species_index_or_None).

    Pattern examples:
      b2npc11_na001.dat   -> ('b2npc11_na', 1)
      b2npco_na000_us.dat -> ('b2npco_na', 0)
      b2news__fhtx.dat    -> ('b2news_fhtx', None)
      b2nph9_fhex.dat     -> ('b2nph9_fhex', None)
      b2tfnb_fchanml_bx000.dat -> ('b2tfnb_fchanml_bx', 0)
      te_eV.dat           -> ('te_eV', None)
      Cv.B_full_us.dat    -> ('Cv.B_full', None)
    """
    import re

    name = filename
    if name.endswith(".dat"):
        name = name[:-4]
    if name.endswith("_us"):
        name = name[:-3]
    if name.startswith("__1__") or name == "info":
        return None

    # Pattern 1: b2XXXX__varname  (double underscore, no species)
    m = re.match(r"^(b2\w+)__(\w+)$", name)
    if m:
        return (f"{m.group(1)}_{m.group(2)}", None)

    # Pattern 2: b2XXXX_varnameNNN  (species-indexed, may have complex varname)
    m = re.match(r"^(b2\w+)_(.+?)(\d{3})$", name)
    if m:
        source = m.group(1)
        varname = m.group(2).rstrip("_")
        idx = int(m.group(3))
        return (f"{source}_{varname}", idx)

    # Pattern 3: b2XXXX_varname  (single variable)
    m = re.match(r"^(b2\w+)_(\w+)$", name)
    if m:
        return (f"{m.group(1)}_{m.group(2)}", None)

    # Pattern 4: Dotted names like Cv.B_full
    if re.match(r"^\w+\.\w+$", name):
        return (name, None)

    # Pattern 5: Standalone name
    if re.match(r"^[a-zA-Z_][\w.]*$", name):
        return (name, None)

    return None


def scan_output_directory(output_path: str | Path) -> dict[str, str]:
    """Scan an output/ directory and return {variable_name: file_path} mapping.

    Variable naming preserves the source prefix (e.g. 'b2npc11_na', 'te_eV').
    Species-indexed variables get separate entries: 'b2npc11_na_001', etc.
    """
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise NotADirectoryError(f"{output_path} is not a directory")

    groups: dict[str, list[tuple[str, int | None]]] = {}

    for fpath in sorted(output_path.iterdir()):
        if not fpath.is_file() or not fpath.suffix == ".dat":
            continue
        info = _parse_dat_filename(fpath.name)
        if info is None:
            continue
        base_name, species_idx = info
        groups.setdefault(base_name, []).append((str(fpath.resolve()), species_idx))

    files: dict[str, str] = {}
    for base_name, entries in groups.items():
        if len(entries) == 1:
            files[base_name] = entries[0][0]
        else:
            has_indices = any(idx is not None for _, idx in entries)
            if has_indices:
                for fpath, idx in entries:
                    key = f"{base_name}_{idx:03d}" if idx is not None else base_name
                    files[key] = fpath
            else:
                files[base_name] = entries[0][0]

    return files
