"""Read EIRENE ``input.dat`` files.

The EIRENE input file is a structured text file with numbered sections
(``*** 3a.``, ``*** 3b.``, ``*** 6a.``, ``*** 7.``, etc.) that define the
geometry, surfaces, materials, and sources for an EIRENE simulation.

Key sections parsed by this module:

- **Block 3a**: Standard surfaces (geometry boundaries, vessel walls, etc.)
- **Block 3b**: Additional (limiter) surfaces defined by coordinates
- **Block 6a**: SURFMOD material definitions (mass -> element mapping)
- **Block 7**: Gas puffing sources (species and their associated surfaces)

This module replicates the functionality of the original MATLAB
``read_input.m``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Mass number -> element symbol lookup (same as get_plasma_composition.m)
# ---------------------------------------------------------------------------

MASS_TO_ELEMENT: dict[int, str] = {
    1: "H",
    2: "D",
    3: "T",
    4: "He",
    7: "Li",
    9: "Be",
    11: "B",
    12: "C",
    14: "N",
    20: "Ne",
    40: "Ar",
    84: "Kr",
    132: "Xe",
    184: "W",
}


def mass_to_element(mass_num: int) -> str:
    """Convert an integer mass number to an element symbol.

    Parameters
    ----------
    mass_num:
        Atomic mass number (rounded from mass × 100 read from SURFMOD).

    Returns
    -------
    str
        Element symbol (e.g. ``'D'``, ``'C'``, ``'W'``), or
        ``'M<mass>'`` for unrecognised masses.
    """
    return MASS_TO_ELEMENT.get(mass_num, f"M{mass_num}")


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_eirene_input(
    path: str | Path,
) -> dict[str, Any]:
    """Read an EIRENE ``input.dat`` file.

    Parameters
    ----------
    path:
        Path to the ``input.dat`` file (or a directory containing it).

    Returns
    -------
    dict
        Dictionary with the following keys:

        **Geometry / surfaces**

        - ``nstd``: number of standard surfaces (block 3a)
        - ``surface_3a``: list of dicts, one per standard surface, each with:
            - ``index``: surface index (from ``data1[-1]``)
            - ``description``: surface description string
            - ``data1``: 1-D array of floats (10–11 values)
            - ``data2``: 1-D array of floats (10 values)
            - ``material``: SURFMOD material name string (``''`` if none)
            - ``element``: element symbol decoded from SURFMOD (``''`` if none)
        - ``element_3a``: list of element symbols per surface
        - ``nlim``: number of additional (limiter) surfaces (block 3b)
        - ``surface``: list of coordinate arrays for block 3b surfaces
        - ``surface_ind``: ``(nlim, 2)`` array with ``[surface_index, ilim]``

        **Gas puffing (block 7)**

        - ``puff_species``: list of species names for gas puffs
        - ``puff_ilims``: list of lists, ilim indices per puff species
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "input.dat"

    if not path.exists():
        raise FileNotFoundError(f"EIRENE input file not found: {path}")

    result: dict[str, Any] = {
        "nstd": 0,
        "surface_3a": [],
        "element_3a": [],
        "nlim": 0,
        "surface": [],
        "surface_ind": np.empty((0, 2), dtype=np.int32),
        "puff_species": [],
        "puff_ilims": [],
    }

    with path.open("r") as f:
        _read_block_3a(f, result)
        _read_block_6a(f, result)
        _read_block_3b(f, result)
        _read_block_7(f, result)

    return result


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------


def _seek_to_section(f, marker: str) -> Optional[str]:
    """Advance *f* until a line containing *marker* is found.

    Returns the matching line (stripped), or ``None`` if EOF is reached.
    """
    while True:
        line = f.readline()
        if not line:
            return None
        stripped = line.strip()
        if marker in stripped:
            return stripped


def _read_block_3a(f, result: dict[str, Any]) -> None:
    """Read *** 3a. Standard surfaces."""
    header = _seek_to_section(f, "*** 3a.")
    if header is None:
        import warnings

        warnings.warn("Block *** 3a. not found in EIRENE input file")
        return

    # Read nstd
    line = f.readline()
    while line and not line.strip():
        line = f.readline()
    if not line:
        return
    nstd = int(line.strip())
    result["nstd"] = nstd

    surfaces: list[dict[str, Any]] = []

    for i in range(nstd):
        # Find the next * header line (skip blank lines and *** lines)
        line = f.readline()
        header_line = ""
        while line:
            stripped = line.strip()
            if stripped.startswith("*") and not stripped.startswith("***"):
                header_line = stripped
                break
            line = f.readline()
        if not line or not header_line:
            raise ValueError(f"EOF reached while reading standard surface {i + 1} header")
        # Extract description after the colon
        desc = ""
        if ":" in header_line:
            desc = header_line.split(":", 1)[1].strip()

        # Data line 1 (10-11 floats)
        line = f.readline()
        while line and not line.strip():
            line = f.readline()
        data1 = np.fromstring(line.strip(), sep=" ", dtype=np.float64) if line else np.array([])

        # Data line 2 (10 floats)
        line = f.readline()
        while line and not line.strip():
            line = f.readline()
        data2 = np.fromstring(line.strip(), sep=" ", dtype=np.float64) if line else np.array([])

        # Peek at next line for optional material name
        pos = f.tell()
        next_line = f.readline()
        material = ""
        if next_line:
            ns = next_line.strip()
            if ns and not ns.startswith("*"):
                material = ns
            else:
                # Rewind — next line is a section marker
                f.seek(pos)

        surface: dict[str, Any] = {
            "index": int(data1[-1]) if len(data1) > 0 else 0,
            "description": desc,
            "data1": data1,
            "data2": data2,
            "material": material,
            "element": "",
        }
        surfaces.append(surface)

    result["surface_3a"] = surfaces


def _read_block_6a(f, result: dict[str, Any]) -> None:
    """Read *** 6a. SURFMOD material definitions.

    Maps material names to element symbols by decoding the mass number
    from the SURFMOD data.
    """
    # Save current position so we can restore it for block 3b
    pos_before = f.tell()

    header = _seek_to_section(f, "*** 6a.")
    if header is None:
        # No SURFMOD data
        nstd = result.get("nstd", 0)
        result["element_3a"] = [""] * nstd
        for s in result.get("surface_3a", []):
            s["element"] = ""
        f.seek(pos_before)
        return

    surfmod_names: list[str] = []
    surfmod_elements: list[str] = []

    # Skip preamble until first SURFMOD
    line = f.readline()
    while line:
        stripped = line.strip()
        if stripped.startswith("SURFMOD"):
            break
        line = f.readline()

    # Read all SURFMOD entries
    while line and line.strip().startswith("SURFMOD"):
        name = line.strip()

        # Skip integer line
        line = f.readline()
        if not line:
            break

        # First double line — first value = mass * 100
        line = f.readline()
        if not line:
            break
        try:
            mass_num = int(float(line.strip().split()[0]) / 100)
        except (ValueError, IndexError):
            mass_num = 0

        element = mass_to_element(mass_num)
        surfmod_names.append(name)
        surfmod_elements.append(element)

        # Skip remaining 2 double lines
        for _ in range(2):
            line = f.readline()
            if not line:
                break

        # Read next SURFMOD or exit
        line = f.readline()

    # Build element_3a by matching material names
    name_to_element = dict(zip(surfmod_names, surfmod_elements))
    nstd = result.get("nstd", 0)
    elements: list[str] = []
    for i in range(nstd):
        mat = result["surface_3a"][i]["material"]
        el = name_to_element.get(mat, "")
        elements.append(el)
        result["surface_3a"][i]["element"] = el

    result["element_3a"] = elements

    # Restore to position before block 6a for next reader
    f.seek(pos_before)


def _read_block_3b(f, result: dict[str, Any]) -> None:
    """Read *** 3b. Additional (limiter) surfaces with coordinates."""
    header = _seek_to_section(f, "*** 3b.")
    if header is None:
        raise ValueError("EOF reached without finding *** 3b.")

    line = f.readline()
    while line and not line.strip():
        line = f.readline()
    if not line:
        return

    nlim = int(line.strip())
    result["nlim"] = nlim

    surfaces: list[np.ndarray] = []
    surface_indices: list[list[int]] = []

    for i in range(nlim):
        # Find * header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"EOF reached without finding surface {i + 1} header")
            if "*" in line.strip():
                break

        surface_str_1 = line.strip()

        # Skip 2 lines (indices)
        f.readline()
        f.readline()

        # Coordinate data line
        line = f.readline()
        if not line:
            raise ValueError(f"EOF reached reading coordinates for surface {i + 1}")

        surface_str_2 = line.strip()

        # Parse 6 coordinates, each in 12-char field width, divided by 100 (cm -> m)
        coords = np.zeros(6)
        for k in range(6):
            start = 12 * k
            end = start + 12
            try:
                val = float(surface_str_2[start:end].strip()) / 100.0
            except ValueError:
                val = 0.0
            coords[k] = val
        surfaces.append(coords)

        # Parse surface index from header
        try:
            # Format: "*<index> : <description>"
            numbers = surface_str_1.replace("*", "").split(":")
            idx = int(numbers[0].strip())
            desc_part = numbers[1] if len(numbers) > 1 else ""
            # Extract digits from description part for ilim
            digits = re.findall(r"\d+", desc_part)
            ilim = int(digits[0]) if digits else i + 1
            surface_indices.append([idx, ilim])
        except (ValueError, IndexError):
            import warnings

            warnings.warn(f"Could not parse surface index for ilim={i + 1}")
            surface_indices.append([i + 1, i + 1])

    result["surface"] = surfaces
    result["surface_ind"] = np.array(surface_indices, dtype=np.int32)


def _read_block_7(f, result: dict[str, Any]) -> None:
    """Read *** 7. Gas puffing sources."""
    header = _seek_to_section(f, "*** 7.")
    if header is None:
        import warnings

        warnings.warn("EOF reached without finding *** 7.")
        return

    puff_species: list[str] = []
    puff_ilims: list[list[int]] = []

    while True:
        # Find next "Gas puffing source" line, or exit on *** 8.
        found = False
        while True:
            line = f.readline()
            if not line:
                return
            stripped = line.strip()
            if "*** 8." in stripped:
                result["puff_species"] = puff_species
                result["puff_ilims"] = puff_ilims
                return
            if "Gas puffing source" in stripped:
                found = True
                break

        if not found:
            break

        # Parse species name
        # Format: "Gas puffing source : <species>"
        after = stripped.split("Gas puffing source", 1)[1]
        parts = after.split(":")
        if len(parts) >= 2:
            species_name = parts[1].strip()
        else:
            species_name = parts[0].strip() if parts else ""
        puff_species.append(species_name)

        # Skip 7 lines to reach the number of puff surfaces
        nlims_line = ""
        for _ in range(7):
            nlims_line = f.readline()
            if not nlims_line:
                break

        try:
            nlims = int(nlims_line.strip()) if nlims_line else 0
        except (ValueError, TypeError):
            nlims = 0

        ilm_list: list[int] = []
        for _ in range(nlims):
            info_line = f.readline()
            if not info_line:
                break
            info_parts = info_line.strip().split()
            if len(info_parts) >= 3:
                try:
                    ilim = int(info_parts[2])
                    ilm_list.append(ilim)
                except ValueError:
                    pass
            # Skip 3 metadata lines
            for _ in range(3):
                f.readline()

        puff_ilims.append(ilm_list)

    result["puff_species"] = puff_species
    result["puff_ilims"] = puff_ilims


# ---------------------------------------------------------------------------
# Convenience: read b2.numerics.parameters as well
# ---------------------------------------------------------------------------

def read_user_parameters(
    path: str | Path,
    pfr_cvs: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Read a SOLPS user parameters file (``b2.numerics.parameters``-style).

    This replicates the MATLAB ``read_user_parameters.m`` which reads custom
    parameters like PFR definitions.

    Parameters
    ----------
    path:
        Path to the user parameters file.
    pfr_cvs:
        Optional default list of CV indices to use if the file does not
        define ``pfr_cvs``.

    Returns
    -------
    dict
        Dictionary with fields:

        - ``pfr_cvs``: list of CV indices in the private flux region
        - ``lpfrb_i``: inner PFR boundary index (+2 offset)
        - ``lpfrb_o``: outer PFR boundary index (+2 offset)
        - ``lpfrt_i``: inner PFR target index (+2 offset)
        - ``lpfrt_o``: outer PFR target index (+2 offset)
        - ``pr_weight``: placeholder for future weighting
    """
    path = Path(path).expanduser().resolve()

    result: dict[str, Any] = {
        "pfr_cvs": np.array(pfr_cvs if pfr_cvs is not None else [], dtype=np.int32),
        "lpfrb_i": 0,
        "lpfrb_o": 0,
        "lpfrt_i": 0,
        "lpfrt_o": 0,
        "pr_weight": np.array([], dtype=np.float64),
    }

    if not path.exists():
        return result

    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("!") or raw.startswith("#") or raw.startswith("c"):
                continue

            # npfr_cvs — ignore
            if "npfr_cvs" in raw:
                continue

            # pfr_cvs — comma-separated list
            if "pfr_cvs" in raw:
                raw_clean = raw.replace("=", ",")
                parts = raw_clean.split(",")
                cvs: list[int] = []
                for p in parts:
                    try:
                        cvs.append(int(p))
                    except ValueError:
                        pass
                result["pfr_cvs"] = np.array(cvs, dtype=np.int32)

            # LPFRB_I / LPFRB_O / LPFRT_I / LPFRT_O
            for key, attr in [
                ("LPFRB_I", "lpfrb_i"),
                ("LPFRB_O", "lpfrb_o"),
                ("LPFRT_I", "lpfrt_i"),
                ("LPFRT_O", "lpfrt_o"),
            ]:
                if key in raw:
                    digits = re.findall(r"\d+", raw)
                    if digits:
                        result[attr] = int(digits[-1]) + 2

    # Build full PFR CV range if boundaries are defined
    if result["lpfrb_i"] != 0:
        pfr_range = list(range(result["lpfrb_i"], result["lpfrt_i"] + 1)) + \
                    list(range(result["lpfrt_o"], result["lpfrb_o"] + 1))
        result["pfr_cvs"] = np.array(pfr_range, dtype=np.int32)

    return result
