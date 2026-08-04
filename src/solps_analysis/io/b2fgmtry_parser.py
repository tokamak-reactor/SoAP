"""Parser for the SOLPS b2fgmtry / b2fstati tagged-ASCII format.

The format uses sections starting with:
    *cf: <type> <count> <field_name(s)>
followed by the data.
Types: int, real (float), char (string), logical (Fortran T/F), array (complex).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, TextIO

import numpy as np


def _parse_count_header(line: str) -> tuple[str, int, str]:
    """Parse a '*cf:' header line.

    Returns (type_name, count, field_names).
    Examples:
        '*cf:    int                7    nCi,nCg,nCv,nFc,nVx,nFs,nFt'
        '*cf:    real               9    zamin'
        '*cf:    char             120    label'
    """
    parts = line.split(None, 3)  # split by whitespace, max 3 splits
    if len(parts) < 4:
        raise ValueError(f"Malformed header line: {line!r}")
    type_name = parts[1]
    count = int(parts[2])
    field_names = parts[3].strip()
    return type_name, count, field_names


def read_tagged_ascii_sections(file: TextIO) -> dict[str, Any]:
    """Read a tagged-ASCII file and return a dict of field_name -> value.

    Single values: str for 'char', int for 'int', float for 'real'
    Arrays: numpy arrays.
    """
    # Skip VERSION line or any non-*cf lines at the start
    first_line = file.readline()
    # Read version if present
    version = None
    if first_line.startswith("VERSION"):
        version = first_line.strip()
    else:
        # Put it back by seeking? No, need to buffer. Simpler: handle inline.
        # Actually, let's just keep the version separate.
        pass

    result: dict[str, Any] = {}
    result["_version"] = version

    while True:
        line = file.readline()
        if not line:  # EOF
            break
        line = line.rstrip("\n")
        if not line:
            continue
        if not line.startswith("*cf:"):
            continue

        type_name, count, field_names = _parse_count_header(line)

        if type_name == "char":
            # Read one line for the string
            value = file.readline().rstrip("\n")
            result[field_names] = value
        elif type_name == "int":
            # Read count integers (may span multiple lines)
            data_text = _read_data_lines(file, count, dtype="int")
            values = np.fromstring(data_text, sep=" ", dtype=np.int32, count=count)
            if count == 1:
                result[field_names] = int(values[0])
            else:
                result[field_names] = values
        elif type_name == "real":
            data_text = _read_data_lines(file, count, dtype="float")
            values = np.fromstring(data_text, sep=" ", dtype=np.float64, count=count)
            if count == 1:
                result[field_names] = float(values[0])
            else:
                result[field_names] = values
        elif type_name in ("logical",):
            # Fortran logical: T/F as characters
            data_text = _read_data_lines(file, count, dtype="str")
            # We'll store as a string for now
            result[field_names] = data_text.strip()
        else:
            # Unknown type - skip count items by reading raw
            _read_data_lines(file, count, dtype="skip")

    return result


def _read_data_lines(file: TextIO, count: int, dtype: str) -> str:
    """Read lines of data until we've collected 'count' tokens or hit the next *cf:"""
    tokens: list[str] = []
    while len(tokens) < count:
        pos = file.tell()
        line = file.readline()
        if not line:  # EOF
            break
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped.startswith("*cf:"):
            # Oops, we read the next header. Rewind.
            file.seek(pos)
            break
        tokens.extend(line_stripped.split())
    return " ".join(tokens)


def read_b2fgmtry(path: str | Path) -> dict[str, Any]:
    """Read a b2fgmtry geometry file, returning a flat dict of all fields."""
    path = Path(path)
    with path.open("r") as f:
        data = read_tagged_ascii_sections(f)

    # Build dimension metadata
    result: dict[str, Any] = {}
    result["_version"] = data.get("_version")

    # Copy everything, but also resolve named fields with commas
    for key, value in data.items():
        if key == "_version":
            continue
        # Fields can be comma-separated: "nCi,nCg,nCv,nFc,nVx,nFs,nFt"
        names = [n.strip() for n in key.split(",")]
        if len(names) == 1:
            result[names[0]] = value
        elif isinstance(value, np.ndarray) and value.size == len(names):
            for i, name in enumerate(names):
                result[name] = value[i]
        else:
            result[key] = value

    return result
