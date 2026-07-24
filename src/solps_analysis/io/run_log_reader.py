"""Read SOLPS-ITER run.log and b2.numerics.parameter files.

Both are plain-text key-value files in a simple Fortran namelist-like format.

run.log
-------
Contains one key-value entry per line of the form::

    ITER = 1000, TIME = 0.5000E+00, NTIM = 1000, DTIM = 0.5000E-03

or::

    Gas puff strength, atom: 7.8343E+15 (index 2)

The MATLAB reader (``read_run_log.m``) searches backwards through the file
for the last occurrence of ITER/TIME/NTIM and Gas puff lines.

b2.numerics.parameters
-----------------------
A simple Fortran namelist where each line contains ``key = value`` pairs::

    ntime       = 2000
    dt          = 1.0e-05
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# run.log reader
# ---------------------------------------------------------------------------

def read_run_log(path: str | Path) -> dict[str, Any]:
    """Read a SOLPS-ITER ``run.log`` file.

    Parameters
    ----------
    path:
        Path to the ``run.log`` file (or a directory containing it).
        If a directory is given, looks for ``run.log`` inside it.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``time``: last physical time in the log (float), or -1 if not found
        - ``iter``: last iteration number (int)
        - ``ntim``: last NTIM value (int)
        - ``dtim``: last DTIM value (float)
        - ``gas_puff``: dict mapping species index (int) -> total puffed particles
        - ``gas_puff_parts``: list of individual puff contributions (in order
          encountered scanning backwards, i.e. most recent first)
        - ``raw_lines``: complete list of non-empty lines from the file
    """
    path = _resolve(path, "run.log")

    result: dict[str, Any] = {
        "time": -1.0,
        "iter": -1,
        "ntim": -1,
        "dtim": -1.0,
        "gas_puff": {},
        "gas_puff_parts": [],
        "raw_lines": [],
    }

    if not path.exists():
        import warnings
        warnings.warn(f"run.log not found: {path}")
        return result

    with path.open("r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    result["raw_lines"] = lines.copy()

    # Search backwards (last occurrence has the final state)
    itim = -1
    ntim = -1

    for line in reversed(lines):
        # Match ITER/TIME/NTIM/DTIM line
        if "ITER" in line and "TIME" in line and "NTIM" in line:
            try:
                _iter = _extract_int(line, "ITER")
                _time = _extract_float(line, "TIME")
                _ntim = _extract_int(line, "NTIM")
                _dtim = _extract_float(line, "DTIM")

                if result["time"] == -1.0 and _time is not None:
                    result["time"] = _time
                if _iter is not None:
                    itim = _iter
                    if result["iter"] == -1:
                        result["iter"] = _iter
                if _ntim is not None:
                    ntim = _ntim
                    if result["ntim"] == -1:
                        result["ntim"] = _ntim
                if _dtim is not None and result["dtim"] == -1.0:
                    result["dtim"] = _dtim
            except (ValueError, IndexError):
                continue

        # Match Gas puff line
        elif "Gas puff strength" in line:
            try:
                # Format: "Gas puff strength, atom: <value> (index <idx>)"
                # or sometimes "Gas puff strength, atom: <value>"
                after_colon = line.split(":", 1)[1].strip()
                # Split into value and possible parenthesised index
                parts = after_colon.split()
                if parts:
                    value = float(parts[0])
                    result["gas_puff_parts"].append(value)

                    # Look for "(index N)" or "(N)"
                    idx_match = re.search(r"index\s+(\d+)", after_colon)
                    if idx_match:
                        idx = int(idx_match.group(1))
                        result["gas_puff"][idx] = (
                            result["gas_puff"].get(idx, 0.0) + value
                        )
                    # If there's a parenthesised number without "index"
                    else:
                        paren_match = re.search(r"\((\d+)\)", after_colon)
                        if paren_match:
                            idx = int(paren_match.group(1))
                            result["gas_puff"][idx] = (
                                result["gas_puff"].get(idx, 0.0) + value
                            )
            except (ValueError, IndexError):
                continue

    # gas_puff_parts is reversed (most recent first from backwards scan)
    result["gas_puff_parts"] = result["gas_puff_parts"]

    return result


# ---------------------------------------------------------------------------
# b2.numerics.parameters reader
# ---------------------------------------------------------------------------

def read_numerics_parameters(path: str | Path) -> dict[str, Any]:
    """Read a SOLPS-ITER ``b2.numerics.parameters`` file.

    The file is a simple Fortran namelist with one key = value per line::

        ntime       = 2000
        dt          = 1.0e-05
        nstop       = 2000
        text_output = 1
        nframe      = -1

    Comment lines start with ``!``, ``#``, or ``c``.  Inline comments after
    a ``!`` are stripped.

    Parameters
    ----------
    path:
        Path to the ``b2.numerics.parameters`` file (or a directory
        containing it).

    Returns
    -------
    dict
        Dictionary of parameter name -> value.  Numeric values are converted
        to ``int`` or ``float``; string values are returned as-is.
    """
    path = _resolve(path, "b2.numerics.parameters")

    result: dict[str, Any] = {}

    if not path.exists():
        import warnings
        warnings.warn(f"b2.numerics.parameters not found: {path}")
        return result

    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            # Skip blank / comment lines
            if not raw or raw.startswith("!") or raw.startswith("#") or raw.startswith("c"):
                continue
            # Strip inline comments (Fortran !)
            if "!" in raw:
                raw = raw.split("!")[0].strip()
            # Parse key = value
            if "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            key = key.strip().lower()
            val = val.strip()

            # Try numeric conversion
            result[key] = _convert_value(val)

    return result


# ---------------------------------------------------------------------------
# Generic key-value file reader
# ---------------------------------------------------------------------------

def read_key_value_file(
    path: str | Path,
    *,
    comment_chars: str = "!#c",
    separator: str = "=",
    lowercase_keys: bool = True,
) -> dict[str, Any]:
    """Read a generic Fortran-style key-value file.

    Lines should be of the form::

        key = value

    Comment lines starting with any character in *comment_chars* are skipped.
    Inline comments (``!``) after the value are also stripped.

    Parameters
    ----------
    path:
        Path to the file.
    comment_chars:
        Characters that indicate a comment line.
    separator:
        The character(s) separating key and value.
    lowercase_keys:
        If *True*, convert all key names to lowercase.

    Returns
    -------
    dict
        Dictionary of parameter name -> value.  Numeric values are converted
        to ``int`` or ``float``; strings are returned as-is.
    """
    path = Path(path).expanduser().resolve()
    result: dict[str, Any] = {}
    if not path.exists():
        return result

    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw[0] in comment_chars:
                continue
            # Strip inline Fortran comment
            if "!" in raw:
                raw = raw.split("!")[0].strip()
            if separator not in raw:
                continue
            key, val = raw.split(separator, 1)
            key = key.strip()
            val = val.strip()
            if lowercase_keys:
                key = key.lower()
            result[key] = _convert_value(val)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str | Path, filename: str) -> Path:
    """If *path* is a directory, append *filename*; otherwise use as-is."""
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        return p / filename
    return p


def _convert_value(val: str) -> Any:
    """Convert a string value to int, float, or leave as str."""
    val = val.strip()
    # Remove trailing Fortran-style suffixes like _r, _dp
    val = re.sub(r"_\w+$", "", val)

    # Try int first
    try:
        # Handle Fortran scientific notation like 1.0e-05
        return int(val)
    except ValueError:
        pass

    # Try float
    try:
        return float(val)
    except ValueError:
        pass

    # Handle Fortran logicals
    upper = val.upper()
    if upper in (".TRUE.", "T", "TRUE", ".T."):
        return True
    if upper in (".FALSE.", "F", "FALSE", ".F."):
        return False

    # Return as string
    return val


def _extract_float(line: str, keyword: str) -> Optional[float]:
    """Extract a float value following *keyword* in a key=value line."""
    m = re.search(rf"\b{keyword}\s*=\s*([\d\.Ee+\-]+)", line)
    if m:
        return float(m.group(1))
    return None


def _extract_int(line: str, keyword: str) -> Optional[int]:
    """Extract an integer value following *keyword* in a key=value line."""
    m = re.search(rf"\b{keyword}\s*=\s*(\d+)", line)
    if m:
        return int(m.group(1))
    return None
