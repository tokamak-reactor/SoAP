"""Read EIRENE neutral data files (fort.44 and fort.46).

Format overview
---------------
Both fort.44 and fort.46 are plain-text ASCII files with labeled sections.
Each data section starts with a header line of the form::

    *eirene data field <name> with size <N>

followed by N space-delimited float values in E-notation that may span
multiple lines.

fort.44 (full neutral solution on the B2 grid):
    Lines 1-2:  dimension header
    Species names (one per line)
    Data sections: dab2, tab2, dmb2, tmb2, dib2, tib2,
                   rfluxa, rfluxm, pfluxa, pfluxm, etc.
    Wall-loading data sections
    Radiation sections

fort.46 (neutral densities on the EIRENE triangle grid):
    Similar sectioned format but the data arrays are indexed over triangles
    (ntri) × species.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, TextIO

import numpy as np


# ---------------------------------------------------------------------------
# Common helper — mimics MATLAB read_ft44_rfield.m
# ---------------------------------------------------------------------------

def _read_eirene_field(
    f: TextIO,
    fieldname: str,
    dims: Optional[tuple[int, ...]] = None,
    *,
    skip_idx_lines: bool = False,
) -> np.ndarray:
    """Find and read one ``*eirene data field <fieldname> with size <N>`` section.

    Searches forward from the current file position.  When found, reads
    *N* floating-point values, reshapes to *dims* (if given), and returns
    the array.  If the section is not found, returns ``np.zeros(dims)`` (or a
    scalar 0 if *dims* is ``None``) and issues a warning.

    Parameters
    ----------
    f:
        Open file handle (text mode).  Position is advanced past the data.
    fieldname:
        Name of the field to search for (e.g. ``'dab2'``, ``'pdena'``).
    dims:
        Shape to reshape the data into.  If *None*, a 1-D array is returned.
    skip_idx_lines:
        If *True*, skip one line per 6 columns of *dims[1]* before reading
        data.  Used for fields whose names contain ``(A)``, ``(M)``, etc.
    """
    start_pos = f.tell()

    # ---- search for the field header line --------------------------------
    found = False
    size = 0
    while True:
        line = f.readline()
        if not line:  # EOF
            break
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Match: *eirene data field <name> with size <N>
        if fieldname in line_stripped:
            found = True
            # Extract the size N
            m = re.search(r"with size (\d+)", line_stripped)
            if m:
                size = int(m.group(1))
            else:
                # Fallback: assume size follows "size" keyword
                # Try to parse the last number on the line
                parts = line_stripped.split()
                for part in reversed(parts):
                    try:
                        size = int(part)
                        break
                    except ValueError:
                        continue
            break

    if not found:
        # Restore position, warn, return zeros
        f.seek(start_pos)
        import warnings
        warnings.warn(f"Field '{fieldname}' does not exist in file")
        if dims is not None:
            return np.zeros(dims, dtype=np.float64)
        return np.array([], dtype=np.float64)

    # ---- consistency check ------------------------------------------------
    if dims is not None:
        expected = int(np.prod(dims))
        if size != expected:
            import warnings
            warnings.warn(
                f"Field '{fieldname}': file reports size {size}, "
                f"expected {expected} (dims={dims}). Using file size."
            )
            # Adjust dims to match file
            dims = (size,)
    else:
        dims = (size,)

    # ---- skip index lines for (A)/(M)/(P)/(I) fields ---------------------
    if skip_idx_lines or re.search(r"\([AMP](?:\)|,)|\(I\)", fieldname):
        ncols = dims[1] if len(dims) > 1 else size
        n_skip_lines = int(np.ceil(ncols / 6.0))
        for _ in range(n_skip_lines):
            f.readline()

    # ---- read the data ----------------------------------------------------
    data_text: list[str] = []
    tokens_collected = 0
    while tokens_collected < size:
        line = f.readline()
        if not line:  # premature EOF
            break
        stripped = line.strip()
        if not stripped:
            continue
        # If we hit another *eirene header stop (unlikely if size is correct,
        # but be safe)
        if stripped.startswith("*eirene data field"):
            break
        tokens = stripped.split()
        data_text.extend(tokens)
        tokens_collected += len(tokens)

    raw = np.fromstring(" ".join(data_text), sep=" ", dtype=np.float64, count=size)
    if len(raw) < size:
        # Pad with zeros if we didn't get enough
        raw = np.pad(raw, (0, size - len(raw)), constant_values=0.0)

    if len(dims) > 1:
        return raw.reshape(dims, order="F")  # Fortran column-major order
    return raw


# ---------------------------------------------------------------------------
# fort.44 reader
# ---------------------------------------------------------------------------

def read_fort44(path: str | Path) -> dict[str, Any]:
    """Read an EIRENE fort.44 file.

    Parameters
    ----------
    path:
        Path to the ``fort.44`` file.

    Returns
    -------
    dict
        A dictionary with the following keys:

        **Header / dimensions**
            - ``nx``, ``ny``: grid dimensions
            - ``version``: format version integer
            - ``git_hash``: optional git hash string
            - ``natm``, ``nmol``, ``nion``: number of species
            - ``nfla``, ``nlwrmsh``: additional dimension counts
            - ``atom_labels``, ``mol_labels``, ``ion_labels``: species name lists

        **Neutral data** (2-D arrays shaped ``(nCv, nspecies)`` after
        unstructured remap — stored as ``(nx, ny, nspecies)`` in the file)
            - ``dab2``, ``tab2``: neutral atom density / temperature
            - ``dmb2``, ``tmb2``: neutral molecule density / temperature
            - ``dib2``, ``tib2``: neutral ion density / temperature
            - ``rfluxa``, ``rfluxm``: atom / molecule reflux
            - ``pfluxa``, ``pfluxm``: atom / molecule particle flux
            - ``refluxa``, ``refluxm``: atom / molecule energy flux
            - ``pefluxa``, ``pefluxm``: atom / molecule particle energy
            - ``emiss``, ``emissmol``: emission
            - ``srcml``, ``edissml``: molecular source / dissociation energy
            - ``eneutrad``, ``emolrad``, ``eionrad``: radiation

        **Wall-loading data** (``wld`` sub-dict)
            - ``nlim``, ``nsts``, ``nstra``: wall dimensions
            - ``wldnek``, ``wldnep``, ``wldna``, ``ewlda``, etc.
            - ``eirdiag``, ``sarea_res``, ``ewlda_res``, etc.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is inconsistent.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"fort.44 file not found: {path}")

    result: dict[str, Any] = {}

    with path.open("r") as f:
        # ---- Line 1: nx ny version [git_hash] or nCv version [git_hash] -----
        header_line = f.readline().strip()
        parts = header_line.split()
        # Check if format is unstructured (nCv version git_hash) or structured (nx ny version git_hash)
        if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit() and int(parts[1]) > 100000:
            # Unstructured: nCv version [git_hash]
            n_cv = int(parts[0])
            version = int(parts[1])
            git_hash = parts[2] if len(parts) > 2 else ""
            result["nCv"] = n_cv
            result["version"] = version
            result["git_hash"] = git_hash
        else:
            # Structured: nx ny version [git_hash]
            nx = int(parts[0])
            ny = int(parts[1])
            version = int(parts[2])
            git_hash = parts[3] if len(parts) > 3 else ""
            result["nx"] = nx
            result["ny"] = ny
            result["version"] = version
            result["git_hash"] = git_hash

        # ---- Line 2: natm nmol nion [nfla] [nlwrmsh] ---------------------
        dims_line = f.readline().strip()
        dim_parts = dims_line.split()
        natm = int(dim_parts[0])
        nmol = int(dim_parts[1])
        nion = int(dim_parts[2])
        nfla = int(dim_parts[3]) if len(dim_parts) > 3 else 1
        nlwrmsh = int(dim_parts[4]) if len(dim_parts) > 4 else 1

        result["natm"] = natm
        result["nmol"] = nmol
        result["nion"] = nion
        result["nfla"] = nfla
        result["nlwrmsh"] = nlwrmsh

        # ---- Species labels ----------------------------------------------
        atom_labels: list[str] = []
        mol_labels: list[str] = []
        ion_labels: list[str] = []

        for _ in range(natm):
            atom_labels.append(f.readline().strip())
        for _ in range(nmol):
            mol_labels.append(f.readline().strip())
        for _ in range(nion):
            ion_labels.append(f.readline().strip())

        result["atom_labels"] = atom_labels
        result["mol_labels"] = mol_labels
        result["ion_labels"] = ion_labels

        # Map atoms -> molecules and atoms -> ions (by same base name)
        atm2mol = np.zeros(natm, dtype=np.int32)
        atm2ion = np.zeros(natm, dtype=np.int32)
        molA = np.zeros(nmol, dtype=np.int32)

        for iatm, albl in enumerate(atom_labels):
            for imol, mlbl in enumerate(mol_labels):
                if albl in mlbl:
                    atm2mol[iatm] = imol + 1  # 1-based like MATLAB
                    digits = re.findall(r"\d+", mlbl)
                    if digits:
                        molA[imol] = int(digits[0])
                    break
            for iion, ilbl in enumerate(ion_labels):
                if albl in ilbl:
                    atm2ion[iatm] = iion + 1  # 1-based
                    break

        result["atm2mol"] = atm2mol
        result["atm2ion"] = atm2ion
        result["molA"] = molA

        # Compute number of cells
        if "nCv" in result:
            n_cv = result["nCv"]
            nx = n_cv
            ny = 1
        else:
            nx = result.get("nx", 0)
            ny = result.get("ny", 0)
            n_cv = nx * ny

        result["nCv"] = n_cv

        # ---- Wall loading dimension defaults --------------------------------
        nlim, nsts, nstra = 0, 0, 0

        # ---- Rewind and read sections by field name ----------------------
        f.seek(0)

        # Basic neutral data
        result["dab2"] = _read_eirene_field(f, "dab2", (nx, ny, natm))
        result["tab2"] = _read_eirene_field(f, "tab2", (nx, ny, natm))
        result["dmb2"] = _read_eirene_field(f, "dmb2", (nx, ny, nmol))
        result["tmb2"] = _read_eirene_field(f, "tmb2", (nx, ny, nmol))
        result["dib2"] = _read_eirene_field(f, "dib2", (nx, ny, nion))
        result["tib2"] = _read_eirene_field(f, "tib2", (nx, ny, nion))

        # Fluxes
        result["rfluxa"] = _read_eirene_field(f, "rfluxa", (nx, ny, natm))
        result["rfluxm"] = _read_eirene_field(f, "rfluxm", (nx, ny, nmol))
        result["pfluxa"] = _read_eirene_field(f, "pfluxa", (nx, ny, natm))
        result["pfluxm"] = _read_eirene_field(f, "pfluxm", (nx, ny, nmol))
        result["refluxa"] = _read_eirene_field(f, "refluxa", (nx, ny, natm))
        result["refluxm"] = _read_eirene_field(f, "refluxm", (nx, ny, nmol))
        result["pefluxa"] = _read_eirene_field(f, "pefluxa", (nx, ny, natm))
        result["pefluxm"] = _read_eirene_field(f, "pefluxm", (nx, ny, nmol))

        # Emission / molecular sources
        result["emiss"] = _read_eirene_field(f, "emiss", (nx, ny, 1))
        result["emissmol"] = _read_eirene_field(f, "emissmol", (nx, ny, 1))
        result["srcml"] = _read_eirene_field(f, "srcml", (nx, ny, nmol))
        result["edissml"] = _read_eirene_field(f, "edissml", (nx, ny, nmol))

        # ---- Wall-loading data -------------------------------------------
        wld: dict[str, Any] = {}
        result["wld"] = wld

        # Dimensions
        dims_line = f.readline().strip()
        while dims_line and not dims_line.startswith("*eirene data field"):
            # Skip blank lines or non-field lines
            dims_line = f.readline().strip()

        # If we consumed a field line, we need to handle it; otherwise fall
        # back to the old logic of reading nlim nsts nstra directly.
        # In practice, older fort.44 files have a plain line: nlim nsts nstra
        # before the wall-loading sections.

        # Try reading the wall dimensions from the current file position.
        # The wall dimensions are NOT in a *eirene data field header.
        # They are simply the next three integers after the last data section.
        # Seek back to the start, find the last *eirene data field, and
        # read the three ints after it.

        # Simpler approach: scan forward past the already-read sections,
        # then read nlim nsts nstra.
        # Since we already moved f past all the data sections above,
        # we should now be positioned right before the wall dimension line.

        # Actually, `_read_eirene_field` scans forward from current position.
        # After reading all the fields above, the file pointer may be at
        # various places depending on which fields existed.  Let's just read
        # wall dims the simple way: scan forward from current position.

        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            # Skip section headers
            if stripped.startswith("*"):
                continue
            # Try to parse three integers
            try:
                vals = [int(x) for x in stripped.split()]
                if len(vals) >= 3:
                    nlim, nsts, nstra = vals[0], vals[1], vals[2]
                    break
            except ValueError:
                continue

        wld["nlim"] = nlim
        wld["nsts"] = nsts
        wld["nstra"] = nstra

        nwl = nlim + nsts  # total wall elements

        # --- strahler 1 (default) ---
        wld["wldnek"] = np.zeros((nwl, nstra + 1))
        wld["wldnep"] = np.zeros((nwl, nstra + 1))
        wld["wldna"] = np.zeros((nwl, natm, nstra + 1))
        wld["ewlda"] = np.zeros((nwl, natm, nstra + 1))
        wld["wldnm"] = np.zeros((nwl, nmol, nstra + 1))
        wld["ewldm"] = np.zeros((nwl, nmol, nstra + 1))

        if nwl > 0:
            try:
                nek = _read_eirene_field(f, "wldnek", (nwl,))
                if nek.size > 0:
                    wld["wldnek"][:, 0] = nek
                nep = _read_eirene_field(f, "wldnep", (nwl,))
                if nep.size > 0:
                    wld["wldnep"][:, 0] = nep
                na = _read_eirene_field(f, "wldna", (nwl, natm))
                if na.size > 0:
                    wld["wldna"][:, :, 0] = na.reshape(nwl, natm)
                ea = _read_eirene_field(f, "ewlda", (nwl, natm))
                if ea.size > 0:
                    wld["ewlda"][:, :, 0] = ea.reshape(nwl, natm)
                nm = _read_eirene_field(f, "wldnm", (nwl, nmol))
                if nm.size > 0:
                    wld["wldnm"][:, :, 0] = nm.reshape(nwl, nmol)
                em = _read_eirene_field(f, "ewldm", (nwl, nmol))
                if em.size > 0:
                    wld["ewldm"][:, :, 0] = em.reshape(nwl, nmol)

                wld["wldra"] = _read_eirene_field(f, "wldra", (nwl, natm))
                wld["wldrm"] = _read_eirene_field(f, "wldrm", (nwl, nmol))

                # Radially resolved — additional strahler
                if nstra > 1:
                    for i in range(2, nstra + 1):
                        nek = _read_eirene_field(f, "wldnek", (nwl,))
                        if nek.size > 0:
                            wld["wldnek"][:, i - 1] = nek
                        nep = _read_eirene_field(f, "wldnep", (nwl,))
                        if nep.size > 0:
                            wld["wldnep"][:, i - 1] = nep
                        na = _read_eirene_field(f, "wldna", (nwl, natm))
                        if na.size > 0:
                            wld["wldna"][:, :, i - 1] = na.reshape(nwl, natm)
                        ea = _read_eirene_field(f, "ewlda", (nwl, natm))
                        if ea.size > 0:
                            wld["ewlda"][:, :, i - 1] = ea.reshape(nwl, natm)
                        nm = _read_eirene_field(f, "wldnm", (nwl, nmol))
                        if nm.size > 0:
                            wld["wldnm"][:, :, i - 1] = nm.reshape(nwl, nmol)
                        em = _read_eirene_field(f, "ewldm", (nwl, nmol))
                        if em.size > 0:
                            wld["ewldm"][:, :, i - 1] = em.reshape(nwl, nmol)
            except Exception:
                import warnings
                warnings.warn("Skipping wall load data section due to dimension mismatch")


            # Additional wall quantities (when ns is provided in MATLAB)
            wld["wldpp"] = _read_eirene_field(f, "wldpp", (nwl, 1))
            wld["wldpa"] = _read_eirene_field(f, "wldpa", (nwl, natm))
            wld["wldpm"] = _read_eirene_field(f, "wldpm", (nwl, nmol))
            wld["wldpeb"] = _read_eirene_field(f, "wldpeb", (nwl,))
            wld["wldspt"] = _read_eirene_field(f, "wldspt", (nwl,))
            wld["wlarea"] = _read_eirene_field(f, "wlarea", (nwl,))

            # Pumping
            wld["wlpumpa"] = _read_eirene_field(
                f, "wlpump(A)", (nwl, natm), skip_idx_lines=True
            )
            wld["wlpumpm"] = _read_eirene_field(
                f, "wlpump(M)", (nwl, nmol), skip_idx_lines=True
            )

            # Combined pumping
            wlpump = wld["wlpumpa"].copy()
            for ia in range(natm):
                ma = atm2mol[ia]
                if ma > 0:
                    wlpump[:, ia] = wld["wlpumpm"][:, ma - 1] * molA[ma - 1]
            wld["wlpump"] = wlpump

            # Radiation
            result["eneutrad"] = _read_eirene_field(f, "eneutrad", (nx, ny, natm))
            result["emolrad"] = _read_eirene_field(f, "emolrad", (nx, ny, nmol))
            result["eionrad"] = _read_eirene_field(f, "eionrad", (nx, ny, nion))

            # EIRDIAG / resolved surfaces
            wld["eirdiag"] = _read_eirene_field(f, "eirdiag")
            if wld["eirdiag"].size > 0:
                ncl = int(wld["eirdiag"].max())
                wld["ncl"] = ncl
                wld["sarea_res"] = _read_eirene_field(f, "sarea_res", (ncl,))
                wld["ewlda_res"] = _read_eirene_field(f, "ewlda_res", (ncl, natm))
                wld["ewldm_res"] = _read_eirene_field(f, "ewldm_res", (ncl, nmol))
                wld["ewldrp_res"] = _read_eirene_field(f, "ewldrp_res", (ncl,))
                wld["wldspt_res"] = _read_eirene_field(f, "wldspt_res", (ncl,))
                wld["ewldt_res"] = _read_eirene_field(f, "ewldt_res", (ncl,))

    return result


# ---------------------------------------------------------------------------
# fort.46 reader
# ---------------------------------------------------------------------------

def read_fort46(path: str | Path) -> dict[str, np.ndarray]:
    """Read an EIRENE fort.46 file.

    fort.46 contains neutral data on the EIRENE triangle grid: particle
    densities (pdena, pdenm, pdeni), energy densities (edena, edenm, edeni),
    and momentum densities (vxdena, vxdenm, vxdeni, etc.).

    Parameters
    ----------
    path:
        Path to the ``fort.46`` file.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``ntri``: number of triangles
        - ``version``: format version integer
        - ``natm``, ``nmol``, ``nion``: species counts
        - ``atom_labels``, ``mol_labels``, ``ion_labels``: species names
        - ``pdena``, ``edena``, ``vxdena``, ``vydena``, ``vzdena``: atom data
        - ``pdenm``, ``edenm``, ``vxdenm``, ``vydenm``, ``vzdenm``: molecule data
        - ``pdeni``, ``edeni``, ``vxdeni``, ``vydeni``, ``vzdeni``: ion data

        All density arrays are in SI units (m⁻³, J m⁻³, kg s⁻¹ m⁻²).
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"fort.46 file not found: {path}")

    result: dict[str, Any] = {}

    with path.open("r") as f:
        # ---- Line 1: ntri version [git_hash] -----------------------------
        header_line = f.readline().strip()
        parts = header_line.split()
        ntri = int(parts[0])
        version = int(parts[1])

        result["ntri"] = ntri
        result["version"] = version

        # ---- Line 2: natm nmol nion --------------------------------------
        dims_line = f.readline().strip()
        dim_parts = dims_line.split()
        natm = int(dim_parts[0])
        nmol = int(dim_parts[1])
        nion = int(dim_parts[2])

        result["natm"] = natm
        result["nmol"] = nmol
        result["nion"] = nion

        # ---- Species labels (one per line) -------------------------------
        atom_labels = [f.readline().strip() for _ in range(natm)]
        mol_labels = [f.readline().strip() for _ in range(nmol)]
        ion_labels = [f.readline().strip() for _ in range(nion)]

        result["atom_labels"] = atom_labels
        result["mol_labels"] = mol_labels
        result["ion_labels"] = ion_labels

        # ---- Read data sections (rewind and search) ----------------------
        f.seek(0)

        # Conversion factors (same as MATLAB)
        eV = 1.602176634e-19  # J

        # Particle densities: cm⁻³ -> m⁻³  (×1e6)
        result["pdena"] = _read_eirene_field(f, "pdena", (ntri, natm)) * 1e6
        result["pdenm"] = _read_eirene_field(f, "pdenm", (ntri, nmol)) * 1e6
        result["pdeni"] = _read_eirene_field(f, "pdeni", (ntri, nion)) * 1e6

        # Energy densities: eV·cm⁻³ -> J·m⁻³  (×1e6 × eV)
        result["edena"] = _read_eirene_field(f, "edena", (ntri, natm)) * 1e6 * eV
        result["edenm"] = _read_eirene_field(f, "edenm", (ntri, nmol)) * 1e6 * eV
        result["edeni"] = _read_eirene_field(f, "edeni", (ntri, nion)) * 1e6 * eV

        # Momentum densities: (10× eV·s·cm⁻³) -> kg·s⁻¹·m⁻²  (×10 conversion)
        result["vxdena"] = _read_eirene_field(f, "vxdena", (ntri, natm)) * 10.0
        result["vxdenm"] = _read_eirene_field(f, "vxdenm", (ntri, nmol)) * 10.0
        result["vxdeni"] = _read_eirene_field(f, "vxdeni", (ntri, nion)) * 10.0

        result["vydena"] = _read_eirene_field(f, "vydena", (ntri, natm)) * 10.0
        result["vydenm"] = _read_eirene_field(f, "vydenm", (ntri, nmol)) * 10.0
        result["vydeni"] = _read_eirene_field(f, "vydeni", (ntri, nion)) * 10.0

        result["vzdena"] = _read_eirene_field(f, "vzdena", (ntri, natm)) * 10.0
        result["vzdenm"] = _read_eirene_field(f, "vzdenm", (ntri, nmol)) * 10.0
        result["vzdeni"] = _read_eirene_field(f, "vzdeni", (ntri, nion)) * 10.0

    return result


# ---------------------------------------------------------------------------
# Convenience: detect file type and dispatch
# ---------------------------------------------------------------------------

def read_eirene(path: str | Path) -> dict[str, Any]:
    """Auto-detect and read an EIRENE output file (fort.44 or fort.46).

    Detection heuristic: if the file has ``pdena`` / ``edena`` sections it
    is treated as fort.46 data (triangular grid); otherwise it is treated as
    fort.44 (structured grid) data.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    dict
        The parsed data dictionary.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"EIRENE file not found: {path}")

    # Quick peek at the first few lines to decide
    with path.open("r") as f:
        first = f.readline().strip()
        second = f.readline().strip()

    # Both formats have integers on the first two lines.
    # fort.46 second line has 3 ints (natm, nmol, nion).
    # Try to read the entire file by heuristic: try fort.44 first.
    try:
        result = read_fort44(path)
        # If we got actual data with at least one field, return it
        if "dab2" in result and result["dab2"].size > 0:
            return result
    except Exception:
        pass

    # Fall back to fort.46
    return read_fort46(path)
