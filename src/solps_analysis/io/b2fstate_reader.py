"""SOLPS-ITER b2fstati/b2fstate reader — full state data.

The b2fstati file uses the same tagged-ASCII format as b2fgmtry.
This module reads all plasma state variables (na, ne, te, ti, fluxes, etc.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solps_analysis.io.b2fgmtry_parser import read_tagged_ascii_sections


def read_b2fstate_full(path: str | Path) -> dict[str, np.ndarray | float | int | str]:
    """Read the full plasma state from a b2fstati/b2fstate file.

    Returns a dict with all variable names as keys and numpy arrays as values.
    Includes: na, ne, te, ti, ua, fna, fhe, fhi, fch, and many more.
    """
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        candidates = [
            path / "b2fstati",
            path / "b2fstate",
            path.parent / "b2fstati",
            path.parent / "b2fstate",
        ]
    else:
        candidates = [path]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as f:
                raw = read_tagged_ascii_sections(f)
            return raw

    raise FileNotFoundError(f"b2fstati/b2fstate not found from {path}")


def extract_state_arrays(raw: dict) -> dict[str, np.ndarray]:
    """Extract just the named arrays from the raw tagged-ASCII data.

    Returns a flat dict with array variable names as keys.
    Comma-separated field names (like 'nx,ny,ns') are ignored here.
    """
    import re

    result = {}
    for key, value in raw.items():
        if key == "_version":
            continue
        # Skip comma-separated names (dimension fields)
        if "," in key:
            continue
        if isinstance(value, (np.ndarray, int, float)):
            result[key] = value
    return result


def get_plasma_composition(state: dict) -> dict:
    """Determine plasma composition from b2fstati state data.

    Groups charged states by species (nuclear charge).
    Returns a dict with:
      - n_species: number of distinct species
      - species_names: list of element symbols
      - species_indices: list of lists, indices into state arrays for each species
      - species_charge: max charge per species
      - species_mass: atomic mass per species
      - ions_list: indices of charged states
      - neut_list: indices of neutral states
    """
    zamax = state.get("zamax", state.get("zn", []))
    am = state.get("am", [])
    ns = len(zamax)

    if isinstance(zamax, np.ndarray):
        zamax = zamax.ravel()
    if isinstance(am, np.ndarray):
        am = am.ravel()

    # Map species by charge drops
    species_indices = []
    current_indices = []
    for k in range(ns - 1):
        current_indices.append(k)
        if zamax[k + 1] < zamax[k]:
            species_indices.append(current_indices)
            current_indices = []
    current_indices.append(ns - 1)
    species_indices.append(current_indices)

    # Build species names
    mass_to_element = {
        1: "H", 2: "D", 3: "T", 4: "He",
        7: "Li", 9: "Be", 11: "B", 12: "C",
        14: "N", 20: "Ne", 40: "Ar", 84: "Kr",
        132: "Xe", 184: "W",
    }

    species_names = []
    for idx_list in species_indices:
        sp_mass = round(float(am[idx_list[0]]))
        name = mass_to_element.get(sp_mass, f"M{sp_mass}")
        species_names.append(name)

    # Get charge and mass per species
    species_charge = []
    species_mass = []
    for idx_list in species_indices:
        species_charge.append(float(zamax[idx_list[-1]]))
        species_mass.append(float(am[idx_list[0]]))

    return {
        "n_species": len(species_indices),
        "species_names": species_names,
        "species_indices": species_indices,
        "species_charge": np.array(species_charge),
        "species_mass": np.array(species_mass),
        "ions_list": [i for i in range(ns) if zamax[i] != 0],
        "neut_list": [i for i in range(ns) if zamax[i] == 0],
    }
