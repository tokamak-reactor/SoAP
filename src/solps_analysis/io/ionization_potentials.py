"""Read ionization potentials for plasma composition (MATLAB read_ionization_potentials.m).

The ionization_potentials file is a lower-triangular table: row i (element with
atomic number i) contains the ionization potentials of charge states 1..i.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ionization_potentials(path: str | Path, comp) -> np.ndarray:
    """Return ion_pot (n_species,) — ionization potential of each species.

    Neutrals get 0. For an ion of element Z at charge state k (k>=1) the
    value is the (k-th) ionization potential of that element.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "ionization_potentials"
    if not p.exists():
        return np.zeros(comp.n_species)

    text = p.read_text(errors="ignore")
    # Fortran double format: 1.35984340d+01 → 1.35984340e+01
    text = text.replace("d+", "e+").replace("d-", "e-").replace("D+", "e+").replace("D-", "e-")
    import re
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals = np.array([float(t) for t in tokens], dtype=np.float64)
    if vals.ndim == 0:
        vals = vals.reshape(1)

    # Build triangular full table: row i has i entries
    n_elements = 86
    table = np.zeros((n_elements, n_elements))
    idx = 0
    for i in range(n_elements):
        for j in range(i + 1):
            if idx < vals.size:
                table[i, j] = vals[idx]
                idx += 1

    ns = comp.n_species
    ion_pot = np.zeros(ns)
    indices = getattr(comp, "element_indices_list", None)
    zn = np.asarray(getattr(comp, "zn", []), dtype=int)  # atomic numbers
    if indices is None or zn.size == 0:
        return ion_pot

    for iel, idx_list in enumerate(indices):
        if len(idx_list) < 2:
            continue
        zatm = zn[idx_list[0]]
        if zatm < 1 or zatm > n_elements:
            continue
        first_ion = idx_list[1]
        for k, is_ in enumerate(idx_list[1:]):
            # charge state k+1 → table[zatm-1, k]
            ion_pot[is_] = table[zatm - 1, k]

    return ion_pot


def get_ion_pot(watch, comp) -> np.ndarray:
    """Ionization potentials per species, cached on watch."""
    cached = getattr(watch, "_ion_pot", None)
    if cached is not None:
        return cached
    ion_pot = read_ionization_potentials(str(watch.path), comp)
    watch._ion_pot = ion_pot
    return ion_pot
