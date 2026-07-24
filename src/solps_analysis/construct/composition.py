"""Plasma composition: B2 and EIRENE species indexing.

B2 indexing: all charge states of each element grouped together.
  Example (D + C): [D⁰, D⁺, C⁰, C⁺, C²⁺, C³⁺, C⁴⁺, C⁵⁺, C⁶⁺]
  → D: indices [0, 1], C: indices [2, 3, 4, 5, 6, 7, 8]

EIRENE indexing: atoms, molecules, ions as separate groups.
  Example (D + C): atoms=[D, C], molecules=[D2], ions=[D2+]
  → D: atom 0, C: atom 1, D2: mol 0, D2+: ion 0

The bridge: B2 neutrals (D⁰, C⁰) are zeroed when EIRENE is active.
EIRENE densities on B2 grid (dab2_eir, dmb2_eir) replace them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────
# Element identification
# ──────────────────────────────────────────────

# Atomic mass → element symbol
MASS_TO_ELEMENT: dict[int, str] = {
    1: "H", 2: "D", 3: "T", 4: "He",
    7: "Li", 9: "Be", 11: "B", 12: "C",
    14: "N", 16: "O", 20: "Ne", 23: "Na",
    27: "Al", 28: "Si", 32: "S", 35: "Cl",
    40: "Ar", 52: "Cr", 55: "Mn", 56: "Fe",
    59: "Ni", 64: "Cu", 84: "Kr", 132: "Xe",
    184: "W",
}


def mass_to_element(mass: float) -> str:
    """Convert atomic mass number to element symbol."""
    return MASS_TO_ELEMENT.get(round(mass), f"M{round(mass)}")


# ──────────────────────────────────────────────
# B2 Plasma Composition
# ──────────────────────────────────────────────


@dataclass
class B2Composition:
    """B2 species composition — maps elements to charge states.

    >>> comp = B2Composition(zamax=[0,1,0,1,2,3,4,5,6], am=[2,2,12,12,12,12,12,12,12])
    >>> comp.element_indices("D")
    [0, 1]
    >>> comp.element_indices("C")
    [2, 3, 4, 5, 6, 7, 8]
    >>> comp.names
    ['D', 'C']
    """

    # Per-species arrays (ns total species)
    zamax: np.ndarray  # (ns,) — max charge of each species
    am: np.ndarray     # (ns,) — atomic mass of each species  
    zn: np.ndarray     # (ns,) — nuclear charge

    # Derived: grouped by element
    n_species: int = 0          # ns — total B2 species
    n_elements: int = 0         # number of distinct elements
    element_names: list[str] = field(default_factory=list)
    element_indices_list: list[list[int]] = field(default_factory=list)
    element_charge: np.ndarray | None = None   # (n_elements,) max charge per element
    element_mass: np.ndarray | None = None     # (n_elements,) mass per element

    # Lookup dicts
    _elem_to_indices: dict = field(default_factory=dict)
    _elem_to_name: dict = field(default_factory=dict)

    def __post_init__(self):
        self._build()

    def _build(self):
        """Build element grouping from zamax array."""
        zamax = np.asarray(self.zamax).ravel()
        am = np.asarray(self.am).ravel()
        ns = len(zamax)

        self.n_species = ns

        # Group: new element starts when zamax decreases
        elem_indices = []
        current = []
        for k in range(ns - 1):
            current.append(k)
            if zamax[k + 1] < zamax[k]:
                elem_indices.append(current)
                current = []
        current.append(ns - 1)
        elem_indices.append(current)

        self.element_indices_list = elem_indices
        self.n_elements = len(elem_indices)

        # Element names and properties
        names = []
        charges = []
        masses = []
        self._elem_to_indices = {}
        self._elem_to_name = {}

        for idx_list in elem_indices:
            mass_val = round(float(am[idx_list[0]]))
            name = mass_to_element(mass_val)
            names.append(name)
            charges.append(float(zamax[idx_list[-1]]))
            masses.append(float(am[idx_list[0]]))
            self._elem_to_indices[name] = idx_list
            self._elem_to_name[name] = name

        self.element_names = names
        self.element_charge = np.array(charges)
        self.element_mass = np.array(masses)

    def element_indices(self, symbol: str) -> list[int]:
        """Get B2 species indices for an element, e.g. comp.element_indices('D') → [0, 1]."""
        return self._elem_to_indices.get(symbol, [])

    def charge_states(self, symbol: str) -> list[float]:
        """Get charge states for an element's species."""
        idx = self.element_indices(symbol)
        return [float(self.zamax[i]) for i in idx]

    def neutral_index(self, symbol: str) -> int | None:
        """Get the B2 index of the neutral species for this element (charge 0)."""
        for i in self.element_indices(symbol):
            if self.zamax[i] == 0:
                return i
        return None

    def first_ion_index(self, symbol: str) -> int | None:
        """Get the B2 index of the first ionized state."""
        for i in self.element_indices(symbol):
            if self.zamax[i] >= 1:
                return i
        return None

    def __repr__(self) -> str:
        return f"B2Composition({self.n_species} species, {self.n_elements} elements: {', '.join(self.element_names)})"


# ──────────────────────────────────────────────
# EIRENE Composition
# ──────────────────────────────────────────────


@dataclass
class EireneComposition:
    """EIRENE species composition — atoms, molecules, ions.

    >>> comp = EireneComposition(atom_labels=['D','C'], mol_labels=['D2'], ion_labels=['D2+'])
    >>> comp.atom_index('D')
    0
    >>> comp.mol_index('D2')
    0
    >>> comp.ion_index('D2+')
    0
    """

    atom_labels: list[str] = field(default_factory=list)
    mol_labels: list[str] = field(default_factory=list)
    ion_labels: list[str] = field(default_factory=list)

    # Mapping from B2/H elemental symbol → EIRENE index
    # E.g. for D + C case: element_to_atom['D'] = 0, element_to_atom['C'] = 1
    element_to_atom: dict = field(default_factory=dict)
    element_to_mol: dict = field(default_factory=dict)

    # atm2mol[i] = molecule index that atom i belongs to (1-based, 0=none)
    # atm2ion[i] = ion index that atom i belongs to
    atm2mol: np.ndarray | None = None
    atm2ion: np.ndarray | None = None

    def __post_init__(self):
        # Build element → atom index mapping
        for i, lbl in enumerate(self.atom_labels):
            # Strip charge info for comparison
            clean = lbl.rstrip("+-0123456789")
            self.element_to_atom[clean] = i

        for i, lbl in enumerate(self.mol_labels):
            clean = lbl.rstrip("+-0123456789")
            if clean not in self.element_to_mol:
                self.element_to_mol[clean] = i

    @property
    def natm(self) -> int:
        return len(self.atom_labels)

    @property
    def nmol(self) -> int:
        return len(self.mol_labels)

    @property
    def nion(self) -> int:
        return len(self.ion_labels)

    def atom_index(self, label: str) -> int | None:
        """Get EIRENE atom index for an element symbol (e.g. 'D' → 0)."""
        clean = label.rstrip("+-0123456789")
        return self.element_to_atom.get(clean)

    def mol_index(self, label: str) -> int | None:
        """Get EIRENE molecule index."""
        clean = label.rstrip("+-0123456789")
        return self.element_to_mol.get(clean)

    def ion_index(self, label: str) -> int | None:
        """Get EIRENE ion index by label."""
        for i, lbl in enumerate(self.ion_labels):
            if lbl.startswith(label):
                return i
        return None

    def __repr__(self) -> str:
        return (f"EireneComposition({self.natm} atoms: {self.atom_labels}, "
                f"{self.nmol} mols: {self.mol_labels}, {self.nion} ions: {self.ion_labels})")


# ──────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────


def build_b2_composition(fstate: dict) -> B2Composition:
    """Build B2Composition from b2fstati/fstate data.
    
    Args:
        fstate: dict with keys 'zamax', 'am', 'zn' (numpy arrays)
    """
    zamax = fstate.get("zamax", fstate.get("zamax", []))
    am = fstate.get("am", [])
    zn = fstate.get("zn", [])
    if isinstance(zamax, (int, float)):
        zamax = [zamax]
    return B2Composition(
        zamax=np.asarray(zamax, dtype=np.float64).ravel(),
        am=np.asarray(am, dtype=np.float64).ravel(),
        zn=np.asarray(zn, dtype=np.float64).ravel(),
    )


def build_eirene_composition(neut: dict) -> EireneComposition:
    """Build EireneComposition from fort.44 data.
    
    Args:
        neut: dict with keys 'atom_labels', 'mol_labels', 'ion_labels', 
              optionally 'atm2mol', 'atm2ion'
    """
    return EireneComposition(
        atom_labels=list(neut.get("atom_labels", [])),
        mol_labels=list(neut.get("mol_labels", [])),
        ion_labels=list(neut.get("ion_labels", [])),
        atm2mol=neut.get("atm2mol"),
        atm2ion=neut.get("atm2ion"),
    )
