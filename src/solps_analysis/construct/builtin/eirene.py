"""EIRENE-derived quantities: na overwrite, neutral fluxes, temperatures, pressure.

MATLAB counterpart: calc_additional.m, section "redefine some variables if EIRENE is used"
(lines 38–347).

Key logic
---------
When EIRENE is active, B2's neutral densities (na[:, neutral_cols]) are
replaced by EIRENE data from fort.44 (dab2, dmb2).  Particle fluxes on
faces (fna_th, fna_r) are also computed from EIRENE data instead of B2's.

All quantities in this module produce *new* variables — the original B2
`na` is never modified.  Use `eirene_na` in downstream quantities (ni, pa,
cs, …) whenever EIRENE data should be preferred.
"""

from __future__ import annotations

import numpy as np

from solps_analysis.construct.registry import quantity

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
QE = 1.602176634e-19   # Elementary charge [C] (CODATA 2018, exact)
MP = 1.672621637e-27   # Proton mass [kg]


# ──────────────────────────────────────────────
# Helper: dab2 structured → 1D
# ──────────────────────────────────────────────


def _unpack_eirene_cell_data(arr: np.ndarray, grid) -> np.ndarray:
    """Convert EIRENE cell data from structured (nx, ny, ns) → (nCv, ns).

    For unstructured grids the input is already (nCv, 1, ns); the middle
    singleton dimension is squeezed.

    NOTE: fort.44 structured data covers only interior cells (nx, ny),
    WITHOUT guard cells. The mapping to full cell array uses
    imap_cv[1:-1, 1:-1] which removes the guard ring.
    """
    if arr.ndim == 3:
        if arr.shape[1] == 1:
            return arr[:, 0, :]
        # Structured fort.44: (nx_int, ny_int, ns) — interior only
        nx_int, ny_int, ns = arr.shape
        if grid.imap_cv is not None and nx_int == grid.nx and ny_int == grid.ny:
            # Map interior fort.44 data to cell array using imap
            interior_imap = grid.imap_cv[1:-1, 1:-1]  # (nx, ny)
            dest_idx = interior_imap.ravel(order='F').astype(np.intp)  # (n_int,) 1-based indices
            result = np.zeros((grid.n_cells, ns), dtype=np.float64)
            valid = dest_idx > 0
            for s in range(ns):
                result[dest_idx[valid] - 1, s] = arr[:, :, s].ravel(order='F')[valid]
            return result
    return np.asarray(arr, dtype=np.float64)


def _unpack_eirene_1d(arr: np.ndarray, grid) -> np.ndarray:
    """Convert a 1-D EIRENE array sized (nx*ny,) → (nCv,) for structured."""
    if arr.ndim == 1:
        n_expected = (grid.nx + 2) * (grid.ny + 2) if grid.is_structured else grid.n_cells
        if grid.is_structured and grid.imap_cv is not None and len(arr) == n_expected:
            imap = grid.imap_cv
            result = np.zeros(grid.n_cells, dtype=np.float64)
            mask = imap > 0
            result[imap[mask].astype(np.intp) - 1] = arr[mask]
            return result
    return np.asarray(arr, dtype=np.float64).ravel()


# ──────────────────────────────────────────────
# EIRENE overwritten na
# ──────────────────────────────────────────────


@quantity(
    name="eirene_na",
    requires=["dab2", "dmb2", "atm2mol", "molA"],
    description="Neutral densities from EIRENE (fort.44), replacing B2 neutral columns",
    unit="m⁻³",
)
def calc_eirene_na(dab2, dmb2, atm2mol, molA, grid=None, comp=None, eirene=None, watch=None, **kw):
    """Build na matrix with neutral densities from EIRENE fort.44 data.

    For each EIRENE atom species:
      na[:, neutral_col] = dab2[:, ia] + molA[mol_idx] * dmb2[:, mol_idx]

    Interior cells (1:nCi) are taken directly from fort.44.
    Boundary cells take the value of their nearest interior neighbour.

    The original B2 na (if loaded) is NOT modified — this function builds
    a separate array.
    """
    # ── get original na shape ────────────────────────────────────────────
    original_na = watch.get("na")
    if original_na is None:
        raise ValueError("eirene_na requires B2 na to be loaded first")

    n_cells, n_species = original_na.data.shape

    # ── unpack EIRENE cell data ──────────────────────────────────────────
    dab2_1d = _unpack_eirene_cell_data(dab2, grid)   # (nCv, natm) or (nCi, natm)
    dmb2_1d = _unpack_eirene_cell_data(dmb2, grid)   # (nCv, nmol) or (nCi, nmol)

    # EIRENE species counts
    natm = dab2_1d.shape[1] if dab2_1d.ndim > 1 else 0
    nmol = dmb2_1d.shape[1] if dmb2_1d.ndim > 1 else 0

    if natm == 0 or eirene is None or comp is None:
        return original_na.data.copy()

    # ── build output ─────────────────────────────────────────────────────
    result = original_na.data.copy()

    nci = grid.n_core_cells or grid.n_cells

    for ia in range(natm):
        # EIRENE atom label → element → B2 neutral column
        atom_label = eirene.atom_labels[ia].strip()
        b2_col = comp.neutral_index(atom_label)

        if b2_col is None:
            continue  # element not tracked in B2 composition

        mol_idx = int(atm2mol[ia])  # 1-based; 0 = no molecule

        # ── interior cells ───────────────────────────────────────────────
        n_avail = min(dab2_1d.shape[0], nci)
        result[:n_avail, b2_col] = dab2_1d[:n_avail, ia]

        if mol_idx > 0:
            mw = float(molA[mol_idx - 1])
            n_avail_mol = min(dmb2_1d.shape[0], nci)
            result[:n_avail_mol, b2_col] += mw * dmb2_1d[:n_avail_mol, mol_idx - 1]

        # ── boundary cells (nCi+1 : nCv) ─────────────────────────────────
        # For structured grids: use imap_cv to find interior neighbors.
        # For unstructured grids: use face connectivity.
        if nci < n_cells:
            _fill_boundary_eirene(
                result, grid, nci, b2_col,
                dab2_1d, dmb2_1d, ia, mol_idx, mw if mol_idx > 0 else None
            )

    return result


def _fill_boundary_eirene(result, grid, nci, b2_col, dab2_1d, dmb2_1d,
                          ia, mol_idx, mw):
    """Fill EIRENE data in boundary cells from interior neighbours."""
    n_cells = grid.n_cells

    # Structured grid: use imap_cv to find the interior cell for each boundary
    if grid.is_structured and grid.imap_cv is not None:
        imap = grid.imap_cv  # (nx+2, ny+2)
        nx, ny = imap.shape
        # Build reverse lookup: cell_index → (i, j) in imap
        cell_to_ij = {}
        for j in range(ny):
            for i in range(nx):
                idx = int(imap[i, j])
                if idx > 0:
                    cell_to_ij[idx - 1] = (i, j)

        for icv in range(nci, n_cells):
            if icv not in cell_to_ij:
                continue
            i, j = cell_to_ij[icv]
            # Try 4-connected neighbors
            for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= ni < nx and 0 <= nj < ny:
                    neigh_idx = int(imap[ni, nj]) - 1
                    if 0 <= neigh_idx < nci:
                        result[icv, b2_col] = dab2_1d[neigh_idx, ia]
                        if mw is not None:
                            result[icv, b2_col] += mw * dmb2_1d[neigh_idx, mol_idx - 1]
                        break
    elif grid.cv_fc_p is not None:
        # Unstructured: use face connectivity
        for icv in range(nci, n_cells):
            start = grid.cv_fc_p[icv, 0]
            fc0 = int(grid.cv_fc[start]) if grid.cv_fc is not None else -1
            if fc0 >= 0 and fc0 < grid.n_faces:
                cv_neigh = grid.fc_cv[fc0]
                cv2 = min(cv_neigh[0], cv_neigh[1])
                if 0 <= cv2 < nci:
                    result[icv, b2_col] = dab2_1d[cv2, ia]
                    if mw is not None:
                        result[icv, b2_col] += mw * dmb2_1d[cv2, mol_idx - 1]


# ──────────────────────────────────────────────
# EIRENE particle fluxes on faces (fna)
# ──────────────────────────────────────────────


def _distribute_flux_to_faces(tot_flux, grid, fna_field: str) -> np.ndarray:
    """Distribute cell-centered total fluxes to faces (fna_th or fna_r).

    For each flux tube face: interpolate using fcHc weights × fcS.
    For each radial surface face: 0 (handled by the caller).
    For boundary cells: use fcQalf projection.
    """
    n_faces = grid.n_faces
    fna = np.zeros(n_faces)

    if grid.ft_fc_p is None or grid.fc_hc is None:
        return fna

    nci = grid.n_core_cells or grid.n_cells

    # Flux tube faces (poloidal θ direction)
    for i_ft in range(grid.n_flux_tubes):
        start = grid.ft_fc_p[i_ft, 0]
        count = grid.ft_fc_p[i_ft, 1]
        for k in range(count):
            fc = int(grid.ft_fc[start + k])
            cv1, cv2 = grid.fc_cv[fc]
            if cv1 < 0 or cv2 < 0 or cv1 >= nci or cv2 >= nci:
                continue
            w1 = grid.fc_hc[fc, 1]
            w2 = grid.fc_hc[fc, 0]
            denom = w1 + w2
            if denom > 1e-30:
                fna[fc] = (tot_flux[cv1] * w1 + tot_flux[cv2] * w2) / denom * grid.fc_s[fc]

    return fna


@quantity(
    name="eirene_tot_flux_th",
    requires=["vxdena", "vydena", "pux", "puy", "pfluxa", "rfluxa", "atm2mol", "molA"],
    description="Total poloidal flux per species from EIRENE (on cells)",
    unit="s⁻¹",
)
def calc_tot_flux_th(vxdena, vydena, pux, puy, pfluxa, rfluxa, atm2mol, molA,
                     grid=None, comp=None, eirene=None, watch=None, **kw):
    """Compute total poloidal (theta) particle flux on cells.

    Two code paths matching MATLAB:
    - version >= 3.002 (unstructured): (vxdena·pux + vydena·puy) / (am * mp)
    - version < 3.002  (structured): neut.pfluxa from fort.44
    """
    if comp is None or eirene is None:
        raise ValueError("eirene_tot_flux_th requires comp and eirene composition")

    natm = len(eirene.atom_labels)
    nci = grid.n_core_cells or grid.n_cells

    result = np.zeros((grid.n_cells, max(1, comp.n_elements)))

    if grid.version_number >= 3002:
        # Unstructured path: use fort.46 vxdena/vydena + pux/puy projections
        # pux/puy are on B2 grid (size nCv or ntri)
        pux_1d = np.asarray(pux, dtype=np.float64).ravel()
        puy_1d = np.asarray(puy, dtype=np.float64).ravel()

        # vxdena/vydena are (ntri, natm); take first nci values
        n_avail = min(vxdena.shape[0], nci)
        for ia in range(natm):
            element = eirene.atom_labels[ia].strip()
            elem_idx = comp.element_names.index(element) if element in comp.element_names else ia
            am_val = float(comp.element_mass[elem_idx]) if elem_idx < len(comp.element_mass) else 2.0

            tot = np.zeros(n_avail)
            tot += vxdena[:n_avail, ia] * pux_1d[:n_avail]
            tot += vydena[:n_avail, ia] * puy_1d[:n_avail]
            tot /= (am_val * MP)

            mol_idx = int(atm2mol[ia])
            if mol_idx > 0:
                mw = float(molA[mol_idx - 1])
                # vxdenm/vydenm from ft46 for molecule contribution
                ft46 = getattr(watch, 'ft46', None)
                if ft46 is not None and 'vxdenm' in ft46:
                    n_avail_m = min(ft46['vxdenm'].shape[0], nci)
                    tot_m = np.zeros(n_avail_m)
                    tot_m += ft46['vxdenm'][:n_avail_m, mol_idx - 1] * pux_1d[:n_avail_m]
                    tot_m += ft46['vydenm'][:n_avail_m, mol_idx - 1] * puy_1d[:n_avail_m]
                    tot_m /= (am_val * MP)
                    tot[:n_avail_m] += tot_m

            result[:n_avail, elem_idx] = tot
    else:
        # Structured path: use pfluxa from fort.44 (already on B2 grid)
        pfluxa_1d = _unpack_eirene_cell_data(pfluxa, grid)

        for ia in range(natm):
            element = eirene.atom_labels[ia].strip()
            elem_idx = comp.element_names.index(element) if element in comp.element_names else ia
            n_avail = min(pfluxa_1d.shape[0], nci)
            result[:n_avail, elem_idx] = pfluxa_1d[:n_avail, ia]

            mol_idx = int(atm2mol[ia])
            if mol_idx > 0:
                mw = float(molA[mol_idx - 1])
                rfluxa_1d = _unpack_eirene_cell_data(rfluxa, grid)
                n_avail_m = min(rfluxa_1d.shape[0], nci)
                result[:n_avail_m, elem_idx] += mw * rfluxa_1d[:n_avail_m, mol_idx - 1]

    return result


@quantity(
    name="eirene_tot_flux_r",
    requires=["vxdena", "vydena", "pvx", "pvy", "pfluxa", "rfluxa", "atm2mol", "molA"],
    description="Total radial particle flux per species from EIRENE (on cells)",
    unit="s⁻¹",
)
def calc_tot_flux_r(vxdena, vydena, pvx, pvy, pfluxa, rfluxa, atm2mol, molA,
                     grid=None, comp=None, eirene=None, watch=None, **kw):
    """Compute total radial particle flux on cells.

    Two code paths matching MATLAB:
    - version >= 3.002 (unstructured): (vxdena·pvx + vydena·pvy) / (am * mp)
    - version < 3.002  (structured): neut.rfluxa from fort.44
    """
    if comp is None or eirene is None:
        raise ValueError("eirene_tot_flux_r requires comp and eirene composition")

    natm = len(eirene.atom_labels)
    nci = grid.n_core_cells or grid.n_cells

    result = np.zeros((grid.n_cells, max(1, comp.n_elements)))

    if grid.version_number >= 3002:
        pvx_1d = np.asarray(pvx, dtype=np.float64).ravel()
        pvy_1d = np.asarray(pvy, dtype=np.float64).ravel()

        n_avail = min(vxdena.shape[0], nci)
        for ia in range(natm):
            element = eirene.atom_labels[ia].strip()
            elem_idx = comp.element_names.index(element) if element in comp.element_names else ia
            am_val = float(comp.element_mass[elem_idx]) if elem_idx < len(comp.element_mass) else 2.0

            tot = np.zeros(n_avail)
            tot += vxdena[:n_avail, ia] * pvx_1d[:n_avail]
            tot += vydena[:n_avail, ia] * pvy_1d[:n_avail]
            tot /= (am_val * MP)

            mol_idx = int(atm2mol[ia])
            if mol_idx > 0:
                mw = float(molA[mol_idx - 1])
                ft46 = getattr(watch, 'ft46', None)
                if ft46 is not None and 'vxdenm' in ft46:
                    n_avail_m = min(ft46['vxdenm'].shape[0], nci)
                    tot_m = np.zeros(n_avail_m)
                    tot_m += ft46['vxdenm'][:n_avail_m, mol_idx - 1] * pvx_1d[:n_avail_m]
                    tot_m += ft46['vydenm'][:n_avail_m, mol_idx - 1] * pvy_1d[:n_avail_m]
                    tot_m /= (am_val * MP)
                    tot[:n_avail_m] += tot_m

            result[:n_avail, elem_idx] = tot
    else:
        rfluxa_1d = _unpack_eirene_cell_data(rfluxa, grid)

        for ia in range(natm):
            element = eirene.atom_labels[ia].strip()
            elem_idx = comp.element_names.index(element) if element in comp.element_names else ia
            n_avail = min(rfluxa_1d.shape[0], nci)
            result[:n_avail, elem_idx] = rfluxa_1d[:n_avail, ia]

            mol_idx = int(atm2mol[ia])
            if mol_idx > 0:
                mw = float(molA[mol_idx - 1])
                pfluxa_1d = _unpack_eirene_cell_data(pfluxa, grid)
                n_avail_m = min(pfluxa_1d.shape[0], nci)
                result[:n_avail_m, elem_idx] += mw * pfluxa_1d[:n_avail_m, mol_idx - 1]

    return result


# ──────────────────────────────────────────────
# Neutral temperatures
# ──────────────────────────────────────────────


@quantity(
    name="eirene_tdena",
    requires=["edena", "pdena"],
    description="Neutral atom temperature from EIRENE: 2/3 * edena / pdena",
    unit="eV",
)
def calc_eirene_tdena(edena, pdena, **kw):
    """Neutral atom temperature on triangle grid or B2 grid."""
    result = np.full_like(edena, 0.0)
    mask = pdena > 1e-30
    result[mask] = (2.0 / 3.0) * edena[mask] / pdena[mask]
    return result


@quantity(
    name="eirene_tdena_tot",
    requires=["edena", "pdena"],
    description="Total (atom+molecule) neutral temperature: 2/3 * edena_tot / ndena_tot",
    unit="eV",
)
def calc_eirene_tdena_tot(edena, pdena, **kw):
    """For triangle grid, this is the same as tdena (molecules handled separately)."""
    return calc_eirene_tdena(edena, pdena)


# ──────────────────────────────────────────────
# Neutral pressure on cells
# ──────────────────────────────────────────────


@quantity(
    name="eirene_P_neut",
    requires=["edena", "pdena"],
    description="Neutral pressure: 2/3 * edena (on cells/B2 grid)",
    unit="Pa",
)
def calc_eirene_P_neut(edena, **kw):
    """P_neut = 2/3 * edena."""
    return (2.0 / 3.0) * np.asarray(edena, dtype=np.float64)
