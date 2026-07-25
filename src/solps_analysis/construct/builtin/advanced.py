"""Advanced quantities: separatrix, budgets, partial pressures."""

from __future__ import annotations

import numpy as np

from solps_analysis.construct.registry import quantity
from solps_analysis.construct.builtin.basic import _resolve_na, QE, MP


@quantity(
    name="pa",
    requires=["ne", "te_eV", "ti_eV", "na"],
    description="Partial pressure by element: neutral*Te + ion*Ti",
    unit="Pa",
)
def calc_pa(ne, te_eV, ti_eV, na, grid=None, comp=None, watch=None, **kw):
    """Partial pressure by element: neutral*Te + ion*Ti.
    
    pa[:, ia] = na_elem_neutral * Te * qe + na_elem_ions * Ti * qe

    Uses eirene_na when EIRENE data is available.
    """
    effective_na = _resolve_na(watch, na)
    te_j = te_eV * QE
    ti_j = ti_eV * QE
    n_elem = comp.n_elements if comp else 1

    result = np.zeros((effective_na.shape[0], max(n_elem, 1)))
    for ia in range(n_elem):
        idx = comp.element_indices(comp.element_names[ia])
        neutral_rows = [i for i in idx if comp.zamax[i] == 0]
        ion_rows = [i for i in idx if comp.zamax[i] > 0]
        p_neut = effective_na[:, neutral_rows].sum(axis=1) * te_j if neutral_rows else 0
        p_ions = effective_na[:, ion_rows].sum(axis=1) * ti_j if ion_rows else 0
        result[:, ia] = p_neut + p_ions
    return result


@quantity(
    name="Zavg",
    requires=["na"],
    description="Average ion charge by element",
    unit="",
)
def calc_zavg(na, grid=None, comp=None, watch=None, **kw):
    """Average ion charge by element.

    Uses eirene_na when EIRENE data is available.
    """
    effective_na = _resolve_na(watch, na)
    n_elem = comp.n_elements if comp else 1
    result = np.zeros((effective_na.shape[0], max(n_elem, 1)))
    for ia in range(n_elem):
        idx = comp.element_indices(comp.element_names[ia])
        ion_rows = [i for i in idx if comp.zamax[i] > 0]
        if not ion_rows:
            continue
        na_ions = effective_na[:, ion_rows]
        charges = np.array([comp.zamax[i] for i in ion_rows])
        result[:, ia] = (na_ions * charges).sum(axis=1) / \
                        np.maximum(na_ions.sum(axis=1), 1e-30)
    return result


@quantity(
    name="te_sep",
    requires=["te_eV"],
    description="Averaged separatrix electron temperature",
    unit="eV",
)
def calc_te_sep(te_eV, grid=None, **kw):
    """Surface-averaged Te at the separatrix (from core_sep_fcs)."""
    fcs = getattr(grid, 'core_sep_fcs', None)
    if fcs is None or len(fcs) == 0:
        return np.array([0.0])
    # Check fc_cv validity — guard against structured grids where fc_cv may be [0,0]
    fc_cv = grid.fc_cv
    fc_hc = grid.fc_hc
    fc_s = getattr(grid, 'fc_s', None)
    if fc_s is None:
        fc_s = np.ones(grid.n_faces)
    total = 0.0
    total_s = 0.0
    for fc in fcs:
        cv1, cv2 = fc_cv[fc]
        if cv1 == cv2 == 0 or cv1 >= grid.n_cells or cv2 >= grid.n_cells:
            continue
        w1, w2 = fc_hc[fc, 1], fc_hc[fc, 0]
        denom = w1 + w2
        if denom > 1e-30:
            tee = (te_eV[cv1] * w1 + te_eV[cv2] * w2) / denom
            total += tee * fc_s[fc]
            total_s += fc_s[fc]
    return np.array([total / max(total_s, 1e-30)])


@quantity(
    name="ti_sep",
    requires=["ti_eV"],
    description="Averaged separatrix ion temperature",
    unit="eV",
)
def calc_ti_sep(ti_eV, grid=None, **kw):
    fcs = getattr(grid, 'core_sep_fcs', None)
    if fcs is None or len(fcs) == 0:
        return np.array([0.0])
    fc_s = grid.fc_s if hasattr(grid, 'fc_s') else np.ones(grid.n_faces)
    fc_hc = grid.fc_hc
    total = 0.0
    total_s = 0.0
    for fc in fcs:
        cv1, cv2 = grid.fc_cv[fc]
        w1, w2 = fc_hc[fc, 1], fc_hc[fc, 0]
        tii = (ti_eV[cv1] * w1 + ti_eV[cv2] * w2) / max(w1 + w2, 1e-30)
        total += tii * fc_s[fc]
        total_s += fc_s[fc]
    return np.array([total / max(total_s, 1e-30)])


@quantity(
    name="n_e_sep",
    requires=["ne", "na"],
    description="Separatrix concentration (n_imp/n_e) per element",
    unit="",
)
def calc_n_e_sep(ne, na, grid=None, comp=None, watch=None, **kw):
    fcs = getattr(grid, 'core_sep_fcs', None)
    if fcs is None or len(fcs) == 0 or comp is None:
        return np.zeros(comp.n_elements if comp else 1)

    effective_na = _resolve_na(watch, na)
    fc_s = grid.fc_s if hasattr(grid, 'fc_s') else np.ones(grid.n_faces)
    fc_hc = grid.fc_hc

    sep_ne = 0.0
    sep_n_imp = np.zeros(comp.n_elements)
    total_s = 0.0

    for fc in fcs:
        cv1, cv2 = grid.fc_cv[fc]
        w1, w2 = fc_hc[fc, 1], fc_hc[fc, 0]
        tee = (ne[cv1] * w1 + ne[cv2] * w2) / max(w1 + w2, 1e-30)
        sep_ne += tee * fc_s[fc]

        for ia in range(comp.n_elements):
            idx = comp.element_indices(comp.element_names[ia])
            ion_rows = [i for i in idx if comp.zamax[i] > 0]
            if ion_rows:
                n_avg = sum(effective_na[cv if cv < effective_na.shape[0] else 0, i] for i in ion_rows for cv in [cv1, cv2]) / 2
                sep_n_imp[ia] += n_avg * fc_s[fc]
        total_s += fc_s[fc]

    ne_avg = sep_ne / max(total_s, 1e-30)
    return np.array([imp_n / max(ne_avg, 1e-30) for imp_n in sep_n_imp])


@quantity(
    name="total_particles",
    requires=["na"],
    description="Total particle inventory per element",
    unit="",
)
def calc_total_particles(na, grid=None, comp=None, watch=None, **kw):
    """Total number of nuclei per element, integrated over core (nCi) cells.

    Uses eirene_na when EIRENE data is available.
    """
    effective_na = _resolve_na(watch, na)
    cv_vol = grid.cv_vol if hasattr(grid, 'cv_vol') and grid.cv_vol is not None else np.ones(grid.n_cells)
    n_core = grid.n_core_cells if hasattr(grid, 'n_core_cells') else grid.n_cells

    if comp is None:
        return np.array([(effective_na[:n_core, :].sum(axis=1) * cv_vol[:n_core]).sum()])

    result = np.zeros(comp.n_elements)
    for ia in range(comp.n_elements):
        idx = comp.element_indices(comp.element_names[ia])
        ion_rows = [i for i in idx if comp.zamax[i] > 0]
        if ion_rows:
            result[ia] = (effective_na[:n_core, :][:, ion_rows].sum(axis=1) * cv_vol[:n_core]).sum()
    return result
