"""calc_additional.m port — face interpolation + divergence block.

MATLAB source: Calc_SPb/calc_additional.m lines 349-495 (and beyond).
Uses the MATLAB-style workspace from io.matlab_vars.
"""

from __future__ import annotations

import numpy as np

from solps_analysis.construct.registry import quantity
from solps_analysis.core.operators import div_us, intface

QE = 1.602176634e-19   # C, CODATA 2018
MP = 1.672621637e-27   # kg, proton mass


def _ws(watch) -> dict:
    from solps_analysis.io.matlab_vars import build_workspace
    return build_workspace(watch)


def _intface_method(grid) -> str:
    """MATLAB line 349-353: version >= 3.002 → 'vol', else 'halfsum'."""
    return "vol" if grid.version_number >= 3002 else "halfsum"


# ──────────────────────────────────────────────────────────────
# Face interpolation block (lines 355-370)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="tef",
    requires=[],
    description="Te interpolated to faces (intface)",
    unit="eV",
    location="face",
)
def calc_tef(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return intface(grid, ws["te"], 1, _intface_method(grid))


@quantity(
    name="tif",
    requires=[],
    description="Ti interpolated to faces (intface)",
    unit="eV",
    location="face",
)
def calc_tif(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return intface(grid, ws["ti"], 1, _intface_method(grid))


@quantity(
    name="nef",
    requires=[],
    description="ne interpolated to faces (intface)",
    unit="m⁻³",
    location="face",
)
def calc_nef(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return intface(grid, ws["ne"], 1, _intface_method(grid))


@quantity(
    name="naf",
    requires=[],
    description="na (all species) interpolated to faces",
    unit="m⁻³",
    location="face",
)
def calc_naf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    ns = ws["na"].shape[1]
    out = np.zeros((grid.n_faces, ns))
    for is_ in range(ns):
        out[:, is_] = intface(grid, ws["na"][:, is_], 1, _intface_method(grid))
    return out


@quantity(
    name="uaf",
    requires=[],
    description="ua (all species) interpolated to faces",
    unit="m/s",
    location="face",
)
def calc_uaf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    ns = ws["ua"].shape[1]
    out = np.zeros((grid.n_faces, ns))
    for is_ in range(ns):
        out[:, is_] = intface(grid, ws["ua"][:, is_], 1, _intface_method(grid))
    return out


@quantity(
    name="uef",
    requires=[],
    description="ue interpolated to faces (intface)",
    unit="m/s",
    location="face",
)
def calc_uef(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return intface(grid, ws["ue"], 1, _intface_method(grid))


@quantity(
    name="pof",
    requires=[],
    description="po interpolated to faces (intface)",
    unit="V",
    location="face",
)
def calc_pof(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return intface(grid, ws["po"], 0, _intface_method(grid))


@quantity(
    name="hzf",
    requires=[],
    description="cvHz interpolated to faces",
    unit="m",
    location="face",
)
def calc_hzf(watch=None, grid=None, **kw):
    if grid.cv_hz is None:
        return np.zeros(grid.n_faces)
    return intface(grid, grid.cv_hz, 1, _intface_method(grid))


# ──────────────────────────────────────────────────────────────
# Divergence block (lines 372-495)
# ──────────────────────────────────────────────────────────────

def _div_pair(grid, f_th: np.ndarray, f_r: np.ndarray) -> np.ndarray:
    """div_us(gmtry, [f_th f_r]) — one species or full matrix."""
    if f_th.ndim == 1:
        return div_us(grid, np.column_stack([f_th, f_r]))
    # (nFc, ns) → packed (nFc, 2*ns)
    ns = f_th.shape[1]
    packed = np.zeros((f_th.shape[0], 2 * ns))
    packed[:, :ns] = f_th
    packed[:, ns:] = f_r
    return div_us(grid, packed)


def _div_th(grid, f_th: np.ndarray) -> np.ndarray:
    """div_us(gmtry, [f_th zeros])"""
    if f_th.ndim == 1:
        return div_us(grid, np.column_stack([f_th, np.zeros_like(f_th)]))
    ns = f_th.shape[1]
    packed = np.zeros((f_th.shape[0], 2 * ns))
    packed[:, :ns] = f_th
    return div_us(grid, packed)


def _div_r(grid, f_r: np.ndarray) -> np.ndarray:
    """div_us(gmtry, [zeros f_r])"""
    if f_r.ndim == 1:
        return div_us(grid, np.column_stack([np.zeros_like(f_r), f_r]))
    ns = f_r.shape[1]
    packed = np.zeros((f_r.shape[0], 2 * ns))
    packed[:, ns:] = f_r
    return div_us(grid, packed)


@quantity(
    name="div_fna_mdf",
    requires=[],
    description="div of total (mdf) particle flux per species",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_mdf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_mdf_th"], ws["fna_mdf_r"])


@quantity(
    name="div_fna",
    requires=[],
    description="div of full particle flux per species",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_th"], ws["fna_r"])


@quantity(
    name="div_fna_mdf_th",
    requires=[],
    description="poloidal part of div(fna_mdf)",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_mdf_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_mdf_th"])


@quantity(
    name="div_fna_mdf_r",
    requires=[],
    description="radial part of div(fna_mdf)",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_mdf_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_mdf_r"])


@quantity(
    name="div_fna_th",
    requires=[],
    description="poloidal part of div(fna)",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_th"])


@quantity(
    name="div_fna_r",
    requires=[],
    description="radial part of div(fna)",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_r"])


@quantity(
    name="div_fna_flo_th",
    requires=[],
    description="poloidal div of flow-flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_flo_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_flo_th"])


@quantity(
    name="div_fna_flo_r",
    requires=[],
    description="radial div of flow-flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_flo_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_flo_r"])


@quantity(
    name="div_fna_Dgradn_th",
    requires=[],
    description="poloidal div of D∇n flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_Dgradn_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_Dgradn_th"])


@quantity(
    name="div_fna_Dgradn_r",
    requires=[],
    description="radial div of D∇n flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_Dgradn_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_Dgradn_r"])


@quantity(
    name="div_fmo",
    requires=[],
    description="div of momentum flux",
    unit="Pa/m",
)
def calc_div_fmo(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fmo_th"], ws["fmo_r"])


@quantity(
    name="div_fmo_th",
    requires=[],
    description="poloidal div of momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fmo_th"])


@quantity(
    name="div_fmo_r",
    requires=[],
    description="radial div of momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fmo_r"])


@quantity(
    name="div_fmo_flo",
    requires=[],
    description="div of momentum flow flux",
    unit="Pa/m",
)
def calc_div_fmo_flo(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fmo_flo_th"], ws["fmo_flo_r"])


@quantity(
    name="div_fmo_vis",
    requires=[],
    description="div of viscous momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_vis(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fmo_vis_th"], ws["fmo_vis_r"])


@quantity(
    name="div_fmo_vis_th",
    requires=[],
    description="poloidal div of viscous momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_vis_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fmo_vis_th"])


@quantity(
    name="div_fmo_vis_r",
    requires=[],
    description="radial div of viscous momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_vis_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fmo_vis_r"])


@quantity(
    name="div_fna_fha",
    requires=[],
    description="div of friction force flux (charged species only)",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_fha(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_fha_th"], ws["fna_fha_r"])


# ──────────────────────────────────────────────────────────────
# Energy flux divergences (lines 481-495)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="div_ue",
    requires=[],
    description="div of electron parallel velocity flux",
    unit="s⁻¹",
)
def calc_div_ue(watch=None, grid=None, **kw):
    ws = _ws(watch)
    uef = intface(grid, ws["ue"], 1, _intface_method(grid))
    fc_pbs = grid.fc_pbs if grid.fc_pbs is not None else np.ones(grid.n_faces)
    return div_us(grid, np.column_stack([uef * fc_pbs, np.zeros(grid.n_faces)]))


@quantity(
    name="div_fch",
    requires=[],
    description="div of total current density",
    unit="A/m³",
)
def calc_div_fch(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fch_th"], ws["fch_r"])


@quantity(
    name="div_fch_th",
    requires=[],
    description="poloidal div of current density",
    unit="A/m³",
)
def calc_div_fch_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fch_th"])


@quantity(
    name="div_fch_r",
    requires=[],
    description="radial div of current density",
    unit="A/m³",
)
def calc_div_fch_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fch_r"])


@quantity(
    name="div_fhi",
    requires=[],
    description="div of ion heat flux (B2 5.0 formulation)",
    unit="W/m³",
)
def calc_div_fhi(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fhi_th"], ws["fhi_r"])


@quantity(
    name="div_fhe",
    requires=[],
    description="div of electron heat flux (B2 5.0 formulation)",
    unit="W/m³",
)
def calc_div_fhe(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fhe_th"], ws["fhe_r"])


@quantity(
    name="div_fhi_mdf",
    requires=[],
    description="div of ion heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhi_mdf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fhi_mdf_th"], ws["fhi_mdf_r"])


@quantity(
    name="div_fhi_mdf_th",
    requires=[],
    description="poloidal div of ion heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhi_mdf_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fhi_mdf_th"])


@quantity(
    name="div_fhi_mdf_r",
    requires=[],
    description="radial div of ion heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhi_mdf_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fhi_mdf_r"])


@quantity(
    name="div_fhe_mdf",
    requires=[],
    description="div of electron heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhe_mdf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fhe_mdf_th"], ws["fhe_mdf_r"])


@quantity(
    name="div_fhe_mdf_th",
    requires=[],
    description="poloidal div of electron heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhe_mdf_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fhe_mdf_th"])


@quantity(
    name="div_fhe_mdf_r",
    requires=[],
    description="radial div of electron heat flux (mdf)",
    unit="W/m³",
)
def calc_div_fhe_mdf_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fhe_mdf_r"])
