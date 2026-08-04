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
    return "vol" if grid.version_float >= 3.002 else "halfsum"


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


# ──────────────────────────────────────────────────────────────
# Currents & effective velocities (lines 406-476)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fna_curr_th",
    requires=[],
    description="current-related particle flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna_curr_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if grid.version_float >= 3.2:
        # version >= 320: from fch components / qe
        fch = (ws.get("fch_inert_th", np.zeros(grid.n_faces))
               + ws.get("fch_vispar_th", np.zeros(grid.n_faces))
               + ws.get("fch_visper_th", np.zeros(grid.n_faces))
               + ws.get("fch_AN_th", np.zeros(grid.n_faces))
               + ws.get("fch_visq_th", np.zeros(grid.n_faces)))
        ns = ws["fna_mdf_th"].shape[1]
        out = np.zeros((grid.n_faces, ns))
        if fch.ndim == 1:
            out[:, 1] = fch / QE  # main ion column
        else:
            out = fch / QE
    else:
        out = (ws["fna_mdf_th"] - (ws["fna_nupar_th"] + ws["fna_dia_mdf_th"]
               + ws["fna_RhieChow_th"] + ws["fna_nuAN_th"] + ws["fna_Dgradn_th"]
               + ws["fna_nuExB_th"]))
    return _zero_neutral_cols(out, comp)


@quantity(
    name="fna_curr_r",
    requires=[],
    description="current-related particle flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna_curr_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if grid.version_float >= 3.2:
        fch = (ws.get("fch_inert_r", np.zeros(grid.n_faces))
               + ws.get("fch_vispar_r", np.zeros(grid.n_faces))
               + ws.get("fch_visper_r", np.zeros(grid.n_faces))
               + ws.get("fch_AN_r", np.zeros(grid.n_faces))
               + ws.get("fch_visq_r", np.zeros(grid.n_faces)))
        ns = ws["fna_mdf_r"].shape[1]
        out = np.zeros((grid.n_faces, ns))
        if fch.ndim == 1:
            out[:, 1] = fch / QE
        else:
            out = fch / QE
    else:
        out = (ws["fna_mdf_r"] - (ws["fna_dia_mdf_r"] + ws["fna_RhieChow_r"]
               + ws["fna_nuAN_r"] + ws["fna_Dgradn_r"] + ws["fna_nuExB_r"]))
    return _zero_neutral_cols(out, comp)


def _zs(comp):
    """Charge zs per species (0 for neutrals, zamax for ions)."""
    if comp is None:
        return None
    zs = getattr(comp, "zs", None)
    if zs is None:
        zamax = getattr(comp, "zamax", None)
        if zamax is not None:
            zs = np.asarray(zamax)
    return zs


def _zero_neutral_cols(out, comp):
    """Zero columns of charged-species matrix for neutral species."""
    zs = _zs(comp)
    if zs is None or out.ndim != 2:
        return out
    out[:, np.asarray(zs) == 0] = 0
    return out


@quantity(
    name="div_fna_curr",
    requires=[],
    description="div of current particle flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_curr(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth = ws.get("fna_curr_th")
    fr = ws.get("fna_curr_r")
    if fth is None:
        fth = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    if fr is None:
        fr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    return _div_pair(grid, fth, fr)


@quantity(
    name="ua_eff_th",
    requires=[],
    description="effective poloidal velocity from friction flux",
    unit="m/s",
    location="face",
)
def calc_ua_eff_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    naf = intface(grid, ws["na"], 1, _intface_method(grid))
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ws["fna_fha_th"] / naf / fc_s[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="ua_eff_r",
    requires=[],
    description="effective radial velocity from friction flux",
    unit="m/s",
    location="face",
)
def calc_ua_eff_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    naf = intface(grid, ws["na"], 1, _intface_method(grid))
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ws["fna_fha_r"] / naf / fc_s[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="ua_diff_th",
    requires=[],
    description="diffusion velocity, poloidal",
    unit="m/s",
    location="face",
)
def calc_ua_diff_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    naf = intface(grid, ws["na"], 1, _intface_method(grid))
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ws["fna_Dgradn_th"] / naf / fc_s[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="ua_diff_r",
    requires=[],
    description="diffusion velocity, radial",
    unit="m/s",
    location="face",
)
def calc_ua_diff_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    naf = intface(grid, ws["na"], 1, _intface_method(grid))
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ws["fna_Dgradn_r"] / naf / fc_s[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="div_ua",
    requires=[],
    description="div of parallel velocity * pbs",
    unit="s⁻¹",
)
def calc_div_ua(watch=None, grid=None, **kw):
    ws = _ws(watch)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_pbs = grid.fc_pbs if grid.fc_pbs is not None else np.ones(grid.n_faces)
    return _div_th(grid, uaf * fc_pbs[:, None])


@quantity(
    name="div_ua_eff",
    requires=[],
    description="div of effective velocity * fcS",
    unit="s⁻¹",
)
def calc_div_ua_eff(watch=None, grid=None, **kw):
    ws = _ws(watch)
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    fth = ws.get("ua_eff_th")
    fr = ws.get("ua_eff_r")
    if fth is None:
        fth = calc_ua_eff_th(watch=watch, grid=grid)
    if fr is None:
        fr = calc_ua_eff_r(watch=watch, grid=grid)
    return _div_pair(grid, fth * fc_s[:, None], fr * fc_s[:, None])


@quantity(
    name="div_ua_ExB",
    requires=[],
    description="div of ExB velocity (fna_nuExB / naf)",
    unit="s⁻¹",
)
def calc_div_ua_ExB(watch=None, grid=None, **kw):
    ws = _ws(watch)
    naf = intface(grid, ws["na"], 1, _intface_method(grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        fth = ws["fna_nuExB_th"] / naf
        fr = ws["fna_nuExB_r"] / naf
    fth = np.nan_to_num(fth, nan=0.0, posinf=0.0, neginf=0.0)
    fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
    return _div_pair(grid, fth, fr)


@quantity(
    name="div_ua_diff",
    requires=[],
    description="div of diffusion velocity * fcS",
    unit="s⁻¹",
)
def calc_div_ua_diff(watch=None, grid=None, **kw):
    ws = _ws(watch)
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    fth = ws.get("ua_diff_th")
    fr = ws.get("ua_diff_r")
    if fth is None:
        fth = calc_ua_diff_th(watch=watch, grid=grid)
    if fr is None:
        fr = calc_ua_diff_r(watch=watch, grid=grid)
    return _div_pair(grid, fth * fc_s[:, None], fr * fc_s[:, None])


@quantity(
    name="div_fmo_viscurv",
    requires=[],
    description="div of curvature viscosity momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_viscurv(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if comp is None:
        return np.zeros(grid.n_cells)
    ams = _species_mass_kg(comp)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_hz = ws.get("fc_hz")
    if fc_hz is None:
        fc_hz = np.ones(grid.n_faces)
    fth = ws["fna_mo_vis_th"] * fc_hz[:, None] * ams[None, :] * uaf
    fr = ws["fna_mo_vis_r"] * fc_hz[:, None] * ams[None, :] * uaf
    return _div_pair(grid, fth, fr)


@quantity(
    name="div_fmo_vis_BgradB",
    requires=[],
    description="div of ∇B viscosity momentum flux",
    unit="Pa/m",
)
def calc_div_fmo_vis_BgradB(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if comp is None:
        return np.zeros(grid.n_cells)
    ams = _species_mass_kg(comp)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_hz = ws.get("fc_hz")
    if fc_hz is None:
        fc_hz = np.ones(grid.n_faces)
    fth = ws["fna_nuBgradB_th"] * fc_hz[:, None] * ams[None, :] * uaf
    fr = ws["fna_nuBgradB_r"] * fc_hz[:, None] * ams[None, :] * uaf
    return _div_pair(grid, fth, fr)


@quantity(
    name="div_fna_dia_mdf",
    requires=[],
    description="div of diamagnetic (mdf) flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_dia_mdf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_dia_mdf_th"], ws["fna_dia_mdf_r"])


@quantity(
    name="div_fna_nuExB",
    requires=[],
    description="div of ExB drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuExB(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_nuExB_th"], ws["fna_nuExB_r"])


@quantity(
    name="div_fna_nuExB_th",
    requires=[],
    description="poloidal div of ExB drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuExB_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_nuExB_th"])


@quantity(
    name="div_fna_nuExB_r",
    requires=[],
    description="radial div of ExB drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuExB_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_nuExB_r"])


@quantity(
    name="div_fna_nuBgradB",
    requires=[],
    description="div of ∇B drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuBgradB(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_nuBgradB_th"], ws["fna_nuBgradB_r"])


@quantity(
    name="div_fna_nuBgradB_th",
    requires=[],
    description="poloidal div of ∇B drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuBgradB_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_nuBgradB_th"])


@quantity(
    name="div_fna_nuBgradB_r",
    requires=[],
    description="radial div of ∇B drift flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuBgradB_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_r(grid, ws["fna_nuBgradB_r"])


@quantity(
    name="div_fna_nupar",
    requires=[],
    description="div of parallel particle flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nupar(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_th(grid, ws["fna_nupar_th"])


@quantity(
    name="div_fna_nuAN",
    requires=[],
    description="div of anomalous pinch flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_nuAN(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_nuAN_th"], ws["fna_nuAN_r"])


@quantity(
    name="div_fna_RhieChow",
    requires=[],
    description="div of Rhie-Chow flux",
    unit="m⁻³ s⁻¹",
)
def calc_div_fna_RhieChow(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return _div_pair(grid, ws["fna_RhieChow_th"], ws["fna_RhieChow_r"])


def _species_mass_kg(comp):
    """Species mass in kg (am * mp), array of length ns."""
    if comp is None:
        return None
    am = getattr(comp, "am", None)
    if am is None:
        am = getattr(comp, "ams", None)
    if am is None:
        return None
    return np.asarray(am, dtype=np.float64) * MP


# ──────────────────────────────────────────────────────────────
# Electron current fluxes & convective fluxes (lines 416-421)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fne_curr_th",
    requires=[],
    description="electron current flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_curr_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fna_curr = ws.get("fna_curr_th")
    if fna_curr is None:
        fna_curr = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    return (ws["fch_th"] - ws["fch_nuBgradB_th"] - ws["fch_par_th"]) / QE \
        - fna_curr.sum(axis=1)


@quantity(
    name="fne_curr_r",
    requires=[],
    description="electron current flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_curr_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fna_curr = ws.get("fna_curr_r")
    if fna_curr is None:
        fna_curr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    return (ws["fch_r"] - ws["fch_nuBgradB_r"] - ws["fch_par_r"]) / QE \
        - fna_curr.sum(axis=1)


@quantity(
    name="fna_conv_th",
    requires=[],
    description="convective particle flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna_conv_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return (ws["fna_nuExB_th"] + ws["fna_nupar_th"] + ws["fna_nuAN_th"]
            + ws["fna_RhieChow_th"] + ws.get("fna_curr_th", 0))


@quantity(
    name="fna_conv_r",
    requires=[],
    description="convective particle flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna_conv_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return (ws["fna_nuExB_r"] + ws["fna_nuAN_r"] + ws["fna_RhieChow_r"]
            + ws.get("fna_curr_r", 0))


# ──────────────────────────────────────────────────────────────
# Derived heat/momentum fluxes (lines 553-628)
# ──────────────────────────────────────────────────────────────

def _sum_over_charged(ws, arr_name: str, zs: np.ndarray) -> np.ndarray:
    """Sum zs(is) * arr[:, is] over charged species (nFc,)."""
    arr = ws[arr_name]
    return (arr * zs[None, :]).sum(axis=1)


@quantity(
    name="fhe_nuExB_th",
    requires=[],
    description="electron heat flux from ExB drift, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_nuExB_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    return 1.5 * zs[None, :] * ws["fna_nuExB_th"] @ np.ones(len(zs)) * 0 + \
        1.5 * _sum_over_charged(ws, "fna_nuExB_th", zs) * tef * QE if zs is not None \
        else np.zeros(grid.n_faces)


@quantity(
    name="fhe_nuExB_r",
    requires=[],
    description="electron heat flux from ExB drift, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_nuExB_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    return 1.5 * _sum_over_charged(ws, "fna_nuExB_r", zs) * tef * QE


@quantity(
    name="fhe_nupar_th",
    requires=[],
    description="electron heat flux from parallel flow, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_nupar_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    return 1.5 * _sum_over_charged(ws, "fna_nupar_th", zs) * tef * QE \
        - 1.5 * ws["fch_par_th"] * tef


@quantity(
    name="fhe_fnaAN_th",
    requires=[],
    description="electron heat flux from anomalous pinch, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_fnaAN_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    return 1.5 * (ws["fna_Dgradn_th"] * zs[None, :]).sum(axis=1) * tef * QE


@quantity(
    name="fhe_fnaAN_r",
    requires=[],
    description="electron heat flux from anomalous pinch, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_fnaAN_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    return 1.5 * ((ws["fna_nuAN_r"] + ws["fna_Dgradn_r"]) * zs[None, :]).sum(axis=1) * tef * QE


def _fmo_drift(watch, grid, ws, fna_th, fna_r, factor=1.0):
    """fmo_* = factor * fna_* * uaf * hzf * mp * am (nFc, ns)."""
    ams = _species_mass_kg(ws.get("comp")) if False else None
    from solps_analysis.io.matlab_vars import species_am
    am = species_am(watch)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_hz = ws.get("fc_hz", np.ones(grid.n_faces))
    pref = factor * fc_hz[:, None] * am[None, :] * MP * uaf
    return fna_th * pref, fna_r * pref


@quantity(
    name="fmo_nuExB_th",
    requires=[],
    description="momentum flux from ExB drift, poloidal",
    unit="Pa",
    location="face",
)
def calc_fmo_nuExB_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth, _ = _fmo_drift(watch, grid, ws, ws["fna_nuExB_th"], ws["fna_nuExB_r"])
    return fth


@quantity(
    name="fmo_nuExB_r",
    requires=[],
    description="momentum flux from ExB drift, radial",
    unit="Pa",
    location="face",
)
def calc_fmo_nuExB_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    _, fr = _fmo_drift(watch, grid, ws, ws["fna_nuExB_th"], ws["fna_nuExB_r"])
    return fr


@quantity(
    name="fmo_nuBgradB_th",
    requires=[],
    description="momentum flux from ∇B drift, poloidal",
    unit="Pa",
    location="face",
)
def calc_fmo_nuBgradB_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth, _ = _fmo_drift(watch, grid, ws, ws["fna_nuBgradB_th"], ws["fna_nuBgradB_r"], factor=2.0)
    return fth


@quantity(
    name="fmo_nuBgradB_r",
    requires=[],
    description="momentum flux from ∇B drift, radial",
    unit="Pa",
    location="face",
)
def calc_fmo_nuBgradB_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    _, fr = _fmo_drift(watch, grid, ws, ws["fna_nuBgradB_th"], ws["fna_nuBgradB_r"], factor=2.0)
    return fr


@quantity(
    name="fmo_fnaAN_th",
    requires=[],
    description="momentum flux from anomalous pinch, poloidal",
    unit="Pa",
    location="face",
)
def calc_fmo_fnaAN_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth, _ = _fmo_drift(watch, grid, ws, ws["fna_Dgradn_th"], ws["fna_Dgradn_r"])
    return fth


@quantity(
    name="fmo_fnaAN_r",
    requires=[],
    description="momentum flux from anomalous pinch, radial",
    unit="Pa",
    location="face",
)
def calc_fmo_fnaAN_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    _, fr = _fmo_drift(watch, grid, ws,
                       (ws["fna_nuAN_r"] + ws["fna_Dgradn_r"]) * 0 + ws["fna_Dgradn_th"],
                       ws["fna_nuAN_r"] + ws["fna_Dgradn_r"])
    return fr


@quantity(
    name="fmo_fnapar_th",
    requires=[],
    description="momentum flux from parallel flow, poloidal",
    unit="Pa",
    location="face",
)
def calc_fmo_fnapar_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth, _ = _fmo_drift(watch, grid, ws, ws["fna_nupar_th"], ws["fna_nupar_th"])
    return fth


@quantity(
    name="fmo_conv_th",
    requires=[],
    description="convective momentum flux, poloidal (ions only)",
    unit="Pa",
    location="face",
)
def calc_fmo_conv_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _species_mass_kg(comp)
    zs = _zs(comp)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_hz = ws.get("fc_hz", np.ones(grid.n_faces))
    out = ams[None, :] * uaf * ws["fna_th"] * fc_hz[:, None]
    if zs is not None:
        out[:, np.asarray(zs) <= 0] = 0
    return out


@quantity(
    name="fmo_conv_r",
    requires=[],
    description="convective momentum flux, radial (ions only)",
    unit="Pa",
    location="face",
)
def calc_fmo_conv_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _species_mass_kg(comp)
    zs = _zs(comp)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_hz = ws.get("fc_hz", np.ones(grid.n_faces))
    out = ams[None, :] * uaf * ws["fna_r"] * fc_hz[:, None]
    if zs is not None:
        out[:, np.asarray(zs) <= 0] = 0
    return out


@quantity(
    name="fna53_th",
    requires=[],
    description="5/2 particle flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna53_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return ws["fna_nuExB_th"] + ws["fna_nupar_th"]


@quantity(
    name="fna53_r",
    requires=[],
    description="5/2 particle flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fna53_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return ws["fna_nuExB_r"]


def _fne_sum(watch, grid, ws, name):
    """fne_* = sum over species zs(is)*fna_*_th/r (nFc,)."""
    zs = _zs(watch.b2_comp if hasattr(watch, "b2_comp") else None)
    if zs is None:
        from solps_analysis.io.matlab_vars import species_zamax
        zs = species_zamax(watch)
    return (ws[name] * zs[None, :]).sum(axis=1)


@quantity(
    name="fne_nuBgradB_th",
    requires=[],
    description="electron flux from ∇B drift, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_nuBgradB_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -(ws["fna_nuBgradB_th"] * (tef / tif)[:, None] * zs[None, :] ** 2).sum(axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="fne_nuBgradB_r",
    requires=[],
    description="electron flux from ∇B drift, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_nuBgradB_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -(ws["fna_nuBgradB_r"] * (tef / tif)[:, None] * zs[None, :] ** 2).sum(axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@quantity(
    name="fne_th",
    requires=[],
    description="total electron flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    fna_curr = ws.get("fna_curr_th")
    if fna_curr is None:
        fna_curr = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        nuBg = -(ws["fna_nuBgradB_th"] * (tef / tif)[:, None] * zs[None, :] ** 2).sum(axis=1)
    nuBg = np.nan_to_num(nuBg, nan=0.0, posinf=0.0, neginf=0.0)
    return (ws["fna_nupar_th"] * zs[None, :]).sum(axis=1) \
        + (ws["fna_nuExB_th"] * zs[None, :]).sum(axis=1) + nuBg \
        + (ws["fna_Dgradn_th"] * zs[None, :]).sum(axis=1) \
        + (ws["fna_nuAN_th"] * zs[None, :]).sum(axis=1) \
        + (ws["fna_RhieChow_th"] * zs[None, :]).sum(axis=1) \
        - ws["fch_par_th"] / QE \
        + (fna_curr * zs[None, :]).sum(axis=1)


@quantity(
    name="fne_r",
    requires=[],
    description="total electron flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    fna_curr = ws.get("fna_curr_r")
    if fna_curr is None:
        fna_curr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        nuBg = -(ws["fna_nuBgradB_r"] * (tef / tif)[:, None] * zs[None, :] ** 2).sum(axis=1)
    nuBg = np.nan_to_num(nuBg, nan=0.0, posinf=0.0, neginf=0.0)
    return (ws["fna_nuExB_r"] * zs[None, :]).sum(axis=1) + nuBg \
        + (ws["fna_Dgradn_r"] * zs[None, :]).sum(axis=1) \
        + (ws["fna_nuAN_r"] * zs[None, :]).sum(axis=1) \
        + (ws["fna_RhieChow_r"] * zs[None, :]).sum(axis=1) \
        + (fna_curr * zs[None, :]).sum(axis=1)


@quantity(
    name="fne53_th",
    requires=[],
    description="5/2 electron flux, poloidal",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne53_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    fna_curr = ws.get("fna_curr_th")
    if fna_curr is None:
        fna_curr = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    return (ws["fna_nuExB_th"] * zs[None, :]).sum(axis=1) \
        - (fna_curr * zs[None, :]).sum(axis=1) \
        + (ws["fna_nupar_th"] * zs[None, :]).sum(axis=1) - ws["fch_par_th"] / QE


@quantity(
    name="fne53_r",
    requires=[],
    description="5/2 electron flux, radial",
    unit="m⁻² s⁻¹",
    location="face",
)
def calc_fne53_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    fna_curr = ws.get("fna_curr_r")
    if fna_curr is None:
        fna_curr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    return (ws["fna_nuExB_r"] * zs[None, :]).sum(axis=1) \
        - (fna_curr * zs[None, :]).sum(axis=1)


@quantity(
    name="c071",
    requires=[],
    description="parallel heat flux coefficient c071",
    unit="",
)
def calc_c071(watch=None, grid=None, **kw):
    ws = _ws(watch)
    zeff = ws["Zeff"]
    if np.max(np.abs(zeff)) == 0:
        na = ws["na"]
        from solps_analysis.io.matlab_vars import species_zamax
        zamax = species_zamax(watch)
        zeff = (na * zamax[None, :] ** 2).sum(axis=1) / np.maximum(ws["ne"], 1e-30)
    return (1.56 * zeff * (1 + 1.4 * zeff) * (1 + 0.52 * zeff)
            / (1 + 2.56 * zeff) / (1 + 0.29 * zeff) / (zeff + np.sqrt(2) / 2))


@quantity(
    name="fhe_qeprll_th",
    requires=[],
    description="electron heat flux from parallel current, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_qeprll_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    c071 = ws.get("c071")
    if c071 is None:
        c071 = calc_c071(watch=watch, grid=grid)
    c071f = intface(grid, c071, 1, "vol")
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    fch_pTe = _b2mn_param(watch, "b2tfhe_fch_pTe", 1.0)
    return -c071f * ws["fch_par_th"] * tef * fch_pTe


@quantity(
    name="fhe_qeprll_r",
    requires=[],
    description="electron heat flux from parallel current, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_qeprll_r(watch=None, grid=None, **kw):
    return np.zeros(grid.n_faces)


@quantity(
    name="div_fhe_qeprll",
    requires=[],
    description="div of electron parallel-current heat flux",
    unit="W/m³",
)
def calc_div_fhe_qeprll(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth = ws.get("fhe_qeprll_th")
    if fth is None:
        fth = calc_fhe_qeprll_th(watch=watch, grid=grid, comp=comp)
    return _div_th(grid, fth)


def _b2mn_param(watch, key: str, default: float) -> float:
    """Read a scalar parameter from b2mn.dat (MATLAB read_b2mn_dat.m)."""
    try:
        from pathlib import Path
        path = Path(watch.path) / ".." / "b2mn.dat"
        if not path.exists():
            path = Path(watch.path) / "b2mn.dat"
        if not path.exists():
            return default
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.replace("'", "").split()
            if len(parts) >= 2 and parts[0] == key:
                try:
                    return float(parts[1])
                except ValueError:
                    return default
    except Exception:
        pass
    return default


@quantity(
    name="fhe_alphaEhat_th",
    requires=[],
    description="electron heat flux from parallel E-hat, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_alphaEhat_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fth = ws.get("fhe_qeprll_th")
    if fth is None:
        fth = calc_fhe_qeprll_th(watch=watch, grid=grid, comp=comp)
    alfTeEh = _b2mn_param(watch, "b2tfhe_alfTeEh", 0.0)
    fch_pTe = _b2mn_param(watch, "b2tfhe_fch_pTe", 1.0)
    if fch_pTe == 1:
        alfTeEh = 0.0
    if alfTeEh > 0:
        # b2xehx contribution (requires external E-field solver)
        tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
        calf = ws.get("calf_clLucFlim_th", np.zeros(grid.n_faces))
        fth = fth + calf * tef * _b2xehx_approx(grid, ws) * QE * alfTeEh
    return fth


def _b2xehx_approx(grid, ws):
    """Approximation of b2xehx (external parallel E-field) — zero for now.

    MATLAB b2xehx solves the parallel current balance; only needed when
    b2tfhe_alfTeEh > 0 (off by default).
    """
    return np.zeros(grid.n_faces)


@quantity(
    name="fhe_alphaEhat_r",
    requires=[],
    description="electron heat flux from parallel E-hat, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_alphaEhat_r(watch=None, grid=None, **kw):
    return np.zeros(grid.n_faces)


def _fhi_conv_sum(ws, fname: str) -> np.ndarray:
    """fhi_* = 1.5 * sum(fna_*) * tif * qe (nFc,)."""
    return 1.5 * ws[fname].sum(axis=1)


@quantity(
    name="fhi_nuExB_th",
    requires=[],
    description="ion heat flux from ExB drift, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_nuExB_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return _fhi_conv_sum(ws, "fna_nuExB_th") * tif * QE


@quantity(
    name="fhi_nuExB_r",
    requires=[],
    description="ion heat flux from ExB drift, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_nuExB_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return _fhi_conv_sum(ws, "fna_nuExB_r") * tif * QE


@quantity(
    name="fhi_nupar_th",
    requires=[],
    description="ion heat flux from parallel flow, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_nupar_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return _fhi_conv_sum(ws, "fna_nupar_th") * tif * QE


@quantity(
    name="fhi_fnaAN_th",
    requires=[],
    description="ion heat flux from anomalous pinch, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_fnaAN_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return 1.5 * (ws["fna_nuAN_th"] + ws["fna_Dgradn_th"]).sum(axis=1) * tif * QE


@quantity(
    name="fhi_fnaAN_r",
    requires=[],
    description="ion heat flux from anomalous pinch, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_fnaAN_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return 1.5 * (ws["fna_nuAN_r"] + ws["fna_Dgradn_r"]).sum(axis=1) * tif * QE


@quantity(
    name="fhi_conv_th",
    requires=[],
    description="convective ion heat flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_conv_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return _fhi_conv_sum(ws, "fna_th") * tif * QE


@quantity(
    name="fhi_conv_r",
    requires=[],
    description="convective ion heat flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_conv_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return _fhi_conv_sum(ws, "fna_r") * tif * QE


@quantity(
    name="fhi_fni_th",
    requires=[],
    description="ion heat flux from friction, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_fni_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return 1.5 * ws["fna_fha_th"].sum(axis=1) * tif * QE


@quantity(
    name="fhi_fni_r",
    requires=[],
    description="ion heat flux from friction, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_fni_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    return 1.5 * ws["fna_fha_r"].sum(axis=1) * tif * QE


@quantity(
    name="fhi_curr_th",
    requires=[],
    description="ion heat flux from current, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_curr_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    fna_curr = ws.get("fna_curr_th")
    if fna_curr is None:
        fna_curr = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    return 1.5 * fna_curr.sum(axis=1) * tif * QE


@quantity(
    name="fhi_curr_r",
    requires=[],
    description="ion heat flux from current, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_curr_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    fna_curr = ws.get("fna_curr_r")
    if fna_curr is None:
        fna_curr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    return 1.5 * fna_curr.sum(axis=1) * tif * QE


@quantity(
    name="fhe_conv_th",
    requires=[],
    description="convective electron heat flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_conv_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    fne_th = ws.get("fne_th")
    if fne_th is None:
        fne_th = calc_fne_th(watch=watch, grid=grid, comp=comp)
    return 1.5 * fne_th * tef * QE


@quantity(
    name="fhe_conv_r",
    requires=[],
    description="convective electron heat flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_conv_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    fne_r = ws.get("fne_r")
    if fne_r is None:
        fne_r = calc_fne_r(watch=watch, grid=grid, comp=comp)
    return 1.5 * fne_r * tef * QE


@quantity(
    name="fhe_gradte_th",
    requires=[],
    description="conductive electron heat flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhe_gradte_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return ws["fhe_cond_th"]


@quantity(
    name="fhe_gradte_r",
    requires=[],
    description="conductive electron heat flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fhe_gradte_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return ws["fhe_cond_r"]


@quantity(
    name="fhi_gradte_th",
    requires=[],
    description="conductive ion heat flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fhi_gradte_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return ws["fhi_cond_th"]


@quantity(
    name="fhi_gradte_r",
    requires=[],
    description="conductive ion heat flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fhi_gradte_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return ws["fhi_cond_r"]


@quantity(
    name="fh_joule_th",
    requires=[],
    description="Joule heating flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fh_joule_th(watch=None, grid=None, **kw):
    ws = _ws(watch)
    pof = ws["pof"] if "pof" in ws else intface(grid, ws["po"], 0, _intface_method(grid))
    return ws["fch_th"] * pof


@quantity(
    name="fh_joule_r",
    requires=[],
    description="Joule heating flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fh_joule_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    pof = ws["pof"] if "pof" in ws else intface(grid, ws["po"], 0, _intface_method(grid))
    return ws["fch_r"] * pof


# ──────────────────────────────────────────────────────────────
# OMP-integrated fluxes (lines 631-767)
# ──────────────────────────────────────────────────────────────

def _omp_integrals(watch, grid, comp) -> dict:
    """Integrate face fluxes along outer/inner midplane flux surfaces.

    Port of calc_additional.m lines 631-767.  Returns dict of arrays:
      - species arrays: (len_cs, ns): Fna_mdf_r, Fna_mdf, Fna_Dgradn_r,
        Fna_flo_r, Fna_nuBgradB_r, Fna_dia_mdf_r, FnaExB_r, Fna_nuAN_r,
        Fmo_r, Fmo_ExB_r, Fmo_nuBgradB_r, Fmo_fnaVan_r, Fmo_vis_r,
        Fmo_conv_r, Fmo_curr_r, Fmo_fna_mo_r, Fch_AN, Fch_in, Fch_inert,
        Fch_vispar, Fch_visper, Fch_visq, Fna_curr_tot
      - scalar arrays: (len_cs,): Fhi_tot, Fhe_tot, Fhe_gradte, Fhn_tot,
        Fch_nuBgradB, Fch, Fhi_nuExB, Fhe_nuExB, Fhi_dia_mdf, Fhe_dia_mdf,
        Fhi_nuBgradB, Fhe_nuBgradB, Fhi_curr, Fhe_curr, Fei_curr, Fei_kin,
        Fhi_AN, Fhe_AN, Fhi_cond, Fhi_flo, Fhi_conv, Fhi_mdf, Fhe_cond,
        Fhe_flo, Fhe_conv, Fhe_mdf, Fee, Fei, Fet
    """
    cached = getattr(watch, "_omp_integrals", None)
    if cached is not None:
        return cached

    ws = _ws(watch)
    from solps_analysis.io.matlab_vars import species_am

    # OMP/IMP coordinates
    watch.compute_regions()
    omp = grid.outer_midplane_cells
    if omp is None or len(omp) < 2:
        watch._omp_integrals = {}
        return {}

    def _fc_coords(cells):
        out = []
        for k in range(1, len(cells)):
            prev_fcs = grid.cv_fc[grid.cv_fc_p[cells[k-1], 0]:grid.cv_fc_p[cells[k-1], 0]+grid.cv_fc_p[cells[k-1], 1]]
            cur_fcs = grid.cv_fc[grid.cv_fc_p[cells[k], 0]:grid.cv_fc_p[cells[k], 0]+grid.cv_fc_p[cells[k], 1]]
            common = np.intersect1d(prev_fcs, cur_fcs)
            if common.size:
                out.append(common[0])
        return np.array(out, dtype=np.intp)

    fc_omp = _fc_coords(omp)
    imp = grid.inner_midplane_cells
    fc_imp = _fc_coords(imp) if imp is not None else np.array([], dtype=np.intp)
    len_cs = len(fc_omp)

    # isCrossOutmidpl / isCrossInmidpl (1-based index into fc_omp/fc_imp)
    fs_fc_p = grid.fs_fc_p
    fs_fc = grid.fs_fc
    n_fs = fs_fc_p.shape[0]
    is_cross_out = np.zeros(n_fs, dtype=int)
    is_cross_in = np.zeros(n_fs, dtype=int)
    for i in range(n_fs):
        fcs = fs_fc[fs_fc_p[i, 0]:fs_fc_p[i, 0] + fs_fc_p[i, 1]]
        inter = np.intersect1d(fcs, fc_omp)
        if inter.size:
            is_cross_out[i] = np.where(fc_omp == inter.min())[0][0] + 1
        inter = np.intersect1d(fcs, fc_imp)
        if inter.size:
            is_cross_in[i] = np.where(fc_imp == inter.min())[0][0] + 1

    ns = ws["fna_mdf_r"].shape[1]
    n_fc = grid.n_faces
    zs = _zs(comp)
    ams = species_am(watch) if zs is not None else None

    names_species = [
        "Fna_mdf_r", "Fna_mdf", "Fna_Dgradn_r", "Fna_flo_r", "Fna_nuBgradB_r",
        "Fna_dia_mdf_r", "FnaExB_r", "Fna_nuAN_r", "Fmo_r", "Fmo_ExB_r",
        "Fmo_nuBgradB_r", "Fmo_fnaVan_r", "Fmo_vis_r", "Fmo_conv_r",
        "Fmo_curr_r", "Fmo_fna_mo_r", "Fch_AN", "Fch_in", "Fch_inert",
        "Fch_vispar", "Fch_visper", "Fch_visq", "Fna_curr_tot",
    ]
    out = {name: np.zeros((len_cs, ns)) for name in names_species}
    names_scalar = [
        "Fhi_tot", "Fhe_tot", "Fhe_gradte", "Fhn_tot", "Fch_nuBgradB", "Fch",
        "Fhi_nuExB", "Fhe_nuExB", "Fhi_dia_mdf", "Fhe_dia_mdf",
        "Fhi_nuBgradB", "Fhe_nuBgradB", "Fhi_curr", "Fhe_curr", "Fei_curr",
        "Fei_kin", "Fhi_AN", "Fhe_AN", "Fhi_cond", "Fhi_flo", "Fhi_conv",
        "Fhi_mdf", "Fhe_cond", "Fhe_flo", "Fhe_conv", "Fhe_mdf",
        "Fee", "Fei", "Fet",
    ]
    for name in names_scalar:
        out[name] = np.zeros(len_cs)

    # face quantities needed
    fq = {
        "fna_mdf_r": ws["fna_mdf_r"], "fna_mdf_th": ws["fna_mdf_th"],
        "fna_Dgradn_r": ws["fna_Dgradn_r"], "fna_flo_r": ws["fna_flo_r"],
        "fna_nuBgradB_r": ws["fna_nuBgradB_r"],
        "fna_dia_mdf_r": ws["fna_dia_mdf_r"], "fna_nuExB_r": ws["fna_nuExB_r"],
        "fna_nuAN_r": ws["fna_nuAN_r"],
        "fmo_r": ws["fmo_r"], "fmo_nuExB_r": ws["fmo_nuExB_r"] if "fmo_nuExB_r" in ws else None,
        "fmo_nuBgradB_r": ws["fmo_nuBgradB_r"] if "fmo_nuBgradB_r" in ws else None,
        "fmo_fnaAN_r": ws["fmo_fnaAN_r"] if "fmo_fnaAN_r" in ws else None,
        "fmo_vis_r": ws["fmo_vis_r"],
        "fmo_conv_r": ws["fmo_conv_r"] if "fmo_conv_r" in ws else None,
        "fch_AN_r": ws["fch_AN_r"], "fch_in_r": ws["fch_in_r"],
        "fch_inert_r": ws["fch_inert_r"], "fch_vispar_r": ws["fch_vispar_r"],
        "fch_visper_r": ws["fch_visper_r"], "fch_visq_r": ws["fch_visq_r"],
        "fna_curr_r": ws.get("fna_curr_r"),
        "fhi_mdf_r": ws["fhi_mdf_r"], "fhe_mdf_r": ws["fhe_mdf_r"],
        "fhi_nuExB_r": ws.get("fhi_nuExB_r"), "fhe_nuExB_r": ws.get("fhe_nuExB_r"),
        "fhi_nuBgradB_r": ws.get("fhi_nuBgradB_r"), "fhe_nuBgradB_r": ws.get("fhe_nuBgradB_r"),
        "fhi_dia_mdf_r": ws["fhi_dia_mdf_r"], "fhe_dia_mdf_r": ws["fhe_dia_mdf_r"],
        "fhi_cond_r": ws["fhi_cond_r"], "fhi_flo_r": ws["fhi_flo_r"],
        "fhi_conv_r": ws.get("fhi_conv_r"), "fhi_fnaAN_r": ws.get("fhi_fnaAN_r"),
        "fhe_cond_r": ws["fhe_cond_r"], "fhe_flo_r": ws["fhe_flo_r"],
        "fhe_conv_r": ws.get("fhe_conv_r"), "fhe_fnaAN_r": ws.get("fhe_fnaAN_r"),
        "fhe_gradte_r": ws.get("fhe_gradte_r"),
        "fch_nuBgradB_r": ws["fch_nuBgradB_r"], "fch_r": ws["fch_r"],
        "fne_curr_r": ws.get("fne_curr_r"), "tef": ws.get("tef"),
        "fee_r": ws.get("fee_r"), "fei_r": ws.get("fei_r"),
        "fet_r": ws.get("fet_r"), "fei_kin_r": ws.get("fei_kin_r"),
        "fei_curr_r": ws.get("fei_curr_r"),
    }

    # lazily compute derived face quantities
    import solps_analysis.construct.builtin.calc_additional as _ca
    import solps_analysis.construct.builtin.energy_balance as _eb

    def _face(name, fallback_fn=None):
        v = fq.get(name)
        if v is None and fallback_fn is not None:
            v = fallback_fn()
        if v is None:
            v = np.zeros(n_fc)
        return v

    def _fallback(module, name):
        fn = getattr(module, f"calc_{name}", None)
        if fn is None:
            return np.zeros(n_fc)
        try:
            return fn(watch=watch, grid=grid, comp=comp)
        except Exception:
            return np.zeros(n_fc)

    fq["fhi_nuExB_r"] = _face("fhi_nuExB_r", lambda: _fallback(_ca, "fhi_nuExB_r"))
    fq["fhe_nuExB_r"] = _face("fhe_nuExB_r", lambda: _fallback(_ca, "fhe_nuExB_r"))
    fq["fhi_nuBgradB_r"] = _face("fhi_nuBgradB_r", lambda: _fallback(_ca, "fhi_nuBgradB_r"))
    fq["fhe_nuBgradB_r"] = _face("fhe_nuBgradB_r", lambda: _fallback(_ca, "fhe_nuBgradB_r"))
    fq["fhi_conv_r"] = _face("fhi_conv_r", lambda: _fallback(_ca, "fhi_conv_r"))
    fq["fhe_conv_r"] = _face("fhe_conv_r", lambda: _fallback(_ca, "fhe_conv_r"))
    fq["fhi_fnaAN_r"] = _face("fhi_fnaAN_r", lambda: _fallback(_ca, "fhi_fnaAN_r"))
    fq["fhe_fnaAN_r"] = _face("fhe_fnaAN_r", lambda: _fallback(_ca, "fhe_fnaAN_r"))
    fq["fhe_gradte_r"] = _face("fhe_gradte_r", lambda: _fallback(_ca, "fhe_gradte_r"))
    fq["fne_curr_r"] = _face("fne_curr_r", lambda: _fallback(_ca, "fne_curr_r"))
    fq["fee_r"] = _face("fee_r", lambda: _fallback(_eb, "fee_r"))
    fq["fei_r"] = _face("fei_r", lambda: _fallback(_eb, "fei_r"))
    fq["fet_r"] = _face("fet_r", lambda: _fallback(_eb, "fet_r"))
    fq["fei_kin_r"] = _face("fei_kin_r", lambda: _fallback(_eb, "fei_kin_r"))
    fq["fei_curr_r"] = _face("fei_curr_r", lambda: _fallback(_eb, "fei_curr_r"))
    fq["tef"] = _face("tef", lambda: _fallback(_ca, "tef"))

    fc_hz = ws.get("fc_hz", np.ones(n_fc))
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    n_ci = grid.n_core_cells
    cv_reg = grid.cv_reg
    fc_cv = grid.fc_cv

    # Precompute fallback values via lazy import of calc functions
    def _lazy(name, *args):
        import solps_analysis.construct.builtin.calc_additional as ca
        fn = getattr(ca, f"calc_{name}")
        return fn(watch=watch, grid=grid, comp=comp)

    for i_fs in range(n_fs):
        fcs = fs_fc[fs_fc_p[i_fs, 0]:fs_fc_p[i_fs, 0] + fs_fc_p[i_fs, 1]]
        if fcs.size == 0:
            continue
        i_cs = is_cross_out[i_fs]
        if i_cs == 0:
            i_cs = is_cross_in[i_fs]
        if i_cs == 0:
            continue
        if i_cs > len_cs:
            continue
        i_cs0 = i_cs - 1  # 0-based row

        for k in range(len(fcs)):
            fc = fcs[k]
            cell_regs = cv_reg[fc_cv[fc]]
            if not (np.any(cell_regs == 1) or np.any(cell_regs == 5)
                    or np.any(cell_regs == 2) or np.any(cell_regs == 6)):
                continue

            for is_ in range(ns):
                if zs is not None and zs[is_] == 0:
                    if watch.neut is not None and i_cs == 1:
                        # EIRENE neutral: source integrated over SOL column
                        tmp = np.where(np.mod(cv_reg, 4) == 1)[0]
                        tmp = tmp[tmp > n_ci]
                        s = ws["sna"][tmp, is_].sum() if "sna" in ws else 0.0
                        out["Fna_mdf_r"][i_cs0, is_] = s
                        out["Fna_mdf"][i_cs0, is_] = s
                    continue
                out["Fna_mdf_r"][i_cs0, is_] += fq["fna_mdf_r"][fc, is_]
                out["Fna_mdf"][i_cs0, is_] += fq["fna_mdf_r"][fc, is_] + fq["fna_mdf_th"][fc, is_]
                out["Fna_Dgradn_r"][i_cs0, is_] += fq["fna_Dgradn_r"][fc, is_]
                out["Fna_flo_r"][i_cs0, is_] += fq["fna_flo_r"][fc, is_]
                out["Fna_nuBgradB_r"][i_cs0, is_] += fq["fna_nuBgradB_r"][fc, is_]
                out["Fna_dia_mdf_r"][i_cs0, is_] += fq["fna_dia_mdf_r"][fc, is_]
                out["FnaExB_r"][i_cs0, is_] += fq["fna_nuExB_r"][fc, is_]
                out["Fna_nuAN_r"][i_cs0, is_] += fq["fna_nuAN_r"][fc, is_]
                out["Fmo_r"][i_cs0, is_] += fq["fmo_r"][fc, is_]
                if fq["fmo_nuExB_r"] is not None:
                    out["Fmo_ExB_r"][i_cs0, is_] += fq["fmo_nuExB_r"][fc, is_]
                if fq["fmo_nuBgradB_r"] is not None:
                    out["Fmo_nuBgradB_r"][i_cs0, is_] += fq["fmo_nuBgradB_r"][fc, is_]
                if fq["fmo_fnaAN_r"] is not None:
                    out["Fmo_fnaVan_r"][i_cs0, is_] += fq["fmo_fnaAN_r"][fc, is_]
                out["Fmo_vis_r"][i_cs0, is_] += fq["fmo_vis_r"][fc, is_]
                if fq["fmo_conv_r"] is not None:
                    out["Fmo_conv_r"][i_cs0, is_] += fq["fmo_conv_r"][fc, is_]
                if fq["fna_curr_r"] is not None:
                    out["Fmo_curr_r"][i_cs0, is_] += (fq["fna_curr_r"][fc, is_]
                                                     * fc_hz[fc] * ams[is_] * MP * uaf[fc, is_])
                    out["Fna_curr_tot"][i_cs0, is_] += fq["fna_curr_r"][fc, is_]
                out["Fmo_fna_mo_r"][i_cs0, is_] += ws["fna_mo_r"][fc, is_] * ams[is_] * MP * uaf[fc, is_]
                out["Fch_AN"][i_cs0, is_] += fq["fch_AN_r"][fc, is_]
                out["Fch_in"][i_cs0, is_] += fq["fch_in_r"][fc, is_]
                out["Fch_inert"][i_cs0, is_] += fq["fch_inert_r"][fc, is_]
                out["Fch_vispar"][i_cs0, is_] += fq["fch_vispar_r"][fc, is_]
                out["Fch_visper"][i_cs0, is_] += fq["fch_visper_r"][fc, is_]
                out["Fch_visq"][i_cs0, is_] += fq["fch_visq_r"][fc, is_]

            out["Fhi_nuExB"][i_cs0] += fq["fhi_nuExB_r"][fc]
            out["Fhe_nuExB"][i_cs0] += fq["fhe_nuExB_r"][fc]
            out["Fhi_tot"][i_cs0] += fq["fhi_mdf_r"][fc] + 2.0/3.0 * fq["fhi_nuExB_r"][fc]
            out["Fhe_tot"][i_cs0] += fq["fhe_mdf_r"][fc] + 2.0/3.0 * fq["fhe_nuExB_r"][fc]
            out["Fhi_mdf"][i_cs0] = out["Fhi_tot"][i_cs0] + fq["fhi_mdf_r"][fc]
            out["Fhe_mdf"][i_cs0] = out["Fhe_tot"][i_cs0] + fq["fhe_mdf_r"][fc]
            out["Fhe_gradte"][i_cs0] += fq["fhe_gradte_r"][fc]
            out["Fee"][i_cs0] += fq["fee_r"][fc]
            out["Fei"][i_cs0] += fq["fei_r"][fc]
            out["Fet"][i_cs0] += fq["fet_r"][fc]
            out["Fhi_nuBgradB"][i_cs0] += fq["fhi_nuBgradB_r"][fc]
            out["Fhe_nuBgradB"][i_cs0] += fq["fhe_nuBgradB_r"][fc]
            out["Fei_kin"][i_cs0] += fq["fei_kin_r"][fc]
            out["Fhe_curr"][i_cs0] += 1.5 * QE * fq["fne_curr_r"][fc] * fq["tef"][fc]
            out["Fei_curr"][i_cs0] += fq["fei_curr_r"][fc]
            out["Fhi_dia_mdf"][i_cs0] += fq["fhi_dia_mdf_r"][fc]
            out["Fhe_dia_mdf"][i_cs0] += fq["fhe_dia_mdf_r"][fc]
            out["Fhi_AN"][i_cs0] += fq["fhi_fnaAN_r"][fc]
            out["Fhe_AN"][i_cs0] += fq["fhe_fnaAN_r"][fc]
            out["Fhi_cond"][i_cs0] += fq["fhi_cond_r"][fc]
            out["Fhi_conv"][i_cs0] += fq["fhi_conv_r"][fc]
            out["Fhi_flo"][i_cs0] += fq["fhi_flo_r"][fc]
            out["Fhe_cond"][i_cs0] += fq["fhe_cond_r"][fc]
            out["Fhe_conv"][i_cs0] += fq["fhe_conv_r"][fc]
            out["Fhe_flo"][i_cs0] += fq["fhe_flo_r"][fc]
            out["Fch_nuBgradB"][i_cs0] += fq["fch_nuBgradB_r"][fc]
            out["Fch"][i_cs0] += fq["fch_r"][fc]

    watch._omp_integrals = out
    return out


def _omp_get(watch, grid, comp, name: str) -> np.ndarray:
    d = _omp_integrals(watch, grid, comp)
    return d.get(name, np.zeros(0))


@quantity(
    name="Fna_mdf_r",
    requires=[],
    description="OMP-integrated radial particle flux (mdf)",
    unit="s⁻¹",
)
def calc_Fna_mdf_r(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fna_mdf_r")


@quantity(
    name="Fhi_tot",
    requires=[],
    description="OMP-integrated total ion heat flux",
    unit="W",
)
def calc_Fhi_tot(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fhi_tot")


@quantity(
    name="Fhe_tot",
    requires=[],
    description="OMP-integrated total electron heat flux",
    unit="W",
)
def calc_Fhe_tot(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fhe_tot")


@quantity(
    name="Fch",
    requires=[],
    description="OMP-integrated total current",
    unit="A",
)
def calc_Fch(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fch")


@quantity(
    name="Fee",
    requires=[],
    description="OMP-integrated electron energy flux (extended balance)",
    unit="W",
)
def calc_Fee(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fee")


@quantity(
    name="Fei",
    requires=[],
    description="OMP-integrated ion energy flux (extended balance)",
    unit="W",
)
def calc_Fei(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fei")


@quantity(
    name="Fet",
    requires=[],
    description="OMP-integrated total energy flux (extended balance)",
    unit="W",
)
def calc_Fet(watch=None, grid=None, comp=None, **kw):
    return _omp_get(watch, grid, comp, "Fet")


# ──────────────────────────────────────────────────────────────
# Shi/She flux-tube sources (lines 768-815)
# ──────────────────────────────────────────────────────────────

def _ft_integrals(watch, grid, comp) -> dict:
    """Integrate heat sources along flux tubes crossing the midplane.

    Port of calc_additional.m lines 768-815.
    """
    cached = getattr(watch, "_ft_integrals", None)
    if cached is not None:
        return cached

    ws = _ws(watch)
    watch.compute_regions()
    omp = grid.outer_midplane_cells
    if omp is None or len(omp) < 2:
        watch._ft_integrals = {}
        return {}

    def _fc_coords(cells):
        out = []
        for k in range(1, len(cells)):
            prev_fcs = grid.cv_fc[grid.cv_fc_p[cells[k-1], 0]:grid.cv_fc_p[cells[k-1], 0]+grid.cv_fc_p[cells[k-1], 1]]
            cur_fcs = grid.cv_fc[grid.cv_fc_p[cells[k], 0]:grid.cv_fc_p[cells[k], 0]+grid.cv_fc_p[cells[k], 1]]
            common = np.intersect1d(prev_fcs, cur_fcs)
            if common.size:
                out.append(common[0])
        return np.array(out, dtype=np.intp)

    fc_omp = _fc_coords(omp)
    imp = grid.inner_midplane_cells
    fc_imp = _fc_coords(imp) if imp is not None else np.array([], dtype=np.intp)
    len_cs = len(omp)  # MATLAB uses length(outmidpl_coords)

    # isCrossOutmidplCv: flux tube index → OMP position
    ft_cv_p = grid.ft_cv_p
    ft_cv = grid.ft_cv
    n_ft = ft_cv_p.shape[0]
    is_cross_out_cv = np.zeros(n_ft, dtype=int)
    is_cross_in_cv = np.zeros(n_ft, dtype=int)
    for i in range(n_ft):
        cvs = ft_cv[ft_cv_p[i, 0]:ft_cv_p[i, 0] + ft_cv_p[i, 1]]
        inter = np.intersect1d(cvs, omp)
        if inter.size:
            is_cross_out_cv[i] = np.where(omp == inter.min())[0][0] + 1
        inter = np.intersect1d(cvs, imp) if imp is not None else np.array([], dtype=int)
        if inter.size:
            is_cross_in_cv[i] = np.where(imp == inter.min())[0][0] + 1

    names = ["Shi", "Shi_vis", "Shi_eir", "Shi_du", "Shi_dd", "Shi_dd1",
             "Shi_fr", "Shi_BC", "She", "She_ei", "She_eir", "She_du",
             "She_dd", "She_fr", "She_BC", "She_rad"]
    out = {name: np.zeros(len_cs) for name in names}

    # shi_dd1 (line 768): ExB-driven ion heat source
    zs = _zs(comp)
    ions = np.where(zs is not None and np.asarray(zs) != 0)[0] if zs is not None else np.arange(ws["na"].shape[1])
    vel_exb_th = ws["vel_ExB_th"]  # (nFc, ns)
    vel_exb_r = ws["vel_ExB_r"]
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    fc_qalf = grid.fc_qalf if grid.fc_qalf is not None else np.ones((grid.n_faces, 2))
    flow_th = vel_exb_th[:, 1] * fc_qalf[:, 0] * fc_s  # species 2 = main ion (0-based 1)
    flow_r = vel_exb_r[:, 1] * fc_qalf[:, 1] * fc_s
    div_vel = div_us(grid, np.column_stack([flow_th, flow_r]))
    shi_dd1_cell = -ws["na"][:, ions].sum(axis=1) * ws["ti"] * QE * div_vel

    cv_reg = grid.cv_reg
    for i_ft in range(n_ft):
        i_cs = is_cross_out_cv[i_ft]
        if i_cs == 0:
            i_cs = is_cross_in_cv[i_ft]
        if i_cs == 0 or i_cs > len_cs:
            continue
        i_cs0 = i_cs - 1
        cvs = ft_cv[ft_cv_p[i_ft, 0]:ft_cv_p[i_ft, 0] + ft_cv_p[i_ft, 1]]
        for i_cv in cvs:
            if np.mod(cv_reg[i_cv], 4) == 1 or np.mod(cv_reg[i_cv], 4) == 2:
                out["Shi"][i_cs0] += ws["shi"][i_cv]
                out["Shi_vis"][i_cs0] += ws.get("shi_viscl", 0)[i_cv] + ws.get("shi_visan", 0)[i_cv]
                out["Shi_eir"][i_cs0] += ws.get("shi_eir", 0)[i_cv]
                out["Shi_du"][i_cs0] += ws.get("shi_du", 0)[i_cv]
                out["Shi_dd"][i_cs0] += ws.get("shi_dd", 0)[i_cv]
                out["Shi_dd1"][i_cs0] += shi_dd1_cell[i_cv]
                out["Shi_fr"][i_cs0] += ws.get("shi_fr", 0)[i_cv]
                out["Shi_BC"][i_cs0] += ws.get("shi_BC", 0)[i_cv]
                out["She"][i_cs0] += ws["she"][i_cv]
                out["She_ei"][i_cs0] += ws.get("she_ei", 0)[i_cv]
                out["She_eir"][i_cs0] += ws.get("she_eir", 0)[i_cv]
                out["She_du"][i_cs0] += ws.get("she_du", 0)[i_cv]
                out["She_dd"][i_cs0] += ws.get("she_dd", 0)[i_cv]
                out["She_fr"][i_cs0] += ws.get("she_fr", 0)[i_cv]
                out["She_BC"][i_cs0] += ws.get("she_BC", 0)[i_cv]
                out["She_rad"][i_cs0] += ws.get("she_rad", 0)[i_cv]

    watch._ft_integrals = out
    return out


def _ft_get(watch, grid, comp, name: str) -> np.ndarray:
    d = _ft_integrals(watch, grid, comp)
    return d.get(name, np.zeros(0))


@quantity(
    name="Shi",
    requires=[],
    description="flux-tube integrated ion heat source",
    unit="W",
)
def calc_Shi(watch=None, grid=None, comp=None, **kw):
    return _ft_get(watch, grid, comp, "Shi")


@quantity(
    name="She",
    requires=[],
    description="flux-tube integrated electron heat source",
    unit="W",
)
def calc_She(watch=None, grid=None, comp=None, **kw):
    return _ft_get(watch, grid, comp, "She")


@quantity(
    name="Shi_dd1",
    requires=[],
    description="flux-tube integrated ExB ion heat source",
    unit="W",
)
def calc_Shi_dd1(watch=None, grid=None, comp=None, **kw):
    return _ft_get(watch, grid, comp, "Shi_dd1")


# ──────────────────────────────────────────────────────────────
# Radiation (lines 821-859)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="Qrad",
    requires=[],
    description="radiation power loss per element (core)",
    unit="W/m³",
)
def calc_Qrad(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if comp is None:
        return np.zeros((grid.n_cells, 1))
    nsorts = comp.n_elements
    n_ci = grid.n_core_cells
    indices = comp.element_indices_list
    out = np.zeros((grid.n_cells, nsorts))
    for ia in range(nsorts):
        is_list = indices[ia]
        q = np.zeros(grid.n_cells)
        q[:n_ci] = (ws["she_radlin"][:n_ci, is_list].sum(axis=1)
                    + ws["she_radbrm"][:n_ci, is_list].sum(axis=1))
        neut = watch.neut
        if neut is not None:
            from solps_analysis.construct.builtin.eirene import _unpack_eirene_cell_data
            eneutrad = neut.get("eneutrad")
            if eneutrad is not None:
                eneutrad_1d = _unpack_eirene_cell_data(np.asarray(eneutrad), grid)
                if eneutrad_1d.ndim == 2 and eneutrad_1d.shape[1] > ia:
                    q[:n_ci] = q[:n_ci] - eneutrad_1d[:n_ci, ia]
            atm2mol = neut.get("atm2mol")
            if atm2mol is not None and len(atm2mol) > ia and atm2mol[ia] != 0:
                emolrad = neut.get("emolrad")
                if emolrad is not None:
                    emolrad_1d = _unpack_eirene_cell_data(np.asarray(emolrad), grid)
                    if emolrad_1d.ndim == 2 and emolrad_1d.shape[1] > atm2mol[ia]:
                        q[:n_ci] = q[:n_ci] - emolrad_1d[:n_ci, atm2mol[ia]]
            atm2ion = neut.get("atm2ion")
            if atm2ion is not None and len(atm2ion) > ia and atm2ion[ia] != 0:
                eionrad = neut.get("eionrad")
                if eionrad is not None:
                    eionrad_1d = _unpack_eirene_cell_data(np.asarray(eionrad), grid)
                    if eionrad_1d.ndim == 2 and eionrad_1d.shape[1] > atm2ion[ia]:
                        q[:n_ci] = q[:n_ci] - eionrad_1d[:n_ci, atm2ion[ia]]
        out[:, ia] = q
    return out


@quantity(
    name="Qrad_tot",
    requires=[],
    description="total radiation power per element (core)",
    unit="W",
)
def calc_Qrad_tot(watch=None, grid=None, comp=None, **kw):
    q = _ws(watch).get("Qrad")
    if q is None:
        q = calc_Qrad(watch=watch, grid=grid, comp=comp)
    return q.sum(axis=0)


@quantity(
    name="Qrad_tot_sum",
    requires=[],
    description="total radiation power (all elements)",
    unit="W",
)
def calc_Qrad_tot_sum(watch=None, grid=None, comp=None, **kw):
    q = _ws(watch).get("Qrad")
    if q is None:
        q = calc_Qrad(watch=watch, grid=grid, comp=comp)
    return np.array([q.sum()])


# ──────────────────────────────────────────────────────────────
# Radial-column sums (lines 862-925)
# ──────────────────────────────────────────────────────────────

def _column_sums(watch, grid, comp) -> dict:
    """Sum sources along radial columns crossing the OMP.

    Port of calc_additional.m lines 862-925.
    """
    cached = getattr(watch, "_column_sums", None)
    if cached is not None:
        return cached

    ws = _ws(watch)
    watch.compute_regions()
    omp = grid.outer_midplane_cells
    if omp is None or len(omp) < 2:
        watch._column_sums = {}
        return {}

    len_cs = len(omp)
    ns = ws["fna_mdf_r"].shape[1]
    nsorts = comp.n_elements if comp is not None else 1

    Sna = np.zeros((len_cs, ns))
    div_Fna_mdf = np.zeros((len_cs, ns))
    Snas = np.zeros((len_cs, nsorts))
    Qrads = np.zeros((len_cs, nsorts))
    ftVol = np.zeros(len_cs)
    te_mean = np.zeros(len_cs)

    qrad = ws.get("Qrad")
    if qrad is None:
        qrad = calc_Qrad(watch=watch, grid=grid, comp=comp)
    snas = ws.get("snas")
    if snas is None:
        snas = calc_snas(watch=watch, grid=grid, comp=comp)
    div_fna_mdf = ws.get("div_fna_mdf")
    if div_fna_mdf is None:
        div_fna_mdf = calc_div_fna_mdf(watch=watch, grid=grid)
    eirene_flag = getattr(watch, "neut", None) is not None
    sna_src = ws["sna_eir"] if eirene_flag and "sna_eir" in ws else ws.get("sna")

    cv_reg = grid.cv_reg
    cv_vol = grid.cv_vol
    te = ws["te"]
    imap_fcy = grid.imap_fcy
    imap_fcx = grid.imap_fcx
    fc_cv = grid.fc_cv
    n_ci = grid.n_core_cells
    wall_mask = grid.fc_lbl != 0 if grid.fc_lbl is not None else np.zeros(grid.n_faces, dtype=bool)

    r_index = 0
    if imap_fcy is None:
        watch._column_sums = {}
        return {}

    n_cols = imap_fcy.shape[1]
    for col in range(n_cols):
        col_fcy = imap_fcy[:, col]
        col_fcy = col_fcy[col_fcy != 0]
        if col_fcy.size == 0:
            continue
        cvs_to_fcy = fc_cv[col_fcy - 1]  # 0-based
        if not np.intersect1d(omp, cvs_to_fcy.ravel()).size:
            continue

        # drop faces whose cells are in regions 3/8/7/4 (PFR/private flux)
        keep = np.ones(len(col_fcy), dtype=bool)
        for k in range(len(col_fcy)):
            if np.any(np.isin(cv_reg[fc_cv[col_fcy[k] - 1]], [3, 8, 7, 4])):
                keep[k] = False
        col_fcy = col_fcy[keep]
        cvs_to_fcy = fc_cv[col_fcy - 1]
        # remove wall faces
        wall_keep = ~wall_mask[col_fcy - 1]
        col_fcy = col_fcy[wall_keep]
        cvs_to_fcy = cvs_to_fcy[wall_keep]

        cvs = np.unique(cvs_to_fcy.ravel())
        if cvs.size == 0:
            continue

        for is_ in range(ns):
            Sna[r_index, is_] = sna_src[cvs, is_].sum() if sna_src is not None else 0.0
            div_Fna_mdf[r_index, is_] = div_fna_mdf[cvs, is_].sum()
        for ia in range(nsorts):
            Snas[r_index, ia] = snas[cvs, ia].sum() if snas.ndim == 2 and snas.shape[1] > ia else 0.0
            Qrads[r_index, ia] = qrad[cvs, ia].sum() if qrad.ndim == 2 and qrad.shape[1] > ia else 0.0
        ftVol[r_index] = cv_vol[cvs].sum()
        if ftVol[r_index] > 0:
            te_mean[r_index] = (te[cvs] * cv_vol[cvs]).sum() / ftVol[r_index]

        r_index += 1

    out = {
        "Sna": Sna[:r_index], "div_Fna_mdf": div_Fna_mdf[:r_index],
        "Snas": Snas[:r_index], "Qrads": Qrads[:r_index],
        "ftVol": ftVol[:r_index], "te_mean": te_mean[:r_index],
    }
    watch._column_sums = out
    return out


@quantity(
    name="Sna",
    requires=[],
    description="radial-column integrated neutral source per species",
    unit="s⁻¹",
)
def calc_Sna(watch=None, grid=None, comp=None, **kw):
    return _column_sums(watch, grid, comp).get("Sna", np.zeros(0))


@quantity(
    name="Snas",
    requires=[],
    description="radial-column integrated neutral source per element",
    unit="s⁻¹",
)
def calc_Snas(watch=None, grid=None, comp=None, **kw):
    return _column_sums(watch, grid, comp).get("Snas", np.zeros(0))


@quantity(
    name="Qrads",
    requires=[],
    description="radial-column integrated radiation per element",
    unit="W",
)
def calc_Qrads(watch=None, grid=None, comp=None, **kw):
    return _column_sums(watch, grid, comp).get("Qrads", np.zeros(0))


@quantity(
    name="te_mean",
    requires=[],
    description="volume-averaged Te along radial column",
    unit="eV",
)
def calc_te_mean(watch=None, grid=None, comp=None, **kw):
    return _column_sums(watch, grid, comp).get("te_mean", np.zeros(0))


# ──────────────────────────────────────────────────────────────
# fh_htpl / fhp / fh_pls (lines 927-945)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fh_htpl_th",
    requires=[],
    description="heat flux to plates, poloidal",
    unit="W",
    location="face",
)
def calc_fh_htpl_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    zs = _zs(comp)
    ams = None
    if comp is not None:
        am = getattr(comp, "am", None)
        if am is not None:
            ams = np.asarray(am, dtype=np.float64)
    fne_th = ws.get("fne_th")
    if fne_th is None:
        fne_th = calc_fne_th(watch=watch, grid=grid, comp=comp)
    out = ws["fhe_th"] + ws["fhi_th"] + fne_th * tef * QE
    if zs is not None and ams is not None:
        for is_ in range(len(zs)):
            if zs[is_] == 0:
                continue
            out = out + ws["fna_th"][:, is_] * (tif * QE + ams[is_] * MP * uaf[:, is_] ** 2 / 2)
    return out


@quantity(
    name="fh_htpl_r",
    requires=[],
    description="heat flux to plates, radial",
    unit="W",
    location="face",
)
def calc_fh_htpl_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    zs = _zs(comp)
    ams = None
    if comp is not None:
        am = getattr(comp, "am", None)
        if am is not None:
            ams = np.asarray(am, dtype=np.float64)
    fne_r = ws.get("fne_r")
    if fne_r is None:
        fne_r = calc_fne_r(watch=watch, grid=grid, comp=comp)
    out = ws["fhe_r"] + ws["fhi_r"] + fne_r * tef * QE
    if zs is not None and ams is not None:
        for is_ in range(len(zs)):
            if zs[is_] == 0:
                continue
            out = out + ws["fna_r"][:, is_] * (tif * QE + ams[is_] * MP * uaf[:, is_] ** 2 / 2)
    return out


def _fhp(watch, grid, comp, direction: str) -> np.ndarray:
    """Ionization-potential energy flux (calc_fhp simple method)."""
    ws = _ws(watch)
    if comp is None:
        return np.zeros(grid.n_faces)
    from solps_analysis.construct.builtin.energy_balance import _E_pot_ion
    rpt = _E_pot_ion(watch, comp)  # (ns,) cumulative potentials
    # rpt_face = intface(rpt per species, halfsum)
    ns = len(rpt)
    rpt_face = np.zeros((grid.n_faces, ns))
    for is_ in range(ns):
        rpt_face[:, is_] = intface(grid, np.full(grid.n_cells, rpt[is_]), 1, "halfsum")
    fna = ws["fna_r"] if direction == "r" else ws["fna_th"]
    return fna * rpt_face * QE


@quantity(
    name="fhp_th",
    requires=[],
    description="ionization-potential energy flux, poloidal",
    unit="W",
    location="face",
)
def calc_fhp_th(watch=None, grid=None, comp=None, **kw):
    return _fhp(watch, grid, comp, "th")


@quantity(
    name="fhp_r",
    requires=[],
    description="ionization-potential energy flux, radial",
    unit="W",
    location="face",
)
def calc_fhp_r(watch=None, grid=None, comp=None, **kw):
    return _fhp(watch, grid, comp, "r")


@quantity(
    name="fh_pls_th",
    requires=[],
    description="total heat flux to plasma-facing surface, poloidal",
    unit="W",
    location="face",
)
def calc_fh_pls_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    htpl = ws.get("fh_htpl_th")
    if htpl is None:
        htpl = calc_fh_htpl_th(watch=watch, grid=grid, comp=comp)
    fhp = ws.get("fhp_th")
    if fhp is None:
        fhp = calc_fhp_th(watch=watch, grid=grid, comp=comp)
    fh_nutpr = ws.get("fh_nutpr_th", np.zeros(grid.n_faces))
    return htpl + fh_nutpr + fhp.sum(axis=1) if fhp.ndim == 2 else htpl + fh_nutpr + fhp


@quantity(
    name="fh_pls_r",
    requires=[],
    description="total heat flux to plasma-facing surface, radial",
    unit="W",
    location="face",
)
def calc_fh_pls_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    htpl = ws.get("fh_htpl_r")
    if htpl is None:
        htpl = calc_fh_htpl_r(watch=watch, grid=grid, comp=comp)
    fhp = ws.get("fhp_r")
    if fhp is None:
        fhp = calc_fhp_r(watch=watch, grid=grid, comp=comp)
    fh_nutpr = ws.get("fh_nutpr_r", np.zeros(grid.n_faces))
    return htpl + fh_nutpr + fhp.sum(axis=1) if fhp.ndim == 2 else htpl + fh_nutpr + fhp


# ──────────────────────────────────────────────────────────────
# Wall heat loads, saturation currents, floating potential
# (lines 947-1113)
# ──────────────────────────────────────────────────────────────

ME = 9.1093837015e-31  # electron mass, kg


def _wall_quantities(watch, grid, comp) -> dict:
    """Compute wall heat/current loads and saturation currents.

    Port of calc_additional.m lines 947-1113 (sputtering skipped —
    compute_sputtering defaults to false).
    """
    cached = getattr(watch, "_wall_quantities", None)
    if cached is not None:
        return cached

    ws = _ws(watch)
    n_fc = grid.n_faces
    from solps_analysis.core.operators import diff_p_us, diff_r_us

    # fhi_th/r fallback (lines 963-968)
    fhi_th = ws["fhi_th"]
    fhi_r = ws["fhi_r"]
    if np.abs(fhi_th).sum() == 0:
        fhi_th = ws["fhi_mdf_th"] - ws["fhi_dia_mdf_th"] + ws.get("fhi_nuBgradB_th", 0)
    if np.abs(fhi_r).sum() == 0:
        fhi_r = ws["fhi_mdf_r"] - ws["fhi_dia_mdf_r"] + ws.get("fhi_nuBgradB_r", 0)

    # heat flux components (lines 971-977)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    zs = _zs(comp)
    ions = np.where(zs is not None and np.asarray(zs) != 0)[0] if zs is not None else np.arange(ws["na"].shape[1])
    fna53_th = ws["fna_nuExB_th"] + ws["fna_nupar_th"]
    fna53_r = ws["fna_nuExB_r"]
    fne_curr_th = ws.get("fne_curr_th")
    if fne_curr_th is None:
        fne_curr_th = calc_fne_curr_th(watch=watch, grid=grid, comp=comp)
    fne_curr_r = ws.get("fne_curr_r")
    if fne_curr_r is None:
        fne_curr_r = calc_fne_curr_r(watch=watch, grid=grid, comp=comp)
    fne53_th = (ws["fna_nuExB_th"] * zs[None, :]).sum(axis=1) \
        - (ws.get("fna_curr_th", np.zeros_like(fna53_th)) * zs[None, :]).sum(axis=1) \
        + (ws["fna_nupar_th"] * zs[None, :]).sum(axis=1) - ws["fch_par_th"] / QE \
        if zs is not None else np.zeros(n_fc)
    fne53_r = (ws["fna_nuExB_r"] * zs[None, :]).sum(axis=1) \
        - (ws.get("fna_curr_r", np.zeros_like(fna53_r)) * zs[None, :]).sum(axis=1) \
        if zs is not None else np.zeros(n_fc)

    fh_heat_th = fhi_th + ws["fhe_th"] + fna53_th[:, ions].sum(axis=1) * tif * QE \
        + fne53_th * tef * QE
    fh_heat_r = fhi_r + ws["fhe_r"] + fna53_r[:, ions].sum(axis=1) * tif * QE \
        + fne53_r * tef * QE

    # kinetic & viscous heat flux (lines 982-1005)
    fh_kinrgy_th = np.zeros(n_fc)
    fh_kinrgy_r = np.zeros(n_fc)
    fh_vis_th = np.zeros(n_fc)
    fh_vis_r = np.zeros(n_fc)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fc_cv = grid.fc_cv
    cetaa_cl_th = ws.get("cetaa_clLucFlim_th")
    cetaa_AN_th = ws.get("cetaa_AN_th")
    cetaa_AN_r = ws.get("cetaa_AN_r")
    if cetaa_cl_th is None:
        cetaa_cl_th = np.zeros(n_fc)
    if cetaa_AN_th is None:
        cetaa_AN_th = np.zeros((n_fc, ws["na"].shape[1]))
    if cetaa_AN_r is None:
        cetaa_AN_r = np.zeros((n_fc, ws["na"].shape[1]))
    ams = None
    if comp is not None:
        am = getattr(comp, "am", None)
        if am is not None:
            ams = np.asarray(am, dtype=np.float64)

    if ams is not None and zs is not None:
        for is_ in range(len(zs)):
            if zs[is_] == 0:
                continue
            ua2 = ws["ua"][:, is_] ** 2 / 2
            # ua2m = 0.5*sum(ua2 over fcCv) — average of the two cells
            ua2m = 0.5 * (ua2[fc_cv[:, 0]] + ua2[fc_cv[:, 1]])
            diff_ua2_th = diff_p_us(grid, 0, ua2)
            diff_ua2_r = diff_r_us(grid, 0, ua2)
            fh_kinrgy_th = fh_kinrgy_th + ws["fna_th"][:, is_] * ua2m * MP * ams[is_]
            fh_kinrgy_r = fh_kinrgy_r + ws["fna_r"][:, is_] * ua2m * MP * ams[is_]
            fh_vis_th = fh_vis_th - (4.0/3.0 * cetaa_cl_th + cetaa_AN_th[:, is_]) * diff_ua2_th
            fh_vis_r = fh_vis_r - (4.0/3.0 * 0 + cetaa_AN_r[:, is_]) * diff_ua2_r

    # sum of components (lines 1006-1015)
    fhp_sum_th = ws["fhp_th"][:, ions].sum(axis=1) if "fhp_th" in ws else np.zeros(n_fc)
    fhp_sum_r = ws["fhp_r"][:, ions].sum(axis=1) if "fhp_r" in ws else np.zeros(n_fc)
    fh_joule_th = ws.get("fh_joule_th")
    fh_joule_r = ws.get("fh_joule_r")
    if fh_joule_th is None or fh_joule_r is None:
        fh_joule_th = calc_fh_joule_th(watch=watch, grid=grid)
        fh_joule_r = calc_fh_joule_r(watch=watch, grid=grid)
    fh_nutpr_th = ws.get("fh_nutpr_th", np.zeros(n_fc))
    fh_neut_tot_th = ws.get("fh_neut_tot_th", np.zeros(n_fc))
    fh_rad_th = ws.get("fh_rad_th", np.zeros(n_fc))

    fh_sum_th = fh_heat_th + fh_kinrgy_th + fh_vis_th + fhp_sum_th + fh_joule_th \
        + fh_nutpr_th + fh_neut_tot_th + fh_rad_th
    fh_sum_r = fh_heat_r + fh_kinrgy_r + fh_vis_r + fhp_sum_r + fh_joule_r

    # boundary loads (lines 1017-1043)
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(n_fc)
    fc_or = grid.fc_or
    fcs_boundary = _fcs_boundary(grid)
    n_b = len(fcs_boundary)

    fh_boundary = np.zeros(n_fc)
    fch_boundary = np.zeros(n_fc)
    fna_wall = np.zeros((n_fc, ws["na"].shape[1]))
    jsat_perp = np.zeros(n_fc)
    jsat_perp_j = np.zeros(n_fc)
    jsat_par_exp = np.zeros(n_fc)
    jsat_par_exp_j = np.zeros(n_fc)
    jsat_par_phys = np.zeros(n_fc)
    po_fl_wall = np.zeros(grid.n_cells)

    ns = ws["na"].shape[1]
    for k in range(n_b):
        fcs = fcs_boundary[k]
        cvs = _cvs_boundary(grid)[k][1]
        if fcs.size == 0:
            continue
        fh_boundary[fcs] = (fh_sum_th[fcs] + fh_sum_r[fcs]) / fc_s[fcs] * fc_or[fcs]
        fch_boundary[fcs] = (ws["fch_th"][fcs] + ws["fch_r"][fcs]) / fc_s[fcs] * fc_or[fcs]
        fna_wall[fcs, :] = (ws["fna_mdf_r"][fcs, :] + ws["fna_mdf_th"][fcs, :]) / fc_s[fcs, None] * fc_or[fcs, None]

        for is_ in range(ns):
            jsat_perp[fcs] = jsat_perp[fcs] + fc_or[fcs] * (
                ws["fna_mdf_th"][fcs, is_] + ws["fna_mdf_r"][fcs, is_]
                - ws["fna_dia_mdf_th"][fcs, is_] - ws["fna_dia_mdf_r"][fcs, is_]
            ) * zs[is_] / fc_s[fcs] * QE
            jsat_par_phys[fcs] = jsat_par_phys[fcs] + (ws["na"][cvs, is_] * ws["ua"][cvs, is_]
                                                        * zs[is_] * QE).sum(axis=0) * np.sign(jsat_perp[fcs])
        fc_qalf = grid.fc_qalf if grid.fc_qalf is not None else np.zeros((n_fc, 2))
        fc_bb = grid.fc_bb if grid.fc_bb is not None else np.ones((n_fc, 4))
        jsat_par_phys[fcs] = jsat_par_phys[fcs] * fc_or[fcs] * np.sign(fc_qalf[fcs, 0])

        if fc_bb[fcs[0], 0] != 0 and fc_qalf[fcs[0], 0] != 0:
            jsat_cos = np.abs(fc_qalf[fcs, 0] * fc_bb[fcs, 0] / fc_bb[fcs, 3])
            qalfmin = _b2mn_param(watch, "b2stbc_Qalfmin", 0.0)
            if qalfmin > 0:
                jsat_cos[jsat_cos < qalfmin] = qalfmin
            jsat_par_exp[fcs] = jsat_perp[fcs] / jsat_cos
            jsat_perp_j[fcs] = (fch_boundary[fcs]
                                + np.sqrt((ws["te"][cvs] * QE) / (2 * np.pi * ME))
                                * np.exp(-ws["po"][cvs] / ws["te"][cvs])
                                * ws["ne"][cvs] * QE * jsat_cos).sum(axis=0) if cvs.size else 0
            jsat_par_exp_j[fcs] = jsat_perp_j[fcs] / jsat_cos

        # floating potential (lines 1048-1051)
        if cvs.size:
            Gamma_e_par = ws["ne"][cvs] * np.sqrt((ws["te"][cvs] * QE) / (2 * np.pi * ME))
            Gamma_i_par = jsat_par_exp[fcs] / QE
            # per-face: use first cell of the boundary (MATLAB uses cvs vector)
            for j, fc in enumerate(fcs):
                cv = cvs[j % len(cvs)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    po_fl_wall[cv] = ws["po"][cv] - ws["te"][cv] * np.log(
                        Gamma_e_par[j % len(cvs)] / Gamma_i_par[j])

    out = {
        "fh_heat_th": fh_heat_th, "fh_heat_r": fh_heat_r,
        "fh_kinrgy_th": fh_kinrgy_th, "fh_kinrgy_r": fh_kinrgy_r,
        "fh_vis_th": fh_vis_th, "fh_vis_r": fh_vis_r,
        "fh_sum_th": fh_sum_th, "fh_sum_r": fh_sum_r, "fh_sum": fh_sum_th + fh_sum_r,
        "fh_boundary": fh_boundary, "fch_boundary": fch_boundary,
        "fna_wall": fna_wall,
        "jsat_perp": jsat_perp, "jsat_perp_j": jsat_perp_j,
        "jsat_par_exp": jsat_par_exp, "jsat_par_exp_j": jsat_par_exp_j,
        "jsat_par_phys": jsat_par_phys, "po_fl_wall": po_fl_wall,
    }
    watch._wall_quantities = out
    return out


def _fcs_boundary(grid) -> list:
    """Boundary faces per label (face counterparts of cvs_boundary)."""
    cached = getattr(grid, "_fcs_boundary_cache", None)
    if cached is not None:
        return cached
    out = []
    if grid.fc_lbl is not None and grid.fc_cv is not None:
        n_ci = grid.n_core_cells
        for lbl in np.unique(grid.fc_lbl[grid.fc_lbl != 0]):
            mask = grid.fc_lbl == lbl
            fcs = np.where(mask)[0]
            # keep only faces adjacent to guard cells (boundary faces)
            guard = (grid.fc_cv[fcs] >= n_ci).any(axis=1)
            out.append(fcs[guard])
    grid._fcs_boundary_cache = out
    return out


def _wall_get(watch, grid, comp, name: str) -> np.ndarray:
    return _wall_quantities(watch, grid, comp).get(name, np.zeros(0))


@quantity(
    name="fh_boundary",
    requires=[],
    description="total heat flux to wall boundary",
    unit="W/m²",
    location="face",
)
def calc_fh_boundary(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "fh_boundary")


@quantity(
    name="fch_boundary",
    requires=[],
    description="current density to wall boundary",
    unit="A/m²",
    location="face",
)
def calc_fch_boundary(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "fch_boundary")


@quantity(
    name="jsat_perp",
    requires=[],
    description="perpendicular saturation current density",
    unit="A/m²",
    location="face",
)
def calc_jsat_perp(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "jsat_perp")


@quantity(
    name="jsat_par_exp",
    requires=[],
    description="parallel saturation current (experimental)",
    unit="A/m²",
    location="face",
)
def calc_jsat_par_exp(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "jsat_par_exp")


@quantity(
    name="po_fl_wall",
    requires=[],
    description="floating potential at boundary cells",
    unit="V",
)
def calc_po_fl_wall(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "po_fl_wall")


@quantity(
    name="fh_sum",
    requires=[],
    description="total heat flux to wall (sum of components)",
    unit="W/m²",
    location="face",
)
def calc_fh_sum(watch=None, grid=None, comp=None, **kw):
    return _wall_get(watch, grid, comp, "fh_sum")


# ──────────────────────────────────────────────────────────────
# Species totals, separatrix/pedestal quantities, compression,
# enrichment (lines 1117-1341)
# ──────────────────────────────────────────────────────────────

def _sep_quantities(watch, grid, comp) -> dict:
    """Separatrix/pedestal parameters + compression/enrichment.

    Port of calc_additional.m lines 1117-1341.
    """
    cached = getattr(watch, "_sep_quantities", None)
    if cached is not None:
        return cached

    ws = _ws(watch)
    watch.compute_regions()
    ns = ws["na"].shape[1]
    nsorts = comp.n_elements if comp is not None else 1
    indices = comp.element_indices_list if comp is not None else [list(range(ns))]
    zs = _zs(comp)
    ams = None
    if comp is not None:
        am = getattr(comp, "am", None)
        if am is not None:
            ams = np.asarray(am, dtype=np.float64)
    n_ci = grid.n_core_cells
    cv_vol = grid.cv_vol
    fc_s = grid.fc_s if grid.fc_s is not None else np.ones(grid.n_faces)
    fc_hc = grid.fc_hc if grid.fc_hc is not None else np.ones((grid.n_faces, 2))
    fc_cv = grid.fc_cv
    cv_reg = grid.cv_reg

    # ni / rho / p_ch / cs / csi (lines 1118-1140)
    ni = np.zeros((grid.n_cells, 2))
    rho = np.zeros(grid.n_cells)
    for is_ in range(ns):
        if zs is not None and zs[is_] != 0:
            ni[:, 1] += ws["na"][:, is_]
            if ams is not None:
                rho += ws["na"][:, is_] * ams[is_] * MP
        else:
            ni[:, 0] += ws["na"][:, is_]
    ni[:, 0] += ni[:, 1]
    p_ch = ws["ne"] * ws["te"] * QE + ni[:, 1] * ws["ti"] * QE
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.sqrt(p_ch / np.maximum(rho, 1e-30))
        csi = np.sqrt(ni[:, 1] * ws["ti"] * QE / np.maximum(rho, 1e-30))
    cs = np.nan_to_num(cs, nan=0.0, posinf=0.0)
    csi = np.nan_to_num(csi, nan=0.0, posinf=0.0)

    # Zavg per element (lines 1142-1150)
    Zavg = np.zeros((grid.n_cells, nsorts))
    for ia in range(nsorts):
        tmp = indices[ia]
        num = np.zeros(grid.n_cells)
        den = np.zeros(grid.n_cells)
        for is_ in tmp[1:]:
            num += zs[is_] * ws["na"][:, is_]
            den += ws["na"][:, is_]
        with np.errstate(divide="ignore", invalid="ignore"):
            Zavg[:, ia] = num / np.maximum(den, 1e-30)
    Zavg = np.nan_to_num(Zavg, nan=0.0)

    sep_fcs = grid.core_sep_fcs
    if sep_fcs is None or len(sep_fcs) == 0:
        watch._sep_quantities = {}
        return {}

    # te_sep / ti_sep (lines 1189-1198): hc-weighted interpolation × fcS
    te_sep = 0.0
    ti_sep = 0.0
    for i_fc in sep_fcs:
        i_cvs = fc_cv[i_fc]
        hc1, hc2 = fc_hc[i_fc]
        w = hc1 + hc2
        te_sep += (ws["te"][i_cvs[0]] * hc2 + ws["te"][i_cvs[1]] * hc1) / w * fc_s[i_fc]
        ti_sep += (ws["ti"][i_cvs[0]] * hc2 + ws["ti"][i_cvs[1]] * hc1) / w * fc_s[i_fc]
    sum_fcS = fc_s[sep_fcs].sum()
    te_sep /= sum_fcS
    ti_sep /= sum_fcS

    # ped_fcs: core boundary faces (fcLbl == -21 | -25)
    ped_fcs = np.where((grid.fc_lbl == -21) | (grid.fc_lbl == -25))[0] if grid.fc_lbl is not None else np.array([], dtype=int)

    # particle inventory (lines 1200-1232)
    N_tot_B25 = np.zeros(nsorts)
    N_sorts_by_Reg = np.zeros((cv_reg.max(), nsorts))
    volReg = np.zeros(cv_reg.max())
    N_core = np.zeros(nsorts)
    N_in_div = np.zeros(nsorts)
    N_out_div = np.zeros(nsorts)
    pa = np.zeros((grid.n_cells, nsorts))
    uaAv = np.zeros((grid.n_cells, nsorts))
    n_in_div_avr = np.zeros(nsorts)
    n_out_div_avr = np.zeros(nsorts)

    for ia in range(nsorts):
        is_list = indices[ia]
        N_tot_B25[ia] = (ws["na"][:n_ci, is_list].sum(axis=1) * cv_vol[:n_ci]).sum()
        for i_reg in range(1, cv_reg.max() + 1):
            reg_cvs = np.where(cv_reg == i_reg)[0]
            reg_cvs = reg_cvs[reg_cvs < n_ci]
            if reg_cvs.size == 0:
                continue
            volReg[i_reg - 1] = cv_vol[reg_cvs].sum()
            N_sorts_by_Reg[i_reg - 1, ia] = (ws["na"][np.ix_(reg_cvs, is_list)].sum(axis=1) * cv_vol[reg_cvs]).sum()

        pa[:, ia] = (ws["na"][:, is_list] * zs[is_list][None, :] * ws["te"][:, None]
                     + ws["na"][:, is_list[1:]].sum(axis=1)[:, None] * ws["ti"][:, None]).sum(axis=1) * QE \
            if len(is_list) > 1 else ws["na"][:, is_list].sum(axis=1) * zs[is_list[0]] * ws["te"] * QE
        naua = ws["na"] * ws["ua"]
        uaAv[:, ia] = (naua[:, is_list[1:]].sum(axis=1)
                       / np.maximum(ws["na"][:, is_list[1:]].sum(axis=1), 1e-30)) \
            if len(is_list) > 1 else np.zeros(grid.n_cells)
        N_core[ia] = N_sorts_by_Reg[np.mod(np.arange(1, cv_reg.max() + 1), 4) == 1, ia].sum()

        in_str_cvs = grid.cv_inner_tar
        out_str_cvs = grid.cv_outer_tar
        if in_str_cvs is not None and len(in_str_cvs):
            in_reg = np.unique(cv_reg[in_str_cvs])[0]
            N_in_div[ia] = N_sorts_by_Reg[in_reg - 1, ia] if in_reg - 1 < N_sorts_by_Reg.shape[0] else 0.0
            n_in_div_avr[ia] = N_in_div[ia] / max(volReg[in_reg - 1], 1e-30)
        if out_str_cvs is not None and len(out_str_cvs):
            out_reg = np.unique(cv_reg[out_str_cvs])[0]
            N_out_div[ia] = N_sorts_by_Reg[out_reg - 1, ia] if out_reg - 1 < N_sorts_by_Reg.shape[0] else 0.0
            n_out_div_avr[ia] = N_out_div[ia] / max(volReg[out_reg - 1], 1e-30)

    # separatrix densities (lines 1234-1255)
    n_sep = np.zeros(nsorts)
    n_sep_Kuk = np.zeros(nsorts)
    ne_sep = 0.0
    ne_sep_Kuk = 0.0
    for i_fc in sep_fcs:
        i_cvs = fc_cv[i_fc]
        hc1, hc2 = fc_hc[i_fc]
        w = hc1 + hc2
        lcfs_cv = i_cvs[np.mod(cv_reg[i_cvs], 4) != 1]
        for ia in range(nsorts):
            is_list = np.asarray(indices[ia], dtype=np.intp)
            na_tmp = ws["na"][np.ix_(i_cvs, is_list)].sum(axis=1)
            na_int = (na_tmp[0] * hc2 + na_tmp[1] * hc1) / w
            n_sep[ia] += na_int * fc_s[i_fc]
            if lcfs_cv.size:
                n_sep_Kuk[ia] += ws["na"][np.ix_(lcfs_cv, is_list)].sum() * fc_s[i_fc]
        ne_int = (ws["ne"][i_cvs[0]] * hc2 + ws["ne"][i_cvs[1]] * hc1) / w
        ne_sep += ne_int * fc_s[i_fc]
        ne_sep_Kuk += ws["ne"][lcfs_cv].sum() * fc_s[i_fc] if lcfs_cv.size else 0.0
    n_sep /= sum_fcS
    n_sep_Kuk /= sum_fcS
    ne_sep /= sum_fcS
    ne_sep_Kuk /= sum_fcS

    # pedestal densities (lines 1257-1282)
    n_ped = np.zeros(nsorts)
    n_ped_Kuk = np.zeros(nsorts)
    ne_ped = 0.0
    ne_ped_Kuk = 0.0
    p_ped_Kuk = np.zeros(nsorts)
    omp = grid.outer_midplane_cells
    cei_ft = grid.cv_ft[omp[1]] if (omp is not None and len(omp) > 1 and grid.cv_ft is not None) else None
    cei_cvs = np.array([], dtype=int)
    if cei_ft is not None and grid.ft_cv is not None and grid.ft_cv_p is not None:
        i_ft = int(cei_ft)
        if 0 <= i_ft < grid.ft_cv_p.shape[0]:
            cei_cvs = grid.ft_cv[grid.ft_cv_p[i_ft, 0]:grid.ft_cv_p[i_ft, 0] + grid.ft_cv_p[i_ft, 1]]

    sum_ped_fcS = fc_s[ped_fcs].sum() if len(ped_fcs) else 1.0
    for k, i_fc in enumerate(ped_fcs):
        i_cvs = fc_cv[i_fc]
        hc1, hc2 = fc_hc[i_fc]
        w = hc1 + hc2
        for ia in range(nsorts):
            is_list = np.asarray(indices[ia], dtype=np.intp)
            na_tmp = ws["na"][np.ix_(i_cvs, is_list)].sum(axis=1)
            na_int = (na_tmp[0] * hc2 + na_tmp[1] * hc1) / w
            n_ped[ia] += na_int * fc_s[i_fc]
        ne_int = (ws["ne"][i_cvs[0]] * hc2 + ws["ne"][i_cvs[1]] * hc1) / w
        ne_ped += ne_int * fc_s[i_fc]
    n_ped /= sum_ped_fcS
    ne_ped /= sum_ped_fcS

    if len(cei_cvs):
        for ia in range(nsorts):
            is_list = np.asarray(indices[ia], dtype=np.intp)
            n_ped_Kuk[ia] = (ws["na"][np.ix_(cei_cvs, is_list)].sum(axis=1) * fc_s[ped_fcs]).sum() / sum_ped_fcS \
                if len(ped_fcs) else 0.0
            p_ped_Kuk[ia] = (pa[cei_cvs, ia] * fc_s[ped_fcs]).sum() / sum_ped_fcS \
                if len(ped_fcs) else 0.0
        ne_ped_Kuk = (ws["ne"][cei_cvs] * fc_s[ped_fcs]).sum() / sum_ped_fcS if len(ped_fcs) else 0.0

    # concentrations (lines 1285-1290)
    n_e_sep = n_sep / max(ne_sep, 1e-30)
    n_D_sep = n_sep / max(n_sep[0], 1e-30) if nsorts else n_sep
    n_e_ped = n_ped / max(ne_ped, 1e-30)
    n_D_ped = n_ped / max(n_ped[0], 1e-30) if nsorts else n_ped

    # compression / enrichment (lines 1292-1317)
    Compr_indiv_sep = n_in_div_avr / np.maximum(n_sep, 1e-30)
    Compr_outdiv_sep = n_out_div_avr / np.maximum(n_sep, 1e-30)
    Compr_indiv_ped = n_in_div_avr / np.maximum(n_ped, 1e-30)
    Compr_outdiv_ped = n_out_div_avr / np.maximum(n_ped, 1e-30)
    in_reg0 = np.unique(cv_reg[grid.cv_inner_tar])[0] if grid.cv_inner_tar is not None and len(grid.cv_inner_tar) else 1
    out_reg0 = np.unique(cv_reg[grid.cv_outer_tar])[0] if grid.cv_outer_tar is not None and len(grid.cv_outer_tar) else 2
    v_div = volReg[in_reg0 - 1] + volReg[out_reg0 - 1] if in_reg0 <= volReg.size and out_reg0 <= volReg.size else 1.0
    Compr_div_sep = (N_in_div + N_out_div) / np.maximum(n_sep, 1e-30) / max(v_div, 1e-30)
    Compr_div_ped = (N_in_div + N_out_div) / np.maximum(n_ped, 1e-30) / max(v_div, 1e-30)
    Compr_indiv_sep_Kuk = n_in_div_avr / np.maximum(n_sep_Kuk, 1e-30)
    Compr_outdiv_sep_Kuk = n_out_div_avr / np.maximum(n_sep_Kuk, 1e-30)
    Compr_indiv_ped_Kuk = n_in_div_avr / np.maximum(n_ped_Kuk, 1e-30)
    Compr_outdiv_ped_Kuk = n_out_div_avr / np.maximum(n_ped_Kuk, 1e-30)
    Compr_div_sep_Kuk = (N_in_div + N_out_div) / np.maximum(n_sep_Kuk, 1e-30) / max(v_div, 1e-30)
    Compr_div_ped_Kuk = (N_in_div + N_out_div) / np.maximum(n_ped_Kuk, 1e-30) / max(v_div, 1e-30)

    Enrich_indiv_sep = Compr_indiv_sep / max(Compr_indiv_sep[0], 1e-30)
    Enrich_outdiv_sep = Compr_outdiv_sep / max(Compr_outdiv_sep[0], 1e-30)
    Enrich_div_sep = Compr_div_sep / max(Compr_div_sep[0], 1e-30)

    out = {
        "ni": ni, "rho": rho, "p_ch": p_ch, "cs": cs, "csi": csi, "Zavg": Zavg,
        "pa": pa, "uaAv": uaAv,
        "te_sep": te_sep, "ti_sep": ti_sep,
        "n_sep": n_sep, "n_sep_Kuk": n_sep_Kuk, "ne_sep": ne_sep, "ne_sep_Kuk": ne_sep_Kuk,
        "n_ped": n_ped, "n_ped_Kuk": n_ped_Kuk, "ne_ped": ne_ped, "ne_ped_Kuk": ne_ped_Kuk,
        "p_ped_Kuk": p_ped_Kuk,
        "n_e_sep": n_e_sep, "n_D_sep": n_D_sep, "n_e_ped": n_e_ped, "n_D_ped": n_D_ped,
        "N_tot_B25": N_tot_B25, "N_sorts_by_Reg": N_sorts_by_Reg, "volReg": volReg,
        "N_core": N_core, "N_in_div": N_in_div, "N_out_div": N_out_div,
        "n_in_div_avr": n_in_div_avr, "n_out_div_avr": n_out_div_avr,
        "Compr_indiv_sep": Compr_indiv_sep, "Compr_outdiv_sep": Compr_outdiv_sep,
        "Compr_indiv_ped": Compr_indiv_ped, "Compr_outdiv_ped": Compr_outdiv_ped,
        "Compr_div_sep": Compr_div_sep, "Compr_div_ped": Compr_div_ped,
        "Compr_indiv_sep_Kuk": Compr_indiv_sep_Kuk, "Compr_outdiv_sep_Kuk": Compr_outdiv_sep_Kuk,
        "Compr_indiv_ped_Kuk": Compr_indiv_ped_Kuk, "Compr_outdiv_ped_Kuk": Compr_outdiv_ped_Kuk,
        "Compr_div_sep_Kuk": Compr_div_sep_Kuk, "Compr_div_ped_Kuk": Compr_div_ped_Kuk,
        "Enrich_indiv_sep": Enrich_indiv_sep, "Enrich_outdiv_sep": Enrich_outdiv_sep,
        "Enrich_div_sep": Enrich_div_sep,
    }
    watch._sep_quantities = out
    return out


def _sep_get(watch, grid, comp, name: str) -> np.ndarray:
    return _sep_quantities(watch, grid, comp).get(name, np.zeros(0))


@quantity(
    name="ni",
    requires=[],
    description="ion+atom / ion density (2 columns)",
    unit="m⁻³",
)
def calc_ni(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "ni")


@quantity(
    name="rho",
    requires=[],
    description="mass density of charged particles",
    unit="kg/m³",
)
def calc_rho(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "rho")


@quantity(
    name="p_ch",
    requires=[],
    description="pressure of charged particles",
    unit="Pa",
)
def calc_p_ch(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "p_ch")


@quantity(
    name="cs",
    requires=[],
    description="sound speed",
    unit="m/s",
)
def calc_cs(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "cs")


@quantity(
    name="Zavg",
    requires=[],
    description="average charge per element",
    unit="",
)
def calc_Zavg(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "Zavg")


@quantity(
    name="te_sep",
    requires=[],
    description="separatrix electron temperature (surface averaged)",
    unit="eV",
)
def calc_te_sep(watch=None, grid=None, comp=None, **kw):
    return np.array([_sep_get(watch, grid, comp, "te_sep")])


@quantity(
    name="ti_sep",
    requires=[],
    description="separatrix ion temperature (surface averaged)",
    unit="eV",
)
def calc_ti_sep(watch=None, grid=None, comp=None, **kw):
    return np.array([_sep_get(watch, grid, comp, "ti_sep")])


@quantity(
    name="n_sep",
    requires=[],
    description="separatrix density per element",
    unit="m⁻³",
)
def calc_n_sep(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "n_sep")


@quantity(
    name="n_e_sep",
    requires=[],
    description="separatrix concentration per element (rel. to ne)",
    unit="",
)
def calc_n_e_sep(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "n_e_sep")


@quantity(
    name="n_D_sep",
    requires=[],
    description="separatrix concentration per element (rel. to main ion)",
    unit="",
)
def calc_n_D_sep(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "n_D_sep")


@quantity(
    name="n_ped",
    requires=[],
    description="pedestal density per element",
    unit="m⁻³",
)
def calc_n_ped(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "n_ped")


@quantity(
    name="Compr_div_sep",
    requires=[],
    description="divertor compression at separatrix",
    unit="",
)
def calc_Compr_div_sep(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "Compr_div_sep")


@quantity(
    name="Compr_div_ped",
    requires=[],
    description="divertor compression at pedestal",
    unit="",
)
def calc_Compr_div_ped(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "Compr_div_ped")


@quantity(
    name="Enrich_div_sep",
    requires=[],
    description="divertor enrichment at separatrix",
    unit="",
)
def calc_Enrich_div_sep(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "Enrich_div_sep")


@quantity(
    name="N_tot_B25",
    requires=[],
    description="total nuclei per element in core",
    unit="",
)
def calc_N_tot_B25(watch=None, grid=None, comp=None, **kw):
    return _sep_get(watch, grid, comp, "N_tot_B25")


# ──────────────────────────────────────────────────────────────
# Gas puff / fueling / seeding (lines 1343-1410)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="Gas_puff",
    requires=[],
    description="gas puff rate per element",
    unit="s⁻¹",
)
def calc_Gas_puff(watch=None, grid=None, comp=None, **kw):
    if comp is None:
        return np.zeros(1)
    nsorts = comp.n_elements
    gas = np.zeros(nsorts)
    run_log = getattr(watch, "run_log", None)
    if run_log is not None and getattr(watch, "neut", None) is not None:
        gp = run_log.get("gas_puff", {})
        for ia in range(nsorts):
            gas[ia] = float(gp.get(ia, 0.0))
    return gas


@quantity(
    name="Fueling",
    requires=[],
    description="fueling rate (main ion gas puff)",
    unit="s⁻¹",
)
def calc_Fueling(watch=None, grid=None, comp=None, **kw):
    gas = calc_Gas_puff(watch=watch, grid=grid, comp=comp)
    if len(gas) and gas[0] > 0:
        return np.array([gas[0]])
    return np.array([0.0])


@quantity(
    name="Seeding",
    requires=[],
    description="seeding rate (radiating impurity gas puff)",
    unit="s⁻¹",
)
def calc_Seeding(watch=None, grid=None, comp=None, **kw):
    gas = calc_Gas_puff(watch=watch, grid=grid, comp=comp)
    if len(gas) and gas[-1] > 0:
        return np.array([gas[-1]])
    return np.array([0.0])


# ──────────────────────────────────────────────────────────────
# Region-integrated quantities (lines 1414-1429)
# ──────────────────────────────────────────────────────────────

def _region_integrals(watch, grid, comp) -> dict:
    cached = getattr(watch, "_region_integrals", None)
    if cached is not None:
        return cached
    ws = _ws(watch)
    ns = ws["na"].shape[1]
    n_reg = grid.cv_reg.max()
    Na_reg = np.zeros((8, ns))
    she_rad_reg = np.zeros(8)
    she_eir_reg = np.zeros(8)
    she_radbrm_reg = np.zeros((8, ns))
    she_radlin_reg = np.zeros((8, ns))
    cv_vol = grid.cv_vol
    cv_reg = grid.cv_reg
    for i_reg in range(1, min(8, n_reg) + 1):
        cvs = np.where(cv_reg == i_reg)[0]
        if cvs.size == 0:
            continue
        Na_reg[i_reg - 1, :] = (ws["na"][cvs] * cv_vol[cvs, None]).sum(axis=0)
        she_rad_reg[i_reg - 1] = ws["she_rad"][cvs].sum()
        she_eir_reg[i_reg - 1] = ws["she_eir"][cvs].sum() if "she_eir" in ws else 0.0
        she_radbrm_reg[i_reg - 1, :] = ws["she_radbrm"][cvs].sum(axis=0)
        she_radlin_reg[i_reg - 1, :] = ws["she_radlin"][cvs].sum(axis=0)
    out = {
        "Na_reg": Na_reg, "she_rad_reg": she_rad_reg, "she_eir_reg": she_eir_reg,
        "she_radbrm_reg": she_radbrm_reg, "she_radlin_reg": she_radlin_reg,
    }
    watch._region_integrals = out
    return out


@quantity(
    name="Na_reg",
    requires=[],
    description="particle inventory per region",
    unit="",
)
def calc_Na_reg(watch=None, grid=None, comp=None, **kw):
    return _region_integrals(watch, grid, comp).get("Na_reg", np.zeros(0))


@quantity(
    name="she_rad_reg",
    requires=[],
    description="radiation power per region",
    unit="W",
)
def calc_she_rad_reg(watch=None, grid=None, comp=None, **kw):
    return _region_integrals(watch, grid, comp).get("she_rad_reg", np.zeros(0))


# ──────────────────────────────────────────────────────────────
# Electric fields and gradients (lines 1431-1471)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="Hx1",
    requires=[],
    description="poloidal connector length per cell",
    unit="m",
)
def calc_Hx1(watch=None, grid=None, **kw):
    hx1 = np.zeros(grid.n_cells)
    if grid.fc_qalf is None or grid.fc_hc is None or grid.fc_cv is None:
        return hx1
    mask = np.abs(grid.fc_qalf[:, 0]) > 0.1
    fc_cv = grid.fc_cv[mask]
    fc_hc = grid.fc_hc[mask]
    np.add.at(hx1, fc_cv[:, 0], fc_hc[:, 0])
    np.add.at(hx1, fc_cv[:, 1], fc_hc[:, 1])
    return hx1


@quantity(
    name="E_r",
    requires=[],
    description="radial electric field",
    unit="V/m",
    location="face",
)
def calc_E_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_r_us
    funv = np.zeros(grid.n_vertices)
    bp_dir = getattr(grid, "bp_dir", 0) or -1
    return -grad_r_us(grid, 1, ws["po"], funv) * bp_dir


@quantity(
    name="E_th",
    requires=[],
    description="poloidal electric field",
    unit="V/m",
    location="face",
)
def calc_E_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_p_us
    funv = np.zeros(grid.n_vertices)
    bp_dir = getattr(grid, "bp_dir", 0) or -1
    return -grad_p_us(grid, 1, ws["po"], funv) * bp_dir


@quantity(
    name="gradPe_r",
    requires=[],
    description="radial electron pressure gradient",
    unit="Pa/m",
    location="face",
)
def calc_gradPe_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_r_us
    funv = np.zeros(grid.n_vertices)
    return grad_r_us(grid, 1, QE * ws["ne"] * ws["te"], funv)


@quantity(
    name="gradPe_th",
    requires=[],
    description="poloidal electron pressure gradient",
    unit="Pa/m",
    location="face",
)
def calc_gradPe_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_p_us
    funv = np.zeros(grid.n_vertices)
    return grad_p_us(grid, 1, QE * ws["ne"] * ws["te"], funv)


@quantity(
    name="gradPi_r",
    requires=[],
    description="radial ion pressure gradient per species",
    unit="Pa/m",
    location="face",
)
def calc_gradPi_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_r_us
    funv = np.zeros(grid.n_vertices)
    ns = ws["na"].shape[1]
    out = np.zeros((grid.n_faces, ns))
    for is_ in range(ns):
        out[:, is_] = grad_r_us(grid, 1, QE * ws["na"][:, is_] * ws["ti"], funv)
    return out


@quantity(
    name="gradPi_th",
    requires=[],
    description="poloidal ion pressure gradient per species",
    unit="Pa/m",
    location="face",
)
def calc_gradPi_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_p_us
    funv = np.zeros(grid.n_vertices)
    ns = ws["na"].shape[1]
    out = np.zeros((grid.n_faces, ns))
    for is_ in range(ns):
        out[:, is_] = grad_p_us(grid, 1, QE * ws["na"][:, is_] * ws["ti"], funv)
    return out


@quantity(
    name="gradTe_r",
    requires=[],
    description="radial electron temperature gradient",
    unit="eV/m",
    location="face",
)
def calc_gradTe_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_r_us
    funv = np.zeros(grid.n_vertices)
    return grad_r_us(grid, 1, ws["te"], funv)


@quantity(
    name="gradTi_r",
    requires=[],
    description="radial ion temperature gradient",
    unit="eV/m",
    location="face",
)
def calc_gradTi_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_r_us
    funv = np.zeros(grid.n_vertices)
    return grad_r_us(grid, 1, ws["ti"], funv)


@quantity(
    name="gradTe_th",
    requires=[],
    description="poloidal electron temperature gradient",
    unit="eV/m",
    location="face",
)
def calc_gradTe_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_p_us
    funv = np.zeros(grid.n_vertices)
    return grad_p_us(grid, 1, ws["te"], funv)


@quantity(
    name="gradTi_th",
    requires=[],
    description="poloidal ion temperature gradient",
    unit="eV/m",
    location="face",
)
def calc_gradTi_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import grad_p_us
    funv = np.zeros(grid.n_vertices)
    return grad_p_us(grid, 1, ws["ti"], funv)


@quantity(
    name="vel_ExBc_th",
    requires=[],
    description="cell-centered ExB velocity, poloidal",
    unit="m/s",
)
def calc_vel_ExBc_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import intcell_us
    ns = ws["vel_ExB_th"].shape[1]
    out = np.zeros((grid.n_cells, ns))
    for is_ in range(ns):
        out[:, is_] = intcell_us(grid, grid.intcell_p, ws["vel_ExB_th"][:, is_])
    return out


@quantity(
    name="ua_th",
    requires=[],
    description="poloidal projection of parallel velocity",
    unit="m/s",
)
def calc_ua_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    cv_bb = grid.cv_bb if grid.cv_bb is not None else np.ones((grid.n_cells, 4))
    return ws["ua"] * cv_bb[:, 0:1] / np.maximum(cv_bb[:, 3:4], 1e-30)


@quantity(
    name="cs_th",
    requires=[],
    description="poloidal projection of sound speed",
    unit="m/s",
)
def calc_cs_th(watch=None, grid=None, comp=None, **kw):
    cs = _sep_get(watch, grid, comp, "cs")
    if cs.size == 0:
        return np.zeros(grid.n_cells)
    cv_bb = grid.cv_bb if grid.cv_bb is not None else np.ones((grid.n_cells, 4))
    return cs * cv_bb[:, 0] / np.maximum(cv_bb[:, 3], 1e-30)


@quantity(
    name="E_up_r",
    requires=[],
    description="radial E-field from ExB (up)",
    unit="V/m",
    location="face",
)
def calc_E_up_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    from solps_analysis.core.operators import intcell_us
    ns = ws["vel_ExB_th"].shape[1]
    vel_c = np.zeros((grid.n_cells, ns))
    for is_ in range(ns):
        vel_c[:, is_] = intcell_us(grid, grid.intcell_p, ws["vel_ExB_th"][:, is_])
    cv_bb = grid.cv_bb if grid.cv_bb is not None else np.ones((grid.n_cells, 4))
    e_upc_r = vel_c[:, 1] * cv_bb[:, 3] ** 2 / np.maximum(cv_bb[:, 2], 1e-30)
    bp_dir = getattr(grid, "bp_dir", 0) or -1
    return intface(grid, e_upc_r, 1, _intface_method(grid)) * bp_dir

@quantity(
    name="smo_vis_tot",
    requires=[],
    description="total viscous momentum source",
    unit="Pa/m",
)
def calc_smo_vis_tot(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return (ws["smo_visq"] + ws["smo_vispar"]
            - (calc_div_fmo_vis(watch=watch, grid=grid)
               + calc_div_fmo_viscurv(watch=watch, grid=grid, comp=comp)))


@quantity(
    name="smo_vis_tot_th",
    requires=[],
    description="poloidal part of total viscous momentum source",
    unit="Pa/m",
)
def calc_smo_vis_tot_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    return (ws["smo_visq"] + ws["smo_vispar"]
            - (calc_div_fmo_vis_th(watch=watch, grid=grid)
               + calc_div_fmo_viscurv(watch=watch, grid=grid, comp=comp)))


@quantity(
    name="smo_vis_tot_r",
    requires=[],
    description="radial part of total viscous momentum source",
    unit="Pa/m",
)
def calc_smo_vis_tot_r(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return -(calc_div_fmo_vis_r(watch=watch, grid=grid))


# ──────────────────────────────────────────────────────────────
# snas / snas_bound_reg (lines 497-521)
# ──────────────────────────────────────────────────────────────

def _element_ion_indices(comp, ns: int) -> list[list[int]]:
    """For each element, indices of charged states (excludes neutral).

    MATLAB comp.sorts_index{isort}(2:end) with 1-based → 0-based here.
    """
    indices = getattr(comp, "element_indices_list", None)
    if indices is None:
        return [list(range(ns))]
    out = []
    for idx_list in indices:
        if len(idx_list) > 1:
            out.append(list(idx_list[1:]))  # drop neutral
        else:
            out.append([])
    return out


def _cvs_boundary(grid):
    """Guard-cell groups per boundary label (lazy, cached on grid).

    MATLAB read_geometry.m: for each fcLbl value, collect cells of those
    faces that are not core cells (cv_indx_0).  Order is irrelevant for
    the sums used in calc_additional.
    """
    cached = getattr(grid, "_cvs_boundary_cache", None)
    if cached is not None:
        return cached
    out = []
    if grid.fc_lbl is not None and grid.fc_cv is not None:
        n_ci = grid.n_core_cells
        for lbl in np.unique(grid.fc_lbl[grid.fc_lbl != 0]):
            mask = grid.fc_lbl == lbl
            cvs = np.unique(grid.fc_cv[mask])  # 0-based
            cvs = cvs[cvs >= n_ci]  # guard cells only
            out.append((int(lbl), cvs))
    grid._cvs_boundary_cache = out
    return out


@quantity(
    name="snas",
    requires=[],
    description="neutral source per element (sum of ion sources)",
    unit="m⁻³ s⁻¹",
)
def calc_snas(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if comp is None:
        return np.zeros(grid.n_cells)
    eirene_flag = getattr(watch, "neut", None) is not None
    src = ws["sna_eir"] if eirene_flag and "sna_eir" in ws else ws.get("sna")
    if src is None:
        return np.zeros(grid.n_cells)
    ns = src.shape[1]
    ion_idx = _element_ion_indices(comp, ns)
    out = np.zeros((grid.n_cells, len(ion_idx)))
    for iel, idxs in enumerate(ion_idx):
        if idxs:
            out[:, iel] = src[:, idxs].sum(axis=1)
    return out


@quantity(
    name="snas_bound_reg",
    requires=[],
    description="neutral source integrated per boundary region and element",
    unit="s⁻¹",
)
def calc_snas_bound_reg(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    if comp is None:
        return np.zeros((max(1, grid.cv_reg.max()), 1))
    eirene_flag = getattr(watch, "neut", None) is not None
    src = ws["sna_eir"] if eirene_flag and "sna_eir" in ws else ws.get("sna")
    if src is None:
        return np.zeros((1, 1))
    ns = src.shape[1]
    ion_idx = _element_ion_indices(comp, ns)
    groups = _cvs_boundary(grid)
    out = np.zeros((len(groups), len(ion_idx)))
    for ig, (lbl, cvs) in enumerate(groups):
        for iel, idxs in enumerate(ion_idx):
            if len(idxs) > 0:
                out[ig, iel] = src[np.ix_(np.asarray(cvs, dtype=np.intp),
                                           np.asarray(idxs, dtype=np.intp))].sum()
    return out
