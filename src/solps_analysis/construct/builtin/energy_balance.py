"""Port of MATLAB energy_balance_Extended.m — energy densities and fluxes.

Provides fee/fei/fet (electron/ion/total energy fluxes) and se_* energy
sources used by the OMP-integral block and the heat-source block of
calc_additional.m.  Uses the MATLAB-style workspace from io.matlab_vars.
"""

from __future__ import annotations

import numpy as np

from solps_analysis.construct.registry import quantity
from solps_analysis.core.operators import div_us, intface
from solps_analysis.io.ionization_potentials import get_ion_pot

QE = 1.602176634e-19
MP = 1.672621637e-27


def _ws(watch) -> dict:
    from solps_analysis.io.matlab_vars import build_workspace
    return build_workspace(watch)


def _intface_method(grid) -> str:
    return "vol" if grid.version_float >= 3.002 else "halfsum"


def _zs(comp):
    """Charge per species (zamax when comp.zs absent)."""
    if comp is None:
        return None
    zs = getattr(comp, "zs", None)
    if zs is None:
        zamax = getattr(comp, "zamax", None)
        if zamax is not None:
            zs = np.asarray(zamax)
    return zs


def _ams_kg(comp) -> np.ndarray | None:
    if comp is None:
        return None
    am = getattr(comp, "am", None)
    if am is None:
        return None
    return np.asarray(am, dtype=np.float64) * MP


def _E_pot_ion(watch, comp) -> np.ndarray:
    """Cumulative ionization potential per species (MATLAB E_pot_ion).

    E_pot_ion(is) = sum(ion_pot[neutral..is]) — total energy spent to
    ionize a neutral into charge state is.
    """
    ion_pot = get_ion_pot(watch, comp)
    indices = getattr(comp, "element_indices_list", None)
    ns = len(ion_pot)
    e_pot = np.zeros(ns)
    if indices is None:
        return e_pot
    for idx_list in indices:
        run = 0.0
        for is_ in idx_list:
            run += ion_pot[is_]
            e_pot[is_] = run
    return e_pot


@quantity(
    name="en_e",
    requires=[],
    description="electron thermal energy density",
    unit="J",
)
def calc_en_e(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return 1.5 * QE * ws["ne"] * ws["te"] * grid.cv_vol


@quantity(
    name="en_i_int",
    requires=[],
    description="ion thermal energy density",
    unit="J",
)
def calc_en_i_int(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    zs = _zs(comp)
    na = ws["na"]
    if zs is not None:
        na = na[:, np.asarray(zs) != 0]
    return 1.5 * QE * na.sum(axis=1) * ws["ti"] * grid.cv_vol


@quantity(
    name="en_i_kin",
    requires=[],
    description="ion kinetic energy density",
    unit="J",
)
def calc_en_i_kin(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    zs = _zs(comp)
    na = ws["na"]
    ua = ws["ua"]
    if zs is not None:
        keep = np.asarray(zs) != 0
        na = na[:, keep]
        ua = ua[:, keep]
        ams = ams[keep] if ams is not None else None
    if ams is None:
        return np.zeros(grid.n_cells)
    return 0.5 * (na * ua ** 2) @ ams * grid.cv_vol


@quantity(
    name="en_i_pot",
    requires=[],
    description="ionization potential energy density",
    unit="J",
)
def calc_en_i_pot(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    e_pot = _E_pot_ion(watch, comp)
    return QE * (ws["na"] * e_pot[None, :]).sum(axis=1) * grid.cv_vol


@quantity(
    name="en",
    requires=[],
    description="total energy density",
    unit="J",
)
def calc_en(watch=None, grid=None, comp=None, **kw):
    return (calc_en_e(watch=watch, grid=grid)
            + calc_en_i_int(watch=watch, grid=grid, comp=comp)
            + calc_en_i_kin(watch=watch, grid=grid, comp=comp)
            + calc_en_i_pot(watch=watch, grid=grid, comp=comp))


# ──────────────────────────────────────────────────────────────
# Electron energy flux (fee)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fee_th",
    requires=[],
    description="total electron energy flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fee_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    zs = _zs(comp)
    if zs is None:
        return np.zeros(grid.n_faces)

    # fne_nuBgradB_th = -(fna_nuBgradB_th * zs^2) * tef/tif
    with np.errstate(divide="ignore", invalid="ignore"):
        fne_nuBg = -(ws["fna_nuBgradB_th"] * zs[None, :] ** 2).sum(axis=1) * (tef / tif)
    fne_nuBg = np.nan_to_num(fne_nuBg, nan=0.0, posinf=0.0, neginf=0.0)
    fhe_nuBg = 2.5 * fne_nuBg * tef * QE

    fne_curr = ws.get("fne_curr_th")
    if fne_curr is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fne_curr_th
        fne_curr = calc_fne_curr_th(watch=watch, grid=grid, comp=comp)
    fne_th = ws.get("fne_th")
    if fne_th is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fne_th
        fne_th = calc_fne_th(watch=watch, grid=grid, comp=comp)

    fhe_conv = 2.5 * fne_th * tef * QE - ws.get("fhe_fnaAN_th", 0.0) / 1.5
    return (ws["fhe_mdf_th"] + (fhe_nuBg - ws["fhe_dia_mdf_th"])
            + fhe_conv / 1.5 - fhe_nuBg / 2.5
            - ws.get("fhe_fnaAN_th", 0.0) / 1.5)


@quantity(
    name="fee_r",
    requires=[],
    description="total electron energy flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fee_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tef = ws["tef"] if "tef" in ws else intface(grid, ws["te"], 1, _intface_method(grid))
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    zs = _zs(comp)
    if zs is None:
        return np.zeros(grid.n_faces)

    with np.errstate(divide="ignore", invalid="ignore"):
        fne_nuBg = -(ws["fna_nuBgradB_r"] * zs[None, :] ** 2).sum(axis=1) * (tef / tif)
    fne_nuBg = np.nan_to_num(fne_nuBg, nan=0.0, posinf=0.0, neginf=0.0)
    fhe_nuBg = 2.5 * fne_nuBg * tef * QE

    fne_curr = ws.get("fne_curr_r")
    if fne_curr is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fne_curr_r
        fne_curr = calc_fne_curr_r(watch=watch, grid=grid, comp=comp)
    fne_r = ws.get("fne_r")
    if fne_r is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fne_r
        fne_r = calc_fne_r(watch=watch, grid=grid, comp=comp)

    fhe_conv = 2.5 * fne_r * tef * QE - ws.get("fhe_fnaAN_r", 0.0) / 1.5
    return (ws["fhe_mdf_r"] + (fhe_nuBg - ws["fhe_dia_mdf_r"])
            + fhe_conv / 1.5 - fhe_nuBg / 2.5
            - ws.get("fhe_fnaAN_r", 0.0) / 1.5)


# ──────────────────────────────────────────────────────────────
# Ion energy flux (fei)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fei_kin_th",
    requires=[],
    description="ion kinetic energy flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fei_kin_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    if ams is None:
        return np.zeros(grid.n_faces)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    flux = ws["fna_mdf_th"] + (ws["fna_nuBgradB_th"] - ws["fna_dia_mdf_th"])
    return (uaf ** 2 * flux) @ ams / 2


@quantity(
    name="fei_kin_r",
    requires=[],
    description="ion kinetic energy flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fei_kin_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    if ams is None:
        return np.zeros(grid.n_faces)
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    flux = ws["fna_mdf_r"] + (ws["fna_nuBgradB_r"] - ws["fna_dia_mdf_r"])
    return (uaf ** 2 * flux) @ ams / 2


@quantity(
    name="fei_curr_th",
    requires=[],
    description="ion current energy flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fei_curr_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fna_curr = ws.get("fna_curr_th")
    if fna_curr is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fna_curr_th
        fna_curr = calc_fna_curr_th(watch=watch, grid=grid, comp=comp)
    fni_curr = fna_curr.sum(axis=1)
    if ams is None:
        return 2.5 * fni_curr * tif * QE
    return 2.5 * fni_curr * tif * QE + (fna_curr * uaf ** 2) @ ams / 2


@quantity(
    name="fei_curr_r",
    requires=[],
    description="ion current energy flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fei_curr_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    uaf = intface(grid, ws["ua"], 1, _intface_method(grid))
    fna_curr = ws.get("fna_curr_r")
    if fna_curr is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fna_curr_r
        fna_curr = calc_fna_curr_r(watch=watch, grid=grid, comp=comp)
    fni_curr = fna_curr.sum(axis=1)
    if ams is None:
        return 2.5 * fni_curr * tif * QE
    return 2.5 * fni_curr * tif * QE + (fna_curr * uaf ** 2) @ ams / 2


@quantity(
    name="fei_th",
    requires=[],
    description="total ion energy flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fei_th(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    # fni_phys_th = fni_mdf + fni_nuBgradB - fni_dia_mdf
    fni_mdf = ws["fna_mdf_th"].sum(axis=1)
    fni_nuBg = ws["fna_nuBgradB_th"].sum(axis=1)
    fni_dia = ws["fna_dia_mdf_th"].sum(axis=1)
    fni_phys = fni_mdf + fni_nuBg - fni_dia
    fhi_nuBg = 2.5 * fni_nuBg * tif * QE
    fei_conv = 2.5 * fni_phys * tif * QE - ws.get("fhi_fnaAN_th", 0.0) / 1.5
    fei_kin = ws.get("fei_kin_th")
    if fei_kin is None:
        fei_kin = calc_fei_kin_th(watch=watch, grid=grid, comp=comp)
    return (ws["fhi_mdf_th"] + (fhi_nuBg - ws["fhi_dia_mdf_th"])
            + fei_conv / 1.5 - fhi_nuBg / 2.5
            + fei_kin - ws.get("fhi_fnaAN_th", 0.0) / 1.5)


@quantity(
    name="fei_r",
    requires=[],
    description="total ion energy flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fei_r(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    tif = intface(grid, ws["ti"], 1, _intface_method(grid))
    fni_mdf = ws["fna_mdf_r"].sum(axis=1)
    fni_nuBg = ws["fna_nuBgradB_r"].sum(axis=1)
    fni_dia = ws["fna_dia_mdf_r"].sum(axis=1)
    fni_phys = fni_mdf + fni_nuBg - fni_dia
    fhi_nuBg = 2.5 * fni_nuBg * tif * QE
    fei_conv = 2.5 * fni_phys * tif * QE - ws.get("fhi_fnaAN_r", 0.0) / 1.5
    fei_kin = ws.get("fei_kin_r")
    if fei_kin is None:
        fei_kin = calc_fei_kin_r(watch=watch, grid=grid, comp=comp)
    return (ws["fhi_mdf_r"] + (fhi_nuBg - ws["fhi_dia_mdf_r"])
            + fei_conv / 1.5 - fhi_nuBg / 2.5
            + fei_kin - ws.get("fhi_fnaAN_r", 0.0) / 1.5)


# ──────────────────────────────────────────────────────────────
# Total energy flux (fet)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="fet_th",
    requires=[],
    description="total energy flux, poloidal",
    unit="W/m²",
    location="face",
)
def calc_fet_th(watch=None, grid=None, comp=None, **kw):
    fee = ws_get(watch, "fee_th", grid, comp)
    fei = ws_get(watch, "fei_th", grid, comp)
    fei_kin = ws_get(watch, "fei_kin_th", grid, comp)
    return fee + fei + fei_kin


@quantity(
    name="fet_r",
    requires=[],
    description="total energy flux, radial",
    unit="W/m²",
    location="face",
)
def calc_fet_r(watch=None, grid=None, comp=None, **kw):
    fee = ws_get(watch, "fee_r", grid, comp)
    fei = ws_get(watch, "fei_r", grid, comp)
    fei_kin = ws_get(watch, "fei_kin_r", grid, comp)
    return fee + fei + fei_kin


def ws_get(watch, name, grid, comp):
    ws = _ws(watch)
    val = ws.get(name)
    if val is not None:
        return val
    from solps_analysis.construct.builtin.energy_balance import (
        calc_fee_th, calc_fee_r, calc_fei_th, calc_fei_r,
        calc_fei_kin_th, calc_fei_kin_r,
    )
    fn = {
        "fee_th": calc_fee_th, "fee_r": calc_fee_r,
        "fei_th": calc_fei_th, "fei_r": calc_fei_r,
        "fei_kin_th": calc_fei_kin_th, "fei_kin_r": calc_fei_kin_r,
    }[name]
    return fn(watch=watch, grid=grid, comp=comp)


# ──────────────────────────────────────────────────────────────
# Energy sources (se_*)
# ──────────────────────────────────────────────────────────────

@quantity(
    name="se_inel",
    requires=[],
    description="inelastic energy source",
    unit="W/m³",
)
def calc_se_inel(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return (ws.get("she_rad", 0) + ws.get("she_eir", 0) + ws.get("shi_eir", 0)
            + ws.get("shi_ion", 0) + ws.get("shi_rec", 0))


@quantity(
    name="se_EM_BzB",
    requires=[],
    description="electromagnetic energy source from Joule heating",
    unit="W/m³",
)
def calc_se_EM_BzB(watch=None, grid=None, **kw):
    ws = _ws(watch)
    fh_joule_th = ws.get("fh_joule_th")
    fh_joule_r = ws.get("fh_joule_r")
    if fh_joule_th is None or fh_joule_r is None:
        from solps_analysis.construct.builtin.calc_additional import (
            calc_fh_joule_th, calc_fh_joule_r)
        fh_joule_th = calc_fh_joule_th(watch=watch, grid=grid)
        fh_joule_r = calc_fh_joule_r(watch=watch, grid=grid)
    return -div_us(grid, np.column_stack([fh_joule_th, fh_joule_r]))


@quantity(
    name="se_EM_LK",
    requires=[],
    description="electromagnetic energy source (Lorentz-Kaveeva)",
    unit="W/m³",
)
def calc_se_EM_LK(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fhe_nupar_th = ws.get("fhe_nupar_th")
    if fhe_nupar_th is None:
        from solps_analysis.construct.builtin.calc_additional import calc_fhe_nupar_th
        fhe_nupar_th = calc_fhe_nupar_th(watch=watch, grid=grid, comp=comp)
    fhe_nuExB_th = ws.get("fhe_nuExB_th")
    fhe_nuExB_r = ws.get("fhe_nuExB_r")
    if fhe_nuExB_th is None or fhe_nuExB_r is None:
        from solps_analysis.construct.builtin.calc_additional import (
            calc_fhe_nuExB_th, calc_fhe_nuExB_r)
        fhe_nuExB_th = calc_fhe_nuExB_th(watch=watch, grid=grid, comp=comp)
        fhe_nuExB_r = calc_fhe_nuExB_r(watch=watch, grid=grid, comp=comp)
    smo_gradpo = ws.get("smo_gradpo", np.zeros(grid.n_cells))
    cv_hz = grid.cv_hz if grid.cv_hz is not None else np.ones(grid.n_cells)
    return (div_us(grid, np.column_stack([fhe_nupar_th, np.zeros(grid.n_faces)])) / 1.5
            + ws.get("she_du", 0) + ws.get("she_fr", 0)
            + smo_gradpo.sum(axis=1) / cv_hz
            + div_us(grid, np.column_stack([fhe_nuExB_th, fhe_nuExB_r])) / 1.5
            + ws.get("she_dd", 0) + ws.get("shi_dd", 0))


@quantity(
    name="se_vis",
    requires=[],
    description="viscous energy source",
    unit="W/m³",
)
def calc_se_vis(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ua = ws["ua"]
    cv_hz = grid.cv_hz if grid.cv_hz is not None else np.ones(grid.n_cells)
    smo_vispar = ws.get("smo_vispar", np.zeros((grid.n_cells, ua.shape[1])))
    smo_visq = ws.get("smo_visq", np.zeros((grid.n_cells, ua.shape[1])))
    smovi = ws.get("smovi", np.zeros((grid.n_cells, ua.shape[1])))
    div_fmo_vis = ws.get("div_fmo_vis")
    if div_fmo_vis is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fmo_vis
        div_fmo_vis = calc_div_fmo_vis(watch=watch, grid=grid)
    div_fmo_viscurv = ws.get("div_fmo_viscurv")
    if div_fmo_viscurv is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fmo_viscurv
        div_fmo_viscurv = calc_div_fmo_viscurv(watch=watch, grid=grid, comp=comp)
    div_fmo_vis_BgradB = ws.get("div_fmo_vis_BgradB")
    if div_fmo_vis_BgradB is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fmo_vis_BgradB
        div_fmo_vis_BgradB = calc_div_fmo_vis_BgradB(watch=watch, grid=grid, comp=comp)

    se_uavispar = (ua * smo_vispar).sum(axis=1) / cv_hz
    se_uavis_cond = -(ua * div_fmo_vis).sum(axis=1) / cv_hz
    se_uavisq = (ua * smo_visq).sum(axis=1) / cv_hz
    se_uavisi = (ua * smovi).sum(axis=1) / cv_hz
    se_uaviscurv = -(ua * div_fmo_viscurv).sum(axis=1) / cv_hz
    se_uavisc_BgradB = -(ua * div_fmo_vis_BgradB).sum(axis=1) / cv_hz

    n_ci = grid.n_core_cells
    se_uavispar[n_ci:] = 0
    se_uavis_cond[n_ci:] = 0
    se_uavisq[n_ci:] = 0
    se_uaviscurv[n_ci:] = 0

    return (ws.get("shi_visan", 0) + ws.get("shi_viscl", 0)
            + se_uavispar + se_uavis_cond + se_uavisq + se_uaviscurv
            + se_uavisc_BgradB + se_uavisi)


@quantity(
    name="se_vis_num",
    requires=[],
    description="viscous energy source (numerical)",
    unit="W/m³",
)
def calc_se_vis_num(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ua = ws["ua"]
    cv_hz = grid.cv_hz if grid.cv_hz is not None else np.ones(grid.n_cells)
    smo_vispar = ws.get("smo_vispar", np.zeros((grid.n_cells, ua.shape[1])))
    smo_visq = ws.get("smo_visq", np.zeros((grid.n_cells, ua.shape[1])))
    smovi = ws.get("smovi", np.zeros((grid.n_cells, ua.shape[1])))
    div_fmo_vis = ws.get("div_fmo_vis")
    if div_fmo_vis is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fmo_vis
        div_fmo_vis = calc_div_fmo_vis(watch=watch, grid=grid)
    se_uavispar = (ua * smo_vispar).sum(axis=1) / cv_hz
    se_uavis_cond = -(ua * div_fmo_vis).sum(axis=1) / cv_hz
    se_uavisq = (ua * smo_visq).sum(axis=1) / cv_hz
    se_uavisi = (ua * smovi).sum(axis=1) / cv_hz
    n_ci = grid.n_core_cells
    se_uavispar[n_ci:] = 0
    se_uavis_cond[n_ci:] = 0
    se_uavisq[n_ci:] = 0
    return (ws.get("shi_visan", 0) + ws.get("shi_viscl", 0)
            + se_uavispar + se_uavis_cond + se_uavisq + se_uavisi)


@quantity(
    name="se_Ekin_divGamma",
    requires=[],
    description="kinetic energy source from continuity",
    unit="W/m³",
)
def calc_se_Ekin_divGamma(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    if ams is None:
        return np.zeros(grid.n_cells)
    div_fna = ws.get("div_fna")
    if div_fna is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fna
        div_fna = calc_div_fna(watch=watch, grid=grid)
    return (div_fna * ws["ua"] ** 2) @ ams


@quantity(
    name="se_Ekin_divGamma_mdf",
    requires=[],
    description="kinetic energy source (mdf formulation)",
    unit="W/m³",
)
def calc_se_Ekin_divGamma_mdf(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ams = _ams_kg(comp)
    if ams is None:
        return np.zeros(grid.n_cells)
    div_fna_mdf = ws.get("div_fna_mdf")
    if div_fna_mdf is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fna_mdf
        div_fna_mdf = calc_div_fna_mdf(watch=watch, grid=grid)
    return (div_fna_mdf * ws["ua"] ** 2) @ ams


@quantity(
    name="se_cf",
    requires=[],
    description="centrifugal energy source",
    unit="W/m³",
)
def calc_se_cf(watch=None, grid=None, **kw):
    ws = _ws(watch)
    return (ws["ua"] * ws.get("smo_cf", np.zeros_like(ws["ua"]))).sum(axis=1)


@quantity(
    name="se_divEkinGamma",
    requires=[],
    description="divergence of kinetic energy flux",
    unit="W/m³",
)
def calc_se_divEkinGamma(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    fei_kin_th = ws.get("fei_kin_th")
    fei_kin_r = ws.get("fei_kin_r")
    if fei_kin_th is None or fei_kin_r is None:
        fei_kin_th = calc_fei_kin_th(watch=watch, grid=grid, comp=comp)
        fei_kin_r = calc_fei_kin_r(watch=watch, grid=grid, comp=comp)
    return div_us(grid, np.column_stack([fei_kin_th, fei_kin_r]))


@quantity(
    name="se_inert",
    requires=[],
    description="inertial energy source",
    unit="W/m³",
)
def calc_se_inert(watch=None, grid=None, comp=None, **kw):
    ws = _ws(watch)
    ua = ws["ua"]
    cv_hz = grid.cv_hz if grid.cv_hz is not None else np.ones(grid.n_cells)
    div_fmo_flo = ws.get("div_fmo_flo")
    if div_fmo_flo is None:
        from solps_analysis.construct.builtin.calc_additional import calc_div_fmo_flo
        div_fmo_flo = calc_div_fmo_flo(watch=watch, grid=grid)
    se_ekin_mdf = ws.get("se_Ekin_divGamma_mdf")
    if se_ekin_mdf is None:
        se_ekin_mdf = calc_se_Ekin_divGamma_mdf(watch=watch, grid=grid, comp=comp)
    se_cf = ws.get("se_cf")
    if se_cf is None:
        se_cf = calc_se_cf(watch=watch, grid=grid)
    se_divEkin = ws.get("se_divEkinGamma")
    if se_divEkin is None:
        se_divEkin = calc_se_divEkinGamma(watch=watch, grid=grid, comp=comp)
    return -(ua * div_fmo_flo).sum(axis=1) / cv_hz + se_ekin_mdf + se_cf + se_divEkin
