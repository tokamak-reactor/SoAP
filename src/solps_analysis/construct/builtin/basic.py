"""Builtin physical quantities — первые величины из calc_additional."""

from __future__ import annotations

import numpy as np

from solps_analysis.construct.registry import quantity

# ──────────────────────────────────────────────
# Physical constants (SI)
# ──────────────────────────────────────────────

QE = 1.602176487e-19   # Elementary charge [C]
MP = 1.672621637e-27   # Proton mass [kg]
ME = 9.10938215e-31    # Electron mass [kg]
EPS0 = 8.8542e-12      # Vacuum permittivity [F/m]


# ──────────────────────────────────────────────
# Density-related
# ──────────────────────────────────────────────


@quantity(
    name="ni",
    requires=["na"],
    description="Ion density: ni[:,0]=total, ni[:,1]=ions only",
    unit="m⁻³",
)
def calc_ni(na, grid=None, comp=None, **kw):
    """Ion density from na.

    ni[:, 0] = total (neutrals + ions)
    ni[:, 1] = ions only (zamax > 0)
    """
    ns = na.shape[1] if na.ndim > 1 else 0
    if comp is None or ns == 0:
        return np.zeros((na.shape[0], 2))

    neutrals = np.zeros(na.shape[0])
    ions = np.zeros(na.shape[0])

    for isp in range(ns):
        if comp.zamax[isp] > 0:
            ions += na[:, isp]
        else:
            neutrals += na[:, isp]

    result = np.column_stack([neutrals + ions, ions])
    return result


@quantity(
    name="rho",
    requires=["na"],
    description="Mass density of charged particles",
    unit="kg/m³",
)
def calc_rho(na, grid=None, comp=None, **kw):
    """Mass density: sum(na[:,is] * am[is] * mp) for all species."""
    ns = na.shape[1] if na.ndim > 1 else 0
    if comp is None or ns == 0:
        return np.zeros(na.shape[0])

    result = np.zeros(na.shape[0])
    for isp in range(ns):
        result += na[:, isp] * comp.am[isp] * MP

    return result


# ──────────────────────────────────────────────
# Pressure
# ──────────────────────────────────────────────


@quantity(
    name="p_ch",
    requires=["ne", "te_eV", "ti_eV", "na"],
    description="Charged particle pressure: ne*Te + ni*Ti",
    unit="Pa",
)
def calc_p_ch(ne, te_eV, ti_eV, na, grid=None, comp=None, **kw):
    """Pressure of charged particles.

    p_ch = ne * Te * qe + ni_ions * Ti * qe
    where ni_ions = sum of na[:, is] for species with zamax > 0
    """
    te_j = te_eV * QE  # eV → J
    ti_j = ti_eV * QE

    ni_ions = np.zeros(na.shape[0])
    ns = na.shape[1] if na.ndim > 1 else 0
    if comp is not None:
        for isp in range(ns):
            if comp.zamax[isp] > 0:
                ni_ions += na[:, isp]

    return ne * te_j + ni_ions * ti_j


# ──────────────────────────────────────────────
# Sound speed
# ──────────────────────────────────────────────


@quantity(
    name="cs",
    requires=["ne", "te_eV", "ti_eV", "na"],
    description="Speed of sound: sqrt(p_ch / rho)",
    unit="m/s",
)
def calc_cs(ne, te_eV, ti_eV, na, grid=None, comp=None, **kw):
    """Speed of sound."""
    # Recompute p_ch and rho inline
    te_j = te_eV * QE
    ti_j = ti_eV * QE

    ni_ions = np.zeros(na.shape[0])
    rho = np.zeros(na.shape[0])
    ns = na.shape[1] if na.ndim > 1 else 0

    for isp in range(ns):
        if comp is not None and comp.zamax[isp] > 0:
            ni_ions += na[:, isp]
        rho += na[:, isp] * (comp.am[isp] if comp else 2) * MP

    p_ch = ne * te_j + ni_ions * ti_j
    return np.sqrt(p_ch / np.maximum(rho, 1e-30))


@quantity(
    name="csi",
    requires=["ti_eV", "na"],
    description="Ion thermal velocity: sqrt(ni*Ti / rho)",
    unit="m/s",
)
def calc_csi(ti_eV, na, grid=None, comp=None, **kw):
    """Ion thermal velocity."""
    ti_j = ti_eV * QE

    ni_ions = np.zeros(na.shape[0])
    rho = np.zeros(na.shape[0])
    ns = na.shape[1] if na.ndim > 1 else 0

    for isp in range(ns):
        mass = (comp.am[isp] if comp else 2) * MP
        if comp is not None and comp.zamax[isp] > 0:
            ni_ions += na[:, isp]
        rho += na[:, isp] * mass

    return np.sqrt(ni_ions * ti_j / np.maximum(rho, 1e-30))


# ──────────────────────────────────────────────
# Zeff
# ──────────────────────────────────────────────


@quantity(
    name="Zeff",
    requires=["ne", "na"],
    description="Effective ion charge: Z_eff",
    unit="",
)
def calc_zeff(ne, na, grid=None, comp=None, **kw):
    """Effective charge Z_eff = sum(ni * Zi^2) / ne."""
    ns = na.shape[1] if na.ndim > 1 else 0
    result = np.zeros(na.shape[0])

    for isp in range(ns):
        z = comp.zamax[isp] if comp is not None else 1
        if z > 0:
            result += na[:, isp] * z ** 2

    return result / np.maximum(ne, 1e-10)
