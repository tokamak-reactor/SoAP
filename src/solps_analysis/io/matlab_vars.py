"""Assemble MATLAB-style workspace variables from a SolpsWatch.

Mirrors MATLAB read_data_3x.m: species-indexed .dat files are stacked into
(n_faces, ns) / (n_cells, ns) matrices with MATLAB's fallback chain
(e.g. fna_mdf_th ← b2npc11_fnax, else b2npc9_fnax, ... else b2npco_fnax).

Face variables are mapped through imap_fcx (dim=1) / imap_fcy (dim=2)
thanks to the variable catalog; cell variables through imap_cv.
"""

from __future__ import annotations

import re

import numpy as np

from solps_analysis.io.variable_catalog import _catalog_transform

MP = 1.672621637e-27  # kg, proton mass (MATLAB mp)


def n_species(watch) -> int:
    """Number of B2 species (ns) from b2fstati."""
    from solps_analysis.io.b2fstate_reader import read_b2fstate_full

    try:
        raw = read_b2fstate_full(str(watch.path))
        dim = raw.get("nx,ny,ns")
        if isinstance(dim, np.ndarray) and dim.size >= 3:
            return int(dim[2])
        if isinstance(dim, (list, tuple)) and len(dim) >= 3:
            return int(dim[2])
    except Exception:
        pass
    return 0


def species_zamax(watch) -> np.ndarray:
    """zamax per species from b2fstati."""
    from solps_analysis.io.b2fstate_reader import read_b2fstate_full

    try:
        raw = read_b2fstate_full(str(watch.path))
        zamax = raw.get("zamax")
        if zamax is not None:
            return np.asarray(zamax, dtype=np.float64).ravel()
    except Exception:
        pass
    return np.zeros(n_species(watch))


def species_am(watch) -> np.ndarray:
    """Atomic mass (amu) per species from b2fstati."""
    from solps_analysis.io.b2fstate_reader import read_b2fstate_full

    try:
        raw = read_b2fstate_full(str(watch.path))
        am = raw.get("am")
        if am is not None:
            return np.asarray(am, dtype=np.float64).ravel()
    except Exception:
        pass
    return np.zeros(n_species(watch))


def _resolve_species_file(watch, base_names: list[str], is_: int) -> str | None:
    """Find the file for species index is_ (0-based) among base_names.

    MATLAB reads b2npc11_fnax%03d first, falls back to b2npc9, b2npc7,
    b2npco, etc.
    """
    files = getattr(watch, "_file_index", None) or {}
    for base in base_names:
        key = f"{base}_{is_:03d}"
        if key in files:
            return files[key]
    return None


def get_species_var(watch, base_names: list[str], dim: int | None = None,
                    sign: float = 1.0, divisor: str | None = None,
                    n_spec: int | None = None) -> np.ndarray:
    """Stack a species-indexed variable into (n, ns).

    base_names: fallback chain of file prefixes (most preferred first).
    dim: 1 → x-faces, 2 → y-faces, None → cells (from catalog if omitted).
    sign: multiplicative sign.
    divisor: grid attribute name to divide by (e.g. 'fc_hz' or None).
    """
    ns = n_spec if n_spec is not None else n_species(watch)
    if ns == 0:
        return np.zeros(0)

    # probe first file to learn size
    first = None
    for is_ in range(ns):
        fpath = _resolve_species_file(watch, base_names, is_)
        if fpath is not None:
            first = fpath
            break
    if first is None:
        # no data at all: return empty (nFc or nCv) sized array
        n = watch.grid.n_faces if (dim in (1, 2)) else watch.grid.n_cells
        return np.zeros((n, ns))

    probe = _read_var(watch, first)
    n = probe.shape[0]
    out = np.zeros((n, ns), dtype=np.float64)

    for is_ in range(ns):
        fpath = _resolve_species_file(watch, base_names, is_)
        if fpath is None:
            continue  # MATLAB leaves zeros + warning
        val = _read_var(watch, fpath)
        if divisor is not None:
            val = _divide_by_grid(watch, val, divisor)
        out[:, is_] = val

    out *= sign
    return out


def get_scalar_var(watch, base_names: list[str], dim: int | None = None,
                   sign: float = 1.0, divisor: str | None = None) -> np.ndarray:
    """Read a non-species .dat variable (n,) with fallback chain."""
    files = getattr(watch, "_file_index", None) or {}
    for base in base_names:
        if base in files:
            val = _read_var(watch, files[base])
            if divisor is not None:
                val = _divide_by_grid(watch, val, divisor)
            return val * sign
    # fallback: zeros of proper size
    n = watch.grid.n_faces if (dim in (1, 2)) else watch.grid.n_cells
    return np.zeros(n)


def _divide_by_grid(watch, val: np.ndarray, divisor: str) -> np.ndarray:
    """Divide by a grid quantity (or special 'fc_qgam_cos')."""
    if divisor == "fc_qgam_cos":
        qgam = getattr(watch.grid, "fc_qgam", None)
        if qgam is not None and qgam.ndim == 2 and qgam.shape[1] >= 1:
            d = qgam[:, 0]
            return val / np.where(d == 0, np.nan, d)
        return val
    div = getattr(watch.grid, divisor, None)
    if div is not None:
        return val / np.where(div == 0, np.nan, div)
    return val


def _read_var(watch, fpath: str) -> np.ndarray:
    """Read a single .dat file through the watch's catalog-aware reader."""
    if watch.get("__catalog_reader__") is not None:
        # use cached raw reads if present (not used by default)
        pass
    return watch._read_dat_file(fpath)


# ──────────────────────────────────────────────────────────────
# MATLAB workspace assembly (read_data_3x.m)
# ──────────────────────────────────────────────────────────────

# Species-indexed face variables: base_names (fallback order), dim, sign
_FACE_SPECIES_VARS: dict[str, tuple] = {
    "fna_mdf_th": (["b2npc11_fnax", "b2npc9_fnax", "b2npc7_fnax", "b2npco_fnax", "b2tfnb_fnb_mdfx"], 1, 1.0),
    "fna_mdf_r":  (["b2npc11_fnay", "b2npc9_fnay", "b2npc7_fnay", "b2npco_fnay", "b2tfnb_fnb_mdfy"], 2, 1.0),
    "fna_th":     (["b2tfnb_fnbx", "b2npc11_fnax"], 1, 1.0),
    "fna_r":      (["b2tfnb_fnby", "b2npc11_fnay"], 2, 1.0),
    "fna_nupar_th": (["b2tfnb_bxuanax"], 1, 1.0),
    "fna_nuBgradB_th": (["b2tfnb_vadianax"], 1, 1.0),
    "fna_nuBgradB_r":  (["b2tfnb_vadianay"], 2, 1.0),
    "fna_nuExB_th": (["b2tfnb_vaecrbnax"], 1, 1.0),
    "fna_nuExB_r":  (["b2tfnb_vaecrbnay"], 2, 1.0),
    "fna_dia_mdf_th": (["b2tfnb_fnbPSchx"], 1, -1.0),
    "fna_dia_mdf_r":  (["b2tfnb_fnbPSchy"], 2, -1.0),
    "fna_nuAN_r":   (["b2tfnb_cvlbnay"], 2, 1.0),
    "fna_RhieChow_th": (["b2tfnb_dpccornax"], 1, 1.0),
    "fna_Dgradn_th": (["b2tfnb_dPat_mdf_gradnax"], 1, 1.0),
    "fna_Dgradn_r":  (["b2tfnb_dPat_mdf_gradnay"], 2, 1.0),
    "fna_fha_th":   (["b2tfnb_fnb_hex"], 1, 1.0),
    "fna_fha_r":    (["b2tfnb_fnb_hey"], 2, 1.0),
    "fna_flo_th":   (["b2tfnb_fnb_32x"], 1, 1.0),
    "fna_flo_r":    (["b2tfnb_fnb_32y"], 2, 1.0),
    "fna_mo_th":    (["b2tfnb_fnb_fcorx"], 1, 1.0),
    "fna_mo_r":     (["b2tfnb_fnb_fcory"], 2, 1.0),
    "fna_mo_vis_th": (["b2npmo_flubvx"], 1, 1.0),
    "fmo_th":       (["b2npmo_fmox"], 1, 1.0),
    "fmo_r":        (["b2npmo_fmoy"], 2, 1.0),
    "fmo_vis_th":   (["b2urmo_etaPat_graduax"], 1, 1.0),
    "fmo_vis_r":    (["b2urmo_etaPat_graduay"], 2, 1.0),
    "fmo_flo_th":   (["b2urmo_etaPat_uax"], 1, 1.0),
    "fmo_flo_r":    (["b2urmo_etaPat_uay"], 2, 1.0),
    "fch_AN_th":    (["b2tfch_fchanml_ax", "b2tfnb_fchanml_bx"], 1, 1.0),
    "fch_AN_r":     (["b2tfch_fchanml_ay", "b2tfnb_fchanml_by"], 2, 1.0),
    "fch_inert_th": (["b2tfch_fchinert_ax", "b2tfnb_fchinert_bx"], 1, 1.0),
    "fch_inert_r":  (["b2tfch_fchinert_ay", "b2tfnb_fchinert_by"], 2, 1.0),
    "fch_vispar_th": (["b2tfch_fchvispar_ax", "b2tfnb_fchvispar_bx"], 1, 1.0),
    "fch_vispar_r":  (["b2tfch_fchvispar_ay", "b2tfnb_fchvispar_by"], 2, 1.0),
    "fch_visper_th": (["b2tfch_fchvisper_ax", "b2tfnb_fchvisper_bx"], 1, 1.0),
    "fch_visper_r":  (["b2tfch_fchvisper_ay", "b2tfnb_fchvisper_by"], 2, 1.0),
    "fch_visq_th":  (["b2tfch_fchvisq_ax", "b2tfnb_fchvisq_bx"], 1, 1.0),
    "fch_visq_r":   (["b2tfch_fchvisq_ay", "b2tfnb_fchvisq_by"], 2, 1.0),
    "fch_in_th":    (["b2tfch_fchin_ax", "b2tfnb_fchin_bx"], 1, 1.0),
    "fch_in_r":     (["b2tfch_fchin_ay", "b2tfnb_fchin_by"], 2, 1.0),
    "vel_ExB_th":   (["b2tfnb_vbecrbx"], 1, 1.0),
    "vel_ExB_r":    (["b2tfnb_vbecrby"], 2, 1.0),
    "cetaahz_clLucFlim_th": (["b2trcl_luciani_fllim_cvsahzx"], 1, 1.0, "fc_qgam_cos"),
    "cetaa_AN_th":  (["b2trno_cvsax"], 1, 1.0, "fc_qgam_cos"),
    "cetaa_AN_r":   (["b2trno_cvsay"], 2, 1.0, "fc_qgam_cos"),
    "cetaahz_AN_th": (["b2trno_cvsahzx"], 1, 1.0, "fc_qgam_cos"),
    "cetaahz_AN_r":  (["b2trno_cvsahzy"], 2, 1.0, "fc_qgam_cos"),
}

# Species-indexed cell variables: base_names, sign
_CELL_SPECIES_VARS: dict[str, tuple] = {
    "na":     (["b2npc11_na", "b2npco_na"], 1.0),
    "ua":     (["b2npmo_ua"], 1.0),
    "dnadt":  (["b2npc11_dnadt", "b2npco_dnadt"], 1.0),
    "sna":    (["b2npc11_sna", "b2npco_sna"], 1.0),
    "resco":  (["b2npc11_resco", "b2npco_resco"], 1.0),
    "smo":    (["b2npmo_smb"], 1.0),
    "resmo":  (["b2npmo_resmo"], 1.0),
    "taua":   (["b2tqca_taua"], 1.0),
    "kbnrgy": (["b2tfnb_kbnrgy"], 1.0),
    "sna_cx": (["b2stcx_sna_"], 1.0),
    "sna_ion": (["b2stel_sna_ion"], 1.0),
    "sna_rec": (["b2stel_sna_rec"], 1.0),
    "sna_BC": (["b2stbc_phys_sna"], 1.0),
    "vel_BgradB_th": (["b2tfnb_vbdiax"], 1.0),
    "vel_BgradB_r":  (["b2tfnb_vbdiay"], 1.0),
    "vel_dia_th": (["b2tfnb_wbdiax"], 1.0),
    "vel_dia_r":  (["b2tfnb_wbdiay"], 1.0),
    "dgradpbx": (["b2tfnb_dgradpbx"], 1.0),
    "dgradpby": (["b2tfnb_dgradpby"], 1.0),
    "dPat_2diagradnax": (["b2tfnb_dPat_2diagradnax"], 1.0),
    "dPat_2diagradnay": (["b2tfnb_dPat_2diagradnay"], 1.0),
    "smo_AN":   (["b2npmo_smoan"], 1.0),
    "smo_cf":   (["b2npmo_smocf"], 1.0),
    "smo_frea": (["b2npmo_smofrea"], 1.0),
    "smo_fria": (["b2npmo_smofria"], 1.0),
    "smo_tfea": (["b2npmo_smotfea"], 1.0),
    "smo_tfia": (["b2npmo_smotfia"], 1.0),
    "smo_vispar": (["b2npmo_smovv"], 1.0),
    "smo_visq": (["b2npmo_smovh"], 1.0),
    "smovi":    (["b2npmo_smovi"], 1.0),
    "smogp":    (["b2sigp_smogp"], 1.0),
    "smo_gradpi": (["b2sigp_smogpi"], 1.0),
    "smo_gradpo": (["b2sigp_smogpo"], 1.0),
    "smo_cx":   (["b2stcx_smq"], 1.0),
    "smo_BC":   (["b2stbc_phys_smo"], 1.0),
    "smo_ion":  (["b2stel_smq_ion"], 1.0),
    "smo_rec":  (["b2stel_smq_rec"], 1.0),
    "she_radbrm": (["b2stel_rqbrm"], 1.0),
    "she_radlin": (["b2stel_rqrad"], 1.0),
    "sna_eir":  (["b2stbr_sna_eir"], 1.0),
    "smo_eir":  (["b2stbr_smo_eir"], 1.0),
}

# Non-species face variables: base_names (fallback), dim, sign, divisor
_FACE_SCALAR_VARS: dict[str, tuple] = {
    "fch_th": (["b2npp7_fchx"], 1, 1.0, None),
    "fch_r":  (["b2npp7_fchy"], 2, 1.0, None),
    "fch_par_th": (["b2tfch_fch_px"], 1, 1.0, None),
    "fch_nuBgradB_th": (["b2tfch_fchdiax"], 1, 1.0, None),
    "fch_nuBgradB_r":  (["b2tfch_fchdiay"], 2, 1.0, None),
    "fch_stoch_r": (["b2tfch_fchstochy"], 2, 1.0, None),
    "fhe_mdf_th": (["b2nph9_fhex"], 1, 1.0, None),
    "fhe_mdf_r":  (["b2nph9_fhey"], 2, 1.0, None),
    "fhi_mdf_th": (["b2nph9_fhix"], 1, 1.0, None),
    "fhi_mdf_r":  (["b2nph9_fhiy"], 2, 1.0, None),
    "fhe_th": (["b2tfhe_fhe_no_mdfx"], 1, 1.0, None),
    "fhe_r":  (["b2tfhe_fhe_no_mdfy"], 2, 1.0, None),
    "fhi_th": (["b2tfhe_fhi_no_mdfx"], 1, 1.0, None),
    "fhi_r":  (["b2tfhe_fhi_no_mdfy"], 2, 1.0, None),
    "fhe_dia_mdf_th": (["b2tfhe_fhePSchx"], 1, -1.0, None),
    "fhe_dia_mdf_r":  (["b2tfhe_fhePSchy"], 2, -1.0, None),
    "fhi_dia_mdf_th": (["b2tfhi_fhiPSchx"], 1, -1.0, None),
    "fhi_dia_mdf_r":  (["b2tfhi_fhiPSchy"], 2, -1.0, None),
    "fhe_flo_th": (["b2tfhe_qe_32x"], 1, 1.0, None),
    "fhe_flo_r":  (["b2tfhe_qe_32y"], 2, 1.0, None),
    "fhi_flo_th": (["b2tfhi_qi_32x"], 1, 1.0, None),
    "fhi_flo_r":  (["b2tfhi_qi_32y"], 2, 1.0, None),
    "fhe_cond_th": (["b2tfhe_qe_ke_gTx"], 1, 1.0, None),
    "fhe_cond_r":  (["b2tfhe_qe_ke_gTy"], 2, 1.0, None),
    "fhi_cond_th": (["b2tfhi_qi_ki_gTx"], 1, 1.0, None),
    "fhi_cond_r":  (["b2tfhi_qi_ki_gTy"], 2, 1.0, None),
    "fhe_nuBgradB_th": (["b2tfhe_qediax"], 1, 1.0, None),
    "fhe_nuBgradB_r":  (["b2tfhe_qediay"], 2, 1.0, None),
    "fhe_alphaEhat_th": (["b2tfhe_qe_alphaTehx"], 1, 1.0, None),
    "fhe_alphaEhat_r":  (["b2tfhe_qe_alphaTehy"], 2, 1.0, None),
    "fht_th": (["b2news_fhtx"], 1, 1.0, None),
    "fht_r":  (["b2news_fhty"], 2, 1.0, None),
    "calf_cl_th": (["b2trcl_calfx"], 1, 1.0, "fc_qgam_cos"),
    "ckappae_cl_th": (["b2trcl_chcex"], 1, 1.0, "fc_qgam_cos"),
    "ckappai_cl_th": (["b2trcl_chcix"], 1, 1.0, "fc_qgam_cos"),
    "csig_cl_th": (["b2trcl_csigx"], 1, 1.0, "fc_qgam_cos"),
    "calf_clLuc_th": (["b2trcl_luciani_calfx"], 1, 1.0, "fc_qgam_cos"),
    "ckappae_clLuc_th": (["b2trcl_luciani_chcex"], 1, 1.0, "fc_qgam_cos"),
    "ckappai_clLuc_th": (["b2trcl_luciani_chcix"], 1, 1.0, "fc_qgam_cos"),
    "csig_clLuc_th": (["b2trcl_luciani_csigx"], 1, 1.0, "fc_qgam_cos"),
    "calf_clLucFlim_th": (["b2trcl_luciani_fllim_calfx"], 1, 1.0, "fc_qgam_cos"),
    "ckappae_clLucFlim_th": (["b2trcl_luciani_fllim_chcex"], 1, 1.0, "fc_qgam_cos"),
    "ckappai_clLucFlim_th": (["b2trcl_luciani_fllim_chcix"], 1, 1.0, "fc_qgam_cos"),
    "csig_clLucFlim_th": (["b2trcl_luciani_fllim_csigx"], 1, 1.0, "fc_qgam_cos"),
}

# Non-species cell variables
_CELL_SCALAR_VARS: dict[str, tuple] = {
    "te": (["b2nph9_te", "b2nph9_te_inp"], 1.0),
    "ti": (["b2nph9_ti", "b2nph9_ti_inp"], 1.0),
    "ne": (["ne"], 1.0),
    "po": (["b2npp7_po"], 1.0),
    "ue": (["b2npmo_ue"], 1.0),
    "Zeff": (["Zeff"], 1.0),
    "taue": (["b2tqce_taue"], 1.0),
    "lnlam": (["b2tqca_lnlam"], 1.0),
    "alf0": (["b2tqna_alf0"], 1.0),
    "hce": (["b2tqna_hce0"], 1.0),
    "hci": (["b2tqna_hcib001"], 1.0),
    "csig_AN_r": (["b2tqna_sig0", "b2tfch_csig_any", "b2trno_csigy"], 1.0),
    "she": (["b2nph9_she"], 1.0),
    "shi": (["b2nph9_shi"], 1.0),
    "she_ei": (["b2nph9_shei"], 1.0),
    "she_rad": (["b2stel_she_rad"], 1.0),
    "she_dd": (["b2sihs_shedd"], 1.0),
    "she_du": (["b2sihs_shedu"], 1.0),
    "she_fr": (["b2sihs_shefr"], 1.0),
    "she_BC": (["b2stbc_phys_she"], 1.0),
    "she_st": (["b2srst_shest"], 1.0),
    "she_eir": (["b2stbr_she_eir"], 1.0),
    "shi_dd": (["b2sihs_shidd"], 1.0),
    "shi_du": (["b2sihs_shidu"], 1.0),
    "shi_fr": (["b2sihs_shifr"], 1.0),
    "shi_visan": (["b2sihs_shiva"], 1.0),
    "shi_viscl": (["b2sihs_shivc"], 1.0),
    "shi_BC": (["b2stbc_phys_shi"], 1.0),
    "shi_st": (["b2srst_shist"], 1.0),
    "shi_cx": (["b2stcx_shi"], 1.0),
    "shi_eir": (["b2stbr_shi_eir"], 1.0),
}


def _ws_dim(watch, dim):
    return watch.grid.n_faces if dim in (1, 2) else watch.grid.n_cells


def build_workspace(watch) -> dict:
    """Assemble all MATLAB-style workspace variables (cached on watch).

    Returns {name: np.ndarray} with MATLAB semantics:
      - face variables are (n_faces, ns) or (n_faces,)
      - cell variables are (n_cells, ns) or (n_cells,)
    """
    cached = getattr(watch, "_matlab_ws", None)
    if cached is not None:
        return cached

    ws: dict = {}
    ns = n_species(watch)

    for name, spec in _FACE_SPECIES_VARS.items():
        if len(spec) == 3:
            bases, dim, sign = spec
            divisor = None
        else:
            bases, dim, sign, divisor = spec
        ws[name] = get_species_var(watch, base_names=bases, dim=dim,
                                   sign=sign, n_spec=ns, divisor=divisor)

    for name, (bases, sign) in _CELL_SPECIES_VARS.items():
        ws[name] = get_species_var(watch, base_names=bases, dim=None,
                                   sign=sign, n_spec=ns)

    for name, (bases, dim, sign, divisor) in _FACE_SCALAR_VARS.items():
        if divisor == "fc_qgam_cos":
            ws[name] = get_scalar_var(watch, bases, dim=dim, sign=sign,
                                      divisor="fc_qgam_cos")
        else:
            ws[name] = get_scalar_var(watch, bases, dim=dim, sign=sign,
                                      divisor=divisor)

    for name, (bases, sign) in _CELL_SCALAR_VARS.items():
        ws[name] = get_scalar_var(watch, bases, sign=sign)

    # MATLAB read_geometry.m line 822: fcHz = intface(cvHz, 1, 'hc')
    from solps_analysis.core.operators import intface as _intface_op
    if watch.grid.cv_hz is not None and watch.grid.fc_cv is not None:
        ws["fc_hz"] = _intface_op(watch.grid, watch.grid.cv_hz, 1, "hc")
    else:
        ws["fc_hz"] = np.ones(watch.grid.n_faces)

    # MATLAB read_data_3x.m lines 873-876:
    # fna_mo_vis_th = flubvx ./ fcHz ./ ams ./ mp
    if "fna_mo_vis_th" in ws and ns > 0:
        fc_hz = ws.get("fc_hz")
        ams = species_am(watch)
        if fc_hz is not None and ams.size == ns:
            denom = fc_hz[:, None] * ams[None, :] * MP
            ws["fna_mo_vis_th"] = ws["fna_mo_vis_th"] / np.where(denom == 0, np.nan, denom)
            ws["fna_mo_vis_th"] = np.nan_to_num(ws["fna_mo_vis_th"], nan=0.0,
                                                 posinf=0.0, neginf=0.0)

    # MATLAB leaves these as zeros (no poloidal/radial counterpart):
    # fna_nuAN_th, fna_RhieChow_r, fna_mo_vis_r (read_data_3x.m)
    n_fc = watch.grid.n_faces
    ws.setdefault("fna_nuAN_th", np.zeros((n_fc, ns)))
    ws.setdefault("fna_RhieChow_r", np.zeros((n_fc, ns)))
    ws.setdefault("fna_mo_vis_r", np.zeros((n_fc, ns)))
    ws.setdefault("fch_par_r", np.zeros(n_fc))

    # MATLAB fallback (read_data_3x.m lines 902-919): if b2urmo_etaPat_uax/y
    # missing, fmo_flo = fmo - fmo_cond
    if "fmo_flo_th" in ws and "fmo_th" in ws and "fmo_vis_th" in ws:
        if not np.any(ws["fmo_flo_th"]):
            ws["fmo_flo_th"] = ws["fmo_th"] - ws["fmo_vis_th"]
    if "fmo_flo_r" in ws and "fmo_r" in ws and "fmo_vis_r" in ws:
        if not np.any(ws["fmo_flo_r"]):
            ws["fmo_flo_r"] = ws["fmo_r"] - ws["fmo_vis_r"]

    # MATLAB: fna_mo_th = fna_mo_th + fna_mo_vis_th (lines 881-882)
    if "fna_mo_th" in ws and "fna_mo_vis_th" in ws:
        ws["fna_mo_th"] = ws["fna_mo_th"] + ws["fna_mo_vis_th"]

    # MATLAB: fna_Dgradn = fna_cond (lines 795-796)
    if "fna_Dgradn_th" in ws and "fna_cond_th" in ws:
        ws["fna_Dgradn_th"] = ws["fna_cond_th"]
    if "fna_Dgradn_r" in ws and "fna_cond_r" in ws:
        ws["fna_Dgradn_r"] = ws["fna_cond_r"]

    # MATLAB read_data_3x.m: te = te ./ qe  (b2nph9_te is in Joules)
    QE = 1.602176634e-19
    if "te" in ws and ws["te"].size:
        ws["te"] = ws["te"] / QE
    if "ti" in ws and ws["ti"].size:
        ws["ti"] = ws["ti"] / QE

    watch._matlab_ws = ws
    return ws

