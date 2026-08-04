"""Variable catalog for SOLPS-ITER output files.

Determines how a structured .dat file should be mapped to the
unstructured-style 1D arrays:
  - "cell"   → via imap_cv  (st_us_transform)
  - "face_x" → via imap_fcx (st_us_transform_fc(input, 1))
  - "face_y" → via imap_fcy (st_us_transform_fc(input, 2))

The mapping mirrors MATLAB read_data_3x.m.  Names are normalized the same
way as scan_output_directory: b2tfch__fchdiax.dat → "b2tfch_fchdiax".
"""

from __future__ import annotations

import re

# {normalized_base_name_prefix: "face_x" | "face_y"}
FACE_PREFIXES: dict[str, str] = {
    # --- particle fluxes (fna) ---
    "b2npc11_fnax": "face_x",
    "b2npc11_fnay": "face_y",
    "b2npc9_fnax": "face_x",
    "b2npc9_fnay": "face_y",
    "b2npc7_fnax": "face_x",
    "b2npc7_fnay": "face_y",
    "b2npco_fnax": "face_x",
    "b2npco_fnay": "face_y",
    "b2tfnb_fnbx": "face_x",
    "b2tfnb_fnby": "face_y",
    "b2tfnb_fnb_mdfx": "face_x",
    "b2tfnb_fnb_mdfy": "face_y",
    "b2tfnb_fnb_32x": "face_x",
    "b2tfnb_fnb_32y": "face_y",
    "b2tfnb_fnb_hex": "face_x",
    "b2tfnb_fnb_hey": "face_y",
    "b2tfnb_fnb_fcorx": "face_x",
    "b2tfnb_fnb_fcory": "face_y",
    "b2tfnb_bxuanax": "face_x",
    "b2tfnb_vadianax": "face_x",
    "b2tfnb_vadianay": "face_y",
    "b2tfnb_vaecrbnax": "face_x",
    "b2tfnb_vaecrbnay": "face_y",
    "b2tfnb_fnbPSchx": "face_x",
    "b2tfnb_fnbPSchy": "face_y",
    "b2tfnb_cvlbnay": "face_y",
    "b2tfnb_dpccornax": "face_x",
    "b2tfnb_dPat_mdf_gradnax": "face_x",
    "b2tfnb_dPat_mdf_gradnay": "face_y",
    # --- momentum fluxes (fmo) ---
    "b2npmo_fmox": "face_x",
    "b2npmo_fmoy": "face_y",
    "b2npmo_flubvx": "face_x",
    "b2urmo_etaPat_graduax": "face_x",
    "b2urmo_etaPat_graduay": "face_y",
    "b2urmo_etaPat_uax": "face_x",
    "b2urmo_etaPat_uay": "face_y",
    # --- currents (fch) ---
    "b2npp7_fchx": "face_x",
    "b2npp7_fchy": "face_y",
    "b2tfch_fch_px": "face_x",
    "b2tfch_fchanml_ax": "face_x",
    "b2tfch_fchanml_ay": "face_y",
    "b2tfch_fchdiax": "face_x",
    "b2tfch_fchdiay": "face_y",
    "b2tfch_fchinert_ax": "face_x",
    "b2tfch_fchinert_ay": "face_y",
    "b2tfch_fchvispar_ax": "face_x",
    "b2tfch_fchvispar_ay": "face_y",
    "b2tfch_fchvisper_ax": "face_x",
    "b2tfch_fchvisper_ay": "face_y",
    "b2tfch_fchvisq_ax": "face_x",
    "b2tfch_fchvisq_ay": "face_y",
    "b2tfch_fchin_ax": "face_x",
    "b2tfch_fchin_ay": "face_y",
    "b2tfch_fchstochy": "face_y",
    "b2tfch_csig_anx": "face_x",
    "b2tfch_csig_any": "face_y",
    "b2tfnb_fchanml_bx": "face_x",
    "b2tfnb_fchanml_by": "face_y",
    "b2tfnb_fchinert_bx": "face_x",
    "b2tfnb_fchinert_by": "face_y",
    "b2tfnb_fchvispar_bx": "face_x",
    "b2tfnb_fchvispar_by": "face_y",
    "b2tfnb_fchvisper_bx": "face_x",
    "b2tfnb_fchvisper_by": "face_y",
    "b2tfnb_fchvisq_bx": "face_x",
    "b2tfnb_fchvisq_by": "face_y",
    "b2tfnb_fchin_bx": "face_x",
    "b2tfnb_fchin_by": "face_y",
    # --- heat fluxes (fhe/fhi/fht) ---
    "b2nph9_fhex": "face_x",
    "b2nph9_fhey": "face_y",
    "b2nph9_fhix": "face_x",
    "b2nph9_fhiy": "face_y",
    "b2tfhe_fhe_no_mdfx": "face_x",
    "b2tfhe_fhe_no_mdfy": "face_y",
    "b2tfhe_fhi_no_mdfx": "face_x",
    "b2tfhe_fhi_no_mdfy": "face_y",
    "b2tfhe_fhePSchx": "face_x",
    "b2tfhe_fhePSchy": "face_y",
    "b2tfhi_fhiPSchx": "face_x",
    "b2tfhi_fhiPSchy": "face_y",
    "b2tfhe_qe_32x": "face_x",
    "b2tfhe_qe_32y": "face_y",
    "b2tfhi_qi_32x": "face_x",
    "b2tfhi_qi_32y": "face_y",
    "b2tfhe_qe_ke_gTx": "face_x",
    "b2tfhe_qe_ke_gTy": "face_y",
    "b2tfhi_qi_ki_gTx": "face_x",
    "b2tfhi_qi_ki_gTy": "face_y",
    "b2tfhe_qediax": "face_x",
    "b2tfhe_qediay": "face_y",
    "b2tfhe_qe_alphaTehx": "face_x",
    "b2tfhe_qe_alphaTehy": "face_y",
    "b2news_fhtx": "face_x",
    "b2news_fhty": "face_y",
    # --- velocities ---
    "b2tfnb_vbecrbx": "face_x",
    "b2tfnb_vbecrby": "face_y",
    "b2tfnb_vbdiax": "face_x",
    "b2tfnb_vbdiay": "face_y",
    "b2tfnb_wbdiax": "face_x",
    "b2tfnb_wbdiay": "face_y",
    # --- transport coefficients (classical/Luciani/anomalous) ---
    "b2trcl_calfx": "face_x",
    "b2trcl_chcex": "face_x",
    "b2trcl_chcix": "face_x",
    "b2trcl_csigx": "face_x",
    "b2trcl_cvsahzx": "face_x",
    "b2trcl_cvsax": "face_x",
    "b2trcl_luciani_calfx": "face_x",
    "b2trcl_luciani_chcex": "face_x",
    "b2trcl_luciani_chcix": "face_x",
    "b2trcl_luciani_csigx": "face_x",
    "b2trcl_luciani_cvsahzx": "face_x",
    "b2trcl_luciani_fllim_calfx": "face_x",
    "b2trcl_luciani_fllim_chcex": "face_x",
    "b2trcl_luciani_fllim_chcix": "face_x",
    "b2trcl_luciani_fllim_csigx": "face_x",
    "b2trcl_luciani_fllim_cvsahzx": "face_x",
    "b2trno_cdnax": "face_x",
    "b2trno_cdnay": "face_y",
    "b2trno_cdpax": "face_x",
    "b2trno_cdpay": "face_y",
    "b2trno_chcex": "face_x",
    "b2trno_chcey": "face_y",
    "b2trno_chcix": "face_x",
    "b2trno_chciy": "face_y",
    "b2trno_csigx": "face_x",
    "b2trno_csigy": "face_y",
    "b2trno_cvsax": "face_x",
    "b2trno_cvsay": "face_y",
    "b2trno_cvsahzx": "face_x",
    "b2trno_cvsahzy": "face_y",
}

# Longer prefixes must win over shorter ones (e.g. b2tfnb_fnb_mdfx
# vs b2tfnb_fnbx).  Sort by length descending for prefix matching.
_SORTED_PREFIXES = sorted(FACE_PREFIXES.items(), key=lambda kv: -len(kv[0]))


def normalize_dat_name(filename: str) -> str:
    """Normalize a .dat filename to the catalog key format.

    b2tfch__fchdiax001.dat → b2tfch_fchdiax_001
    b2news__fhtx.dat       → b2news_fhtx
    """
    name = filename
    if name.endswith(".dat"):
        name = name[:-4]
    if name.startswith("__1__"):
        return name
    name = re.sub(r"__", "_", name)
    return name


def _catalog_transform(file_path: str) -> str:
    """Return 'cell', 'face_x' or 'face_y' for a .dat file path."""
    import os

    base = normalize_dat_name(os.path.basename(str(file_path)))
    # strip trailing species index _NNN
    base_no_idx = re.sub(r"_\d{3}$", "", base)
    for prefix, kind in _SORTED_PREFIXES:
        if base_no_idx.startswith(prefix):
            return kind
    return "cell"
