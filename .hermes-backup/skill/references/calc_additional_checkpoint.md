# calc_additional.m — Checkpoint

**Stop point**: end of EIRENE main loop, line 170.

## Already implemented (lines 1-170)

- Lines 1-29: header comments (documentation-only)
- Lines 30-37: skip logic (mtime check — not relevant for Python)
- Lines 38-170: EIRENE main section
  - Line 43-45: P_main_b25 (simple pressure product)
  - Lines 48-170: main loop over species (ia=1:nsorts)
    - atm2mol mapping, fna_th/r from EIRENE data (fort.44 pfluxa/rfluxa
      for structured, fort.46 vxdena+pux/puy for unstructured)
    - **na overwrite**: na[:, neutral_col] = dab2 + molA * dmb2 ✓
    - **Boundary cells**: copy from nearest interior neighbour ✓
    - **Triangle accumulators**: pdena_tot, edena_tot, ndena_tot,
      gxdena_tot, gydena_tot, gzdena_tot
    - **Temperatures**: tdena, tdena_tot
    - **Neutral pressure**: P_neut

  All quantities registered as `@quantity` in `eirene.py`:
  - `eirene_na` ✓
  - `eirene_tot_flux_th` ✓
  - `eirene_tot_flux_r` ✓
  - `eirene_tdena` ✓
  - `eirene_tdena_tot` ✓
  - `eirene_P_neut` ✓

## Still to implement (lines 171-500+)

### P_pfr (lines 171-177)
Average neutral pressure in PFR region:
```matlab
tmp = neut.dab2(:,1)*neut.tab2(:,1) + neut.dmb2(:,1)*neut.tmb2(:,1);
P_pfr = sum(tmp(user_params.pfr_cvs,1).*user_params.pr_weight) ./ sum(user_params.pr_weight);
```
Requires `user_params.pfr_cvs` from b2.user.parameters.

### Recycling / wall data (lines 187-341)
- fcLbl → nsts mapping (neutral station tracking)
- fn_sput_wall, fh_nutpr_th — sputtering and neutral energy fluxes
- wld (wall data) recycling: sna modification for boundary cells
- pumped_flux

### Face interpolation (lines 349-370)
tef, tif, nef, naf, uaf, uef, pof via `intface()`. Our `cell_to_face()`
in `interpolate.py` already implements this — just need to compute and store.

### Divergence operators (lines 372-476)
**~30 operations × 9 species**. For each species:
- div_fna_mdf, div_fna_mdf_th/r, div_fna_flo_th/r, div_fna_Dgradn_th/r
- div_fmo, div_fmo_flo, div_fmo_vis, div_fmo_viscurv, div_fmo_vis_BgradB
- div_fna_fha, div_fna_dia_mdf, div_fna_nuExB, div_fna_nuBgradB
- div_fna_nupar, div_fna_nuAN, div_fna_RhieChow, div_fna_curr
- ua_eff_th/r, ua_diff_th/r, div_ua, div_ua_eff, div_ua_ExB, div_ua_diff
- smo_vis_tot, smo_vis_tot_th, smo_vis_tot_r

All use `div_us(gmtry, [f_th f_r])` — we need a `divergence()` function.

### Remaining (lines 477-1762)
- gradPi_th/r, div_ue, div_fch, div_fhi, div_fhe
- snas, snas_bound_reg
- Many more derived quantities

## Strategy for remaining work

The divergence operators are repetitive (same fn × 9 species × 30 components).
Consider generating them programmatically via loops rather than porting
each one individually. Each follows:
```python
result[:, is] = divergence(grid, [f_th[:, is], f_r[:, is]])
```
