# EIRENE implementation notes

## Structured fort.44: no guard cells

fort.44 data for structured grids has shape (nx, ny, ns) — interior cells ONLY (nx=98, ny=32 for Globus_44644).
The grid's imap_cv has shape (nx+2, ny+2) = (100, 34) — includes guard cells.

**Correct mapping:** `imap_cv[1:-1, 1:-1]` gives interior cell indices. Use this to map
fort.44 data to the full nCv cell array. See `_unpack_eirene_cell_data()` in
`construct/builtin/eirene.py`.

## Boundary cell filling (structured grids)

For structured grids, the face connectivity (cv_fc_p) may point to invalid faces
(fc_cv[0] = [0,0]) for boundary cells. **Do not use face connectivity for boundary
cell neighbour lookup on structured grids.** Use imap-based neighbour lookup instead:

1. Build reverse mapping `cell_index → (i, j)` in imap_cv
2. For each boundary cell (index ≥ nCi), find its (i,j) in the imap
3. Try 4-connected neighbours: `(i-1, j)`, `(i+1, j)`, `(i, j-1)`, `(i, j+1)`
4. Copy EIRENE density from the first interior neighbour (index < nCi) found

This was the key fix that made `eirene_na` match MATLAB 100%. The MATLAB code
uses `gmtry.cvFc(gmtry.cvFcP(iCv,1))` which works for unstructured grids but
may produce face index 0 for structured guard cells.

See `_fill_boundary_eirene()` in `construct/builtin/eirene.py`.

## atm2mol mapping
- fort.44 `atm2mol[ia]` = molecule index for atom ia (1-based, 0 = no molecule)
- fort.44 `molA[mi]` = atoms per molecule (e.g. 2 for D2)
- EIRENE atom → B2 species: use `comp.neutral_index(element)`
