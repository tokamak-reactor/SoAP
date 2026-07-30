---
name: solps-analysis-project
description: "SoAP — SOLPS-ITER analysis Python package: architecture, methodology, design decisions"
---

# SOLPS Analysis Project (SoAP)

Python package for SOLPS-ITER edge plasma simulation analysis.
Located at `/home/kirill/projects/solps_analysis/`.

## Methodology — how to add new plotting functionality

**DO NOT design in a vacuum.** Follow this sequence:

1. **Consult the MATLAB catalog first.** The `references/matlab-plotting-catalog.md` file
   lists every plotting function from the MATLAB codebase. These represent years of
   accumulated domain knowledge — do not start from zero.

2. **Go function-by-function.** For each MATLAB plotting function:
   - Read its header and implementation
   - Determine: do we have an extractor + plotter that covers its functionality?
   - If yes: create a demo/test, move on
   - If no: add the missing extractor or plotter to match MATLAB's capability
   - If the function only makes sense for unstructured grids (e.g. wall/boundary profiles),
     add a clear error for structured grids rather than doing nothing
   - Log skipped functions (multi-watch/multi-case) in `skipped-matlab-functions.md`

3. **Every MATLAB function needs coverage.** Not a port — a check that the extractor+plotter
   system can produce the same plot. If it can't, add the missing piece.

4. **Extractors separate from plotters.** Each MATLAB plotting function maps to:
   - One or more **extractors** in `extract/profile.py` (mesh data → (x, y) vectors)
   - Possibly a new **plotter** in `plot/` (draws the (x, y) vectors)
   - Never mix extraction into the rendering code

5. **Mode 2 (multi-watch) and Mode 3 (time-dependent) are separate — skip them during
   initial per-function review.** Note skipped functions and come back later.

## Workflow (MATLAB-style, explicit steps)

```python
watch = SolpsWatch.from_directory("/path/to/watch")
watch.compute_regions()       # ← explicit, one call
watch.plot("te_eV", along="omp")  # ← plot many times
```

- `compute_regions()` is NEVER called automatically inside extractors or plotters.
  The user calls it once. Extractors only CHECK if regions exist and raise a clear
  error if not (`_require_regions()`).
- Similarly, variable assembly (na) and composition building are NOT automatic.

## Key design decisions

- **Regions explicit**: Like MATLAB — compute regions once, then plot many times.
- **Extractors separate**: `extract/profile.py` owns the geometry→data logic.
- **Single plotter class per plot type**: Plot1D, Plot2D, PlotWall, PlotMesh.
  One plotter handles `str | list[str]` for both `variable` and `along`.
- **Style presets**: screen (default), journal (17×11 cm, serif, 300dpi),
  presentation (large text). Only valid matplotlib rcParams keys are applied.
- **No auto-magic**: No hidden region computation, no hidden na assembly.

## Plot2D cell vertex reconstruction (three paths)

`_build_cell_vertices(grid)` tries in order:

1. **cv_crn_r / cv_crn_z** — structured grids. Stored by geometry_reader from
   b2fgmtry's crx[:], cry[:] arrays. Exact 4 corners per cell. Fastest and most
   accurate path.

2. **cv_vx_p / cv_vx / vx_x / vx_y** — unstructured grids. Cell→vertex pointer
   arrays + vertex coordinate arrays. Supports arbitrary polygons (3+ vertices).

3. **Fallback averaging** — estimate vertices from neighbouring cell centres.
   Used when neither corner data nor vertex mapping is available.

Do NOT use tricontourf for Plot2D — it fills the central hole (private flux
region / X-point region) because it triangulates the convex hull of cell centres.

## Backup / disaster recovery

Before wiping a machine, changing computers, or any scenario where you might lose the repo:

1. **`git status`** — confirm working tree is clean. Commit or stash anything dirty.
2. **Set up remote** — either via `gh` CLI (`gh auth login` with token or SSH) or direct `git remote add origin git@github.com:<user>/<repo>.git`.
3. **Push all branches**: `git push -u origin master` (and any other branches).
4. **Snapshot memory into the repo**: save relevant Hermes memory entries into `.hermes/memory-solps.md` and the current skill content into `.hermes/skill-solps-analysis-project.md`. Commit these so they travel with the source.
5. **Create a `RECOVERY.md`** in the repo root with instructions: clone URL, required data paths, Hermes skill setup, and any environment prerequisites.
6. **Push again** to include the backup files.

See `references/backup-workflow.md` for a full session transcript of this process.

## Pitfalls

- **`is_structured` flag can be wrong**: Some unstructured grids (Globus-3_WG)
  have `is_structured=True` in data files. Always check `imap_cv.ndim == 2`
  to distinguish structured from unstructured.
- **`cv_r`, `cv_theta` NOT set by geometry reader**. Extractors fall back
  to computing from cv_x, cv_y (sqrt/arctan2).
- **Target cell indices NOT computed for structured grids**. `regions_structured.py`
  doesn't set inner/outer_target_cells or cv_lbl_len. Target extraction on
  structured grids will fail with a clear error.
- **`na` is assembled**, not read directly. Call `watch._assemble_na()` before
  extracting profiles of `na`.
- **`watch.b2_comp` is built by `watch.construct()`** or by calling
  `build_b2_composition(read_b2fstate_full(...))` explicitly.

## Physical constants

- `QE = 1.602176634e-19` (CODATA 2018, exact by SI definition)
- `MP = 1.672621637e-27`
