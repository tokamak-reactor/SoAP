"""Region computation for structured SOLPS-ITER grids (3.0.x).

Uses matrix indices and cut variables — no topology needed.
Supports SN (nncut=1), DDN (nncut=2), CDN topologies.
"""

from __future__ import annotations

import numpy as np

from solps_analysis.core.grid import GridTopology


def compute_regions_structured(grid: GridTopology) -> dict:
    """Compute all region data for a structured grid using (i,j) indices.

    Produces the same output attributes as the unstructured compute_all_regions():
      - outer_midplane_cells / inner_midplane_cells
      - cv_inner_tar / cv_outer_tar / fc_inner_tar / fc_outer_tar
      - core_sep_fcs
      - cv_r, cv_theta
      - xp_vx, sep_vx, sep_fc
    """
    if not grid.is_structured or grid.imap_cv is None:
        return {}

    nx, ny = grid.nx, grid.ny
    nncut = grid.nncut
    imap = grid.imap_cv  # (nx+2, ny+2) — 1-based cell indices, 0 = guard
    imap_fcx = grid.imap_fcx
    imap_fcy = grid.imap_fcy

    # --- 1. Compute cut indices from raw data ---
    # leftcut/rightcut/topcut/bottomcut should be stored in grid by now
    leftcut = getattr(grid, '_cut_leftcut', None)
    rightcut = getattr(grid, '_cut_rightcut', None)
    topcut = getattr(grid, '_cut_topcut', None)
    bottomcut = getattr(grid, '_cut_bottomcut', None)

    if leftcut is None:
        return _compute_regions_from_cvreg(grid, imap, nx, ny, nncut)

    # --- 2. Build (iy, ix) → cell index mapping for reverse lookup ---
    # imap[ix, iy] = cell_index (1-based), 0 = no cell
    # We need ix_from_cell and iy_from_cell
    cell_to_ix = np.zeros(grid.n_cells + 1, dtype=np.int32)
    cell_to_iy = np.zeros(grid.n_cells + 1, dtype=np.int32)
    for ix in range(nx + 2):
        for iy in range(ny + 2):
            c = imap[ix, iy]
            if c > 0:
                cell_to_ix[c] = ix
                cell_to_iy[c] = iy

    # --- 3. Determine topology type from nncut ---
    # SN (nncut=1): one X-point, one separatrix
    # DDN (nncut=2): two X-points, two separatrices, both active
    # CDN (nncut=2, connected): single null + secondary X-point
    is_single_null = (nncut == 1)

    # --- 4. Compute zone boundaries ---
    # nc1 = first core-SOL transition on inner side
    # nc2 = end of inner SOL / start of outer PFR
    # nc3 = end of inner PFR / start of outer SOL  
    # nc4 = outer SOL-core transition
    # nsep = first poloidal separatrix crossing (lower)
    # nsep2 = second poloidal separatrix crossing (upper, if applicable)
    # ntt = poloidal cut index

    # For nncut=2 DDN: the two cuts are at (leftcut[0], topcut[0]) and (leftcut[1], topcut[1])
    xp1_ix = leftcut[0] + 2 if len(leftcut) > 0 else 0
    xp1_iy = topcut[0] + 2 if len(topcut) > 0 else 0

    if not is_single_null and len(leftcut) > 1:
        xp2_ix = leftcut[1] + 2
        xp2_iy = topcut[1] + 2
    else:
        xp2_ix, xp2_iy = 0, 0

    # Compute nsep: row where core transitions to SOL on lower side
    # Scan from bottom (iy=2) upward, find where cvReg transitions
    nsep = 0
    nsep2 = 0
    for iy in range(2, ny + 1):
        has_core = False
        has_sol = False
        for ix in range(2, nx + 1):
            c = imap[ix, iy]
            if c == 0:
                continue
            reg = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
            if reg in (1, 5):
                has_core = True
            if reg in (2, 6):
                has_sol = True
        if has_core and has_sol and nsep == 0:
            nsep = iy
        if has_core and has_sol and nsep > 0:
            nsep2 = iy

    if nsep2 == nsep:
        nsep2 = 0

    # --- 5. Find OMP radial column ---
    # Walk radially using rightix array, starting from SOL at the OMP row
    outer_midplane_cells = []
    inner_midplane_cells = []

    # Find OMP row: last row with core+SOL but no PFR
    rightix = grid._rightix if hasattr(grid, '_rightix') else None
    iy_omp = 0
    for iy in range(ny, 1, -1):
        has_core = has_sol = has_pfr = False
        for ix in range(2, nx + 1):
            c = imap[ix, iy]
            if c == 0: continue
            reg = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
            if reg in (1, 5): has_core = True
            elif reg in (2, 6): has_sol = True
            elif reg in (3, 4, 7, 8): has_pfr = True
        if has_core and has_sol and not has_pfr:
            iy_omp = iy
            break

    if iy_omp == 0:
        iy_omp = ny // 2 + 1

    # Walk radially from left to right at iy_omp using rightix
    if rightix is not None:
        # Start from the first cell at iy_omp
        start_ix = 2
        while start_ix <= nx + 1 and imap[start_ix, iy_omp] == 0:
            start_ix += 1
        if start_ix <= nx + 1:
            ix = start_ix
            while ix <= nx + 1:
                c = imap[ix, iy_omp]
                if c > 0:
                    outer_midplane_cells.append(c - 1)
                    inner_midplane_cells.append(c - 1)
                # Move to next cell radially
                if rightix[ix, iy_omp] <= nx + 1:
                    ix = rightix[ix, iy_omp] + 2
                else:
                    break
    else:
        # Fallback: all cells at OMP row
        for ix in range(2, nx + 1):
            c = imap[ix, iy_omp]
            if c > 0:
                outer_midplane_cells.append(c - 1)
                inner_midplane_cells.append(c - 1)

    # --- 6. Find targets ---
    # Inner target: cells at iy=2 (bottom), inner side (left of core)
    # Outer target: cells at iy=2 (bottom), outer side (right of core)
    cv_inner_tar = []
    cv_outer_tar = []
    fc_inner_tar = []
    fc_outer_tar = []

    # Find target at bottom (iy=2)
    for ix in range(2, nx + 1):
        c = imap[ix, 2]
        if c == 0:
            continue
        reg = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
        if reg in (3, 7):  # Inner divertor/PFR
            cv_inner_tar.append(c - 1)
        elif reg in (4, 8):  # Outer divertor/PFR
            cv_outer_tar.append(c - 1)

    # If no targets found at bottom, check top (iy = ny+1)
    if not cv_inner_tar and not cv_outer_tar:
        for ix in range(2, nx + 1):
            c = imap[ix, ny + 1]
            if c == 0:
                continue
            reg = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
            if reg in (3, 7):
                cv_inner_tar.append(c - 1)
            elif reg in (4, 8):
                cv_outer_tar.append(c - 1)

    # --- 7. Separatrix faces ---
    # Find faces between core (reg 1,5) and SOL (reg 2,6)
    core_sep_fcs = []
    for iy in range(2, ny + 1):
        for ix in range(2, nx + 1):
            c = imap[ix, iy]
            if c == 0:
                continue
            # Check right neighbor
            if ix < nx + 1:
                cr = imap[ix + 1, iy]
                if cr > 0:
                    r1 = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
                    r2 = grid.cv_reg[cr - 1] if grid.cv_reg is not None else 0
                    if (r1 in (1, 5) and r2 in (2, 6)) or (r1 in (2, 6) and r2 in (1, 5)):
                        if imap_fcx is not None:
                            fc = imap_fcx[ix + 1, iy]
                            if fc > 0:
                                core_sep_fcs.append(fc - 1)
            # Check top neighbor
            if iy < ny + 1:
                ct = imap[ix, iy + 1]
                if ct > 0:
                    r1 = grid.cv_reg[c - 1] if grid.cv_reg is not None else 0
                    r2 = grid.cv_reg[ct - 1] if grid.cv_reg is not None else 0
                    if (r1 in (1, 5) and r2 in (2, 6)) or (r1 in (2, 6) and r2 in (1, 5)):
                        if imap_fcy is not None:
                            fc = imap_fcy[ix, iy + 1]
                            if fc > 0:
                                core_sep_fcs.append(fc - 1)

    # --- 8. Store results ---
    grid.outer_midplane_cells = np.array(outer_midplane_cells, dtype=np.int32) if outer_midplane_cells else None
    grid.inner_midplane_cells = np.array(inner_midplane_cells, dtype=np.int32) if inner_midplane_cells else None
    grid.cv_inner_tar = np.array(cv_inner_tar, dtype=np.int32) if cv_inner_tar else None
    grid.cv_outer_tar = np.array(cv_outer_tar, dtype=np.int32) if cv_outer_tar else None
    grid.fc_inner_tar = np.array(fc_inner_tar, dtype=np.int32) if fc_inner_tar else None
    grid.fc_outer_tar = np.array(fc_outer_tar, dtype=np.int32) if fc_outer_tar else None
    grid.core_sep_fcs = np.array(core_sep_fcs, dtype=np.int32) if core_sep_fcs else None

    # X-points (at cut locations)
    xp_list = []
    if xp1_ix > 0 and xp1_iy > 0:
        vx_idx = xp1_ix * (ny + 2) + xp1_iy  # approximate vertex index
        xp_list.append(vx_idx)
    if xp2_ix > 0 and xp2_iy > 0:
        vx_idx = xp2_ix * (ny + 2) + xp2_iy
        xp_list.append(vx_idx)
    grid.xp_vx = np.array(xp_list, dtype=np.int32) if xp_list else None

    return _build_regions_dict_structured(grid)


def _compute_regions_from_cvreg(grid, imap, nx, ny, nncut):
    """Fallback: derive regions from cvReg values when cut data unavailable."""
    cv_reg = grid.cv_reg
    if cv_reg is None:
        return {}

    # OMP: cells with cv_reg in (2, 6) at the row just above the divertor
    outer_midplane = []
    inner_midplane = []

    # Find the highest iy that has both core and SOL cells
    sep_iy = 0
    for iy in range(2, ny + 1):
        has_core = False
        has_sol = False
        for ix in range(2, nx + 1):
            c = imap[ix, iy]
            if c == 0:
                continue
            reg = cv_reg[c - 1]
            if reg in (1, 5):
                has_core = True
            if reg in (2, 6):
                has_sol = True
        if has_core and has_sol:
            sep_iy = iy

    if sep_iy > 2:
        omp_row = sep_iy - 1
        for ix in range(2, nx + 1):
            c = imap[ix, omp_row]
            if c == 0:
                continue
            reg = cv_reg[c - 1]
            if reg in (2, 6):
                outer_midplane.append(c - 1)
                inner_midplane.append(c - 1)

    grid.outer_midplane_cells = np.array(outer_midplane, dtype=np.int32) if outer_midplane else None
    grid.inner_midplane_cells = np.array(inner_midplane, dtype=np.int32) if inner_midplane else None
    return _build_regions_dict_structured(grid)


def _build_regions_dict_structured(grid):
    """Build a regions dict from structured grid attributes."""
    attrs = [
        "outer_midplane_cells", "inner_midplane_cells",
        "cv_inner_tar", "cv_outer_tar",
        "fc_inner_tar", "fc_outer_tar",
        "core_sep_fcs", "xp_vx",
    ]
    return {a: getattr(grid, a, None) for a in attrs}
