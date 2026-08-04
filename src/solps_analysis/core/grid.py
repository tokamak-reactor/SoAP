"""GridTopology — handles both structured and unstructured SOLPS grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GridTopology:
    """Unified grid topology for SOLPS-ITER.

    Handles both structured (3.0.x) and unstructured (3.2.x) grids.
    Internal representation is always unstructured (1D cell-indexed),
    but structured grids retain their (i,j) indices via imap fields.
    """

    # --- Dimensions ---
    n_cells: int          # nCv — total number of control volumes
    n_faces: int          # nFc — total number of faces
    n_vertices: int       # nVx — total number of vertices
    n_flux_surfaces: int  # nFs — number of flux surfaces
    n_flux_tubes: int     # nFt — number of flux tubes
    n_core_cells: int = 0 # nCi — cells in core region (B2 computational)

    # --- Grid type ---
    is_structured: bool = False
    nx: int = 0
    ny: int = 0
    nncut: int = 0

    # --- Cell coordinates ---
    cv_x: np.ndarray | None = None   # (n_cells,) X coordinate of cell center [m]
    cv_y: np.ndarray | None = None   # (n_cells,) Y coordinate of cell center [m]
    cv_r: np.ndarray | None = None   # (n_cells,) radial coordinate [m] (computed)
    cv_theta: np.ndarray | None = None  # (n_cells,) poloidal coord [m] (computed)
    cv_vol: np.ndarray | None = None # (n_cells,) cell volume [m³]
    cv_sz: np.ndarray | None = None  # (n_cells,) area perp. to z [m²]
    cv_hz: np.ndarray | None = None  # (n_cells,) length in z [m]

    # --- Magnetic field at cells ---
    cv_bb: np.ndarray | None = None  # (n_cells, 4) B_pol, B_rad, Bz, |B|

    # --- Face coordinates/geometry ---
    fc_x: np.ndarray | None = None   # (n_faces,) X coordinate of face center
    fc_y: np.ndarray | None = None   # (n_faces,) Y coordinate of face center
    fc_s: np.ndarray | None = None   # (n_faces,) face area
    fc_hc: np.ndarray | None = None  # (n_faces, 2) connector lengths
    fc_ht: np.ndarray | None = None  # (n_faces,) tangential length
    fc_bb: np.ndarray | None = None  # (n_faces, 4) B_pol, B_rad, Bz, |B|
    fc_lbl: np.ndarray | None = None # (n_faces,) boundary label
    fc_reg: np.ndarray | None = None # (n_faces,) region number

    # --- Vertex coordinates ---
    vx_x: np.ndarray | None = None   # (n_vertices,) X coordinate
    vx_y: np.ndarray | None = None   # (n_vertices,) Y coordinate
    vx_fpsi: np.ndarray | None = None  # (n_vertices,) poloidal flux

    # --- Connectivity ---
    cv_fc_p: np.ndarray | None = None  # (n_cells, 2) start index + count into cv_fc
    cv_fc: np.ndarray | None = None    # flat face list per cell
    fc_cv: np.ndarray | None = None    # (n_faces, 2) neighboring cells
    fc_vx: np.ndarray | None = None    # (n_faces, 2) vertices of each face
    cv_ft: np.ndarray | None = None    # (n_cells,) flux tube index per cell

    # --- Flux surface / flux tube data ---
    ft_cv_p: np.ndarray | None = None  # (n_flux_tubes, 2)
    ft_cv: np.ndarray | None = None    # flat cell list per flux tube
    ft_fc_p: np.ndarray | None = None  # (n_flux_tubes, 2)
    ft_fc: np.ndarray | None = None    # flat face list per flux tube
    ft_reg: np.ndarray | None = None   # (n_flux_tubes,) region numbers
    fs_fc_p: np.ndarray | None = None  # (n_flux_surfaces, 2)
    fs_fc: np.ndarray | None = None    # flat face list per flux surface
    fs_psi: np.ndarray | None = None   # (n_flux_surfaces,) poloidal flux values
    cv_reg: np.ndarray | None = None   # (n_cells,) region numbers
    cv_conn: np.ndarray | None = None  # (n_cells,) connection length

    # --- Additional connectivity (loaded from unstructured file) ---
    vx_fc_p: np.ndarray | None = None  # (n_vertices, 2) pointer to vx_fc
    vx_fc: np.ndarray | None = None    # flat face list per vertex
    vx_cv_p: np.ndarray | None = None  # (n_vertices, 2) pointer to vx_cv
    vx_cv: np.ndarray | None = None    # flat cell list per vertex
    cv_vx_p: np.ndarray | None = None  # (n_cells, 2) pointer to cv_vx
    cv_vx: np.ndarray | None = None    # flat vertex list per cell

    # --- Additional geometry arrays ---
    fc_qalf: np.ndarray | None = None  # (n_faces, 2) cos/sin of alpha
    fc_qgam: np.ndarray | None = None  # (n_faces, 2) cos/sin of gamma
    fc_qbet: np.ndarray | None = None  # (n_faces, 2) cos/sin of beta
    fc_pbs: np.ndarray | None = None   # (n_faces,) magnetic contact area
    fc_hz: np.ndarray | None = None    # (n_faces,) toroidal metric hz at faces
    cv_eb: np.ndarray | None = None    # (n_cells, 3) unit vector along B
    cv_qgam: np.ndarray | None = None  # (n_cells, 2) cos/sin of gamma
    intcell_p: np.ndarray | None = None  # interpolation factors (poloidal)
    intcell_r: np.ndarray | None = None  # interpolation factors (radial)

    # --- Derived arrays (computed in regions module) ---
    fc_fs: np.ndarray | None = None    # (n_faces,) flux surface index (computed)
    fc_ft: np.ndarray | None = None    # (n_faces,) flux tube index (computed)
    fc_theta: np.ndarray | None = None # (n_faces,) poloidal coordinate (computed)
    fc_r: np.ndarray | None = None     # (n_faces,) radial coordinate (computed)
    cv_theta_n: np.ndarray | None = None  # (n_cells,) normalized poloidal coord
    fc_theta_n: np.ndarray | None = None  # (n_faces,) normalized poloidal coord
    ft_conn: np.ndarray | None = None  # (n_flux_tubes,) connection length
    cv_lbl_len: np.ndarray | None = None  # (n_cells,) boundary distance
    fc_lbl_len: np.ndarray | None = None  # (n_faces,) boundary distance
    fc_lbl_group: np.ndarray | None = None  # (n_faces,) grouped boundary label
    fc_lbl_group_len: np.ndarray | None = None  # (n_faces,) grouped boundary distance

    # --- Derived regions (computed in regions module) ---
    xp_vx: np.ndarray | None = None     # X-point vertex indices
    fs_sep: np.ndarray | None = None    # flux surfaces at separatrix
    fs_sep2: np.ndarray | None = None   # second separatrix flux surfaces (diverted)
    sep_vx: np.ndarray | None = None    # separatrix vertices
    sep_fc: np.ndarray | None = None    # separatrix faces
    sep_fc_sol: np.ndarray | None = None  # separatrix faces at SOL boundary
    sep2_vx: np.ndarray | None = None
    sep2_fc: np.ndarray | None = None
    inner_midplane_cells: np.ndarray | None = None
    outer_midplane_cells: np.ndarray | None = None
    fc_inner_midplane: np.ndarray | None = None
    fc_outer_midplane: np.ndarray | None = None
    inner_midplane_cells_sol: np.ndarray | None = None
    outer_midplane_cells_sol: np.ndarray | None = None
    inner_target_cells: np.ndarray | None = None
    outer_target_cells: np.ndarray | None = None
    inner_target_faces: np.ndarray | None = None
    outer_target_faces: np.ndarray | None = None
    inner_top_target_cells: np.ndarray | None = None
    outer_top_target_cells: np.ndarray | None = None
    inner_top_target_faces: np.ndarray | None = None
    outer_top_target_faces: np.ndarray | None = None
    inner_active_target_cells: np.ndarray | None = None
    outer_active_target_cells: np.ndarray | None = None
    inner_active_target_faces: np.ndarray | None = None
    outer_active_target_faces: np.ndarray | None = None
    inner_inactive_target_cells: np.ndarray | None = None
    outer_inactive_target_cells: np.ndarray | None = None
    inner_inactive_target_faces: np.ndarray | None = None
    outer_inactive_target_faces: np.ndarray | None = None
    separatrix_cells: np.ndarray | None = None
    core_sep_faces: np.ndarray | None = None
    cv_or: np.ndarray | None = None    # (n_cells,) orientation
    fc_or: np.ndarray | None = None    # (n_faces,) orientation
    bp_dir: int = 0  # poloidal B direction (±1)
    bt_dir: int = 0  # toroidal B direction (±1)
    wall_faces: np.ndarray | None = None
    wall_cells: np.ndarray | None = None
    wall_cells_vol: np.ndarray | None = None
    wall_cells_len: np.ndarray | None = None
    fcs_boundary: list | None = None
    cvs_boundary: list | None = None
    cvs_close_boundary: list | None = None
    vx_bound_ends: np.ndarray | None = None
    is_cross_outmidpl: np.ndarray | None = None
    is_cross_inmidpl: np.ndarray | None = None
    is_cross_outmidpl_cv: np.ndarray | None = None
    is_cross_inmidpl_cv: np.ndarray | None = None

    # --- Structured grid mapping (only for structured or converted grids) ---
    imap_cv: np.ndarray | None = None  # (nx+2, ny+2) cell index mapping
    imap_fcx: np.ndarray | None = None # (nx+2, ny+2) x-face mapping
    imap_fcy: np.ndarray | None = None # (nx+2, ny+2) y-face mapping
    imap_vx: np.ndarray | None = None  # (nx+2, ny+2) vertex mapping

    # --- Poloidal flux (structured grids: averaged from fpsi corners) ---
    cv_fpsi: np.ndarray | None = None  # (n_cells,) poloidal flux at cell centers
    fc_fpsi: np.ndarray | None = None  # (n_faces,) poloidal flux at face centers

    # --- Derived regions (computed) ---
    inner_midplane_cells: np.ndarray | None = None
    outer_midplane_cells: np.ndarray | None = None
    inner_target_cells: np.ndarray | None = None
    outer_target_cells: np.ndarray | None = None
    inner_target_faces: np.ndarray | None = None
    outer_target_faces: np.ndarray | None = None
    separatrix_cells: np.ndarray | None = None

    # --- Version info ---
    version: str = ""
    version_number: int = 0  # e.g. 302 for 03.002.000

    # --- Species ---
    n_species: int = 0
    species_charges: np.ndarray | None = None  # (n_species,) zamin
    species_charge_max: np.ndarray | None = None  # (n_species,) zamax
    species_n: np.ndarray | None = None  # (n_species,) zn — nuclear charge
    species_mass: np.ndarray | None = None  # (n_species,) am — atomic mass

    # --- Boundary labels from b2.boundary.parameters ---
    boundary_labels: dict[int, str] = field(default_factory=dict)

    @property
    def n_core(self) -> int:
        return self.n_core_cells

    @property
    def n_sol(self) -> int:
        return self.n_cells - self.n_core_cells

    def cell_radial_neighbors(self, cell_index: int) -> list[int]:
        """Get radial neighbor indices for a given cell.
        Uses connectivity to walk radially (poloidally-aligned)."""
        # This will be implemented properly later
        raise NotImplementedError

    def cell_poloidal_neighbors(self, cell_index: int) -> list[int]:
        """Get poloidal neighbor indices for a given cell."""
        raise NotImplementedError

    def __repr__(self) -> str:
        kind = "structured" if self.is_structured else "unstructured"
        return (
            f"GridTopology({kind}, nCv={self.n_cells}, nFc={self.n_faces}, "
            f"nVx={self.n_vertices}, v={self.version})"
        )
