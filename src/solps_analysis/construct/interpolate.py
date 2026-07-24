"""Interpolation tools for SOLPS-ITER structured/unstructured grids.

Provides face↔cell interpolation matching MATLAB's intface and intcell.
"""

from __future__ import annotations

import numpy as np

from solps_analysis.core.grid import GridTopology


def face_to_cell(grid: GridTopology, fc_data: np.ndarray) -> np.ndarray:
    """Interpolate face-centered data to cell centers.
    
    Uses inverse-distance weighting via fcHc (connector lengths).
    MATLAB equivalent: intcell_us (unstructured) or intcell_P/intcell_R.
    
    Args:
        grid: GridTopology
        fc_data: (n_faces,) or (n_faces, n_species) — data on faces
    
    Returns:
        (n_cells,) or (n_cells, n_species) — data on cell centers
    """
    if fc_data.ndim == 1:
        fc_data = fc_data[:, np.newaxis]
        one_dim = True
    else:
        one_dim = False

    n_cells = grid.n_cells
    n_spec = fc_data.shape[1]
    result = np.zeros((n_cells, n_spec), dtype=np.float64)
    weights = np.zeros((n_cells, n_spec), dtype=np.float64)

    for i_cv in range(n_cells):
        start = grid.cv_fc_p[i_cv, 0]
        count = grid.cv_fc_p[i_cv, 1]
        faces = grid.cv_fc[start:start + count]
        
        for fc in faces:
            # Which side of this face is our cell?
            side = 0 if grid.fc_cv[fc, 0] == i_cv else 1
            w = grid.fc_hc[fc, side]  # connector length
            if w > 0:
                result[i_cv] += fc_data[fc] * w
                weights[i_cv] += w

    # Normalize
    mask = weights[:, 0] > 0
    result[mask] /= weights[mask]
    
    if one_dim:
        return result[:, 0]
    return result


def cell_to_face(grid: GridTopology, cv_data: np.ndarray) -> np.ndarray:
    """Interpolate cell-centered data to faces.
    
    Uses inverse-distance weighting via fcHc.
    MATLAB equivalent: intface.
    
    Args:
        grid: GridTopology
        cv_data: (n_cells,) or (n_cells, n_species) — data on cells
    
    Returns:
        (n_faces,) or (n_faces, n_species) — data on faces
    """
    if cv_data.ndim == 1:
        cv_data = cv_data[:, np.newaxis]
        one_dim = True
    else:
        one_dim = False

    n_faces = grid.n_faces
    n_spec = cv_data.shape[1]
    result = np.zeros((n_faces, n_spec), dtype=np.float64)

    for i_fc in range(n_faces):
        cv1, cv2 = grid.fc_cv[i_fc]
        w1 = grid.fc_hc[i_fc, 1]  # weight from cv2 side
        w2 = grid.fc_hc[i_fc, 0]  # weight from cv1 side
        total = w1 + w2
        if total > 0:
            result[i_fc] = (cv_data[cv1] * w1 + cv_data[cv2] * w2) / total

    if one_dim:
        return result[:, 0]
    return result
