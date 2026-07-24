"""SolpsDataset — the main container for SOLPS-ITER watch analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from solps_analysis.core.grid import GridTopology
from solps_analysis.core.variable import SolpsVariable, VariableMeta
from solps_analysis.io.geometry_reader import read_geometry, read_b2fstati
from solps_analysis.io.data_readers import (
    read_structured_dat,
    read_unstructured_dat,
    scan_output_directory,
)


@dataclass
class SolpsWatch:
    """Represents a single SOLPS-ITER watch (one time snapshot).

    This is the main entry point for analyzing a watch.

    Typical usage::

        watch = SolpsWatch.from_directory("/path/to/watch")

        # Access geometry
        print(watch.grid.n_cells)

        # Access a variable
        te = watch.get("te_eV")
        print(te.data.shape)

        # List all available variables
        print(watch.list_variables())
    """

    path: Path
    grid: GridTopology
    eirene_flag: bool = False

    # Variable storage: {var_name: SolpsVariable}
    _variables: dict[str, SolpsVariable] = field(default_factory=dict)

    # File index: {var_name: file_path}
    _file_index: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        load_variables: bool = True,
        variables: list[str] | None = None,
    ) -> "SolpsWatch":
        """Load a complete watch from its directory.

        Args:
            path: Path to the watch directory (contains output/, b2fgmtry, etc.)
            load_variables: If True, scan and load .dat files
            variables: If given, only load these specific variables (partial read)
        """
        path = Path(path).expanduser().resolve()

        # 1. Read geometry
        try:
            grid = cls._find_and_read_geometry(path)
        except FileNotFoundError:
            # Try loading just b2fstati for minimal info
            grid = read_b2fstati(path)
            if grid.n_cells == 0:
                raise

        watch = cls(path=path, grid=grid)

        # 2. Detect version and grid type
        watch._detect_eirene()

        # 3. Scan and load variables
        if load_variables:
            output_dir = path / "output"
            if output_dir.is_dir():
                file_index = scan_output_directory(str(output_dir))
                watch._file_index = file_index
                if variables:
                    watch._load_selected(variables)
                else:
                    watch._load_all()

        return watch

    @staticmethod
    def _find_and_read_geometry(path: Path) -> GridTopology:
        """Find and read b2fgmtry, searching up the tree."""
        # Try different locations
        candidates = [
            path / "b2fgmtry",
            path.parent / "b2fgmtry",
            path.parent.parent / "b2fgmtry",
        ]
        for c in candidates:
            if c.exists():
                return read_geometry(str(c))
        raise FileNotFoundError(f"No b2fgmtry found for {path}")

    def _detect_eirene(self) -> None:
        """Check if EIRENE data is present (fort.44, fort.46, input.dat)."""
        has_ft44 = (self.path.parent / "fort.44").exists()
        has_ft46 = (self.path.parent / "fort.46").exists()
        has_input = (self.path / "input.dat").exists() or (self.path.parent / "input.dat").exists()
        self.eirene_flag = has_ft44 and has_ft46 and has_input

    def _load_all(self) -> None:
        """Load all variables found in the output/ directory."""
        for var_name, file_path in self._file_index.items():
            try:
                data = self._read_dat_file(file_path)
                meta = VariableMeta(
                    name=var_name,
                    location=self._infer_location(var_name),
                    source_file=file_path,
                )
                self._variables[var_name] = SolpsVariable(data=data, meta=meta)
            except Exception as e:
                pass  # Skip unreadable files

    def _load_selected(self, var_names: list[str]) -> None:
        """Load only selected variables."""
        for var_name in var_names:
            file_path = self._file_index.get(var_name)
            if file_path:
                try:
                    data = self._read_dat_file(file_path)
                    meta = VariableMeta(
                        name=var_name,
                        location=self._infer_location(var_name),
                        source_file=file_path,
                    )
                    self._variables[var_name] = SolpsVariable(data=data, meta=meta)
                except Exception as e:
                    pass

    def _read_dat_file(self, file_path: str) -> np.ndarray:
        """Read a .dat file using the appropriate reader.

        If the grid is structured (stored nx, ny), convert 2D → 1D.
        """
        # First read raw
        from solps_analysis.io.data_readers import read_watch_file
        values = read_watch_file(file_path)

        # If the grid has structured mapping, convert 2D → unstructured 1D
        if self.grid.is_structured and self.grid.imap_cv is not None and values.ndim == 2:
            # values is (ny+2, nx+2) after read_structured_dat
            if values.shape == (self.grid.ny + 2, self.grid.nx + 2):
                values = self._structured_to_unstructured(values)

        values = np.ascontiguousarray(values, dtype=np.float64)
        return values

    def _structured_to_unstructured(self, data_2d: np.ndarray) -> np.ndarray:
        """Convert structured 2D data to unstructured 1D using imapCv mapping.

        data_2d: (ny+2, nx+2) — already flipped (row 0 = outermost flux surface)
        """
        if self.grid.imap_cv is None:
            return data_2d.ravel()

        ncells = self.grid.n_cells
        result = np.zeros(ncells, dtype=np.float64)
        imap = self.grid.imap_cv  # (nx+2, ny+2)
        ny, nx = data_2d.shape

        for j in range(ny):  # j = poloidal (iy)
            for i in range(nx):  # i = radial (ix)
                idx = imap[i, j]
                if idx != 0:
                    result[idx - 1] = data_2d[j, i]

        return result

    def _infer_location(self, var_name: str) -> str:
        """Infer whether a variable is cell-centered or face-centered."""
        # Heuristic: face data often has _r, _th, _mdf_, _fht, _fhe etc.
        # Cell data: na, sna, resco, te_eV, ne, etc.
        face_indicators = [
            "_r", "_th", "_fht", "_fhe", "_fhi", "_fhn",
            "_mdf", "_fna", "_fnax", "_fnay",
        ]
        for indicator in face_indicators:
            if indicator in var_name:
                return "face_r" if "_th" not in var_name else "face_th"
        return "cell"

    # --- Public API ---

    def get(self, name: str) -> SolpsVariable | None:
        """Get a variable by name."""
        return self._variables.get(name)

    def list_variables(self) -> list[str]:
        """List all loaded variables."""
        return sorted(self._variables.keys())

    def list_all_files(self) -> list[str]:
        """List all available variable file names (loaded or not)."""
        return sorted(self._file_index.keys())

    def __getitem__(self, name: str) -> SolpsVariable:
        var = self.get(name)
        if var is None:
            raise KeyError(f"Variable '{name}' not found. Available: {self.list_variables()}")
        return var

    def __repr__(self) -> str:
        return (
            f"SolpsWatch({self.path.name}, nCv={self.grid.n_cells}, "
            f"vars={len(self._variables)}, EIRENE={self.eirene_flag})"
        )
