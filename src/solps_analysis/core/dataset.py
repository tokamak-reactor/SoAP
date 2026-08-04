"""SolpsDataset — main container for SOLPS-ITER watch analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from solps_analysis.core.grid import GridTopology
from solps_analysis.core.variable import SolpsVariable, VariableMeta
from solps_analysis.io.geometry_reader import read_geometry, read_b2fstati

# Units for MATLAB-style workspace variables (read_data_3x.m semantics).
# Base name before the first "_" is used as fallback.
_WORKSPACE_UNITS: dict[str, str] = {
    "te": "eV", "ti": "eV", "tn": "eV", "ne": "m⁻³", "na": "m⁻³",
    "ua": "m/s", "ue": "m/s", "uap": "m/s", "po": "V", "Zeff": "",
    "she": "W/m³", "shi": "W/m³", "reshe": "W/m³", "reshi": "W/m³",
    "resht": "W/m³", "respo": "A/m³", "sch": "A/m²", "taua": "s",
    "taue": "s", "sna": "m⁻³s⁻¹", "dnadt": "m⁻³s⁻¹", "fna": "m⁻²s⁻¹",
    "fhe": "W/m²", "fhi": "W/m²", "fch": "A/m²", "fmo": "N/m²",
    "smo": "N/m³", "she_rad": "W/m³", "she_radlin": "W/m³",
    "she_radbrm": "W/m³", "vel_ExB": "m/s", "vel_BgradB": "m/s",
}
from solps_analysis.io.data_readers import (
    read_structured_dat,
    read_unstructured_dat,
    scan_output_directory,
    read_watch_file,
)
from solps_analysis.io.b2fstate_reader import (
    extract_state_arrays,
    get_plasma_composition,
)
from solps_analysis.io.variable_catalog import _catalog_transform


@dataclass
class SolpsWatch:
    """Represents a single SOLPS-ITER watch (one time snapshot).

    Usage::

        watch = SolpsWatch.from_directory("/path/to/watch")
        print(watch.list_variables())
        te = watch["te_eV"]
    """

    path: Path
    grid: GridTopology
    eirene_flag: bool = False
    plasma_composition: dict | None = None

    _variables: dict[str, SolpsVariable] = field(default_factory=dict)
    _file_index: dict[str, str] = field(default_factory=dict)
    _construct_cache: dict[str, SolpsVariable] = field(default_factory=dict)
    _workspace_cache: dict | None = None

    # EIRENE data
    neut: dict | None = None
    ft46: dict | None = None
    input_file: dict | None = None

    # Numerical data
    numerics: dict | None = None
    run_log: dict | None = None

    # Composition
    b2_comp: Any = None
    eirene_comp: Any = None

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        load_variables: bool = True,
        variables: list[str] | None = None,
        load_eirene: bool = False,
    ) -> "SolpsWatch":
        """Load a complete watch from its directory.

        Args:
            path: Path to the watch directory.
            load_variables: Scan and load .dat files from output/.
            variables: If given, only load these specific variables.
            load_eirene: Also load EIRENE and numerical data.
        """
        path = Path(path).expanduser().resolve()

        # 1. Read geometry (searches up directory tree)
        try:
            grid = read_geometry(str(path))
        except FileNotFoundError:
            grid = read_b2fstati(path)

        watch = cls(path=path, grid=grid)
        watch._detect_eirene()

        # 2. Read plasma composition from b2fstati
        try:
            fstate_raw = read_b2fstati(str(path))
            if hasattr(fstate_raw, 'n_species') and fstate_raw.n_species > 0:
                watch.plasma_composition = get_plasma_composition({
                    "zamax": fstate_raw.species_charge_max,
                    "am": fstate_raw.species_mass,
                    "zn": fstate_raw.species_n,
                })
        except Exception:
            pass

        # 3. Scan and load variables from output/
        if load_variables:
            output_dir = path / "output"
            if output_dir.is_dir():
                file_index = scan_output_directory(str(output_dir))
                watch._file_index = file_index
                if variables:
                    watch._load_selected(variables)
                else:
                    watch._load_all()

        # 4. Load EIRENE and numerical data
        if load_eirene:
            watch._load_eirene_data()
            watch._load_numerical_data()

        return watch

    def _load_eirene_data(self) -> None:
        """Load EIRENE data (fort.44, fort.46, input.dat)."""
        from solps_analysis.io.eirene_reader import read_fort44, read_fort46
        from solps_analysis.io.input_reader import read_eirene_input, read_user_parameters
        import warnings

        ft44_path = self.path / "fort.44"
        ft46_path = self.path / "fort.46"
        input_path = self.path / "input.dat"

        if ft44_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.neut = read_fort44(str(ft44_path))
            except Exception as e:
                pass

        if ft46_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.ft46 = read_fort46(str(ft46_path))
            except Exception as e:
                pass

        if input_path.exists():
            try:
                self.input_file = read_eirene_input(str(input_path))
            except Exception:
                pass

        self.eirene_flag = (self.neut is not None and self.ft46 is not None)

    def _load_numerical_data(self) -> None:
        """Load numerical parameters and run log."""
        from solps_analysis.io.run_log_reader import read_run_log, read_numerics_parameters

        run_log_path = self.path / "run.log"
        if run_log_path.exists():
            try:
                self.run_log = read_run_log(str(run_log_path))
            except Exception:
                pass

        b2num_path = self.path / "b2.numerics.parameters"
        if b2num_path.exists():
            try:
                self.numerics = read_numerics_parameters(str(b2num_path))
            except Exception:
                pass

    def _detect_eirene(self) -> None:
        """Check if EIRENE files are present."""
        has_ft44 = (self.path / "fort.44").exists()
        has_ft46 = (self.path / "fort.46").exists()
        has_input = (self.path / "input.dat").exists()
        self.eirene_flag = has_ft44 and has_ft46

    def _load_all(self) -> None:
        """Load all variables from output directory."""
        for var_name, file_path in self._file_index.items():
            try:
                data = self._read_dat_file(file_path)
                meta = VariableMeta(
                    name=var_name,
                    location=self._infer_location(var_name),
                    source_file=str(file_path),
                )
                self._variables[var_name] = SolpsVariable(data=data, meta=meta)
            except Exception:
                pass

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
                        source_file=str(file_path),
                    )
                    self._variables[var_name] = SolpsVariable(data=data, meta=meta)
                except Exception:
                    pass

    def _read_dat_file(self, file_path: str) -> np.ndarray:
        """Read a .dat file and convert to 1D if structured grid.

        Uses the variable catalog to decide cell vs face transformation.
        """
        values = read_watch_file(file_path)

        # Convert 2D → 1D for structured grids
        if self.grid.is_structured and self.grid.imap_cv is not None and values.ndim == 2:
            ny, nx = values.shape
            ey, ex = self.grid.ny + 2, self.grid.nx + 2
            if ny == ey and nx == ex:
                # Determine transform: cell (default) or face (dim 1/2)
                transform = _catalog_transform(file_path)
                if transform == "face_x":
                    values = self._structured_to_faces(values, dim=1)
                elif transform == "face_y":
                    values = self._structured_to_faces(values, dim=2)
                else:
                    values = self._structured_to_unstructured(values)
            elif ny == ex and nx == ey:
                transform = _catalog_transform(file_path)
                if transform == "face_x":
                    values = self._structured_to_faces(values.T, dim=1)
                elif transform == "face_y":
                    values = self._structured_to_faces(values.T, dim=2)
                else:
                    values = self._structured_to_unstructured(values.T)

        return np.ascontiguousarray(values, dtype=np.float64)

    def _structured_to_faces(self, data_2d: np.ndarray, dim: int = 1) -> np.ndarray:
        """Convert structured 2D (ny+2, nx+2) to 1D face data.

        dim=1 → x-faces via imap_fcx (west faces)
        dim=2 → y-faces via imap_fcy (bottom faces)
        MATLAB equivalent: st_us_transform_fc(input, dim).
        """
        n_faces = self.grid.n_faces
        result = np.zeros(n_faces, dtype=np.float64)
        if dim == 1:
            imap = self.grid.imap_fcx
        else:
            imap = self.grid.imap_fcy
        if imap is None:
            return result
        mask = imap > 0
        # data_2d is (ny+2, nx+2); imap is (nx+2, ny+2); MATLAB indexes var(iy,ix)
        result[imap[mask].astype(np.intp) - 1] = data_2d.T[mask]
        return result

    def _structured_to_unstructured(self, data_2d: np.ndarray) -> np.ndarray:
        """Convert structured 2D to unstructured 1D via imap_cv — vectorized."""
        if self.grid.imap_cv is None:
            return data_2d.ravel()

        ncells = self.grid.n_cells
        result = np.zeros(ncells, dtype=np.float64)
        imap = self.grid.imap_cv  # (nx+2, ny+2)

        # data_2d is (ny+2, nx+2) after flipud. imap is (nx+2, ny+2).
        # We need rd_idx[j, i] → imap[i, j], so use .T
        mask = imap > 0
        result[imap[mask].astype(np.intp) - 1] = data_2d.T[mask]
        return result

    def _infer_location(self, var_name: str) -> str:
        face_indicators = ["_r", "_th", "_fht", "_fhe", "_fhi", "_fhn", "_mdf", "_fna", "_fnax", "_fnay"]
        for indicator in face_indicators:
            if indicator in var_name:
                return "face_r" if "_th" not in var_name else "face_th"
        return "cell"

    # --- Public API ---

    def get(self, name: str) -> SolpsVariable | None:
        return self._variables.get(name)

    @property
    def workspace(self) -> dict[str, np.ndarray]:
        """MATLAB-style workspace (read_data_3x.m port), built lazily.

        Contains the *assembled* physical variables under MATLAB names:
        te/ti (in eV), ne, po, na (nCv×ns), ua, reshe/reshi/resht/respo,
        sch, fna_mdf_th/r, sna, taua, … — 178 variables. This is the
        third data level:
          raw .dat files  → watch.get("b2nph9_te")     (list_variables)
          MATLAB workspace → watch.workspace["te"]     (list_workspace_vars)
          derived physics → watch.construct("te_sep")  (list_quantities)
        """
        cached = getattr(self, "_workspace_cache", None)
        if cached is None:
            from solps_analysis.io.matlab_vars import build_workspace
            cached = build_workspace(self)
            self._workspace_cache = cached
        return cached

    def list_workspace_vars(self) -> list[str]:
        """Names of MATLAB-style workspace variables (assembled physics)."""
        return sorted(self.workspace.keys())

    def ws_var(self, name: str) -> SolpsVariable | None:
        """A workspace variable as a SolpsVariable (with unit metadata).

        Example: watch.ws_var("na") → SolpsVariable (nCv×ns densities),
        watch.ws_var("reshe") → residual electron heat source.
        """
        data = self.workspace.get(name)
        if data is None:
            return None
        return SolpsVariable(
            data=np.asarray(data, dtype=np.float64),
            meta=VariableMeta(name=name, unit=_WORKSPACE_UNITS.get(name, "")),
        )

    def list_variables(self) -> list[str]:
        return sorted(self._variables.keys())

    def list_all_files(self) -> list[str]:
        return sorted(self._file_index.keys())

    def construct(self, name: str) -> SolpsVariable | None:
        """Compute a derived physical quantity by name.

        Uses the quantity registry (@quantity decorator).
        Automatically loads composition data if needed.

        Results are memoized on the watch (quantities are pure functions
        of immutable watch data), so repeated calls are ~0.02 ms.
        Clear the cache with ``watch._construct_cache.clear()`` if you
        register new quantities at runtime and want them fresh.
        """
        cached = self._construct_cache.get(name)
        if cached is not None:
            return cached

        from solps_analysis.construct.builtin import basic  # noqa: F401
        from solps_analysis.construct.registry import construct_quantity
        from solps_analysis.construct.composition import (
            build_b2_composition,
            build_eirene_composition,
        )
        from solps_analysis.io.b2fstate_reader import read_b2fstate_full

        # Build compositions if not yet available
        if self.b2_comp is None:
            try:
                fstate = read_b2fstate_full(str(self.path))
                self.b2_comp = build_b2_composition(fstate)
            except Exception:
                pass

        # Auto-load EIRENE data when fort.44/46 exist (needed for
        # snas, Qrad, Shi/She etc.)
        if self.neut is None and (self.path / "fort.44").exists():
            try:
                self._load_eirene_data()
                self._load_numerical_data()
            except Exception:
                pass
        if self.eirene_comp is None and self.neut is not None:
            try:
                self.eirene_comp = build_eirene_composition(self.neut)
            except Exception:
                pass

        # Auto-assemble na if needed and not present
        if self.get("na") is None:
            self._assemble_na()

        var = construct_quantity(
            name=name,
            watch=self,
            grid=self.grid,
            comp=self.b2_comp,
            eirene_comp=self.eirene_comp,
        )
        if var is not None:
            self._variables[name] = var
            self._construct_cache[name] = var
        return var

    def _assemble_na(self) -> None:
        """Assemble na matrix from B2 species density files.
        
        Looks for files matching b2npc*_na_NNN or b2npco_na_NNN.
        """
        import re
        na_vars = {}
        for vname in self._file_index:
            # Match: b2npc11_na_NNN or b2npco_na_NNN (B2 species density)
            m = re.match(r"^(b2npc\d+|b2npco)_na_(\d{3})$", vname)
            if m:
                idx = int(m.group(2))
                na_vars[idx] = self._file_index[vname]

        if not na_vars:
            return

        # Read, convert to unstructured, and stack
        from solps_analysis.io.data_readers import read_watch_file
        cols = []
        max_idx = max(na_vars.keys())
        for i in range(max_idx + 1):
            if i in na_vars:
                data = read_watch_file(na_vars[i])
                # Convert 2D structured → 1D unstructured
                if data.ndim == 2 and self.grid.imap_cv is not None:
                    data = self._structured_to_unstructured(data)
                cols.append(data)
            else:
                if cols:
                    cols.append(np.zeros(cols[0].shape if cols else 1))
                else:
                    cols.append(np.zeros(self.grid.n_cells if self.grid.n_cells else 1))

        if cols:
            na_data = np.column_stack(cols)
            self._variables["na"] = SolpsVariable(
                data=np.ascontiguousarray(na_data, dtype=np.float64),
                meta=VariableMeta(
                    name="na",
                    unit="m⁻³",
                    description="Particle density (assembled from species files)",
                    location="cell",
                    is_constructed=True,
                ),
            )

    def construct_all(self) -> list[str]:
        """Compute all registered quantities. Returns list of computed names."""
        from solps_analysis.construct.builtin import basic  # noqa: F401
        from solps_analysis.construct.registry import list_quantities

        computed = []
        for name in list_quantities():
            try:
                if self.construct(name) is not None:
                    computed.append(name)
            except Exception:
                pass
        return computed

    def compute_regions(self) -> list[str]:
        """Compute all region data (targets, midplanes, separatrix)."""
        from solps_analysis.core.regions import compute_all_regions
        from solps_analysis.core.regions_structured import compute_regions_structured

        imap = getattr(self.grid, "imap_cv", None)
        is_struct = self.grid.is_structured and (imap is None or imap.ndim == 2)

        if is_struct:
            result = compute_regions_structured(self.grid)
        else:
            result = compute_all_regions(self.grid)

        if result:
            for key, val in result.items():
                setattr(self.grid, key, val)
            return list(result.keys())
        return []

    def plot(self, variable: str, type: str = "1d", **kwargs) -> "PlotResult":
        """Quick ad-hoc plot. Shortcut for Plot1D (or Plot2D in future)."""
        from solps_analysis.plot import Plot1D, PlotConfig
        config = PlotConfig(type=type, variable=variable, **kwargs)
        return Plot1D(self, config).render()

    def __getitem__(self, name: str) -> SolpsVariable:
        var = self.get(name)
        if var is None:
            raise KeyError(f"Variable '{name}' not found")
        return var

    def __repr__(self) -> str:
        return (
            f"SolpsWatch({self.path.name}, nCv={self.grid.n_cells}, "
            f"vars={len(self._variables)}, EIRENE={self.eirene_flag})"
        )
