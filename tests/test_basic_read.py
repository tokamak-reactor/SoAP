"""Test basic reading of structured and unstructured watches."""

from pathlib import Path

from solps_analysis.core.dataset import SolpsWatch
from solps_analysis.io.geometry_reader import read_geometry, read_b2fstati
from solps_analysis.io.data_readers import (
    scan_output_directory,
    read_structured_dat,
    read_unstructured_dat,
    read_watch_file,
)


def test_structured_geometry():
    """Read geometry from the structured Globus_44644 watch."""

    path = Path("/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus_44644_ng_new_equ/v_tor_Cvel/watch_15.06.2026_12_26/")
    b2fgmtry_path = path.parent.parent / "b2fgmtry"

    print(f"Looking for b2fgmtry at: {b2fgmtry_path}")
    print(f"Exists: {b2fgmtry_path.exists()}")

    try:
        grid = read_geometry(str(path))
        print(f"Grid (structured from geometry): {grid}")
        print(f"  n_cells={grid.n_cells}, n_faces={grid.n_faces}")
        print(f"  nx={grid.nx}, ny={grid.ny}")
        print(f"  version={grid.version}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # Try b2fstati instead
        print("Trying b2fstati...")
        state = read_b2fstati(str(path))
        print(f"  n_cells={state.n_cells}, n_species={state.n_species}")
        print(f"  nx={state.nx}, ny={state.ny}")
        print(f"  is_structured={state.is_structured}")


def test_unstructured_geometry():
    """Read geometry from the unstructured Globus-3 watch."""

    path = Path("/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus-3_WG/sonya_refined_no_redef_v2/watch_02.06.2026_15_30/")

    try:
        grid = read_geometry(str(path))
        print(f"Grid (unstructured): {grid}")
        print(f"  n_cells={grid.n_cells}, n_faces={grid.n_faces}")
        print(f"  n_vertices={grid.n_vertices}")
        print(f"  nx={grid.nx}, ny={grid.ny}")
        print(f"  is_structured={grid.is_structured}")
        print(f"  cv_x shape = {grid.cv_x.shape if grid.cv_x is not None else 'None'}")
        print(f"  cv_y shape = {grid.cv_y.shape if grid.cv_y is not None else 'None'}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")


def test_scan_output_structured():
    """Scan output directory of structured watch."""
    path = Path("/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus_44644_ng_new_equ/v_tor_Cvel/watch_15.06.2026_12_26/")
    output_dir = path / "output"
    if output_dir.exists():
        files = scan_output_directory(str(output_dir))
        print(f"Found {len(files)} files in structured output/")
        # Show some examples
        keys = sorted(files.keys())[:10]
        for k in keys:
            print(f"  {k} -> {files[k]}")


def test_scan_output_unstructured():
    """Scan output directory of unstructured watch."""
    path = Path("/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus-3_WG/sonya_refined_no_redef_v2/watch_02.06.2026_15_30/")
    output_dir = path / "output"
    if output_dir.exists():
        files = scan_output_directory(str(output_dir))
        print(f"Found {len(files)} files in unstructured output/")
        keys = sorted(files.keys())[:10]
        for k in keys:
            print(f"  {k} -> {files[k]}")


def test_read_dat_files():
    """Test reading actual .dat files."""
    # Structured .dat file
    struct_file = "/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus_44644_ng_new_equ/v_tor_Cvel/watch_15.06.2026_12_26/output/te_eV.dat"
    try:
        data = read_structured_dat(struct_file)
        print(f"Structured te_eV.dat: shape={data.shape}")
        print(f"  min={data.min():.3e}, max={data.max():.3e}")
        print(f"  first row: {data[0, :5]}")
    except Exception as e:
        print(f"ERROR reading structured .dat: {e}")

    # Unstructured .dat file
    unstruct_file = "/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus-3_WG/sonya_refined_no_redef_v2/watch_02.06.2026_15_30/output/b2npco_dnadt000_us.dat"
    try:
        data = read_unstructured_dat(unstruct_file)
        print(f"Unstructured dnadt000_us.dat: shape={data.shape}")
        print(f"  min={data.min():.3e}, max={data.max():.3e}")
        print(f"  first 5: {data[:5]}")
    except Exception as e:
        print(f"ERROR reading unstructured .dat: {e}")


def test_full_watch_structured():
    """Test loading a complete structured watch."""
    path = "/run/media/kirill/Fusion/Science/SOLPS_ITER/Watches/Globus_44644_ng_new_equ/v_tor_Cvel/watch_15.06.2026_12_26/"
    try:
        watch = SolpsWatch.from_directory(path, load_variables=True)
        print(f"Structured watch loaded: {watch}")
        print(f"  Variables: {watch.list_variables()[:10]}...")
        te = watch.get("te_eV")
        if te:
            print(f"  te_eV: shape={te.data.shape}")
    except Exception as e:
        print(f"ERROR loading structured watch: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Structured geometry")
    print("=" * 60)
    test_structured_geometry()
    print()

    print("=" * 60)
    print("TEST 2: Unstructured geometry")
    print("=" * 60)
    test_unstructured_geometry()
    print()

    print("=" * 60)
    print("TEST 3: Scan output (structured)")
    print("=" * 60)
    test_scan_output_structured()
    print()

    print("=" * 60)
    print("TEST 4: Scan output (unstructured)")
    print("=" * 60)
    test_scan_output_unstructured()
    print()

    print("=" * 60)
    print("TEST 5: Read .dat files")
    print("=" * 60)
    test_read_dat_files()
    print()

    print("=" * 60)
    print("TEST 6: Full structured watch")
    print("=" * 60)
    test_full_watch_structured()
    print()
