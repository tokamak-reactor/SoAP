# Visualization Architecture

## Class hierarchy

```
Plot (ABC)
 ├── Plot1D       — profiles along OMP/IMP/targets/flux tubes
 ├── Plot2D       — 2D filled contours on SOLPS grid
 ├── PlotWall     — wall geometry
 ├── PlotMesh     — computational mesh
 └── PlotMulti1D  — multiple variables on one axis
```

## PlotConfig (dataclass)

```python
@dataclass
class PlotConfig:
    type: str                    # "1d" | "2d" | "wall" | "mesh"
    variable: str | list[str] | None = None
    along: str | None = None     # "omp", "imp", "target_in", "target_out", "ft_N"
    region: str = "all"
    log: bool = False
    cmap: str | None = None      # None = from preset
    title: str | None = None     # auto from metadata
    style: str = "screen"        # "screen" | "journal" | "presentation"
    style_overrides: dict = field(default_factory=dict)
```

## Style presets (style.py)

```python
PRESETS = {
    "screen":      # 10×8in, 100dpi, sans-serif, for work-in-progress
    "journal":     # 17×11cm, 300dpi, serif, for A4 papers (LaTeX/Word)
    "presentation":  # 16×9, large fonts, for projector
}
```

## PlotList

```python
@dataclass
class PlotList:
    name: str
    description: str
    plots: list[PlotConfig]
    style: str = "screen"

    def run(watch, tweaks=None) -> list[PlotResult]
    def save(path)
    @classmethod def load(path) -> PlotList
    @classmethod def builtin(name) -> PlotList
```

Storage:
- Builtin: `src/solps_analysis/plot/lists/*.yaml` (under git)
- User: `plot_lists/*.yaml` in project root (.gitignore)

## PlotResult

```python
@dataclass
class PlotResult:
    config: PlotConfig
    fig: Figure
    ax: Axes
```

User can post-process:
```python
result.ax.legend(loc="upper left")
result.fig.savefig("te_profile.pdf")
```

## Tweaks

Optional callback for repeated post-processing:
```python
def fix_legend(fig, ax):
    ax.legend(loc="upper left", fontsize=7)

list.run(watch, tweaks=fix_legend)
```

## Plot2D implementation notes

- Uses `PatchCollection` with per-cell quadrilateral patches — NOT `tricontourf`
  (which fills the central hole / private flux region via convex hull).
- Cell vertices come from b2fgmtry `crx`/`cry` arrays (4 corner coordinates per cell).
  Stored on GridTopology as `cv_crn_r` / `cv_crn_z`.
- `_build_cell_vertices(grid)` in `plot/plot2d.py` maps imap_cv indices to corner
  data. Fallback path averages cell centers for grids without corner data.
- Separatrix overlay uses `sep_fc` (face indices along separatrix). Wall overlay
  uses `wall_faces`.

```python
# Ad-hoc
watch.plot("te_eV", type="2d")
watch.plot("te_eV", type="2d", style="journal")

# Constructor + edit
p = Plot2D(watch, config={"style": "journal", "log": True})
fig, ax = p.render()
ax.set_xlim(0.5, 1.5)

# One-command batch
PlotList.builtin("student_overview").run(watch)
PlotList.load("plot_lists/my_list.yaml").run(watch, tweaks=fix_legend)
```

## File structure

```
src/solps_analysis/plot/
    __init__.py         # exports Plot, PlotConfig, PlotList, PlotResult
    base.py             # Plot (ABC)
    plot1d.py           # Plot1D, PlotMulti1D
    plot2d.py           # Plot2D
    plot_wall.py        # PlotWall
    plot_mesh.py        # PlotMesh
    style.py            # PRESETS, apply_preset()
    lists/              # builtin YAMLs (under git)
        student_overview.yaml
        divertor_analysis.yaml
        quick_look.yaml
```

## Compatibility checks

- location == "cell" → 2D and 1D both allowed
- location == "face_r" / "face_th" → 1D only
- along="omp" → requires grid.inner_midplane_cells / outer_midplane_cells
- along="target_in" → requires inner_target_cells
