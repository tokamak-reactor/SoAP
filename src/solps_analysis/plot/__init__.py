"""Plotting module for SOLPS-ITER analysis."""
from solps_analysis.plot.base import Plot, PlotConfig, PlotResult, PRESETS
from solps_analysis.plot.plot1d import Plot1D, PlotMulti1D
from solps_analysis.plot.plot2d import Plot2D
from solps_analysis.plot.style import apply_preset, merge_style, PRESETS as STYLE_PRESETS

__all__ = [
    "Plot", "PlotConfig", "PlotResult",
    "Plot1D", "PlotMulti1D", "Plot2D",
    "PRESETS", "STYLE_PRESETS", "apply_preset", "merge_style",
]
