"""Style presets for SoAP plotting.

Three built-in presets cover the three user groups:

- ``screen`` — work-in-progress, quick checks (default)
- ``journal`` — A4/LaTeX-ready figures for papers (17×11 cm, 300 dpi, serif)
- ``presentation`` — projector/screen (16×9, large fonts)

Only valid matplotlib rcParams keys are stored in presets; plot-specific
keys (``cmap``, ``vmin``, ``vmax``, ...) live in :class:`PlotConfig` /
``style_overrides`` and are handled by the individual renderers.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

# Keys that are legitimate matplotlib rcParams (applied via plt.rcParams).
# Everything else in a preset/style dict is plot-specific and must be
# consumed by the renderers themselves.
VALID_RCPARAMS = {
    "figure.figsize", "figure.dpi", "font.family", "font.size",
    "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
    "legend.fontsize", "lines.linewidth", "grid.alpha",
    "savefig.bbox", "savefig.pad_inches",
}

PRESETS: dict[str, dict[str, Any]] = {
    "screen": {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    },
    "journal": {
        "figure.figsize": (17 / 2.54, 11 / 2.54),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    },
    "presentation": {
        "figure.figsize": (16, 9),
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
    },
}


def apply_preset(name: str = "screen", overrides: dict[str, Any] | None = None) -> None:
    """Apply a style preset to matplotlib rcParams (in-place, global).

    Only keys in :data:`VALID_RCPARAMS` are applied; unknown keys are
    ignored (they belong to plot-specific handling).
    """
    style = PRESETS.get(name, PRESETS["screen"]).copy()
    if overrides:
        style.update(overrides)
    for key, val in style.items():
        if key in VALID_RCPARAMS:
            try:
                plt.rcParams[key] = val
            except KeyError:
                pass


def merge_style(name: str = "screen", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the merged style dict (preset + overrides) without applying it."""
    style = PRESETS.get(name, PRESETS["screen"]).copy()
    if overrides:
        style.update(overrides)
    return style
