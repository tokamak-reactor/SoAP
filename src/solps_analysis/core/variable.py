"""Variable and dataset definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class VariableMeta:
    """Metadata for a single SOLPS variable.

    Frozen: metadata is written once at creation. The ``extra`` dict is
    the single extension point for arbitrary additional information
    (source, formula, MATLAB reference lines, uncertainty, plot hints,
    …); its *contents* are mutable even though the container is frozen.
    """

    name: str
    unit: str = ""
    description: str = ""
    location: str = "cell"
    source_file: str = ""
    is_constructed: bool = False
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SolpsVariable:
    """A single variable with data and metadata.

    Frozen: variables are created once when the watch is read (or a
    quantity is constructed) and must not be mutated afterwards. To
    attach extra information use ``meta.extra`` (mutable dict), never
    dynamic attributes — they are forbidden on frozen instances.
    """

    data: np.ndarray
    meta: VariableMeta

    def __post_init__(self) -> None:
        if self.data.ndim not in (1, 2):
            raise ValueError(f"Variable data must be 1D or 2D, got {self.data.ndim}D")

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return (
            f"SolpsVariable('{self.name}', shape={self.data.shape}, "
            f"unit='{self.meta.unit}', loc={self.meta.location})"
        )
