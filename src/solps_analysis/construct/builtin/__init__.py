"""Builtin quantities — автоматически импортируется для регистрации."""

# Импорт необходим для регистрации декораторов @quantity
from solps_analysis.construct.builtin import basic  # noqa: F401
from solps_analysis.construct.builtin import advanced  # noqa: F401
from solps_analysis.construct.builtin import eirene  # noqa: F401
from solps_analysis.construct.builtin import calc_additional  # noqa: F401
from solps_analysis.construct.builtin import energy_balance  # noqa: F401
