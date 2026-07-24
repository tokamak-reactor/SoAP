"""Quantity registry — декоратор @quantity для добавления новых величин."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

from solps_analysis.core.variable import SolpsVariable, VariableMeta

# Глобальный реестр: {name: QuantityDef}
_QUANTITY_REGISTRY: dict[str, "QuantityDef"] = {}


class QuantityDef:
    """Описание одной расчётной величины."""

    def __init__(
        self,
        name: str,
        func: Callable,
        *,
        requires: list[str] | None = None,
        description: str = "",
        unit: str = "",
        location: str = "cell",
    ):
        self.name = name
        self.func = func
        self.requires = requires or []
        self.description = description
        self.unit = unit
        self.location = location

    def __repr__(self) -> str:
        return (
            f"Quantity('{self.name}', requires={self.requires}, "
            f"unit='{self.unit}')"
        )


def quantity(
    name: str | None = None,
    *,
    requires: list[str] | None = None,
    description: str = "",
    unit: str = "",
    location: str = "cell",
) -> Callable:
    """Декоратор для регистрации новой расчётной величины.

    Пример::

        @quantity(
            name="cs",
            requires=["te_eV", "ti_eV"],
            description="Speed of sound",
            unit="m/s",
        )
        def calc_cs(te, ti, grid=None, comp=None, watch=None):
            return np.sqrt((te + ti) * const.e / (const.m_D * const.m_p))
    """

    def decorator(func: Callable) -> Callable:
        qname = name or func.__name__
        qdef = QuantityDef(
            name=qname,
            func=func,
            requires=requires or [],
            description=description or func.__doc__ or "",
            unit=unit,
            location=location,
        )
        _QUANTITY_REGISTRY[qname] = qdef
        return func

    return decorator


def get_quantity(name: str) -> QuantityDef | None:
    """Получить описание величины по имени."""
    return _QUANTITY_REGISTRY.get(name)


def list_quantities() -> list[str]:
    """Список всех зарегистрированных величин."""
    return sorted(_QUANTITY_REGISTRY.keys())


def list_quantities_info() -> list[dict]:
    """Подробная информация о всех величинах."""
    return [
        {
            "name": q.name,
            "requires": q.requires,
            "description": q.description,
            "unit": q.unit,
            "location": q.location,
        }
        for q in _QUANTITY_REGISTRY.values()
    ]


def construct_quantity(
    name: str,
    watch: Any,
    grid: Any,
    comp: Any,
    eirene_comp: Any,
) -> SolpsVariable | None:
    """Вычислить величину по имени, используя данные из watch.

    Args:
        name: Имя величины (из реестра)
        watch: SolpsWatch с данными
        grid: GridTopology
        comp: B2Composition
        eirene_comp: EireneComposition

    Returns:
        SolpsVariable или None, если величина не найдена
    """
    qdef = _QUANTITY_REGISTRY.get(name)
    if qdef is None:
        return None

    # Собираем требуемые переменные
    kwargs: dict[str, Any] = {}
    missing = []

    for req in qdef.requires:
        # Проверяем в watch
        var = watch.get(req)
        if var is not None:
            kwargs[req] = var.data
            continue

        # Проверяем в EIRENE данных (fort.44)
        if watch.neut and req in watch.neut:
            kwargs[req] = watch.neut[req]
            continue

        # Проверяем в fort.46
        if watch.ft46 and req in watch.ft46:
            kwargs[req] = watch.ft46[req]
            continue

        missing.append(req)

    if missing:
        raise ValueError(
            f"Cannot compute '{name}': missing dependencies {missing}"
        )

    # Добавляем контекст
    kwargs["grid"] = grid
    kwargs["comp"] = comp
    kwargs["eirene"] = eirene_comp
    kwargs["watch"] = watch

    # Вычисляем
    data = qdef.func(**kwargs)

    # Оборачиваем в SolpsVariable
    meta = VariableMeta(
        name=name,
        unit=qdef.unit,
        description=qdef.description,
        location=qdef.location,
        is_constructed=True,
    )
    return SolpsVariable(data=np.asarray(data, dtype=np.float64), meta=meta)
