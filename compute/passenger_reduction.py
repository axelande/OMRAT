"""Passenger ship / fast ferry causation reduction (IWRAP's "reduction factor"), QGIS-free.

IWRAP lowers the causation factor of passenger ships and fast ferries,
on the reasoning that their bridge manning and equipment make a human
error less likely.  The factor ``F`` (IWRAP's default 20) acts per ship:

* ship-ship collisions: the causation factor of an encounter is multiplied
  by ``1 / sqrt(F)`` for every passenger ship or fast ferry in it, so a
  passenger ship meeting a cargo ship gets ``Pc / sqrt(20)`` and two
  passenger ships ``Pc / 20``;
* powered grounding and allision: the ship's frequency is divided by ``F``.

Drifting is not affected (IWRAP covers it with the lower blackout rate).

OMRAT stores the two factors in the ``pc`` block (:data:`PASSENGER_KEY`,
:data:`FAST_FERRY_KEY`), edits them in the Causation factors dialog and
exports them to IWRAP's ``global_settings/misc`` as set, so the two tools
apply the factor once each (KNOWN_ISSUES 19).  The default is IWRAP's 20
for both (user's decision, 2026-11-09); 1 means no reduction.  A project
that stores a value keeps it.

Which rows count as passenger ships / fast ferries follows the IWRAP
export's type grouping (``compute.iwrap_convertion._UI_TO_IWRAP``) and,
for other taxonomies, the ship type name (:func:`ship_type_class`).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

PASSENGER_KEY = 'passenger_reduction'
FAST_FERRY_KEY = 'fast_ferry_reduction'
IWRAP_DEFAULT = 20.0
"""IWRAP's own default for both factors."""
DEFAULTS: dict[str, float] = {PASSENGER_KEY: IWRAP_DEFAULT, FAST_FERRY_KEY: IWRAP_DEFAULT}
"""OMRAT's defaults = IWRAP's (a project without the keys gets them)."""

_FAST_FERRY_TOKENS = ('high speed craft', 'hsc', 'fast ferry', 'fast-ferry')
_PASSENGER_TOKENS = ('passenger', 'ro-pax', 'ropax', 'cruise', 'ferry')


def factor(pc: dict[str, Any] | None, key: str) -> float:
    """The stored factor; the default (20) when missing; 1 (no reduction) when not a number >= 1."""
    try:
        value = float((pc or {}).get(key, DEFAULTS[key]))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value < 1.0:
        return 1.0
    return value


def ship_type_class(name: Any) -> str | None:
    """``'fast_ferry'``, ``'passenger'`` or ``None`` for one ship type name."""
    n = str(name or '').lower()
    if any(tok in n for tok in _FAST_FERRY_TOKENS):
        return 'fast_ferry'
    if any(tok in n for tok in _PASSENGER_TOKENS):
        return 'passenger'
    return None


def type_factors(type_names: Sequence[Any] | None, pc: dict[str, Any] | None,
                 n_types: int | None = None) -> list[float]:
    """Reduction factor ``F`` per ship-type row (1.0 for every other type).

    ``n_types`` pads (with 1.0) or trims the list to the traffic matrix size
    when the type list and the matrices disagree.
    """
    f_pass = factor(pc, PASSENGER_KEY)
    f_fast = factor(pc, FAST_FERRY_KEY)
    out = []
    for name in (type_names or []):
        cls = ship_type_class(name)
        out.append(f_fast if cls == 'fast_ferry' else f_pass if cls == 'passenger' else 1.0)
    if n_types is not None:
        out = (out + [1.0] * n_types)[:n_types]
    return out


def collision_scales(type_names: Sequence[Any] | None, pc: dict[str, Any] | None,
                     n_types: int | None = None) -> list[float]:
    """Per-type multiplier on the collision causation factor: ``1 / sqrt(F)``.

    The causation factor of an encounter between rows ``i`` and ``j`` is
    ``Pc * scale[i] * scale[j]``.
    """
    return [1.0 / math.sqrt(f) for f in type_factors(type_names, pc, n_types)]


def powered_scales(type_names: Sequence[Any] | None, pc: dict[str, Any] | None,
                   n_types: int | None = None) -> list[float]:
    """Per-type multiplier on powered grounding / allision: ``1 / F``."""
    return [1.0 / f for f in type_factors(type_names, pc, n_types)]


def is_active(pc: dict[str, Any] | None) -> bool:
    """True when either factor reduces anything."""
    return factor(pc, PASSENGER_KEY) > 1.0 or factor(pc, FAST_FERRY_KEY) > 1.0


def describe(pc: dict[str, Any] | None) -> str:
    """One line for the result report."""
    return (f"Passenger ship reduction factor {factor(pc, PASSENGER_KEY):g}, "
            f"fast ferry reduction factor {factor(pc, FAST_FERRY_KEY):g} "
            "(collisions: causation / sqrt(F) per such ship; powered: frequency / F)")
