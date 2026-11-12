"""Squat added to the draught in the grounding models, QGIS-free.

Two tick boxes in **Settings > Causation factors** (``pc`` block), both on by
default (user's decision 2026-11-10):

* ``squat_drifting`` -- drifting grounding: a ship blacks out at its speed on
  the leg, so the squat at that speed decides which depth areas it can
  ground on;
* ``squat_powered`` -- powered grounding, Category I and II: the ship sails
  at its speed on the leg.

Squat = Barras (1979), open water: ``S = Cb * V^2 / 100`` (V in knots, S in
metres; :func:`compute.basic_equations.squat_m`), V = the cell's speed, the
block coefficient from the ship type's name (:func:`cb_for_type_name`).  A
ship grounds on every depth area shallower than ``draught + S``.

Anchoring keeps the static draught (the maximum anchoring depth is
``anchor_d`` x draught), and so does the allision height clearance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from compute.basic_equations import _CB_BY_SHIP_TYPE, squat_m

KEY_DRIFTING = 'squat_drifting'
KEY_POWERED = 'squat_powered'
SQUAT_DEFAULTS: dict[str, bool] = {KEY_DRIFTING: True, KEY_POWERED: True}
DEFAULT_CB = 0.75
"""Block coefficient for a type the name rules do not know (conservative)."""

# First match wins, so the specific names come before the generic ones
# ("Chemical / gas tanker" before "tanker", "Passenger cruise" before
# "passenger").  Values follow ``basic_equations._CB_BY_SHIP_TYPE``.
_CB_BY_NAME: tuple[tuple[tuple[str, ...], float], ...] = (
    (('search', 'rescue'), 0.50),
    (('pleasure',), 0.55),
    (('sailing',), 0.65),
    (('high speed', 'hsc', 'fast ferry'), 0.65),
    (('fishing',), 0.62),
    (('tug', 'push', 'towing', 'pilot', 'dredg'), 0.65),
    (('cruise',), 0.70),
    (('ro-pax', 'ropax', 'ro-ro', 'roro'), 0.65),
    (('passenger', 'ferry'), 0.70),
    (('container',), 0.67),
    (('chemical', 'gas'), 0.67),
    (('tanker',), 0.82),
    (('bulk',), 0.80),
    (('cargo',), 0.72),
)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def squat_on(pc_vals: Mapping[str, Any] | None, key: str) -> bool:
    """Is squat ticked for ``key`` (:data:`KEY_DRIFTING` / :data:`KEY_POWERED`)?  Default on."""
    pc = pc_vals if isinstance(pc_vals, Mapping) else {}
    return _as_bool(pc.get(key), SQUAT_DEFAULTS.get(key, True))


def cb_for_type_name(name: str | None, index: int | None = None) -> float:
    """Block coefficient for a ship type, from its name; by OMRAT index without a name."""
    n = str(name or '').lower()
    if n:
        for tokens, cb in _CB_BY_NAME:
            if any(tok in n for tok in tokens):
                return cb
        return DEFAULT_CB
    if index is not None and index in _CB_BY_SHIP_TYPE:
        return _CB_BY_SHIP_TYPE[index]
    return DEFAULT_CB


def cell_squat_m(speed_kts: Any, cb: float) -> float:
    """Squat (m) of a cell sailing at ``speed_kts``; 0 for a missing or non-positive speed."""
    try:
        v = float(speed_kts)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0:
        return 0.0
    return squat_m(v, cb=cb)


def type_names(data: Mapping[str, Any] | None) -> list[str]:
    """The project's ship-type names (``ship_categories.types``), ``[]`` when absent."""
    cats = (data or {}).get('ship_categories') if isinstance(data, Mapping) else None
    return list((cats or {}).get('types') or []) if isinstance(cats, Mapping) else []


def squat_for_row(names: list[str], row: int, speed_kts: Any) -> float:
    """Squat of traffic-matrix row ``row`` (ship type) at ``speed_kts``."""
    return cell_squat_m(speed_kts, cb_for_type_name(names[row] if 0 <= row < len(names) else None, row))
