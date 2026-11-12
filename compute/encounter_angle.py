"""Smallest angle between two routes that meet at a waypoint.

Crossing, merging and bend collisions are counted per pair of routes at a
waypoint (``ShipCollisionModelMixin._junction_route_encounters``) with
Pedersen's crossing formula, which divides by ``|sin theta|``.  Two routes
whose arrival courses are almost parallel (overtaking-like) or almost
opposite (head-on-like) are therefore left out: a pair counts only when

    limit <= theta <= 180 - limit      (degrees)

Friis-Hansen (2008, section 4.1) gives the crossing equations for angles
between 10 and 170 degrees, and IWRAP uses the same limit, so the default
is **10 degrees**.  For a bend the angle between the two reverse routes is
180 degrees minus the turn, so a turn smaller than the limit gives no bend.
Until 2026-11-10 OMRAT used 0.1 rad (5.7 degrees).

The key lives in the project's ``pc`` block (Settings > Causation factors).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

KEY = 'min_encounter_angle_deg'

DEFAULT_DEG = 10.0
MIN_DEG = 1.0
MAX_DEG = 45.0

DEFAULTS: dict[str, float] = {KEY: DEFAULT_DEG}


def min_angle_deg(pc_vals: Mapping[str, Any] | None) -> float:
    """Configured limit in degrees, clamped to ``MIN_DEG..MAX_DEG`` (default 10)."""
    try:
        deg = float((pc_vals or {}).get(KEY, DEFAULT_DEG))
    except (TypeError, ValueError):
        deg = DEFAULT_DEG
    if deg != deg:  # NaN
        deg = DEFAULT_DEG
    return min(MAX_DEG, max(MIN_DEG, deg))


def min_angle_rad(pc_vals: Mapping[str, Any] | None) -> float:
    """The limit in radians."""
    return math.radians(min_angle_deg(pc_vals))


def angle_counts(theta_rad: float, limit_rad: float) -> bool:
    """Is a route pair at ``theta_rad`` (0..pi) far enough from parallel and opposite?"""
    return limit_rad <= theta_rad <= math.pi - limit_rad


def describe(pc_vals: Mapping[str, Any] | None) -> str:
    """One line for reports."""
    deg = min_angle_deg(pc_vals)
    return f'{deg:g} to {180.0 - deg:g} degrees between the arrival courses'
