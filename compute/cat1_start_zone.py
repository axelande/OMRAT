"""Category I "start zone": ignore in-lane hits right after the origin waypoint.

Category I casts rays parallel to a leg from the waypoint the ships come
from and counts every ray that meets an obstacle before the far waypoint
(:func:`geometries.get_powered_overlap._compute_cat1_in_lane`).  Close to
the origin that picture is wrong: the ships are still arriving from the
previous leg, and their lateral position is not yet described by this
leg's fitted distribution.  A wind farm lying next to a junction then
collects the far tail of the distribution of a leg whose ships never
sail there (ESCOW case study, LEG_1_2_b East going, 2026-10-01).

With the setting on, a ray whose first edge crossing lies within the
first ``pct`` % of the leg is dropped.  Rays that cross an edge further
along still count, so an obstacle straddling the line keeps the part
beyond it (credited to the edge where the ray leaves the start zone).
Rays of the opposite direction start at the far waypoint, so the same
obstacle near the junction is still hit by them at 90-100 % of the leg.
Category II is untouched.

Both keys live in the project's ``pc`` block next to the Cat I causation
factors (Settings > Causation factors) and default to **on, 10 %**.
"""
from __future__ import annotations

from typing import Any, Mapping

KEY_ENABLED = 'cat1_skip_start'
KEY_PCT = 'cat1_skip_start_pct'

DEFAULT_ENABLED = True
DEFAULT_PCT = 10.0
MAX_PCT = 90.0

CAT1_START_ZONE_DEFAULTS: dict[str, Any] = {
    KEY_ENABLED: DEFAULT_ENABLED,
    KEY_PCT: DEFAULT_PCT,
}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def start_zone_pct(pc_vals: Mapping[str, Any] | None) -> float:
    """Configured percentage, clamped to ``0..MAX_PCT`` (default 10)."""
    try:
        pct = float((pc_vals or {}).get(KEY_PCT, DEFAULT_PCT))
    except (TypeError, ValueError):
        pct = DEFAULT_PCT
    if pct != pct:  # NaN
        pct = DEFAULT_PCT
    return min(MAX_PCT, max(0.0, pct))


def start_zone_enabled(pc_vals: Mapping[str, Any] | None) -> bool:
    return _as_bool((pc_vals or {}).get(KEY_ENABLED), DEFAULT_ENABLED)


def start_zone_fraction(pc_vals: Mapping[str, Any] | None) -> float:
    """Fraction of the leg length the Cat I rays skip (0 when off)."""
    if not start_zone_enabled(pc_vals):
        return 0.0
    return start_zone_pct(pc_vals) / 100.0


def start_zone_settings(pc_vals: Mapping[str, Any] | None) -> dict[str, Any]:
    """``{'enabled': bool, 'pct': float}`` for reports and sidecars."""
    return {'enabled': start_zone_enabled(pc_vals), 'pct': start_zone_pct(pc_vals)}


def describe_start_zone(pc_vals: Mapping[str, Any] | None) -> str:
    """One line for the results Markdown."""
    if not start_zone_enabled(pc_vals):
        return 'off (hits counted from the origin waypoint)'
    return f'on, hits within the first {start_zone_pct(pc_vals):g} % of each leg ignored'
