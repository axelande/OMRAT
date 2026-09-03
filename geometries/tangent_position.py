"""Pure-geometry helpers for the movable tangent line ("width marker").

Every leg carries a ``Tangent_Pos`` value in ``segment_data``: the
fraction along the leg (``0`` = start point, ``1`` = end point) where the
perpendicular tangent line -- and therefore the AIS passage line -- is
placed.  The default ``0.5`` reproduces the historical midpoint
behaviour.

No QGIS imports: the QGIS layer code and the PostGIS query builder both
call these so the drawn tangent and the sampled cross-section stay in
step.  Feed projected (metre) coordinates to the perpendicular helpers.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_TANGENT_POS = 0.5
TANGENT_POS_KEY = 'Tangent_Pos'

XY = tuple[float, float]


def clamp_fraction(t: float) -> float:
    """Clamp ``t`` to the closed interval ``[0, 1]``."""
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return float(t)


def normalize_tangent_pos(value: Any, default: float = DEFAULT_TANGENT_POS) -> float:
    """Coerce a stored / typed ``Tangent_Pos`` into a clamped fraction.

    Accepts floats, ints and numeric strings.  ``None``, non-finite or
    unparsable input returns ``default``.
    """
    try:
        t = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(t):
        return default
    return clamp_fraction(t)


def percent_from_fraction(t: float) -> str:
    """Table text for a fraction: ``0.5 -> '50'``, ``0.333 -> '33.3'``."""
    return f"{round(clamp_fraction(t) * 100.0, 1):g}"


def fraction_from_percent(text: Any) -> float | None:
    """Parse a typed percentage (``'30'``, ``'30 %'``, ``30.0``) into a
    clamped fraction.  Returns ``None`` when the text is not a number so
    the caller can restore the previous cell value."""
    if isinstance(text, str):
        text = text.replace('%', '').replace(',', '.').strip()
        if not text:
            return None
    try:
        pct = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pct):
        return None
    return clamp_fraction(pct / 100.0)


def point_along(start: XY, end: XY, t: float) -> XY:
    """Point at fraction ``t`` of the straight segment ``start -> end``."""
    t = clamp_fraction(t)
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def project_fraction(start: XY, end: XY, point: XY) -> float | None:
    """Fraction along ``start -> end`` of the orthogonal projection of
    ``point`` onto the leg line, clamped to ``[0, 1]``.

    Returns ``None`` for a degenerate (zero-length) leg.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-18:
        return None
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    return clamp_fraction(t)


def perpendicular_through_point(
    start: XY, end: XY, half_width: float, t: float = DEFAULT_TANGENT_POS,
) -> tuple[XY, XY] | None:
    """Endpoints of the segment perpendicular to ``start -> end`` through
    the point at fraction ``t``, extending ``half_width`` to each side.

    Planar maths -- feed projected coordinates.  ``None`` for a
    degenerate leg.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return None
    px, py = -dy / length, dx / length
    mx, my = point_along(start, end, t)
    return (
        (mx - px * half_width, my - py * half_width),
        (mx + px * half_width, my + py * half_width),
    )
