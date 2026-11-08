"""Length binning of AIS pings into the traffic matrix (pure; no QGIS).

Both binners (``AIS._bin_one_ping`` and ``AisUpdateTask._bin_ping``) use
:func:`length_bin`, so a ping lands in the same column either way.  The
columns are the project's length classes (``ship_categories.length_intervals``),
not fixed 25 m steps, and the matrix has exactly one column per class: the
Traffic table's extra *Total* column is never a data column.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

UNKNOWN_LOA_M = 100.0
"""Length used when AIS reports none (NULL) or a non-positive one (0 m)."""

FALLBACK_CLASS_M = 25.0
"""Class width used when no length classes are known."""

SOG_NOT_AVAILABLE_KN = 102.3
"""AIS speed over ground 1023 (x 0.1 kn) means "not available" (ITU-R M.1371)."""


def valid_sog(value: Any) -> float | None:
    """Speed over ground in knots, or None for a missing / "not available" value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0.0 or v >= SOG_NOT_AVAILABLE_KN - 0.05:
        return None
    return v


def valid_dimension(value: Any) -> float | None:
    """Draught, beam or air draught in metres, or None when missing.

    AIS reports an unknown dimension as 0, which would pull a cell's mean down.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0.0 else None


def add_observations(cells: dict[str, Any], type_cat: int, loa_cat: int,
                     sog: Any, air_draught: Any, beam: Any, draught: Any) -> None:
    """Append one ping's valid observations to the cell lists of a traffic block."""
    for key, value in (('Speed (knots)', valid_sog(sog)),
                       ('Ship heights (meters)', valid_dimension(air_draught)),
                       ('Ship Beam (meters)', valid_dimension(beam)),
                       ('Draught (meters)', valid_dimension(draught))):
        if value is not None:
            cells[key][type_cat][loa_cat].append(value)


def interval_bounds(length_intervals: Sequence[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """``(min, max)`` per length class; a non-numeric max ("1000+") is open-ended."""
    bounds: list[tuple[float, float]] = []
    for interval in length_intervals or []:
        try:
            lo = float(interval.get('min', 0.0))
        except (TypeError, ValueError):
            lo = 0.0
        try:
            hi = float(interval.get('max'))
        except (TypeError, ValueError):
            hi = math.inf
        bounds.append((lo, hi))
    return bounds


def length_bin(loa: Any, bounds: Sequence[tuple[float, float]] | None, n_cols: int = 0) -> int:
    """Column of the length class that holds ``loa`` (``min < loa <= max``).

    A missing or non-positive LOA counts as :data:`UNKNOWN_LOA_M`.  A ship
    shorter than the first class goes to column 0, one longer than the last
    class to the last column.  Without ``bounds`` the classes are
    ``n_cols`` steps of :data:`FALLBACK_CLASS_M`.
    """
    try:
        length = float(loa)
    except (TypeError, ValueError):
        length = UNKNOWN_LOA_M
    if not math.isfinite(length) or length <= 0.0:
        length = UNKNOWN_LOA_M
    if not bounds:
        n = max(int(n_cols), 1)
        bounds = [(i * FALLBACK_CLASS_M, (i + 1) * FALLBACK_CLASS_M) for i in range(n)]
    for col, (lo, hi) in enumerate(bounds):
        if lo < length <= hi:
            return col
    return 0 if length <= bounds[0][0] else len(bounds) - 1
