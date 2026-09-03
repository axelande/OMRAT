"""Natural ordering of legs for the route table and the Traffic tab.

``LEG_1_2`` must sort before ``LEG_1_10`` and ``LEG_5_12_a`` before
``LEG_5_12_b``, so text is split into digit and non-digit runs and the
digit runs compare numerically.  Pure Python, no Qt.
"""
from __future__ import annotations

import re
from typing import Any

_CHUNK_RE = re.compile(r'(\d+)')

SORTABLE_COLUMNS: dict[int, str] = {0: 'Segment_Id', 1: 'Route_Id', 2: 'Leg_name'}


def natural_key(text: Any) -> tuple:
    """Sort key that compares embedded integers numerically and the rest
    case-insensitively.  ``None`` sorts first."""
    if text is None:
        return ()
    s = str(text)
    parts = _CHUNK_RE.split(s)
    key: list[tuple[int, Any]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))
    return tuple(key)


def leg_sort_key(seg_id: str, seg: dict[str, Any], column_key: str) -> tuple:
    """Key for one ``segment_data`` entry by ``column_key``; falls back to
    the segment id so ties are stable and deterministic."""
    if column_key == 'Segment_Id':
        primary = natural_key(seg.get('Segment_Id', seg_id))
    elif column_key == 'Route_Id':
        primary = natural_key(seg.get('Route_Id', 0))
    else:
        primary = natural_key(seg.get(column_key, ''))
    return primary, natural_key(seg_id)


def sort_segment_data(
    segment_data: dict[str, Any], column_key: str = 'Leg_name', reverse: bool = False,
) -> dict[str, Any]:
    """New dict with the same entries ordered by ``column_key``."""
    items = [(k, v) for k, v in segment_data.items() if isinstance(v, dict)]
    others = [(k, v) for k, v in segment_data.items() if not isinstance(v, dict)]
    items.sort(key=lambda kv: leg_sort_key(kv[0], kv[1], column_key), reverse=reverse)
    return dict(items + others)
