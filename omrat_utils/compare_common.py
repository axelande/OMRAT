"""Helpers shared by the Compare-tab modules (QGIS-free).

Two snapshots of *different* layouts reuse leg ids for unrelated legs
(``"9"`` is ``LEG_2_1_b`` in one file and ``LEG_4_2_a`` in the other),
so the Compare modules match a leg only when both its ``Segment_Id``
key **and** its ``Leg_name`` agree (:func:`match_legs`).  Everything
else is reported as present on one side only.

Distances use a local equirectangular projection
(:func:`local_projector`), the same idea as ``SimpleProjector`` in the
powered model, so no pyproj / QGIS is needed.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

__all__ = [
    'finite', 'parse_xy', 'leg_title', 'leg_identity', 'leg_sort_key',
    'match_legs', 'direction_frequency', 'leg_traffic', 'local_projector',
    'haversine_m', 'fmt_xy',
]


def finite(value: Any) -> float | None:
    """``float(value)`` when it is a finite number, else ``None``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_xy(text: Any) -> tuple[float, float] | None:
    """``"16.33 55.75"`` (or ``"16.33,55.75"``) -> ``(16.33, 55.75)``."""
    if not isinstance(text, str):
        return None
    parts = text.replace(',', ' ').split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def fmt_xy(xy: tuple[float, float] | None) -> str:
    return '—' if xy is None else f'{xy[0]:.6f} {xy[1]:.6f}'


def _seg(segment_data: Mapping[str, Any] | None, seg_id: Any) -> Mapping[str, Any]:
    seg = (segment_data or {}).get(str(seg_id))
    return seg if isinstance(seg, Mapping) else {}


def leg_title(segment_data: Mapping[str, Any] | None, seg_id: Any) -> str:
    """``"LEG_7_1 (17)"``; ``"Leg 17"`` when the leg has no name."""
    name = str(_seg(segment_data, seg_id).get('Leg_name') or '').strip()
    return f'{name} ({seg_id})' if name else f'Leg {seg_id}'


def leg_identity(segment_data: Mapping[str, Any] | None, seg_id: Any) -> tuple[str, str]:
    """``(seg_id, Leg_name)`` -- the key two runs must share to be the same leg."""
    return str(seg_id), str(_seg(segment_data, seg_id).get('Leg_name') or '').strip()


def leg_sort_key(seg_id: Any) -> tuple:
    text = str(seg_id)
    return (0, int(text), '') if text.isdigit() else (1, 0, text)


def match_legs(
    seg_a: Mapping[str, Any] | None, seg_b: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], list[str]]:
    """``(matched, only_a, only_b)`` leg ids, each sorted numerically.

    A leg is matched when the same id exists in both runs with the same
    ``Leg_name`` (or no name on either side).
    """
    seg_a = seg_a or {}
    seg_b = seg_b or {}
    matched: list[str] = []
    only_a: list[str] = []
    only_b: list[str] = []
    for k in sorted({str(k) for k in seg_a} | {str(k) for k in seg_b}, key=leg_sort_key):
        in_a, in_b = k in seg_a, k in seg_b
        if in_a and in_b and leg_identity(seg_a, k) == leg_identity(seg_b, k):
            matched.append(k)
            continue
        if in_a:
            only_a.append(k)
        if in_b:
            only_b.append(k)
    return matched, only_a, only_b


def direction_frequency(block: Mapping[str, Any] | None) -> float:
    """Ships per year in one ``traffic_data[seg][dir]`` block (finite, positive cells)."""
    total = 0.0
    if not isinstance(block, Mapping):
        return total
    for row in block.get('Frequency (ships/year)') or []:
        cells = row if isinstance(row, (list, tuple)) else [row]
        for cell in cells:
            v = finite(cell)
            if v is not None and v > 0:
                total += v
    return total


def leg_traffic(traffic_data: Mapping[str, Any] | None, seg_id: Any) -> float:
    """Ships per year on a leg, both directions."""
    dirs = (traffic_data or {}).get(str(seg_id)) or {}
    return sum(direction_frequency(b) for b in dirs.values()) if isinstance(dirs, Mapping) else 0.0


def local_projector(lat0: float) -> Callable[..., tuple[float, float]]:
    """Equirectangular lon/lat -> metres around latitude ``lat0``.

    Usable with ``shapely.ops.transform``.  Accurate to well under 1 %
    over the extent of an OMRAT model.
    """
    phi = math.radians(lat0)
    m_lat = 111132.954 - 559.822 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)

    def _proj(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        return x * m_lon, y * m_lat

    return _proj


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    r = 6_371_000.0
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))
