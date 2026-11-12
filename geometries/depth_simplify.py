"""Simplify a depth layer before a run (pure; no QGIS).

The drift model builds one merged layer per distinct depth class and works
through every one of them for each leg, direction and wind sector, so the
depth layer drives the run time (Kattegat, 20 Oct 2026: 17 -> 11 classes
took a run from 188 to 59 minutes).  Two levels of detail are kept:

* **Everywhere, the classes down to the detail limit** -- one class below
  the deepest draught in the traffic.  That is all grounding needs: a ship
  can anchor where the depth is below ``anchor_d`` x its draught, so the
  anchoring area always encloses the grounding area and a drifting ship
  crosses it first; where it starts does not change the grounding result.
* **All classes in the drift corridors that reach a structure.**  A ship
  that can anchor anywhere between its leg and a structure never reaches
  it, and big ships anchor in 40-70 m, so every depth matters on the way.
  :func:`corridor_zone` rebuilds the drift model's own corridor
  (``compute.drift_corridor_geometry._create_drift_corridor``: the leg, 5
  weighted sigma of its lateral distribution to each side, swept the model
  reach) for every leg, travel direction and drift direction of the rose,
  keeps the corridors that touch a structure and cuts each just past the
  structure.  The model reads nothing beyond its reach (99 % repaired,
  ``compute.drift_reach``), so a longer corridor only adds margin.

The zone is exact only because the cascade clips every obstacle to the part
of the corridor ahead of the leg before it builds the shadow (KNOWN_ISSUES
11, fixed 20 Oct 2026); before, water far behind a leg could shield a
structure and no zone short of the whole strip behind the leg was exact.

:func:`simplify_depths` keeps every polygon whose class is at most the
detail limit, clips deeper polygons to the zone and drops the rest.  Depth
values are the lower bound of their band (``3`` = 3-6 m).
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
from shapely import get_num_coordinates
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from compute.drift_corridor_geometry import _create_drift_corridor
from compute.drift_reach import REACH_QUANTILE, drift_reach_m
from geometries.route_validation import leg_length_m
from geometries.shapely_compat import transform

NM_M = 1852.0
REPAIR_QUANTILES = (REACH_QUANTILE, 0.999, 0.9999)
"""Corridor lengths offered: the model reach first, longer ones for margin."""
DEFAULT_REPAIR_QUANTILE = REACH_QUANTILE
FALLBACK_SPREAD_M = 2500.0
"""Corridor half-width the drift model uses for a leg without a lateral distribution."""
MARGIN = 1.05
"""Corridors are built 5 % longer and wider than the model's (projection, rounding)."""
COMPASS = tuple(range(0, 360, 45))
_EARTH_R = 6371008.8
WKT_DECIMALS = 5


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


_INTERVAL = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$')


def depth_value(value: Any) -> float:
    """Class value of a depth cell: the number, or the *shallower* end of an
    interval -- ``"48-51"`` -> 48, ``"-10--5"`` -> -10 (``nan`` if neither).

    The one rule for a depth class: compute, the save path, the depth layer
    on the map (GEBCO fetch and project load) and the Drift Analysis
    corridors all use it, so a ship grounds on the same water everywhere.
    """
    v = _num(value)
    if math.isnan(v) and isinstance(value, str):
        m = _INTERVAL.match(value)
        if m:
            v = min(float(m.group(1)), float(m.group(2)))
    return v


def class_values(depths: Iterable[Sequence[Any]]) -> list[float]:
    """Sorted distinct depth classes of ``[id, depth, wkt]`` rows."""
    return sorted({v for row in depths if len(row) > 1 and math.isfinite(v := depth_value(row[1]))})


def max_traffic_draught(traffic_data: dict[str, Any] | None, with_squat: bool = False,
                        type_names: Sequence[str] | None = None) -> float | None:
    """Deepest finite draught of a cell that carries ships, or None.

    ``with_squat`` adds each cell's squat at its speed (``compute/squat.py``),
    as the grounding models do when a Squat box is ticked.
    """
    from compute.squat import squat_for_row
    names = list(type_names or [])
    deepest: float | None = None
    for dirs in (traffic_data or {}).values():
        if not isinstance(dirs, dict):
            continue
        for block in dirs.values():
            freq = block.get('Frequency (ships/year)') if isinstance(block, dict) else None
            draught = block.get('Draught (meters)') if isinstance(block, dict) else None
            speed = block.get('Speed (knots)') if isinstance(block, dict) else None
            if not isinstance(freq, list) or not isinstance(draught, list):
                continue
            for i, row in enumerate(freq):
                for j, q in enumerate(row if isinstance(row, list) else []):
                    q = _num(q)
                    if not (math.isfinite(q) and q > 0):
                        continue
                    try:
                        d = _num(draught[i][j])
                    except (IndexError, TypeError):
                        continue
                    if not (math.isfinite(d) and d > 0):
                        continue
                    if with_squat:
                        try:
                            v = _num(speed[i][j]) if isinstance(speed, list) else 0.0
                        except (IndexError, TypeError):
                            v = 0.0
                        d += squat_for_row(names, i, v)
                    if deepest is None or d > deepest:
                        deepest = d
    return deepest


def suggest_detail_limit(classes: Sequence[float], max_draught: float | None) -> float | None:
    """The first class deeper than the deepest draught.

    Every grounding class (below the draught) is kept, plus one class beyond
    it so that anchoring has a layer of its own before grounding.  Falls back
    to the deepest class when no class is deeper than the draught.
    """
    if not classes:
        return None
    if max_draught is None:
        return max(classes)
    deeper = [c for c in classes if c > max_draught]
    return min(deeper) if deeper else max(classes)


def _local_projection(lat0: float):
    k = math.cos(math.radians(lat0))

    def to_m(x, y, z=None):
        return (math.radians(1.0) * _EARTH_R * k * x if k else x, math.radians(1.0) * _EARTH_R * y)

    def to_deg(x, y, z=None):
        return (x / (math.radians(1.0) * _EARTH_R * k), y / (math.radians(1.0) * _EARTH_R))

    return to_m, to_deg


def lateral_spread_m(segment: dict[str, Any], direction: int) -> float:
    """Half-width of the drift model's corridor for one travel direction (0 = drawn).

    Mirrors ``DriftingModelMixin._precompute_leg_lateral_params``: 5 x the
    weight-averaged standard deviation of the lateral mixture (normal
    components and the uniform part), or :data:`FALLBACK_SPREAD_M`.
    """
    d = direction + 1
    weights: list[float] = []
    variances: list[float] = []
    for i in range(1, 4):
        w = _num(segment.get(f'weight{d}_{i}', 0.0))
        std = _num(segment.get(f'std{d}_{i}', 0.0))
        if math.isfinite(w) and w > 0 and math.isfinite(std) and std > 0:
            weights.append(w)
            variances.append(std * std)
    u_p = _num(segment.get(f'u_p{d}', 0.0))
    if math.isfinite(u_p) and u_p > 0:
        width = _num(segment.get(f'u_max{d}', 0.0)) - _num(segment.get(f'u_min{d}', 0.0))
        if math.isfinite(width):
            weights.append(u_p)
            variances.append(width * width / 12.0)
    total = sum(weights)
    if total <= 0:
        return FALLBACK_SPREAD_M
    std = math.sqrt(sum(w * v for w, v in zip(weights, variances)) / total)
    return 5.0 * std if math.isfinite(std) and std > 0 else FALLBACK_SPREAD_M


def corridor_length_m(drift: dict[str, Any] | None, segment_data: dict[str, Any] | None,
                      quantile: float = REACH_QUANTILE) -> float | None:
    """Corridor length: the drift model's reach, or the drift to ``quantile`` when that is longer.

    The model's reach is the drift distance by which 99 % of the ships are
    repaired, capped at 10 x the longest leg
    (``DriftingModelMixin._compute_reach_distance``).
    """
    legs = [leg_length_m(seg) for seg in (segment_data or {}).values() if isinstance(seg, dict)]
    cap = 10.0 * max(legs) if legs else 0.0
    model = drift_reach_m(drift, REACH_QUANTILE)
    if model is None:
        model = cap
    elif cap > 0:
        model = min(model, cap)
    if quantile > REACH_QUANTILE:
        longer = drift_reach_m(drift, quantile)
        if longer is not None:
            model = max(model, longer)
    return model if model > 0 else None


def _has_traffic(block: Any) -> bool:
    freq = block.get('Frequency (ships/year)') if isinstance(block, dict) else None
    for row in freq if isinstance(freq, list) else []:
        for q in row if isinstance(row, list) else []:
            q = _num(q)
            if math.isfinite(q) and q > 0:
                return True
    return False


def drift_lines(segment_data: dict[str, Any], traffic_data: dict[str, Any] | None
                ) -> list[tuple[str, int, LineString, float]]:
    """``(segment id, travel direction, lon/lat line, half-width m)`` for every direction with ships.

    Blocks are read by position like the compute (block 0 = drawn direction).
    """
    out: list[tuple[str, int, LineString, float]] = []
    for seg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        try:
            a = tuple(float(v) for v in str(seg['Start_Point']).split()[:2])
            b = tuple(float(v) for v in str(seg['End_Point']).split()[:2])
        except (KeyError, ValueError):
            continue
        if len(a) < 2 or len(b) < 2 or a == b:
            continue
        blocks = list(((traffic_data or {}).get(seg_id) or {}).values())
        for k in (0, 1):
            if k < len(blocks) and _has_traffic(blocks[k]):
                line = LineString([a, b] if k == 0 else [b, a])
                out.append((str(seg_id), k, line, lateral_spread_m(seg, k)))
    return out


def corridor_zone(
    lines: Sequence[tuple[str, int, LineString, float]],
    object_wkts: Iterable[str],
    reach_m: float | None,
    rose: dict[Any, Any] | None = None,
) -> tuple[BaseGeometry | None, list[dict[str, Any]]]:
    """Union of the drift corridors that reach a structure (lon/lat) and which ones they are.

    Every corridor is the drift model's own (:data:`MARGIN` longer and
    wider), built in a local metric frame on the leg's latitude and cut
    just past the farthest point of the structures it touches.  Drift
    directions with no probability in ``rose`` are skipped.
    """
    structures = []
    for text in object_wkts:
        try:
            geom = make_valid(shapely_wkt.loads(text))
        except Exception:  # nosec B112 - a broken structure WKT is skipped, not fatal
            continue
        if not geom.is_empty:
            structures.append(geom)
    if not structures or not lines or not reach_m or reach_m <= 0:
        return None, []
    probs = {int(float(k)): _num(v) for k, v in (rose or {}).items()}
    compass = [c for c in COMPASS if not probs or probs.get(c, 0.0) > 0]
    parts: list[BaseGeometry] = []
    hits: list[dict[str, Any]] = []
    length = reach_m * MARGIN
    for seg_id, direction, line, spread in lines:
        to_m, to_deg = _local_projection(line.centroid.y)
        line_m = transform(to_m, line)
        targets = unary_union([transform(to_m, g) for g in structures])
        ends = np.asarray(line_m.coords, dtype=float)
        for c in compass:
            math_angle = (90 - c) % 360
            corridor = _create_drift_corridor(line_m, math_angle, length, spread * MARGIN)
            if corridor is None or not corridor.intersects(targets):
                continue
            hit = _polygonal(corridor.intersection(targets))
            if hit is None:
                continue
            u = np.array([math.cos(math.radians(math_angle)), math.sin(math.radians(math_angle))])
            pts = np.array([xy for g in getattr(hit, 'geoms', [hit]) for xy in g.exterior.coords])
            far = float(((pts[:, None, :] - ends[None, :, :]) @ u).max())
            cut = min(length, max(far, 0.0) + (MARGIN - 1.0) * reach_m)
            trimmed = _create_drift_corridor(line_m, math_angle, cut, spread * MARGIN)
            if trimmed is None:
                continue
            parts.append(transform(to_deg, trimmed))
            hits.append({'segment': seg_id, 'direction': direction, 'compass': c})
    return (unary_union(parts) if parts else None), hits


def area_km2(geom: BaseGeometry | None) -> float:
    """Area of a lon/lat geometry in km^2 (local equirectangular frame)."""
    if geom is None or geom.is_empty:
        return 0.0
    to_m, _ = _local_projection(geom.centroid.y)
    return transform(to_m, geom).area / 1e6


def _polygonal(geom: BaseGeometry) -> BaseGeometry | None:
    """Only the polygon parts of an intersection (drop slivers that became lines/points)."""
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = [g for g in getattr(geom, 'geoms', []) if isinstance(g, (Polygon, MultiPolygon))]
    return unary_union(polys) if polys else None


def _vertices(text: str) -> int:
    try:
        return int(get_num_coordinates(shapely_wkt.loads(text)))
    except Exception:  # nosec B110 - an unreadable WKT counts as empty
        return 0


def simplify_depths(
    depths: Sequence[Sequence[Any]],
    detail_limit: float,
    keep_zone: BaseGeometry | None = None,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Return the simplified ``[id, depth, wkt]`` rows and a summary.

    Polygons with a class at most ``detail_limit`` are kept unchanged;
    deeper ones are clipped to ``keep_zone`` (their own class kept) or
    dropped when they do not reach it.
    """
    out: list[list[str]] = []
    kept = clipped = dropped = 0
    for row in depths:
        if len(row) < 3:
            continue
        depth = depth_value(row[1])
        if not math.isfinite(depth) or depth <= detail_limit:
            out.append([str(row[0]), str(row[1]), str(row[2])])
            kept += 1
            continue
        part = None
        if keep_zone is not None:
            try:
                part = _polygonal(make_valid(shapely_wkt.loads(str(row[2]))).intersection(keep_zone))
            except Exception:  # nosec B110 - unreadable deep polygon: dropped like one outside the zone
                part = None
        if part is None:
            dropped += 1
            continue
        out.append([str(row[0]), str(row[1]), shapely_wkt.dumps(part, rounding_precision=WKT_DECIMALS)])
        clipped += 1
    summary = {
        'detail_limit': float(detail_limit),
        'classes_before': class_values(depths),
        'classes_after': class_values(out),
        'polygons_before': sum(1 for r in depths if len(r) >= 3),
        'polygons_after': len(out),
        'vertices_before': sum(_vertices(str(r[2])) for r in depths if len(r) >= 3),
        'vertices_after': sum(_vertices(r[2]) for r in out),
        'kept': kept,
        'clipped': clipped,
        'dropped': dropped,
    }
    return out, summary


def describe(summary: dict[str, Any]) -> str:
    """One line for the message bar / log."""
    return (f"Depth classes {len(summary['classes_before'])} -> {len(summary['classes_after'])} "
            f"(all classes to {summary['detail_limit']:g} m, deeper ones only in the drift corridors that "
            f"reach a structure); vertices {summary['vertices_before']:,} -> {summary['vertices_after']:,}; "
            f"depth areas {summary['polygons_before']} -> {summary['polygons_after']} ({summary['clipped']} "
            f"clipped to the corridors, {summary['dropped']} removed)")
