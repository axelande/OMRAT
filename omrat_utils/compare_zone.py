"""Compare two runs inside a common zone around the structures (QGIS-free).

Two layouts of the same area rarely cover the same area: one model
stops at the park's neighbourhood, the other runs on to the next strait.
Their totals then differ for reasons that have nothing to do with the
park.  This module compares what both runs share -- the water within
``radius_km`` of the structures (the union of the structures of run A
and run B, so a run without the park still gets the same zone):

* **Leg length** and **traffic work** (ships/year x km) inside the zone,
  on the legs the run actually calculated (suppressed legs are left out).
* **Collisions in the zone**: head-on and overtaking per leg, scaled by
  the share of the leg inside the zone (both grow with leg length);
  crossing, merging and bend at waypoints inside the zone.
* **Collisions per million ship-km** -- the collision total over the
  traffic work, which separates "more traffic modelled here" from
  "riskier geometry".
* **Allision**: the run totals.  Allision only happens at structures and
  every structure lies in the zone.

Distances use :func:`omrat_utils.compare_common.local_projector`.
:func:`build_zone_table` returns ``(rows, info)`` where rows are
``[item, run_a, run_b, delta_abs, delta_pct]`` text, like the accident
table.
"""
from __future__ import annotations

from typing import Any, Mapping

from .compare_common import finite, leg_traffic, local_projector, parse_xy

__all__ = ['DEFAULT_RADIUS_KM', 'zone_from_structures', 'zone_values', 'build_zone_table']

DEFAULT_RADIUS_KM = 20.0


def _structure_wkts(snap: Mapping[str, Any] | None) -> list[str]:
    out = []
    for row in (snap or {}).get('objects') or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3 and str(row[2]).strip():
            out.append(str(row[2]).strip())
    return out


def zone_from_structures(
    snap_a: Mapping[str, Any] | None,
    snap_b: Mapping[str, Any] | None,
    radius_km: float,
) -> tuple[Any, Any, int]:
    """``(zone_in_metres, projector, n_structures)``; zone is ``None`` without structures.

    A structure present in both runs counts once.
    """
    from shapely import wkt as _wkt
    from shapely.ops import transform, unary_union
    geoms = []
    for text in dict.fromkeys(_structure_wkts(snap_a) + _structure_wkts(snap_b)):
        try:
            geom = _wkt.loads(text)
        except Exception:  # nosec B110 B112 -- a broken WKT is skipped
            continue
        if not geom.is_empty:
            geoms.append(geom)
    if not geoms:
        return None, None, 0
    union = unary_union(geoms)
    proj = local_projector(union.centroid.y)
    zone = transform(proj, union).buffer(max(float(radius_km), 0.0) * 1000.0)
    return zone, proj, len(geoms)


def zone_values(
    snap: Mapping[str, Any] | None,
    outputs: Mapping[str, Any] | None,
    zone: Any,
    proj: Any,
) -> dict[str, float | None]:
    """The zone figures of one run; collision values are ``None`` without a collision sidecar."""
    from shapely.geometry import LineString, Point
    from shapely.ops import transform

    snap = snap or {}
    outputs = outputs or {}
    segs = {str(k): v for k, v in (snap.get('segment_data') or {}).items() if isinstance(v, Mapping)}
    traffic = snap.get('traffic_data') or {}
    leg_km = 0.0
    ship_km = 0.0
    fraction: dict[str, float] = {}
    for k, s in segs.items():
        a, b = parse_xy(s.get('Start_Point')), parse_xy(s.get('End_Point'))
        if a is None or b is None or a == b:
            continue
        line = transform(proj, LineString([a, b]))
        inside = line.intersection(zone).length
        fraction[k] = inside / line.length if line.length > 0 else 0.0
        if inside <= 0 or s.get('suppressed'):
            continue
        leg_km += inside / 1000.0
        ship_km += leg_traffic(traffic, k) * inside / 1000.0

    vals: dict[str, float | None] = {
        'leg_km': leg_km,
        'ship_km_m': ship_km / 1e6,
    }
    coll = outputs.get('collision')
    if isinstance(coll, Mapping):
        sums = dict.fromkeys(('head_on', 'overtaking', 'crossing', 'merging', 'bend'), 0.0)
        for k, rec in (coll.get('by_leg') or {}).items():
            if not isinstance(rec, Mapping):
                continue
            f = fraction.get(str(k), 0.0)
            for t in ('head_on', 'overtaking'):
                sums[t] += (finite(rec.get(t)) or 0.0) * f
        for wp, rec in (coll.get('by_waypoint') or {}).items():
            xy = parse_xy(wp)
            if xy is None or not isinstance(rec, Mapping):
                continue
            if zone.covers(Point(proj(*xy))):
                for t in ('crossing', 'merging', 'bend'):
                    sums[t] += finite(rec.get(t)) or 0.0
        vals.update(sums)
        vals['collisions'] = sum(sums.values())
        vals['rate'] = vals['collisions'] / vals['ship_km_m'] if vals['ship_km_m'] else None
    else:
        for key in ('head_on', 'overtaking', 'crossing', 'merging', 'bend', 'collisions', 'rate'):
            vals[key] = None

    drift = (outputs.get('drifting') or {}).get('totals') or {}
    vals['drift_allision'] = finite(drift.get('allision')) if 'allision' in drift else None
    powered = ((outputs.get('powered') or {}).get('allision') or {}).get('totals') or {}
    vals['powered_allision'] = finite(powered.get('allision')) if 'allision' in powered else None
    parts = [vals[k] for k in ('collisions', 'drift_allision', 'powered_allision') if vals[k] is not None]
    vals['total'] = sum(parts) if parts else None
    return vals


_ROWS: tuple[tuple[str, str, str], ...] = (
    ('leg_km', 'Leg length in zone (km)', ',.1f'),
    ('ship_km_m', 'Traffic work in zone (million ship-km/year)', '.3f'),
    ('head_on', 'Head-on collision in zone', '.3e'),
    ('overtaking', 'Overtaking collision in zone', '.3e'),
    ('crossing', 'Crossing collision in zone', '.3e'),
    ('merging', 'Merging collision in zone', '.3e'),
    ('bend', 'Bend collision in zone', '.3e'),
    ('collisions', 'All collisions in zone', '.3e'),
    ('rate', 'Collisions per million ship-km in zone', '.4g'),
    ('drift_allision', 'Drifting allision (run total)', '.3e'),
    ('powered_allision', 'Powered allision (run total)', '.3e'),
    ('total', 'Collisions in zone + allision', '.3e'),
)


def build_zone_table(
    snap_a: Mapping[str, Any] | None,
    snap_b: Mapping[str, Any] | None,
    outputs_a: Mapping[str, Any] | None,
    outputs_b: Mapping[str, Any] | None,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> tuple[list[list[str]], str]:
    """``(rows, info)``; rows ``[item, run_a, run_b, delta_abs, delta_pct]``.

    Without structures in either run the rows are empty and ``info``
    says why.
    """
    from .compare import _value_row

    try:
        zone, proj, n = zone_from_structures(snap_a, snap_b, radius_km)
    except ImportError:
        return [], 'The zone comparison needs shapely.'
    if zone is None:
        return [], ('Neither run has structures; the zone is drawn around the '
                    'structures of run A and run B.')
    va = zone_values(snap_a, outputs_a, zone, proj)
    vb = zone_values(snap_b, outputs_b, zone, proj)
    rows = [_value_row(label, va.get(key), vb.get(key), fmt) for key, label, fmt in _ROWS]
    info = (f'Zone: within {radius_km:g} km of {n} structure(s) of run A and B '
            f'({zone.area / 1e6:,.0f} km²).  Head-on and overtaking count the share of each '
            'leg inside the zone; crossing, merging and bend count waypoints inside it.')
    return rows, info
