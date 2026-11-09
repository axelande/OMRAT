"""Speed restriction zones (pure; no QGIS).

A *speed restriction zone* is a polygon (EPSG:4326 WKT) in which ships sail
at most ``limit_kn``.  The zones live in the ``.omrat`` block
``speed_restriction_zones``::

    [{"id": 1, "name": "Anholt OWF", "enabled": true,
      "limit_kn": 10.0, "compliance_pct": 100.0,
      "wkt": "Polygon ((...))",
      "rules": [{"ship_type": 17, "length_min": null, "length_max": null,
                 "restricted": false, "limit_kn": null, "compliance_pct": null}]}]

Rules (who the zone applies to)
-------------------------------
A zone without rules applies to every ship.  The rules are tried from the
top and the **first rule that matches a traffic cell wins**; a cell no rule
matches gets the zone default.  A rule matches on ship type (``None`` = all
types, otherwise the row index of the traffic matrix) and on length:
``length_min`` / ``length_max`` are boundaries of the project's length
classes, and a class ``(lo, hi]`` matches when ``lo >= length_min`` and
``hi <= length_max`` (the traffic is stored per class, so a ship of 30 m
cannot be told from one of 45 m inside the class 25-50).  A matching rule
either exempts the cell (``restricted`` false) or restricts it with its own
limit / compliance, blank meaning the zone default.

Compute (:func:`apply_speed_restrictions`)
------------------------------------------
Runs on the calculation's private copy right after
:func:`compute.traffic_redirect.apply_traffic_redirects` (so missing speeds
are filled and moved traffic is restricted too).  Per traffic cell:

* the speed in a zone is ``1 / (c / min(V, limit) + (1 - c) / V)`` -- the
  compliant share ``c`` sails at the limit, the rest keeps its speed, mixed
  by the time spent (``1/V``);
* ``Speed (knots)`` becomes the **time-equivalent speed along the leg**,
  ``1 / V_leg = (1 - sum f_z) / V + sum f_z / V_z`` with ``f_z`` the share
  of the leg length inside zone ``z``.  Head-on and overtaking candidates
  (linear in ``1/V`` along the leg) and the drifting exposure hours (``L/V``)
  are then exact; the drift exposure is spread evenly along the leg (the
  user's choice: the effect follows the share of the leg in the zone);
* :data:`SPEED_AT_START_KEY` / :data:`SPEED_AT_END_KEY` hold the speed *at*
  the leg's ``Start_Point`` / ``End_Point`` (the zone speed when the
  waypoint is inside a zone, else ``V``).  Crossing, merging and bend
  collisions are counted at a waypoint and powered Cat II starts at the
  turning point, so those read the waypoint speed (:func:`speed_matrix`).

The keys only exist on the compute copy; a leg no zone touches is left
untouched, and :func:`speed_matrix` falls back to ``Speed (knots)``.

Zones may not overlap (only *enabled* zones are checked, so two
alternatives can live in one project, one switched on at a time).  Overlap
or an invalid enabled zone makes :func:`apply_speed_restrictions` raise
:class:`SpeedRestrictionError` -- no path can compute with them.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

from geometries.route_validation import parse_wkt_point

__all__ = [
    'ZONES_KEY', 'SPEED_KEY', 'SPEED_AT_START_KEY', 'SPEED_AT_END_KEY', 'FREQ_KEY',
    'DEFAULT_LIMIT_KN', 'DEFAULT_COMPLIANCE_PCT', 'SpeedRestrictionError', 'LegExposure',
    'new_zone', 'new_rule', 'normalise_zones', 'zones_from_data', 'next_zone_id',
    'zone_geometry', 'length_class_bounds', 'length_boundaries', 'rule_matches', 'cell_limit',
    'restricted_speed', 'time_equivalent_speed', 'leg_exposure', 'find_overlaps',
    'zone_problems', 'overlap_message', 'blocking_problems', 'apply_speed_restrictions',
    'speed_matrix', 'turning_point',
    'zone_leg_effects', 'describe_zone', 'describe_rule', 'zone_label', 'summary_log_lines',
    'summary_markdown', 'zone_area_km2', 'settings_view',
]

ZONES_KEY = 'speed_restriction_zones'
FREQ_KEY = 'Frequency (ships/year)'
SPEED_KEY = 'Speed (knots)'
SPEED_AT_START_KEY = 'Speed at start point (knots)'
SPEED_AT_END_KEY = 'Speed at end point (knots)'

DEFAULT_LIMIT_KN = 10.0
DEFAULT_COMPLIANCE_PCT = 100.0

OVERLAP_TOLERANCE_M2 = 1.0
"""Zones sharing an edge are fine; an intersection above this area overlaps."""

POINT_TOLERANCE_M = 1.0
"""A waypoint within this distance of a zone counts as inside it (boundary included)."""

_LENGTH_TOL = 1e-6


class SpeedRestrictionError(ValueError):
    """Enabled zones that cannot be used: overlapping or invalid input."""


# ---------------------------------------------------------------------------
# Zone / rule records
# ---------------------------------------------------------------------------

def new_rule(ship_type: int | None = None, length_min: float | None = None,
             length_max: float | None = None, restricted: bool = True,
             limit_kn: float | None = None, compliance_pct: float | None = None) -> dict[str, Any]:
    """One rule; ``None`` = all ship types / no length bound / the zone default."""
    return {'ship_type': ship_type, 'length_min': length_min, 'length_max': length_max,
            'restricted': bool(restricted), 'limit_kn': limit_kn, 'compliance_pct': compliance_pct}


def new_zone(zone_id: int, wkt: str, name: str = '', limit_kn: float = DEFAULT_LIMIT_KN,
             compliance_pct: float = DEFAULT_COMPLIANCE_PCT) -> dict[str, Any]:
    """A zone that applies to every ship (no rules), enabled."""
    return {'id': int(zone_id), 'name': name or f'Zone {int(zone_id)}', 'enabled': True,
            'limit_kn': float(limit_kn), 'compliance_pct': float(compliance_pct),
            'wkt': str(wkt or ''), 'rules': []}


def _opt_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _opt_int(value: Any) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float) -> float:
    v = _opt_float(value)
    return default if v is None else v


def _normalise_rule(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    return new_rule(
        ship_type=_opt_int(raw.get('ship_type')),
        length_min=_opt_float(raw.get('length_min')),
        length_max=_opt_float(raw.get('length_max')),
        restricted=bool(raw.get('restricted', True)),
        limit_kn=_opt_float(raw.get('limit_kn')),
        compliance_pct=_opt_float(raw.get('compliance_pct')),
    )


def normalise_zones(raw: Any) -> list[dict[str, Any]]:
    """Clean list of zones: defaults seeded, numbers coerced, ids unique.

    Entries that are not mappings are dropped; a missing or repeated id
    gets the next free one.  The input is not changed.
    """
    zones: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return zones
    used: set[int] = set()
    pending: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        zid = _opt_int(entry.get('id'))
        zone = {
            'id': zid if zid is not None and zid > 0 and zid not in used else None,
            'name': str(entry.get('name') or ''),
            'enabled': bool(entry.get('enabled', True)),
            'limit_kn': _float(entry.get('limit_kn'), DEFAULT_LIMIT_KN),
            'compliance_pct': _float(entry.get('compliance_pct'), DEFAULT_COMPLIANCE_PCT),
            'wkt': str(entry.get('wkt') or ''),
            'rules': [r for r in (_normalise_rule(x) for x in (entry.get('rules') or [])) if r is not None],
        }
        if zone['id'] is not None:
            used.add(zone['id'])
        else:
            pending.append(zone)
        zones.append(zone)
    for zone in pending:
        zone['id'] = next_zone_id(used)
        used.add(zone['id'])
    for zone in zones:
        if not zone['name']:
            zone['name'] = f"Zone {zone['id']}"
    return zones


def zones_from_data(data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The normalised ``speed_restriction_zones`` of a project dict."""
    return normalise_zones((data or {}).get(ZONES_KEY))


def next_zone_id(zones: Iterable[Any]) -> int:
    """Smallest id above every id in ``zones`` (zone dicts or plain ids)."""
    ids = [z.get('id') if isinstance(z, Mapping) else z for z in zones]
    return max([int(i) for i in ids if isinstance(i, int)] + [0]) + 1


def zone_label(zone: Mapping[str, Any]) -> str:
    """``Zone 2 "Approach N"`` -- how messages name a zone."""
    zid = zone.get('id')
    name = str(zone.get('name') or '')
    default = f'Zone {zid}'
    return default if not name or name == default else f'{default} "{name}"'


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def zone_geometry(zone: Mapping[str, Any]):
    """The zone's (multi)polygon in lon/lat, or ``None`` when it has none."""
    text = str(zone.get('wkt') or '').strip()
    if not text:
        return None
    try:
        geom = shapely_wkt.loads(text)
    except Exception:  # noqa: BLE001 - any parse failure = no polygon
        return None
    if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
        return None
    return geom


def _projector(lat0: float):
    """Equirectangular lon/lat -> metres around ``lat0`` (for shapely ``transform``)."""
    phi = math.radians(lat0)
    m_lat = 111132.954 - 559.822 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)

    def _proj(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        return x * m_lon, y * m_lat
    return _proj


@dataclass(frozen=True)
class LegExposure:
    """How one leg meets the zones."""
    shares: dict[int, float] = field(default_factory=dict)
    """Zone id -> share of the leg length inside the zone (0..1)."""
    start_zone: int | None = None
    """Zone holding the leg's ``Start_Point``, if any."""
    end_zone: int | None = None
    """Zone holding the leg's ``End_Point``, if any."""

    @property
    def inside(self) -> float:
        return min(1.0, sum(self.shares.values()))

    def zone_ids(self) -> list[int]:
        ids = set(self.shares) | {z for z in (self.start_zone, self.end_zone) if z is not None}
        return sorted(ids)


def leg_exposure(seg: Mapping[str, Any], zone_geoms: Sequence[tuple[int, Any]]) -> LegExposure | None:
    """Shares of the leg inside each zone and the zones of its two ends.

    ``zone_geoms`` = ``[(zone_id, lon/lat geometry), ...]``.  ``None`` when
    the leg has no usable endpoints or touches no zone.
    """
    start = parse_wkt_point(seg.get('Start_Point'))
    end = parse_wkt_point(seg.get('End_Point'))
    if start is None or end is None:
        return None
    proj = _projector((start[1] + end[1]) / 2.0)
    ps, pe = proj(*start), proj(*end)
    line = LineString([ps, pe])
    length = line.length
    if length <= 0.0:
        return None
    shares: dict[int, float] = {}
    start_zone = end_zone = None
    for zid, geom in zone_geoms:
        if geom is None:
            continue
        pgeom = shapely_transform(proj, geom)
        if not pgeom.is_valid:
            pgeom = pgeom.buffer(0)
        if line.distance(pgeom) > POINT_TOLERANCE_M:
            continue
        inside = line.intersection(pgeom).length / length
        if inside > 0.0:
            shares[int(zid)] = min(1.0, inside)
        if start_zone is None and pgeom.distance(Point(ps)) <= POINT_TOLERANCE_M:
            start_zone = int(zid)
        if end_zone is None and pgeom.distance(Point(pe)) <= POINT_TOLERANCE_M:
            end_zone = int(zid)
    if not shares and start_zone is None and end_zone is None:
        return None
    return LegExposure(shares=shares, start_zone=start_zone, end_zone=end_zone)


def find_overlaps(zones: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, float]]:
    """``(id_a, id_b, overlap m2)`` for every pair of enabled zones that overlap.

    Zones that only share an edge (or overlap by less than
    :data:`OVERLAP_TOLERANCE_M2`) do not count.
    """
    geoms = [(z['id'], zone_geometry(z)) for z in zones if z.get('enabled', True)]
    geoms = [(zid, g) for zid, g in geoms if g is not None]
    if len(geoms) < 2:
        return []
    lat0 = sum(g.centroid.y for _, g in geoms) / len(geoms)
    proj = _projector(lat0)
    projected = []
    for zid, g in geoms:
        pg = shapely_transform(proj, g)
        projected.append((zid, pg if pg.is_valid else pg.buffer(0)))
    found: list[tuple[int, int, float]] = []
    for i, (id_a, ga) in enumerate(projected):
        for id_b, gb in projected[i + 1:]:
            if not ga.intersects(gb):
                continue
            area = ga.intersection(gb).area
            if area > OVERLAP_TOLERANCE_M2:
                found.append((int(id_a), int(id_b), float(area)))
    return found


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def length_class_bounds(length_intervals: Sequence[Mapping[str, Any]] | None) -> list[tuple[float, float]]:
    """``(min, max)`` per length class; a non-numeric max ("1000+") is open (inf).

    Same reading as ``omrat_utils.ais_binning.interval_bounds``.
    """
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


def _project_bounds(data: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    cats = (data or {}).get('ship_categories') or {}
    return length_class_bounds(cats.get('length_intervals') if isinstance(cats, Mapping) else None)


def _project_types(data: Mapping[str, Any] | None) -> list[str]:
    cats = (data or {}).get('ship_categories') or {}
    types = cats.get('types') if isinstance(cats, Mapping) else None
    return [str(t) for t in (types or [])]


def length_boundaries(bounds: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Values a rule may use: ``(lower bounds, finite upper bounds)`` of the classes."""
    lows = sorted({float(lo) for lo, _ in bounds})
    highs = sorted({float(hi) for _, hi in bounds if math.isfinite(hi)})
    return lows, highs


def rule_matches(rule: Mapping[str, Any], ship_type: int, length_idx: int,
                 bounds: Sequence[tuple[float, float]]) -> bool:
    """Whether ``rule`` covers the traffic cell (ship type row, length column)."""
    rtype = rule.get('ship_type')
    if rtype is not None and int(rtype) != int(ship_type):
        return False
    lmin, lmax = rule.get('length_min'), rule.get('length_max')
    if lmin is None and lmax is None:
        return True
    if not 0 <= length_idx < len(bounds):
        return False
    lo, hi = bounds[length_idx]
    if lmin is not None and lo < float(lmin) - _LENGTH_TOL:
        return False
    if lmax is not None and hi > float(lmax) + _LENGTH_TOL:
        return False
    return True


def cell_limit(zone: Mapping[str, Any], ship_type: int, length_idx: int,
               bounds: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """``(limit_kn, compliance 0..1)`` for one traffic cell, ``None`` when exempt.

    The first matching rule from the top wins; no match = the zone default.
    """
    limit = float(zone.get('limit_kn', DEFAULT_LIMIT_KN))
    compliance = float(zone.get('compliance_pct', DEFAULT_COMPLIANCE_PCT))
    for rule in zone.get('rules') or []:
        if not rule_matches(rule, ship_type, length_idx, bounds):
            continue
        if not rule.get('restricted', True):
            return None
        if rule.get('limit_kn') is not None:
            limit = float(rule['limit_kn'])
        if rule.get('compliance_pct') is not None:
            compliance = float(rule['compliance_pct'])
        break
    return limit, min(1.0, max(0.0, compliance / 100.0))


# ---------------------------------------------------------------------------
# Speeds
# ---------------------------------------------------------------------------

def restricted_speed(speed_kn: float, limit_kn: float, compliance: float) -> float:
    """Speed in the zone: compliant share at ``min(V, limit)``, the rest at ``V``.

    Mixed by time (``1/V``), so the ship density in the zone is exact.  A
    missing (``inf`` / NaN / <= 0) speed, a limit at or above the speed and
    zero compliance leave the speed unchanged.
    """
    v = float(speed_kn)
    if not math.isfinite(v) or v <= 0.0 or limit_kn <= 0.0 or limit_kn >= v or compliance <= 0.0:
        return v
    c = min(1.0, float(compliance))
    return 1.0 / (c / float(limit_kn) + (1.0 - c) / v)


def time_equivalent_speed(speed_kn: float, shares: Mapping[int, float],
                          zone_speeds: Mapping[int, float]) -> float:
    """Speed giving the same time on the leg: ``1/V_leg = (1-sum f)/V + sum f/V_z``."""
    v = float(speed_kn)
    if not math.isfinite(v) or v <= 0.0 or not shares:
        return v
    inside = min(1.0, sum(shares.values()))
    inv = (1.0 - inside) / v
    for zid, share in shares.items():
        vz = float(zone_speeds.get(zid, v))
        if not math.isfinite(vz) or vz <= 0.0:
            vz = v
        inv += share / vz
    return 1.0 / inv if inv > 0.0 else v


def _cell_value(cell: Any) -> float:
    if isinstance(cell, (list, tuple, np.ndarray)):
        vals = [float(x) for x in np.asarray(cell, dtype=float).ravel()]
        return float(np.mean(vals)) if vals else math.inf
    try:
        return float(cell)
    except (TypeError, ValueError):
        return math.inf


def _rows(matrix: Any) -> list[list[float]]:
    if matrix is None:
        return []
    out: list[list[float]] = []
    for row in matrix:
        if isinstance(row, (list, tuple, np.ndarray)):
            out.append([_cell_value(c) for c in row])
        else:
            out.append([_cell_value(row)])
    return out


def _like(original: Any, rows: list[list[float]]) -> Any:
    return np.array(rows, dtype=float) if isinstance(original, np.ndarray) else rows


def _weighted_mean(freq: list[list[float]], speed: list[list[float]]) -> float | None:
    tot_q = tot_qv = 0.0
    for i, row in enumerate(speed):
        for j, v in enumerate(row):
            q = freq[i][j] if i < len(freq) and j < len(freq[i]) else 0.0
            if q > 0.0 and math.isfinite(q) and math.isfinite(v) and v > 0.0:
                tot_q += q
                tot_qv += q * v
    return tot_qv / tot_q if tot_q > 0.0 else None


def speed_matrix(dir_data: Mapping[str, Any], at: str | None = None) -> Any:
    """The speed matrix a model should read.

    ``at='start'`` / ``'end'``: the speed at the leg's ``Start_Point`` /
    ``End_Point`` (crossing, merging, bend, powered Cat II); ``None``: the
    speed along the leg.  Falls back to ``Speed (knots)`` when no zone
    touches the leg.
    """
    if at == 'start' and SPEED_AT_START_KEY in dir_data:
        return dir_data[SPEED_AT_START_KEY]
    if at == 'end' and SPEED_AT_END_KEY in dir_data:
        return dir_data[SPEED_AT_END_KEY]
    return dir_data.get(SPEED_KEY, [])


def turning_point(dir_idx: int) -> str:
    """The end where direction ``dir_idx`` turns (powered Cat II starts there).

    Direction 0 sails the drawn way (``Start_Point -> End_Point``) and turns
    at ``'end'``; direction 1 turns at ``'start'``.
    """
    return 'end' if int(dir_idx) == 0 else 'start'


def _restrict_block(block: dict[str, Any], exposure: LegExposure,
                    zones_by_id: Mapping[int, Mapping[str, Any]],
                    bounds: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
    original = block.get(SPEED_KEY)
    if original is None:
        return None
    speed = _rows(original)
    freq = _rows(block.get(FREQ_KEY))
    zone_ids = exposure.zone_ids()
    leg_rows: list[list[float]] = []
    start_rows: list[list[float]] = []
    end_rows: list[list[float]] = []
    for i, row in enumerate(speed):
        leg_row, start_row, end_row = [], [], []
        for j, v in enumerate(row):
            zone_speeds: dict[int, float] = {}
            for zid in zone_ids:
                lim = cell_limit(zones_by_id[zid], i, j, bounds)
                zone_speeds[zid] = v if lim is None else restricted_speed(v, lim[0], lim[1])
            leg_row.append(time_equivalent_speed(v, exposure.shares, zone_speeds))
            start_row.append(zone_speeds.get(exposure.start_zone, v) if exposure.start_zone is not None else v)
            end_row.append(zone_speeds.get(exposure.end_zone, v) if exposure.end_zone is not None else v)
        leg_rows.append(leg_row)
        start_rows.append(start_row)
        end_rows.append(end_row)
    block[SPEED_KEY] = _like(original, leg_rows)
    block[SPEED_AT_START_KEY] = _like(original, start_rows)
    block[SPEED_AT_END_KEY] = _like(original, end_rows)
    return {
        'before': _weighted_mean(freq, speed),
        'leg': _weighted_mean(freq, leg_rows),
        'start': _weighted_mean(freq, start_rows),
        'end': _weighted_mean(freq, end_rows),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _pct_ok(value: float | None) -> bool:
    return value is None or 0.0 <= float(value) <= 100.0


def zone_problems(zone: Mapping[str, Any], ship_types: Sequence[str] | None = None,
                  bounds: Sequence[tuple[float, float]] | None = None) -> list[str]:
    """What is wrong with one zone (empty list = usable).

    ``ship_types`` / ``bounds`` (the project's ship categories) are checked
    when given: a rule's ship type must exist and its length bounds must be
    class boundaries.
    """
    problems: list[str] = []
    geom = zone_geometry(zone)
    if geom is None:
        problems.append('has no polygon')
    elif not geom.is_valid:
        problems.append('polygon is invalid (it crosses itself)')
    limit = zone.get('limit_kn')
    if limit is None or not math.isfinite(float(limit)) or float(limit) <= 0.0:
        problems.append('speed limit must be above 0 kn')
    if not _pct_ok(zone.get('compliance_pct')):
        problems.append('compliance must be 0-100 %')
    lows, highs = length_boundaries(bounds or [])
    for k, rule in enumerate(zone.get('rules') or [], start=1):
        stype = rule.get('ship_type')
        if stype is not None and (int(stype) < 0 or (ship_types is not None and int(stype) >= len(ship_types))):
            problems.append(f'rule {k}: ship type {stype} is not in the project')
        if rule.get('limit_kn') is not None and float(rule['limit_kn']) <= 0.0:
            problems.append(f'rule {k}: speed limit must be above 0 kn')
        if not _pct_ok(rule.get('compliance_pct')):
            problems.append(f'rule {k}: compliance must be 0-100 %')
        lmin, lmax = rule.get('length_min'), rule.get('length_max')
        if lmin is not None and lmax is not None and float(lmin) >= float(lmax):
            problems.append(f'rule {k}: the length range is empty ({lmin:g}-{lmax:g} m)')
        if bounds is not None and (lmin is not None or lmax is not None):
            if not bounds:
                problems.append(f'rule {k}: the project has no length classes')
            else:
                if lmin is not None and not any(abs(float(lmin) - x) <= _LENGTH_TOL for x in lows):
                    problems.append(f'rule {k}: {float(lmin):g} m is not a length class boundary')
                if lmax is not None and not any(abs(float(lmax) - x) <= _LENGTH_TOL for x in highs):
                    problems.append(f'rule {k}: {float(lmax):g} m is not a length class boundary')
    return problems


def overlap_message(zone_a: Mapping[str, Any], zone_b: Mapping[str, Any], area_m2: float) -> str:
    return (f'{zone_label(zone_b)} overlaps {zone_label(zone_a)} ({area_m2 / 1e6:.3g} km2). '
            'Speed restriction zones cannot overlap; adjust one of them or switch one off.')


def blocking_problems(data: Mapping[str, Any] | None) -> list[str]:
    """Messages that stop a run: overlapping or invalid *enabled* zones."""
    zones = [z for z in zones_from_data(data) if z['enabled']]
    if not zones:
        return []
    by_id = {z['id']: z for z in zones}
    messages = [overlap_message(by_id[a], by_id[b], area) for a, b, area in find_overlaps(zones)]
    types, bounds = _project_types(data), _project_bounds(data)
    for zone in zones:
        for problem in zone_problems(zone, types or None, bounds):
            messages.append(f'{zone_label(zone)}: {problem}.')
    return messages


# ---------------------------------------------------------------------------
# Compute step
# ---------------------------------------------------------------------------

def apply_speed_restrictions(data: dict[str, Any]) -> dict[str, Any]:
    """Restrict the speeds of ``data['traffic_data']`` in place (compute copy only).

    Returns ``{'zones': [...enabled zones...], 'legs': [{leg, name, shares,
    start_zone, end_zone, dirs: [{dir, before, leg, start, end}]}]}``;
    both lists are empty without enabled zones.  Raises
    :class:`SpeedRestrictionError` for overlapping / invalid enabled zones.
    """
    zones = [z for z in zones_from_data(data) if z['enabled']]
    summary: dict[str, Any] = {'zones': copy.deepcopy(zones), 'legs': []}
    if not zones:
        return summary
    problems = blocking_problems(data)
    if problems:
        raise SpeedRestrictionError('\n'.join(problems))
    bounds = _project_bounds(data)
    zones_by_id = {z['id']: z for z in zones}
    geoms = [(z['id'], zone_geometry(z)) for z in zones]
    segment_data = data.get('segment_data') or {}
    traffic_data = data.get('traffic_data') or {}
    for seg_id, seg in segment_data.items():
        if not isinstance(seg, Mapping):
            continue
        exposure = leg_exposure(seg, geoms)
        if exposure is None:
            continue
        leg_dirs = traffic_data.get(seg_id)
        dirs = []
        for dir_idx, (dir_name, block) in enumerate(leg_dirs.items() if isinstance(leg_dirs, dict) else ()):
            if not isinstance(block, dict):
                continue
            means = _restrict_block(block, exposure, zones_by_id, bounds)
            if means is not None:
                dirs.append({'dir': dir_idx, 'name': str(dir_name), **means})
        summary['legs'].append({
            'leg': str(seg_id), 'name': str(seg.get('Leg_name') or seg_id),
            'shares': dict(exposure.shares), 'start_zone': exposure.start_zone,
            'end_zone': exposure.end_zone, 'dirs': dirs,
        })
    return summary


def zone_leg_effects(data: Mapping[str, Any], zone: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Effect of one zone on its own (for the dialog): the ``legs`` rows of
    :func:`apply_speed_restrictions`, computed on a private copy."""
    if zone_geometry(zone) is None:
        return []
    work = {
        'segment_data': copy.deepcopy(dict(data.get('segment_data') or {})),
        'traffic_data': copy.deepcopy(dict(data.get('traffic_data') or {})),
        'ship_categories': copy.deepcopy(data.get('ship_categories') or {}),
        ZONES_KEY: [dict(copy.deepcopy(dict(zone)), enabled=True)],
    }
    try:
        return apply_speed_restrictions(work)['legs']
    except SpeedRestrictionError:
        return []


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _len_text(rule: Mapping[str, Any]) -> str:
    lmin, lmax = rule.get('length_min'), rule.get('length_max')
    if lmin is None and lmax is None:
        return 'all lengths'
    if lmax is None:
        return f'>= {float(lmin):g} m'
    if lmin is None:
        return f'<= {float(lmax):g} m'
    return f'{float(lmin):g}-{float(lmax):g} m'


def describe_rule(rule: Mapping[str, Any], ship_types: Sequence[str] | None = None) -> str:
    """``Fishing, 25-50 m: 12 kn`` / ``Passenger, all lengths: exempt``."""
    stype = rule.get('ship_type')
    if stype is None:
        who = 'All ship types'
    elif ship_types is not None and 0 <= int(stype) < len(ship_types):
        who = str(ship_types[int(stype)])
    else:
        who = f'Ship type {stype}'
    head = f'{who}, {_len_text(rule)}'
    if not rule.get('restricted', True):
        return f'{head}: exempt'
    parts = []
    parts.append(f"{float(rule['limit_kn']):g} kn" if rule.get('limit_kn') is not None else 'zone limit')
    if rule.get('compliance_pct') is not None:
        parts.append(f"{float(rule['compliance_pct']):g} % compliance")
    return f"{head}: {', '.join(parts)}"


def describe_zone(zone: Mapping[str, Any]) -> str:
    """``Zone 1 "Anholt OWF": 10 kn, 100 % compliance, all ships``."""
    rules = zone.get('rules') or []
    who = 'all ships' if not rules else f'{len(rules)} rule{"s" if len(rules) != 1 else ""}'
    state = '' if zone.get('enabled', True) else ' (off)'
    return (f"{zone_label(zone)}{state}: {float(zone.get('limit_kn', 0.0)):g} kn, "
            f"{float(zone.get('compliance_pct', 0.0)):g} % compliance, {who}")


def _kn(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1f}'


def _zone_names(leg: Mapping[str, Any], zones_by_id: Mapping[int, Mapping[str, Any]]) -> str:
    parts = [f"{zone_label(zones_by_id.get(zid, {'id': zid}))} {100.0 * share:.0f} %"
             for zid, share in sorted(leg.get('shares', {}).items())]
    for key, end in (('start_zone', 'start'), ('end_zone', 'end')):
        zid = leg.get(key)
        if zid is not None and zid not in leg.get('shares', {}):
            parts.append(f"{zone_label(zones_by_id.get(zid, {'id': zid}))} ({end} point only)")
    return ', '.join(parts)


def zone_area_km2(zone: Mapping[str, Any]) -> float | None:
    """Area of the zone's polygon in km2 (``None`` without a polygon)."""
    geom = zone_geometry(zone)
    if geom is None:
        return None
    return float(shapely_transform(_projector(geom.centroid.y), geom).area) / 1e6


def settings_view(zones: Any, ship_types: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """The zones as ``{'Zone <id>': {field: value}}`` for a settings diff
    (Compare tab): keyed by id, the polygon as points + area, the rules as text."""
    out: dict[str, dict[str, Any]] = {}
    for zone in normalise_zones(zones):
        geom = zone_geometry(zone)
        if geom is None:
            polygon = 'none'
        else:
            n_points = len(shapely_wkt.loads(zone['wkt']).exterior.coords) if geom.geom_type == 'Polygon' else None
            area = zone_area_km2(zone) or 0.0
            polygon = (f'{n_points} points, ' if n_points else '') + f'{area:.3f} km2'
        out[f"Zone {zone['id']}"] = {
            'name': zone['name'], 'enabled': zone['enabled'], 'limit_kn': zone['limit_kn'],
            'compliance_pct': zone['compliance_pct'], 'polygon': polygon,
            'rules': [describe_rule(r, ship_types) for r in zone['rules']],
        }
    return out


def summary_log_lines(summary: Mapping[str, Any] | None) -> list[str]:
    """One line per restricted leg + direction for the QGIS log."""
    if not summary or not summary.get('zones'):
        return []
    zones_by_id = {z['id']: z for z in summary['zones']}
    lines = [f"Speed restriction zones: {len(summary['zones'])} enabled, "
             f"{len(summary.get('legs') or [])} leg(s) affected"]
    for leg in summary.get('legs') or []:
        for d in leg.get('dirs') or []:
            lines.append(
                f"  {leg['name']} dir {d['dir'] + 1} ({_zone_names(leg, zones_by_id)}): "
                f"{_kn(d.get('before'))} -> {_kn(d.get('leg'))} kn along the leg")
    return lines


def summary_markdown(summary: Mapping[str, Any] | None,
                     ship_types: Sequence[str] | None = None) -> list[str]:
    """``## Speed restriction zones`` section of the run report (empty without zones)."""
    if not summary or not summary.get('zones'):
        return []
    zones = summary['zones']
    zones_by_id = {z['id']: z for z in zones}
    md = ['## Speed restriction zones', '']
    md.append('Speeds are capped in the zones below (frequency effect only; causation and '
              'consequence do not depend on speed).  Along a leg the time-equivalent speed is '
              'used; crossing, merging, bend and powered Cat II use the speed at the waypoint.')
    md.append('')
    for zone in zones:
        md.append(f'- {describe_zone(zone)}')
        for k, rule in enumerate(zone.get('rules') or [], start=1):
            md.append(f'  - Rule {k}: {describe_rule(rule, ship_types)}')
    md.append('')
    legs = summary.get('legs') or []
    if not legs:
        md.append('_No leg passes through an enabled zone._')
        md.append('')
        return md
    md.append('| Leg | Direction | In zone | Mean speed (kn) | Along the leg | At start | At end |')
    md.append('|---|---|---|---:|---:|---:|---:|')
    for leg in legs:
        where = _zone_names(leg, zones_by_id)
        for d in leg.get('dirs') or []:
            md.append(f"| {leg['name']} | {d.get('name', d['dir'] + 1)} | {where} | {_kn(d.get('before'))} | "
                      f"{_kn(d.get('leg'))} | {_kn(d.get('start'))} | {_kn(d.get('end'))} |")
    md.append('')
    md.append('_Mean speeds are weighted by ships per year._')
    md.append('')
    return md
