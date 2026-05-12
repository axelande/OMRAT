"""Pure-geometry route-validation primitives.

Detects two classes of authoring problems in an OMRAT segment set:

* **Close waypoints** — distinct leg endpoints that lie within ``tol_frac``
  of the shortest leg length and should be snapped to a single junction.
* **Leg crossings** — true geometric X-intersections between two legs that
  do not share an endpoint and should be split into four sub-legs at the
  intersection point.

The module is deliberately QGIS-free so it can be exercised under the
plain OSGeo4W interpreter via ``pytest --noconftest``.

Coordinates are EPSG:4326 (lon, lat); distances reported in meters use
an equirectangular approximation around the mean latitude of the legs
involved.  This matches the convention used elsewhere in the standalone
geometry modules (see ``drifting/engine.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sqrt
from typing import Any, Callable, Iterable

# Earth radius (meters), WGS84 mean.
_EARTH_R_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def parse_wkt_point(text: str | None) -> tuple[float, float] | None:
    """Parse the OMRAT segment-table point format.

    Accepts ``"lon lat"`` (the format written by ``HandleQGISIface
    .format_wkt``) and the WKT shapes ``"POINT(lon lat)"`` or
    ``"Point (lon lat)"``.  Returns ``None`` for missing or malformed
    input.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if '(' in s and ')' in s:
        s = s.split('(', 1)[1].split(')', 1)[0]
    parts = s.replace(',', ' ').split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def format_wkt_point(x: float, y: float) -> str:
    """Format ``(x, y)`` as ``"lon lat"`` with six-decimal precision.

    Mirrors ``HandleQGISIface.format_wkt`` so merged/split endpoints
    round-trip identically through ``segment_data``.
    """
    return f"{x:.6f} {y:.6f}"


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in meters between two (lon, lat) points."""
    from math import asin, sin
    lon1, lat1 = p1
    lon2, lat2 = p2
    rlat1, rlat2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_R_M * asin(sqrt(a))


def _equirect_xy(
    p: tuple[float, float], lat0_rad: float
) -> tuple[float, float]:
    """Project (lon, lat) to local equirectangular meters around ``lat0``."""
    lon, lat = p
    return (radians(lon) * _EARTH_R_M * cos(lat0_rad),
            radians(lat) * _EARTH_R_M)


# ---------------------------------------------------------------------------
# Leg helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Leg:
    """Lightweight projection of a segment_data entry used internally."""

    leg_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float

    @property
    def length_safe_m(self) -> float:
        """Use the stored length if positive; else fall back to haversine."""
        if self.length_m and self.length_m > 0:
            return float(self.length_m)
        return haversine_m(self.start, self.end)


def _legs_from_segment_data(segment_data: dict[str, Any]) -> list[_Leg]:
    """Build the lightweight leg list, skipping any unparseable rows."""
    out: list[_Leg] = []
    for leg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        sp = parse_wkt_point(seg.get('Start_Point'))
        ep = parse_wkt_point(seg.get('End_Point'))
        if sp is None or ep is None:
            continue
        try:
            length = float(seg.get('line_length', 0.0) or 0.0)
        except (TypeError, ValueError):
            length = 0.0
        out.append(_Leg(str(leg_id), sp, ep, length))
    return out


# ---------------------------------------------------------------------------
# Close-waypoint detection
# ---------------------------------------------------------------------------


@dataclass
class CloseWaypointPair:
    """Two distinct waypoint locations that should likely be merged.

    Each location is represented by the actual coordinate tuple stored on
    the affected legs.  ``leg_endpoints`` maps each location back to the
    ``(leg_id, "start"|"end")`` references that must be rewritten when the
    merge is applied.
    """

    point_a: tuple[float, float]
    point_b: tuple[float, float]
    distance_m: float
    threshold_m: float
    leg_endpoints: dict[
        tuple[float, float], list[tuple[str, str]]
    ] = field(default_factory=dict)

    @property
    def midpoint(self) -> tuple[float, float]:
        return (
            0.5 * (self.point_a[0] + self.point_b[0]),
            0.5 * (self.point_a[1] + self.point_b[1]),
        )


def _collect_endpoint_index(
    legs: Iterable[_Leg],
) -> dict[tuple[float, float], list[tuple[str, str]]]:
    """Group leg endpoints by their (lon, lat) location.

    Identical coordinates already share a junction; this index is what
    the close-waypoint detector compares across distinct locations.
    """
    idx: dict[tuple[float, float], list[tuple[str, str]]] = {}
    for leg in legs:
        idx.setdefault(leg.start, []).append((leg.leg_id, 'start'))
        idx.setdefault(leg.end, []).append((leg.leg_id, 'end'))
    return idx


def find_close_waypoint_pairs(
    segment_data: dict[str, Any],
    tol_frac: float = 0.05,
) -> list[CloseWaypointPair]:
    """Find unordered pairs of distinct waypoint locations to consider merging.

    Two locations qualify when their separation is less than ``tol_frac``
    times the shorter of the two **incident** leg lengths.  Using the
    shorter incident leg matches the IWRAP convention and avoids
    spuriously flagging two faraway endpoints just because some other
    leg in the project happens to be tiny.

    Pairs are returned sorted by ``distance_m`` ascending so the UI can
    walk the most-egregious cases first.
    """
    legs = _legs_from_segment_data(segment_data)
    if len(legs) < 2:
        return []

    endpoint_idx = _collect_endpoint_index(legs)
    locations = list(endpoint_idx.keys())
    # Map (lon, lat) -> minimum length of any leg touching that endpoint.
    loc_min_len: dict[tuple[float, float], float] = {}
    for leg in legs:
        for loc in (leg.start, leg.end):
            cur = loc_min_len.get(loc)
            if cur is None or leg.length_safe_m < cur:
                loc_min_len[loc] = leg.length_safe_m

    pairs: list[CloseWaypointPair] = []
    for i, a in enumerate(locations):
        for b in locations[i + 1:]:
            la = loc_min_len.get(a, 0.0)
            lb = loc_min_len.get(b, 0.0)
            shortest = min(la, lb) if la and lb else max(la, lb)
            if shortest <= 0:
                continue
            threshold = tol_frac * shortest
            d = haversine_m(a, b)
            if d < threshold:
                pairs.append(CloseWaypointPair(
                    point_a=a,
                    point_b=b,
                    distance_m=d,
                    threshold_m=threshold,
                    leg_endpoints={
                        a: list(endpoint_idx.get(a, [])),
                        b: list(endpoint_idx.get(b, [])),
                    },
                ))
    pairs.sort(key=lambda p: p.distance_m)
    return pairs


# ---------------------------------------------------------------------------
# Apply waypoint merge
# ---------------------------------------------------------------------------


def apply_waypoint_merge(
    segment_data: dict[str, Any],
    pair: CloseWaypointPair,
    target: tuple[float, float],
) -> int:
    """Rewrite every leg endpoint matching ``pair.point_a`` or ``pair.point_b``
    to ``target``.

    Returns the number of leg endpoints updated.  Mutates ``segment_data``
    in place — callers are expected to refresh any UI mirrors (the route
    table, offset lines) afterwards.

    The function does **not** recompute ``line_length``; that field is
    a UTM-projected meter value owned by ``HandleQGISIface`` and best
    refreshed in the calling QGIS layer.  For pure-geometry callers
    (tests, headless runs) the new haversine length is written so the
    downstream models still see a sensible number.
    """
    if not isinstance(segment_data, dict) or not segment_data:
        return 0
    src_a = pair.point_a
    src_b = pair.point_b
    target_str = format_wkt_point(*target)
    moved = 0
    affected_legs: set[str] = set()
    for end_pt in (src_a, src_b):
        for leg_id, which in pair.leg_endpoints.get(end_pt, []):
            seg = segment_data.get(leg_id)
            if not isinstance(seg, dict):
                continue
            field_name = 'Start_Point' if which == 'start' else 'End_Point'
            current = parse_wkt_point(seg.get(field_name))
            if current is None or current != end_pt:
                continue
            seg[field_name] = target_str
            moved += 1
            affected_legs.add(leg_id)
    # Refresh haversine length for legs whose endpoints actually moved.
    for leg_id in affected_legs:
        seg = segment_data.get(leg_id)
        if not isinstance(seg, dict):
            continue
        sp = parse_wkt_point(seg.get('Start_Point'))
        ep = parse_wkt_point(seg.get('End_Point'))
        if sp is not None and ep is not None:
            seg['line_length'] = haversine_m(sp, ep)
    return moved


# ---------------------------------------------------------------------------
# Leg-intersection detection
# ---------------------------------------------------------------------------


@dataclass
class LegIntersection:
    """A true X-crossing between two legs.

    ``t1`` / ``t2`` are the fractional positions along leg 1 / leg 2
    (open interval, both strictly between 0 and 1).
    """

    leg1_id: str
    leg2_id: str
    point: tuple[float, float]
    t1: float
    t2: float


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    eps: float = 1e-9,
) -> tuple[float, float, tuple[float, float]] | None:
    """Strict X-intersection test for two lon/lat segments.

    Returns ``(t1, t2, point)`` with ``0 < t1, t2 < 1`` or ``None`` if
    the segments are parallel, collinear, only meet at an endpoint, or
    miss each other entirely.  The lon/lat plane is treated as Euclidean
    here — for the proximity scales OMRAT cares about (a few hundred km
    max per leg) the curvature error is well below the 5%-of-leg snap
    tolerance.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < eps:
        return None  # parallel or collinear
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom
    # Strictly interior on both segments.
    if not (eps < t < 1 - eps) or not (eps < u < 1 - eps):
        return None
    ix = x1 + t * dx1
    iy = y1 + t * dy1
    return t, u, (ix, iy)


def find_leg_intersections(
    segment_data: dict[str, Any],
) -> list[LegIntersection]:
    """Find all leg pairs that cross at an interior point.

    Pairs that merely share an endpoint (a junction) are skipped — those
    are exactly the topologies the transition-matrix model is designed
    for.  The returned list is sorted by ``(leg1_id, leg2_id)`` for
    deterministic UI ordering.
    """
    legs = _legs_from_segment_data(segment_data)
    out: list[LegIntersection] = []
    for i, leg1 in enumerate(legs):
        for leg2 in legs[i + 1:]:
            # If the two legs share an endpoint they are a junction,
            # not a crossing.
            shared = (
                leg1.start == leg2.start
                or leg1.start == leg2.end
                or leg1.end == leg2.start
                or leg1.end == leg2.end
            )
            if shared:
                continue
            hit = _segments_intersect(
                leg1.start, leg1.end, leg2.start, leg2.end,
            )
            if hit is None:
                continue
            t1, t2, pt = hit
            out.append(LegIntersection(
                leg1_id=leg1.leg_id,
                leg2_id=leg2.leg_id,
                point=pt,
                t1=t1,
                t2=t2,
            ))
    out.sort(key=lambda x: (x.leg1_id, x.leg2_id))
    return out


# ---------------------------------------------------------------------------
# Apply intersection split
# ---------------------------------------------------------------------------


def _next_leg_id(segment_data: dict[str, Any]) -> str:
    """Pick a fresh integer-string leg id one past the current max.

    OMRAT keys legs by stringified integers (see ``HandleQGISIface
    .save_route``).  We honour that convention so the UI's twRouteList
    machinery does not have to special-case our generated ids.
    """
    max_id = 0
    for k in (segment_data or {}).keys():
        try:
            v = int(str(k))
            if v > max_id:
                max_id = v
        except ValueError:
            continue
    return str(max_id + 1)


def _split_one_leg(
    segment_data: dict[str, Any],
    leg_id: str,
    split_point: tuple[float, float],
    new_id_provider: Callable[[], str],
) -> tuple[str, str] | None:
    """Replace ``leg_id`` with two sub-legs that meet at ``split_point``.

    Returns ``(first_id, second_id)`` for the new sub-legs, or ``None``
    if the source leg cannot be parsed.  The first sub-leg keeps the
    original id (so any UI references to it stay valid); the second
    receives a freshly minted id.

    Distributions, traffic, depth/object references are *not* duplicated
    here — that's the caller's job (the UI flow needs to ask the user
    whether to copy the parent leg's traffic data into both sub-legs).
    The returned legs inherit the parent's non-geometry fields by deep
    copy so callers can subsequently overwrite anything they want.
    """
    import copy as _copy
    parent = segment_data.get(leg_id)
    if not isinstance(parent, dict):
        return None
    sp = parse_wkt_point(parent.get('Start_Point'))
    ep = parse_wkt_point(parent.get('End_Point'))
    if sp is None or ep is None:
        return None
    second_id = new_id_provider()
    first = _copy.deepcopy(parent)
    second = _copy.deepcopy(parent)
    mid_str = format_wkt_point(*split_point)
    first['End_Point'] = mid_str
    first['line_length'] = haversine_m(sp, split_point)
    first['Segment_Id'] = leg_id
    if 'Leg_name' in first:
        first['Leg_name'] = f"{first['Leg_name']}_a"
    second['Start_Point'] = mid_str
    second['End_Point'] = format_wkt_point(*ep)
    second['line_length'] = haversine_m(split_point, ep)
    second['Segment_Id'] = second_id
    if 'Leg_name' in second:
        second['Leg_name'] = f"{second['Leg_name']}_b"
    segment_data[leg_id] = first
    segment_data[second_id] = second
    return leg_id, second_id


def apply_intersection_split(
    segment_data: dict[str, Any],
    intersection: LegIntersection,
    new_id_provider: Callable[[], str] | None = None,
    traffic_data: dict[str, Any] | None = None,
) -> dict[str, tuple[str, str]]:
    """Split both legs of ``intersection`` at the crossing point.

    ``traffic_data`` is optional; when provided each sub-leg inherits the
    parent leg's directional traffic block (deep copy).  This matches
    the user's request: "the sub-legs inherit the parent's traffic" — a
    1000 ships/year leg becomes two 1000 ships/year sub-legs, and the
    transition matrix at the new junction governs how much continues
    straight versus turns.

    Returns ``{original_leg_id: (first_sub_id, second_sub_id)}`` for both
    legs involved.  The first sub-leg always reuses the parent id so any
    persistent references in the UI remain valid; the second sub-leg
    gets a freshly minted id.
    """
    import copy as _copy
    if new_id_provider is None:
        # Closure that increments after each allocation so successive
        # calls inside one validation pass don't collide.
        state = {'next': None}

        def _provider() -> str:
            if state['next'] is None:
                state['next'] = int(_next_leg_id(segment_data))
            else:
                state['next'] += 1
            return str(state['next'])

        new_id_provider = _provider

    result: dict[str, tuple[str, str]] = {}
    for leg_id in (intersection.leg1_id, intersection.leg2_id):
        split = _split_one_leg(
            segment_data, leg_id, intersection.point, new_id_provider,
        )
        if split is None:
            continue
        result[leg_id] = split
        if traffic_data is not None and leg_id in traffic_data:
            parent_traffic = _copy.deepcopy(traffic_data[leg_id])
            # First sub-leg keeps the original id; second gets the new
            # id.  Both inherit the same traffic block.
            traffic_data[split[1]] = parent_traffic
    return result


# ---------------------------------------------------------------------------
# Public top-level helper
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Summary of the validation pass over a project's routes."""

    close_pairs: list[CloseWaypointPair]
    intersections: list[LegIntersection]

    @property
    def empty(self) -> bool:
        return not self.close_pairs and not self.intersections


def validate_routes(
    segment_data: dict[str, Any],
    tol_frac: float = 0.05,
) -> ValidationReport:
    """Run both detectors and bundle the results.

    The UI driver calls this once when the user clicks ``Update all
    distributions`` and walks the report interactively.
    """
    return ValidationReport(
        close_pairs=find_close_waypoint_pairs(segment_data, tol_frac=tol_frac),
        intersections=find_leg_intersections(segment_data),
    )
