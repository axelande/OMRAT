"""Waypoint registry: legs reference shared nodes instead of owning
their end coordinates.

``waypoints`` is ``{wp_id: (lon, lat)}`` and every leg in
``segment_data`` carries ``start_wp`` / ``end_wp`` naming its two nodes.
``Start_Point`` / ``End_Point`` stay on the leg as a *derived* copy so
compute, the visualisers, the report and IWRAP export are untouched.

Two directions of truth, used deliberately:

* :func:`rebuild_waypoints` -- **coordinates win**.  Used after anything
  that rewrites endpoints as text: file load, IWRAP import, close-
  waypoint merges and intersection splits from ``route_validation``.
  Existing ids are kept wherever the coordinate is unchanged so the
  registry is stable across reloads.
* :func:`move_waypoint` / :func:`merge_waypoints` /
  :func:`sync_endpoints_from_waypoints` -- **the registry wins**.  Used
  by the canvas editing code: dragging a junction vertex moves one node
  and every incident leg follows by construction.

Coordinates are EPSG:4326 ``(lon, lat)``.  Pure Python, no QGIS.
"""
from __future__ import annotations

from typing import Any, Iterable

from geometries.route_validation import format_wkt_point, haversine_m, parse_wkt_point

#: Leg fields holding the node ids.
WP_START = 'start_wp'
WP_END = 'end_wp'

#: Two coordinates closer than this (degrees, ~1 cm) are the same node.
COORD_TOL_DEG = 1e-7

XY = tuple[float, float]
Waypoints = dict[str, XY]


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def _store_xy(xy: XY) -> XY:
    """Registry coordinates are kept at the six decimals the legs' text
    endpoints use, so the two never disagree by a rounding residue."""
    return (round(float(xy[0]), 6), round(float(xy[1]), 6))


def points_equal(a: XY | None, b: XY | None, tol: float = COORD_TOL_DEG) -> bool:
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def new_waypoint_id(waypoints: Waypoints | None) -> str:
    """First integer id above every id in use (ids are integer strings)."""
    max_id = 0
    for key in (waypoints or {}):
        try:
            max_id = max(max_id, int(str(key)))
        except (TypeError, ValueError):
            continue
    return str(max_id + 1)


def find_waypoint_at(waypoints: Waypoints | None, xy: XY, tol: float = COORD_TOL_DEG) -> str | None:
    """Id of the node at exactly ``xy`` (within ``tol`` degrees), else ``None``.

    The query is rounded like a stored coordinate first, so a raw canvas
    point and its six-decimal text form resolve to the same node."""
    q = _store_xy(xy)
    for wid, wxy in (waypoints or {}).items():
        if points_equal(wxy, q, tol):
            return str(wid)
    return None


def find_waypoint_near(
    waypoints: Waypoints | None, xy: XY, tol_m: float, exclude: Iterable[str] = (),
) -> tuple[str, float] | None:
    """Nearest node within ``tol_m`` metres of ``xy`` as ``(id, distance_m)``."""
    skip = {str(e) for e in exclude}
    best: tuple[str, float] | None = None
    for wid, wxy in (waypoints or {}).items():
        if str(wid) in skip:
            continue
        d = haversine_m(wxy, xy)
        if d <= tol_m and (best is None or d < best[1]):
            best = (str(wid), d)
    return best


def ensure_waypoint(waypoints: Waypoints, xy: XY, tol: float = COORD_TOL_DEG) -> str:
    """Id of the node at ``xy``, creating it when there is none."""
    wid = find_waypoint_at(waypoints, xy, tol)
    if wid is None:
        wid = new_waypoint_id(waypoints)
        waypoints[wid] = _store_xy(xy)
    return wid


def incident_legs(segment_data: dict[str, Any] | None, wp_id: str) -> list[tuple[str, str]]:
    """``[(leg_id, 'start'|'end'), ...]`` for every leg touching ``wp_id``."""
    out: list[tuple[str, str]] = []
    for leg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        if str(seg.get(WP_START)) == str(wp_id):
            out.append((str(leg_id), 'start'))
        if str(seg.get(WP_END)) == str(wp_id):
            out.append((str(leg_id), 'end'))
    return out


def leg_endpoint_field(side: str) -> str:
    return 'Start_Point' if side == 'start' else 'End_Point'


# ---------------------------------------------------------------------------
# Coordinates -> registry
# ---------------------------------------------------------------------------


def rebuild_waypoints(
    segment_data: dict[str, Any] | None,
    waypoints: Waypoints | None = None,
    tol: float = COORD_TOL_DEG,
) -> Waypoints:
    """Derive the registry from the legs' ``Start_Point`` / ``End_Point``
    and write ``start_wp`` / ``end_wp`` on every leg, in place.

    Coincident endpoints (within ``tol``) share one node.  An id from the
    old registry is reused when its coordinate still matches, so nodes
    that did not move keep their identity; nodes no leg touches are
    dropped; new coordinates get fresh ids.  Legs whose endpoints cannot
    be parsed are left alone.
    """
    old: Waypoints = {str(k): _store_xy(v) for k, v in (waypoints or {}).items()}
    new: Waypoints = {}

    def _node_for(xy: XY, preferred: Any) -> str:
        wid = find_waypoint_at(new, xy, tol)
        if wid is not None:
            return wid
        pref = str(preferred) if preferred is not None else None
        if pref is not None and pref in old and pref not in new and points_equal(old[pref], xy, tol):
            new[pref] = old[pref]
            return pref
        for oid, oxy in old.items():
            if oid not in new and points_equal(oxy, xy, tol):
                new[oid] = oxy
                return oid
        wid = new_waypoint_id({**old, **new})
        new[wid] = _store_xy(xy)
        return wid

    for seg in (segment_data or {}).values():
        if not isinstance(seg, dict):
            continue
        sp = parse_wkt_point(seg.get('Start_Point'))
        ep = parse_wkt_point(seg.get('End_Point'))
        if sp is None or ep is None:
            continue
        seg[WP_START] = _node_for(sp, seg.get(WP_START))
        seg[WP_END] = _node_for(ep, seg.get(WP_END))
    return new


# ---------------------------------------------------------------------------
# Registry -> coordinates
# ---------------------------------------------------------------------------


def sync_endpoints_from_waypoints(segment_data: dict[str, Any] | None, waypoints: Waypoints | None) -> int:
    """Write every leg's ``Start_Point`` / ``End_Point`` from its nodes.
    Legs with an unknown node are skipped.  Returns endpoints changed."""
    wps = waypoints or {}
    changed = 0
    for seg in (segment_data or {}).values():
        if not isinstance(seg, dict):
            continue
        touched = False
        for ref, field in ((WP_START, 'Start_Point'), (WP_END, 'End_Point')):
            wid = seg.get(ref)
            if wid is None or str(wid) not in wps:
                continue
            text = format_wkt_point(*wps[str(wid)])
            if seg.get(field) != text:
                seg[field] = text
                changed += 1
                touched = True
        if touched:
            _refresh_length(seg)
    return changed


def _refresh_length(seg: dict[str, Any]) -> None:
    sp = parse_wkt_point(seg.get('Start_Point'))
    ep = parse_wkt_point(seg.get('End_Point'))
    if sp is not None and ep is not None:
        seg['line_length'] = haversine_m(sp, ep)


def move_waypoint(
    waypoints: Waypoints, segment_data: dict[str, Any] | None, wp_id: str, xy: XY,
) -> list[str]:
    """Move node ``wp_id`` to ``xy`` and every incident leg endpoint with
    it.  Returns the ids of the legs whose geometry changed."""
    wp_id = str(wp_id)
    if wp_id not in waypoints:
        return []
    waypoints[wp_id] = _store_xy(xy)
    text = format_wkt_point(*waypoints[wp_id])
    moved: list[str] = []
    for leg_id, side in incident_legs(segment_data, wp_id):
        seg = segment_data[leg_id]
        seg[leg_endpoint_field(side)] = text
        _refresh_length(seg)
        moved.append(leg_id)
    return moved


def merge_waypoints(
    waypoints: Waypoints, segment_data: dict[str, Any] | None, keep_id: str, drop_id: str,
) -> list[str]:
    """Fold node ``drop_id`` into ``keep_id``: legs on the dropped node
    are re-pointed and moved to the kept node's coordinate, and the
    dropped node is removed.  Returns the legs whose geometry changed."""
    keep_id, drop_id = str(keep_id), str(drop_id)
    if keep_id == drop_id or keep_id not in waypoints:
        return []
    text = format_wkt_point(*waypoints[keep_id])
    moved: list[str] = []
    for leg_id, side in incident_legs(segment_data, drop_id):
        seg = segment_data[leg_id]
        seg[WP_START if side == 'start' else WP_END] = keep_id
        seg[leg_endpoint_field(side)] = text
        _refresh_length(seg)
        moved.append(leg_id)
    waypoints.pop(drop_id, None)
    return moved


def prune_unused_waypoints(waypoints: Waypoints, segment_data: dict[str, Any] | None) -> list[str]:
    """Drop nodes no leg references.  Returns the removed ids."""
    used: set[str] = set()
    for seg in (segment_data or {}).values():
        if isinstance(seg, dict):
            for ref in (WP_START, WP_END):
                if seg.get(ref) is not None:
                    used.add(str(seg[ref]))
    removed = [wid for wid in list(waypoints) if str(wid) not in used]
    for wid in removed:
        waypoints.pop(wid, None)
    return removed


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def validate_waypoint_refs(segment_data: dict[str, Any] | None, waypoints: Waypoints | None) -> list[str]:
    """Human-readable problems: missing refs, unknown ids, endpoints that
    disagree with their node.  Empty list means consistent."""
    wps = waypoints or {}
    problems: list[str] = []
    for leg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        for ref, field in ((WP_START, 'Start_Point'), (WP_END, 'End_Point')):
            wid = seg.get(ref)
            if wid is None:
                problems.append(f"leg {leg_id}: no {ref}")
                continue
            if str(wid) not in wps:
                problems.append(f"leg {leg_id}: {ref}={wid} is not a waypoint")
                continue
            xy = parse_wkt_point(seg.get(field))
            if not points_equal(xy, wps[str(wid)]):
                problems.append(f"leg {leg_id}: {field} {seg.get(field)!r} != waypoint {wid} {wps[str(wid)]}")
    return problems


# ---------------------------------------------------------------------------
# File form
# ---------------------------------------------------------------------------


def waypoints_to_serializable(waypoints: Waypoints | None) -> dict[str, list[float]]:
    """``{id: [lon, lat]}`` for the ``waypoints`` block of ``.omrat``."""
    return {str(k): [float(v[0]), float(v[1])] for k, v in (waypoints or {}).items()}


def waypoints_from_serializable(block: Any) -> Waypoints:
    """Inverse of :func:`waypoints_to_serializable`; tolerant of junk."""
    out: Waypoints = {}
    if not isinstance(block, dict):
        return out
    for k, v in block.items():
        try:
            if isinstance(v, dict):
                out[str(k)] = (float(v['x']), float(v['y']))
            else:
                out[str(k)] = (float(v[0]), float(v[1]))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return out
