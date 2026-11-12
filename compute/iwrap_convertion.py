# Detailed mapping between .omrat and IWRAP XML schema
"""IWRAP XML conversion module for .omrat format.

This module provides bidirectional conversion between OMRAT's .omrat JSON format
and IWRAP's XML format.

Usage Examples:
    # Export .omrat to IWRAP XML:
    write_iwrap_xml('project.omrat', 'output.xml')

    # Import IWRAP XML to .omrat:
    read_iwrap_xml('input.xml', 'project.omrat')

    # Parse XML to dictionary (without saving):
    data = parse_iwrap_xml('input.xml')

Note: The import function gracefully handles missing fields like 'dist1' and 'dist2'
      which don't exist in IWRAP XML files, preventing crashes during import.
"""
import csv
import json
import math
import os
import re
import uuid
import xml.etree.ElementTree as ET  # nosec B405 - parsing delegated to defusedxml; used only to build trees
from typing import Any, Dict, List, Optional, Tuple

# defusedxml protects against XML attacks (XXE, billion laughs, etc.) when
# we read XML files from disk.  We still use the stdlib ``ElementTree`` for
# *building* trees (ET.Element / ET.SubElement / ET.tostring) since those
# don't parse external input.
from defusedxml import ElementTree as DefusedET
from defusedxml import minidom as defused_minidom

from compute import passenger_reduction
from compute.ray_sampling import mixture_span
from geometries.distribution_curves import mixture
from geometries.model_projection import ModelProjector, projected_bearing

# Geometry fixing utilities
try:
    from shapely import wkt as shp_wkt
    from shapely.geometry import (  # noqa: E501
        Polygon as ShpPolygon,
        MultiPolygon as ShpMultiPolygon,
        GeometryCollection as ShpGeometryCollection,
        LineString as ShpLineString,
        box as shp_box,
    )
    from shapely.ops import split as shp_split
    HAVE_SHAPELY = True
except Exception:  # nosec B110 B112
    HAVE_SHAPELY = False


def parse_wkt_polygon(wkt):
    if not isinstance(wkt, str):
        return []
    wkt = wkt.strip()
    if not wkt.lower().startswith('polygon'):
        return []
    try:
        inner = wkt[wkt.find('((') + 2:wkt.rfind('))')]
        coords = []
        for pair in inner.split(','):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split()
            if len(parts) < 2:
                continue
            # WKT order: lon lat
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lat, lon))
        return coords
    except Exception:  # nosec B110 B112
        return []


def parse_point_str(point_str):
    try:
        lon_str, lat_str = point_str.strip().split()
        return float(lat_str), float(lon_str)
    except Exception:  # nosec B110 B112
        return None


def new_guid() -> str:
    return str(uuid.uuid4()).upper()


def prettify_xml(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = defused_minidom.parseString(rough_string)
    # toprettyxml with encoding returns bytes; decode and strip the BOM-style byte prefix.
    # We need encoding="UTF-8" in the declaration so IWRAP (and VSCode) recognise the file.
    raw = reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')
    # toprettyxml prepends <?xml version="1.0" encoding="utf-8"?> — normalise to uppercase UTF-8.
    return raw.replace('encoding="utf-8"', 'encoding="UTF-8"', 1)


MIN_ANCHOR_DIST_FROM_GROUND = 3
"""IWRAP's default ``min_anchor_dist_from_ground``.  OMRAT has no such rule;
the value is written explicitly (or the project's own, from an import) so
the export no longer depends on IWRAP's default."""


def _most_common(values: list[float]) -> float:
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def blackout_classes(drift: dict, type_names: list | None) -> tuple[float | None, float | None]:
    """``(blackout_roro_passenger, blackout_other)`` from OMRAT's per-type blackout rates.

    IWRAP has two classes; OMRAT keeps one rate per ship type.  The
    roro / passenger class is the one ``default_blackout_by_ship_type``
    gives the lower default; each class gets the rate most of its types
    carry.  ``None`` where the project has no per-type rates (or no type
    list) -- the caller then falls back to the legacy keys.
    """
    from compute.basic_equations import default_blackout_by_ship_type

    by_type = drift.get('blackout_by_ship_type') if isinstance(drift, dict) else None
    if not isinstance(by_type, dict) or not by_type or not type_names:
        return None, None
    low = {i for i, v in default_blackout_by_ship_type(list(type_names)).items() if v < 1.0}
    roro, other = [], []
    for k, v in by_type.items():
        try:
            idx, rate = int(k), float(v)
        except (TypeError, ValueError):
            continue
        (roro if idx in low else other).append(rate)
    return (_most_common(roro) if roro else None), (_most_common(other) if other else None)


def build_drifting(parent: ET.Element, drift: dict, type_names: list | None = None):
    drifting = ET.SubElement(parent, 'drifting')
    # Set attributes on drifting per XSD
    if (v := drift.get('anchor_p')) is not None:
        drifting.set('anchor_probability', str(v))
    elif (v := drift.get('drift_p')) is not None:
        # Backward-compatible fallback for legacy data where anchor_p may be missing.
        drifting.set('anchor_probability', str(v))
    if (v := drift.get('anchor_d')) is not None:
        drifting.set('max_anchor_depth', str(v))
    if (v := drift.get('speed') or drift.get('drift_speed')) is not None:
        drifting.set('drift_speed', str(v))
    # Blackout: both IWRAP classes from the per-type rates the drifting model
    # uses.  Only ``blackout_other`` was written before 2026-11-09, so IWRAP
    # always used its default 0.1 for ro-ro / passenger ships (KNOWN_ISSUES 20).
    roro_rate, other_rate = blackout_classes(drift, type_names)
    if other_rate is not None:
        drifting.set('blackout_other', f"{other_rate:g}")
    elif (v := drift.get('drift_blackout_other')) is not None:
        drifting.set('blackout_other', str(v))
    elif (v := drift.get('drift_p')) is not None:
        # Best effort mapping if provided
        drifting.set('blackout_other', str(v))
    if roro_rate is not None:
        drifting.set('blackout_roro_passenger', f"{roro_rate:g}")
    drifting.set('min_anchor_dist_from_ground',
                 f"{_as_float(drift.get('min_anchor_dist_from_ground', MIN_ANCHOR_DIST_FROM_GROUND)):g}")
    # Required children per XSD: repair_time and drift_directions
    repair = drift.get('repair', {}) if isinstance(drift.get('repair'), dict) else {}
    rep = ET.SubElement(drifting, 'repair_time')
    # Attributes on repair_time are optional
    for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type', 'weight']:
        if attr in repair and repair.get(attr) is not None:
            rep.set(attr, str(repair.get(attr)))
    # If OMRAT's repair dict carries no explicit IWRAP attributes, synthesize
    # them from the internal representation.  Exporting an EMPTY repair_time
    # makes IWRAP fall back to its default Normal(mean 0, std 1), which has a
    # far shorter tail than a typical lognormal -- silently changing drifting
    # results by orders of magnitude on legs with distant shallow areas.
    has_attrs = any(rep.get(a) is not None for a in
                    ['combi', 'param_0', 'param_1', 'param_2', 'type'])
    if not has_attrs:
        dist_type = str(repair.get('dist_type', '') or '')
        if repair.get('use_lognormal'):
            # OMRAT lognormal(s=std, loc, scale) -> IWRAP 3-param LogNormal
            # ("Mean / Std. Dev. / Lower bound").  Inverse of the import
            # mapping in _fill_repair_combi:
            #   adj_mean = scale * exp(s^2/2);  std = adj_mean * sqrt(exp(s^2)-1)
            try:
                s = float(repair.get('std', 0.0))
                loc = float(repair.get('loc', 0.0))
                scale = float(repair.get('scale', 1.0))
                adj_mean = scale * math.exp(s * s / 2.0)
                std_out = adj_mean * math.sqrt(math.exp(s * s) - 1.0)
                rep.set('combi', '/Mean/Std. Dev./Lower bound')
                rep.set('type', 'LogNormal')
                rep.set('param_0', str(adj_mean + loc))
                rep.set('param_1', str(std_out))
                rep.set('param_2', str(loc))
            except Exception:  # nosec B110 B112
                pass
        elif dist_type == 'normal':
            rep.set('combi', '/Mean/Std. Dev.')
            rep.set('type', 'Normal')
            rep.set('param_0', str(repair.get('norm_mean', 0.0)))
            rep.set('param_1', str(repair.get('norm_std', 1.0)))
        elif dist_type == 'weibull':
            rep.set('combi', '/Scale/Shape/Location')
            rep.set('type', 'Weibull')
            rep.set('param_0', str(repair.get('wb_scale', 1.0)))
            rep.set('param_1', str(repair.get('wb_shape', 1.0)))
            rep.set('param_2', str(repair.get('wb_loc', 0.0)))
    # Optional repair_time_func.
    # IWRAP treats presence of repair_time_func as Function mode; when a known
    # distribution is provided via repair_time attributes, omit function node.
    func = repair.get('func') if isinstance(repair, dict) else None
    has_distribution_attrs = any(
        repair.get(attr) is not None
        for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type']
    )
    if func and not has_distribution_attrs:
        rf = ET.SubElement(drifting, 'repair_time_func')
        rf.set('name', str(func))
    # Drift directions as attributes
    dd = ET.SubElement(drifting, 'drift_directions')
    rose = drift.get('rose', {}) if isinstance(drift.get('rose'), dict) else {}
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        akey = str(angle)
        if akey in rose and rose.get(akey) is not None:
            # IWRAP uses 180° offset from OMRAT compass convention
            iwrap_angle = (angle + 180) % 360
            dd.set(f'angle_{iwrap_angle}', str(rose.get(akey)))


def build_bridges(parent: ET.Element, objects: list):
    if not objects:
        return
    bridges = ET.SubElement(parent, 'bridges')
    for obj in objects:
        # Support dict or row
        if isinstance(obj, dict):
            name = str(obj.get('id', ''))
            height = obj.get('height', obj.get('heights', ''))
            polygon = obj.get('polygon', '')
        else:
            name = str(obj[0]) if len(obj) > 0 else ''
            height = obj[1] if len(obj) > 1 else ''
            polygon = obj[2] if len(obj) > 2 else ''
        bridge = ET.SubElement(bridges, 'bridge')
        bridge.set('guid', new_guid())
        bridge.set('name', name)
        bp = ET.SubElement(bridge, 'bridge_polyline')
        coords = parse_wkt_polygon(polygon)
        for lat, lon in coords:
            item = ET.SubElement(bp, 'item')
            item.set('guid', new_guid())
            if height not in (None, ''):
                item.set('height', str(height))
            item.set('lat', str(lat))
            item.set('lon', str(lon))


def build_waypoints(parent: ET.Element, segment_data: dict):
    # Build waypoints with attributes per XSD and required child leg_leg_distributions
    waypoints = ET.SubElement(parent, 'waypoints')
    added = {}
    for seg_id, seg in segment_data.items():
        for tag in ('Start_Point', 'End_Point'):
            pt = seg.get(tag)
            coords = parse_point_str(pt)
            if not coords:
                continue
            key = f"{coords[0]},{coords[1]}"
            if key in added:
                continue
            w = ET.SubElement(waypoints, 'waypoint')
            guid = new_guid()
            w.set('guid', guid)
            w.set('name', f"WP_{seg_id}_{tag}")
            w.set('latitude', str(coords[0]))
            w.set('longitude', str(coords[1]))
            w.set('bend_causation_rf', "1")
            w.set('crossing_causation_rf', "1")
            ET.SubElement(w, 'leg_leg_distributions')  # required container
            added[key] = guid
    return added


LEG_EXTENSION_M = '50000'
"""Leg extension written for a direction whose Cat II is enabled."""

POWERED_WIDTH_STEP_M = 100
"""``max_width_powered`` is rounded up to a multiple of this (metres)."""


def powered_width(seg: dict) -> int | None:
    """IWRAP ``max_width_powered`` for a leg: wide enough for every ray OMRAT's powered model casts.

    IWRAP counts powered grounding and allision only for ships within
    ``max_width_powered / 2`` of the leg line.  OMRAT casts its Cat I and
    Cat II rays over the whole lateral mixture of both directions: each
    normal component's mean +/- ``RAY_SPAN_SIGMA`` sigma and the uniform
    bounds (:func:`compute.ray_sampling.mixture_span`).  The export wrote the
    leg ``Width`` (the AIS passage line) here, so IWRAP dropped every ship
    outside it -- most of the Cat I hits of a leg whose sigma is a third of
    its width.  The IWRAP width is symmetric about the leg, so this is twice
    the largest offset of either direction, rounded up, never narrower than
    ``Width``; ``max_width`` stays ``Width``.  ``None`` when the leg has
    neither a width nor a distribution.
    """
    width = _as_float(seg.get('Width'))
    half = 0.0
    for direction in (0, 1):
        span = mixture_span(mixture(seg, direction))
        if span is not None:
            half = max(half, abs(span[0]), abs(span[1]))
    need = math.ceil(2.0 * half / POWERED_WIDTH_STEP_M) * POWERED_WIDTH_STEP_M
    out = max(width, float(need))
    return int(math.ceil(out)) if out > 0 else None


def build_legs(parent: ET.Element, segment_data: dict, waypoint_lookup: dict):
    legs = ET.SubElement(parent, 'legs')
    leg_map = {}
    for seg_id, seg in segment_data.items():
        leg = ET.SubElement(legs, 'leg')
        leg_guid = new_guid()
        leg.set('guid', leg_guid)
        leg.set('name', str(seg.get('Leg_name', '')))
        if seg.get('Width') is not None:
            leg.set('max_width', str(seg.get('Width')))
        if (w_powered := powered_width(seg)) is not None:
            leg.set('max_width_powered', str(w_powered))
        # A leg-direction with ``ai <= 0`` has Cat II switched off in OMRAT.
        # IWRAP has no such switch (``grounding_check_time = 0`` means "use
        # the global default"), so the extension past the waypoint where
        # that flow would fail to turn is zeroed instead: direction 1 runs
        # first -> last and exits past the *last* waypoint, direction 2 the
        # reverse.
        leg.set('max_extension_first', '0' if _as_float(seg.get('ai2', 0)) <= 0 else LEG_EXTENSION_M)
        leg.set('max_extension_last', '0' if _as_float(seg.get('ai1', 0)) <= 0 else LEG_EXTENSION_M)
        leg.set('max_bearing_angle', '0')
        # Waypoints GUIDs
        sp = parse_point_str(seg.get('Start_Point', ''))
        ep = parse_point_str(seg.get('End_Point', ''))
        start_key = end_key = None
        if sp:
            start_key = f"{sp[0]},{sp[1]}"
            wpg = waypoint_lookup.get(start_key, '')
            if wpg:
                leg.set('first_waypoint_guid', wpg)
        if ep:
            end_key = f"{ep[0]},{ep[1]}"
            wpg = waypoint_lookup.get(end_key, '')
            if wpg:
                leg.set('last_waypoint_guid', wpg)
        # Link to manoeuvring aspects and traffic distributions via tags
        # These will be set later by the caller using leg_map
        # Required empty tags element per XSD
        ET.SubElement(leg, 'tags')
        leg_map[str(seg_id)] = {
            'guid': leg_guid,
            'start_key': start_key,
            'end_key': end_key,
            # placeholders
            'ftl_td_guid': None,
            'ltf_td_guid': None,
        }
    return leg_map


def export_junction_rows(segment_data: dict, junctions_payload: dict | None) -> list[tuple[str, str, str, float]]:
    """``(waypoint key, in-leg, out-leg, fraction)`` for every routing row to export.

    The registry is the one a calculation uses: the stored ``junctions``
    block refreshed against ``segment_data`` (``user`` / ``ais`` matrices
    kept, geometric defaults for the rest).  A leg never routes to itself;
    a row is renormalised after a self share is dropped.  Every other leg
    of the junction gets a row, a zero share included, and the rounded
    shares of an in-leg sum to exactly 1, so IWRAP has nothing left to
    distribute automatically (see ``build_leg_leg_distributions``).  The
    waypoint key is ``"lat,lon"`` as ``build_waypoints`` writes it.
    """
    from geometries.junctions import deserialize_junctions, refresh_junction_registry

    registry = refresh_junction_registry(deserialize_junctions(junctions_payload), segment_data)
    rows: list[tuple[str, str, str, float]] = []
    for junction in registry.values():
        key = None
        for leg_id, side in junction.legs.items():
            seg = segment_data.get(leg_id) or segment_data.get(str(leg_id)) or {}
            coords = parse_point_str(seg.get('Start_Point' if side == 'start' else 'End_Point', '') or '')
            if coords:
                key = f"{coords[0]},{coords[1]}"
                break
        if key is None:
            continue
        for in_leg, row in junction.transitions.items():
            if in_leg not in junction.legs:
                continue
            row = row or {}
            outs = [str(o) for o in junction.legs if str(o) != str(in_leg)]
            shares = {o: max(0.0, float(row.get(o, 0.0) or 0.0)) for o in outs}
            total = sum(shares.values())
            if total <= 0:
                continue
            rounded = {o: round(v / total, 6) for o, v in shares.items()}
            biggest = max(rounded, key=lambda o: rounded[o])
            rounded[biggest] = round(1.0 - sum(v for o, v in rounded.items() if o != biggest), 6)
            for out_leg in outs:
                rows.append((key, str(in_leg), out_leg, rounded[out_leg]))
    return rows


def build_leg_leg_distributions(parent_waypoints: ET.Element, segment_data: dict, leg_map: dict,
                                junctions_payload: dict | None = None) -> int:
    """Write OMRAT's junction matrices as IWRAP routing rows; returns the row count.

    IWRAP routes traffic through a waypoint only with ``leg_leg_distribution``
    rows ``from_leg -> to_leg`` (a native file: in-leg -> every other leg,
    row-stochastic, never itself).  The rows are written as
    ``method="MANUAL"`` (IWRAP's "Manually specified overall percentage",
    ``fraction`` = share 0..1): a ``method="AUTOMATIC"`` row is IWRAP's own
    split -- the in-leg's remainder shared equally between its automatic
    rows (33.33 % each at a four-leg junction) -- and IWRAP ignores the
    fraction written there (checked in IWRAP's Waypoint Editor).
    The importer reads ``fraction`` on every row: in a model IWRAP has
    saved, the automatic rows hold the split IWRAP used.  Until 2026-11-09 the export wrote one
    self pair per leg start, so IWRAP saw no traffic through any waypoint:
    merging 0, bend 0, no waypoint crossing (KNOWN_ISSUES 17).
    """
    waypoint_nodes = {}
    for w_el in list(parent_waypoints.findall('waypoint')):
        lat = w_el.get('latitude')
        lon = w_el.get('longitude')
        if lat is None or lon is None:
            continue
        waypoint_nodes[f"{lat},{lon}"] = w_el

    counter = 0
    for key, in_leg, out_leg, fraction in export_junction_rows(segment_data, junctions_payload):
        w_el = waypoint_nodes.get(key)
        from_guid = (leg_map.get(in_leg) or {}).get('guid')
        to_guid = (leg_map.get(out_leg) or {}).get('guid')
        if w_el is None or not from_guid or not to_guid:
            continue
        llds = w_el.find('leg_leg_distributions')
        if llds is None:
            llds = ET.SubElement(w_el, 'leg_leg_distributions')
        counter += 1
        dist = ET.SubElement(llds, 'leg_leg_distribution')
        dist.set('guid', new_guid())
        dist.set('from_leg_uid', from_guid)
        dist.set('to_leg_uid', to_guid)
        dist.set('method', 'MANUAL')
        dist.set('name', f"WTD_{counter}")
        dist.set('season', '-1')
        dist.set('fraction', repr(round(fraction, 6)))
    return counter


def _as_float(v) -> float:
    try:
        f = float(v)
        if f != f or f in (float('inf'), float('-inf')):
            return 0.0
        return f
    except Exception:  # nosec B110 B112
        return 0.0


_MAL_DEFAULT_ATTRS = [
    ('allision_causation_rf', '1'), ('allision_drifting_rf', '1'), ('allision_no_turn_rf', '1'),
    ('aton_reduction_factor', '1'), ('grounding_causation_rf', '1'), ('grounding_check_time', '0'),
    ('grounding_drifting_rf', '1'), ('grounding_no_turn_rf', '1'), ('headon_causation_rf', '1'),
    ('limit_width', 'false'), ('overtaking_causation_rf', '1'),
]


def _fill_mal_mixed_dist(md_el: ET.Element, seg: dict, dir_num: int) -> None:
    # OMRAT keeps the mixture weights in percent; IWRAP's own files use
    # fractions (``weight="1"``) and ``_fill_segment_from_mal`` multiplies by
    # 100 on import, so export divides by 100.  Writing percent made every
    # export -> import round trip multiply the weights by 100 (KNOWN_ISSUES
    # item 3).
    #
    # Lateral sign (KNOWN_ISSUES 16).  OMRAT keeps one axis for both
    # directions: negative = starboard of the *drawn* direction.  IWRAP
    # measures each direction from its own heading: positive = starboard
    # ("towards the right side in the sailing direction").  Direction 1
    # sails the drawn way, so its offsets change sign; direction 2 sails
    # against it, so OMRAT's positive (port of the drawn direction) is
    # already its starboard and is written as stored.  Until 2026-11-09
    # direction 1 was written as is and direction 2 negated, which put both
    # directions on the port side in IWRAP.
    flip = dir_num == 1
    for i in range(1, 4):
        w = _as_float(seg.get(f'weight{dir_num}_{i}', 0))
        if w <= 0:
            continue
        w = w / 100.0
        item = ET.SubElement(md_el, 'mixed_dist_item')
        item.set('combi', '/Mean/Std. Dev.')
        mean = _as_float(seg.get(f'mean{dir_num}_{i}', 0))
        std = _as_float(seg.get(f'std{dir_num}_{i}', 0))
        std = std if std > 0 else 1e-6  # IWRAP rejects Normal Std == 0; clamp matches importer
        if flip:
            mean = -mean
        item.set('param_0', str(mean))
        item.set('param_1', str(std))
        item.set('type', 'Normal')
        item.set('weight', str(w))
    u_p = _as_float(seg.get(f'u_p{dir_num}', 0))
    if u_p > 0:
        u_item = ET.SubElement(md_el, 'mixed_dist_item')
        u_item.set('combi', '/Lower Bound/Upper Bound')
        u_min = _as_float(seg.get(f'u_min{dir_num}', 0))
        u_max = _as_float(seg.get(f'u_max{dir_num}', 0))
        if flip:
            u_min, u_max = -u_max, -u_min
        u_item.set('param_0', str(u_min))
        u_item.set('param_1', str(u_max))
        u_item.set('type', 'Uniform')
        u_item.set('weight', str(u_p / 100.0))


def _build_mal_for_direction(
    mals: ET.Element, seg: dict, seg_id, dir_suffix: str, dir_num: int,
) -> str:
    mal = ET.SubElement(mals, 'manoeuvring_aspects_leg')
    guid = '{' + new_guid() + '}'
    mal.set('guid', guid)
    mal.set('name', str(seg.get('Leg_name', f'LEG_{seg_id}_{dir_suffix}')))
    for attr, default in _MAL_DEFAULT_ATTRS:
        mal.set(attr, default)
    ai = _as_float(seg.get(f'ai{dir_num}', 0))
    if ai > 0:
        # Per-direction position check interval; 0 would make IWRAP fall
        # back to its global default, which is *not* "off" (see build_legs).
        mal.set('grounding_check_time', f'{ai:g}')
    md = ET.SubElement(mal, 'mixed_dist')
    md.set('scale', '1')
    _fill_mal_mixed_dist(md, seg, dir_num)
    return guid


def build_manoeuvring_aspects_legs(parent: ET.Element, segment_data: dict, td_guid_map: dict[str, dict[str, str]]):
    mals = ET.SubElement(parent, 'manoeuvring_aspects_legs')
    mal_guid_map: dict[str, dict[str, str]] = {}
    for seg_id, seg in segment_data.items():
        guid_ftl = _build_mal_for_direction(mals, seg, seg_id, 'FTL', 1)
        guid_ltf = _build_mal_for_direction(mals, seg, seg_id, 'LTF', 2)
        mal_guid_map[str(seg_id)] = {'ftl': guid_ftl, 'ltf': guid_ltf}
    return mal_guid_map


_ROOT_META_DEFAULTS: dict[str, str] = {
    'fv': '2', 'major': '1', 'minor': '0', 'seasons': '0', 'current_season': '-1',
    'data_job_id': '', 'denc_used': '', 'data_country': '', 'data_start': '', 'data_end': '',
    'ship_type_version': '1.0', 'trafficAdjustmentFactor': '1', 'grounding_safety_margin': '0',
    'grounding_safety_margin_by_type': 'false', 'calc_squat': 'false',
    'min_squat_area_size_ship_size_ratio': '100', 'timezone': '(GMT+01:00) Amsterdam, Berlin, Bern',
    'default_max_leg_width': '5000', 'org_model_dir': '',
    'data_work_base_dir': '', 'data_set_dir': '', 'data_import_dir': '', 'density_output_dir': '',
    'speed_density_output_dir': '', 'heatmap_output_dir': '', 'passageline_output_dir': '',
    'simulator_config_dir': '', 'simulator_output_dir': '', 'shared_bathymetry_file': '',
    'shared_structure_file': '', 'default_max_leg_extension_length': '49000', 'use_leg_interaction': 'true',
    'use_32bit_compatability_mode': '1', 'default_max_leg_interaction_angle': '3',
    'use_avoid_own_ship_algo': 'false', 'use_fixed_draught': 'false', 'use_built_in_shiptypes': 'true',
    'use_class_b': 'true', 'use_check_for_legs_on_straight_line_algo': 'false', 'use_anchor_check': 'true',
    'use_spatial_index': 'true', 'grounding_accuracy': '10', 'use_width_from_data': 'false',
    'ship_type_def_file': '', 'ship_type_def_file_base': '', 'ship_type_signature': '',
    'ship_type_def_dont_care': '', 'shared_area_store_file': '', 'height_mode': '0',
    'height_scale_factor': '1', 'height_test': '0', 'fixed_to_year_factor': '0',
    'use_fixed_to_year_factor': 'false',
    'misc': (
        'B|calc_allision|1@B|calc_area|1@B|calc_collisions|1@B|calc_crossing|1'
        '@B|calc_drifting_allision|1@B|calc_drifting_grounding|1@B|calc_grounding|1'
        '@B|calc_headon_overtaking|1@B|calc_powered_allision|1@B|calc_powered_grounding|1'
    ),
    'default_class_a_length': '50', 'default_class_b_length': '25',
    'default_class_a_type': 'Other ship', 'default_class_b_type': 'Other ship',
}


def _set_root_metadata(root: ET.Element, data: dict) -> None:
    root.set('name', data.get('project_name', 'OMRAT Project'))
    root.set('guid', new_guid())
    for k, v in _ROOT_META_DEFAULTS.items():
        root.set(k, str(data.get(k, v)))


def _wire_leg_guids(
    legs_el: ET.Element, data: dict,
    td_guid_map: dict, mal_guid_map: dict,
) -> None:
    for seg_id, seg in data.get('segment_data', {}).items():
        td_pair = td_guid_map.get(str(seg_id), {})
        mal_pair = mal_guid_map.get(str(seg_id), {})
        target_name = str(seg.get('Leg_name', ''))
        for leg_el in legs_el.findall('leg'):
            if leg_el.get('name', '') == target_name:
                if td_pair.get('ftl'):
                    leg_el.set('traffic_distribution_first_to_last_guid', td_pair['ftl'])
                if td_pair.get('ltf'):
                    leg_el.set('traffic_distribution_last_to_first_guid', td_pair['ltf'])
                if mal_pair.get('ftl'):
                    leg_el.set('man_aspects_first_to_last_guid', mal_pair['ftl'])
                if mal_pair.get('ltf'):
                    leg_el.set('man_aspects_last_to_first_guid', mal_pair['ltf'])
                break


def _structure_depth(height: Any) -> str:
    """IWRAP ``depth`` of a structure area: ``-height``, or ``-1`` without a height.

    IWRAP stores a structure's clearance as a negative depth: the areas it
    makes from a bridge polyline carry ``depth = -height`` of the span, its
    other structures (pylons) ``-1``.  The import reads it back the same way
    (``_classify_area_el``).  A height that is missing, 0 or not a number
    means "every ship hits" in OMRAT and goes out as ``-1``.
    """
    try:
        h = float(height)
    except (TypeError, ValueError):
        return '-1'
    if not math.isfinite(h) or h <= 0:
        return '-1'
    return f"{-h:g}"


def _build_objects_as_depths(objects_list: list) -> list:
    """The project's structures as IWRAP structure areas (``type="1"``).

    Each entry is ``{'id', 'depth', 'polygon', 'is_object': True}`` with the
    height as a negative depth (:func:`_structure_depth`).  Until 2026-11-12
    every structure went out as ``-1`` and its height was lost; an import
    then gave every structure a height of 1 m.
    """
    result = []
    for obj in objects_list:
        if isinstance(obj, dict):
            oid = obj.get('id', '')
            height = obj.get('height', obj.get('heights', ''))
            poly = obj.get('polygon', '')
        else:
            try:
                oid, height, poly = obj
            except Exception:  # nosec B110 B112
                continue
        result.append({'id': str(oid), 'depth': _structure_depth(height),
                       'polygon': str(poly), 'is_object': True})
    return result


# IWRAP ``causation_factors`` attribute -> (OMRAT ``pc`` keys to try, default).
#
# The first key that is present in the project's ``pc`` block wins; the
# default is used only when none of them are set, so an incomplete ``pc``
# still produces valid IWRAP XML.  The alternatives exist because the
# same quantity has been stored under more than one name over time --
# ``grounding`` is what the powered model reads, ``p_pc`` is what an
# IWRAP import writes.
#
# Keep this in step with ``cf_mapping`` in ``_parse_global_settings_el``:
# export and import must name the same IWRAP attributes, or a
# round-trip loses values.
_CF_EXPORT_MAP: dict[str, tuple[tuple[str, ...], str]] = {
    # IWRAP splits powered grounding / allision into two categories:
    #
    #   p_*_causation           Category I  -- the obstacle already lies in
    #                           the lane; no course change was required.
    #   p_*_no_turn_causation   Category II -- a turn *was* required at a
    #                           waypoint and the ship failed to make it.
    #
    # OMRAT models both (see ``compute/powered_model.py``):
    # ``pc['grounding_cat1']`` / ``pc['allision_cat1']`` are the Category-I
    # factors and ``pc['grounding']`` / ``pc['allision']`` the Category-II
    # ones, so each goes to its own attribute.
    #
    # A project written before the Category-I keys existed falls back to
    # its Category-II value on the Category-I attribute rather than to a
    # constant: IWRAP computes the Category-I geometry whatever we write
    # there, and scoring it with a factor the user never chose is how the
    # old hardcoded block produced numbers nobody could account for.
    'p_allision_causation': (('allision_cat1', 'allision'), '0.000155'),
    'p_allision_no_turn_causation': (('allision',), '0.000155'),
    'p_allision_drifting_causation': (('allision_drifting_rf',), '1'),
    'p_bend_causation': (('bend',), '0.00013'),
    'p_crossing_causation': (('crossing',), '0.00013'),
    'p_grounding_causation': (('grounding_cat1', 'grounding', 'p_pc'), '0.000155'),
    'p_grounding_no_turn_causation': (('grounding', 'p_pc'), '0.000155'),
    'p_grounding_drifting_causation': (('grounding_drifting_rf',), '1'),
    'p_headon_causation': (('headon',), '5e-05'),
    'p_merging_causation': (('merging',), '0.00013'),
    'p_overtaking_causation': (('overtaking',), '0.00011'),
}


# OMRAT ``pc`` key -> IWRAP attributes to read, in *increasing* precedence.
# Later entries overwrite earlier ones.  The Category-II keys
# (``grounding`` / ``allision``) also accept the Category-I attribute so
# an IWRAP file that sets only ``p_grounding_causation`` still yields a
# value, but the turn-failure attribute wins when both are present.  The
# Category-I keys (``*_cat1``) read only their own attribute.  Mirror of
# ``_CF_EXPORT_MAP``; changing one without the other breaks the round-trip.
_CF_IMPORT_MAP: dict[str, tuple[str, ...]] = {
    'headon': ('p_headon_causation',),
    'overtaking': ('p_overtaking_causation',),
    'crossing': ('p_crossing_causation',),
    'merging': ('p_merging_causation',),
    'bend': ('p_bend_causation',),
    'grounding': (
        'p_grounding_causation', 'p_grounding_no_turn_causation',
    ),
    'allision': (
        'p_allision_causation', 'p_allision_no_turn_causation',
    ),
    'grounding_cat1': ('p_grounding_causation',),
    'allision_cat1': ('p_allision_causation',),
    'grounding_drifting_rf': ('p_grounding_drifting_causation',),
    'allision_drifting_rf': ('p_allision_drifting_causation',),
}


def _fmt_pc(value: Any) -> str | None:
    """Render one causation factor for the XML, or None if unusable."""
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return None


def _add_global_settings(root: ET.Element, data: dict | None = None) -> None:
    """Write ``global_settings`` using the project's own causation factors.

    Before v0.14.0 this emitted hardcoded constants and ignored ``data``
    entirely, so an exported model was scored with IWRAP defaults rather
    than the values set under **Settings -> Causation Factors**.  Any
    OMRAT-vs-IWRAP comparison built on such an export was comparing two
    different sets of inputs.
    """
    pc = (data or {}).get('pc') or {}
    if not isinstance(pc, dict):
        pc = {}

    gs = ET.SubElement(root, 'global_settings')
    cf = ET.SubElement(gs, 'causation_factors')
    for attr, (keys, default) in _CF_EXPORT_MAP.items():
        rendered = None
        for key in keys:
            if key in pc:
                rendered = _fmt_pc(pc[key])
                if rendered is not None:
                    break
        cf.set(attr, rendered if rendered is not None else default)

    # Passenger ship / fast ferry reduction: the project's own factors
    # (``compute.passenger_reduction``, default 20 like IWRAP).  The export
    # hard-coded 20 until 2026-11-09 while OMRAT had no such factor, so IWRAP
    # reduced risks OMRAT never reduced (KNOWN_ISSUES 19).
    misc = ET.SubElement(gs, 'misc')
    misc.set('fastferry_reduction_factor', f"{passenger_reduction.factor(pc, passenger_reduction.FAST_FERRY_KEY):g}")
    mtbc = _fmt_pc(pc.get('mean_time_between_checks'))
    misc.set('meantime_between_checks', mtbc if mtbc is not None else '240')
    misc.set('passengership_reductionfactor',
             f"{passenger_reduction.factor(pc, passenger_reduction.PASSENGER_KEY):g}")


_BEAM_KEY = 'Ship Beam (meters)'
_FREQ_KEY = 'Frequency (ships/year)'


def _fill_missing_beams(data: dict) -> int:
    """Beam = class midpoint length / 6.5 where a cell has ships but no beam
    (the collision model's ``estimate_beam``); returns the number of cells filled."""
    intervals = (data.get('ship_categories') or {}).get('length_intervals') or []
    filled = 0
    for leg in (data.get('traffic_data') or {}).values():
        if not isinstance(leg, dict):
            continue
        for block in leg.values():
            if not isinstance(block, dict) or _BEAM_KEY not in block or _FREQ_KEY not in block:
                continue
            for i, row in enumerate(block[_FREQ_KEY]):
                for j, q in enumerate(row):
                    try:
                        beam = float(block[_BEAM_KEY][i][j])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if _sanitize_num(q) <= 0 or (math.isfinite(beam) and beam > 0):
                        continue
                    try:
                        loa = (float(intervals[j]['min']) + float(intervals[j]['max'])) / 2.0
                    except (IndexError, KeyError, TypeError, ValueError):
                        loa = 150.0
                    block[_BEAM_KEY][i][j] = round(loa / 6.5, 2)
                    filled += 1
    return filled


def _with_filled_traffic(data: dict) -> dict:
    """``data`` with a private copy of the traffic in which ``Scaling (%)`` is
    applied to the frequencies and the gaps a calculation fills are filled
    (``fill_traffic_gaps``: draught and speed; beam = L / 6.5).

    Missing values were exported as 0 until 2026-11-09, IWRAP's "not given",
    so IWRAP used its own draughts for those ships (KNOWN_ISSUES 22).
    """
    import copy
    from compute.traffic_gaps import fill_traffic_gaps
    from compute.traffic_scaling import bake_traffic_scaling

    out = dict(data)
    out['traffic_data'] = copy.deepcopy(data.get('traffic_data') or {})
    out['traffic_scaling'] = copy.deepcopy(data.get('traffic_scaling') or {})
    # IWRAP has no Scaling (%): write the frequencies the calculation uses.
    # A no-op after ``prepare_export_data`` (which bakes it before the
    # redirects, in the run's order); here for direct callers.
    bake_traffic_scaling(out)
    fill_traffic_gaps(out)
    _fill_missing_beams(out)
    return out


def generate_iwrap_xml(data: dict) -> ET.Element:
    data = _with_filled_traffic(data)
    root = ET.Element('riskmodel')
    _set_root_metadata(root, data)
    ET.SubElement(root, 'ship_type_data')
    ET.SubElement(root, 'passagelines')
    td_guid_map = build_traffic_distributions(
        root, data.get('traffic_data', {}), data.get('segment_data', {}), data.get('ship_categories'))
    wl = build_waypoints(root, data.get('segment_data', {}))
    mal_guid_map = build_manoeuvring_aspects_legs(root, data.get('segment_data', {}), td_guid_map)
    leg_map = build_legs(root, data.get('segment_data', {}), wl)
    waypoints_el = root.find('waypoints')
    if waypoints_el is not None:
        build_leg_leg_distributions(waypoints_el, data.get('segment_data', {}), leg_map, data.get('junctions'))
    legs_el = root.find('legs')
    if legs_el is not None:
        _wire_leg_guids(legs_el, data, td_guid_map, mal_guid_map)
    combined_areas = list(data.get('depths', []) or []) + _build_objects_as_depths(data.get('objects', []) or [])
    build_areas(root, combined_areas)
    ET.SubElement(root, 'traffic_areas')
    build_drifting(root, data.get('drift', {}), (data.get('ship_categories') or {}).get('types'))
    ET.SubElement(root, 'area_traffic')
    ET.SubElement(root, 'routes')
    ET.SubElement(root, 'tug_boats')
    _add_global_settings(root, data)
    return root


def _make_simple_polygons(
    parts: List[List[Tuple[float, float]]]
) -> List[List[Tuple[float, float]]]:
    if not parts or not HAVE_SHAPELY:
        return parts
    result: List[List[Tuple[float, float]]] = []
    for coords in parts:
        try:
            ring = [(lon, lat) for (lat, lon) in coords]
            poly = ShpPolygon(ring)
            geom = poly
            if (not poly.is_valid) or (not poly.is_simple):
                try:
                    from shapely import make_valid as _make_valid
                    geom = _make_valid(poly)
                except Exception:  # nosec B110 B112
                    geom = poly.buffer(0)
            if isinstance(geom, ShpPolygon):
                parts_out = [geom]
            elif isinstance(geom, ShpMultiPolygon):
                parts_out = list(geom.geoms)
            elif isinstance(geom, ShpGeometryCollection):
                parts_out = [g for g in geom.geoms if isinstance(g, ShpPolygon)]
            else:
                parts_out = []
            for p in parts_out:
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and p.area > 0:
                    ext = list(p.exterior.coords)
                    result.append([(lat, lon) for (lon, lat) in ext])
        except Exception:  # nosec B110 B112
            result.append(coords)
    return result


def _parse_area_entry(dep, idx: int) -> tuple:
    if isinstance(dep, dict):
        dep_id = str(dep.get('id', idx))
        dep_depth = str(dep.get('depth', ''))
        polygon = dep.get('polygon', '')
    else:
        dep_id = str(dep[0]) if len(dep) > 0 else str(idx)
        dep_depth = str(dep[1]) if len(dep) > 1 else ''
        polygon = dep[2] if len(dep) > 2 else ''
    if isinstance(dep, dict) and 'is_object' in dep:
        # Structures from ``_build_objects_as_depths`` say so: their depth
        # is minus the height, so "-1" alone no longer identifies them.
        return dep_id, dep_depth, polygon, bool(dep['is_object'])
    is_object_area = False
    try:
        is_object_area = float(dep_depth) == -1.0
    except Exception:  # nosec B110 B112
        is_object_area = dep_depth.strip() == '-1'
    return dep_id, dep_depth, polygon, is_object_area


def _hole_free_parts(poly) -> list:
    """Cut a polygon with holes into hole-free polygons covering the same area.

    An IWRAP ``area_polygon`` is a single ring, so a hole cannot be written.
    Each hole is opened by a vertical cut through it (a vertical line inside
    the hole's x-range always crosses it), which removes it from both pieces.
    """
    out, todo = [], [poly]
    while todo:
        piece = todo.pop()
        if not piece.interiors:
            out.append(piece)
            continue
        x = piece.interiors[0].centroid.x
        minx, miny, maxx, maxy = piece.bounds
        parts = [g for g in shp_split(piece, ShpLineString([(x, miny - 1), (x, maxy + 1)])).geoms
                 if isinstance(g, ShpPolygon) and g.area > 0]
        if len(parts) < 2:
            # Degenerate cut (the line only touches a vertex): halve by boxes.
            halves = (piece.intersection(shp_box(minx - 1, miny - 1, x, maxy + 1)),
                      piece.intersection(shp_box(x, miny - 1, maxx + 1, maxy + 1)))
            parts = [g for h in halves for g in getattr(h, 'geoms', [h])
                     if isinstance(g, ShpPolygon) and g.area > 0]
            if len(parts) < 2:
                out.append(ShpPolygon(piece.exterior))  # cannot cut: keep the outline
                continue
        todo.extend(parts)
    return out


def _wkt_area_rings(polygon: str) -> List[List[Tuple[float, float]]] | None:
    """Outer rings (lat, lon) of hole-free parts of a WKT (multi)polygon, or None.

    Every polygon is kept, holes included: the regex parsers below drop any
    polygon that has a hole, which lost 59 % of the Kattegatt depth area in
    an export -> import round trip (KNOWN_ISSUES item 10).
    """
    if not HAVE_SHAPELY:
        return None
    try:
        geom = shp_wkt.loads(polygon)
    except Exception:  # nosec B110 - not WKT: the generic parsers below handle it
        return None
    if not geom.is_valid:
        from shapely import make_valid as _make_valid
        geom = _make_valid(geom)
    rings: List[List[Tuple[float, float]]] = []
    for part in getattr(geom, 'geoms', [geom]):
        if not isinstance(part, ShpPolygon) or part.is_empty:
            continue
        for piece in _hole_free_parts(part):
            rings.append([(lat, lon) for (lon, lat) in piece.exterior.coords])
    return rings


def _collect_coords_from_polygon(polygon) -> List[List[Tuple[float, float]]]:
    if isinstance(polygon, str):
        rings = _wkt_area_rings(polygon)
        if rings is not None:
            return rings
        if polygon.strip().upper().startswith('MULTIPOLYGON'):
            return parse_wkt_multipolygon(polygon)
        coords = parse_wkt_polygon(polygon)
        if coords:
            return [coords]
        coords = parse_generic_polygon(polygon)
        return [coords] if coords else []
    if isinstance(polygon, list):
        coords = []
        for pair in polygon:
            try:
                a, b = float(pair[0]), float(pair[1])
                coords.append((b, a))  # (lat, lon) from [lon, lat] input
            except Exception:  # nosec B110 B112
                continue
        return [coords] if coords else []
    return []


def _build_area_attrs(
    dep_id: str, dep_depth: str, is_object_area: bool, part_suffix: str = '',
) -> dict:
    attrs: dict = {
        'area_style_id': '', 'causationReductionFactor': '1',
        'depth': dep_depth, 'guid': new_guid(),
        'is_line': 'false', 'is_right': 'false',
    }
    if is_object_area:
        attrs.update({
            'filename': 'object', 'name': f"object_{dep_id}",
            'structure_type': 'Other', 'style_mode': '1', 'type': '1',
        })
    else:
        attrs.update({
            'filename': 'depth', 'name': f"depth_{dep_id}{part_suffix}",
            'structure_type': '', 'style_mode': '0', 'type': '0',
        })
    return attrs


def _emit_area_polygons(
    areas: ET.Element, dep_id: str, dep_depth: str,
    is_object_area: bool, polygons: List[List[Tuple[float, float]]],
) -> None:
    if not polygons:
        area = ET.SubElement(areas, 'area_polygon')
        for k, v in _build_area_attrs(dep_id, dep_depth, is_object_area).items():
            area.set(k, v)
        ET.SubElement(area, 'polygon')
        return
    for part_idx, coords in enumerate(polygons, start=1):
        part_suffix = f"_{part_idx}" if len(polygons) > 1 else ""
        area = ET.SubElement(areas, 'area_polygon')
        for k, v in _build_area_attrs(dep_id, dep_depth, is_object_area, part_suffix).items():
            area.set(k, v)
        poly_el = ET.SubElement(area, 'polygon')
        for lat, lon in coords:
            item = ET.SubElement(poly_el, 'item')
            item.set('guid', new_guid())
            item.set('lat', str(lat))
            item.set('lon', str(lon))


def build_areas(parent: ET.Element, depths: list):
    if not depths:
        return
    areas = ET.SubElement(parent, 'areas')
    for idx, dep in enumerate(depths):
        dep_id, dep_depth, polygon, is_object_area = _parse_area_entry(dep, idx)
        polygons = _make_simple_polygons(_collect_coords_from_polygon(polygon))
        _emit_area_polygons(areas, dep_id, dep_depth, is_object_area, polygons)


def parse_generic_polygon(s: str) -> list:
    """Parse generic polygon coordinate strings.

    Supports patterns like:
    - "lon lat; lon lat; ..." (semicolon or comma separated)
    - "lat lon; lat lon; ..."
    - "lon,lat; lon,lat" with commas inside pairs
    Returns list of (lat, lon).
    """
    coords: list[tuple[float, float]] = []
    if not s:
        return coords
    # Normalize separators: replace commas between numbers with spaces, then split by ';'
    try:
        parts = [p.strip() for p in s.replace(',', ' ').split(';') if p.strip()]
        for p in parts:
            nums = [n for n in p.split() if n]
            if len(nums) < 2:
                continue
            # Assume order: lon lat (WKT convention)
            lon, lat = float(nums[0]), float(nums[1])
            lat, lon = float(lat), float(lon)
            coords.append((lat, lon))
    except Exception:  # nosec B110 B112
        return []
    return coords


def parse_wkt_multipolygon(s: str) -> list:
    """Parse WKT MULTIPOLYGON into list of polygons, each a list of (lat, lon).

    Supports formats like:
    MULTIPOLYGON(((lon lat, lon lat, ...)), ((lon lat, ...)))
    Returns list of outer rings only.
    """
    s = s.strip()
    polygons: list[list[tuple[float, float]]] = []
    if not s.upper().startswith('MULTIPOLYGON'):
        return polygons
    try:
        # Extract the content within MULTIPOLYGON( ... )
        start = s.upper().find('MULTIPOLYGON')
        open_paren = s.find('(', start)
        close_paren = s.rfind(')')
        inner = s[open_paren + 1:close_paren].strip()

        # Find all outer rings: sequences wrapped as (( ... )) ignoring inner holes
        import re
        rings = re.findall(r"\(\(\s*([^\)]+?)\s*\)\)", inner)
        for ring in rings:
            coords: list[tuple[float, float]] = []
            for pair in ring.split(','):
                nums = pair.strip().split()
                if len(nums) < 2:
                    continue
                # WKT order: lon lat
                lon, lat = float(nums[0]), float(nums[1])
                coords.append((lat, lon))
            if coords:
                polygons.append(coords)
    except Exception:  # nosec B110 B112
        return []
    return polygons


# ── build_traffic_distributions helpers ───────────────────────────────────────

def _round2(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:  # nosec B110 B112
        return "0"


def _sanitize_num(v) -> float:
    try:
        f = float(v.lower()) if isinstance(v, str) else float(v)
        return 0.0 if (f != f or f in (float('inf'), float('-inf'))) else f
    except Exception:  # nosec B110 B112
        return 0.0


def _get_matrix(direction_dict: dict, key: str):
    if not isinstance(direction_dict, dict):
        return None
    return direction_dict.get(key) or direction_dict.get(key.replace('_', ' '))


def _add_category_attr(categories_el: ET.Element, cat: dict):
    c_el = ET.SubElement(categories_el, 'category')
    c_el.set('causation_reduction_factor', '0')
    c_el.set('depth', _round2(cat.get('depth', 0)))
    c_el.set('draught', _round2(cat.get('draft', cat.get('draught', 0))))
    try:
        freq_val = int(round(float(cat.get('freq', 0))))
    except Exception:  # nosec B110 B112
        freq_val = 0
    c_el.set('freq', str(freq_val))
    c_el.set('height_1', _round2(cat.get('height', cat.get('height_1', 0))))
    c_el.set('height_2', _round2(cat.get('height_2', 0)))
    c_el.set('height_3', _round2(cat.get('height_3', 0)))
    c_el.set('name', str(cat.get('name', 'Unknown')))
    c_el.set('p_ballast', _round2(cat.get('p_ballast', 0)))
    speed = cat.get('speed')
    if speed is None:
        sm = cat.get('speed_mean')
        if sm is not None:
            speed = sm
        else:
            smin, smax = cat.get('speed_min'), cat.get('speed_max')
            try:
                speed = (float(smin) + float(smax)) / 2.0 if smin is not None and smax is not None else 0
            except Exception:  # nosec B110 B112
                speed = 0
    c_el.set('speed', _round2(speed))
    c_el.set('width', _round2(cat.get('beam', cat.get('width', 0))))


def _create_td_element(td_root: ET.Element, leg_key, direction_name: str) -> tuple[str, ET.Element]:
    td = ET.SubElement(td_root, 'traffic_distribution')
    guid = '{' + new_guid() + '}'
    td.set('adjustment_factor', '1')
    td.set('dont_use', 'false')
    td.set('guid', guid)
    td.set('name', f"TD_{leg_key}_{direction_name}")
    td.set('only_uniform', 'false')
    td.set('season', '-1')
    return guid, ET.SubElement(td, 'shiptypes')


_UI_TO_IWRAP: dict[str, str] = {
    'Fishing': 'Fishing ship', 'Towing': 'Support ship',
    'Dredging or underwater ops': 'Support ship', 'Diving ops': 'Support ship',
    'Military ops': 'Other ship', 'Sailing': 'Pleasure boat',
    'Pleasure Craft': 'Pleasure boat', 'High speed craft (HSC)': 'Fast ferry',
    'Pilot Vessel': 'Support ship', 'Search and Rescue vessel': 'Support ship',
    'Tug': 'Support ship', 'Port Tender': 'Support ship',
    'Anti-pollution equipment': 'Support ship', 'Law Enforcement': 'Support ship',
    'Spare': 'Other ship', 'Medical Transport': 'Support ship',
    'Noncombatant ship according to RR Resolution No. 18': 'Other ship',
    'Passenger, all ships of this type': 'Passenger ship',
    'Cargo, all ships of this type': 'General cargo ship',
    'Tanker, all ships of this type': 'Oil products tanker',
    'Other Type, all ships of this type': 'Other ship',
}

# All 14 IWRAP ship types in canonical order — always emitted even when empty,
# so IWRAP can read the distribution without falling back to re-extraction.
_IWRAP_SHIP_TYPE_ORDER: list[str] = [
    'Crude oil tanker', 'Oil products tanker', 'Chemical tanker', 'Gas tanker',
    'Container ship', 'General cargo ship', 'Bulk carrier', 'Ro-Ro cargo ship',
    'Passenger ship', 'Fast ferry', 'Support ship', 'Fishing ship',
    'Pleasure boat', 'Other ship',
]


def _mat_safe(mat, r: int, c: int) -> float:
    return _sanitize_num(mat[r][c]) if mat and r < len(mat) and c < len(mat[r]) else 0.0


def _weighted_mean(pairs) -> float:
    """Ship-weighted mean of ``(ships, value)`` pairs.

    A value that is not positive is unknown (``_sanitize_num`` turns ``inf``
    into 0, and 0 is IWRAP's "not given") and is left out; so are rows
    without ships.  0.0 when nothing is known.
    """
    known = [(w, v) for w, v in pairs if w > 0 and v > 0]
    total = sum(w for w, _ in known)
    return sum(w * v for w, v in known) / total if total > 0 else 0.0


def _mat_avg(mat, freq_mat, col: int, rows: list) -> float:
    """Mean of ``mat[r][col]`` over the OMRAT ``rows`` of one IWRAP group,
    weighted by their ships (KNOWN_ISSUES item 2: it was a plain mean over
    every row, empty ones included, so a group of ten OMRAT rows with one
    row of traffic exported a tenth of its speed)."""
    pairs = [(_mat_safe(freq_mat, r, col), _mat_safe(mat, r, col)) for r in rows]
    return round(_weighted_mean(pairs), 2)


def _emit_shiptypes_prebuilt(shiptypes_el: ET.Element, leg_shiptypes: list):
    for st in leg_shiptypes:
        st_el = ET.SubElement(shiptypes_el, 'shiptype')
        st_el.set('causation_reduction_factor', str(st.get('causation_reduction_factor', 0)))
        st_el.set('freq_adjustment', str(st.get('freq_adjustment', 1)))
        st_el.set('grounding_safety_margin', str(st.get('grounding_safety_margin', -1)))
        st_el.set('name', str(st.get('name', 'Unknown')))
        cats_el = ET.SubElement(st_el, 'categories')
        for cat in (st.get('categories', []) or []):
            if isinstance(cat, dict):
                _add_category_attr(cats_el, cat)


def _emit_shiptypes_from_matrices(shiptypes_el: ET.Element, freq_mat, speed_mat, draft_mat, height_mat, beam_mat):
    if not freq_mat:
        return
    st_el = ET.SubElement(shiptypes_el, 'shiptype')
    for attr, val in [('causation_reduction_factor', '0'), ('freq_adjustment', '1'),
                      ('grounding_safety_margin', '-1'), ('name', 'General cargo ship')]:
        st_el.set(attr, val)
    cats_el = ET.SubElement(st_el, 'categories')
    for r, row in enumerate(freq_mat):
        for c, fv in enumerate(row):
            if _sanitize_num(fv) <= 0:
                continue
            _add_category_attr(cats_el, {
                'name': f'cat_{r}_{c}', 'freq': _sanitize_num(fv),
                'speed': _mat_safe(speed_mat, r, c), 'draught': _mat_safe(draft_mat, r, c),
                'height_1': _mat_safe(height_mat, r, c), 'width': _mat_safe(beam_mat, r, c),
                'depth': 0, 'p_ballast': 0,
            })


IWRAP_LENGTH_CLASSES: list[str] = [f"{lo}-{lo + 25}" for lo in range(0, 350, 25)]
"""IWRAP's own length categories (``0-25`` ... ``325-350``).  IWRAP looks a
category up by this name; an unknown name (OMRAT's ``350-400``) was written
into the ``0-25`` slot on import, replacing the real small craft (KNOWN_ISSUES 18)."""


def _omrat_interval_label(interval: dict, c: int) -> str:
    # Integer "min-max" ("0-25"), not the OMRAT label with decimals ("0.0 - 25.0").
    try:
        return f"{int(float(interval['min']))}-{int(float(interval['max']))}"
    except (KeyError, TypeError, ValueError):
        return re.sub(r"\s*-\s*", "-", str(interval.get('label', f'{c}'))).strip()


def iwrap_length_class(interval: dict, c: int = 0) -> str:
    """The IWRAP length category an OMRAT length interval is exported into.

    An interval whose name IWRAP knows keeps it.  Any other goes to the IWRAP
    class that contains its midpoint, and everything from 350 m up to the
    last class (``325-350``).
    """
    label = _omrat_interval_label(interval, c)
    if label in IWRAP_LENGTH_CLASSES:
        return label
    try:
        mid = (float(interval['min']) + float(interval['max'])) / 2.0
    except (KeyError, TypeError, ValueError):
        return IWRAP_LENGTH_CLASSES[-1]
    return IWRAP_LENGTH_CLASSES[max(0, min(int(mid // 25), len(IWRAP_LENGTH_CLASSES) - 1))]


def length_class_merges(ship_categories: dict | None) -> list[tuple[str, str]]:
    """``(OMRAT interval, IWRAP class)`` for every interval the export renames or merges."""
    if not isinstance(ship_categories, dict):
        return []
    intervals = ship_categories.get('length_intervals') or []
    out = []
    for c, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            continue
        label, target = _omrat_interval_label(interval, c), iwrap_length_class(interval, c)
        if label != target:
            out.append((label, target))
    return out


def describe_length_class_merges(merges: list[tuple[str, str]]) -> str:
    """Text for the export message ('' when every length class is IWRAP's own)."""
    if not merges:
        return ''
    lines = '\n'.join(f'  {omrat} m -> {iwrap} m' for omrat, iwrap in merges)
    return ('\n\nIWRAP has no length class for:\n' + lines
            + '\nThose ships were exported in the class shown (ships added, speed, draught, '
              'beam and height as ship-weighted means).')


def _cells_avg(mat, freq_mat, cells: list[tuple[int, int]]) -> float:
    """Ship-weighted mean of ``mat`` over ``(row, col)`` cells (unknown values left out)."""
    pairs = [(_mat_safe(freq_mat, r, c), _mat_safe(mat, r, c)) for r, c in cells]
    return round(_weighted_mean(pairs), 2)


def _emit_shiptypes_grouped(shiptypes_el: ET.Element, freq_mat, speed_mat, draft_mat,
                            height_mat, beam_mat, types: list, intervals: list):
    groups: dict[str, list[int]] = {}
    for r, type_name in enumerate(types):
        groups.setdefault(_UI_TO_IWRAP.get(type_name, 'Other ship'), []).append(r)
    # OMRAT length columns per IWRAP category, in IWRAP's order; columns that
    # land in the same category are merged like the ship types (ships add up,
    # the other values are ship-weighted means).
    columns: dict[str, list[int]] = {}
    for c, interval in enumerate(intervals):
        columns.setdefault(iwrap_length_class(interval, c), []).append(c)
    # Always emit all 14 canonical IWRAP ship types (empty categories for unmapped ones)
    # so IWRAP does not need to fall back to its own AIS re-extraction.
    for iwrap_name in _IWRAP_SHIP_TYPE_ORDER:
        rows = groups.get(iwrap_name, [])
        st_el = ET.SubElement(shiptypes_el, 'shiptype')
        for attr, val in [('causation_reduction_factor', '0'), ('freq_adjustment', '1'),
                          ('grounding_safety_margin', '-1'), ('name', iwrap_name)]:
            st_el.set(attr, val)
        cats_el = ET.SubElement(st_el, 'categories')
        for label in IWRAP_LENGTH_CLASSES:
            cells = [(r, c) for c in columns.get(label, []) for r in rows]
            freq_sum = sum(_mat_safe(freq_mat, r, c) for r, c in cells)
            if freq_sum <= 0:
                continue
            _add_category_attr(cats_el, {
                'name': label, 'freq': round(freq_sum, 2),
                'speed': _cells_avg(speed_mat, freq_mat, cells),
                'draught': _cells_avg(draft_mat, freq_mat, cells),
                'height_1': _cells_avg(height_mat, freq_mat, cells),
                'width': _cells_avg(beam_mat, freq_mat, cells),
                'depth': 0, 'p_ballast': 0,
            })


def _emit_td_shiptypes(shiptypes_el: ET.Element, dir_data: dict, types: list, intervals: list,
                       global_shiptypes: list, global_categories: list, leg_td: dict):
    freq_mat = _get_matrix(dir_data, 'Frequency (ships/year)')
    speed_mat = _get_matrix(dir_data, 'Speed (knots)')
    draft_mat = _get_matrix(dir_data, 'Draught (meters)')
    height_mat = _get_matrix(dir_data, 'Ship heights (meters)')
    beam_mat = _get_matrix(dir_data, 'Ship Beam (meters)')
    if not types:
        leg_st = global_shiptypes or [{'name': 'General cargo ship',
                                       'categories': global_categories or leg_td.get('categories', [])}]
        pre_cats = [c for st in leg_st for c in (st.get('categories', []) or [])]
        if pre_cats:
            _emit_shiptypes_prebuilt(shiptypes_el, leg_st)
        else:
            _emit_shiptypes_from_matrices(shiptypes_el, freq_mat, speed_mat, draft_mat, height_mat, beam_mat)
        return
    _emit_shiptypes_grouped(shiptypes_el, freq_mat, speed_mat, draft_mat, height_mat, beam_mat, types, intervals)


def build_traffic_distributions(parent: ET.Element, traffic_data: dict, segment_data: dict,
                                ship_categories: dict | None = None) -> dict[str, dict[str, str]]:
    """Map .omrat traffic_data into IWRAP traffic_distributions XML elements."""
    td_root = ET.SubElement(parent, 'traffic_distributions')
    td_guid_map: dict[str, dict[str, str]] = {}
    global_categories = traffic_data.get('categories', []) if isinstance(traffic_data, dict) else []
    global_shiptypes = traffic_data.get('shiptypes', []) if isinstance(traffic_data, dict) else []
    types: list = []
    intervals: list = []
    if isinstance(ship_categories, dict):
        types = ship_categories.get('types', []) or []
        intervals = ship_categories.get('length_intervals', []) or []
    for leg_key in segment_data:
        leg_td = traffic_data.get(str(leg_key), {}) if isinstance(traffic_data, dict) else {}
        # By position, like compute: block 0 = drawn direction = FTL.  The
        # names can lag behind a re-derived ``Dirs`` (vertex drag across a
        # naming sector), and a lookup by name then exported no traffic.
        blocks = [v for v in leg_td.values() if isinstance(v, dict)] if isinstance(leg_td, dict) else []
        ftl_data = blocks[0] if len(blocks) > 0 else {}
        ltf_data = blocks[1] if len(blocks) > 1 else {}
        guid_ftl, shiptypes_e = _create_td_element(td_root, leg_key, 'FTL')
        guid_ltf, shiptypes_w = _create_td_element(td_root, leg_key, 'LTF')
        td_guid_map[str(leg_key)] = {'ftl': guid_ftl, 'ltf': guid_ltf}
        for dir_data, shiptypes_el in [(ftl_data, shiptypes_e), (ltf_data, shiptypes_w)]:
            _emit_td_shiptypes(shiptypes_el, dir_data, types, intervals,
                               global_shiptypes, global_categories, leg_td)
    return td_guid_map


def write_iwrap_xml(json_path: str, output_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    root = generate_iwrap_xml(data)
    xml_str = prettify_xml(root)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)


# ── IWRAP Ship Type Codes mapping ──────────────────────────────────────
# Maps IWRAP XML ship type names to numeric Ship Code (1-14) used in the
# Ship Type Codes lookup table.
IWRAP_SHIP_TYPE_CODES: Dict[str, int] = {
    'Crude oil tanker': 1,
    'Oil products tanker': 2,
    'Chemical tanker': 3,
    'Gas tanker': 4,
    'Container ship': 5,
    'General cargo ship': 6,
    'Bulk carrier': 7,
    'Ro-Ro cargo ship': 8,
    'Passenger ship': 9,
    'Fast ferry': 10,
    'Support ship': 11,
    'Fishing ship': 12,
    'Pleasure boat': 13,
    'Other ship': 14,
}


def _load_ship_type_codes(csv_path: str) -> Dict[Tuple[int, int], dict]:
    """Load the Ship Type Codes CSV into a lookup dictionary.

    The CSV contains expected ship dimensions per (ship_code, lpp_min) pair.
    Key columns used: E(L), E(L/B), E(B/D), E(D/T), E(V).

    From these, we derive:
        Beam     = E(L) / E(L/B)
        Depth_m  = Beam / E(B/D)
        Draught  = Depth_m / E(D/T)
        Speed    = E(V)

    Args:
        csv_path: Path to Ship Type Codes.csv

    Returns:
        Dict mapping (ship_code, lpp_min) -> row dict with derived values.
        Returns empty dict if file not found or parsing fails.
    """
    lookup: Dict[Tuple[int, int], dict] = {}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    ship_code = int(row['Ship Code'])
                    lpp_min = int(row['Lpp min'])
                    ntotal = int(row.get('Ntotal', 0))

                    # Skip aggregate rows (lpp_min=-1) and empty rows
                    if lpp_min < 0 or ntotal == 0:
                        continue

                    # Parse the key ratio columns
                    e_l = float(row.get('E(L)', 0) or 0)
                    e_lb = float(row.get('E(L/B)', 0) or 0)
                    e_bd = float(row.get('E(B/D)', 0) or 0)
                    e_dt = float(row.get('E(D/T)', 0) or 0)
                    e_v = float(row.get('E(V)', 0) or 0)

                    # Derive dimensions (only if ratios are valid)
                    beam = e_l / e_lb if e_lb > 0 else 0
                    depth_m = beam / e_bd if e_bd > 0 else 0
                    draught = depth_m / e_dt if e_dt > 0 else 0

                    lookup[(ship_code, lpp_min)] = {
                        'e_l': e_l,
                        'beam': beam,
                        'depth_moulded': depth_m,
                        'draught': draught,
                        'speed': e_v,
                    }
                except (ValueError, KeyError, ZeroDivisionError):
                    continue
    except (FileNotFoundError, OSError):
        pass
    return lookup


def _find_ship_type_csv(xml_path: str) -> Optional[str]:
    """Ship Type Codes.csv next to the XML file, else the copy shipped in ``compute/``."""
    candidates = [
        os.path.join(os.path.dirname(xml_path), 'Ship Type Codes.csv'),
        os.path.join(os.path.dirname(__file__), 'Ship Type Codes.csv'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ── parse_iwrap_xml helpers ────────────────────────────────────────────────────

_IWRAP_TO_OMRAT_TYPE: dict[str, str] = {
    'Fishing ship': 'Fishing', 'Support ship': 'Towing',
    'Pleasure boat': 'Pleasure Craft', 'Fast ferry': 'High speed craft (HSC)',
    'Passenger ship': 'Passenger, all ships of this type',
    'General cargo ship': 'Cargo, all ships of this type',
    'Container ship': 'Cargo, all ships of this type',
    'Bulk carrier': 'Cargo, all ships of this type',
    'Ro-Ro cargo ship': 'Cargo, all ships of this type',
    'Crude oil tanker': 'Tanker, all ships of this type',
    'Oil products tanker': 'Tanker, all ships of this type',
    'Chemical tanker': 'Tanker, all ships of this type',
    'Gas tanker': 'Tanker, all ships of this type',
    'Other ship': 'Other Type, all ships of this type',
}

_OMRAT_SHIP_TYPES_FULL = [
    'Fishing', 'Towing', 'Dredging or underwater ops', 'Diving ops', 'Military ops',
    'Sailing', 'Pleasure Craft', 'High speed craft (HSC)', 'Pilot Vessel',
    'Search and Rescue vessel', 'Tug', 'Port Tender', 'Anti-pollution equipment',
    'Law Enforcement', 'Spare', 'Medical Transport',
    'Noncombatant ship according to RR Resolution No. 18',
    'Passenger, all ships of this type', 'Cargo, all ships of this type',
    'Tanker, all ships of this type', 'Other Type, all ships of this type',
]


def _init_parse_result(root) -> dict:
    result = {
        'project_name': root.get('name', 'Imported Project'),
        'traffic_data': {}, 'segment_data': {}, 'drift': {},
        'depths': [], 'objects': [], 'ship_categories': None,
        'pc': {'p_pc': 0.00016, 'd_pc': 1},
    }
    for attr in ['fv', 'major', 'minor', 'seasons', 'current_season']:
        if root.get(attr) is not None:
            result[attr] = root.get(attr)
    return result


def _fill_repair_combi(repair_el, repair_dict: dict, debug: bool) -> None:
    combi = repair_el.get('combi', '')
    dist_type = repair_el.get('type', '')
    p0, p1, p2 = (float(repair_el.get(f'param_{i}', 0)) for i in range(3))
    if ('Mean' in combi and 'Std' in combi and 'Lower' not in combi) or dist_type == 'Normal':
        norm_std = p1 if p1 > 0 else 1e-6
        repair_dict.update({
            'use_lognormal': False, 'dist_type': 'normal', 'norm_mean': p0, 'norm_std': norm_std,
            'func': f"__import__('scipy.stats', fromlist=['norm']).norm(loc={p0}, scale={norm_std}).cdf(x)",
        })
        if debug:
            print(f"    Normal repair: mean={p0}, std={norm_std}")
    elif ('Delta' in combi and 'Beta' in combi) or dist_type == 'Weibull':
        repair_dict.update({
            'use_lognormal': False, 'dist_type': 'weibull',
            'wb_shape': p1, 'wb_loc': p2, 'wb_scale': p0,
            'func': (
                f"__import__('scipy.stats', fromlist=['weibull_min'])"
                f".weibull_min(c={p1}, loc={p2}, scale={p0}).cdf(x)"
            ),
        })
        if debug:
            print(f"    Weibull repair: shape={p1}, loc={p2}, scale={p0}")
    elif 'Mean' in combi and 'Std' in combi and 'Lower' in combi:
        adj_mean = p0 - p2
        if adj_mean > 0 and p1 > 0:
            sigma2 = math.log(1 + (p1 / adj_mean) ** 2)
            sigma = math.sqrt(sigma2)
            repair_dict.update({'std': sigma, 'loc': p2, 'scale': math.exp(math.log(adj_mean) - sigma2 / 2)})
        elif adj_mean > 0:
            repair_dict.update({'std': 0.01, 'loc': p2, 'scale': adj_mean})
    else:
        for i, key in enumerate(['loc', 'scale', 'std']):
            if f'param_{i}' in repair_el.attrib:
                repair_dict[key] = (p0, p1, p2)[i]


_BLACKOUT_CLASSES_KEY = '_iwrap_blackout_classes'
"""Transient ``(roro_passenger, other)`` blackout rates; turned into
``blackout_by_ship_type`` by :func:`_apply_blackout_classes` once the ship types are known."""


def _apply_blackout_classes(result: dict) -> None:
    """Per-type blackout rates from IWRAP's two classes (mirror of :func:`blackout_classes`)."""
    from compute.basic_equations import default_blackout_by_ship_type

    drift = result.get('drift') or {}
    classes = drift.pop(_BLACKOUT_CLASSES_KEY, None)
    types = (result.get('ship_categories') or {}).get('types')
    if not classes or not types:
        return
    roro, other = classes
    by_type = {}
    for idx, default in default_blackout_by_ship_type(list(types)).items():
        value = roro if default < 1.0 else other
        by_type[idx] = float(value) if value is not None else default
    drift['blackout_by_ship_type'] = by_type


def _parse_drifting_el(drifting_el, debug: bool) -> dict:
    drift = {
        'drift_p': 1, 'anchor_p': 0.95, 'anchor_d': 7, 'speed': 1.0,
        'rose': {str(a): 0.125 for a in [0, 45, 90, 135, 180, 225, 270, 315]},
        'repair': {'func': '', 'std': 0.95, 'loc': 0.2, 'scale': 0.85, 'use_lognormal': True},
    }
    if drifting_el is None:
        return drift
    if (v := drifting_el.get('anchor_probability')) is not None:
        drift['anchor_p'] = float(v)
    if (v := drifting_el.get('blackout_other')) is not None:
        drift['drift_p'] = float(v)
    elif (v := drifting_el.get('blackout_roro_passenger')) is not None:
        drift['drift_p'] = float(v)
    elif (v := drifting_el.get('anchor_probability')) is not None:
        drift['drift_p'] = float(v)
    roro = drifting_el.get('blackout_roro_passenger')
    other = drifting_el.get('blackout_other')
    if roro is not None or other is not None:
        drift[_BLACKOUT_CLASSES_KEY] = (
            float(roro) if roro is not None else None, float(other) if other is not None else None)
    if (v := drifting_el.get('min_anchor_dist_from_ground')) is not None:
        drift['min_anchor_dist_from_ground'] = _as_float(v)
    if (v := drifting_el.get('max_anchor_depth')) is not None:
        drift['anchor_d'] = int(round(float(v)))
    if (v := drifting_el.get('drift_speed')) is not None:
        drift['speed'] = float(v)
    repair_el = drifting_el.find('repair_time')
    if repair_el is not None:
        for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type', 'weight']:
            if (val := repair_el.get(attr)) is not None:
                try:
                    drift['repair'][attr] = float(val)
                except ValueError:
                    drift['repair'][attr] = val
        _fill_repair_combi(repair_el, drift['repair'], debug)
    dd_el = drifting_el.find('drift_directions')
    if dd_el is not None:
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            if (val := dd_el.get(f'angle_{angle}')) is not None:
                drift['rose'][str((angle + 180) % 360)] = float(val)
    return drift


def _parse_category_el(cat_el, omrat_type: str, ship_code: int, length_interval: str,
                       ship_type_codes: dict, debug: bool) -> dict:
    draught_val = float(cat_el.get('draught', 0))
    height_val = float(cat_el.get('height_1', 0))
    beam_val = float(cat_el.get('width', 0))
    speed_val = float(cat_el.get('speed', 0))
    if (draught_val == 0 or beam_val == 0 or speed_val == 0) and '-' in length_interval:
        try:
            lpp_min = int(length_interval.split('-')[0])
        except (ValueError, IndexError):
            lpp_min = -1
        stc = ship_type_codes.get((ship_code, lpp_min))
        if stc:
            if draught_val == 0:
                draught_val = round(stc['draught'], 2)
            if beam_val == 0:
                beam_val = round(stc['beam'], 2)
            if speed_val == 0:
                speed_val = round(stc['speed'], 2)
            if debug:
                print(f"    STC lookup ({omrat_type}, {length_interval}): "
                      f"draught={draught_val}m, beam={beam_val}m, speed={speed_val}kts")
        elif draught_val == 0:
            try:
                parts = length_interval.split('-')
                draught_val = round((float(parts[0]) + float(parts[1])) / 2 / 12.0, 2)
            except (ValueError, IndexError):
                pass
    return {
        'shiptype': omrat_type, 'length_interval': length_interval,
        'freq': float(cat_el.get('freq', 0)), 'speed': speed_val,
        'draught': draught_val, 'height': height_val, 'beam': beam_val,
    }


_CATEGORY_MEAN_KEYS = ('speed', 'draught', 'height', 'beam')


def _merge_categories(a: dict, b: dict) -> dict:
    """One OMRAT cell from two IWRAP categories that import to it.

    Several IWRAP ship types share an OMRAT row (Container ship, Bulk
    carrier, Ro-Ro and General cargo -> Cargo; the four tankers -> Tanker).
    Ships add up and the other values become ship-weighted means.  The
    category used to be overwritten, so only the last IWRAP type of the
    file kept its ships.
    """
    merged = dict(a)
    merged['freq'] = a['freq'] + b['freq']
    for key in _CATEGORY_MEAN_KEYS:
        mean = _weighted_mean([(a['freq'], a[key]), (b['freq'], b[key])])
        # No ships on either side (IWRAP files list empty categories): keep
        # whichever value is known.
        merged[key] = mean if mean > 0 else (a[key] if a[key] > 0 else b[key])
    return merged


def _parse_tds_el(tds_el, ship_type_codes: dict, debug: bool) -> tuple[dict, set, set]:
    td_map: dict = {}
    all_ship_types: set = set()
    all_length_intervals: set = set()
    if tds_el is None:
        return td_map, all_ship_types, all_length_intervals
    if debug:
        print(f"\nFound {len(list(tds_el.findall('traffic_distribution')))} traffic distributions")
    for td_el in tds_el.findall('traffic_distribution'):
        guid, td_name = td_el.get('guid', ''), td_el.get('name', '')
        cats_by_key: dict = {}
        shiptypes_el = td_el.find('shiptypes')
        if shiptypes_el is not None:
            for st_el in shiptypes_el.findall('shiptype'):
                st_name = st_el.get('name', '')
                omrat_type = _IWRAP_TO_OMRAT_TYPE.get(st_name, 'Other Type, all ships of this type')
                all_ship_types.add(omrat_type)
                ship_code = IWRAP_SHIP_TYPE_CODES.get(st_name, 0)
                cats_el = st_el.find('categories')
                if cats_el is not None:
                    for cat_el in cats_el.findall('category'):
                        li = cat_el.get('name', '0-25')
                        all_length_intervals.add(li)
                        cat_data = _parse_category_el(cat_el, omrat_type, ship_code, li, ship_type_codes, debug)
                        key = (omrat_type, li)
                        cats_by_key[key] = (_merge_categories(cats_by_key[key], cat_data)
                                            if key in cats_by_key else cat_data)
                        if debug and cat_data['freq'] > 0:
                            print(f"  TD {td_name}: {omrat_type} ({li}) = {cat_data['freq']} ships/yr")
        td_map[guid] = {'name': td_name, 'categories': cats_by_key}
    return td_map, all_ship_types, all_length_intervals


def _parse_lld_rows(wp_el) -> list[tuple[str, str, float]]:
    """``(from_leg_uid, to_leg_uid, fraction)`` of one waypoint's routing rows."""
    rows = []
    for lld in wp_el.iter('leg_leg_distribution'):
        try:
            fraction = float(lld.get('fraction', 0))
        except (TypeError, ValueError):
            continue
        rows.append((lld.get('from_leg_uid', ''), lld.get('to_leg_uid', ''), fraction))
    return rows


def _parse_waypoints_el(wps_el) -> dict:
    if wps_el is None:
        return {}
    return {
        wp_el.get('guid', ''): {
            'name': wp_el.get('name', ''),
            'lat': float(wp_el.get('latitude', 0)),
            'lon': float(wp_el.get('longitude', 0)),
            'llds': _parse_lld_rows(wp_el),
        }
        for wp_el in wps_el.findall('waypoint')
    }


def _leg_guid_map(legs_el) -> dict[str, str]:
    """IWRAP leg guid -> OMRAT segment id (same numbering as ``_parse_legs_el``)."""
    if legs_el is None:
        return {}
    return {leg_el.get('guid', ''): str(idx)
            for idx, leg_el in enumerate(legs_el.findall('leg'), start=1) if leg_el.get('guid')}


def _junctions_from_llds(waypoint_map: dict, guid_map: dict[str, str], segment_data: dict) -> dict:
    """Junction matrices from IWRAP's routing rows (mirror of ``build_leg_leg_distributions``).

    Rows to the same leg (the self pairs older OMRAT exports wrote) are
    ignored.  A junction with at least one row is stored as ``user`` so a
    refresh keeps it; in-legs without a row get the geometric default row;
    rows are normalised.  Junctions without rows are left out: loading
    gives them the geometric default, as before.
    """
    from geometries.junctions import (
        build_junctions, compute_geometric_transition_matrix, junction_id_for_point, serialize_junctions,
    )

    registry = build_junctions(segment_data)
    found: dict[str, dict[str, dict[str, float]]] = {}
    for wp in waypoint_map.values():
        rows = wp.get('llds') or []
        if not rows:
            continue
        jid = junction_id_for_point((float(wp.get('lon', 0)), float(wp.get('lat', 0))))
        junction = registry.get(jid)
        if junction is None:
            continue
        for from_guid, to_guid, fraction in rows:
            in_leg, out_leg = guid_map.get(from_guid), guid_map.get(to_guid)
            if (in_leg is None or out_leg is None or in_leg == out_leg or fraction <= 0
                    or in_leg not in junction.legs or out_leg not in junction.legs):
                continue
            row = found.setdefault(jid, {}).setdefault(in_leg, {})
            row[out_leg] = row.get(out_leg, 0.0) + fraction
    out = {}
    for jid, rows in found.items():
        junction = registry[jid]
        geometric = compute_geometric_transition_matrix(junction, segment_data)
        transitions = {}
        for in_leg in junction.legs:
            row = rows.get(in_leg) or geometric.get(in_leg) or {}
            total = sum(row.values())
            if total > 0:
                transitions[in_leg] = {k: v / total for k, v in row.items()}
        junction.transitions = transitions
        junction.source = 'user'
        out[jid] = junction
    return serialize_junctions(out)


def _parse_mal_el(mals_el) -> dict:
    mal_map: dict = {}
    if mals_el is None:
        return mal_map
    for mal_el in mals_el.findall('manoeuvring_aspects_leg'):
        guid = mal_el.get('guid', '')
        dp: dict = {'means': [], 'stds': [], 'weights': [], 'u_min': None, 'u_max': None, 'u_p': 0,
                    'check_time': _as_float(mal_el.get('grounding_check_time', 0))}
        md_el = mal_el.find('mixed_dist')
        if md_el is not None:
            norm_items, unif_items = [], []
            for item_el in md_el.findall('mixed_dist_item'):
                entry = {k: float(item_el.get(k, 0)) for k in ('param_0', 'param_1', 'weight')}
                (norm_items if item_el.get('type') == 'Normal' else
                 unif_items if item_el.get('type') == 'Uniform' else []).append(entry)
            for item in norm_items:
                dp['means'].append(item['param_0'])
                dp['stds'].append(item['param_1'])
                dp['weights'].append(item['weight'])
            if unif_items:
                dp['u_min'] = unif_items[0]['param_0']
                dp['u_max'] = unif_items[0]['param_1']
                dp['u_p'] = unif_items[0]['weight']
        mal_map[guid] = dp
    return mal_map


def _fill_segment_from_mal(seg: dict, ftl_guid: str, ltf_guid: str, mal_map: dict) -> None:
    # Mirror of ``_fill_mal_mixed_dist``: IWRAP's positive = starboard of
    # the sailing direction, OMRAT's negative = starboard of the drawn
    # direction.  The first-to-last flow sails the drawn way (sign changes),
    # the last-to-first flow against it (sign kept).  Before 2026-11-09 the
    # importer flipped direction 2 instead, so a native IWRAP file came in
    # with both lanes mirrored (KNOWN_ISSUES 16).
    if ftl_guid in mal_map:
        mal = mal_map[ftl_guid]
        for i, (m, s, w) in enumerate(zip(mal['means'], mal['stds'], mal['weights']), 1):
            seg[f'mean1_{i}'] = -m
            seg[f'std1_{i}'] = s
            seg[f'weight1_{i}'] = w * 100.0
        if mal['u_min'] is not None:
            seg['u_min1'] = -mal['u_max']
            seg['u_max1'] = -mal['u_min']
            seg['u_p1'] = int(round(mal['u_p'] * 100.0))
        if mal.get('check_time', 0) > 0:
            seg['ai1'] = mal['check_time']
    if ltf_guid in mal_map:
        mal = mal_map[ltf_guid]
        for i, (m, s, w) in enumerate(zip(mal['means'], mal['stds'], mal['weights']), 1):
            seg[f'mean2_{i}'] = m
            seg[f'std2_{i}'] = s
            seg[f'weight2_{i}'] = w * 100.0
        if mal['u_min'] is not None:
            seg['u_min2'] = mal['u_min']
            seg['u_max2'] = mal['u_max']
            seg['u_p2'] = int(round(mal['u_p'] * 100.0))
        if mal.get('check_time', 0) > 0:
            seg['ai2'] = mal['check_time']


def _haversine_m(start_point: str, end_point: str) -> float:
    from math import radians, cos, sin, asin, sqrt
    try:
        sp, ep = start_point.split(), end_point.split()
        if len(sp) == 2 and len(ep) == 2:
            lon1, lat1, lon2, lat2 = map(radians, [float(sp[0]), float(sp[1]), float(ep[0]), float(ep[1])])
            a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
            return 2 * asin(sqrt(a)) * 6371000
    except Exception:  # nosec B110 B112
        pass
    return 0.0


_CAT2_OFF_KEY = '_iwrap_cat2_off'
"""Transient per-segment marker (list of direction numbers), stripped by parse_iwrap_xml."""


def _parse_legs_el(legs_el, waypoint_map: dict, mal_map: dict, debug: bool) -> dict:
    segment_data: dict = {}
    if legs_el is None:
        return segment_data
    for idx, leg_el in enumerate(legs_el.findall('leg'), start=1):
        leg_name = leg_el.get('name', f'Leg_{idx}')
        start_wp = waypoint_map.get(leg_el.get('first_waypoint_guid', ''), {})
        end_wp = waypoint_map.get(leg_el.get('last_waypoint_guid', ''), {})
        sp = f"{start_wp.get('lon', 0)} {start_wp.get('lat', 0)}"
        ep = f"{end_wp.get('lon', 0)} {end_wp.get('lat', 0)}"
        seg: dict = {
            'Leg_name': leg_name, 'Width': int(round(float(leg_el.get('max_width', 0)))),
            'Tangent_Pos': 0.5,
            'Start_Point': sp, 'End_Point': ep,
            'Dirs': ['East going', 'West going'], 'line_length': _haversine_m(sp, ep),
            'Route_Id': 0, 'Segment_Id': str(idx),
            **{f'mean1_{i}': 0.0 for i in range(1, 4)},
            **{f'std1_{i}': 0.0 for i in range(1, 4)},
            **{f'weight1_{i}': 0.0 for i in range(1, 4)},
            **{f'mean2_{i}': 0.0 for i in range(1, 4)},
            **{f'std2_{i}': 0.0 for i in range(1, 4)},
            **{f'weight2_{i}': 0.0 for i in range(1, 4)},
            'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0,
            'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0,
            'ai1': 0.0, 'ai2': 0.0, 'dist1': [], 'dist2': [],
        }
        # Zero extension past a waypoint = no Cat II for the flow that would
        # overshoot it (mirror of build_legs).  Remembered transiently so the
        # global mean-time-between-checks does not re-enable it.
        cat2_off = []
        if (v := leg_el.get('max_extension_last')) is not None and _as_float(v) <= 0:
            cat2_off.append(1)
        if (v := leg_el.get('max_extension_first')) is not None and _as_float(v) <= 0:
            cat2_off.append(2)
        if cat2_off:
            seg[_CAT2_OFF_KEY] = cat2_off
        _fill_segment_from_mal(seg, leg_el.get('man_aspects_first_to_last_guid', ''),
                               leg_el.get('man_aspects_last_to_first_guid', ''), mal_map)
        if debug:
            print(f"  Leg {idx} ({leg_name}): length={seg['line_length']:.1f}m")
        segment_data[str(idx)] = seg
    return segment_data


def _build_ship_categories_from_intervals(all_length_intervals: set) -> tuple[list, list]:
    parsed: list = []
    for s in all_length_intervals:
        parts = s.split('-')
        try:
            parsed.append({'min': float(parts[0]), 'max': float(parts[1]), 'label': s})
        except (IndexError, ValueError):
            parsed.append({'min': 0, 'max': 25, 'label': s})
    parsed.sort(key=lambda x: x['min'])
    return _OMRAT_SHIP_TYPES_FULL, parsed


def _fill_direction_from_td(direction_data: dict, td_info: dict,
                            omrat_types: list, intervals_list: list, debug: bool) -> None:
    for (ship_type, li), cat_data in td_info['categories'].items():
        try:
            ti = omrat_types.index(ship_type)
            ii = intervals_list.index(li)
            direction_data['Frequency (ships/year)'][ti][ii] = cat_data['freq']
            direction_data['Speed (knots)'][ti][ii] = cat_data['speed']
            direction_data['Draught (meters)'][ti][ii] = cat_data['draught']
            direction_data['Ship heights (meters)'][ti][ii] = cat_data['height']
            direction_data['Ship Beam (meters)'][ti][ii] = cat_data['beam']
        except (ValueError, IndexError) as e:
            if debug:
                print(f"      ERROR filling data: {e} ({ship_type}, {li})")


def _build_traffic_matrices_el(legs_el, td_map: dict, omrat_types: list,
                               intervals_list: list, debug: bool) -> dict:
    traffic_data: dict = {}
    if legs_el is None:
        return traffic_data
    nt, ni = len(omrat_types), len(intervals_list)
    td_keys = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)',
               'Ship heights (meters)', 'Ship Beam (meters)']
    for idx, leg_el in enumerate(legs_el.findall('leg'), start=1):
        ftl_guid = leg_el.get('traffic_distribution_first_to_last_guid', '')
        ltf_guid = leg_el.get('traffic_distribution_last_to_first_guid', '')
        east_data = {k: [[0.0] * ni for _ in range(nt)] for k in td_keys}
        east_data['Scaling (%)'] = [[100.0] * ni for _ in range(nt)]
        west_data = {k: [[0.0] * ni for _ in range(nt)] for k in td_keys}
        west_data['Scaling (%)'] = [[100.0] * ni for _ in range(nt)]
        if ftl_guid in td_map:
            _fill_direction_from_td(east_data, td_map[ftl_guid], omrat_types, intervals_list, debug)
        if ltf_guid in td_map:
            _fill_direction_from_td(west_data, td_map[ltf_guid], omrat_types, intervals_list, debug)
        traffic_data[str(idx)] = {'East going': east_data, 'West going': west_data}
        if debug:
            te = sum(sum(r) for r in east_data['Frequency (ships/year)'])
            tw = sum(sum(r) for r in west_data['Frequency (ships/year)'])
            print(f"    Leg {idx} traffic: East={int(te)}, West={int(tw)} ships/year")
    return traffic_data


def _classify_area_el(area_el) -> tuple[str, str, str, bool, str]:
    area_type = area_el.get('type', '0')
    depth_val = area_el.get('depth', '0')
    name = area_el.get('name', '')
    coords = []
    poly_el = area_el.find('polygon')
    if poly_el is not None:
        for item_el in poly_el.findall('item'):
            coords.append((float(item_el.get('lon', 0)), float(item_el.get('lat', 0))))
    if coords:
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        wkt = f"POLYGON(({', '.join(f'{lon} {lat}' for lon, lat in coords)}))"
    else:
        wkt = ""
    try:
        depth_num = float(depth_val)
        is_obj = depth_num < 0 or str(area_type).strip() != '0'
        height_val = str(abs(depth_num)) if depth_num < 0 else ('0' if is_obj else depth_val)
    except ValueError:
        is_obj = depth_val.strip().startswith('-') or str(area_type).strip() != '0'
        stripped = depth_val.strip()
        height_val = stripped.lstrip('-') if stripped.startswith('-') else ('0' if is_obj else depth_val)
    return name, depth_val, wkt, is_obj, height_val


def _parse_areas_el(areas_el) -> tuple[list, list]:
    depths, objects = [], []
    if areas_el is None:
        return depths, objects
    used: dict[bool, set[str]] = {True: set(), False: set()}

    def _unique(area_id: str, is_obj: bool) -> str:
        # Every area keeps its own id.  Until 2026-11-09 the part suffix of
        # ``depth_<id>_<part>`` was cut off, so the 12,010 parts of the
        # Kattegat export came back with 17 ids and the powered model took a
        # whole depth class for one obstacle (KNOWN_ISSUES 21).
        out, n = area_id, 1
        while out in used[is_obj]:
            n += 1
            out = f"{area_id}_{n}"
        used[is_obj].add(out)
        return out

    for area_el in areas_el.findall('area_polygon'):
        name, depth_val, wkt, is_obj, height_val = _classify_area_el(area_el)
        if is_obj:
            objects.append([_unique(name.replace('object_', '', 1), True), height_val, wkt])
        else:
            depths.append([_unique(name.replace('depth_', '', 1), False), depth_val, wkt])
    return depths, objects


def _parse_global_settings_el(gs_el, result: dict, debug: bool) -> None:
    if gs_el is None:
        return
    cf_el = gs_el.find('causation_factors')
    if cf_el is not None:
        pc = result['pc']
        for omrat_key, iwrap_keys in _CF_IMPORT_MAP.items():
            for iwrap_key in iwrap_keys:
                if (val := cf_el.get(iwrap_key)) is None:
                    continue
                try:
                    pc[omrat_key] = float(val)
                except (ValueError, TypeError):
                    pass
        # ``p_pc`` is the dialog's "Powered causation factor" and a legacy
        # alias the powered grounding model falls back to.  Keep it in step
        # so the dialog does not show a stale default next to an imported
        # grounding factor.
        if 'grounding' in pc:
            pc['p_pc'] = pc['grounding']
    misc_el = gs_el.find('misc')
    if misc_el is not None:
        for attr, key in (('passengership_reductionfactor', passenger_reduction.PASSENGER_KEY),
                          ('fastferry_reduction_factor', passenger_reduction.FAST_FERRY_KEY)):
            try:
                value = float(misc_el.get(attr, ''))
            except (TypeError, ValueError):
                continue
            result['pc'][key] = value if value >= 1.0 else 1.0
    if misc_el is not None and (mtbc_val := misc_el.get('meantime_between_checks')) is not None:
        try:
            mtbc = float(mtbc_val)
            result['pc']['mean_time_between_checks'] = mtbc
            for seg in result['segment_data'].values():
                off = seg.get(_CAT2_OFF_KEY, ())
                if seg.get('ai1', 0.0) == 0.0 and 1 not in off:
                    seg['ai1'] = mtbc
                if seg.get('ai2', 0.0) == 0.0 and 2 not in off:
                    seg['ai2'] = mtbc
            if debug:
                print(f"  Mean time between checks: {mtbc}s -> set ai where the leg has no check time")
        except (ValueError, TypeError):
            pass


def _compute_bend_angles(segment_data: dict, debug: bool) -> None:
    seg_ids = sorted(segment_data.keys(), key=lambda x: int(x))
    for seg_id in seg_ids:
        seg = segment_data[seg_id]
        seg_end = seg.get('End_Point', '')
        for other_id in seg_ids:
            if other_id == seg_id:
                continue
            other = segment_data[other_id]
            if seg_end and other.get('Start_Point', '') == seg_end:
                try:
                    sp1 = tuple(float(v) for v in seg['Start_Point'].split()[:2])
                    shared = tuple(float(v) for v in seg_end.split()[:2])
                    ep2 = tuple(float(v) for v in other['End_Point'].split()[:2])
                    # In a projection centred on the shared waypoint (raw degrees until
                    # 2026-11-12, KNOWN_ISSUES 36); the value is informational only.
                    proj = ModelProjector(*shared)
                    b1 = projected_bearing(sp1, shared, proj)
                    b2 = projected_bearing(shared, ep2, proj)
                    bend = abs(b2 - b1)
                    if bend > 180:
                        bend = 360 - bend
                    seg['bend_angle'] = other['bend_angle'] = bend
                    if debug:
                        print(f"  Bend angle ({seg['Leg_name']}<->{other['Leg_name']}): {bend:.1f} deg")
                except (ValueError, IndexError):
                    pass


def parse_iwrap_xml(xml_path: str, debug: bool = False) -> dict:
    """Parse IWRAP XML file and convert to .omrat JSON format."""
    tree = DefusedET.parse(xml_path)
    root = tree.getroot()
    stc_csv = _find_ship_type_csv(xml_path)
    ship_type_codes = _load_ship_type_codes(stc_csv) if stc_csv else {}
    if debug:
        if ship_type_codes:
            print(f"  Loaded Ship Type Codes from {stc_csv} ({len(ship_type_codes)} entries)")
        else:
            print("  Ship Type Codes.csv not found; will fall back to L/12 draught estimate")

    result = _init_parse_result(root)
    result['drift'] = _parse_drifting_el(root.find('drifting'), debug)
    td_map, all_ship_types, all_length_intervals = _parse_tds_el(
        root.find('traffic_distributions'), ship_type_codes, debug
    )
    waypoint_map = _parse_waypoints_el(root.find('waypoints'))
    mal_map = _parse_mal_el(root.find('manoeuvring_aspects_legs'))
    result['segment_data'] = _parse_legs_el(root.find('legs'), waypoint_map, mal_map, debug)
    junctions = _junctions_from_llds(waypoint_map, _leg_guid_map(root.find('legs')), result['segment_data'])
    if junctions:
        result['junctions'] = junctions

    if all_ship_types and all_length_intervals:
        omrat_types, parsed_intervals = _build_ship_categories_from_intervals(all_length_intervals)
        result['ship_categories'] = {
            'types': omrat_types, 'length_intervals': parsed_intervals, 'selection_mode': 'ais',
        }
        intervals_list = [pi['label'] for pi in parsed_intervals]
        result['traffic_data'] = _build_traffic_matrices_el(
            root.find('legs'), td_map, omrat_types, intervals_list, debug
        )

    _apply_blackout_classes(result)
    result['depths'], result['objects'] = _parse_areas_el(root.find('areas'))
    _parse_global_settings_el(root.find('global_settings'), result, debug)
    for seg in result['segment_data'].values():
        seg.pop(_CAT2_OFF_KEY, None)
    _compute_bend_angles(result['segment_data'], debug)
    if debug:
        print(f"\n{'=' * 70}")
        print(f"Segments: {len(result['segment_data'])}, Depths: {len(result['depths'])}, "
              f"Objects: {len(result['objects'])}, Traffic legs: {len(result['traffic_data'])}")
        print(f"{'=' * 70}\n")
    return result


def read_iwrap_xml(xml_path: str, output_path: str):
    """Read IWRAP XML file and save as .omrat JSON file.

    Args:
        xml_path: Path to input IWRAP XML file
        output_path: Path to output .omrat JSON file
    """
    data = parse_iwrap_xml(xml_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'export' and len(sys.argv) >= 4:
            # Export: python iwrap_convertion.py export input.omrat output.xml
            write_iwrap_xml(sys.argv[2], sys.argv[3])
            print(f"✓ Exported {sys.argv[2]} to {sys.argv[3]}")

        elif command == 'import' and len(sys.argv) >= 4:
            # Import: python iwrap_convertion.py import input.xml output.omrat
            read_iwrap_xml(sys.argv[2], sys.argv[3])
            print(f"✓ Imported {sys.argv[2]} to {sys.argv[3]}")

        else:
            print("Usage:")
            print("  Export: python iwrap_convertion.py export <input.omrat> <output.xml>")
            print("  Import: python iwrap_convertion.py import <input.xml> <output.omrat>")
    else:
        # Default behavior for backward compatibility
        write_iwrap_xml('tests/example_data/proj.omrat', 'tests/example_data/generated.xml')
        print("✓ Default export completed: proj.omrat -> generated.xml")
