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
import xml.etree.ElementTree as ET  # nosec B405 - parsing is delegated to defusedxml; this import is only used to build trees (ET.Element / ET.SubElement / ET.tostring)
from typing import Dict, List, Optional, Tuple

# defusedxml protects against XML attacks (XXE, billion laughs, etc.) when
# we read XML files from disk.  We still use the stdlib ``ElementTree`` for
# *building* trees (ET.Element / ET.SubElement / ET.tostring) since those
# don't parse external input.
from defusedxml import ElementTree as DefusedET
from defusedxml import minidom as defused_minidom

# Geometry fixing utilities
try:
    from shapely.geometry import Polygon as ShpPolygon, MultiPolygon as ShpMultiPolygon, GeometryCollection as ShpGeometryCollection
    HAVE_SHAPELY = True
except Exception:
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
    except Exception:
        return []

def parse_point_str(point_str):
    try:
        lon_str, lat_str = point_str.strip().split()
        return float(lat_str), float(lon_str)
    except Exception:
        return None


def new_guid() -> str:
    return str(uuid.uuid4()).upper()

def prettify_xml(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = defused_minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def build_drifting(parent: ET.Element, drift: dict):
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
    if (v := drift.get('drift_blackout_other')) is not None:
        drifting.set('blackout_other', str(v))
    elif (v := drift.get('drift_p')) is not None:
        # Best effort mapping if provided
        drifting.set('blackout_other', str(v))
    # Required children per XSD: repair_time and drift_directions
    repair = drift.get('repair', {}) if isinstance(drift.get('repair'), dict) else {}
    rep = ET.SubElement(drifting, 'repair_time')
    # Attributes on repair_time are optional
    for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type', 'weight']:
        if attr in repair and repair.get(attr) is not None:
            rep.set(attr, str(repair.get(attr)))
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
            name = str(obj.get('id',''))
            height = obj.get('height', obj.get('heights',''))
            polygon = obj.get('polygon','')
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
            leg.set('max_width_powered', str(seg.get('Width')))
        # Use fixed extensions as requested
        leg.set('max_extension_first', '50000')
        leg.set('max_extension_last', '50000')
        if 'ai1' in seg and seg.get('ai1') is not None:
            leg.set('max_bearing_angle', str(seg['ai1']))
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

def build_leg_leg_distributions(parent_waypoints: ET.Element, segment_data: dict, waypoint_lookup: dict, leg_map: dict):
    # Build a mapping from waypoint coordinates to waypoint elements (attributes present)
    waypoint_nodes = {}
    for w_el in list(parent_waypoints.findall('waypoint')):
        lat = w_el.get('latitude')
        lon = w_el.get('longitude')
        if lat is None or lon is None:
            continue
        key = f"{lat},{lon}"
        waypoint_nodes[key] = w_el

    # Attach one automatic self-distribution per starting waypoint
    counter = 1
    for seg_id, info in leg_map.items():
        start_key = info.get('start_key')
        if not start_key:
            continue
        sk_norm = start_key
        w_el = waypoint_nodes.get(sk_norm)
        if w_el is None:
            continue
        llds = w_el.find('leg_leg_distributions')
        if llds is None:
            llds = ET.SubElement(w_el, 'leg_leg_distributions')
        dist = ET.SubElement(llds, 'leg_leg_distribution')
        dist.set('guid', new_guid())
        dist.set('from_leg_uid', info.get('guid', ''))
        # Self-pair for now
        dist.set('to_leg_uid', info.get('guid', ''))
        dist.set('method', 'AUTOMATIC')
        dist.set('name', f"WTD_{counter}")
        dist.set('season', '-1')
        dist.set('fraction', '1')
        counter += 1

def _as_float(v) -> float:
    try:
        f = float(v)
        if f != f or f in (float('inf'), float('-inf')):
            return 0.0
        return f
    except Exception:
        return 0.0


_MAL_DEFAULT_ATTRS = [
    ('allision_causation_rf', '1'), ('allision_drifting_rf', '1'), ('allision_no_turn_rf', '1'),
    ('aton_reduction_factor', '1'), ('grounding_causation_rf', '1'), ('grounding_check_time', '0'),
    ('grounding_drifting_rf', '1'), ('grounding_no_turn_rf', '1'), ('headon_causation_rf', '1'),
    ('limit_width', 'false'), ('overtaking_causation_rf', '1'),
]


def _fill_mal_mixed_dist(md_el: ET.Element, seg: dict, dir_num: int) -> None:
    for i in range(1, 4):
        w = _as_float(seg.get(f'weight{dir_num}_{i}', 0))
        if w <= 0:
            continue
        item = ET.SubElement(md_el, 'mixed_dist_item')
        item.set('combi', '/Mean/Std. Dev.')
        mean = _as_float(seg.get(f'mean{dir_num}_{i}', 0))
        std = _as_float(seg.get(f'std{dir_num}_{i}', 0))
        if dir_num == 2:
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
        if dir_num == 2:
            u_min, u_max = -u_min, -u_max
            if u_min > u_max:
                u_min, u_max = u_max, u_min
        u_item.set('param_0', str(u_min))
        u_item.set('param_1', str(u_max))
        u_item.set('type', 'Uniform')
        u_item.set('weight', str(u_p))


def _build_mal_for_direction(
    mals: ET.Element, seg: dict, seg_id, dir_suffix: str, dir_num: int,
) -> str:
    mal = ET.SubElement(mals, 'manoeuvring_aspects_leg')
    guid = '{' + new_guid() + '}'
    mal.set('guid', guid)
    mal.set('name', str(seg.get('Leg_name', f'LEG_{seg_id}_{dir_suffix}')))
    for attr, default in _MAL_DEFAULT_ATTRS:
        mal.set(attr, default)
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
    'misc': 'B|calc_allision|1@B|calc_area|1@B|calc_collisions|1@B|calc_crossing|1@B|calc_drifting_allision|1@B|calc_drifting_grounding|1@B|calc_grounding|1@B|calc_headon_overtaking|1@B|calc_powered_allision|1@B|calc_powered_grounding|1',
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


def _build_objects_as_depths(objects_list: list) -> list:
    result = []
    for obj in objects_list:
        if isinstance(obj, dict):
            oid = obj.get('id', '')
            poly = obj.get('polygon', '')
        else:
            try:
                oid, _height, poly = obj
            except Exception:
                continue
        result.append([str(oid), str(-1), str(poly)])
    return result


def _add_global_settings(root: ET.Element) -> None:
    gs = ET.SubElement(root, 'global_settings')
    cf = ET.SubElement(gs, 'causation_factors')
    cf.set('p_allision_causation', '0.000155')
    cf.set('p_allision_drifting_causation', '1')
    cf.set('p_allision_no_turn_causation', '0.000155')
    cf.set('p_bend_causation', '0.00013')
    cf.set('p_crossing_causation', '0.00013')
    cf.set('p_grounding_causation', '0.000155')
    cf.set('p_grounding_drifting_causation', '1')
    cf.set('p_grounding_no_turn_causation', '0.000155')
    cf.set('p_headon_causation', '5e-05')
    cf.set('p_merging_causation', '0.00013')
    cf.set('p_overtaking_causation', '0.00011')
    misc = ET.SubElement(gs, 'misc')
    misc.set('fastferry_reduction_factor', '20')
    misc.set('meantime_between_checks', '240')
    misc.set('passengership_reductionfactor', '20')


def generate_iwrap_xml(data: dict) -> ET.Element:
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
        build_leg_leg_distributions(waypoints_el, data.get('segment_data', {}), wl, leg_map)
    legs_el = root.find('legs')
    if legs_el is not None:
        _wire_leg_guids(legs_el, data, td_guid_map, mal_guid_map)
    combined_areas = list(data.get('depths', []) or []) + _build_objects_as_depths(data.get('objects', []) or [])
    build_areas(root, combined_areas)
    ET.SubElement(root, 'traffic_areas')
    build_drifting(root, data.get('drift', {}))
    ET.SubElement(root, 'area_traffic')
    ET.SubElement(root, 'routes')
    ET.SubElement(root, 'tug_boats')
    _add_global_settings(root)
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
                except Exception:
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
        except Exception:
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
    is_object_area = False
    try:
        is_object_area = float(dep_depth) == -1.0
    except Exception:
        is_object_area = dep_depth.strip() == '-1'
    return dep_id, dep_depth, polygon, is_object_area


def _collect_coords_from_polygon(polygon) -> List[List[Tuple[float, float]]]:
    if isinstance(polygon, str):
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
            except Exception:
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
    except Exception:
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
        inner = s[open_paren+1:close_paren].strip()

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
    except Exception:
        return []
    return polygons


# ── build_traffic_distributions helpers ───────────────────────────────────────

def _round2(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "0"


def _sanitize_num(v) -> float:
    try:
        f = float(v.lower()) if isinstance(v, str) else float(v)
        return 0.0 if (f != f or f in (float('inf'), float('-inf'))) else f
    except Exception:
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
    except Exception:
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
            except Exception:
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


def _mat_safe(mat, r: int, c: int) -> float:
    return _sanitize_num(mat[r][c]) if mat and r < len(mat) and c < len(mat[r]) else 0.0


def _mat_avg(mat, col: int, rows: list) -> float:
    vals = [_mat_safe(mat, r, col) for r in rows if mat and r < len(mat) and col < len(mat[r])]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


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


def _emit_shiptypes_grouped(shiptypes_el: ET.Element, freq_mat, speed_mat, draft_mat,
                             height_mat, beam_mat, types: list, intervals: list):
    groups: dict[str, list[int]] = {}
    for r, type_name in enumerate(types):
        groups.setdefault(_UI_TO_IWRAP.get(type_name, 'Other ship'), []).append(r)
    for iwrap_name, rows in groups.items():
        st_el = ET.SubElement(shiptypes_el, 'shiptype')
        for attr, val in [('causation_reduction_factor', '0'), ('freq_adjustment', '1'),
                          ('grounding_safety_margin', '-1'), ('name', iwrap_name)]:
            st_el.set(attr, val)
        cats_el = ET.SubElement(st_el, 'categories')
        for c, interval in enumerate(intervals):
            label = re.sub(r"\s*-\s*", "-", str(interval.get('label', f'{c}'))).strip()
            freq_sum = sum(_mat_safe(freq_mat, r, c) for r in rows)
            if freq_sum <= 0:
                continue
            _add_category_attr(cats_el, {
                'name': label, 'freq': round(freq_sum, 2),
                'speed': _mat_avg(speed_mat, c, rows), 'draught': _mat_avg(draft_mat, c, rows),
                'height_1': _mat_avg(height_mat, c, rows), 'width': _mat_avg(beam_mat, c, rows),
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
        east = leg_td.get('East going', {})
        west = leg_td.get('West going', {})
        guid_ftl, shiptypes_e = _create_td_element(td_root, leg_key, 'FTL')
        guid_ltf, shiptypes_w = _create_td_element(td_root, leg_key, 'LTF')
        td_guid_map[str(leg_key)] = {'ftl': guid_ftl, 'ltf': guid_ltf}
        for dir_data, shiptypes_el in [(east, shiptypes_e), (west, shiptypes_w)]:
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
    """Try to locate Ship Type Codes.csv near the XML file or in known paths."""
    candidates = [
        os.path.join(os.path.dirname(xml_path), 'Ship Type Codes.csv'),
        os.path.join(os.path.dirname(__file__), 'Ship Type Codes.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'tests', 'example_data', 'Ship Type Codes.csv'),
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
            'use_lognormal': False, 'dist_type': 'weibull', 'wb_shape': p1, 'wb_loc': p2, 'wb_scale': p0,
            'func': f"__import__('scipy.stats', fromlist=['weibull_min']).weibull_min(c={p1}, loc={p2}, scale={p0}).cdf(x)",
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
                        cats_by_key[(omrat_type, li)] = cat_data
                        if debug and cat_data['freq'] > 0:
                            print(f"  TD {td_name}: {omrat_type} ({li}) = {cat_data['freq']} ships/yr")
        td_map[guid] = {'name': td_name, 'categories': cats_by_key}
    return td_map, all_ship_types, all_length_intervals


def _parse_waypoints_el(wps_el) -> dict:
    if wps_el is None:
        return {}
    return {
        wp_el.get('guid', ''): {
            'name': wp_el.get('name', ''),
            'lat': float(wp_el.get('latitude', 0)),
            'lon': float(wp_el.get('longitude', 0)),
        }
        for wp_el in wps_el.findall('waypoint')
    }


def _parse_mal_el(mals_el) -> dict:
    mal_map: dict = {}
    if mals_el is None:
        return mal_map
    for mal_el in mals_el.findall('manoeuvring_aspects_leg'):
        guid = mal_el.get('guid', '')
        dp: dict = {'means': [], 'stds': [], 'weights': [], 'u_min': None, 'u_max': None, 'u_p': 0}
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
    if ftl_guid in mal_map:
        mal = mal_map[ftl_guid]
        for i, (m, s, w) in enumerate(zip(mal['means'], mal['stds'], mal['weights']), 1):
            seg[f'mean1_{i}'] = m
            seg[f'std1_{i}'] = s
            seg[f'weight1_{i}'] = w * 100.0
        if mal['u_min'] is not None:
            seg['u_min1'] = mal['u_min']
            seg['u_max1'] = mal['u_max']
            seg['u_p1'] = int(round(mal['u_p'] * 100.0))
    if ltf_guid in mal_map:
        mal = mal_map[ltf_guid]
        for i, (m, s, w) in enumerate(zip(mal['means'], mal['stds'], mal['weights']), 1):
            seg[f'mean2_{i}'] = -m
            seg[f'std2_{i}'] = s
            seg[f'weight2_{i}'] = w * 100.0
        if mal['u_min'] is not None:
            seg['u_min2'] = -mal['u_max']
            seg['u_max2'] = -mal['u_min']
            seg['u_p2'] = int(round(mal['u_p'] * 100.0))


def _haversine_m(start_point: str, end_point: str) -> float:
    from math import radians, cos, sin, asin, sqrt
    try:
        sp, ep = start_point.split(), end_point.split()
        if len(sp) == 2 and len(ep) == 2:
            lon1, lat1, lon2, lat2 = map(radians, [float(sp[0]), float(sp[1]), float(ep[0]), float(ep[1])])
            a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
            return 2 * asin(sqrt(a)) * 6371000
    except Exception:
        pass
    return 0.0


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
        if (v := leg_el.get('max_bearing_angle')) is not None:
            seg['ai1'] = seg['ai2'] = float(v)
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
        east_data = {k: [[0.0]*ni for _ in range(nt)] for k in td_keys}
        east_data['Scaling (%)'] = [[100.0]*ni for _ in range(nt)]
        west_data = {k: [[0.0]*ni for _ in range(nt)] for k in td_keys}
        west_data['Scaling (%)'] = [[100.0]*ni for _ in range(nt)]
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
        height_val = depth_val.strip().lstrip('-') if depth_val.strip().startswith('-') else ('0' if is_obj else depth_val)
    return name, depth_val, wkt, is_obj, height_val


def _parse_areas_el(areas_el) -> tuple[list, list]:
    depths, objects = [], []
    if areas_el is None:
        return depths, objects
    for area_el in areas_el.findall('area_polygon'):
        name, depth_val, wkt, is_obj, height_val = _classify_area_el(area_el)
        if is_obj:
            objects.append([name.replace('object_', ''), height_val, wkt])
        else:
            depths.append([name.replace('depth_', '').split('_')[0], depth_val, wkt])
    return depths, objects


def _parse_global_settings_el(gs_el, result: dict, debug: bool) -> None:
    if gs_el is None:
        return
    cf_el = gs_el.find('causation_factors')
    if cf_el is not None:
        pc = result['pc']
        cf_mapping = {
            'p_headon_causation': 'headon', 'p_overtaking_causation': 'overtaking',
            'p_crossing_causation': 'crossing', 'p_bend_causation': 'bend',
            'p_merging_causation': 'merging', 'p_grounding_causation': 'p_pc',
            'p_allision_causation': 'allision_pc',
            'p_grounding_drifting_causation': 'grounding_drifting_rf',
            'p_allision_drifting_causation': 'allision_drifting_rf',
            'p_grounding_no_turn_causation': 'grounding_no_turn_pc',
            'p_allision_no_turn_causation': 'allision_no_turn_pc',
        }
        for iwrap_key, omrat_key in cf_mapping.items():
            if (val := cf_el.get(iwrap_key)) is not None:
                try:
                    pc[omrat_key] = float(val)
                except (ValueError, TypeError):
                    pass
    misc_el = gs_el.find('misc')
    if misc_el is not None and (mtbc_val := misc_el.get('meantime_between_checks')) is not None:
        try:
            mtbc = float(mtbc_val)
            result['pc']['mean_time_between_checks'] = mtbc
            for seg in result['segment_data'].values():
                if seg.get('ai1', 0.0) == 0.0:
                    seg['ai1'] = mtbc
                if seg.get('ai2', 0.0) == 0.0:
                    seg['ai2'] = mtbc
            if debug:
                print(f"  Mean time between checks: {mtbc}s -> set ai on all segments")
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
                    sp1, ep1 = seg['Start_Point'].split(), seg['End_Point'].split()
                    sp2, ep2 = other['Start_Point'].split(), other['End_Point'].split()
                    b1 = math.degrees(math.atan2(float(ep1[0])-float(sp1[0]), float(ep1[1])-float(sp1[1]))) % 360
                    b2 = math.degrees(math.atan2(float(ep2[0])-float(sp2[0]), float(ep2[1])-float(sp2[1]))) % 360
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

    if all_ship_types and all_length_intervals:
        omrat_types, parsed_intervals = _build_ship_categories_from_intervals(all_length_intervals)
        result['ship_categories'] = {
            'types': omrat_types, 'length_intervals': parsed_intervals, 'selection_mode': 'ais',
        }
        intervals_list = [pi['label'] for pi in parsed_intervals]
        result['traffic_data'] = _build_traffic_matrices_el(
            root.find('legs'), td_map, omrat_types, intervals_list, debug
        )

    result['depths'], result['objects'] = _parse_areas_el(root.find('areas'))
    _parse_global_settings_el(root.find('global_settings'), result, debug)
    _compute_bend_angles(result['segment_data'], debug)
    if debug:
        print(f"\n{'='*70}")
        print(f"Segments: {len(result['segment_data'])}, Depths: {len(result['depths'])}, "
              f"Objects: {len(result['objects'])}, Traffic legs: {len(result['traffic_data'])}")
        print(f"{'='*70}\n")
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
