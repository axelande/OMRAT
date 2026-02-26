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
import xml.etree.ElementTree as ET
import uuid
from xml.dom import minidom
from typing import Dict, List, Optional, Tuple

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
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def build_drifting(parent: ET.Element, drift: dict):
    drifting = ET.SubElement(parent, 'drifting')
    # Set attributes on drifting per XSD
    if (v := drift.get('drift_p')) is not None:
        drifting.set('anchor_probability', str(v))
    if (v := drift.get('anchor_d')) is not None:
        drifting.set('max_anchor_depth', str(v))
    if (v := drift.get('speed') or drift.get('drift_speed')) is not None:
        drifting.set('drift_speed', str(v))
    if (v := drift.get('drift_blackout_other') or drift.get('drift_p')) is not None:
        # Best effort mapping if provided
        drifting.set('blackout_other', str(v))
    # Required children per XSD: repair_time and drift_directions
    repair = drift.get('repair', {}) if isinstance(drift.get('repair'), dict) else {}
    rep = ET.SubElement(drifting, 'repair_time')
    # Attributes on repair_time are optional
    for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type', 'weight']:
        if attr in repair and repair.get(attr) is not None:
            rep.set(attr, str(repair.get(attr)))
    # Optional repair_time_func
    func = repair.get('func') if isinstance(repair, dict) else None
    if func:
        rf = ET.SubElement(drifting, 'repair_time_func')
        rf.set('name', str(func))
    # Drift directions as attributes
    dd = ET.SubElement(drifting, 'drift_directions')
    rose = drift.get('rose', {}) if isinstance(drift.get('rose'), dict) else {}
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        akey = str(angle)
        if akey in rose and rose.get(akey) is not None:
            dd.set(f'angle_{angle}', str(rose.get(akey)))

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

def build_manoeuvring_aspects_legs(parent: ET.Element, segment_data: dict, td_guid_map: dict[str, dict[str, str]]):
    mals = ET.SubElement(parent, 'manoeuvring_aspects_legs')
    mal_guid_map: dict[str, dict[str, str]] = {}
    for seg_id, seg in segment_data.items():
        # One manoeuvring_aspects_leg per direction for this leg
        def _as_float(v):
            try:
                f = float(v)
                if f != f or f in (float('inf'), float('-inf')):
                    return 0.0
                return f
            except Exception:
                return 0.0

        # First-to-last (direction 1)
        mal1 = ET.SubElement(mals, 'manoeuvring_aspects_leg')
        guid_ftl = '{' + new_guid() + '}'
        mal1.set('guid', guid_ftl)
        mal1.set('name', str(seg.get('Leg_name', f'LEG_{seg_id}_FTL')))
        # Set requested default RF attributes
        for attr, default in [
            ('allision_causation_rf', '1'), ('allision_drifting_rf', '1'), ('allision_no_turn_rf', '1'),
            ('aton_reduction_factor', '1'), ('grounding_causation_rf', '1'), ('grounding_check_time', '0'),
            ('grounding_drifting_rf', '1'), ('grounding_no_turn_rf', '1'), ('headon_causation_rf', '1'),
            ('limit_width', 'false'), ('overtaking_causation_rf', '1')
        ]:
            mal1.set(attr, default)
        md1 = ET.SubElement(mal1, 'mixed_dist')
        md1.set('scale', '1')
        for i in range(1, 3+1):
            w = _as_float(seg.get(f'weight1_{i}', 0))
            if w <= 0:
                continue
            item = ET.SubElement(md1, 'mixed_dist_item')
            item.set('combi', '/Mean/Std. Dev.')
            item.set('param_0', str(seg.get(f'mean1_{i}', 0)))
            item.set('param_1', str(seg.get(f'std1_{i}', 0)))
            item.set('type', 'Normal')
            item.set('weight', str(w))
        w1 = _as_float(seg.get('u_p1', 0))
        if w1 > 0:
            u1 = ET.SubElement(md1, 'mixed_dist_item')
            u1.set('combi', '/Lower Bound/Upper Bound')
            u1.set('param_0', str(seg.get('u_min1', 0)))
            u1.set('param_1', str(seg.get('u_max1', 0)))
            u1.set('type', 'Uniform')
            u1.set('weight', str(w1))

        # Last-to-first (direction 2)
        mal2 = ET.SubElement(mals, 'manoeuvring_aspects_leg')
        guid_ltf = '{' + new_guid() + '}'
        mal2.set('guid', guid_ltf)
        mal2.set('name', str(seg.get('Leg_name', f'LEG_{seg_id}_LTF')))
        for attr, default in [
            ('allision_causation_rf', '1'), ('allision_drifting_rf', '1'), ('allision_no_turn_rf', '1'),
            ('aton_reduction_factor', '1'), ('grounding_causation_rf', '1'), ('grounding_check_time', '0'),
            ('grounding_drifting_rf', '1'), ('grounding_no_turn_rf', '1'), ('headon_causation_rf', '1'),
            ('limit_width', 'false'), ('overtaking_causation_rf', '1')
        ]:
            mal2.set(attr, default)
        md2 = ET.SubElement(mal2, 'mixed_dist')
        md2.set('scale', '1')
        for i in range(1, 3+1):
            w = _as_float(seg.get(f'weight2_{i}', 0))
            if w <= 0:
                continue
            item = ET.SubElement(md2, 'mixed_dist_item')
            item.set('combi', '/Mean/Std. Dev.')
            # IWRAP expects opposite sign for direction-2 parameters
            mean2 = -_as_float(seg.get(f'mean2_{i}', 0))
            std2 = _as_float(seg.get(f'std2_{i}', 0))
            item.set('param_0', str(mean2))
            item.set('param_1', str(std2))
            item.set('type', 'Normal')
            item.set('weight', str(w))
        w2 = _as_float(seg.get('u_p2', 0))
        if w2 > 0:
            u2 = ET.SubElement(md2, 'mixed_dist_item')
            u2.set('combi', '/Lower Bound/Upper Bound')
            # Negate bounds for direction-2 and keep ordering low<=high
            umin2 = -_as_float(seg.get('u_min2', 0))
            umax2 = -_as_float(seg.get('u_max2', 0))
            lo, hi = (umin2, umax2) if umin2 <= umax2 else (umax2, umin2)
            u2.set('param_0', str(lo))
            u2.set('param_1', str(hi))
            u2.set('type', 'Uniform')
            u2.set('weight', str(w2))

        # Track generated MAL GUIDs per leg and direction
        mal_guid_map[str(seg_id)] = {'ftl': guid_ftl, 'ltf': guid_ltf}

    return mal_guid_map

def generate_iwrap_xml(data: dict) -> ET.Element:
    root = ET.Element('riskmodel')
    # Basic attributes (minimal placeholders)
    root.set('name', data.get('project_name', 'OMRAT Project'))
    root.set('guid', new_guid())
    # Populate common metadata seen in case2_23.xml with safe defaults (only those allowed by XSD)
    meta_defaults = {
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
        'use_fixed_to_year_factor': 'false', 'misc': 'B|calc_allision|1@B|calc_area|1@B|calc_collisions|1@B|calc_crossing|1@B|calc_drifting_allision|1@B|calc_drifting_grounding|1@B|calc_grounding|1@B|calc_headon_overtaking|1@B|calc_powered_allision|1@B|calc_powered_grounding|1',
        'default_class_a_length': '50', 'default_class_b_length': '25',
        'default_class_a_type': 'Other ship', 'default_class_b_type': 'Other ship',
    }
    for k, v in meta_defaults.items():
        root.set(k, str(data.get(k, v)))

    # Emit elements in XSD order
    ET.SubElement(root, 'ship_type_data')  # empty
    ET.SubElement(root, 'passagelines')    # empty
    # Optional model_area omitted

    # Traffic distributions from traffic_data (optional but included)
    td_guid_map = build_traffic_distributions(root, data.get('traffic_data', {}), data.get('segment_data', {}), data.get('ship_categories'))

    # waypoints populated; add manoeuvring_aspects_legs BEFORE legs to satisfy XSD order
    wl = build_waypoints(root, data.get('segment_data', {}))
    mal_guid_map = build_manoeuvring_aspects_legs(root, data.get('segment_data', {}), td_guid_map)
    leg_map = build_legs(root, data.get('segment_data', {}), wl)
    waypoints_el = root.find('waypoints')
    if waypoints_el is not None:
        build_leg_leg_distributions(waypoints_el, data.get('segment_data', {}), wl, leg_map)

    # After MAL and TDs are built, set leg references to corresponding GUIDs
    legs_el = root.find('legs')
    if legs_el is not None:
        for seg_id, seg in data.get('segment_data', {}).items():
            td_pair = td_guid_map.get(str(seg_id), {})
            ftl_td_guid = td_pair.get('ftl')
            ltf_td_guid = td_pair.get('ltf')
            mal_pair = mal_guid_map.get(str(seg_id), {})
            ftl_mal_guid = mal_pair.get('ftl')
            ltf_mal_guid = mal_pair.get('ltf')
            # Find leg by name
            target_name = str(seg.get('Leg_name', ''))
            for leg_el in legs_el.findall('leg'):
                if leg_el.get('name', '') == target_name:
                    if ftl_td_guid:
                        leg_el.set('traffic_distribution_first_to_last_guid', ftl_td_guid)
                    if ltf_td_guid:
                        leg_el.set('traffic_distribution_last_to_first_guid', ltf_td_guid)
                    if ftl_mal_guid:
                        leg_el.set('man_aspects_first_to_last_guid', ftl_mal_guid)
                    if ltf_mal_guid:
                        leg_el.set('man_aspects_last_to_first_guid', ltf_mal_guid)
                    break

    # Areas from depths in .omrat (required).
    # Treat objects the same as depths with a fixed depth of -1.
    depths_list = data.get('depths', []) or []
    objects_list = data.get('objects', []) or []
    objects_as_depths = []
    for obj in objects_list:
        # Support either dict or list rows [id, height, polygon]
        if isinstance(obj, dict):
            oid = obj.get('id', '')
            poly = obj.get('polygon', '')
            objects_as_depths.append([str(oid), str(-1), str(poly)])
        else:
            try:
                oid, _height, poly = obj
                objects_as_depths.append([str(oid), str(-1), str(poly)])
            except Exception:
                continue
    combined_areas = list(depths_list) + objects_as_depths
    build_areas(root, combined_areas)

    # traffic_areas empty
    ET.SubElement(root, 'traffic_areas')

    # Drifting (required child elements inside)
    build_drifting(root, data.get('drift', {}))

    # area_traffic empty
    ET.SubElement(root, 'area_traffic')

    # routes and tug_boats empty
    ET.SubElement(root, 'routes')
    ET.SubElement(root, 'tug_boats')

    # bridges: not used; objects are exported as areas with depth=-1

    # Global settings hardcoded as requested (attributes on elements)
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
    return root

def build_areas(parent: ET.Element, depths: list):
    if not depths:
        return
    areas = ET.SubElement(parent, 'areas')
    # depths can be list of dicts {'id': id_, 'depth': depth, 'polygon': polygon}
    # or list rows [id, depth, polygon]; support both
    for idx, dep in enumerate(depths):
        if isinstance(dep, dict):
            dep_id = str(dep.get('id', idx))
            dep_depth = str(dep.get('depth', ''))
            polygon = dep.get('polygon', '')
        else:
            # assume [id, depth, polygon]
            dep_id = str(dep[0]) if len(dep) > 0 else str(idx)
            dep_depth = str(dep[1]) if len(dep) > 1 else ''
            polygon = dep[2] if len(dep) > 2 else ''
        # Determine if this entry represents an object-as-area (depth == -1)
        is_object_area = False
        try:
            is_object_area = float(dep_depth) == -1.0
        except Exception:
            is_object_area = dep_depth.strip() == '-1'
        # Collect one or more polygons from WKT or generic formats
        polygons: list[list[tuple[float, float]]] = []
        if isinstance(polygon, str):
            if polygon.strip().upper().startswith('MULTIPOLYGON'):
                polygons = parse_wkt_multipolygon(polygon)
            else:
                coords = parse_wkt_polygon(polygon)
                if coords:
                    polygons = [coords]
                else:
                    # Try generic parser
                    coords = parse_generic_polygon(polygon)
                    if coords:
                        polygons = [coords]
        elif isinstance(polygon, list):
            coords = []
            for pair in polygon:
                try:
                    a, b = float(pair[0]), float(pair[1])
                    # Assume [lon, lat]
                    lon, lat = a, b
                    coords.append((lat, lon))
                except Exception:
                    continue
            if coords:
                polygons = [coords]

        # Ensure polygons are simple and valid using Shapely if available
        def make_simple(parts: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
            if not parts:
                return []
            if not HAVE_SHAPELY:
                return parts
            result: List[List[Tuple[float, float]]] = []
            for coords in parts:
                try:
                    # coords are (lat, lon); Shapely expects (lon, lat)
                    ring = [(lon, lat) for (lat, lon) in coords]
                    poly = ShpPolygon(ring)
                    geom = poly
                    if (not poly.is_valid) or (not poly.is_simple):
                        try:
                            # Prefer Shapely 2.0 make_valid when available
                            from shapely import make_valid as _make_valid
                            geom = _make_valid(poly)
                        except Exception:
                            # Classic fix for many self-intersections
                            geom = poly.buffer(0)
                    # Collect polygon parts
                    parts_out = []
                    if isinstance(geom, ShpPolygon):
                        parts_out = [geom]
                    elif isinstance(geom, ShpMultiPolygon):
                        parts_out = list(geom.geoms)
                    elif isinstance(geom, ShpGeometryCollection):
                        parts_out = [g for g in geom.geoms if isinstance(g, ShpPolygon)]
                    for p in parts_out:
                        if not p.is_valid:
                            # Try final cleanup
                            p = p.buffer(0)
                        if p.is_valid and p.area > 0:
                            ext = list(p.exterior.coords)
                            # Back to (lat, lon)
                            result.append([(lat, lon) for (lon, lat) in ext])
                except Exception:
                    # On any error, fall back to original coords
                    result.append(coords)
            return result

        polygons = make_simple(polygons)

        # Emit one area_polygon per polygon part
        if not polygons:
            # Create empty polygon container to keep structure
            area = ET.SubElement(areas, 'area_polygon')
            attrs = {
                'area_style_id': '',
                'causationReductionFactor': '1',
                'depth': dep_depth,
                'guid': new_guid(),
                'is_line': 'false',
                'is_right': 'false',
            }
            if is_object_area:
                attrs.update({
                    'filename': 'object',
                    'name': f"object_{dep_id}",
                    'structure_type': 'Other',
                    'style_mode': '1',
                    'type': '1',
                })
            else:
                attrs.update({
                    'filename': 'depth',
                    'name': f"depth_{dep_id}",
                    'structure_type': '',
                    'style_mode': '0',
                    'type': '0',
                })
            for k, v in attrs.items():
                area.set(k, v)
            ET.SubElement(area, 'polygon')
        else:
            for part_idx, coords in enumerate(polygons, start=1):
                area = ET.SubElement(areas, 'area_polygon')
                # Name depth_{id}_{part} if multiple parts
                part_suffix = f"_{part_idx}" if len(polygons) > 1 else ""
                attrs = {
                    'area_style_id': '',
                    'causationReductionFactor': '1',
                    'depth': dep_depth,
                    'guid': new_guid(),
                    'is_line': 'false',
                    'is_right': 'false',
                }
                if is_object_area:
                    attrs.update({
                        'filename': 'object',
                        'name': f"object_{dep_id}",
                        'structure_type': 'Other',
                        'style_mode': '1',
                        'type': '1',
                    })
                else:
                    attrs.update({
                        'filename': 'depth',
                        'name': f"depth_{dep_id}{part_suffix}",
                        'structure_type': '',
                        'style_mode': '0',
                        'type': '0',
                    })
                for k, v in attrs.items():
                    area.set(k, v)
                poly_el = ET.SubElement(area, 'polygon')
                for lat, lon in coords:
                    item = ET.SubElement(poly_el, 'item')
                    item.set('guid', new_guid())
                    # Correct emission: input coords are (lat, lon)
                    item.set('lat', str(lat))
                    item.set('lon', str(lon))

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

def build_traffic_distributions(parent: ET.Element, traffic_data: dict, segment_data: dict, ship_categories: dict | None = None):
    """Map .omrat traffic_data into traffic_distributions schema seen in case2_23.xml.

    Schema:
    <traffic_distributions>
      <traffic_distribution adjustment_factor="1" dont_use="false" guid="{}" name="TD_1" only_uniform="false" season="-1">
        <shiptypes>
          <shiptype causation_reduction_factor="0" freq_adjustment="1" grounding_safety_margin="-1" name="...">
            <categories>
              <category causation_reduction_factor="0" depth="0" draught="0" freq="150" height_1="11.3" height_2="0" height_3="0" name="25-50" p_ballast="0" speed="9.6" width="0"/>
            </categories>
          </shiptype>
        </shiptypes>
      </traffic_distribution>
    </traffic_distributions>
    """
    td_root = ET.SubElement(parent, 'traffic_distributions')
    td_guid_map: dict[str, dict[str, str]] = {}

    def round2(v):
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "0"

    # Create one traffic_distribution per leg to keep linkage simple
    global_categories = traffic_data.get('categories', []) if isinstance(traffic_data, dict) else []
    global_shiptypes = traffic_data.get('shiptypes', []) if isinstance(traffic_data, dict) else []
    types = []
    intervals = []
    if isinstance(ship_categories, dict):
        types = ship_categories.get('types', []) or []
        intervals = ship_categories.get('length_intervals', []) or []

    def add_category_attr(categories_el: ET.Element, cat: dict):
        c_el = ET.SubElement(categories_el, 'category')
        # Attributes per case schema
        c_el.set('causation_reduction_factor', '0')
        c_el.set('depth', round2(cat.get('depth', 0)))
        # draught (water draft)
        c_el.set('draught', round2(cat.get('draft', cat.get('draught', 0))))
        # freq must be integer per XSD
        try:
            freq_val = int(round(float(cat.get('freq', 0))))
        except Exception:
            freq_val = 0
        c_el.set('freq', str(freq_val))
        # height_1 primary air draft; height_2/3 optional extras default 0
        c_el.set('height_1', round2(cat.get('height', cat.get('height_1', 0))))
        c_el.set('height_2', round2(cat.get('height_2', 0)))
        c_el.set('height_3', round2(cat.get('height_3', 0)))
        c_el.set('name', str(cat.get('name', 'Unknown')))
        c_el.set('p_ballast', round2(cat.get('p_ballast', 0)))
        # speed as scalar per case file
        # Prefer mean; fall back to explicit 'speed' or min/max average
        speed = cat.get('speed')
        if speed is None:
            sm = cat.get('speed_mean')
            if sm is not None:
                speed = sm
            else:
                smin = cat.get('speed_min')
                smax = cat.get('speed_max')
                if smin is not None and smax is not None:
                    try:
                        speed = (float(smin) + float(smax)) / 2.0
                    except Exception:
                        speed = 0
                else:
                    speed = 0
        c_el.set('speed', round2(speed))
        # width from beam
        c_el.set('width', round2(cat.get('beam', cat.get('width', 0))))

    # Helper to safely get a matrix by name from a direction dict
    def get_matrix(direction_dict: dict, key: str):
        if not isinstance(direction_dict, dict):
            return None
        return direction_dict.get(key) or direction_dict.get(key.replace('_', ' '))

    def sanitize_num(v):
        """Replace inf/-inf and non-numeric with 0; pass through finite numbers."""
        try:
            # Strings like 'inf' or 'nan'
            if isinstance(v, str):
                lv = v.lower()
                if lv in ('inf', '+inf', '-inf', 'nan'):
                    return 0.0
                f = float(v)
                if f == float('inf') or f == float('-inf') or f != f:
                    return 0.0
                return f
            # Numeric
            f = float(v)
            if f == float('inf') or f == float('-inf') or f != f:
                return 0.0
            return f
        except Exception:
            return 0.0

    for leg_key, leg in segment_data.items():
        # Build two distributions: first_to_last (East going) and last_to_first (West going)
        td_guid_map[str(leg_key)] = {}
        def build_td(direction_name: str, dir_data: dict):
            td = ET.SubElement(td_root, 'traffic_distribution')
            guid = '{' + new_guid() + '}'
            td.set('adjustment_factor', '1')
            td.set('dont_use', 'false')
            td.set('guid', guid)
            td.set('name', f"TD_{leg_key}_{direction_name}")
            td.set('only_uniform', 'false')
            td.set('season', str(-1))
            shiptypes_el = ET.SubElement(td, 'shiptypes')
            return guid, shiptypes_el

        leg_td = traffic_data.get(str(leg_key), {}) if isinstance(traffic_data, dict) else {}
        east = leg_td.get('East going', {})
        west = leg_td.get('West going', {})

        guid_ftl, shiptypes_el_e = build_td('FTL', east)
        td_guid_map[str(leg_key)]['ftl'] = guid_ftl
        guid_ltf, shiptypes_el_w = build_td('LTF', west)
        td_guid_map[str(leg_key)]['ltf'] = guid_ltf

        # Determine shiptypes for this leg based on saved ship_categories (types/intervals)
        # matrices by variable per direction
        freq_e = get_matrix(east, 'Frequency (ships/year)')
        speed_e = get_matrix(east, 'Speed (knots)')
        draft_e = get_matrix(east, 'Draught (meters)')
        height_e = get_matrix(east, 'Ship heights (meters)')
        beam_e = get_matrix(east, 'Ship Beam (meters)')

        freq_w = get_matrix(west, 'Frequency (ships/year)')
        speed_w = get_matrix(west, 'Speed (knots)')
        draft_w = get_matrix(west, 'Draught (meters)')
        height_w = get_matrix(west, 'Ship heights (meters)')
        beam_w = get_matrix(west, 'Ship Beam (meters)')

        # Helper to emit categories into a target shiptypes container
        def emit_shiptypes(shiptypes_target: ET.Element, freq_mat, speed_mat, draft_mat, height_mat, beam_mat):
            if not types:
                leg_shiptypes = global_shiptypes or [{'name': 'General cargo ship', 'categories': global_categories or leg_td.get('categories', [])}]
                for st in leg_shiptypes:
                    st_el = ET.SubElement(shiptypes_target, 'shiptype')
                    st_el.set('causation_reduction_factor', str(st.get('causation_reduction_factor', 0)))
                    st_el.set('freq_adjustment', str(st.get('freq_adjustment', 1)))
                    st_el.set('grounding_safety_margin', str(st.get('grounding_safety_margin', -1)))
                    st_el.set('name', str(st.get('name', 'Unknown')))
                    cats_el = ET.SubElement(st_el, 'categories')
                    cats = st.get('categories', []) or []
                    for cat in cats:
                        if isinstance(cat, dict):
                            add_category_attr(cats_el, cat)
                return

            ui_to_iwrap = {
                'Fishing': 'Fishing ship',
                'Towing': 'Support ship',
                'Dredging or underwater ops': 'Support ship',
                'Diving ops': 'Support ship',
                'Military ops': 'Other ship',
                'Sailing': 'Pleasure boat',
                'Pleasure Craft': 'Pleasure boat',
                'High speed craft (HSC)': 'Fast ferry',
                'Pilot Vessel': 'Support ship',
                'Search and Rescue vessel': 'Support ship',
                'Tug': 'Support ship',
                'Port Tender': 'Support ship',
                'Anti-pollution equipment': 'Support ship',
                'Law Enforcement': 'Support ship',
                'Spare': 'Other ship',
                'Medical Transport': 'Support ship',
                'Noncombatant ship according to RR Resolution No. 18': 'Other ship',
                'Passenger, all ships of this type': 'Passenger ship',
                'Cargo, all ships of this type': 'General cargo ship',
                'Tanker, all ships of this type': 'Oil products tanker',
                'Other Type, all ships of this type': 'Other ship',
            }

            groups: dict[str, list[int]] = {}
            for r, type_name in enumerate(types):
                target = ui_to_iwrap.get(type_name, 'Other ship')
                groups.setdefault(target, []).append(r)

            for iwrap_name, rows in groups.items():
                st_el = ET.SubElement(shiptypes_target, 'shiptype')
                st_el.set('causation_reduction_factor', '0')
                st_el.set('freq_adjustment', '1')
                st_el.set('grounding_safety_margin', '-1')
                st_el.set('name', iwrap_name)
                cats_el = ET.SubElement(st_el, 'categories')
                for c, interval in enumerate(intervals):
                    label_raw = str(interval.get('label', f'{c}'))
                    # Normalize label formatting: remove spaces around hyphens (e.g., "0 - 25" -> "0-25")
                    label = re.sub(r"\s*-\s*", "-", label_raw).strip()
                    freq_sum = 0.0
                    speed_vals = []
                    draught_vals = []
                    height_vals = []
                    width_vals = []
                    for r in rows:
                        fv = freq_mat[r][c] if freq_mat and r < len(freq_mat) and c < len(freq_mat[r]) else 0
                        freq_sum += sanitize_num(fv)
                        sv = speed_mat[r][c] if speed_mat and r < len(speed_mat) and c < len(speed_mat[r]) else None
                        if sv is not None:
                            speed_vals.append(sanitize_num(sv))
                        dv = draft_mat[r][c] if draft_mat and r < len(draft_mat) and c < len(draft_mat[r]) else None
                        if dv is not None:
                            draught_vals.append(sanitize_num(dv))
                        hv = height_mat[r][c] if height_mat and r < len(height_mat) and c < len(height_mat[r]) else None
                        if hv is not None:
                            height_vals.append(sanitize_num(hv))
                        bv = beam_mat[r][c] if beam_mat and r < len(beam_mat) and c < len(beam_mat[r]) else None
                        if bv is not None:
                            width_vals.append(sanitize_num(bv))
                    if freq_sum <= 0:
                        continue
                    speed = round(sum(speed_vals)/len(speed_vals), 2) if speed_vals else 0
                    draught = round(sum(draught_vals)/len(draught_vals), 2) if draught_vals else 0
                    height1 = round(sum(height_vals)/len(height_vals), 2) if height_vals else 0
                    width = round(sum(width_vals)/len(width_vals), 2) if width_vals else 0
                    cat_dict = {
                        'name': label,
                        'freq': round(freq_sum, 2),
                        'speed': speed,
                        'draught': draught,
                        'height_1': height1,
                        'width': width,
                        'depth': 0,
                        'p_ballast': 0,
                    }
                    add_category_attr(cats_el, cat_dict)

        # Emit shiptypes separately per direction
        emit_shiptypes(shiptypes_el_e, freq_e, speed_e, draft_e, height_e, beam_e)
        emit_shiptypes(shiptypes_el_w, freq_w, speed_w, draft_w, height_w, beam_w)

    # Return GUID mapping for use by legs and manoeuvring aspects
    return td_guid_map

def write_iwrap_xml(json_path: str, output_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    root = generate_iwrap_xml(data)
    xml_str = prettify_xml(root)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

# Example usage:
#main('tests/example_data/proj.omrat', 'tests/example_data/iwrap.xml')
#run_structure_checks('tests/example_data/proj.omrat')

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


def parse_iwrap_xml(xml_path: str, debug: bool = False) -> dict:
    """Parse IWRAP XML file and convert to .omrat JSON format.

    Handles missing fields gracefully, particularly 'dist1' and 'dist2' which
    don't exist in IWRAP XML files.

    Args:
        xml_path: Path to the IWRAP XML file
        debug: If True, print debug information during parsing

    Returns:
        Dictionary in .omrat JSON format
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Load Ship Type Codes lookup table for deriving dimensions when IWRAP
    # provides zero values for draught, beam, etc.
    stc_csv = _find_ship_type_csv(xml_path)
    ship_type_codes = _load_ship_type_codes(stc_csv) if stc_csv else {}
    if debug:
        if ship_type_codes:
            print(f"  Loaded Ship Type Codes from {stc_csv} ({len(ship_type_codes)} entries)")
        else:
            print("  Ship Type Codes.csv not found; will fall back to L/12 draught estimate")

    result = {
        'project_name': root.get('name', 'Imported Project'),
        'traffic_data': {},
        'segment_data': {},
        'drift': {},
        'depths': [],
        'objects': [],
        'ship_categories': None,
        'pc': {
            'p_pc': 0.00016,  # Default value
            'd_pc': 1  # Default value
        },
    }

    # Import basic metadata
    for attr in ['fv', 'major', 'minor', 'seasons', 'current_season']:
        if root.get(attr) is not None:
            result[attr] = root.get(attr)

    # Import drifting settings with required defaults
    drifting_el = root.find('drifting')
    drift_data = {
        'drift_p': 1,  # Default (must be int)
        'anchor_p': 0.95,  # Default
        'anchor_d': 7,  # Default (must be int)
        'speed': 1.0,  # Default
        'rose': {str(angle): 0.125 for angle in [0, 45, 90, 135, 180, 225, 270, 315]},  # Equal distribution
        'repair': {
            'func': '',  # Required field
            'std': 0.95,
            'loc': 0.2,
            'scale': 0.85,
            'use_lognormal': True
        }
    }

    if drifting_el is not None:
        if drifting_el.get('anchor_probability') is not None:
            drift_data['anchor_p'] = float(drifting_el.get('anchor_probability'))
            # drift_p must be int (probability as 0 or 1)
            drift_data['drift_p'] = int(round(float(drifting_el.get('anchor_probability'))))
        if drifting_el.get('max_anchor_depth') is not None:
            drift_data['anchor_d'] = int(round(float(drifting_el.get('max_anchor_depth'))))
        if drifting_el.get('drift_speed') is not None:
            drift_data['speed'] = float(drifting_el.get('drift_speed'))

        # Parse repair time
        repair_el = drifting_el.find('repair_time')
        if repair_el is not None:
            for attr in ['combi', 'param_0', 'param_1', 'param_2', 'type', 'weight']:
                val = repair_el.get(attr)
                if val is not None:
                    try:
                        drift_data['repair'][attr] = float(val)
                    except ValueError:
                        drift_data['repair'][attr] = val

            # Map IWRAP repair time to scipy.stats.lognorm(s, loc, scale) format.
            # IWRAP combi "/Mean/Std. Dev./Lower Bound" means:
            #   param_0 = Mean of repair time (hours)
            #   param_1 = Std deviation of repair time (hours)
            #   param_2 = Lower bound (minimum repair time, hours)
            # scipy.stats.lognorm uses: s=shape(sigma), loc=shift, scale=exp(mu)
            combi = repair_el.get('combi', '')
            p0 = float(repair_el.get('param_0', 0))
            p1 = float(repair_el.get('param_1', 0))
            p2 = float(repair_el.get('param_2', 0))
            if 'Mean' in combi and 'Std' in combi and 'Lower' in combi:
                iwrap_mean = p0
                iwrap_std = p1
                iwrap_lower = p2
                adj_mean = iwrap_mean - iwrap_lower
                if adj_mean > 0 and iwrap_std > 0:
                    sigma2 = math.log(1 + (iwrap_std / adj_mean) ** 2)
                    sigma = math.sqrt(sigma2)
                    mu = math.log(adj_mean) - sigma2 / 2
                    drift_data['repair']['std'] = sigma       # shape parameter s
                    drift_data['repair']['loc'] = iwrap_lower  # location (shift)
                    drift_data['repair']['scale'] = math.exp(mu)  # scale
                elif adj_mean > 0:
                    # Zero std dev: use a small sigma to avoid degenerate distribution
                    drift_data['repair']['std'] = 0.01
                    drift_data['repair']['loc'] = iwrap_lower
                    drift_data['repair']['scale'] = adj_mean
            else:
                # Unknown combi format: use params directly as scipy params
                if 'param_0' in repair_el.attrib:
                    drift_data['repair']['loc'] = p0
                if 'param_1' in repair_el.attrib:
                    drift_data['repair']['scale'] = p1
                if 'param_2' in repair_el.attrib:
                    drift_data['repair']['std'] = p2

        # Parse drift directions (rose)
        dd_el = drifting_el.find('drift_directions')
        if dd_el is not None:
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                val = dd_el.get(f'angle_{angle}')
                if val is not None:
                    drift_data['rose'][str(angle)] = float(val)

    result['drift'] = drift_data

    # Import traffic distributions and build ship categories
    td_map = {}  # Map guid to distribution data
    all_ship_types = set()
    all_length_intervals = set()

    # Define IWRAP to OMRAT ship type mapping
    iwrap_to_omrat_type = {
        'Fishing ship': 'Fishing',
        'Support ship': 'Towing',
        'Pleasure boat': 'Pleasure Craft',
        'Fast ferry': 'High speed craft (HSC)',
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

    tds_el = root.find('traffic_distributions')
    if tds_el is not None:
        if debug:
            print(f"\n📊 Found {len(list(tds_el.findall('traffic_distribution')))} traffic distributions")

        for td_el in tds_el.findall('traffic_distribution'):
            guid = td_el.get('guid', '')
            td_name = td_el.get('name', '')

            # Parse shiptypes and categories
            shiptypes_el = td_el.find('shiptypes')
            categories_by_type_and_length = {}  # {(type, length_interval): cat_data}

            if shiptypes_el is not None:
                for st_el in shiptypes_el.findall('shiptype'):
                    st_name = st_el.get('name', '')
                    omrat_type = iwrap_to_omrat_type.get(st_name, 'Other Type, all ships of this type')
                    all_ship_types.add(omrat_type)
                    ship_code = IWRAP_SHIP_TYPE_CODES.get(st_name, 0)

                    cats_el = st_el.find('categories')
                    if cats_el is not None:
                        for cat_el in cats_el.findall('category'):
                            length_interval = cat_el.get('name', '0-25')
                            all_length_intervals.add(length_interval)

                            draught_val = float(cat_el.get('draught', 0))
                            height_val = float(cat_el.get('height_1', 0))
                            beam_val = float(cat_el.get('width', 0))

                            # Derive missing dimensions from Ship Type Codes table
                            # when IWRAP provides 0 values
                            if (draught_val == 0 or beam_val == 0) and '-' in length_interval:
                                try:
                                    lpp_min = int(length_interval.split('-')[0])
                                except (ValueError, IndexError):
                                    lpp_min = -1

                                stc_entry = ship_type_codes.get((ship_code, lpp_min))
                                if stc_entry:
                                    if draught_val == 0:
                                        draught_val = round(stc_entry['draught'], 2)
                                    if beam_val == 0:
                                        beam_val = round(stc_entry['beam'], 2)
                                    if debug:
                                        print(f"    STC lookup ({st_name}, {length_interval}): "
                                              f"draught={draught_val}m, beam={beam_val}m")
                                elif draught_val == 0:
                                    # Fallback: estimate draught from mid-length / 12
                                    try:
                                        parts = length_interval.split('-')
                                        mid_length = (float(parts[0]) + float(parts[1])) / 2
                                        draught_val = round(mid_length / 12.0, 2)
                                        if debug:
                                            print(f"    L/12 fallback ({st_name}, {length_interval}): "
                                                  f"draught={draught_val}m")
                                    except (ValueError, IndexError):
                                        pass

                            cat_data = {
                                'shiptype': omrat_type,
                                'length_interval': length_interval,
                                'freq': float(cat_el.get('freq', 0)),
                                'speed': float(cat_el.get('speed', 0)),
                                'draught': draught_val,
                                'height': height_val,
                                'beam': beam_val,
                            }
                            categories_by_type_and_length[(omrat_type, length_interval)] = cat_data

                            if debug and cat_data['freq'] > 0:
                                print(f"  TD {td_name}: {omrat_type} ({length_interval}) = "
                                      f"{cat_data['freq']} ships/yr, T={draught_val}m, B={beam_val}m")

            td_map[guid] = {
                'name': td_name,
                'categories': categories_by_type_and_length,
            }

            if debug:
                print(f"  Stored TD {guid}: {td_name} with {len(categories_by_type_and_length)} categories")

    # Import waypoints (for building segment endpoints)
    waypoint_map = {}  # Map guid to waypoint data
    wps_el = root.find('waypoints')
    if wps_el is not None:
        for wp_el in wps_el.findall('waypoint'):
            guid = wp_el.get('guid', '')
            waypoint_map[guid] = {
                'name': wp_el.get('name', ''),
                'lat': float(wp_el.get('latitude', 0)),
                'lon': float(wp_el.get('longitude', 0)),
            }

    # Import manoeuvring aspects legs (distribution parameters)
    mal_map = {}  # Map guid to manoeuvring aspect data
    mals_el = root.find('manoeuvring_aspects_legs')
    if mals_el is not None:
        for mal_el in mals_el.findall('manoeuvring_aspects_leg'):
            guid = mal_el.get('guid', '')
            md_el = mal_el.find('mixed_dist')

            dist_params = {
                'means': [],
                'stds': [],
                'weights': [],
                'u_min': None,
                'u_max': None,
                'u_p': 0,
            }

            if md_el is not None:
                normal_items = []
                uniform_items = []

                for item_el in md_el.findall('mixed_dist_item'):
                    item_type = item_el.get('type', '')
                    if item_type == 'Normal':
                        normal_items.append({
                            'param_0': float(item_el.get('param_0', 0)),
                            'param_1': float(item_el.get('param_1', 0)),
                            'weight': float(item_el.get('weight', 0)),
                        })
                    elif item_type == 'Uniform':
                        uniform_items.append({
                            'param_0': float(item_el.get('param_0', 0)),
                            'param_1': float(item_el.get('param_1', 0)),
                            'weight': float(item_el.get('weight', 0)),
                        })

                # Store normal distribution parameters
                for item in normal_items:
                    dist_params['means'].append(item['param_0'])
                    dist_params['stds'].append(item['param_1'])
                    dist_params['weights'].append(item['weight'])

                # Store uniform distribution parameters
                if uniform_items:
                    item = uniform_items[0]  # Take first uniform item
                    dist_params['u_min'] = item['param_0']
                    dist_params['u_max'] = item['param_1']
                    dist_params['u_p'] = item['weight']

            mal_map[guid] = dist_params

    # Import legs (segments)
    legs_el = root.find('legs')
    if legs_el is not None:
        for idx, leg_el in enumerate(legs_el.findall('leg'), start=1):
            leg_guid = leg_el.get('guid', '')
            leg_name = leg_el.get('name', f'Leg_{idx}')

            # Get waypoint coordinates
            first_wp_guid = leg_el.get('first_waypoint_guid', '')
            last_wp_guid = leg_el.get('last_waypoint_guid', '')

            start_wp = waypoint_map.get(first_wp_guid, {})
            end_wp = waypoint_map.get(last_wp_guid, {})

            start_point = f"{start_wp.get('lon', 0)} {start_wp.get('lat', 0)}"
            end_point = f"{end_wp.get('lon', 0)} {end_wp.get('lat', 0)}"

            # Initialize segment with all required fields and defaults
            segment = {
                'Leg_name': leg_name,
                'Width': int(round(float(leg_el.get('max_width', 0)))),
                'Start_Point': start_point,
                'End_Point': end_point,
                'Dirs': ['East going', 'West going'],  # Required field
                'line_length': 0.0,  # Will be calculated by system
                'Route_Id': 0,  # Default route
                'Segment_Id': str(idx),
                # Initialize all distribution parameters with defaults
                'mean1_1': 0.0, 'std1_1': 0.0, 'weight1_1': 0.0,
                'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
                'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
                'mean2_1': 0.0, 'std2_1': 0.0, 'weight2_1': 0.0,
                'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
                'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
                'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0,
                'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0,
                'ai1': 0.0, 'ai2': 0.0,
                'dist1': [], 'dist2': [],  # Empty by default
            }

            # Get bearing angle
            if leg_el.get('max_bearing_angle') is not None:
                segment['ai1'] = float(leg_el.get('max_bearing_angle'))
                segment['ai2'] = float(leg_el.get('max_bearing_angle'))  # Same for both directions

            # Get manoeuvring aspects for both directions
            ftl_guid = leg_el.get('man_aspects_first_to_last_guid', '')
            ltf_guid = leg_el.get('man_aspects_last_to_first_guid', '')

            # Direction 1 (first to last)
            if ftl_guid in mal_map:
                mal_data = mal_map[ftl_guid]
                for i, (mean, std, weight) in enumerate(zip(
                    mal_data['means'], mal_data['stds'], mal_data['weights']
                ), start=1):
                    segment[f'mean1_{i}'] = mean
                    segment[f'std1_{i}'] = std
                    # Convert weight from 0-1 to 0-100 (percentage)
                    segment[f'weight1_{i}'] = weight * 100.0

                if mal_data['u_min'] is not None:
                    segment['u_min1'] = mal_data['u_min']
                    segment['u_max1'] = mal_data['u_max']
                    # Convert weight from 0-1 to 0-100 (percentage)
                    segment['u_p1'] = int(round(mal_data['u_p'] * 100.0))

            # Direction 2 (last to first) - Note: IWRAP negates these values
            if ltf_guid in mal_map:
                mal_data = mal_map[ltf_guid]
                for i, (mean, std, weight) in enumerate(zip(
                    mal_data['means'], mal_data['stds'], mal_data['weights']
                ), start=1):
                    # Negate mean back to original sign
                    segment[f'mean2_{i}'] = -mean
                    segment[f'std2_{i}'] = std
                    # Convert weight from 0-1 to 0-100 (percentage)
                    segment[f'weight2_{i}'] = weight * 100.0

                if mal_data['u_min'] is not None:
                    # Negate bounds back and reorder
                    umin = -mal_data['u_max']
                    umax = -mal_data['u_min']
                    segment['u_min2'] = umin
                    segment['u_max2'] = umax
                    # Convert weight from 0-1 to 0-100 (percentage)
                    segment['u_p2'] = int(round(mal_data['u_p'] * 100.0))

            # Calculate line_length from coordinates (lat/lon to meters)
            try:
                from math import radians, cos, sin, asin, sqrt

                # Parse coordinates from "lon lat" format
                start_parts = start_point.split()
                end_parts = end_point.split()

                if len(start_parts) == 2 and len(end_parts) == 2:
                    lon1, lat1 = float(start_parts[0]), float(start_parts[1])
                    lon2, lat2 = float(end_parts[0]), float(end_parts[1])

                    # Haversine formula to calculate distance in meters
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    r = 6371000  # Earth radius in meters
                    segment['line_length'] = c * r

                    if debug:
                        print(f"  ✓ Calculated line_length for {leg_name}: {segment['line_length']:.1f}m")
            except Exception as e:
                if debug:
                    print(f"  ⚠️ Warning: Could not calculate line_length for segment {idx}: {e}")
                segment['line_length'] = 0.0

            result['segment_data'][str(idx)] = segment

    # Build ship_categories and traffic_data matrices
    # Use complete OMRAT ship type list (not just those in IWRAP XML)
    # This ensures UI alignment
    omrat_ship_types_full = [
        'Fishing',
        'Towing',
        'Dredging or underwater ops',
        'Diving ops',
        'Military ops',
        'Sailing',
        'Pleasure Craft',
        'High speed craft (HSC)',
        'Pilot Vessel',
        'Search and Rescue vessel',
        'Tug',
        'Port Tender',
        'Anti-pollution equipment',
        'Law Enforcement',
        'Spare',
        'Medical Transport',
        'Noncombatant ship according to RR Resolution No. 18',
        'Passenger, all ships of this type',
        'Cargo, all ships of this type',
        'Tanker, all ships of this type',
        'Other Type, all ships of this type',
    ]

    if all_ship_types and all_length_intervals:
        ship_types_list = omrat_ship_types_full  # Use full list, not just imported types
        length_intervals_list = sorted(list(all_length_intervals))

        # Parse length intervals and create proper structure
        parsed_intervals = []
        for interval_str in length_intervals_list:
            parts = interval_str.split('-')
            if len(parts) == 2:
                try:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    parsed_intervals.append({
                        'min': min_val,
                        'max': max_val,
                        'label': interval_str
                    })
                except ValueError:
                    parsed_intervals.append({
                        'min': 0,
                        'max': 25,
                        'label': interval_str
                    })
            else:
                parsed_intervals.append({
                    'min': 0,
                    'max': 25,
                    'label': interval_str
                })

        # Sort intervals by min value to ensure proper order (25-50, 50-75, etc.)
        parsed_intervals.sort(key=lambda x: x['min'])
        # Rebuild length_intervals_list to match numeric sort order of parsed_intervals
        # (the original sorted() used lexicographic order which puts "100-125" before "25-50")
        length_intervals_list = [pi['label'] for pi in parsed_intervals]

        result['ship_categories'] = {
            'types': omrat_ship_types_full,  # Always use full list
            'length_intervals': parsed_intervals,
            'selection_mode': 'ais'
        }

        # Build traffic data for ALL segments
        # Use full OMRAT ship type list to ensure UI alignment
        if debug:
            print(f"\n🚢 Building traffic matrices...")
            print(f"  Ship types: {len(omrat_ship_types_full)}")
            print(f"  Length intervals: {len(parsed_intervals)}")
            print(f"  Matrix dimensions: {len(omrat_ship_types_full)} × {len(parsed_intervals)}")

        legs_el = root.find('legs')
        if legs_el is not None:
            for idx, leg_el in enumerate(legs_el.findall('leg'), start=1):
                # Get traffic distribution GUIDs for this leg
                ftl_guid = leg_el.get('traffic_distribution_first_to_last_guid', '')
                ltf_guid = leg_el.get('traffic_distribution_last_to_first_guid', '')

                if debug:
                    print(f"\n  Leg {idx}:")
                    print(f"    FTL GUID: {ftl_guid}")
                    print(f"    LTF GUID: {ltf_guid}")
                    print(f"    FTL in td_map: {ftl_guid in td_map}")
                    print(f"    LTF in td_map: {ltf_guid in td_map}")

                # Initialize empty matrices using full OMRAT ship type list
                num_types = len(omrat_ship_types_full)
                num_intervals = len(parsed_intervals)

                def create_empty_matrix():
                    return [[0.0 for _ in range(num_intervals)] for _ in range(num_types)]

                east_data = {
                    'Frequency (ships/year)': create_empty_matrix(),
                    'Speed (knots)': create_empty_matrix(),
                    'Draught (meters)': create_empty_matrix(),
                    'Ship heights (meters)': create_empty_matrix(),
                    'Ship Beam (meters)': create_empty_matrix(),
                }

                west_data = {
                    'Frequency (ships/year)': create_empty_matrix(),
                    'Speed (knots)': create_empty_matrix(),
                    'Draught (meters)': create_empty_matrix(),
                    'Ship heights (meters)': create_empty_matrix(),
                    'Ship Beam (meters)': create_empty_matrix(),
                }

                # Fill in data from IWRAP traffic distributions
                if ftl_guid in td_map:
                    td_info = td_map[ftl_guid]
                    filled_count = 0
                    for (ship_type, length_interval), cat_data in td_info['categories'].items():
                        try:
                            type_idx = omrat_ship_types_full.index(ship_type)
                            interval_idx = length_intervals_list.index(length_interval)
                            east_data['Frequency (ships/year)'][type_idx][interval_idx] = cat_data['freq']
                            east_data['Speed (knots)'][type_idx][interval_idx] = cat_data['speed']
                            east_data['Draught (meters)'][type_idx][interval_idx] = cat_data['draught']
                            east_data['Ship heights (meters)'][type_idx][interval_idx] = cat_data['height']
                            east_data['Ship Beam (meters)'][type_idx][interval_idx] = cat_data['beam']
                            filled_count += 1
                            if debug and cat_data['freq'] > 0:
                                print(f"      East [{type_idx},{interval_idx}] {ship_type} ({length_interval}): {cat_data['freq']} ships")
                        except (ValueError, IndexError) as e:
                            # Ship type not in full list or interval not found - skip
                            if debug:
                                print(f"      ERROR filling East data: {e}")
                                print(f"        ship_type={ship_type}, interval={length_interval}")
                    if debug:
                        print(f"    Filled {filled_count} cells for East going")

                if ltf_guid in td_map:
                    td_info = td_map[ltf_guid]
                    for (ship_type, length_interval), cat_data in td_info['categories'].items():
                        try:
                            type_idx = omrat_ship_types_full.index(ship_type)
                            interval_idx = length_intervals_list.index(length_interval)
                            west_data['Frequency (ships/year)'][type_idx][interval_idx] = cat_data['freq']
                            west_data['Speed (knots)'][type_idx][interval_idx] = cat_data['speed']
                            west_data['Draught (meters)'][type_idx][interval_idx] = cat_data['draught']
                            west_data['Ship heights (meters)'][type_idx][interval_idx] = cat_data['height']
                            west_data['Ship Beam (meters)'][type_idx][interval_idx] = cat_data['beam']
                        except (ValueError, IndexError) as e:
                            # Ship type not in full list or interval not found - skip
                            pass

                result['traffic_data'][str(idx)] = {
                    'East going': east_data,
                    'West going': west_data,
                }

                if debug:
                    total_east = sum(sum(row) for row in east_data['Frequency (ships/year)'])
                    total_west = sum(sum(row) for row in west_data['Frequency (ships/year)'])
                    print(f"    ✓ Leg {idx} traffic: East={int(total_east)}, West={int(total_west)} ships/year")

    # Import areas (depths and objects)
    areas_el = root.find('areas')
    if areas_el is not None:
        for area_el in areas_el.findall('area_polygon'):
            area_type = area_el.get('type', '0')
            depth_val = area_el.get('depth', '0')
            name = area_el.get('name', '')

            # Parse polygon coordinates
            poly_el = area_el.find('polygon')
            coords = []
            if poly_el is not None:
                for item_el in poly_el.findall('item'):
                    lat = float(item_el.get('lat', 0))
                    lon = float(item_el.get('lon', 0))
                    coords.append((lon, lat))

            # Build WKT polygon string (must be a closed ring per OGC spec)
            if coords:
                # Close the ring if first != last
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                coord_str = ', '.join([f"{lon} {lat}" for lon, lat in coords])
                wkt = f"POLYGON(({coord_str}))"
            else:
                wkt = ""

            # Check if this is an object (negative depth) or depth area
            # IWRAP uses negative depth values for objects:
            #   -1 for general objects
            #   -10, -12, etc. for bridges and other structures
            try:
                depth_num = float(depth_val)
                is_object = depth_num < 0
                height_val = str(abs(depth_num)) if is_object else depth_val
            except ValueError:
                is_object = depth_val.strip().startswith('-')
                height_val = depth_val.strip().lstrip('-') if is_object else depth_val

            if is_object:
                # This is an object (negative depth) - store as list: [id, height, polygon]
                obj_id = name.replace('object_', '').replace('BRIDGE_', 'BRIDGE_')
                result['objects'].append([obj_id, height_val, wkt])
            else:
                # This is a depth area - store as list: [id, depth, polygon]
                depth_id = name.replace('depth_', '').split('_')[0]
                result['depths'].append([depth_id, depth_val, wkt])

    # Import global settings (causation factors)
    gs_el = root.find('global_settings')
    if gs_el is not None:
        cf_el = gs_el.find('causation_factors')
        if cf_el is not None:
            # Store causation factors if needed
            pass

    if debug:
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  Segments: {len(result['segment_data'])}")
        print(f"  Depth areas: {len(result['depths'])}")
        print(f"  Objects: {len(result['objects'])}")
        print(f"\nDistribution parameters (IWRAP 0-1 weights → OMRAT 0-100%):")
        for seg_id, seg in list(result['segment_data'].items())[:1]:  # Show first segment
            print(f"  Segment {seg_id} ({seg['Leg_name']}):")
            print(f"    Line length: {seg['line_length']:.1f}m")
            print(f"    Dir1 weights: {seg['weight1_1']:.1f}%, {seg['weight1_2']:.1f}%, {seg['weight1_3']:.1f}%, uniform: {seg['u_p1']}%")
            print(f"    Dir2 weights: {seg['weight2_1']:.1f}%, {seg['weight2_2']:.1f}%, {seg['weight2_3']:.1f}%, uniform: {seg['u_p2']}%")
        print(f"  Traffic data legs: {len(result['traffic_data'])}")
        if result['ship_categories']:
            print(f"  Ship types: {len(result['ship_categories']['types'])}")
            print(f"  Length intervals: {len(result['ship_categories']['length_intervals'])}")
        for seg_id in sorted(result['traffic_data'].keys(), key=int)[:3]:
            traffic = result['traffic_data'][seg_id]
            total = sum(sum(row) for row in traffic['East going']['Frequency (ships/year)'])
            print(f"  Segment {seg_id}: {int(total)} ships/year (East going)")
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
