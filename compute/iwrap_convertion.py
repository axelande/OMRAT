# Detailed mapping between .omrat and IWRAP XML schema
import json
import re
import xml.etree.ElementTree as ET
import uuid
from xml.dom import minidom
from typing import List, Tuple

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

if __name__ == '__main__':
    write_iwrap_xml('tests/example_data/proj.omrat', 'tests/example_data/generated.xml')
