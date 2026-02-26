import json
import sys
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable

from qgis.PyQt.QtWidgets import QLabel
import matplotlib as mpl
mpl.use('Qt5Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
from numpy import exp, log
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem, QTreeWidgetItem, QTreeWidget, QWidget
from scipy import stats
from scipy.stats import norm, uniform
import shapely.wkt as sw
from shapely.errors import GEOSException
from shapely.ops import transform
try:
    from shapely import make_valid as shp_make_valid
except Exception:
    shp_make_valid = None
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

sys.path.append('.')

from compute.basic_equations import (
    get_head_on_collision_candidates,
    get_overtaking_collision_candidates,
    get_crossing_collision_candidates,
    get_bend_collision_candidates,
)


def _compass_idx_to_math_idx(compass_d_idx: int) -> int:
    """
    Convert compass direction index to math convention index.

    The wind rose uses compass convention (d_idx * 45):
    - d_idx=0 → compass 0° = North
    - d_idx=1 → compass 45° = NE
    - d_idx=2 → compass 90° = East
    - etc.

    The probability_holes arrays use math convention indices (index * 45):
    - index=0 → math 0° = East
    - index=1 → math 45° = NE
    - index=2 → math 90° = North
    - etc.

    Conversion: math_angle = (90 - compass_angle) % 360
                math_index = math_angle // 45
    """
    compass_angle = compass_d_idx * 45
    math_angle = (90 - compass_angle) % 360
    return math_angle // 45


def _extract_obstacle_segments(geom: BaseGeometry) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Extract individual line segments from a polygon boundary.

    IMPORTANT: This function normalizes polygon orientation to CCW (counter-clockwise)
    before extracting segments. This ensures consistent outward normal calculation
    in _segment_intersects_corridor().

    For CCW polygons:
    - Exterior ring goes counter-clockwise
    - Interior (hole) rings go clockwise
    - Outward normal = rotate segment vector 90° clockwise (right-hand rule)

    Args:
        geom: A shapely geometry (Polygon, MultiPolygon, etc.)

    Returns:
        List of ((x1, y1), (x2, y2)) tuples representing line segments
    """
    from shapely.geometry import polygon as shapely_polygon

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def extract_from_ring(ring_coords):
        coords = list(ring_coords)
        for i in range(len(coords) - 1):
            p1 = (float(coords[i][0]), float(coords[i][1]))
            p2 = (float(coords[i + 1][0]), float(coords[i + 1][1]))
            if p1 != p2:  # Skip zero-length segments
                segments.append((p1, p2))

    if isinstance(geom, Polygon):
        # Normalize polygon to CCW exterior, CW holes using shapely's orient()
        # This ensures consistent outward normal calculation
        oriented_geom = shapely_polygon.orient(geom, sign=1.0)  # 1.0 = CCW exterior
        extract_from_ring(oriented_geom.exterior.coords)
        for interior in oriented_geom.interiors:
            extract_from_ring(interior.coords)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            segments.extend(_extract_obstacle_segments(poly))
    elif hasattr(geom, 'boundary'):
        boundary = geom.boundary
        if hasattr(boundary, 'coords'):
            extract_from_ring(boundary.coords)
        elif hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                extract_from_ring(line.coords)

    return segments


def _create_drift_corridor(
    leg: LineString,
    drift_angle: float,
    distance: float,
    lateral_spread: float,
) -> Polygon | None:
    """
    Create the drift corridor polygon for a given leg and drift direction.

    Creates a polygon representing the area a ship could drift through,
    from the leg starting position to the maximum drift distance.

    This matches the approach in pdf_corrected_fast_probability_holes.py
    but uses convex hull to handle self-intersection cases.

    Args:
        leg: The traffic leg LineString
        drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
                     This matches pdf_corrected_fast_probability_holes.py
        distance: Maximum drift distance in meters
        lateral_spread: Half-width of corridor (in meters)

    Returns:
        Polygon representing the drift corridor, or None if invalid
    """
    import numpy as np
    from shapely.ops import unary_union

    leg_coords = np.array(leg.coords)
    if len(leg_coords) < 2:
        return None

    leg_start = leg_coords[0]
    leg_end = leg_coords[-1]
    leg_vec = leg_end - leg_start
    leg_length = np.linalg.norm(leg_vec)

    if leg_length == 0:
        return None

    leg_dir = leg_vec / leg_length
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    # Drift direction vector (math convention: 0=East, 90=North)
    drift_angle_rad = np.radians(drift_angle)
    drift_vec = np.array([np.cos(drift_angle_rad), np.sin(drift_angle_rad)]) * distance

    # Create leg rectangle corners (CCW order)
    p1 = leg_start - lateral_spread * perp_dir
    p2 = leg_start + lateral_spread * perp_dir
    p3 = leg_end + lateral_spread * perp_dir
    p4 = leg_end - lateral_spread * perp_dir

    # Create drifted rectangle corners (CCW order)
    p1_drift = p1 + drift_vec
    p2_drift = p2 + drift_vec
    p3_drift = p3 + drift_vec
    p4_drift = p4 + drift_vec

    # Create the two rectangles as separate polygons and union them
    # This avoids self-intersection issues when drift is along the leg direction
    leg_rect = Polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
    drift_rect = Polygon([tuple(p1_drift), tuple(p2_drift), tuple(p3_drift), tuple(p4_drift)])

    corridor = unary_union([leg_rect, drift_rect])

    # If union creates MultiPolygon (shouldn't happen but handle it), take convex hull
    if isinstance(corridor, MultiPolygon):
        corridor = corridor.convex_hull

    if corridor.is_empty or corridor.area == 0:
        return None

    return corridor


def _segment_intersects_corridor(
    segment: tuple[tuple[float, float], tuple[float, float]],
    corridor: Polygon,
    drift_angle: float | None = None,
    leg_centroid: tuple[float, float] | None = None,
    leg_line: LineString | None = None,
) -> bool:
    """
    Check if a line segment would be hit by ships drifting from the leg.

    A segment is hit if:
    1. The corridor geometrically intersects the segment (substantially, not just a point touch)
    2. The segment is ahead of the leg in the drift direction
    3. The drift direction "faces into" the segment's outward normal
       (ships must be moving toward the segment's blocking face)

    The key insight for obstacle polygons (assumed CCW): each edge has an outward normal
    pointing to the right of the edge vector. For a ship to hit an edge, it must be
    drifting INTO that outward normal (positive dot product).

    Args:
        segment: ((x1, y1), (x2, y2)) tuple
        corridor: Drift corridor polygon
        drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
        leg_centroid: (x, y) centroid of the leg
        leg_line: Optional LineString of the leg

    Returns:
        True if segment would be hit by drift
    """
    import numpy as np
    from shapely.geometry import Point

    p1, p2 = segment
    seg_line = LineString([p1, p2])

    # Basic intersection check
    if not corridor.intersects(seg_line):
        return False

    # Check if the intersection is substantial (not just a point touch)
    intersection = corridor.intersection(seg_line)

    if intersection.is_empty:
        return False
    if intersection.geom_type == 'Point':
        t = 0.01
        interior_p1 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        interior_p2 = (p1[0] + (1-t) * (p2[0] - p1[0]), p1[1] + (1-t) * (p2[1] - p1[1]))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        if not (corridor.contains(Point(interior_p1)) or
                corridor.contains(Point(interior_p2)) or
                corridor.contains(Point(mid))):
            return False

    if drift_angle is None or leg_centroid is None:
        return True

    # Drift direction vector (unit vector)
    drift_angle_rad = np.radians(drift_angle)
    drift_dir = np.array([np.cos(drift_angle_rad), np.sin(drift_angle_rad)])

    # Calculate segment vector and normal
    seg_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    seg_len = np.linalg.norm(seg_vec)
    if seg_len == 0:
        return False

    # Outward normal for CCW polygon: rotate segment vector 90° clockwise
    # For segment (p1 → p2), outward normal points to the RIGHT of the direction
    # Rotate (dx, dy) by -90°: (dy, -dx)
    seg_outward_normal = np.array([seg_vec[1], -seg_vec[0]]) / seg_len

    # Check if drift is parallel to segment (can't hit a parallel segment)
    drift_into_segment = np.dot(drift_dir, seg_outward_normal)
    if abs(drift_into_segment) < 0.17:  # Nearly parallel (< ~10° from parallel)
        return False

    # KEY CHECK: For a ship to hit this segment (enter the polygon through this face),
    # the drift direction must oppose the outward normal (negative dot product).
    # If drift_into_segment > 0, ships are moving in the same direction as the
    # outward normal, meaning they would EXIT through this face, not enter.
    if drift_into_segment > 0:
        return False

    # Check that the segment is not significantly behind the leg in the drift direction.
    # This prevents false positives where a wide corridor intersects a segment that is
    # BEHIND the leg in the drift direction (e.g., Leg 2 south of structure cannot hit
    # the top edge via S/SW/SE drift because those drift directions go away from structure).
    #
    # We check if the segment midpoint is ahead of the leg centroid in drift direction.
    # "Ahead" means the dot product of (segment_mid - leg_centroid) with drift_dir is positive.
    seg_mid = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    leg_center = np.array(leg_centroid)
    vec_to_segment = seg_mid - leg_center
    dist_to_segment = np.linalg.norm(vec_to_segment)

    # Dot product: positive means segment is in front of leg in drift direction
    distance_ahead = np.dot(vec_to_segment, drift_dir)

    # Allow significant tolerance because the corridor has lateral spread.
    # A segment that is slightly behind in the drift direction can still be
    # reachable by ships that start from the lateral edges of the leg.
    # Only reject if the segment is more than 50% of the way "behind" the leg.
    # This catches cases like Leg 2 (south) trying to hit Segment 2 (north top edge)
    # via S/SE/SW drift where the segment is very far behind.
    if distance_ahead < -0.5 * dist_to_segment:
        # Segment is substantially behind the leg in the drift direction
        return False

    return True
from basic_equations import (
    get_drifting_prob,
    get_Fcoll,
    powered_na,
    get_not_repaired,
    get_powered_grounding_cat1,
    get_powered_grounding_cat2,
)
from geometries.route import get_multiple_ed
from geometries.route import get_multi_drift_distance
from geometries.get_drifting_overlap import (
    DriftingOverlapVisualizer,
    compute_min_distance_by_object,
    compute_leg_overlap_fraction,
    compute_dir_overlap_fraction_by_object,
    compute_dir_leg_overlap_fraction_by_object,
)
from geometries.get_powered_overlap import (
    PoweredOverlapVisualizer,
    SimpleProjector as _PoweredProjector,
    _build_legs_and_obstacles,
    _parse_point,
    _run_all_computations,
)
# Use accurate dblquad method with parallel processing for all calculations
from geometries.calculate_probability_holes import compute_probability_holes
from geometries.result_layers import create_result_layers
from ui.show_geom_res import ShowGeomRes

if TYPE_CHECKING:
    from omrat import OMRAT
    
def get_distribution(segment_data:dict[str, Any], direction:int) -> tuple[list[Any], list[float]]:
    d = direction + 1 # given as 0, 1 and should be 1, 2
    distributions: list[Any] = []
    weights: list[float] = []
    
    for i in range(1, 4):
        if f'weight{d}_{i}' in segment_data:
            if segment_data[f'weight{d}_{i}'] > 0:
                di = norm(loc=float(segment_data[f'mean{d}_{i}']), scale=float(segment_data[f'std{d}_{i}']))
                distributions.append(di)
                weights.append(float(segment_data[f'weight{d}_{i}']))
            else:
                distributions.append(norm(loc=0, scale=1))
                weights.append(0)
        else:
            distributions.append(norm(loc=0, scale=1))
            weights.append(0)
    if float(segment_data.get(f'u_p{d}', 0)) > 0:
        low = float(segment_data[f'u_min{d}'])
        high = float(segment_data[f'u_max{d}'])
        distributions.append(uniform(loc=low, scale=high - low))
        weights.append(float(segment_data[f'u_p{d}']))
    else:
        distributions.append(uniform(loc=0, scale=1))
        weights.append(0)
    return distributions, weights
    
def clean_traffic(data:dict[str, Any]) -> list[tuple[LineString, list[Any], list[float], list[dict[str, float]], str]]:
    """List all ships that on each segment/direction"""
    traffics: list[tuple[LineString, list[Any], list[float], list[dict[str, float]], str]] = []
    for segment, dirs in data["traffic_data"].items():
        for k, (di, var) in enumerate(dirs.items()):
            distributions, weights = get_distribution(data["segment_data"][segment], k)
            if k == 0:
                geom_base: BaseGeometry = sw.loads(f'LineString({data["segment_data"][segment]["Start_Point"]}, {data["segment_data"][segment]["End_Point"]})')
            else:
                geom_base: BaseGeometry = sw.loads(f'LineString({data["segment_data"][segment]["End_Point"]}, {data["segment_data"][segment]["Start_Point"]})')
            assert(isinstance(geom_base, LineString))
            geom: LineString = geom_base
            leg_traffic: list[dict[str, float]] = []
            name = f"Leg {segment}-{data['segment_data'][segment]['Dirs'][k]}"
            for i, row in enumerate(var['Frequency (ships/year)']):
                for j, value in enumerate(row):
                    if value == '':
                        continue
                    if isinstance(value, str):
                        value = int(value)
                    if value > 0:
                        info: dict[str, float] = {'freq': value,
                                'speed': float(var['Speed (knots)'][i][j]), 
                                'draught': float(var['Draught (meters)'][i][j]), 
                                'height': float(var['Ship heights (meters)'][i][j]),
                                'ship_type': i,
                                'ship_size': j,
                                'direction': di,
                                }
                        leg_traffic.append(info)
            traffics.append((geom, distributions, weights, leg_traffic, name))
    return traffics


def safe_load_wkt(wkt: str) -> BaseGeometry | None:
    """Safely load a WKT string, returning None if invalid/empty."""
    if wkt is None:
        return None
    s = str(wkt).strip()
    if not s:
        return None
    try:
        return sw.loads(s)
    except (GEOSException, ValueError):
        return None

def load_areas(data:dict[str, Any])-> list[dict[str, Any]]:
    """Loads the objects and depths into one objs dict"""
    objs:list[dict[str, Any]] = []
    # Support list-based storage format per RootModelSchema: [id, value, polygon]
    for obj in data.get('objects', []):
        try:
            oid, height, wkt = obj
        except Exception:
            continue
        geom = safe_load_wkt(wkt)
        if geom is None:
            continue
        objs.append({'type': 'Structure', 'id': oid, 'height': height, 'wkt': geom})
    for dep in data.get('depths', []):
        try:
            did, depth, wkt = dep
        except Exception:
            continue
        geom = safe_load_wkt(wkt)
        if geom is None:
            continue
        objs.append({'type': 'Depth', 'id': did, 'depth': depth, 'wkt': geom})
    return objs

def split_structures_and_depths(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return two lists with attributes preserved, splitting MultiPolygons.

    - structures: [{'id': str, 'height': float, 'wkt': BaseGeometry}]
    - depths: [{'id': str, 'depth': float, 'wkt': BaseGeometry}]

    MultiPolygons are split into individual Polygons, each retaining the
    original attributes (id, height/depth). This ensures the depths list
    has the same length as depths_gdfs after transformation.
    """
    structures: list[dict[str, Any]] = []
    depths: list[dict[str, Any]] = []
    for obj in data.get('objects', []):
        try:
            oid, height, wkt = obj
        except Exception:
            continue
        geom = safe_load_wkt(wkt)
        if geom is None:
            continue
        try:
            hval = float(height)
        except Exception:
            continue
        # Split MultiPolygons into individual Polygons
        if geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                structures.append({'id': f'{oid}_{i}', 'height': hval, 'wkt': poly})
        else:
            structures.append({'id': str(oid), 'height': hval, 'wkt': geom})
    for dep in data.get('depths', []):
        try:
            did, depth, wkt = dep
        except Exception:
            continue
        geom = safe_load_wkt(wkt)
        if geom is None:
            continue
        try:
            dval = float(depth)
        except Exception:
            continue
        # Split MultiPolygons into individual Polygons
        if geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                depths.append({'id': f'{did}_{i}', 'depth': dval, 'wkt': poly})
        else:
            depths.append({'id': str(did), 'depth': dval, 'wkt': geom})
    return structures, depths

def write_res2dict(res, lin, o_name, obj_sum):
    """Write results to the res_dict"""
    if o_name not in res['o']:
        res['o'][o_name] = obj_sum
    else:
        res['o'][o_name] += obj_sum
    res['l'][lin['segment']][lin['direction']] += obj_sum
    res['l'][lin['segment']]['lin_sum'] += obj_sum
    res['all'][f"{lin['segment']} - {lin['direction']}"][o_name] = obj_sum
    res['tot_sum'] += obj_sum

def add_empty_segment_to_res_dict(res, lin):
    """Adds the segment and direction to the res dict"""
    if lin['segment'] not in res['l']:
        res['l'][lin['segment']] = {'lin_sum': 0}
        res['l'][lin['segment']][lin['direction']] = 0
    else:
        res['l'][lin['segment']][lin['direction']] = 0
    res['all'][f"{lin['segment']} - {lin['direction']}"] = {}

def populate_details(twRes: QTableWidget, res_dict: dict[str, Any]):
    twRes.clear()
    twRes.setColumnCount(len(res_dict['o']))
    twRes.setHorizontalHeaderLabels(list(res_dict['o'].keys()))
    twRes.setRowCount(len(res_dict['all']))
    twRes.setVerticalHeaderLabels(list(res_dict['all'].keys()))
    for row, r_key in enumerate(res_dict['all'].keys()):
        for col, c_key in enumerate(res_dict['o'].keys()):
            item = QTableWidgetItem(f"{res_dict['all'][r_key][c_key]:.2e}")
            twRes.setItem(row, col, item)

def populate_segment(tree: QTreeWidget, res_dict: dict[str, Any]):
    tree.clear()
    tree.setColumnCount(3)
    tree.setHeaderLabels(['Segment', 'Direction', 'Probability'])
    tree.setColumnWidth(0, 55)
    tree.setColumnWidth(1, 75)
    tree.setColumnWidth(2, 35)
    main_item = QTreeWidgetItem(tree)
    main_item.setText(0, 'All')
    main_item.setText(2, f"{res_dict['tot_sum']:.2e}")
    for segment, s_val in res_dict['l'].items():
        segment_item = QTreeWidgetItem()
        segment_item.setText(0, segment)
        segment_item.setText(2, f"{s_val['lin_sum']:.2e}")
        main_item.addChild(segment_item)
        # set the child
        for di, val in s_val.items():
            dir_item = QTreeWidgetItem()
            dir_item.setText(1, di)
            dir_item.setText(2, f"{val:.2e}")
            segment_item.addChild(dir_item)
 
def populate_object(tree: QTreeWidget, res_dict: dict[str, Any]):
    tree.clear()
    tree.setColumnCount(2)
    tree.setHeaderLabels(['Object', 'Probability'])
    tree.setColumnWidth(0, 75)
    tree.setColumnWidth(1, 35)
    main_item = QTreeWidgetItem(tree)
    main_item.setText(0, 'All')
    main_item.setText(1, f"{res_dict['tot_sum']:.2e}")
    for object, s_val in res_dict['o'].items():
        item = QTreeWidgetItem()
        item.setText(0, object)
        item.setText(1, f"{s_val:.2e}")
        main_item.addChild(item)

def powered_accidents(data) -> dict:
    width = 100
    max_distance = 25_000
    
    objs = load_areas(data)
    res = {'tot_sum': 0, 'l':{}, 'o':{}, 'all':{}}
    for lin in clean_traffic(data):
        distances, lines, points = get_multiple_ed(lin["geom"], objs, lin["mean"], lin["std"], max_distance, width)
        mean_time = lin['ai']
        add_empty_segment_to_res_dict(res, lin)
        for i in range(len(objs)):
            o_name = f"{objs[i]['type']} - {objs[i]['id']}"
            powered_na_sum = 0
            for dist in distances[i]:
                if dist == 0:
                    continue
                powered = powered_na(dist, mean_time, float(lin['speed']) * 1852 / 3600)
                powered_na_sum += powered * lin['freq'] * data['pc']['p_pc'] / width
            write_res2dict(res, lin, o_name, powered_na_sum)
    return res

def transform_to_utm(lines, objects):
    """
    Transform lines and objects from WGS84 (EPSG:4326) to the appropriate UTM zone.

    Uses QGIS native coordinate transformation to avoid pyproj conflicts in QGIS.

    Parameters:
    - lines: List of LineString geometries in EPSG:4326.
    - objects: List of Polygon geometries in EPSG:4326.

    Returns:
    - transformed_lines: List of LineString geometries in UTM.
    - transformed_objects: List of Polygon geometries in UTM.
    - utm_epsg: The EPSG code of the UTM zone used.
    """
    # Combine all geometries to find the centroid
    all_geometries = lines + objects
    combined_centroid_x = sum([geom.centroid.x for geom in all_geometries]) / len(all_geometries)
    combined_centroid_y = sum([geom.centroid.y for geom in all_geometries]) / len(all_geometries)

    # Determine the UTM zone based on the centroid longitude
    # Northern hemisphere: EPSG 326XX, Southern hemisphere: EPSG 327XX
    utm_zone = int((combined_centroid_x + 180) // 6) + 1
    if combined_centroid_y >= 0:
        utm_epsg = 32600 + utm_zone  # Northern hemisphere
    else:
        utm_epsg = 32700 + utm_zone  # Southern hemisphere

    # Create QGIS CRS objects
    wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")

    # Create coordinate transform
    transform_context = QgsProject.instance().transformContext()
    coord_transform = QgsCoordinateTransform(wgs84_crs, utm_crs, transform_context)

    def transform_coords(x, y):
        """Transform a single coordinate pair from WGS84 to UTM."""
        from qgis.core import QgsPointXY
        point = coord_transform.transform(QgsPointXY(x, y))
        return point.x(), point.y()

    def transform_geometry(geom):
        """Transform a shapely geometry from WGS84 to UTM."""
        return transform(transform_coords, geom)

    # Transform lines
    transformed_lines = [transform_geometry(line) for line in lines]

    # Transform objects
    transformed_objects = [transform_geometry(obj) for obj in objects]

    return transformed_lines, transformed_objects, utm_epsg

def prepare_traffic_lists(data: dict[str, Any]) -> tuple[
    list[LineString], list[list[Any]], list[list[float]], list[str]
]:
    """
    Prepare lists of lines, distributions, weights, and line names from traffic data.
    """
    lines: list[LineString] = []
    distributions: list[list[Any]] = []
    weights: list[list[float]] = []
    line_names: list[str] = []
    for geom, distribution, weight, _, name in clean_traffic(data):
        lines.append(geom)
        distributions.append(distribution)
        weights.append(weight)
        line_names.append(name)
    return lines, distributions, weights, line_names

class Calculation:
    def __init__(self, parent: "OMRAT") -> None:
        self.p = parent
        self.canvas: QWidget | None = None
        self.drifting_report: dict[str, Any] | None = None
        self._progress_callback: Callable[[int, int, str], bool] | None = None
        # Store metadata for result layer generation
        self._last_structures: list[dict[str, Any]] = []
        self._last_depths: list[dict[str, Any]] = []
        self.allision_result_layer = None
        self.grounding_result_layer = None
        # Ship-ship collision attributes
        self.ship_collision_prob: float = 0.0
        self.collision_report: dict[str, Any] | None = None

    def set_progress_callback(self, callback: Callable[[int, int, str], bool]) -> None:
        """
        Set a callback function for progress updates.

        Args:
            callback: Function that takes (completed, total, message) and returns bool.
                     Should return False to cancel the operation, True to continue.
        """
        self._progress_callback = callback

    def _report_progress(self, phase: str, phase_progress: float, message: str) -> bool:
        """
        Report progress across multiple calculation phases.

        Phases and their weight in overall progress:
        - 'spatial': 0-60% (probability hole calculations - expensive)
        - 'cascade': 60-90% (traffic cascade - moderate)
        - 'layers': 90-100% (result layer creation - fast)

        Args:
            phase: One of 'spatial', 'cascade', 'layers'
            phase_progress: Progress within the phase (0.0 to 1.0)
            message: Status message to display

        Returns:
            True to continue, False to cancel
        """
        if not self._progress_callback:
            return True

        # Phase weights (must sum to 1.0)
        phase_weights = {
            'spatial': (0.0, 0.60),   # 0% to 60%
            'cascade': (0.60, 0.90),  # 60% to 90%
            'layers': (0.90, 1.0),    # 90% to 100%
        }

        start, end = phase_weights.get(phase, (0.0, 1.0))
        overall_progress = start + (end - start) * min(1.0, max(0.0, phase_progress))

        # Report as percentage (0-100)
        return self._progress_callback(
            int(overall_progress * 100),
            100,
            message
        )
        
    def get_no_ship_h(self, data:dict[str, Any]) -> list[float]:
        no_ships:list[float] = []
        td:dict[str, dict[str, np.ndarray]] = data['traffic_data']
        for leg, leg_dirs in td.items():
            leg_length = data['segment_data'][leg]['line_length']
            for v in leg_dirs.values():
                freq = np.array(v['Frequency (ships/year)'])
                speed = np.array(v['Speed (knots)'])
                h = leg_length / (speed * 1852 / 3600)
                no_ships.append(float(np.sum(freq*h)))
        return no_ships

    # --- Drifting model helpers ---
    def _compute_reach_distance(self, data: dict[str, Any], longest_length: float) -> float:
            reach_distance = longest_length * 10.0
            try:
                rep = data.get('drift', {}).get('repair', {})
                use_ln = rep.get('use_lognormal', False)
                if use_ln:
                    s = float(rep.get('std', 0.0))
                    loc = float(rep.get('loc', 0.0))
                    scale = float(rep.get('scale', 1.0))
                    t99_h = float(stats.lognorm(s, loc=loc, scale=scale).ppf(0.99))
                    drift_speed_kts = float(data.get('drift', {}).get('speed', 0.0))
                    drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s
                    if t99_h > 0 and drift_speed > 0:
                        reach_distance = drift_speed * 3600.0 * t99_h
                        reach_distance = min(reach_distance, longest_length * 10.0)
            except Exception:
                pass
            return reach_distance

    def _build_transformed(self, data: dict[str, Any]) -> tuple[
            list[LineString], list[list[Any]], list[list[float]], list[str],
            list[dict[str, Any]], list[dict[str, Any]],
            list[gpd.GeoDataFrame], list[gpd.GeoDataFrame],
            list[LineString]
        ]:
            lines, distributions, weights, line_names = prepare_traffic_lists(data)
            structures, depths = split_structures_and_depths(data)
            structure_geoms = [s['wkt'] for s in structures]
            depth_geoms = [d['wkt'] for d in depths]
            transformed_lines, transformed_objs_all, utm_epsg = transform_to_utm(lines, structure_geoms + depth_geoms)
            n_struct = len(structure_geoms)
            transformed_structs = transformed_objs_all[:n_struct]
            transformed_depths = transformed_objs_all[n_struct:]

            # Create reverse transform (UTM -> WGS84) for converting fixed geometries back
            # This ensures wkt_wgs84 has the same vertex order as wkt (UTM)
            wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
            utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")
            transform_context = QgsProject.instance().transformContext()
            reverse_transform = QgsCoordinateTransform(utm_crs, wgs84_crs, transform_context)

            def transform_utm_to_wgs84(geom):
                """Transform a shapely geometry from UTM back to WGS84."""
                from qgis.core import QgsPointXY
                def reverse_coords(x, y):
                    point = reverse_transform.transform(QgsPointXY(x, y))
                    return point.x(), point.y()
                return transform(reverse_coords, geom)

            # Fix invalid geometries and split any MultiPolygons that may arise from make_valid
            # Note: split_structures_and_depths already splits MultiPolygons, but make_valid
            # can sometimes create new MultiPolygons from invalid geometries
            fixed_structs = []
            fixed_structs_meta = []  # Track original structure metadata
            for i, g in enumerate(transformed_structs):
                try:
                    fixed = shp_make_valid(g) if shp_make_valid is not None else g.buffer(0)
                except Exception:
                    fixed = g

                # Split MultiPolygons into individual Polygons (safety for make_valid results)
                orig = structures[i] if i < len(structures) else {'id': f'struct_{i}', 'height': 0.0}
                if fixed.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(fixed.geoms):
                        fixed_structs.append(poly)
                        # Transform the UTM polygon back to WGS84 so segment indices match
                        poly_wgs84 = transform_utm_to_wgs84(poly)
                        fixed_structs_meta.append({
                            'id': f"{orig['id']}_{j}" if len(fixed.geoms) > 1 else orig['id'],
                            'height': orig['height'],
                            'wkt': poly,
                            'wkt_wgs84': poly_wgs84,  # Transformed back from UTM for consistent segment indices
                        })
                else:
                    fixed_structs.append(fixed)
                    # Transform the UTM geometry back to WGS84 so segment indices match
                    fixed_wgs84 = transform_utm_to_wgs84(fixed)
                    fixed_structs_meta.append({
                        'id': orig['id'],
                        'height': orig['height'],
                        'wkt': fixed,
                        'wkt_wgs84': fixed_wgs84,  # Transformed back from UTM for consistent segment indices
                    })

            fixed_depths = []
            fixed_depths_meta = []  # Track original depth metadata
            for i, g in enumerate(transformed_depths):
                try:
                    fixed = shp_make_valid(g) if shp_make_valid is not None else g.buffer(0)
                except Exception:
                    fixed = g

                # Get the depth value for this geometry
                depth_val = depths[i]['depth'] if i < len(depths) else 0.0
                depth_id = depths[i]['id'] if i < len(depths) else f'depth_{i}'

                # Split MultiPolygons into individual Polygons (safety for make_valid results)
                if fixed.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(fixed.geoms):
                        fixed_depths.append(poly)
                        # Transform the UTM polygon back to WGS84 so segment indices match
                        poly_wgs84 = transform_utm_to_wgs84(poly)
                        fixed_depths_meta.append({
                            'id': f"{depth_id}_{j}" if len(fixed.geoms) > 1 else depth_id,
                            'depth': depth_val,
                            'wkt': poly,
                            'wkt_wgs84': poly_wgs84,  # Transformed back from UTM for consistent segment indices
                        })
                else:
                    fixed_depths.append(fixed)
                    # Transform the UTM geometry back to WGS84 so segment indices match
                    fixed_wgs84 = transform_utm_to_wgs84(fixed)
                    fixed_depths_meta.append({
                        'id': depth_id,
                        'depth': depth_val,
                        'wkt': fixed,
                        'wkt_wgs84': fixed_wgs84,  # Transformed back from UTM for consistent segment indices
                    })

            structs_gdfs = [gpd.GeoDataFrame(geometry=[g]) for g in fixed_structs]
            # Include depth values in the GeoDataFrame
            depths_gdfs = [gpd.GeoDataFrame({'depth': [fixed_depths_meta[i]['depth']], 'geometry': [g]})
                          for i, g in enumerate(fixed_depths)]
            return (
                lines, distributions, weights, line_names,
                fixed_structs_meta, fixed_depths_meta,
                structs_gdfs, depths_gdfs,
                transformed_lines,
            )

    def _precompute_spatial(self,
            transformed_lines: list[LineString],
            distributions: list[list[Any]],
            weights: list[list[float]],
            structs_gdfs: list[gpd.GeoDataFrame],
            depths_gdfs: list[gpd.GeoDataFrame],
            reach_distance: float,
            data: dict[str, Any] | None = None,
        ) -> tuple[list, list, list, list, list, list, list, list, list]:
            struct_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            depth_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            struct_overlap_fracs_dir = compute_dir_overlap_fraction_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            struct_overlap_fracs_dir_leg = compute_dir_leg_overlap_fraction_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            depth_overlap_fracs_dir = compute_dir_overlap_fraction_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            depth_overlap_fracs_dir_leg = compute_dir_leg_overlap_fraction_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            depth_overlap_fracs_leg = compute_leg_overlap_fraction(
                transformed_lines, distributions, weights, depths_gdfs
            ) if len(depths_gdfs) > 0 else []
            # Calculate probability holes using FAST Monte Carlo method
            # Unified progress tracking across structures AND depths
            # Count actual objects for progress estimation
            def count_objects(gdf_list):
                return sum(len(gdf) for gdf in gdf_list)

            struct_obj_count = count_objects(structs_gdfs) if len(structs_gdfs) > 0 else 0
            depth_obj_count = count_objects(depths_gdfs) if len(depths_gdfs) > 0 else 0

            # Estimate total work (8 directions × objects per leg)
            # Structures use dblquad (~slow), depths use fast method (~quick)
            # Weight: 1 structure ≈ 100 depth objects in terms of computation time
            weighted_struct = struct_obj_count * 100
            weighted_depth = depth_obj_count * 1
            total_weighted_work = max(1, weighted_struct + weighted_depth)

            # Track progress across BOTH calculations within the 'spatial' phase
            struct_done = False

            def spatial_progress_callback(completed: int, total: int, msg: str) -> bool:
                """Report progress within the spatial phase (0-60% of overall)"""
                # Calculate weighted progress within spatial phase
                if not struct_done:
                    # Currently calculating structures (first half of spatial)
                    weighted_progress = (completed / max(total, 1)) * weighted_struct
                    label = f"Drifting - structure probabilities ({completed}/{total})"
                else:
                    # Currently calculating depths (second half of spatial)
                    weighted_progress = weighted_struct + (completed / max(total, 1)) * weighted_depth
                    label = f"Drifting - depth probabilities ({completed}/{total})"

                # Convert to fraction of spatial phase (0.0 to 1.0)
                phase_progress = weighted_progress / total_weighted_work
                return self._report_progress('spatial', phase_progress, label)

            # Calculate structures using accurate dblquad integration (allision)
            # This computes the true geometric probability that a drifting ship
            # hits the obstacle, integrating the lateral PDF along the leg.
            # Distance-dependent repair probability is handled separately in the
            # cascade via get_not_repaired().
            struct_probability_holes = compute_probability_holes(
                transformed_lines, distributions, weights, structs_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(structs_gdfs) > 0 else []

            struct_done = True  # Switch to depths

            # Calculate depths using accurate dblquad integration (grounding)
            # NOTE: No draught filtering here - the cascade calculation
            # filters by draught per vessel category
            depth_probability_holes = compute_probability_holes(
                transformed_lines, distributions, weights, depths_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(depths_gdfs) > 0 else []
            return (
                struct_min_dists, depth_min_dists,
                struct_overlap_fracs_dir, depth_overlap_fracs_dir,
                depth_overlap_fracs_leg,
                depth_overlap_fracs_dir_leg,
                struct_overlap_fracs_dir_leg,
                struct_probability_holes,
                depth_probability_holes,
            )

    def _find_nearest_targets(self,
            leg_idx: int,
            d_idx: int,
            height: float,
            draught: float,
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            anchor_d: float,
        ) -> tuple[float | None, int | None, float | None, int | None, float | None]:
        """Find nearest allision target, grounding target, and anchor depth.

        Returns:
            (allision_dist, allision_idx, grounding_dist, grounding_idx, anchor_dist)
        """
        # Convert compass d_idx to math index for array lookups
        # The min_dists arrays use math convention (index 0 = East, index 2 = North)
        math_dir_idx = _compass_idx_to_math_idx(d_idx)

        # Find nearest allision candidate (structure lower than ship height)
        allision_dist, allision_idx = None, None
        if struct_min_dists:
            for s_idx, s in enumerate(structures):
                if s['height'] < height:
                    md = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                    if md is not None and (allision_dist is None or md < allision_dist):
                        allision_dist, allision_idx = md, s_idx

        # Find nearest grounding candidate (depth shallower than draught)
        grounding_dist, grounding_idx = None, None
        if depth_min_dists:
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < draught:
                    md = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                    if md is not None and (grounding_dist is None or md < grounding_dist):
                        grounding_dist, grounding_idx = md, dep_idx

        # Find nearest anchor candidate
        anchor_dist = None
        if depth_min_dists and anchor_d > 0.0:
            thr = anchor_d * draught
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < thr:
                    md = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                    if md is not None and (anchor_dist is None or md < anchor_dist):
                        anchor_dist = md

        return allision_dist, allision_idx, grounding_dist, grounding_idx, anchor_dist

    def _compute_overlap_fractions(self,
            leg_idx: int,
            d_idx: int,
            allision_idx: int | None,
            grounding_idx: int | None,
            allision_dist: float | None,
            grounding_dist: float | None,
            struct_overlap_fracs_dir: list,
            struct_overlap_fracs_dir_leg: list,
            depth_overlap_fracs_leg: list,
            depth_overlap_fracs_dir_leg: list,
            depth_overlap_fracs_dir: list,
            struct_probability_holes: list,
        ) -> tuple[float, float, bool]:
        """Compute overlap fractions for allision and grounding.

        Returns:
            (ov_all, ov_gro, gro_has_true_overlap)
        """
        # Convert compass d_idx to math index for array lookups
        math_dir_idx = _compass_idx_to_math_idx(d_idx)

        # Allision overlap - use probability hole as primary metric
        ov_all = 0.0
        if allision_idx is not None and allision_dist is not None:
            # Primary: use probability hole (integrated probability mass)
            try:
                if struct_probability_holes:
                    ov_all = struct_probability_holes[leg_idx][math_dir_idx][allision_idx]
            except Exception:
                pass

            # Fallback: use traditional overlap metrics if hole calculation failed
            if ov_all <= 0.0 and struct_overlap_fracs_dir:
                ov_all = struct_overlap_fracs_dir[leg_idx][d_idx][allision_idx]
                # Use leg-based directional overlap if larger
                try:
                    if struct_overlap_fracs_dir_leg:
                        ov_all = max(ov_all, struct_overlap_fracs_dir_leg[leg_idx][d_idx][allision_idx])
                except Exception:
                    pass

        # Grounding overlap
        ov_gro = 0.0
        gro_has_true_overlap = False
        if grounding_idx is not None and grounding_dist is not None and depth_overlap_fracs_leg:
            # Primary: fraction of original leg overlapping shallow depth
            ov_gro = depth_overlap_fracs_leg[leg_idx][grounding_idx]
            if ov_gro > 0.0:
                gro_has_true_overlap = True
            # Fallback: if leg overlap is zero, use directional corridor measured along the leg
            if ov_gro <= 0.0 and depth_overlap_fracs_dir_leg:
                try:
                    ov_gro = depth_overlap_fracs_dir_leg[leg_idx][d_idx][grounding_idx]
                    if ov_gro > 0.0:
                        gro_has_true_overlap = True
                except Exception:
                    pass
            # Secondary fallback: use directional centre-line overlap
            if ov_gro <= 0.0 and depth_overlap_fracs_dir:
                try:
                    ov_gro = depth_overlap_fracs_dir[leg_idx][d_idx][grounding_idx]
                except Exception:
                    pass

        return ov_all, ov_gro, gro_has_true_overlap

    def _select_event_type(self,
            allision_dist: float | None,
            grounding_dist: float | None,
            allision_idx: int | None,
            grounding_idx: int | None,
            ov_all: float,
            ov_gro: float,
            gro_has_true_overlap: bool,
        ) -> tuple[str, float | None, int | None]:
        """Determine whether this is an allision or grounding event.

        Returns:
            (event_type, distance, target_idx)
        """
        # Initial choice based on distance
        choose_allision = False
        if allision_dist is not None and grounding_dist is not None:
            choose_allision = allision_dist <= grounding_dist
        elif allision_dist is not None:
            choose_allision = True
        else:
            choose_allision = False

        event = 'allision' if choose_allision else 'grounding'
        dist = allision_dist if choose_allision else grounding_dist
        idx = allision_idx if choose_allision else grounding_idx

        # If chosen event has zero overlap but the other has overlap, switch
        # Only switch to grounding if grounding has a real leg-based overlap
        if event == 'allision' and ov_all <= 0.0 and gro_has_true_overlap and ov_gro > 0.0:
            event = 'grounding'
            dist = grounding_dist
            idx = grounding_idx
        elif event == 'grounding' and ov_gro <= 0.0 and ov_all > 0.0:
            event = 'allision'
            dist = allision_dist
            idx = allision_idx

        return event, dist, idx

    def _update_report(self,
            report: dict[str, Any],
            event: str,
            contrib: float,
            idx: int,
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            seg_id: str,
            cell: dict[str, float],
            d_idx: int,
            dist: float,
            base: float,
            rp: float,
            anchor_factor: float,
            p_nr: float,
            ov_frac: float,
            freq: float,
            ship_type: int,
            ship_size: int,
            drift_corridor: Polygon | None = None,
            leg: LineString | None = None,
        ) -> None:
        """Update report dictionaries with contribution.

        Now also tracks per-segment contributions when drift_corridor is provided.
        """
        # Per-object accumulation
        try:
            if event == 'allision' and idx is not None:
                o = structures[idx]
                okey = f"Structure - {o.get('id', str(idx))}"
                ob = report['by_object'].setdefault(okey, {'allision': 0.0, 'grounding': 0.0})
                ob['allision'] += contrib
            elif event == 'grounding' and idx is not None:
                o = depths[idx]
                okey = f"Depth - {o.get('id', str(idx))}"
                ob = report['by_object'].setdefault(okey, {'allision': 0.0, 'grounding': 0.0})
                ob['grounding'] += contrib
        except Exception:
            pass

        # Per leg-direction accumulation
        leg_dir_label = str(cell.get('direction', '')).strip()
        leg_dir_key = f"{seg_id}:{leg_dir_label}:{d_idx*45}"
        rec = report['by_leg_direction'].setdefault(leg_dir_key, {
            'base_hours': 0.0,
            'contrib_allision': 0.0,
            'contrib_grounding': 0.0,
            'ship_categories': {},
            'min_distance_allision': None,
            'min_distance_grounding': None,
            'anchor_factor_sum': 0.0,
            'not_repaired_sum': 0.0,
            'overlap_sum': 0.0,
            'weight_sum': 0.0,
        })
        rec['base_hours'] += base * rp
        if event == 'allision':
            rec['contrib_allision'] += contrib
            md = rec['min_distance_allision']
            rec['min_distance_allision'] = dist if md is None or dist < md else md
        else:
            rec['contrib_grounding'] += contrib
            md = rec['min_distance_grounding']
            rec['min_distance_grounding'] = dist if md is None or dist < md else md

        # Weighted diagnostics
        w = base * rp
        rec['anchor_factor_sum'] += anchor_factor * w
        rec['not_repaired_sum'] += p_nr * w
        rec['overlap_sum'] += ov_frac * w
        rec['weight_sum'] += w

        # Ship category accumulation
        cat_key = f"{ship_type}-{ship_size}"
        scat = rec['ship_categories'].setdefault(cat_key, {'allision': 0.0, 'grounding': 0.0, 'freq': 0.0})
        scat[event] += contrib
        scat['freq'] += freq

        # Per-structure per leg-direction accumulation (allision only)
        try:
            if event == 'allision' and idx is not None:
                s = structures[idx]
                skey = f"Structure - {s.get('id', str(idx))}"
                s_map = report['by_structure_legdir'].setdefault(skey, {})
                s_map[leg_dir_key] = s_map.get(leg_dir_key, 0.0) + contrib

                # Per-segment tracking: determine which segments of this structure
                # actually intersect with the drift corridor
                if drift_corridor is not None:
                    obs_geom = s.get('wkt')
                    if obs_geom is not None:
                        # Convert compass angle (d_idx*45) to math convention
                        compass_angle = d_idx * 45
                        math_drift_angle = (90 - compass_angle) % 360
                        self._update_segment_contributions(
                            report, 'by_structure_segment_legdir',
                            skey, leg_dir_key, contrib, obs_geom, drift_corridor,
                            math_drift_angle, leg
                        )
        except Exception:
            pass

        # Per-depth per leg-direction accumulation (grounding only)
        try:
            if event == 'grounding' and idx is not None:
                d = depths[idx]
                dkey = f"Depth - {d.get('id', str(idx))}"
                d_map = report['by_depth_legdir'].setdefault(dkey, {})
                d_map[leg_dir_key] = d_map.get(leg_dir_key, 0.0) + contrib

                # Per-segment tracking for depths
                if drift_corridor is not None:
                    obs_geom = d.get('wkt')
                    if obs_geom is not None:
                        # Convert compass angle (d_idx*45) to math convention
                        compass_angle = d_idx * 45
                        math_drift_angle = (90 - compass_angle) % 360
                        self._update_segment_contributions(
                            report, 'by_depth_segment_legdir',
                            dkey, leg_dir_key, contrib, obs_geom, drift_corridor,
                            math_drift_angle, leg
                        )
        except Exception:
            pass

    def _update_segment_contributions(
        self,
        report: dict[str, Any],
        report_key: str,
        obstacle_key: str,
        leg_dir_key: str,
        contrib: float,
        obs_geom: BaseGeometry,
        drift_corridor: Polygon,
        drift_angle: float | None = None,
        leg: LineString | None = None,
    ) -> None:
        """
        Track which segments of an obstacle are hit by a drift corridor.

        Distributes the contribution among segments that actually intersect
        with the drift corridor AND are in the drift direction from the leg.

        Args:
            report: The report dictionary to update
            report_key: 'by_structure_segment_legdir' or 'by_depth_segment_legdir'
            obstacle_key: Key for the obstacle (e.g., "Structure - id")
            leg_dir_key: Key for leg-direction (e.g., "1:North:0")
            contrib: Total contribution for this obstacle from this leg-direction
            obs_geom: Obstacle geometry (UTM)
            drift_corridor: Drift corridor polygon (UTM)
            drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
            leg: The traffic leg LineString for direction checking
        """
        try:
            # Extract obstacle segments
            segments = _extract_obstacle_segments(obs_geom)
            if not segments:
                return

            # Get leg centroid for direction checking
            leg_centroid = None
            if leg is not None:
                centroid = leg.centroid
                leg_centroid = (centroid.x, centroid.y)

            # Find which segments intersect with the corridor in the drift direction
            intersecting_indices: list[int] = []
            for seg_idx, segment in enumerate(segments):
                if _segment_intersects_corridor(segment, drift_corridor, drift_angle, leg_centroid):
                    intersecting_indices.append(seg_idx)

            if not intersecting_indices:
                return

            # Distribute contribution equally among intersecting segments
            contrib_per_segment = contrib / len(intersecting_indices)

            # Initialize data structure if needed
            obs_seg_map = report.setdefault(report_key, {}).setdefault(obstacle_key, {})

            # Store contribution for each intersecting segment
            for seg_idx in intersecting_indices:
                seg_key = f"seg_{seg_idx}"
                seg_data = obs_seg_map.setdefault(seg_key, {})
                seg_data[leg_dir_key] = seg_data.get(leg_dir_key, 0.0) + contrib_per_segment

        except Exception:
            pass

    def _update_anchoring_report(
        self,
        report: dict[str, Any],
        anchor_contrib: float,
        obs_idx: int,
        depths: list[dict[str, Any]],
        seg_id: str,
        d_idx: int,
        dist: float,
        hole_pct: float,
        drift_corridor: Polygon | None,
        leg: LineString,
    ) -> None:
        """
        Track anchoring contributions per depth and per segment.

        Anchoring is now tracked like grounding and allision - we record which
        segments of a depth obstacle would receive the anchoring shadow.

        Args:
            report: The report dictionary to update
            anchor_contrib: Anchoring contribution value
            obs_idx: Index of the depth obstacle
            depths: List of depth dictionaries
            seg_id: Segment (leg) ID
            d_idx: Direction index (compass convention: 0=N, 1=NE, ...)
            dist: Distance to obstacle
            hole_pct: Probability hole percentage
            drift_corridor: Drift corridor polygon (UTM)
            leg: The traffic leg LineString
        """
        try:
            # Direction names for reporting
            dir_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            dir_name = dir_names[d_idx % 8]
            compass_angle = d_idx * 45
            leg_dir_key = f"{seg_id}:{dir_name}:{compass_angle}"

            # Per-depth per leg-direction accumulation
            d = depths[obs_idx]
            dkey = f"Anchoring - {d.get('id', str(obs_idx))}"
            d_map = report['by_anchoring_legdir'].setdefault(dkey, {})
            d_map[leg_dir_key] = d_map.get(leg_dir_key, 0.0) + anchor_contrib

            # Per-segment tracking for anchoring
            if drift_corridor is not None:
                obs_geom = d.get('wkt')
                if obs_geom is not None:
                    # Convert compass angle to math convention
                    math_drift_angle = (90 - compass_angle) % 360
                    self._update_segment_contributions(
                        report, 'by_anchoring_segment_legdir',
                        dkey, leg_dir_key, anchor_contrib, obs_geom, drift_corridor,
                        math_drift_angle, leg
                    )
        except Exception:
            pass

    def _iterate_traffic_and_sum(self,
            data: dict[str, Any],
            line_names: list[str],
            transformed_lines: list[LineString],
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            struct_overlap_fracs_dir: list,
            depth_overlap_fracs_dir: list,
            depth_overlap_fracs_leg: list,
            depth_overlap_fracs_dir_leg: list,
            struct_overlap_fracs_dir_leg: list,
            struct_probability_holes: list,
            depth_probability_holes: list,
            distributions: list[list[Any]] | None = None,
            weights: list[list[float]] | None = None,
            reach_distance: float = 0.0,
        ) -> tuple[float, float, dict[str, Any]]:
        drift = data['drift']
        blackout_per_hour = float(drift.get('drift_p', 0.0)) / (365.0 * 24.0)
        anchor_p = float(drift.get('anchor_p', 0.0))
        anchor_d = float(drift.get('anchor_d', 0.0))
        drift_speed_kts = float(drift.get('speed', 0.0))
        drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s

        # Rose helper
        rose_vals = {int(k): float(v) for k, v in drift.get('rose', {}).items()}
        rose_total = sum(rose_vals.values())
        def rose_prob(idx: int) -> float:
            angle = idx * 45
            v = rose_vals.get(angle, 0.0)
            return (v / rose_total) if rose_total > 0 else 0.0

        # Compose traffic per leg
        traffic_by_leg: list[list[dict[str, float]]] = []
        for geom, _, _, leg_traffic, _ in clean_traffic(data):
            traffic_by_leg.append(leg_traffic)

        # Prepare report structure
        report: dict[str, Any] = {
            'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0},
            'by_leg_direction': {},
            'by_object': {},
            'by_structure_legdir': {},
            'by_depth_legdir': {},  # Per-depth per leg-direction contributions for grounding
            'by_anchoring_legdir': {},  # Per-depth per leg-direction contributions for anchoring
            'by_structure_segment_legdir': {},  # Per-segment per leg-direction for structures
            'by_depth_segment_legdir': {},  # Per-segment per leg-direction for depths
            'by_anchoring_segment_legdir': {},  # Per-segment per leg-direction for anchoring
        }

        total_allision = 0.0
        total_grounding = 0.0
        total_anchoring = 0.0

        # Count total cascade iterations for progress tracking
        # Each leg × each ship cell × 8 directions
        total_cascade_work = sum(
            len(traffic_by_leg[i]) * 8 if i < len(traffic_by_leg) else 0
            for i in range(len(transformed_lines))
        )
        cascade_progress = 0

        for leg_idx, line in enumerate(transformed_lines):
            # Segment id and length
            try:
                nm = line_names[leg_idx]
                seg_id = nm.split('Leg ')[1].split('-')[0].strip()
            except Exception:
                seg_id = str(leg_idx)
            line_length = float(data.get('segment_data', {}).get(seg_id, {}).get('line_length', line.length))

            ship_cells = traffic_by_leg[leg_idx] if leg_idx < len(traffic_by_leg) else []
            for cell in ship_cells:
                freq = float(cell.get('freq', 0.0))
                speed_kts = float(cell.get('speed', 0.0))
                draught = float(cell.get('draught', 0.0))
                height = float(cell.get('height', 0.0))
                ship_type = int(cell.get('ship_type', -1))
                ship_size = int(cell.get('ship_size', -1))
                if speed_kts <= 0.0 or freq <= 0.0:
                    continue
                hours_present = (line_length / (speed_kts * 1852.0)) * freq
                base = hours_present * blackout_per_hour

                for d_idx in range(8):
                    rp = rose_prob(d_idx)
                    if rp <= 0.0:
                        continue

                    # Create drift corridor for per-segment intersection checking
                    drift_corridor: Polygon | None = None
                    if distributions is not None and weights is not None and reach_distance > 0:
                        try:
                            # Calculate lateral spread from distributions
                            dists = distributions[leg_idx] if leg_idx < len(distributions) else []
                            wgts = weights[leg_idx] if leg_idx < len(weights) else []
                            if dists and wgts:
                                w = np.array(wgts)
                                if w.sum() > 0:
                                    w = w / w.sum()
                                    weighted_std = float(np.sqrt(sum(
                                        wt * (dist.std() ** 2) for dist, wt in zip(dists, w) if wt > 0
                                    )))
                                    lateral_spread = 5.0 * weighted_std  # 5 sigma range
                                    compass_angle = d_idx * 45  # Compass angle (0=N, 45=NE, 90=E, etc.)
                                    # Convert compass to math convention for _create_drift_corridor
                                    # Compass: 0=North (CW), Math: 0=East (CCW)
                                    math_angle = (90 - compass_angle) % 360
                                    drift_corridor = _create_drift_corridor(
                                        line, math_angle, reach_distance, lateral_spread
                                    )
                        except Exception:
                            drift_corridor = None

                    # Build list of all obstacles with their distances and holes
                    obstacles: list[tuple[str, int, float, float]] = []

                    # Convert compass d_idx to math index for array lookups
                    # The min_dists and probability_holes arrays use math convention
                    math_dir_idx = _compass_idx_to_math_idx(d_idx)

                    # Add all structures (allision targets)
                    if struct_min_dists and struct_probability_holes:
                        for s_idx, s in enumerate(structures):
                            if s['height'] < height:
                                try:
                                    dist = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                                    hole_pct = struct_probability_holes[leg_idx][math_dir_idx][s_idx]
                                    if dist is not None and hole_pct > 0.0:
                                        obstacles.append(('allision', s_idx, dist, hole_pct))
                                except (IndexError, TypeError):
                                    pass

                    # Add all depths (anchoring or grounding)
                    if depth_min_dists and depth_probability_holes:
                        anchor_threshold = anchor_d * draught if anchor_d > 0.0 else 0.0
                        for dep_idx, dep in enumerate(depths):
                            try:
                                dist = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                                hole_pct = depth_probability_holes[leg_idx][math_dir_idx][dep_idx]
                                if dist is None or hole_pct <= 0.0:
                                    continue

                                # Determine if this depth is for anchoring or grounding
                                if anchor_threshold > 0.0 and dep['depth'] < anchor_threshold:
                                    obstacles.append(('anchoring', dep_idx, dist, hole_pct))
                                if dep['depth'] < draught:
                                    obstacles.append(('grounding', dep_idx, dist, hole_pct))
                            except (IndexError, TypeError):
                                pass

                    if not obstacles:
                        continue

                    # Sort obstacles by distance (closest first)
                    obstacles.sort(key=lambda x: x[2])

                    # Process cascade: track remaining probability
                    remaining_prob = 1.0

                    for obs_type, obs_idx, dist, hole_pct in obstacles:
                        if remaining_prob <= 0.0:
                            break

                        if obs_type == 'anchoring':
                            # Anchoring: calculate the probability reduction and track per-segment
                            # The "anchor contribution" is the probability of successfully anchoring
                            # at this depth, which shadows obstacles behind it.
                            anchor_contrib = base * rp * remaining_prob * anchor_p * hole_pct
                            total_anchoring += anchor_contrib

                            # Update report with per-segment anchoring tracking
                            self._update_anchoring_report(
                                report, anchor_contrib, obs_idx, depths, seg_id,
                                d_idx, dist, hole_pct, drift_corridor, line
                            )

                            # Anchoring reduces remaining probability
                            remaining_prob *= (1.0 - anchor_p * hole_pct)

                        elif obs_type == 'allision':
                            # Allision: calculate contribution from this structure
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_allision += contrib

                            # Update report with drift corridor for per-segment tracking
                            self._update_report(
                                report, 'allision', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size, drift_corridor, line
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

                        elif obs_type == 'grounding':
                            # Grounding: calculate contribution from this depth
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_grounding += contrib

                            # Update report with drift corridor for per-segment tracking
                            self._update_report(
                                report, 'grounding', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size, drift_corridor, line
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

                    # Update cascade progress after each direction
                    cascade_progress += 1
                    if total_cascade_work > 0 and cascade_progress % max(1, total_cascade_work // 20) == 0:
                        phase_progress = cascade_progress / total_cascade_work
                        if not self._report_progress(
                            'cascade', phase_progress,
                            f"Drifting - traffic cascade (leg {leg_idx + 1}/{len(transformed_lines)})"
                        ):
                            # Cancelled - return early with partial results
                            report['totals']['allision'] = total_allision
                            report['totals']['grounding'] = total_grounding
                            report['totals']['anchoring'] = total_anchoring
                            return total_allision, total_grounding, report

        report['totals']['allision'] = total_allision
        report['totals']['grounding'] = total_grounding
        report['totals']['anchoring'] = total_anchoring
        return total_allision, total_grounding, report

    def _auto_generate_drifting_report(self, data: dict[str, Any]) -> str | None:
        """Auto-generate the drifting Markdown report to disk.

        Path resolution priority:
        - If the UI field LEReportPath has a value, write to that path
        - Otherwise, write to '<cwd>/drifting_report.md'

        Returns the written content on success, else None.
        """
        try:
            # Prefer UI-provided path if present
            ui_path = None
            try:
                if hasattr(self.p.main_widget, 'LEReportPath') and self.p.main_widget.LEReportPath is not None:
                    t = self.p.main_widget.LEReportPath.text()
                    if isinstance(t, str) and t.strip():
                        ui_path = t.strip()
            except Exception:
                ui_path = None

            path = ui_path or str(Path(os.getcwd()) / 'drifting_report.md')
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return self.write_drifting_report_markdown(path, data)
        except Exception:
            # Silent failure: do not interrupt calculations/UI/tests
            return None

    def run_drifting_model(self, data: dict[str, Any]) -> tuple[float, float]:
        """Compute drifting allision and grounding, and store a breakdown report."""
        if not data.get('traffic_data') or not data.get('segment_data'):
            self.p.main_widget.LEPDriftAllision.setText(f"{float(0):.3e}")
            try:
                self.p.main_widget.LEPDriftingGrounding.setText(f"{float(0):.3e}")
            except Exception:
                pass
            self.drifting_allision_prob = 0.0
            self.drifting_grounding_prob = 0.0
            return 0.0, 0.0

        # Build transformed inputs once
        (
            lines, distributions, weights, line_names,
            structures, depths,
            structs_gdfs, depths_gdfs,
            transformed_lines,
        ) = self._build_transformed(data)

        if len(structs_gdfs) == 0 and len(depths_gdfs) == 0:
            self.p.main_widget.LEPDriftAllision.setText(f"{float(0):.3e}")
            try:
                self.p.main_widget.LEPDriftingGrounding.setText(f"{float(0):.3e}")
            except Exception:
                pass
            self.drifting_allision_prob = 0.0
            self.drifting_grounding_prob = 0.0
            return 0.0, 0.0

        longest_length = max(line.length for line in transformed_lines) if transformed_lines else 0.0
        reach_distance = self._compute_reach_distance(data, longest_length)
        drift = data.get('drift', {})
        (
            struct_min_dists, depth_min_dists,
            struct_overlap_fracs_dir, depth_overlap_fracs_dir,
            depth_overlap_fracs_leg,
            depth_overlap_fracs_dir_leg,
            struct_overlap_fracs_dir_leg,
            struct_probability_holes,
            depth_probability_holes,
        ) = self._precompute_spatial(
            transformed_lines, distributions, weights,
            structs_gdfs, depths_gdfs, reach_distance, data
        )

        total_allision, total_grounding, report = self._iterate_traffic_and_sum(
            data, line_names, transformed_lines, structures, depths,
            struct_min_dists, depth_min_dists,
            struct_overlap_fracs_dir, depth_overlap_fracs_dir, depth_overlap_fracs_leg,
            depth_overlap_fracs_dir_leg, struct_overlap_fracs_dir_leg,
            struct_probability_holes, depth_probability_holes,
            distributions, weights, reach_distance
        )

        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))

        self.drifting_allision_prob = float(total_allision * allision_rf)
        self.drifting_grounding_prob = float(total_grounding * grounding_rf)
        self.drifting_report = report

        # Store structures and depths for result layer generation
        self._last_structures = structures
        self._last_depths = depths

        self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")
        except Exception:
            pass
        # Auto-generate Markdown report to disk (best-effort, non-blocking)
        self._report_progress('layers', 0.0, "Drifting - generating report...")
        self._auto_generate_drifting_report(data)

        # Create result layers showing where allisions/groundings occurred
        self._report_progress('layers', 0.3, "Drifting - creating result layers...")
        try:
            self.allision_result_layer, self.grounding_result_layer = create_result_layers(
                report, structures, depths, add_to_project=True
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create result layers: {e}")

        self._report_progress('layers', 1.0, "Drifting model complete")
        return self.drifting_allision_prob, self.drifting_grounding_prob

    def run_ship_collision_model(self, data: dict[str, Any]) -> dict[str, float]:
        """
        Run ship-ship collision calculations.

        Calculates head-on, overtaking, crossing, and bend collision frequencies
        based on the traffic data and leg geometries.

        Args:
            data: Dictionary containing traffic_data, segment_data, pc (causation factors),
                  and ship_categories

        Returns:
            dict with keys: 'head_on', 'overtaking', 'crossing', 'bend', 'total'
        """
        result: dict[str, float] = {
            'head_on': 0.0,
            'overtaking': 0.0,
            'crossing': 0.0,
            'bend': 0.0,
            'total': 0.0,
        }

        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data:
            self.ship_collision_prob = 0.0
            self.collision_report = {'totals': result, 'by_leg': {}}
            return result

        # Get causation factors
        pc_headon = float(pc_vals.get('headon', 4.9e-5))
        pc_overtaking = float(pc_vals.get('overtaking', 1.1e-4))
        pc_crossing = float(pc_vals.get('crossing', 1.3e-4))
        pc_bend = float(pc_vals.get('bend', 1.3e-4))

        # Get ship categories for LOA estimates
        ship_categories = data.get('ship_categories', {})
        length_intervals = ship_categories.get('length_intervals', [])

        # Helper to estimate ship dimensions from LOA category index
        def get_loa_midpoint(loa_idx: int) -> float:
            """Get midpoint of LOA category for length estimates."""
            if loa_idx < len(length_intervals):
                interval = length_intervals[loa_idx]
                try:
                    min_val = float(interval.get('min', 50))
                    max_val = float(interval.get('max', 100))
                    return (min_val + max_val) / 2.0
                except (ValueError, TypeError):
                    pass
            # Default midpoints for typical LOA categories
            default_midpoints = [25.0, 75.0, 150.0, 250.0, 350.0]
            return default_midpoints[loa_idx] if loa_idx < len(default_midpoints) else 150.0

        def estimate_beam(loa: float) -> float:
            """Estimate beam from LOA using typical ship ratios (L/B ~ 6-7)."""
            return loa / 6.5

        # Helper: extract weighted mu and sigma from the lateral traffic distributions
        def _get_weighted_mu_sigma(seg_info: dict[str, Any], direction: int) -> tuple[float, float]:
            """Extract weighted mean and std from segment lateral distributions.

            Returns (mu, sigma) in meters.  Raises ValueError when
            segment_data has no distribution information.
            """
            dists, wgts = get_distribution(seg_info, direction)

            w = np.array(wgts, dtype=float)
            w_sum = w.sum()
            if w_sum <= 0:
                raise ValueError(
                    f"No lateral distribution weights found for direction {direction} "
                    f"in segment data (keys: {list(seg_info.keys())})"
                )
            w = w / w_sum

            # Weighted mean: E[X] = Σ w_i * mu_i
            weighted_mu = float(sum(
                wi * dist.mean() for dist, wi in zip(dists, w) if wi > 0
            ))
            # Total variance: Var[X] = Σ w_i*(sigma_i² + mu_i²) - E[X]²
            weighted_var = float(sum(
                wi * (dist.var() + dist.mean() ** 2)
                for dist, wi in zip(dists, w) if wi > 0
            )) - weighted_mu ** 2
            weighted_sigma = float(np.sqrt(max(weighted_var, 0.0)))

            if weighted_sigma < 1.0:
                raise ValueError(
                    f"Lateral distribution sigma too small ({weighted_sigma:.4f} m) "
                    f"for direction {direction} – check distribution data"
                )

            return weighted_mu, weighted_sigma

        # Report structures
        by_leg: dict[str, dict[str, float]] = {}
        total_head_on = 0.0
        total_overtaking = 0.0
        total_crossing = 0.0
        total_bend = 0.0

        leg_keys = list(traffic_data.keys())
        total_legs = len(leg_keys)
        processed = 0

        self._report_progress('spatial', 0.0, "Starting ship collision calculations...")

        # Iterate through legs for head-on and overtaking (same-leg collisions)
        for leg_key in leg_keys:
            leg_dirs = traffic_data.get(leg_key, {})
            seg_info = segment_data.get(leg_key, {})
            leg_length_m = float(seg_info.get('line_length', 1000.0))

            leg_head_on = 0.0
            leg_overtaking = 0.0
            leg_bend = 0.0

            # Get directions for this leg
            dir_keys = list(leg_dirs.keys())

            # Process each direction pair for head-on collisions
            # Head-on: ships in opposite directions
            if len(dir_keys) >= 2:
                dir1, dir2 = dir_keys[0], dir_keys[1]
                data1 = leg_dirs.get(dir1, {})
                data2 = leg_dirs.get(dir2, {})

                freq1 = np.array(data1.get('Frequency (ships/year)', []))
                freq2 = np.array(data2.get('Frequency (ships/year)', []))
                speed1 = np.array(data1.get('Speed (knots)', []))
                speed2 = np.array(data2.get('Speed (knots)', []))
                beam1 = np.array(data1.get('Ship Beam (meters)', []))
                beam2 = np.array(data2.get('Ship Beam (meters)', []))

                # Get lateral distribution parameters from loaded data
                mu1_lat, sigma1_lat = _get_weighted_mu_sigma(seg_info, 0)
                mu2_lat, sigma2_lat = _get_weighted_mu_sigma(seg_info, 1)

                # Iterate ship categories (LOA x Type)
                for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
                    for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                        q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                        if q1 <= 0 or not np.isfinite(q1):
                            continue

                        # Get speed for dir1 ships
                        v1_kts = 10.0  # Default
                        if loa_i < len(speed1) and type_j < len(speed1[loa_i]):
                            s_list = speed1[loa_i][type_j]
                            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                v1_kts = float(np.mean(s_list))
                            elif isinstance(s_list, (int, float)):
                                v1_kts = float(s_list)
                        v1_ms = v1_kts * 1852.0 / 3600.0  # Convert knots to m/s

                        # Get beam for dir1 ships
                        b1 = estimate_beam(get_loa_midpoint(loa_i))
                        if loa_i < len(beam1) and type_j < len(beam1[loa_i]):
                            b_list = beam1[loa_i][type_j]
                            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                b1 = float(np.mean(b_list))
                            elif isinstance(b_list, (int, float)):
                                b1 = float(b_list)

                        # Iterate over dir2 ship categories
                        for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                            for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                                q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                                if q2 <= 0 or not np.isfinite(q2):
                                    continue

                                # Get speed for dir2 ships
                                v2_kts = 10.0
                                if loa_k < len(speed2) and type_l < len(speed2[loa_k]):
                                    s_list = speed2[loa_k][type_l]
                                    if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                        v2_kts = float(np.mean(s_list))
                                    elif isinstance(s_list, (int, float)):
                                        v2_kts = float(s_list)
                                v2_ms = v2_kts * 1852.0 / 3600.0

                                # Get beam for dir2 ships
                                b2 = estimate_beam(get_loa_midpoint(loa_k))
                                if loa_k < len(beam2) and type_l < len(beam2[loa_k]):
                                    b_list = beam2[loa_k][type_l]
                                    if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                        b2 = float(np.mean(b_list))
                                    elif isinstance(b_list, (int, float)):
                                        b2 = float(b_list)

                                # Calculate head-on collision candidates using loaded lateral distributions
                                n_g_headon = get_head_on_collision_candidates(
                                    Q1=q1, Q2=q2,
                                    V1=v1_ms, V2=v2_ms,
                                    mu1=mu1_lat, mu2=mu2_lat,
                                    sigma1=sigma1_lat, sigma2=sigma2_lat,
                                    B1=b1, B2=b2,
                                    L_w=leg_length_m
                                )
                                leg_head_on += n_g_headon * pc_headon

            # Process overtaking collisions (same direction, different speeds)
            for dir_idx, dir_key in enumerate(dir_keys):
                dir_data = leg_dirs.get(dir_key, {})
                freq = np.array(dir_data.get('Frequency (ships/year)', []))
                speed = np.array(dir_data.get('Speed (knots)', []))
                beam = np.array(dir_data.get('Ship Beam (meters)', []))

                # Get lateral distribution for this direction from loaded data
                mu_ot, sigma_ot = _get_weighted_mu_sigma(seg_info, dir_idx)

                # Collect all ship cells in this direction
                ship_cells: list[tuple[int, int, float, float, float]] = []  # (loa_i, type_j, freq, speed_ms, beam)
                for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
                    for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                        q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                        if q <= 0 or not np.isfinite(q):
                            continue

                        v_kts = 10.0
                        if loa_i < len(speed) and type_j < len(speed[loa_i]):
                            s_list = speed[loa_i][type_j]
                            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                v_kts = float(np.mean(s_list))
                            elif isinstance(s_list, (int, float)):
                                v_kts = float(s_list)
                        v_ms = v_kts * 1852.0 / 3600.0

                        b = estimate_beam(get_loa_midpoint(loa_i))
                        if loa_i < len(beam) and type_j < len(beam[loa_i]):
                            b_list = beam[loa_i][type_j]
                            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                b = float(np.mean(b_list))
                            elif isinstance(b_list, (int, float)):
                                b = float(b_list)

                        ship_cells.append((loa_i, type_j, q, v_ms, b))

                # Pairwise overtaking between all ship cells in same direction
                for i, (loa_i, type_i, q_fast, v_fast, b_fast) in enumerate(ship_cells):
                    for j, (loa_j, type_j, q_slow, v_slow, b_slow) in enumerate(ship_cells):
                        if i == j:
                            continue
                        if v_fast <= v_slow:
                            continue  # No overtaking if not faster

                        n_g_overtaking = get_overtaking_collision_candidates(
                            Q_fast=q_fast, Q_slow=q_slow,
                            V_fast=v_fast, V_slow=v_slow,
                            mu_fast=mu_ot, mu_slow=mu_ot,
                            sigma_fast=sigma_ot, sigma_slow=sigma_ot,
                            B_fast=b_fast, B_slow=b_slow,
                            L_w=leg_length_m
                        )
                        leg_overtaking += n_g_overtaking * pc_overtaking

            # Bend collisions (at waypoints between consecutive legs)
            # Simplified: use average ship dimensions and traffic for this leg
            avg_freq = 0.0
            avg_length = 150.0
            avg_beam = 25.0
            count = 0
            for dir_key in dir_keys:
                dir_data = leg_dirs.get(dir_key, {})
                freq = np.array(dir_data.get('Frequency (ships/year)', []))
                for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
                    for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                        q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                        if q > 0:
                            avg_freq += q
                            avg_length = (avg_length * count + get_loa_midpoint(loa_i)) / (count + 1)
                            avg_beam = (avg_beam * count + estimate_beam(get_loa_midpoint(loa_i))) / (count + 1)
                            count += 1

            # Bend collisions should only be calculated when there's an actual bend
            # at a waypoint between consecutive legs. Default to 0 (no bend).
            # Only calculate if segment_data explicitly specifies a bend_angle > 5 degrees.
            bend_angle_deg = float(seg_info.get('bend_angle', 0.0))
            bend_angle_rad = bend_angle_deg * np.pi / 180.0

            # Only calculate bend collision if there's a meaningful angle change (>5 degrees)
            if avg_freq > 0 and bend_angle_deg > 5.0:
                p_no_turn = 0.01  # Probability of failing to turn at bend
                n_g_bend = get_bend_collision_candidates(
                    Q=avg_freq,
                    P_no_turn=p_no_turn,
                    L=avg_length,
                    B=avg_beam,
                    theta=bend_angle_rad
                )
                leg_bend += n_g_bend * pc_bend

            # Store leg results
            by_leg[leg_key] = {
                'head_on': leg_head_on,
                'overtaking': leg_overtaking,
                'bend': leg_bend,
            }

            total_head_on += leg_head_on
            total_overtaking += leg_overtaking
            total_bend += leg_bend

            processed += 1
            self._report_progress(
                'spatial',
                processed / total_legs * 0.8,
                f"Processing leg {leg_key} ({processed}/{total_legs})..."
            )

        # Crossing collisions between different legs
        self._report_progress('cascade', 0.0, "Calculating crossing collisions...")
        crossing_pairs_processed = 0
        total_pairs = total_legs * (total_legs - 1) // 2

        def _parse_point(pt_str: str) -> tuple[float, float] | None:
            """Parse 'x y' coordinate string to (x, y) tuple."""
            if not pt_str:
                return None
            parts = str(pt_str).strip().split()
            if len(parts) >= 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except (ValueError, TypeError):
                    pass
            return None

        def _calc_bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
            """Calculate bearing in degrees (0=N, CW) from start to end (lon/lat)."""
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
            return bearing

        def _points_match(p1: tuple[float, float] | None, p2: tuple[float, float] | None,
                          tol: float = 1e-6) -> bool:
            """Check if two coordinate points are the same within tolerance."""
            if p1 is None or p2 is None:
                return False
            return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

        for i, leg1_key in enumerate(leg_keys):
            for j, leg2_key in enumerate(leg_keys):
                if j <= i:
                    continue  # Avoid double counting

                seg1 = segment_data.get(leg1_key, {})
                seg2 = segment_data.get(leg2_key, {})

                # Parse endpoints
                s1_start = _parse_point(seg1.get('Start_Point', ''))
                s1_end = _parse_point(seg1.get('End_Point', ''))
                s2_start = _parse_point(seg2.get('Start_Point', ''))
                s2_end = _parse_point(seg2.get('End_Point', ''))

                # Only compute crossing if the legs share a waypoint
                shares_waypoint = (
                    _points_match(s1_start, s2_start) or _points_match(s1_start, s2_end) or
                    _points_match(s1_end, s2_start) or _points_match(s1_end, s2_end)
                )
                if not shares_waypoint:
                    crossing_pairs_processed += 1
                    continue

                # Calculate bearing from Start_Point/End_Point if not in segment_data
                if 'bearing' in seg1 and seg1['bearing']:
                    bearing1 = float(seg1['bearing'])
                elif s1_start and s1_end:
                    bearing1 = _calc_bearing(s1_start, s1_end)
                else:
                    crossing_pairs_processed += 1
                    continue

                if 'bearing' in seg2 and seg2['bearing']:
                    bearing2 = float(seg2['bearing'])
                elif s2_start and s2_end:
                    bearing2 = _calc_bearing(s2_start, s2_end)
                else:
                    crossing_pairs_processed += 1
                    continue

                crossing_angle = abs(bearing1 - bearing2) % 180.0
                if crossing_angle > 90:
                    crossing_angle = 180 - crossing_angle
                crossing_angle_rad = crossing_angle * np.pi / 180.0

                if crossing_angle_rad < 0.1:  # Nearly parallel, not a crossing
                    crossing_pairs_processed += 1
                    continue

                # Get traffic from both legs
                leg1_dirs = traffic_data.get(leg1_key, {})
                leg2_dirs = traffic_data.get(leg2_key, {})

                for dir1_key in leg1_dirs:
                    dir1_data = leg1_dirs.get(dir1_key, {})
                    freq1 = np.array(dir1_data.get('Frequency (ships/year)', []))
                    speed1 = np.array(dir1_data.get('Speed (knots)', []))
                    beam1 = np.array(dir1_data.get('Ship Beam (meters)', []))

                    for dir2_key in leg2_dirs:
                        dir2_data = leg2_dirs.get(dir2_key, {})
                        freq2 = np.array(dir2_data.get('Frequency (ships/year)', []))
                        speed2 = np.array(dir2_data.get('Speed (knots)', []))
                        beam2 = np.array(dir2_data.get('Ship Beam (meters)', []))

                        # Iterate ship categories
                        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
                            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                                if q1 <= 0 or not np.isfinite(q1):
                                    continue

                                v1_kts = 10.0
                                if loa_i < len(speed1) and type_j < len(speed1[loa_i]):
                                    s_list = speed1[loa_i][type_j]
                                    if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                        v1_kts = float(np.mean(s_list))
                                    elif isinstance(s_list, (int, float)):
                                        v1_kts = float(s_list)
                                v1_ms = v1_kts * 1852.0 / 3600.0

                                l1 = get_loa_midpoint(loa_i)
                                b1 = estimate_beam(l1)
                                if loa_i < len(beam1) and type_j < len(beam1[loa_i]):
                                    b_list = beam1[loa_i][type_j]
                                    if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                        b1 = float(np.mean(b_list))
                                    elif isinstance(b_list, (int, float)):
                                        b1 = float(b_list)

                                for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                                    for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                                        q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                                        if q2 <= 0 or not np.isfinite(q2):
                                            continue

                                        v2_kts = 10.0
                                        if loa_k < len(speed2) and type_l < len(speed2[loa_k]):
                                            s_list = speed2[loa_k][type_l]
                                            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                                v2_kts = float(np.mean(s_list))
                                            elif isinstance(s_list, (int, float)):
                                                v2_kts = float(s_list)
                                        v2_ms = v2_kts * 1852.0 / 3600.0

                                        l2 = get_loa_midpoint(loa_k)
                                        b2 = estimate_beam(l2)
                                        if loa_k < len(beam2) and type_l < len(beam2[loa_k]):
                                            b_list = beam2[loa_k][type_l]
                                            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                                b2 = float(np.mean(b_list))
                                            elif isinstance(b_list, (int, float)):
                                                b2 = float(b_list)

                                        n_g_crossing = get_crossing_collision_candidates(
                                            Q1=q1, Q2=q2,
                                            V1=v1_ms, V2=v2_ms,
                                            L1=l1, L2=l2,
                                            B1=b1, B2=b2,
                                            theta=crossing_angle_rad
                                        )
                                        total_crossing += n_g_crossing * pc_crossing

                crossing_pairs_processed += 1
                if total_pairs > 0:
                    self._report_progress(
                        'cascade',
                        crossing_pairs_processed / total_pairs,
                        f"Processing crossing pair {leg1_key}-{leg2_key}..."
                    )

        # Compile results
        result['head_on'] = total_head_on
        result['overtaking'] = total_overtaking
        result['crossing'] = total_crossing
        result['bend'] = total_bend
        result['total'] = total_head_on + total_overtaking + total_crossing + total_bend

        self.ship_collision_prob = result['total']
        self.collision_report = {
            'totals': result,
            'by_leg': by_leg,
            'causation_factors': {
                'headon': pc_headon,
                'overtaking': pc_overtaking,
                'crossing': pc_crossing,
                'bend': pc_bend,
            },
        }

        self._report_progress('layers', 1.0, "Ship collision calculation complete")

        # Update UI with collision results
        try:
            self.p.main_widget.LEPHeadOnCollision.setText(f"{result['head_on']:.3e}")
            self.p.main_widget.LEPOvertakingCollision.setText(f"{result['overtaking']:.3e}")
            self.p.main_widget.LEPCrossingCollision.setText(f"{result['crossing']:.3e}")
            self.p.main_widget.LEPMergingCollision.setText(f"{result['bend']:.3e}")
        except Exception as e:
            pass  # UI update failed, but calculation succeeded

        return result

    def run_powered_grounding_model(self, data: dict[str, Any]) -> float:
        """Calculate powered grounding probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially running aground on shallow depth areas.

        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))

        Shadow effect: closer depth areas block the distribution for areas
        behind them.  Only depths shallower than ship draught count.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        depths_list = data.get('depths', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not depths_list:
            try:
                self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        pc_grounding = float(pc_vals.get('grounding', pc_vals.get('p_pc', 1.6e-4)))

        # Build projector
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
        except Exception:
            try:
                self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        # Collect all unique draughts from traffic data so we can compute
        # shadow-aware results once per draught bracket.
        draught_set: set[float] = set()
        for leg_key, leg_dirs in traffic_data.items():
            for dir_key, dir_data in leg_dirs.items():
                draught_array = dir_data.get('Draught (meters)', [])
                for row in draught_array:
                    if not hasattr(row, '__iter__'):
                        continue
                    for d_val in row:
                        try:
                            v = float(d_val) if d_val != '' else 0.0
                            if v > 0:
                                draught_set.add(v)
                        except (ValueError, TypeError):
                            pass
        if not draught_set:
            draught_set = {5.0}  # Default draught

        # For each unique draught, build obstacle list and run shadow computation
        # Cache: draught -> {(seg_id, dir_idx, obs_key) -> {mass, mean_dist}}
        draught_results: dict[float, list[dict]] = {}
        for max_draft in sorted(draught_set):
            try:
                legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                    data, proj, mode="grounding", max_draft=max_draft)
                if all_obstacles:
                    comps = _run_all_computations(legs, all_obstacles)
                    draught_results[max_draft] = comps
                else:
                    draught_results[max_draft] = []
            except Exception:
                draught_results[max_draft] = []

        # Sum per-ship-type contributions using pre-computed shadow results
        for leg_key, leg_dirs in traffic_data.items():
            seg_info = segment_data.get(leg_key, {})
            ai_per_dir = [
                float(seg_info.get('ai1', 180.0)),
                float(seg_info.get('ai2', 180.0)),
            ]

            for dir_idx, (dir_key, dir_data) in enumerate(leg_dirs.items()):
                ai_seconds = ai_per_dir[min(dir_idx, 1)]
                freq_array = dir_data.get('Frequency (ships/year)', [])
                draught_array = dir_data.get('Draught (meters)', [])
                speed_array = dir_data.get('Speed (knots)', [])

                for loa_i, freq_row in enumerate(freq_array):
                    if not hasattr(freq_row, '__iter__'):
                        continue
                    for type_j, freq_val in enumerate(freq_row):
                        try:
                            q = float(freq_val) if freq_val != '' else 0.0
                        except (ValueError, TypeError):
                            q = 0.0
                        if q <= 0:
                            continue

                        # Get ship draught
                        draught = 5.0
                        try:
                            if loa_i < len(draught_array) and type_j < len(draught_array[loa_i]):
                                d_val = draught_array[loa_i][type_j]
                                if isinstance(d_val, (int, float)) and d_val > 0:
                                    draught = float(d_val)
                                elif isinstance(d_val, str) and d_val != '':
                                    draught = float(d_val)
                        except Exception:
                            pass

                        # Get ship speed
                        speed_kts = 10.0
                        try:
                            if loa_i < len(speed_array) and type_j < len(speed_array[loa_i]):
                                s_val = speed_array[loa_i][type_j]
                                if isinstance(s_val, (int, float)) and s_val > 0:
                                    speed_kts = float(s_val)
                                elif isinstance(s_val, str) and s_val != '':
                                    speed_kts = float(s_val)
                        except Exception:
                            pass
                        speed_ms = speed_kts * 1852.0 / 3600.0

                        # Find the closest matching draught bracket
                        best_draft = min(draught_set,
                                         key=lambda d: abs(d - draught))
                        comps = draught_results.get(best_draft, [])

                        # Sum over matching leg/direction computations
                        for comp in comps:
                            if comp["seg_id"] != leg_key:
                                continue
                            if comp["dir_idx"] != dir_idx:
                                continue
                            for key, s in comp["summaries"].items():
                                mass = s["mass"]
                                d_mean = s["mean_dist"]
                                if mass <= 0 or d_mean <= 0:
                                    continue
                                recovery = ai_seconds * speed_ms
                                if recovery <= 0:
                                    continue
                                prob_not_rec = exp(-d_mean / recovery)
                                total += pc_grounding * q * mass * prob_not_rec

        try:
            self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
        except Exception:
            pass
        return total

    def run_powered_allision_model(self, data: dict[str, Any]) -> float:
        """Calculate powered allision probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially hitting structures (objects).

        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))

        Shadow effect: closer obstacles block the distribution for obstacles
        behind them.  Mass = fraction of lateral distribution intercepted.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        objects_list = data.get('objects', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not objects_list:
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        pc_allision = float(pc_vals.get('allision', 1.9e-4))

        # Build projector and geometry once
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
            legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                data, proj, mode="allision", max_draft=0)
        except Exception:
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        if not all_obstacles:
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        # Shadow-aware ray casting: compute mass & d_mean per obstacle
        # per leg/direction (geometry only, independent of ship type)
        computations = _run_all_computations(legs, all_obstacles)

        # For each computation (leg/dir with hits), sum per-ship-type
        for comp in computations:
            seg_id = comp["seg_id"]
            dir_idx = comp["dir_idx"]
            d_info = comp["dir_info"]
            ai_seconds = d_info["ai"]

            leg_dirs = traffic_data.get(seg_id, {})
            dir_keys = list(leg_dirs.keys())
            if dir_idx >= len(dir_keys):
                continue
            dir_data = leg_dirs[dir_keys[dir_idx]]

            freq_array = dir_data.get('Frequency (ships/year)', [])
            speed_array = dir_data.get('Speed (knots)', [])

            for loa_i, freq_row in enumerate(freq_array):
                if not hasattr(freq_row, '__iter__'):
                    continue
                for type_j, freq_val in enumerate(freq_row):
                    try:
                        q = float(freq_val) if freq_val != '' else 0.0
                    except (ValueError, TypeError):
                        q = 0.0
                    if q <= 0:
                        continue

                    speed_kts = 10.0
                    try:
                        if loa_i < len(speed_array) and type_j < len(speed_array[loa_i]):
                            s_val = speed_array[loa_i][type_j]
                            if isinstance(s_val, (int, float)) and s_val > 0:
                                speed_kts = float(s_val)
                            elif isinstance(s_val, str) and s_val != '':
                                speed_kts = float(s_val)
                    except Exception:
                        pass
                    speed_ms = speed_kts * 1852.0 / 3600.0

                    # Sum over obstacles hit by this leg/direction
                    for key, s in comp["summaries"].items():
                        mass = s["mass"]
                        d_mean = s["mean_dist"]
                        if mass <= 0 or d_mean <= 0:
                            continue
                        recovery = ai_seconds * speed_ms
                        if recovery <= 0:
                            continue
                        prob_not_rec = exp(-d_mean / recovery)
                        total += pc_allision * q * mass * prob_not_rec

        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:
            pass
        return total

    def get_drifting_report(self) -> dict[str, Any] | None:
        return self.drifting_report
    
    def generate_drifting_report_markdown(self, data: dict[str, Any] | None = None) -> str:
        """
        Build a human-readable Markdown appendix report from the last drifting run.

        Includes:
        - Parameter summary (drift, anchoring, repair)
        - Overall totals (allision, grounding)
        - Directional aggregates
        - Per leg-direction highlights
        - Ship category breakdown
        """
        rep = self.drifting_report or {}
        totals = rep.get('totals', {})
        bld: dict[str, Any] = rep.get('by_leg_direction', {})
        by_obj: dict[str, Any] = rep.get('by_object', {})
        by_struct_legdir: dict[str, Any] = rep.get('by_structure_legdir', {})

        # Parameter summary
        drift = {} if data is None else data.get('drift', {})
        repair = drift.get('repair', {}) if isinstance(drift, dict) else {}
        rose = drift.get('rose', {}) if isinstance(drift, dict) else {}

        # Aggregates
        total_base_hours = 0.0
        dir_agg: dict[str, dict[str, float]] = {}
        ship_cat_totals: dict[str, dict[str, float]] = {}
        leg_rows: list[tuple[str, float, float, float, float, float]] = []
        # seg:dir key => metrics
        for key, rec in bld.items():
            total_base_hours += float(rec.get('base_hours', 0.0))
            try:
                angle = key.split(':')[-1]
            except Exception:
                angle = '0'
            da = dir_agg.setdefault(angle, {
                'allision': 0.0,
                'grounding': 0.0,
                'base_hours': 0.0,
                'anchor_factor_sum': 0.0,
                'not_repaired_sum': 0.0,
                'overlap_sum': 0.0,
                'weight_sum': 0.0,
            })
            a = float(rec.get('contrib_allision', 0.0))
            g = float(rec.get('contrib_grounding', 0.0))
            da['allision'] += a
            da['grounding'] += g
            da['base_hours'] += float(rec.get('base_hours', 0.0))
            da['anchor_factor_sum'] += float(rec.get('anchor_factor_sum', 0.0))
            da['not_repaired_sum'] += float(rec.get('not_repaired_sum', 0.0))
            da['overlap_sum'] += float(rec.get('overlap_sum', 0.0))
            da['weight_sum'] += float(rec.get('weight_sum', 0.0))

            # Per leg-direction highlight row
            af = float(rec.get('anchor_factor_sum', 0.0))
            wf = float(rec.get('weight_sum', 0.0))
            nrs = float(rec.get('not_repaired_sum', 0.0))
            ovs = float(rec.get('overlap_sum', 0.0))
            avg_anchor = (af / wf) if wf > 0 else 0.0
            avg_not_rep = (nrs / wf) if wf > 0 else 0.0
            avg_overlap = (ovs / wf) if wf > 0 else 0.0
            # store: key, allision, grounding, avg_anchor, avg_not_rep, avg_overlap
            leg_rows.append((key, a, g, avg_anchor, avg_not_rep, avg_overlap))

            # Ship categories
            for cat, vals in rec.get('ship_categories', {}).items():
                sct = ship_cat_totals.setdefault(cat, {'allision': 0.0, 'grounding': 0.0, 'freq': 0.0})
                sct['allision'] += float(vals.get('allision', 0.0))
                sct['grounding'] += float(vals.get('grounding', 0.0))
                sct['freq'] += float(vals.get('freq', 0.0))

        # Sort per-leg-direction rows by segment then direction for readability
        def _parse_key(k: str) -> tuple[int, str, int]:
            try:
                parts = k.split(':')
                if len(parts) == 3:
                    seg, legdir, ang = parts
                    return int(str(seg)), str(legdir), int(str(ang))
                elif len(parts) == 2:
                    seg, ang = parts
                    return int(str(seg)), '', int(str(ang))
                else:
                    return 0, '', 0
            except Exception:
                return (0, '', 0)
        leg_rows_sorted = sorted(leg_rows, key=lambda r: _parse_key(r[0]))

        # Directional table rows
        def dir_row(angle: str, d: dict[str, float]) -> str:
            w = d.get('weight_sum', 0.0)
            avg_anchor = (d.get('anchor_factor_sum', 0.0) / w) if w > 0 else 0.0
            avg_not_rep = (d.get('not_repaired_sum', 0.0) / w) if w > 0 else 0.0
            avg_overlap = (d.get('overlap_sum', 0.0) / w) if w > 0 else 0.0
            return f"| {angle}° | {d.get('base_hours', 0.0):.2f} | {d.get('allision', 0.0):.3e} | {d.get('grounding', 0.0):.3e} | {avg_anchor:.3f} | {avg_not_rep:.3f} | {avg_overlap:.3f} |"

        md_lines: list[str] = []
        # Apply reduction factors to match GUI display
        pc_vals = data.get('pc', {}) if isinstance(data, dict) and isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))

        final_allision = float(totals.get('allision', 0.0)) * allision_rf
        final_grounding = float(totals.get('grounding', 0.0)) * grounding_rf

        md_lines.append("# Drifting Model Appendix Report")
        md_lines.append("")
        md_lines.append("## Summary")
        md_lines.append(f"- Total allision: {final_allision:.3e}")
        md_lines.append(f"- Total grounding: {final_grounding:.3e}")
        md_lines.append(f"- Total allision (before RF): {float(totals.get('allision', 0.0)):.3e}")
        md_lines.append(f"- Total grounding (before RF): {float(totals.get('grounding', 0.0)):.3e}")
        md_lines.append(f"- Allision reduction factor: {allision_rf:.3f}")
        md_lines.append(f"- Grounding reduction factor: {grounding_rf:.3f}")
        md_lines.append(f"- Aggregated ship-hours on legs used in model: {total_base_hours:.2f}")
        md_lines.append("")
        md_lines.append("## Parameters")
        drift_speed_kts = float(drift.get('speed', 0.0)) if isinstance(drift, dict) else 0.0
        md_lines.append(f"- Drift speed: {drift_speed_kts} knots ({drift_speed_kts * 1852.0 / 3600.0:.3f} m/s)")
        md_lines.append(f"- Blackout prob per ship-year: {float(drift.get('drift_p', 0.0)) if isinstance(drift, dict) else 0.0}")
        md_lines.append(f"- Anchor prob: {float(drift.get('anchor_p', 0.0)) if isinstance(drift, dict) else 0.0}, distance factor: {float(drift.get('anchor_d', 0.0)) if isinstance(drift, dict) else 0.0}")
        if isinstance(repair, dict) and repair.get('use_lognormal', False):
            md_lines.append(f"- Repair lognormal (std={float(repair.get('std', 0.0))}, loc={float(repair.get('loc', 0.0))}, scale={float(repair.get('scale', 1.0))})")
        else:
            md_lines.append("- Repair model: not specified/lognormal disabled")
        if isinstance(rose, dict) and rose:
            md_lines.append("- Wind rose: " + ", ".join([f"{k}°={v}" for k, v in rose.items()]))
        md_lines.append("")

        # Per-leg details: lat/lon and distributions
        try:
            seg_data = {} if data is None else (data.get('segment_data', {}) or {})
            if isinstance(seg_data, dict) and seg_data:
                md_lines.append("## Leg Details")
                for seg_id, sd in seg_data.items():
                    sp = str(sd.get('Start_Point', '')).strip()
                    ep = str(sd.get('End_Point', '')).strip()
                    def _fmt_point(pt: str) -> str:
                        try:
                            s = pt.strip().lstrip('(').rstrip(')')
                            parts = [p for p in s.replace(',', ' ').split() if p]
                            if len(parts) >= 2:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                return f"({lon:.6f}, {lat:.6f})"
                        except Exception:
                            pass
                        return pt
                    start_txt = _fmt_point(sp)
                    end_txt = _fmt_point(ep)
                    length = float(sd.get('line_length', 0.0))
                    md_lines.append(f"### Leg {seg_id}")
                    md_lines.append(f"- Start: {start_txt}")
                    md_lines.append(f"- End: {end_txt}")
                    md_lines.append(f"- Length (m): {length:.2f}")
                    # Directions and distributions
                    dirs = list(sd.get('Dirs', []) or [])
                    for d_idx in (1, 2):
                        label = str(dirs[d_idx-1]) if 0 <= (d_idx-1) < len(dirs) else str(d_idx)
                        # Gather normal components
                        comps: list[str] = []
                        for i in range(1, 4):
                            w = float(sd.get(f'weight{d_idx}_{i}', 0.0) or 0.0)
                            m = float(sd.get(f'mean{d_idx}_{i}', 0.0) or 0.0)
                            sdev = float(sd.get(f'std{d_idx}_{i}', 0.0) or 0.0)
                            if w > 0.0:
                                comps.append(f"w={w:.2f}: N({m:.2f}, {sdev:.2f})")
                        # Uniform component
                        up = float(sd.get(f'u_p{d_idx}', 0.0) or 0.0)
                        if up > 0.0:
                            umin = float(sd.get(f'u_min{d_idx}', 0.0) or 0.0)
                            umax = float(sd.get(f'u_max{d_idx}', 0.0) or 0.0)
                            comps.append(f"u_p={up:.2f}: U[{umin:.2f}, {umax:.2f}]")
                        comp_txt = ", ".join(comps) if comps else "(no active distributions)"
                        md_lines.append(f"- Dir {label}: {comp_txt}")
                    md_lines.append("")
                md_lines.append("")
        except Exception:
            # Do not block report generation if segment details are malformed
            pass

        md_lines.append("## Directional Aggregates")
        md_lines.append("| Direction | Base hours | Allision | Grounding | Avg anchor | Avg not-repaired | Avg overlap |")
        md_lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for ang in sorted(dir_agg.keys(), key=lambda x: int(x)):
            md_lines.append(dir_row(ang, dir_agg[ang]))
        md_lines.append("")

        md_lines.append("## Directional Aggregates per Leg-Direction")
        md_lines.append("| Leg:Dir:Angle | Allision | Grounding | Avg anchor | Avg not-repaired | Avg overlap |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for key, a, g, avg_anchor, avg_not_rep, avg_overlap in leg_rows_sorted:
            md_lines.append(f"| {key} | {a:.3e} | {g:.3e} | {avg_anchor:.3f} | {avg_not_rep:.3f} | {avg_overlap:.3f} |")
        md_lines.append("")

        # Per-structure, per leg-direction contributions (allision only)
        if by_struct_legdir:
            md_lines.append("## Per Structure: Directional Aggregates per Leg-Direction")
            for skey in sorted(by_struct_legdir.keys()):
                md_lines.append(f"### {skey}")
                md_lines.append("| Leg:Dir:Angle | Allision |")
                md_lines.append("|---|---:|")
                s_map = by_struct_legdir[skey] or {}
                for k in sorted(s_map.keys(), key=lambda x: _parse_key(x)):
                    md_lines.append(f"| {k} | {float(s_map[k]):.3e} |")
                md_lines.append("")

        # Per object totals
        if by_obj:
            md_lines.append("## Per Object")
            md_lines.append("| Object | Allision | Grounding |")
            md_lines.append("|---|---:|---:|")
            # Only include structures; omit depths as requested
            for okey in sorted([k for k in by_obj.keys() if str(k).startswith('Structure - ')]):
                ob = by_obj[okey]
                md_lines.append(f"| {okey} | {float(ob.get('allision', 0.0)):.3e} | {float(ob.get('grounding', 0.0)):.3e} |")
            md_lines.append("")

        # Ship Category Breakdown with names
        md_lines.append("## Ship Category Breakdown (by Leg:Dir:Angle)")
        md_lines.append("| Leg:Dir:Angle | Type-Size | Annual Frequency | Allision | Grounding |")
        md_lines.append("|---|---|---:|---:|---:|")
        # Build mapping from indices to labels if provided
        type_labels: list[str] = []
        size_labels: list[str] = []
        if isinstance(data, dict):
            try:
                sc = data.get('ship_categories', {})
                if isinstance(sc, dict):
                    type_labels = list(sc.get('types', []) or [])
                    size_labels = [str(x.get('label', '')) for x in (sc.get('length_intervals', []) or [])]
            except Exception:
                pass
        # Emit by walking leg-direction keys to include granularity
        for legdir_key in sorted(bld.keys(), key=lambda x: _parse_key(x)):
            rec = bld[legdir_key]
            cats = rec.get('ship_categories', {}) or {}
            for cat in sorted(cats.keys()):
                vals = cats[cat]
                disp = cat
                try:
                    s_type, s_size = cat.split('-')
                    ti = int(s_type)
                    si = int(s_size)
                    tname = type_labels[ti] if 0 <= ti < len(type_labels) else s_type
                    sname = size_labels[si] if 0 <= si < len(size_labels) else s_size
                    disp = f"{tname} - {sname}"
                except Exception:
                    pass
                md_lines.append(
                    f"| {legdir_key} | {disp} | {float(vals.get('freq', 0.0)):.2f} | {float(vals.get('allision', 0.0)):.3e} | {float(vals.get('grounding', 0.0)):.3e} |"
                )

        md_lines.append("")

        # Summary of legs
        md_lines.append("## Leg Summary")
        md_lines.append("| Leg | Allision | Grounding |")
        md_lines.append("|---:|---:|---:|")
        leg_sums: dict[str, dict[str, float]] = {}
        for key, rec in bld.items():
            try:
                leg = key.split(':')[0]
            except Exception:
                leg = key
            agg = leg_sums.setdefault(leg, {'allision': 0.0, 'grounding': 0.0})
            agg['allision'] += float(rec.get('contrib_allision', 0.0))
            agg['grounding'] += float(rec.get('contrib_grounding', 0.0))
        for leg in sorted(leg_sums.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
            ls = leg_sums[leg]
            md_lines.append(f"| {leg} | {ls['allision']:.3e} | {ls['grounding']:.3e} |")
        md_lines.append("")

        md_lines.append("")
        md_lines.append("_Generated by OMRA Tool drift model. Values are annual probabilities unless otherwise stated._")
        return "\n".join(md_lines)

    def write_drifting_report_markdown(self, file_path: str, data: dict[str, Any] | None = None) -> str:
        """Generate and write the drifting Markdown report to file_path. Returns the content."""
        content = self.generate_drifting_report_markdown(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return content
    
    def run_drift_visualization(self, data: dict[str, Any]) -> None:
        if not data.get('traffic_data'):
            return

        lines, distributions, weights, line_names = prepare_traffic_lists(data)

        # Prepare objects
        objects = [sw.loads(wkt) for _, _, wkt in data['objects']]
        transformed_lines, transformed_objects, utm_crs = transform_to_utm(lines, objects)
        # Fix invalid geometries for visualization and overlap computation
        fixed_objects = []
        for obj in transformed_objects:
            try:
                if shp_make_valid is not None:
                    fixed = shp_make_valid(obj)
                else:
                    fixed = obj.buffer(0)
            except Exception:
                fixed = obj
            if fixed is not None:
                fixed_objects.append(fixed)
        transformed_objects = fixed_objects
        longest_length = max(line.length for line in transformed_lines)
        transformed_objects_gdf = [gpd.GeoDataFrame(geometry=[obj]) for obj in transformed_objects]

        # Create and show dialog
        dialog = ShowGeomRes(self.p.main_widget)
        DriftingOverlapVisualizer.show_in_dialog(
            dialog,
            transformed_lines,
            line_names,
            transformed_objects_gdf,
            distributions,
            weights,
            data = data,
            distance=longest_length * 3.0
        )
        dialog.exec_()

    def run_powered_allision_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive Cat II powered allision visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware Cat II ray-casting plots showing how
        ships that miss a turn may hit objects (structures).
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        dialog = ShowGeomRes(self.p.main_widget)
        PoweredOverlapVisualizer.show_in_dialog(
            dialog, data, mode="allision", max_draft=max_draft,
        )
        dialog.exec_()

    def run_powered_grounding_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive Cat II powered grounding visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware Cat II ray-casting plots showing how
        ships that miss a turn may run aground on shallow depth areas.
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        dialog = ShowGeomRes(self.p.main_widget)
        PoweredOverlapVisualizer.show_in_dialog(
            dialog, data, mode="grounding", max_draft=max_draft,
        )
        dialog.exec_()


if __name__ == '__main__':
    # No CLI entry; this module is used via the plugin and tests.
    pass
