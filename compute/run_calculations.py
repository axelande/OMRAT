import json
import sys
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable

from qgis.PyQt.QtWidgets import QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
mpl.use('Qt5Agg')
import geopandas as gpd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import exp, log
import numpy as np
from pyproj import CRS, Transformer
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
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

sys.path.append('.')
from basic_equations import get_drifting_prob, get_Fcoll, powered_na, get_not_repaired
from geometries.route import get_multiple_ed
from geometries.route import get_multi_drift_distance, get_best_utm 
from geometries.get_drifting_overlap import (
    DriftingOverlapVisualizer,
    compute_min_distance_by_object,
    compute_leg_overlap_fraction,
    compute_dir_overlap_fraction_by_object,
    compute_dir_leg_overlap_fraction_by_object,
)
# Smart hybrid: fast for depths, accurate for structures
from geometries.smart_hybrid_probability_holes import compute_probability_holes_smart_hybrid
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

    Parameters:
    - lines: List of LineString geometries in EPSG:4326.
    - objects: List of Polygon geometries in EPSG:4326.

    Returns:
    - transformed_lines: List of LineString geometries in UTM.
    - transformed_objects: List of Polygon geometries in UTM.
    """
    # Combine all geometries to find the centroid
    all_geometries = lines + objects
    combined_centroid = sum([geom.centroid.x for geom in all_geometries]) / len(all_geometries), \
                        sum([geom.centroid.y for geom in all_geometries]) / len(all_geometries)

    # Determine the UTM zone based on the centroid
    utm_crs = CRS.from_epsg(32600 + int((combined_centroid[0] + 180) // 6) + 1)

    # Create a transformer from WGS84 to the determined UTM CRS
    transformer = Transformer.from_crs(CRS('EPSG:4326'), utm_crs, always_xy=True)

    # Transform lines
    transformed_lines = [transform(transformer.transform, line) for line in lines]

    # Transform objects
    transformed_objects = [transform(transformer.transform, obj) for obj in objects]

    return transformed_lines, transformed_objects, utm_crs

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

    def set_progress_callback(self, callback: Callable[[int, int, str], bool]) -> None:
        """
        Set a callback function for progress updates.

        Args:
            callback: Function that takes (completed, total, message) and returns bool.
                     Should return False to cancel the operation, True to continue.
        """
        self._progress_callback = callback
        
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
            transformed_lines, transformed_objs_all, _ = transform_to_utm(lines, structure_geoms + depth_geoms)
            n_struct = len(structure_geoms)
            transformed_structs = transformed_objs_all[:n_struct]
            transformed_depths = transformed_objs_all[n_struct:]

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
                if fixed.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(fixed.geoms):
                        fixed_structs.append(poly)
                        # Create new metadata entry with unique id
                        orig = structures[i] if i < len(structures) else {'id': f'struct_{i}', 'height': 0.0}
                        fixed_structs_meta.append({
                            'id': f"{orig['id']}_{j}" if len(fixed.geoms) > 1 else orig['id'],
                            'height': orig['height'],
                            'wkt': poly
                        })
                else:
                    fixed_structs.append(fixed)
                    orig = structures[i] if i < len(structures) else {'id': f'struct_{i}', 'height': 0.0}
                    fixed_structs_meta.append({
                        'id': orig['id'],
                        'height': orig['height'],
                        'wkt': fixed
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
                        fixed_depths_meta.append({
                            'id': f"{depth_id}_{j}" if len(fixed.geoms) > 1 else depth_id,
                            'depth': depth_val,
                            'wkt': poly
                        })
                else:
                    fixed_depths.append(fixed)
                    fixed_depths_meta.append({
                        'id': depth_id,
                        'depth': depth_val,
                        'wkt': fixed
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
            total_weighted_work = weighted_struct + weighted_depth

            # Track progress across BOTH calculations
            struct_done = False

            def unified_progress_callback(completed: int, total: int, msg: str) -> bool:
                """Combine progress from structures (phase 1) and depths (phase 2)"""
                if not self._progress_callback:
                    return True

                # Calculate weighted progress
                if not struct_done:
                    # Currently calculating structures (first phase)
                    weighted_progress = (completed / max(total, 1)) * weighted_struct
                else:
                    # Currently calculating depths (second phase)
                    weighted_progress = weighted_struct + (completed / max(total, 1)) * weighted_depth

                # Convert to percentage of total work
                overall_progress = int(weighted_progress)
                overall_total = int(total_weighted_work)

                return self._progress_callback(overall_progress, overall_total, msg)

            # Calculate structures using FAST geometric method (allision)
            struct_probability_holes = compute_probability_holes_smart_hybrid(
                transformed_lines, distributions, weights, structs_gdfs,
                distance=reach_distance,
                progress_callback=unified_progress_callback,
                use_fast=True,
                is_structure=True
            ) if len(structs_gdfs) > 0 else []

            struct_done = True  # Switch to phase 2

            # Calculate depths using FAST geometric method (grounding)
            # NOTE: No draught filtering here - the cascade calculation
            # filters by draught per vessel category
            depth_probability_holes = compute_probability_holes_smart_hybrid(
                transformed_lines, distributions, weights, depths_gdfs,
                distance=reach_distance,
                progress_callback=unified_progress_callback,
                use_fast=True,
                is_structure=False
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
        # Find nearest allision candidate (structure lower than ship height)
        allision_dist, allision_idx = None, None
        if struct_min_dists:
            for s_idx, s in enumerate(structures):
                if s['height'] < height:
                    md = struct_min_dists[leg_idx][d_idx][s_idx]
                    if md is not None and (allision_dist is None or md < allision_dist):
                        allision_dist, allision_idx = md, s_idx

        # Find nearest grounding candidate (depth shallower than draught)
        grounding_dist, grounding_idx = None, None
        if depth_min_dists:
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < draught:
                    md = depth_min_dists[leg_idx][d_idx][dep_idx]
                    if md is not None and (grounding_dist is None or md < grounding_dist):
                        grounding_dist, grounding_idx = md, dep_idx

        # Find nearest anchor candidate
        anchor_dist = None
        if depth_min_dists and anchor_d > 0.0:
            thr = anchor_d * draught
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < thr:
                    md = depth_min_dists[leg_idx][d_idx][dep_idx]
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
        # Allision overlap - use probability hole as primary metric
        ov_all = 0.0
        if allision_idx is not None and allision_dist is not None:
            # Primary: use probability hole (integrated probability mass)
            try:
                if struct_probability_holes:
                    ov_all = struct_probability_holes[leg_idx][d_idx][allision_idx]
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
        ) -> None:
        """Update report dictionaries with contribution."""
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
            'totals': {'allision': 0.0, 'grounding': 0.0},
            'by_leg_direction': {},
            'by_object': {},
            'by_structure_legdir': {}
        }

        total_allision = 0.0
        total_grounding = 0.0

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

                    # Build list of all obstacles with their distances and holes
                    obstacles: list[tuple[str, int, float, float]] = []

                    # Add all structures (allision targets)
                    if struct_min_dists and struct_probability_holes:
                        for s_idx, s in enumerate(structures):
                            if s['height'] < height:
                                try:
                                    dist = struct_min_dists[leg_idx][d_idx][s_idx]
                                    hole_pct = struct_probability_holes[leg_idx][d_idx][s_idx]
                                    if dist is not None and hole_pct > 0.0:
                                        obstacles.append(('allision', s_idx, dist, hole_pct))
                                except (IndexError, TypeError):
                                    pass

                    # Add all depths (anchoring or grounding)
                    if depth_min_dists and depth_probability_holes:
                        anchor_threshold = anchor_d * draught if anchor_d > 0.0 else 0.0
                        for dep_idx, dep in enumerate(depths):
                            try:
                                dist = depth_min_dists[leg_idx][d_idx][dep_idx]
                                hole_pct = depth_probability_holes[leg_idx][d_idx][dep_idx]
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
                            # Anchoring reduces remaining by (P_anchor × hole%)
                            remaining_prob *= (1.0 - anchor_p * hole_pct)

                        elif obs_type == 'allision':
                            # Allision: calculate contribution from this structure
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_allision += contrib

                            # Update report
                            self._update_report(
                                report, 'allision', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

                        elif obs_type == 'grounding':
                            # Grounding: calculate contribution from this depth
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_grounding += contrib

                            # Update report
                            self._update_report(
                                report, 'grounding', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

        report['totals']['allision'] = total_allision
        report['totals']['grounding'] = total_grounding
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
            # Grounding line edit name in UI is LEPDriftingGrounding
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
            struct_probability_holes, depth_probability_holes
        )

        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))

        self.drifting_allision_prob = float(total_allision * allision_rf)
        self.drifting_grounding_prob = float(total_grounding * grounding_rf)
        self.drifting_report = report

        self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")
        except Exception:
            pass
        # Auto-generate Markdown report to disk (best-effort, non-blocking)
        self._auto_generate_drifting_report(data)
        return self.drifting_allision_prob, self.drifting_grounding_prob

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


if __name__ == '__main__':
    # No CLI entry; this module is used via the plugin and tests.
    pass
