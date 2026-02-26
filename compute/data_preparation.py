"""
Data preparation utilities for OMRAT calculations.

Functions for loading, parsing, and transforming raw .omrat data dictionaries
into the geometric/numeric structures consumed by calculation modules.
"""
from typing import Any

import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
from scipy.stats import norm, uniform
import shapely.wkt as sw
from shapely.errors import GEOSException
from shapely.ops import transform
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry


def get_distribution(segment_data: dict[str, Any], direction: int) -> tuple[list[Any], list[float]]:
    d = direction + 1  # given as 0, 1 and should be 1, 2
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


def clean_traffic(data: dict[str, Any]) -> list[tuple[LineString, list[Any], list[float], list[dict[str, float]], str]]:
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


def load_areas(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Loads the objects and depths into one objs dict"""
    objs: list[dict[str, Any]] = []
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
