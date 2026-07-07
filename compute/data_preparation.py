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


SCALING_KEY = 'Scaling (%)'
FREQ_KEY = 'Frequency (ships/year)'


def _scale_direction_freq(var: dict) -> None:
    freq = var.get(FREQ_KEY)
    if freq is None:
        return
    scaling = var.get(SCALING_KEY)
    for i, row in enumerate(freq):
        if not hasattr(row, '__iter__'):
            continue
        for j, q in enumerate(row):
            factor = 1.0
            if scaling is not None and i < len(scaling):
                s_row = scaling[i]
                if hasattr(s_row, '__iter__') and j < len(s_row):
                    try:
                        factor = float(s_row[j]) / 100.0
                    except (TypeError, ValueError):
                        factor = 1.0
            if factor == 1.0:
                continue
            if q == '' or q is None:
                continue
            try:
                q_val = float(q)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(q_val):
                continue
            row[j] = q_val * factor


def apply_traffic_scaling(data: dict[str, Any]) -> None:
    """Multiply every ``Frequency (ships/year)`` cell by its ``Scaling (%)`` / 100.

    Mutates ``data['traffic_data']`` in place.  Callers must pass a deep-copy
    of the live UI state so the user's stored Q values stay untouched.
    """
    traffic = data.get('traffic_data') or {}
    if not isinstance(traffic, dict):
        return
    for _seg, dirs in traffic.items():
        if not isinstance(dirs, dict):
            continue
        for _di, var in dirs.items():
            if isinstance(var, dict):
                _scale_direction_freq(var)


def _is_qgis_available() -> bool:
    """Return True only when a real (non-mocked) QGIS environment is active.

    MagicMock stubs are detected because their ``isValid()`` returns a Mock
    object, not a Python bool.
    """
    try:
        crs = QgsCoordinateReferenceSystem("EPSG:4326")
        return isinstance(crs.isValid(), bool)
    except Exception:
        return False


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


def _expand_geom_entries(items: list, value_key: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in items:
        try:
            eid, val, wkt = entry
        except Exception:
            continue
        geom = safe_load_wkt(wkt)
        if geom is None:
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                result.append({'id': f'{eid}_{i}', value_key: fval, 'wkt': poly})
        else:
            result.append({'id': str(eid), value_key: fval, 'wkt': geom})
    return result


def split_structures_and_depths(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (structures, depths), splitting MultiPolygons into individual Polygons."""
    structures = _expand_geom_entries(data.get('objects', []), 'height')
    depths = _expand_geom_entries(data.get('depths', []), 'depth')
    return structures, depths


def _get_utm_epsg(lon: float, lat: float) -> int:
    utm_zone = int((lon + 180) // 6) + 1
    return 32600 + utm_zone if lat >= 0 else 32700 + utm_zone


def _make_coord_transform(utm_epsg: int):
    if _is_qgis_available():
        wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")
        transform_context = QgsProject.instance().transformContext()
        coord_transform = QgsCoordinateTransform(wgs84_crs, utm_crs, transform_context)
        def transform_coords(x, y):
            from qgis.core import QgsPointXY
            point = coord_transform.transform(QgsPointXY(x, y))
            return point.x(), point.y()
    else:
        from pyproj import Transformer as _Transformer
        _proj = _Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        def transform_coords(x, y):
            return _proj.transform(x, y)
    return transform_coords


def transform_to_utm(lines, objects):
    """Transform lines and objects from WGS84 (EPSG:4326) to the appropriate UTM zone."""
    all_geometries = lines + objects
    cx = sum(g.centroid.x for g in all_geometries) / len(all_geometries)
    cy = sum(g.centroid.y for g in all_geometries) / len(all_geometries)
    utm_epsg = _get_utm_epsg(cx, cy)
    tc = _make_coord_transform(utm_epsg)
    xform = lambda geom: transform(tc, geom)
    return [xform(l) for l in lines], [xform(o) for o in objects], utm_epsg


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
