"""Pure-data geometry helpers for the drifting-overlap visualisation.

This module groups the math/shapely helpers that were originally inlined
at the top of :mod:`geometries.get_drifting_overlap`.  They have no
QGIS / matplotlib / Qt dependencies so they can be unit-tested in
isolation and reused by ``compute.drifting_model``.

Functions
---------
* :func:`create_polygon_from_line` -- buffer a leg by the weighted
  distribution to get a base coverage polygon.
* :func:`extend_polygon_in_directions` -- sweep that polygon in 8
  drift directions to build the per-direction corridor polygons.
* :func:`compare_polygons_with_objs` -- per (polygon, object) hit
  matrix.
* :func:`estimate_weighted_overlap` -- weighted PDF coverage of an
  intersection polygon.
* :func:`compute_coverages_and_distances` -- batch coverage + distance
  arrays for every (polygon, object).
* :func:`directional_distances_to_points` -- vectorised reverse-ray
  along-drift distances for arbitrary points.
* :func:`directional_min_distance_reverse_ray` -- min along-drift
  distance from a leg to any vertex of an intersection polygon.
"""
from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def create_polygon_from_line(
    line: LineString,
    distributions: list[Any],
    weights: list[float],
) -> Polygon:
    """Buffer ``line`` by the weighted distribution to get a coverage polygon.

    The returned polygon covers the lateral 99.9999% (~4.89σ) of the
    weighted-distribution mixture, translated by the weighted mean.
    """
    weights = np.array(weights) / np.sum(weights)
    weighted_mu = sum(
        weight * dist.mean() for dist, weight in zip(distributions, weights)
    )
    weighted_std = np.sqrt(sum(
        weight * (dist.std() ** 2)
        for dist, weight in zip(distributions, weights)
    ))
    moved_line = translate(line, xoff=0, yoff=weighted_mu)
    coverage_range = 4.89 * weighted_std
    return moved_line.buffer(coverage_range)


def extend_polygon_in_directions(
    polygon: Polygon,
    distance: float,
) -> tuple[list[BaseGeometry], list[LineString]]:
    """Extend ``polygon`` in 8 directions (math angles 0,45,...,315).

    Returns one swept polygon per direction plus the centre-line that
    connects the original polygon to the translated one.  Degenerate
    inputs return 8 empty placeholders so downstream consumers don't
    have to handle ``None``.
    """
    extended_polygons: list[BaseGeometry] = []
    centre_lines: list[LineString] = []
    if polygon is None or polygon.is_empty:
        return [Polygon()] * 8, [LineString()] * 8

    for angle in range(0, 360, 45):
        dx = distance * np.cos(np.radians(angle))
        dy = distance * np.sin(np.radians(angle))
        translated_polygon = translate(polygon, xoff=dx, yoff=dy)
        connecting_polygon = polygon.union(translated_polygon).convex_hull
        extended_polygons.append(connecting_polygon)
        try:
            c0 = polygon.representative_point()
            c1 = translated_polygon.representative_point()
            centre_lines.append(LineString([(c0.x, c0.y), (c1.x, c1.y)]))
        except Exception:
            centre_lines.append(LineString())
    return extended_polygons, centre_lines


def compare_polygons_with_objs(
    extended_polygons: list[BaseGeometry],
    objs_gdf_list: list[gpd.GeoDataFrame],
) -> dict[str, list[list[bool]]]:
    """Per (polygon, gdf, obj) overlap matrix indexed by ``"Polygon_<i>"``."""
    results: dict[str, Any] = {}
    for i, polygon in enumerate(extended_polygons):
        results[f"Polygon_{i}"] = []
        for objs_gdf in objs_gdf_list:
            intersects = objs_gdf.intersects(polygon)
            results[f"Polygon_{i}"].append(intersects.tolist())
    return results


def estimate_weighted_overlap(
    intersection: BaseGeometry,
    line: LineString,
    distributions: list[Any],
    weights: list[float],
) -> tuple[float, np.ndarray]:
    """Weighted PDF coverage of ``intersection`` against the mixture."""
    weights = np.array(weights) / np.sum(weights)

    closest_point = line.interpolate(line.project(intersection.centroid))
    if isinstance(intersection, Polygon):
        sample_points = np.array(intersection.exterior.coords)
    elif isinstance(intersection, MultiPolygon):
        sample_points = np.vstack(
            [np.array(poly.exterior.coords) for poly in intersection.geoms]
        )
    else:
        raise ValueError("Unkown geom type type")
    sample_points = np.atleast_2d(sample_points)
    distances = np.sqrt(
        (sample_points[:, 0] - closest_point.x) ** 2
        + (sample_points[:, 1] - closest_point.y) ** 2
    )

    combined_probabilities = np.zeros_like(distances)
    for dist, weight in zip(distributions, weights):
        combined_probabilities += weight * dist.pdf(distances)
    weighted_overlap = combined_probabilities.sum() * 100
    return weighted_overlap, distances


def compute_coverages_and_distances(
    extended_polygons: list[BaseGeometry],
    centre_lines: list[LineString],
    distributions: list[Any],
    weights: list[float],
    objs_gdf_list: list[gpd.GeoDataFrame],
    results: dict[str, list[list[bool]]],
) -> tuple[list[float], list[Any], list[bool]]:
    """Compute weighted coverage + distance arrays per (polygon, object).

    Returns
    -------
    (coverages, distances, covered)
        - coverages: flat list, one entry per (polygon, gdf, obj) triple.
        - distances: flat list of np.ndarray distance arrays, same length.
        - covered: per-polygon flag (any object intersected).
    """
    coverages: list[float] = []
    distances: list[Any] = []
    covered: list[bool] = []
    for i, polygon in enumerate(extended_polygons):
        covered.append(False)
        for gdf_idx, objs_gdf in enumerate(objs_gdf_list):
            for j, obj in enumerate(objs_gdf.geometry):
                if results[f"Polygon_{i}"][gdf_idx][j]:
                    intersection = polygon.intersection(obj)
                    coverage, dists = estimate_weighted_overlap(
                        intersection,
                        centre_lines[i],
                        distributions,
                        weights,
                    )
                    coverages.append(coverage)
                    distances.append(dists)
                    covered[i] = True
                else:
                    coverages.append(0)
                    distances.append(np.ndarray([]))
    return coverages, distances, covered


def directional_distances_to_points(
    points: np.ndarray,
    leg: LineString,
    compass_angle_deg: float,
    use_leg_offset: bool = False,
) -> np.ndarray:
    """Vectorised along-drift distances from ``leg`` to ``points``.

    For each point ``p`` casts a reverse ray against
    ``compass_angle_deg`` and returns the along-drift distance to the
    first leg-segment hit.  Misses fall back to a "nearest point on
    leg projected onto drift direction" rule.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty(0)
    if points.ndim == 1:
        points = points.reshape(1, 2)

    n = points.shape[0]
    if leg is None or leg.is_empty:
        return np.full(n, np.inf)

    from drifting.engine import compass_to_math_deg

    leg_coords = np.asarray(leg.coords, dtype=float)
    if leg_coords.shape[0] < 2:
        return np.full(n, np.inf)

    math_deg = compass_to_math_deg(float(compass_angle_deg))
    rad = math.radians(math_deg)
    ux = math.cos(rad)
    uy = math.sin(rad)
    u = np.array([ux, uy])
    u_perp = np.array([-uy, ux])

    origin = leg_coords[0]
    verts_rel = points - origin
    leg_rel = leg_coords - origin
    verts_along = verts_rel @ u
    verts_perp = verts_rel @ u_perp
    leg_along = leg_rel @ u
    leg_perp = leg_rel @ u_perp

    seg_a0 = leg_along[:-1]
    seg_a1 = leg_along[1:]
    seg_p0 = leg_perp[:-1]
    seg_p1 = leg_perp[1:]

    ray_a = verts_along[:, None]
    ray_p = verts_perp[:, None]
    p_min = np.minimum(seg_p0, seg_p1)[None, :]
    p_max = np.maximum(seg_p0, seg_p1)[None, :]
    dp = (seg_p1 - seg_p0)[None, :]

    crosses = (ray_p >= p_min) & (ray_p <= p_max) & (dp != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ray_p - seg_p0[None, :]) / dp
        along_int = seg_a0[None, :] + t * (seg_a1 - seg_a0)[None, :]
    dist = ray_a - along_int
    valid = crosses & (dist >= 0)
    dist = np.where(valid, dist, np.inf)
    per_point_min = np.min(dist, axis=1)

    finite = np.isfinite(per_point_min)
    if finite.all():
        return per_point_min

    miss_idx = np.where(~finite)[0]
    miss_pts = points[miss_idx]

    seg_p0_xy = leg_coords[:-1]
    seg_p1_xy = leg_coords[1:]
    seg_v = seg_p1_xy - seg_p0_xy
    seg_len_sq = (seg_v * seg_v).sum(axis=1)
    safe_len_sq = np.where(seg_len_sq > 0, seg_len_sq, 1.0)

    diff = miss_pts[:, None, :] - seg_p0_xy[None, :, :]
    t = np.einsum('kmi,mi->km', diff, seg_v) / safe_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    t = np.where(seg_len_sq[None, :] > 0, t, 0.0)
    nearest = seg_p0_xy[None, :, :] + t[:, :, None] * seg_v[None, :, :]
    delta = miss_pts[:, None, :] - nearest
    dist2 = (delta * delta).sum(axis=2)
    best_seg = np.argmin(dist2, axis=1)
    k_range = np.arange(miss_pts.shape[0])
    near = nearest[k_range, best_seg]

    vec = miss_pts - near
    dot = vec[:, 0] * ux + vec[:, 1] * uy
    positive = dot >= 0
    per_point_min[miss_idx[positive]] = dot[positive]
    return per_point_min


def directional_min_distance_reverse_ray(
    intersection: BaseGeometry,
    leg: LineString,
    compass_angle_deg: float,
) -> float | None:
    """Min along-drift distance from ``leg`` to any vertex of ``intersection``.

    Returns ``None`` when no vertex is reachable by drifting from the
    leg in the given direction.
    """
    if leg is None or leg.is_empty:
        return None

    coords: list[tuple[float, float]] = []
    if isinstance(intersection, Polygon):
        coords = list(intersection.exterior.coords)
        for hole in intersection.interiors:
            coords.extend(hole.coords)
    elif isinstance(intersection, MultiPolygon):
        for poly in intersection.geoms:
            coords.extend(poly.exterior.coords)
            for hole in poly.interiors:
                coords.extend(hole.coords)
    else:
        return None

    if not coords:
        return None

    verts = np.asarray(coords, dtype=float)
    dists = directional_distances_to_points(
        verts, leg, compass_angle_deg, use_leg_offset=False,
    )
    finite = np.isfinite(dists)
    if not finite.any():
        return None
    return float(dists[finite].min())


def compute_min_distance_by_object(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float,
) -> list[list[list[float | None]]]:
    """For each leg + 8 drift directions, min reverse-ray distance per object.

    Returns a 3-level list indexed as
    ``[leg_index][direction_index][object_index]``.  ``None`` means the
    object is not reachable by drifting from that leg/direction.
    """
    per_leg_dir_obj: list[list[list[float | None]]] = []
    for line, dist, wgt in zip(lines, distributions, weights):
        n_objs = sum(len(gdf) for gdf in objs_gdf_list)
        per_dir: list[list[float | None]] = []

        base_polygon = create_polygon_from_line(line, dist, wgt)
        extended_polygons, _centre_lines = extend_polygon_in_directions(
            base_polygon, distance,
        )

        for d_idx, polygon in enumerate(extended_polygons):
            math_angle = (d_idx * 45) % 360
            compass_angle = (90 - math_angle) % 360

            min_dists: list[float | None] = [None] * n_objs
            flat_idx = 0
            for objs_gdf in objs_gdf_list:
                for obj in objs_gdf.geometry:
                    if polygon.intersects(obj):
                        intersection = polygon.intersection(obj)
                        try:
                            md = directional_min_distance_reverse_ray(
                                intersection, line, compass_angle,
                            )
                            if md is not None:
                                prev = min_dists[flat_idx]
                                if prev is None or md < prev:
                                    min_dists[flat_idx] = md
                        except Exception:
                            pass
                    flat_idx += 1
            per_dir.append(min_dists)
        per_leg_dir_obj.append(per_dir)
    return per_leg_dir_obj
