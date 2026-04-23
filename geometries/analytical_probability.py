"""
Analytical cross-section CDF integration for drift probability holes.

Instead of Monte Carlo sampling, this computes exact probability holes by:
1. Slicing the leg into N cross-sections (positions s along the leg)
2. For each slice, analytically determining the y-intervals where drift rays
   hit the obstacle polygon (using vectorized linear algebra on polygon edges)
3. Integrating the lateral PDF over those intervals using CDF (exact)

This is deterministic (no MC noise) and typically faster than MC.

Geometry overview:
    For a fixed position s along the leg, a ship starts at:
        P(s, y) = leg_start + s * leg_vec + y * perp_dir

    and drifts in a straight line:
        ray(t) = P(s, y) + t * distance * drift_vec,  t in [0, 1]

    The set of y-values where this ray intersects a polygon edge (A, B)
    is found by solving:
        P(s, y) + t * D * drift = A + u * (B - A)

    This gives t(y) and u(y) as LINEAR functions of y.
    The constraints t in [0,1] and u in [0,1] each define a y-interval.
    Their intersection is the y-range where the ray crosses that edge.

    The hit region for a slice is the UNION of all edge crossing intervals.

    The probability for each interval [y_lo, y_hi] is:
        sum(weight_i * (CDF_i(y_hi) - CDF_i(y_lo)))

PERFORMANCE: The core computation is fully vectorized with numpy,
processing all slices x all edges in a single batch operation.
"""

from typing import Any, Optional, Callable
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
import geopandas as gpd
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)


def _extract_polygon_rings(geom) -> list[np.ndarray]:
    """Extract polygon rings as numpy arrays from a Shapely geometry."""
    rings = []
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            rings.append(np.array(geom.exterior.coords))
            for hole in geom.interiors:
                rings.append(np.array(hole.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if not poly.is_empty:
                rings.append(np.array(poly.exterior.coords))
                for hole in poly.interiors:
                    rings.append(np.array(hole.coords))
    return rings


def _vectorized_edge_y_intervals(
    s_values: np.ndarray,
    leg_start: np.ndarray,
    leg_vec: np.ndarray,
    perp_dir: np.ndarray,
    drift_vec: np.ndarray,
    distance: float,
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    lateral_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of y-intervals for ALL slices x ALL edges at once.

    Args:
        s_values: (N_slices,) array of s positions along leg
        edge_starts: (N_edges, 2) array of edge start points
        edge_ends: (N_edges, 2) array of edge end points

    Returns:
        y_lo: (N_slices, N_edges) lower bounds of valid y-intervals
        y_hi: (N_slices, N_edges) upper bounds of valid y-intervals
        valid: (N_slices, N_edges) boolean mask of valid intervals
    """
    N_s = len(s_values)
    N_e = len(edge_starts)

    # Edge vectors: (N_edges, 2)
    edge_vecs = edge_ends - edge_starts

    # Matrix coefficients (same for all slices):
    D = distance
    a11 = D * drift_vec[0]
    a12 = -edge_vecs[:, 0]  # (N_edges,)
    a21 = D * drift_vec[1]
    a22 = -edge_vecs[:, 1]  # (N_edges,)

    # Determinant: (N_edges,)
    det = a11 * a22 - a12 * a21
    valid_det = np.abs(det) > 1e-12
    safe_det = np.where(valid_det, det, 1.0)
    inv_det = 1.0 / safe_det  # (N_edges,)

    # Origin points for each s: (N_slices, 2)
    origins = leg_start[np.newaxis, :] + s_values[:, np.newaxis] * leg_vec[np.newaxis, :]

    # c = edge_start - origin: (N_slices, N_edges, 2)
    c = edge_starts[np.newaxis, :, :] - origins[:, np.newaxis, :]

    # t0(s, e) = inv_det[e] * (a22[e] * c[s,e,0] - a12[e] * c[s,e,1])
    t0 = inv_det[np.newaxis, :] * (
        a22[np.newaxis, :] * c[:, :, 0] - a12[np.newaxis, :] * c[:, :, 1]
    )  # (N_slices, N_edges)

    # t_slope = inv_det * (a22 * (-perp.x) - a12 * (-perp.y))
    # Same for all slices: (N_edges,)
    t_slope = inv_det * (a22 * (-perp_dir[0]) - a12 * (-perp_dir[1]))

    # u0(s, e) = inv_det[e] * (-a21 * c[s,e,0] + a11 * c[s,e,1])
    u0 = inv_det[np.newaxis, :] * (
        -a21 * c[:, :, 0] + a11 * c[:, :, 1]
    )  # (N_slices, N_edges)

    # u_slope: (N_edges,)
    u_slope = inv_det * (-a21 * (-perp_dir[0]) + a11 * (-perp_dir[1]))

    # Initialize y bounds
    y_lo = np.full((N_s, N_e), -lateral_range)
    y_hi = np.full((N_s, N_e), lateral_range)
    valid = valid_det[np.newaxis, :].repeat(N_s, axis=0)  # (N_slices, N_edges)

    # --- Constrain by t in [0, 1] ---
    t_slope_2d = t_slope[np.newaxis, :].repeat(N_s, axis=0)
    has_t_slope = np.abs(t_slope_2d) > 1e-15
    safe_t_slope = np.where(has_t_slope, t_slope_2d, 1.0)

    yt0 = -t0 / safe_t_slope
    yt1 = (1.0 - t0) / safe_t_slope
    t_lo = np.minimum(yt0, yt1)
    t_hi = np.maximum(yt0, yt1)

    # Apply t-slope constraints
    y_lo = np.where(has_t_slope, np.maximum(y_lo, t_lo), y_lo)
    y_hi = np.where(has_t_slope, np.minimum(y_hi, t_hi), y_hi)

    # Where t_slope ~ 0, check if constant t0 is in [0, 1]
    const_t_invalid = ~has_t_slope & ((t0 < 0.0) | (t0 > 1.0))
    valid &= ~const_t_invalid

    # --- Constrain by u in [0, 1] ---
    u_slope_2d = u_slope[np.newaxis, :].repeat(N_s, axis=0)
    has_u_slope = np.abs(u_slope_2d) > 1e-15
    safe_u_slope = np.where(has_u_slope, u_slope_2d, 1.0)

    yu0 = -u0 / safe_u_slope
    yu1 = (1.0 - u0) / safe_u_slope
    u_lo = np.minimum(yu0, yu1)
    u_hi = np.maximum(yu0, yu1)

    y_lo = np.where(has_u_slope, np.maximum(y_lo, u_lo), y_lo)
    y_hi = np.where(has_u_slope, np.minimum(y_hi, u_hi), y_hi)

    const_u_invalid = ~has_u_slope & ((u0 < 0.0) | (u0 > 1.0))
    valid &= ~const_u_invalid

    # Final validity check
    valid &= (y_lo < y_hi)

    return y_lo, y_hi, valid


def _merge_intervals_vectorized(y_lo: np.ndarray, y_hi: np.ndarray,
                                valid: np.ndarray) -> list[tuple[float, float]]:
    """
    Merge valid intervals for a single slice into non-overlapping intervals.

    Args:
        y_lo: (N_edges,) lower bounds
        y_hi: (N_edges,) upper bounds
        valid: (N_edges,) boolean mask
    """
    mask = valid
    if not np.any(mask):
        return []

    lo = y_lo[mask]
    hi = y_hi[mask]

    # Sort by lower bound
    order = np.argsort(lo)
    lo = lo[order]
    hi = hi[order]

    # Merge overlapping intervals
    merged = [(lo[0], hi[0])]
    for i in range(1, len(lo)):
        if lo[i] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi[i]))
        else:
            merged.append((lo[i], hi[i]))

    return merged


def compute_probability_analytical(
    leg_start: np.ndarray,
    leg_vec: np.ndarray,
    perp_dir: np.ndarray,
    drift_vec: np.ndarray,
    distance: float,
    lateral_range: float,
    polygon_rings: list[np.ndarray],
    dists: list,
    weights: np.ndarray,
    n_slices: int = 100,
) -> float:
    """
    Compute probability hole using analytical cross-section CDF integration.

    Fully vectorized: computes all slices x all edges in a single batch
    operation using numpy broadcasting, then integrates PDF via CDF.

    Returns:
        Probability value between 0 and 1
    """
    # Collect all edges from all rings
    all_edge_starts = []
    all_edge_ends = []
    for ring in polygon_rings:
        if len(ring) < 3:
            continue
        all_edge_starts.append(ring[:-1])
        all_edge_ends.append(ring[1:])

    if not all_edge_starts:
        return 0.0

    edge_starts = np.vstack(all_edge_starts)  # (N_edges, 2)
    edge_ends = np.vstack(all_edge_ends)      # (N_edges, 2)

    # Slice positions along the leg
    s_values = (np.arange(n_slices) + 0.5) / n_slices

    # Vectorized computation of all y-intervals
    y_lo, y_hi, valid = _vectorized_edge_y_intervals(
        s_values, leg_start, leg_vec, perp_dir, drift_vec, distance,
        edge_starts, edge_ends, lateral_range,
    )

    # Pre-compute CDF arrays for efficiency
    # Collect all unique interval boundaries, then batch-evaluate CDFs
    total_prob = 0.0

    for si in range(n_slices):
        intervals = _merge_intervals_vectorized(y_lo[si], y_hi[si], valid[si])
        if not intervals:
            continue

        # Integrate PDF over merged intervals using CDF
        for iv_lo, iv_hi in intervals:
            for weight, dist in zip(weights, dists):
                total_prob += weight * (dist.cdf(iv_hi) - dist.cdf(iv_lo))

    probability = total_prob / n_slices
    return float(np.clip(probability, 0.0, 1.0))


def _compute_single_direction_analytical(
    leg_idx: int,
    dir_idx: int,
    angle_deg: float,
    line: LineString,
    dists: list[Any],
    wgts: list[float],
    objs_gdf_list: list[gpd.GeoDataFrame],
    obj_index_map: list[tuple[int, int]],
    distance: float,
    lateral_range: float,
    n_slices: int = 100,
) -> tuple[int, int, list[float], int, int]:
    """
    Worker function for one (leg, direction) combination.
    Same interface as the MC version but uses analytical integration.
    """
    # Normalize weights
    w = np.array(wgts)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()

    # Get leg geometry
    leg_coords = np.array(line.coords)
    leg_start = leg_coords[0]
    leg_end = leg_coords[-1]
    leg_vec = leg_end - leg_start
    leg_len = line.length
    leg_dir = leg_vec / leg_len if leg_len > 0 else np.array([1, 0])

    # Perpendicular direction
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    # Drift direction vector
    angle_rad = np.radians(angle_deg)
    drift_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Extract polygon rings
    polygon_coords_dict = {}
    for gi, ri in obj_index_map:
        geom = objs_gdf_list[gi].geometry.iloc[ri]
        polygon_coords_dict[(gi, ri)] = _extract_polygon_rings(geom)

    dir_holes: list[float] = []
    skipped_full_coverage = 0
    skipped_too_far = 0

    for obj_idx, (gi, ri) in enumerate(obj_index_map):
        obj = objs_gdf_list[gi].geometry.iloc[ri]
        polygon_rings = polygon_coords_dict[(gi, ri)]

        # Quick distance check
        min_dist = line.distance(obj)
        max_possible_reach = distance + lateral_range
        if min_dist > max_possible_reach:
            dir_holes.append(0.0)
            skipped_too_far += 1
            continue

        probability_hole = compute_probability_analytical(
            leg_start=leg_start,
            leg_vec=leg_vec,
            perp_dir=perp_dir,
            drift_vec=drift_vec,
            distance=distance,
            lateral_range=lateral_range,
            polygon_rings=polygon_rings,
            dists=dists,
            weights=w,
            n_slices=n_slices,
        )

        dir_holes.append(probability_hole)

    return (leg_idx, dir_idx, dir_holes, skipped_full_coverage, skipped_too_far)


def compute_probability_holes_analytical(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
    n_slices: int = 100,
) -> list[list[list[float]]]:
    """
    Calculate probability holes using analytical cross-section CDF integration.

    Drop-in replacement for compute_probability_holes() from
    calculate_probability_holes.py, but deterministic and typically faster.

    Args:
        lines: List of LineString geometries representing traffic legs
        distributions: List of distribution lists (one per leg)
        weights: List of weight lists (one per leg)
        objs_gdf_list: List of GeoDataFrames containing object geometries
        distance: Maximum drift distance in meters
        progress_callback: Optional callback(completed, total, message) -> bool
        n_slices: Number of cross-section slices per leg (default 100)

    Returns:
        3-level nested list: [leg_idx][direction_idx][object_idx] = probability
    """
    # Flatten object indexing
    obj_index_map: list[tuple[int, int]] = []
    for gi, gdf in enumerate(objs_gdf_list):
        for ri in range(len(gdf)):
            obj_index_map.append((gi, ri))

    drift_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    total_holes = len(lines) * len(drift_angles) * len(obj_index_map)
    start_time = time.time()

    # Pre-calculate lateral ranges
    lateral_ranges = []
    for dists, wgts in zip(distributions, weights):
        w = np.array(wgts)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        weighted_std = float(np.sqrt(
            sum(weight * (dist.std() ** 2) for dist, weight in zip(dists, w))
        ))
        lateral_ranges.append(5.0 * weighted_std)

    # Initialize result structure
    per_leg_dir_obj = [
        [[0.0] * len(obj_index_map) for _ in drift_angles]
        for _ in lines
    ]

    # Prepare tasks
    tasks = []
    for leg_idx, (line, dists, wgts) in enumerate(zip(lines, distributions, weights)):
        leg_coords = np.array(line.coords)
        if len(leg_coords) < 2:
            continue
        lateral_range = lateral_ranges[leg_idx]
        for dir_idx, angle_deg in enumerate(drift_angles):
            tasks.append((
                leg_idx, dir_idx, angle_deg, line, dists, wgts,
                objs_gdf_list, obj_index_map, distance, lateral_range,
                n_slices,
            ))

    # Execute in parallel
    max_workers = max(1, cpu_count() - 1)
    total_tasks = len(tasks)
    completed_tasks = 0
    skipped_full = 0
    skipped_far = 0

    logger.info(
        f"Analytical probability holes: {len(lines)} legs x "
        f"{len(drift_angles)} dirs x {len(obj_index_map)} objects = "
        f"{total_holes} holes, {n_slices} slices/leg, {max_workers} threads"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _compute_single_direction_analytical, *task
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            try:
                leg_idx, dir_idx, dir_holes, skip_cov, skip_far = future.result()
                per_leg_dir_obj[leg_idx][dir_idx] = dir_holes
                skipped_full += skip_cov
                skipped_far += skip_far
                completed_tasks += 1

                if progress_callback:
                    elapsed = time.time() - start_time
                    eta = elapsed / completed_tasks * (total_tasks - completed_tasks)
                    msg = f"Analytical: {completed_tasks}/{total_tasks} | ETA: {int(eta)}s"
                    if not progress_callback(completed_tasks, total_tasks, msg):
                        for f in future_to_task:
                            f.cancel()
                        return per_leg_dir_obj

            except Exception as e:
                logger.error(f"Analytical task failed: {e}")

    total_time = time.time() - start_time
    actually_computed = total_holes - skipped_full - skipped_far
    logger.info(
        f"Analytical complete: {total_time:.1f}s, "
        f"computed={actually_computed}, skipped_far={skipped_far}, "
        f"skipped_full={skipped_full}"
    )

    return per_leg_dir_obj
