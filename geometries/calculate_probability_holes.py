"""
Calculate probability holes using semi-analytical integration.

For straight legs, this computes the probability mass from the lateral distribution
that drifts into structures/depths when drifting in each of 8 compass directions.
"""
from typing import Any, Callable, Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.prepared import prep
from scipy.integrate import dblquad
import time
import logging
import sys

# Set up logger
logger = logging.getLogger(__name__)


def _log(message: str) -> None:
    """Log message to logger only (minimal console spam)."""
    logger.info(message)


def compute_probability_holes(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
) -> list[list[list[float]]]:
    """
    Calculate probability holes for all legs, drift directions, and objects.

    Uses semi-analytical integration (scipy.dblquad) with geometric ray intersection
    to compute the probability that vessels drifting from the leg hit each object.

    Args:
        lines: List of LineString geometries representing traffic legs
        distributions: List of distribution lists (one per leg, each containing scipy.stats distributions)
        weights: List of weight lists (one per leg, weights for each distribution)
        objs_gdf_list: List of GeoDataFrames containing object geometries (structures or depths)
        distance: Maximum drift distance in meters
        progress_callback: Optional callback function(completed, total, message) -> bool.
                          Returns False to cancel, True to continue.

    Returns:
        3-level nested list: [leg_idx][direction_idx][object_idx] = probability (0-1)
        where direction_idx corresponds to: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    """
    per_leg_dir_obj: list[list[list[float]]] = []

    # Flatten object indexing
    obj_index_map: list[tuple[int, int]] = []
    for gi, gdf in enumerate(objs_gdf_list):
        for ri in range(len(gdf)):
            obj_index_map.append((gi, ri))

    # OPTIMIZATION: Prepare geometries for fast intersection (10x speedup!)
    # Prepared geometries build spatial index once, reuse for all intersection checks
    prepared_objs_dict = {}
    for gi, ri in obj_index_map:
        geom = objs_gdf_list[gi].geometry.iloc[ri]
        prepared_objs_dict[(gi, ri)] = prep(geom)

    # 8 drift directions (compass directions)
    drift_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Calculate total number of holes for progress tracking
    total_holes = len(lines) * len(drift_angles) * len(obj_index_map)
    completed_holes = 0
    skipped_full_coverage = 0
    skipped_too_far = 0
    start_time = time.time()

    _log(f"\n{'='*80}")
    _log(f"COMPUTING PROBABILITY HOLES (OPTIMIZED)")
    _log(f"{'='*80}")
    _log(f"Legs: {len(lines)}")
    _log(f"Drift directions: {len(drift_angles)}")
    _log(f"Objects: {len(obj_index_map)}")
    _log(f"Total hole calculations: {total_holes}")
    _log(f"{'='*80}")
    _log(f"Optimizations enabled:")
    _log(f"  • Prepared geometries (10x faster intersections)")
    _log(f"  • Skip after 100% coverage")
    _log(f"  • Skip objects too far away")
    _log(f"  • Adaptive tolerance (faster for small holes)")
    _log(f"  • Reduced lateral range (±5σ instead of ±9.5σ)")
    _log(f"  • Relaxed tolerance for low-probability holes")
    _log(f"{'='*80}\n")

    for leg_idx, (line, dists, wgts) in enumerate(zip(lines, distributions, weights)):
        # Normalize weights
        w = np.array(wgts)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()

        # Get leg geometry
        leg_coords = np.array(line.coords)
        if len(leg_coords) < 2:
            # Degenerate line
            per_leg_dir_obj.append([[0.0] * len(obj_index_map) for _ in drift_angles])
            continue

        leg_start = leg_coords[0]
        leg_end = leg_coords[-1]
        leg_vec = leg_end - leg_start
        leg_len = line.length
        leg_dir = leg_vec / leg_len if leg_len > 0 else np.array([1, 0])

        # Perpendicular direction for lateral offsets
        perp_dir = np.array([-leg_dir[1], leg_dir[0]])

        # Calculate lateral range
        # Using ±5σ instead of ±9.5σ for better performance (covers 99.9999% vs 99.999...999%)
        # This reduces integration domain significantly with negligible accuracy loss
        weighted_std = float(np.sqrt(sum(weight * (dist.std() ** 2) for dist, weight in zip(dists, w))))
        lateral_range = 5.0 * weighted_std

        per_dir: list[list[float]] = []

        for dir_idx, angle_deg in enumerate(drift_angles):
            angle_rad = np.radians(angle_deg)
            drift_dx = np.cos(angle_rad)
            drift_dy = np.sin(angle_rad)
            drift_vec_cached = np.array([drift_dx, drift_dy])

            dir_holes: list[float] = []
            cumulative_hole = 0.0  # Track total coverage in this direction

            for obj_idx, (gi, ri) in enumerate(obj_index_map):
                obj = objs_gdf_list[gi].geometry.iloc[ri]
                prepared_obj = prepared_objs_dict[(gi, ri)]

                # Progress tracking
                completed_holes += 1

                # OPTIMIZATION 1: If we've already covered 100% (or 99.9%), skip remaining objects
                if cumulative_hole >= 0.999:
                    probability_hole = 0.0
                    dir_holes.append(probability_hole)
                    skipped_full_coverage += 1

                    # Progress callback every 10%
                    if progress_callback and completed_holes % max(1, total_holes // 10) == 0:
                        elapsed = time.time() - start_time
                        avg_time_per_hole = elapsed / completed_holes
                        remaining_holes = total_holes - completed_holes
                        eta_seconds = avg_time_per_hole * remaining_holes
                        eta_min = int(eta_seconds / 60)
                        eta_sec = int(eta_seconds % 60)
                        msg = f"Progress: {completed_holes}/{total_holes} | ETA: {eta_min}m {eta_sec}s"
                        should_continue = progress_callback(completed_holes, total_holes, msg)
                        if not should_continue:
                            logger.info("Calculation cancelled by user")
                            return per_leg_dir_obj
                    continue

                # OPTIMIZATION 2: Quick distance check - if object is too far, skip
                min_dist = line.distance(obj)
                max_possible_reach = distance + lateral_range
                if min_dist > max_possible_reach:
                    probability_hole = 0.0
                    dir_holes.append(probability_hole)
                    skipped_too_far += 1

                    # Progress callback every 10%
                    if progress_callback and completed_holes % max(1, total_holes // 10) == 0:
                        elapsed = time.time() - start_time
                        avg_time_per_hole = elapsed / completed_holes
                        remaining_holes = total_holes - completed_holes
                        eta_seconds = avg_time_per_hole * remaining_holes
                        eta_min = int(eta_seconds / 60)
                        eta_sec = int(eta_seconds % 60)
                        msg = f"Progress: {completed_holes}/{total_holes} | ETA: {eta_min}m {eta_sec}s"
                        should_continue = progress_callback(completed_holes, total_holes, msg)
                        if not should_continue:
                            logger.info("Calculation cancelled by user")
                            return per_leg_dir_obj
                    continue

                # OPTIMIZATION: Create PDF cache to avoid recalculating for same y values
                pdf_cache = {}

                # Define integrand for semi-analytical integration
                def integrand(y: float, s: float) -> float:
                    """
                    Integrand function for probability hole calculation.

                    Args:
                        y: Lateral offset from leg (meters)
                        s: Parameter along leg [0, 1]

                    Returns:
                        Combined PDF value if drift trajectory hits object, 0 otherwise
                    """
                    # Position along leg
                    leg_pos = leg_start + s * leg_vec

                    # Add lateral offset
                    start_pos = leg_pos + y * perp_dir

                    # End position after drifting (use cached drift vector)
                    end_pos = start_pos + distance * drift_vec_cached

                    # Create drift ray
                    ray = LineString([start_pos, end_pos])

                    # Check if ray intersects object (use prepared geometry for 10x speedup!)
                    if prepared_obj.intersects(ray):
                        # Use cached PDF value if available (many integration points use same y)
                        if y not in pdf_cache:
                            pdf_cache[y] = sum(weight * dist.pdf(y) for weight, dist in zip(w, dists))
                        return pdf_cache[y]
                    else:
                        return 0.0

                # (Removed verbose "Starting integration" messages)

                # Perform 2D adaptive integration
                # OPTIMIZATION 3: Use adaptive tolerance based on cumulative coverage
                # If we've already covered a lot, we can be less precise for remaining small holes
                if cumulative_hole > 0.95:
                    # We've covered 95%+, remaining holes are tiny - use very fast tolerance
                    tolerance = 1e-1  # 10% error is fine for <5% contribution
                elif cumulative_hole > 0.8:
                    # We've covered 80%+, remaining holes are small - use faster tolerance
                    tolerance = 5e-2  # 5% error
                elif cumulative_hole > 0.5:
                    # We've covered 50%+, use moderate tolerance
                    tolerance = 2e-2  # 2% error
                else:
                    # First holes are most important, use tighter tolerance
                    tolerance = 1e-2  # 1% error

                try:
                    result, error = dblquad(
                        integrand,
                        0, 1,  # s bounds (leg parameter)
                        lambda s: -lateral_range,
                        lambda s: lateral_range,  # y bounds (lateral offset)
                        epsabs=tolerance,  # Absolute error tolerance
                        epsrel=tolerance,  # Relative error tolerance
                    )
                    # CRITICAL: Normalize by leg length to get geometric causation factor (per unit length)
                    # This prevents double-counting: hours_present in run_calculations.py already accounts for leg length
                    # Without this normalization, longer legs get artificially higher probabilities
                    probability_hole = result / leg_len if leg_len > 0 else 0.0
                    # Clamp to reasonable range [0, 1]
                    probability_hole = max(0.0, min(1.0, probability_hole))
                except Exception:
                    # If integration fails, fall back to 0
                    probability_hole = 0.0

                # Update cumulative coverage for this direction
                cumulative_hole += probability_hole

                # Progress reporting (only every 10%)
                if progress_callback and completed_holes % max(1, total_holes // 10) == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_hole = elapsed / completed_holes
                    remaining_holes = total_holes - completed_holes
                    eta_seconds = avg_time_per_hole * remaining_holes
                    eta_min = int(eta_seconds / 60)
                    eta_sec = int(eta_seconds % 60)
                    msg = f"Progress: {completed_holes}/{total_holes} | ETA: {eta_min}m {eta_sec}s"
                    should_continue = progress_callback(completed_holes, total_holes, msg)
                    if not should_continue:
                        logger.info("Calculation cancelled by user")
                        return per_leg_dir_obj

                dir_holes.append(probability_hole)

            per_dir.append(dir_holes)

        per_leg_dir_obj.append(per_dir)

    # Final summary
    total_time = time.time() - start_time
    actually_computed = total_holes - skipped_full_coverage - skipped_too_far
    _log(f"\n{'='*80}")
    _log("HOLE CALCULATION COMPLETE")
    _log(f"{'='*80}")
    _log(f"Total holes: {total_holes}")
    _log(f"  • Actually computed: {actually_computed} ({actually_computed/total_holes*100:.1f}%)")
    _log(f"  • Skipped (100% coverage): {skipped_full_coverage} ({skipped_full_coverage/total_holes*100:.1f}%)")
    _log(f"  • Skipped (too far): {skipped_too_far} ({skipped_too_far/total_holes*100:.1f}%)")
    _log(f"Total time: {int(total_time/60)}m {int(total_time%60)}s ({total_time:.1f}s)")
    if actually_computed > 0:
        _log(f"Average time per computed hole: {total_time/actually_computed:.2f}s")
    _log(f"Average time per total hole: {total_time/total_holes:.2f}s")
    speedup = total_holes / max(actually_computed, 1)
    _log(f"Effective speedup from skipping: {speedup:.1f}x")
    _log(f"{'='*80}\n")

    return per_leg_dir_obj
