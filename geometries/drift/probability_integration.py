# -*- coding: utf-8 -*-
"""
Integration between drift corridor shadow algorithm and probability calculation.

Provides shadow-adjusted probability holes that account for upstream obstacles
blocking the drift corridor. This creates more realistic probability cascades
where obstacles that physically block drift paths reduce downstream probabilities.

The key improvement over independent probability calculation:
- Obstacles are processed in distance order (closest to leg first)
- Each obstacle clips the corridor, creating a shadow
- Downstream obstacles can only be reached through the remaining corridor area
- This naturally handles scattered obstacles with gaps (ships drift through gaps)
"""

from typing import Callable, Optional
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import logging

from .constants import DIRECTIONS
from .coordinates import compass_to_vector
from .corridor import create_projected_corridor
from .shadow import create_obstacle_shadow, extract_polygons

logger = logging.getLogger(__name__)


def compute_shadow_adjusted_holes(
    legs_utm: list[LineString],
    obstacles_utm: list[tuple[Polygon, float, str, int]],
    half_width: float,
    projection_dist: float,
    directions: dict[str, int] | None = None,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
) -> dict:
    """
    Compute probability holes with shadow adjustment.

    For each leg and direction:
    1. Create drift corridor
    2. Process obstacles in distance order (closest first)
    3. Track remaining reachable corridor area after each obstacle
    4. Return effective probability = intersection_area / original_area

    The shadow effect: when an obstacle is hit, it blocks the corridor behind it.
    Downstream obstacles can only be reached through the remaining corridor area.

    Args:
        legs_utm: List of leg LineStrings in UTM coordinates
        obstacles_utm: List of (geometry, depth/height value, type, original_index) tuples
                      where type is 'depth' or 'structure'
        half_width: Half the lateral distribution width (meters)
        projection_dist: Drift projection distance (meters)
        directions: Direction dict (default: DIRECTIONS from constants)
        progress_callback: Optional callback(completed, total, message) -> bool

    Returns:
        Dict with structure:
        {
            'effective_holes': [leg][direction][obstacle] = probability (0-1),
            'shadow_factors': [leg][direction][obstacle] = remaining corridor fraction (0-1),
            'corridors': [leg][direction] = final clipped corridor geometry,
            'obstacle_order': [leg][direction] = list of obstacle indices in distance order,
        }
    """
    if directions is None:
        directions = DIRECTIONS

    num_legs = len(legs_utm)
    num_dirs = len(directions)
    num_obs = len(obstacles_utm)
    total_work = num_legs * num_dirs
    completed = 0

    effective_holes: list[list[list[float]]] = []
    shadow_factors: list[list[list[float]]] = []
    corridors: list[list[Polygon]] = []
    obstacle_orders: list[list[list[int]]] = []

    logger.info(f"Computing shadow-adjusted holes: {num_legs} legs, {num_dirs} dirs, {num_obs} obstacles")

    for leg_idx, leg in enumerate(legs_utm):
        leg_holes: list[list[float]] = []
        leg_shadows: list[list[float]] = []
        leg_corridors: list[Polygon] = []
        leg_orders: list[list[int]] = []

        # Pre-compute distances from leg to all obstacles (once per leg)
        obs_distances = []
        for obs_idx, (geom, value, obs_type, orig_idx) in enumerate(obstacles_utm):
            try:
                dist = leg.distance(geom)
            except Exception:
                dist = float('inf')
            obs_distances.append((dist, obs_idx))

        for dir_name, angle in directions.items():
            if progress_callback:
                msg = f"Shadow adjustment: Leg {leg_idx + 1}/{num_legs}, {dir_name}"
                if not progress_callback(completed, total_work, msg):
                    logger.info("Shadow calculation cancelled")
                    return {
                        'effective_holes': effective_holes,
                        'shadow_factors': shadow_factors,
                        'corridors': corridors,
                        'obstacle_order': obstacle_orders,
                        'cancelled': True,
                    }

            # Create base corridor for this direction
            corridor = create_projected_corridor(leg, half_width, angle, projection_dist)

            if corridor.is_empty or corridor.area == 0:
                leg_holes.append([0.0] * num_obs)
                leg_shadows.append([1.0] * num_obs)
                leg_corridors.append(corridor)
                leg_orders.append([])
                completed += 1
                continue

            original_area = corridor.area
            corridor_bounds = corridor.bounds

            # Sort obstacles by distance to leg
            sorted_obs = sorted(obs_distances, key=lambda x: x[0])
            dir_order = [obs_idx for _, obs_idx in sorted_obs]

            # Initialize results for this direction
            dir_holes = [0.0] * num_obs
            dir_shadows = [1.0] * num_obs

            # Process obstacles in distance order, tracking corridor clipping
            current_corridor = corridor

            for dist, obs_idx in sorted_obs:
                geom, value, obs_type, orig_idx = obstacles_utm[obs_idx]

                # Record shadow factor (remaining corridor fraction) at this obstacle
                if current_corridor.is_empty:
                    dir_shadows[obs_idx] = 0.0
                    dir_holes[obs_idx] = 0.0
                    continue

                current_area = current_corridor.area
                dir_shadows[obs_idx] = current_area / original_area

                # Check if obstacle intersects current (possibly clipped) corridor
                try:
                    if not current_corridor.intersects(geom):
                        dir_holes[obs_idx] = 0.0
                        continue

                    # Calculate intersection with current corridor
                    intersection = current_corridor.intersection(geom)
                    intersection = make_valid(intersection)

                    if intersection.is_empty:
                        dir_holes[obs_idx] = 0.0
                        continue

                    # Effective probability = intersection area / original corridor area
                    # This accounts for upstream shadows reducing the reachable area
                    intersection_area = intersection.area
                    effective_prob = intersection_area / original_area
                    dir_holes[obs_idx] = min(1.0, max(0.0, effective_prob))

                    # Clip corridor at this obstacle for subsequent obstacles
                    # This creates the shadow effect
                    current_corridor = _clip_corridor_at_obstacle(
                        current_corridor, geom, angle, corridor_bounds
                    )

                except Exception as e:
                    logger.warning(f"Error processing obstacle {obs_idx}: {e}")
                    dir_holes[obs_idx] = 0.0

            leg_holes.append(dir_holes)
            leg_shadows.append(dir_shadows)
            leg_corridors.append(current_corridor)
            leg_orders.append(dir_order)
            completed += 1

        effective_holes.append(leg_holes)
        shadow_factors.append(leg_shadows)
        corridors.append(leg_corridors)
        obstacle_orders.append(leg_orders)

    logger.info(f"Shadow-adjusted holes computed for {num_legs} legs × {num_dirs} directions")

    return {
        'effective_holes': effective_holes,
        'shadow_factors': shadow_factors,
        'corridors': corridors,
        'obstacle_order': obstacle_orders,
        'cancelled': False,
    }


def _clip_corridor_at_obstacle(
    corridor: Polygon,
    obstacle: Polygon,
    drift_angle_deg: float,
    corridor_bounds: tuple[float, float, float, float],
) -> Polygon:
    """
    Clip corridor at an obstacle, creating a shadow in the drift direction.

    Args:
        corridor: Current corridor polygon
        obstacle: Obstacle to clip at
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)
        corridor_bounds: Original corridor bounds for shadow calculation

    Returns:
        Clipped corridor polygon
    """
    try:
        # Get intersection of corridor with obstacle
        intersection = corridor.intersection(obstacle)
        intersection = make_valid(intersection)

        if intersection.is_empty:
            return corridor

        # Extract polygon parts from intersection
        parts = extract_polygons(intersection)

        if not parts:
            return corridor

        # Create shadow/blocking zone for each part
        blocking_zones = []
        for part in parts:
            if part.is_empty:
                continue
            shadow = create_obstacle_shadow(part, drift_angle_deg, corridor_bounds)
            if not shadow.is_empty:
                blocking_zones.append(shadow)

        if not blocking_zones:
            return corridor

        # Subtract all blocking zones from corridor
        all_blocking = unary_union(blocking_zones)
        all_blocking = make_valid(all_blocking)

        result = corridor.difference(all_blocking)
        result = make_valid(result)

        # Handle MultiPolygon result - keep all reachable parts
        if isinstance(result, MultiPolygon):
            parts = [p for p in result.geoms if isinstance(p, Polygon) and p.area > 0]
            if parts:
                result = unary_union(parts)
            else:
                result = Polygon()

        return result if isinstance(result, Polygon) else Polygon()

    except Exception as e:
        logger.warning(f"Error clipping corridor: {e}")
        return corridor


def separate_obstacles_by_type(
    shadow_result: dict,
    obstacles_utm: list[tuple[Polygon, float, str, int]],
    num_structures: int,
) -> tuple[list, list]:
    """
    Separate shadow-adjusted holes into structure and depth lists.

    The shadow calculation processes all obstacles together for correct shadowing,
    but the cascade needs them separated by type.

    Args:
        shadow_result: Output from compute_shadow_adjusted_holes()
        obstacles_utm: Original obstacle list with type info
        num_structures: Number of structure obstacles (first N in list)

    Returns:
        (struct_holes, depth_holes) - each as [leg][direction][type_idx]
    """
    effective_holes = shadow_result['effective_holes']
    num_legs = len(effective_holes)
    num_dirs = len(effective_holes[0]) if num_legs > 0 else 0

    struct_holes = []
    depth_holes = []

    for leg_idx in range(num_legs):
        leg_struct = []
        leg_depth = []

        for dir_idx in range(num_dirs):
            dir_struct = []
            dir_depth = []

            for obs_idx, (geom, value, obs_type, orig_idx) in enumerate(obstacles_utm):
                hole_val = effective_holes[leg_idx][dir_idx][obs_idx]

                if obs_type == 'structure':
                    # Map to original structure index
                    dir_struct.append((orig_idx, hole_val))
                else:
                    # Map to original depth index
                    dir_depth.append((orig_idx, hole_val))

            # Sort by original index and extract values
            dir_struct.sort(key=lambda x: x[0])
            dir_depth.sort(key=lambda x: x[0])

            leg_struct.append([v for _, v in dir_struct])
            leg_depth.append([v for _, v in dir_depth])

        struct_holes.append(leg_struct)
        depth_holes.append(leg_depth)

    return struct_holes, depth_holes


def blend_with_pdf_holes(
    shadow_holes: list[list[list[float]]],
    pdf_holes: list[list[list[float]]],
    blend_factor: float = 0.0,
) -> list[list[list[float]]]:
    """
    Blend shadow-adjusted holes with PDF-based probability holes.

    This allows combining:
    - Shadow-adjusted: Accurate geometric blocking (hard limit)
    - PDF-based: Accounts for lateral distribution weighting

    The shadow value acts as an upper bound - you can't hit an obstacle
    that's completely shadowed, regardless of PDF weighting.

    Args:
        shadow_holes: Shadow-adjusted holes [leg][direction][object]
        pdf_holes: PDF-based probability holes [leg][direction][object]
        blend_factor: 0.0 = use shadow as cap on PDF, 1.0 = pure shadow

    Returns:
        Blended probability holes [leg][direction][object]
    """
    blended = []

    for leg_idx in range(len(shadow_holes)):
        leg_blended = []
        for dir_idx in range(len(shadow_holes[leg_idx])):
            dir_blended = []
            for obs_idx in range(len(shadow_holes[leg_idx][dir_idx])):
                shadow_val = shadow_holes[leg_idx][dir_idx][obs_idx]

                # Get PDF value if available
                pdf_val = 0.0
                if (pdf_holes and
                    leg_idx < len(pdf_holes) and
                    dir_idx < len(pdf_holes[leg_idx]) and
                    obs_idx < len(pdf_holes[leg_idx][dir_idx])):
                    pdf_val = pdf_holes[leg_idx][dir_idx][obs_idx]

                # Shadow acts as upper bound - can't exceed geometric possibility
                # Blend between PDF (weighted by distribution) and shadow (geometric)
                if blend_factor >= 1.0:
                    blended_val = shadow_val
                elif blend_factor <= 0.0:
                    # Use PDF but cap at shadow value
                    blended_val = min(shadow_val, pdf_val)
                else:
                    # Weighted blend, still capped at shadow
                    weighted = pdf_val * (1 - blend_factor) + shadow_val * blend_factor
                    blended_val = min(shadow_val, weighted)

                dir_blended.append(max(0.0, min(1.0, blended_val)))
            leg_blended.append(dir_blended)
        blended.append(leg_blended)

    return blended


def get_direction_index(angle_deg: float) -> int:
    """
    Convert compass angle to direction index (0-7).

    Args:
        angle_deg: Compass angle in degrees (0=N, 45=NW, 90=W, etc.)

    Returns:
        Direction index 0-7
    """
    # Normalize to 0-360
    angle = angle_deg % 360

    # Map to index (each direction spans 45°)
    # 0° (N) → 0, 45° (NW) → 1, 90° (W) → 2, etc.
    return int(angle / 45) % 8


def direction_index_to_angle(dir_idx: int) -> float:
    """
    Convert direction index to compass angle.

    Args:
        dir_idx: Direction index 0-7

    Returns:
        Compass angle in degrees
    """
    return (dir_idx * 45) % 360
