# -*- coding: utf-8 -*-
"""
Drift Corridor Generation for QGIS

Creates drift corridors for shipping legs based on:
- Base surface (leg × distribution width)
- 8 wind directions (N, NE, E, SE, S, SW, W, NW)
- Projection distance based on repair time probability
- Shadows/holes from depth and structure obstacles
"""

import numpy as np
from scipy import stats
from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection, MultiLineString
from shapely.ops import unary_union, transform
from shapely.validation import make_valid
from shapely.affinity import translate
from pyproj import Transformer, CRS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omrat import OMRAT


def get_utm_crs(lon: float, lat: float) -> CRS:
    """Get the appropriate UTM CRS for a given lon/lat coordinate."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    return CRS(f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84")


def transform_geometry(geom, from_crs: CRS, to_crs: CRS):
    """Transform a Shapely geometry between coordinate systems."""
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return transform(transformer.transform, geom)

# Wind directions (8 compass directions)
# Angle 0 = East, 90 = North (standard math convention)
DIRECTIONS = {
    'E': 0,
    'NE': 45,
    'N': 90,
    'NW': 135,
    'W': 180,
    'SW': 225,
    'S': 270,
    'SE': 315,
}


def get_projection_distance(repair_params: dict, drift_speed_ms: float,
                            target_prob: float = 1e-3,
                            max_distance: float = 50000) -> float:
    """
    Calculate the projection distance where prob_not_repaired drops to target_prob.

    Args:
        repair_params: Lognormal distribution parameters (std, loc, scale, use_lognormal)
        drift_speed_ms: Drift speed in m/s
        target_prob: Target probability for P(not repaired), default 1e-3 (0.1%)
        max_distance: Maximum distance cap in meters, default 50km

    Returns:
        Projection distance in meters (capped at max_distance)
    """
    # Sanity check on drift speed
    if drift_speed_ms <= 0 or drift_speed_ms > 10:  # Max ~20 knots
        drift_speed_ms = 1.0  # Default ~2 knots

    if repair_params.get('use_lognormal', True):
        try:
            std_val = repair_params.get('std', 0.95)
            loc_val = repair_params.get('loc', 0.2)
            scale_val = repair_params.get('scale', 0.85)

            # Validate parameters
            if std_val <= 0 or scale_val <= 0:
                return min(10000, max_distance)  # Default 10km

            dist = stats.lognorm(std_val, loc_val, scale_val)

            # Find drift_time where CDF = 1 - target_prob
            # Use a reasonable target to avoid extreme values
            cdf_target = min(1 - target_prob, 0.999)
            drift_time_hours = dist.ppf(cdf_target)

            # Sanity check - drift time should be reasonable (< 48 hours max)
            if drift_time_hours > 48 or drift_time_hours < 0 or not np.isfinite(drift_time_hours):
                return min(10000, max_distance)

            # Convert to distance
            distance_m = drift_time_hours * 3600 * drift_speed_ms

            # Final bounds check
            if not np.isfinite(distance_m) or distance_m < 0:
                return min(10000, max_distance)

            return min(distance_m, max_distance)
        except Exception:
            return min(10000, max_distance)
    else:
        return min(10000, max_distance)


def get_distribution_width(std: float, coverage: float = 0.99) -> float:
    """
    Calculate the width that covers 'coverage' fraction of a normal distribution.

    For 99% coverage: width = 2 * Z_0.995 * std ≈ 2 * 2.576 * std
    """
    z = stats.norm.ppf((1 + coverage) / 2)
    return 2 * z * std


def create_base_surface(leg: LineString, half_width: float) -> Polygon:
    """
    Create the base distribution surface around the leg.
    This is leg_length × distribution_width.
    """
    coords = np.array(leg.coords)
    start, end = coords[0], coords[-1]

    leg_vec = end - start
    leg_length = np.linalg.norm(leg_vec)
    if leg_length == 0:
        return Polygon()
    leg_dir = leg_vec / leg_length
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    p1 = start - half_width * perp_dir
    p2 = start + half_width * perp_dir
    p3 = end + half_width * perp_dir
    p4 = end - half_width * perp_dir

    return Polygon([p1, p2, p3, p4])


def create_projected_corridor(leg: LineString, half_width: float,
                              drift_angle_deg: float, projection_dist: float) -> Polygon:
    """
    Project the base surface in the drift direction.

    The corridor is formed by:
    - The base surface (leg × distribution width)
    - Extended in the drift direction by projection_dist
    """
    coords = np.array(leg.coords)
    start, end = coords[0], coords[-1]

    leg_vec = end - start
    leg_length = np.linalg.norm(leg_vec)
    if leg_length == 0:
        return Polygon()
    leg_dir = leg_vec / leg_length
    perp_to_leg = np.array([-leg_dir[1], leg_dir[0]])

    # Drift direction
    drift_angle_rad = np.radians(drift_angle_deg)
    drift_dir = np.array([np.cos(drift_angle_rad), np.sin(drift_angle_rad)])
    drift_vec = drift_dir * projection_dist

    # Base surface corners
    b1 = start - half_width * perp_to_leg
    b2 = start + half_width * perp_to_leg
    b3 = end + half_width * perp_to_leg
    b4 = end - half_width * perp_to_leg

    # Projected surface corners
    p1 = b1 + drift_vec
    p2 = b2 + drift_vec
    p3 = b3 + drift_vec
    p4 = b4 + drift_vec

    # Create union and convex hull
    base_poly = Polygon([b1, b2, b3, b4])
    projected_poly = Polygon([p1, p2, p3, p4])

    corridor = unary_union([base_poly, projected_poly]).convex_hull
    return make_valid(corridor)


def get_blocking_line_for_obstacle(obstacle: Polygon, drift_angle_deg: float,
                                    corridor_bounds: tuple) -> MultiLineString | None:
    """
    Get the blocking lines for an obstacle - vertical/horizontal/diagonal lines from
    the front edge vertices of the obstacle extending in the drift direction.

    For curvy/irregular obstacles (like S-shaped depth contours), this finds ALL
    vertices on the "front edge" facing against the drift and creates a blocking
    line from each one.

    For N drift: Find all vertices at local minima (southernmost points of each "bump")
                 and draw vertical lines northward from each.

    Args:
        obstacle: The obstacle polygon
        drift_angle_deg: Drift direction in degrees (0=E, 90=N, etc.)
        corridor_bounds: (minx, miny, maxx, maxy) of the corridor

    Returns:
        MultiLineString with the blocking lines, or None if not found
    """
    if obstacle.is_empty:
        return None

    try:
        obstacle_valid = make_valid(obstacle)
        if obstacle_valid.is_empty:
            return None

        # For multipolygon results from make_valid, get the largest polygon
        if isinstance(obstacle_valid, MultiPolygon):
            obstacle_valid = max(obstacle_valid.geoms, key=lambda g: g.area)

        if not hasattr(obstacle_valid, 'exterior'):
            return None

        corr_minx, corr_miny, corr_maxx, corr_maxy = corridor_bounds

        # Normalize angle to 0-360
        angle = drift_angle_deg % 360

        # Get all exterior coordinates
        coords = list(obstacle_valid.exterior.coords)

        # Find "front edge" vertices - these are vertices where the obstacle
        # protrudes toward the direction OPPOSITE to drift (where ships would hit first)
        # For N drift, we want vertices at the SOUTH side (local y minima)

        lines = []

        if 67.5 <= angle < 112.5:
            # N drift (90°): Find vertices at local y minima (south-facing points)
            # Draw vertical lines northward from each
            front_vertices = _find_front_vertices_n(coords)
            for x, y in front_vertices:
                lines.append(LineString([(x, y), (x, corr_maxy)]))

        elif 112.5 <= angle < 157.5:
            # NW drift (135°): Find SE-facing vertices, draw NW diagonal lines
            front_vertices = _find_front_vertices_nw(coords)
            for x, y in front_vertices:
                dist_to_top = corr_maxy - y
                lines.append(LineString([(x, y), (x - dist_to_top, corr_maxy)]))

        elif 157.5 <= angle < 202.5:
            # W drift (180°): Find vertices at local x maxima (east-facing points)
            # Draw horizontal lines westward from each
            front_vertices = _find_front_vertices_w(coords)
            for x, y in front_vertices:
                lines.append(LineString([(x, y), (corr_minx, y)]))

        elif 202.5 <= angle < 247.5:
            # SW drift (225°): Find NE-facing vertices, draw SW diagonal lines
            front_vertices = _find_front_vertices_sw(coords)
            for x, y in front_vertices:
                dist_to_bottom = y - corr_miny
                lines.append(LineString([(x, y), (x - dist_to_bottom, corr_miny)]))

        elif 247.5 <= angle < 292.5:
            # S drift (270°): Find vertices at local y maxima (north-facing points)
            # Draw vertical lines southward from each
            front_vertices = _find_front_vertices_s(coords)
            for x, y in front_vertices:
                lines.append(LineString([(x, y), (x, corr_miny)]))

        elif 292.5 <= angle < 337.5:
            # SE drift (315°): Find NW-facing vertices, draw SE diagonal lines
            front_vertices = _find_front_vertices_se(coords)
            for x, y in front_vertices:
                dist_to_bottom = y - corr_miny
                lines.append(LineString([(x, y), (x + dist_to_bottom, corr_miny)]))

        elif angle >= 337.5 or angle < 22.5:
            # E drift (0°): Find vertices at local x minima (west-facing points)
            # Draw horizontal lines eastward from each
            front_vertices = _find_front_vertices_e(coords)
            for x, y in front_vertices:
                lines.append(LineString([(x, y), (corr_maxx, y)]))

        else:
            # NE drift (45°): Find SW-facing vertices, draw NE diagonal lines
            # 22.5 <= angle < 67.5
            front_vertices = _find_front_vertices_ne(coords)
            for x, y in front_vertices:
                dist_to_top = corr_maxy - y
                lines.append(LineString([(x, y), (x + dist_to_top, corr_maxy)]))

        if not lines:
            return None

        return MultiLineString(lines)

    except Exception:
        return None


def _find_front_vertices_n(coords: list) -> list:
    """Find vertices that are local y minima (south-facing points for N drift)."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    front = []
    # Use bounding box south edge as baseline - find all vertices near it
    ys = [c[1] for c in coords]
    min_y = min(ys)
    max_y = max(ys)
    threshold = (max_y - min_y) * 0.1  # 10% tolerance

    # Find vertices that are at or near the southern edge
    for i, (x, y) in enumerate(coords[:-1]):  # Skip last (same as first)
        if y <= min_y + threshold:
            front.append((x, y))

    # If no vertices found, use the absolute minimum
    if not front:
        min_idx = ys.index(min_y)
        front.append(coords[min_idx])

    return front


def _find_front_vertices_s(coords: list) -> list:
    """Find vertices that are local y maxima (north-facing points for S drift)."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    front = []
    ys = [c[1] for c in coords]
    max_y = max(ys)
    min_y = min(ys)
    threshold = (max_y - min_y) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if y >= max_y - threshold:
            front.append((x, y))

    if not front:
        max_idx = ys.index(max_y)
        front.append(coords[max_idx])

    return front


def _find_front_vertices_e(coords: list) -> list:
    """Find vertices that are local x minima (west-facing points for E drift)."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    front = []
    xs = [c[0] for c in coords]
    min_x = min(xs)
    max_x = max(xs)
    threshold = (max_x - min_x) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if x <= min_x + threshold:
            front.append((x, y))

    if not front:
        min_idx = xs.index(min_x)
        front.append(coords[min_idx])

    return front


def _find_front_vertices_w(coords: list) -> list:
    """Find vertices that are local x maxima (east-facing points for W drift)."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    front = []
    xs = [c[0] for c in coords]
    max_x = max(xs)
    min_x = min(xs)
    threshold = (max_x - min_x) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if x >= max_x - threshold:
            front.append((x, y))

    if not front:
        max_idx = xs.index(max_x)
        front.append(coords[max_idx])

    return front


def _find_front_vertices_ne(coords: list) -> list:
    """Find vertices that face SW (for NE drift) - low x and low y."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    # For NE drift, front faces SW: minimize (x + y)
    front = []
    values = [c[0] + c[1] for c in coords]
    min_val = min(values)
    max_val = max(values)
    threshold = (max_val - min_val) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if (x + y) <= min_val + threshold:
            front.append((x, y))

    if not front:
        min_idx = values.index(min_val)
        front.append(coords[min_idx])

    return front


def _find_front_vertices_nw(coords: list) -> list:
    """Find vertices that face SE (for NW drift) - high x and low y."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    # For NW drift, front faces SE: maximize x, minimize y -> maximize (x - y)
    front = []
    values = [c[0] - c[1] for c in coords]
    max_val = max(values)
    min_val = min(values)
    threshold = (max_val - min_val) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if (x - y) >= max_val - threshold:
            front.append((x, y))

    if not front:
        max_idx = values.index(max_val)
        front.append(coords[max_idx])

    return front


def _find_front_vertices_se(coords: list) -> list:
    """Find vertices that face NW (for SE drift) - low x and high y."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    # For SE drift, front faces NW: minimize x, maximize y -> minimize (x - y)
    front = []
    values = [c[0] - c[1] for c in coords]
    min_val = min(values)
    max_val = max(values)
    threshold = (max_val - min_val) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if (x - y) <= min_val + threshold:
            front.append((x, y))

    if not front:
        min_idx = values.index(min_val)
        front.append(coords[min_idx])

    return front


def _find_front_vertices_sw(coords: list) -> list:
    """Find vertices that face NE (for SW drift) - high x and high y."""
    if len(coords) < 3:
        return [(coords[0][0], coords[0][1])] if coords else []

    # For SW drift, front faces NE: maximize (x + y)
    front = []
    values = [c[0] + c[1] for c in coords]
    max_val = max(values)
    min_val = min(values)
    threshold = (max_val - min_val) * 0.1

    for i, (x, y) in enumerate(coords[:-1]):
        if (x + y) >= max_val - threshold:
            front.append((x, y))

    if not front:
        max_idx = values.index(max_val)
        front.append(coords[max_idx])

    return front


def get_blocking_line(obstacle: Polygon, drift_angle_deg: float) -> LineString | None:
    """
    Get the "front edge" of an obstacle - the edge facing AGAINST the drift direction.

    This is a simplified version that just returns the obstacle's bounding box edge.
    For the full blocking line that spans the corridor, use get_blocking_line_for_obstacle.

    Args:
        obstacle: The obstacle polygon
        drift_angle_deg: Drift direction in degrees (0=E, 90=N, etc.)

    Returns:
        LineString representing the blocking edge, or None if not found
    """
    if obstacle.is_empty:
        return None

    try:
        obstacle_valid = make_valid(obstacle)
        if obstacle_valid.is_empty:
            return None

        if isinstance(obstacle_valid, MultiPolygon):
            obstacle_valid = max(obstacle_valid.geoms, key=lambda g: g.area)

        if not hasattr(obstacle_valid, 'exterior'):
            return None

        minx, miny, maxx, maxy = obstacle_valid.bounds
        angle = drift_angle_deg % 360

        if 45 <= angle < 135:
            return LineString([(minx, miny), (maxx, miny)])
        elif 135 <= angle < 225:
            return LineString([(maxx, miny), (maxx, maxy)])
        elif 225 <= angle < 315:
            return LineString([(minx, maxy), (maxx, maxy)])
        else:
            return LineString([(minx, miny), (minx, maxy)])

    except Exception:
        return None


def create_blocking_shadow(obstacle, drift_angle_deg: float,
                           projection_dist: float, corridor: Polygon):
    """
    Create a shadow that blocks everything behind the obstacle.

    This approach creates a shadow by sweeping/extruding the obstacle in the drift
    direction. IMPORTANT: We do NOT use convex hull because for large irregular
    obstacles (like depth contours), convex hull would create a massive area that
    blocks far more than intended.

    Args:
        obstacle: The obstacle polygon
        drift_angle_deg: Drift direction in degrees (0=E, 90=N, etc.)
        projection_dist: How far the shadow extends
        corridor: The corridor being clipped (used to determine extent)

    Returns:
        Polygon representing the blocked area behind the obstacle
    """
    # Use the sweep approach which preserves the actual obstacle shape
    return create_shadow_behind_obstacle_sweep(obstacle, drift_angle_deg, projection_dist)


def create_shadow_behind_obstacle_sweep(obstacle, drift_angle_deg: float,
                                         projection_dist: float):
    """
    Create shadow by extending obstacle in drift direction with STRAIGHT edges.

    For each polygon part, creates a rectangular blocking area from the obstacle's
    front edge to the projection distance in the drift direction. This produces
    clean straight blocking lines, not jagged edges.

    For N drift: Create a box from (minx, front_y) to (maxx, front_y + projection_dist)
    """
    from shapely.geometry import box

    if obstacle.is_empty:
        return Polygon()

    try:
        obstacle_valid = make_valid(obstacle)
        if obstacle_valid.is_empty:
            return Polygon()

        angle = drift_angle_deg % 360

        # Get all polygons
        if isinstance(obstacle_valid, Polygon):
            polys = [obstacle_valid]
        elif isinstance(obstacle_valid, MultiPolygon):
            polys = list(obstacle_valid.geoms)
        else:
            # Handle GeometryCollection
            polys = []
            if hasattr(obstacle_valid, 'geoms'):
                for g in obstacle_valid.geoms:
                    if isinstance(g, Polygon) and not g.is_empty:
                        polys.append(g)
                    elif isinstance(g, MultiPolygon):
                        polys.extend(p for p in g.geoms if not p.is_empty)
            if not polys:
                return Polygon()

        shadow_parts = []

        for poly in polys:
            if poly.is_empty:
                continue

            obs_minx, obs_miny, obs_maxx, obs_maxy = poly.bounds

            # Create a rectangular shadow extending from the obstacle in drift direction
            if 67.5 <= angle < 112.5:  # N drift
                # Shadow extends from obstacle's south edge northward
                shadow_box = box(obs_minx, obs_miny, obs_maxx, obs_miny + projection_dist)
            elif 247.5 <= angle < 292.5:  # S drift
                shadow_box = box(obs_minx, obs_maxy - projection_dist, obs_maxx, obs_maxy)
            elif angle >= 337.5 or angle < 22.5:  # E drift
                shadow_box = box(obs_minx, obs_miny, obs_minx + projection_dist, obs_maxy)
            elif 157.5 <= angle < 202.5:  # W drift
                shadow_box = box(obs_maxx - projection_dist, obs_miny, obs_maxx, obs_maxy)
            elif 22.5 <= angle < 67.5:  # NE drift
                # Diagonal: create a parallelogram-like shape using box approximation
                # Use bounding box extended in NE direction
                shadow_box = box(obs_minx, obs_miny,
                                obs_maxx + projection_dist * 0.707,
                                obs_maxy + projection_dist * 0.707)
            elif 112.5 <= angle < 157.5:  # NW drift
                shadow_box = box(obs_minx - projection_dist * 0.707, obs_miny,
                                obs_maxx,
                                obs_maxy + projection_dist * 0.707)
            elif 202.5 <= angle < 247.5:  # SW drift
                shadow_box = box(obs_minx - projection_dist * 0.707,
                                obs_miny - projection_dist * 0.707,
                                obs_maxx, obs_maxy)
            else:  # SE drift
                shadow_box = box(obs_minx, obs_miny - projection_dist * 0.707,
                                obs_maxx + projection_dist * 0.707, obs_maxy)

            shadow_parts.append(shadow_box)

        if not shadow_parts:
            return Polygon()

        shadow = unary_union(shadow_parts)
        shadow = make_valid(shadow)

        return shadow

    except Exception:
        try:
            return make_valid(obstacle)
        except Exception:
            return obstacle


def create_shadow_behind_obstacle(obstacle: Polygon, drift_angle_deg: float,
                                   projection_dist: float,
                                   include_obstacle: bool = True) -> Polygon:
    """
    Create the "shadow" cast by an obstacle in the drift direction.

    The shadow represents the area BEHIND the obstacle (in drift direction)
    that ships cannot reach because they would hit the obstacle first.

    Args:
        obstacle: The obstacle polygon
        drift_angle_deg: Drift direction in degrees (0=E, 90=N, etc.)
        projection_dist: How far the shadow extends
        include_obstacle: If True, shadow includes the obstacle. If False,
                         shadow is only the area behind the obstacle.

    NOTE: Does NOT use convex hull to preserve the obstacle's shape.
    Uses a sweep approach with multiple intermediate steps.
    """
    if obstacle.is_empty:
        return Polygon()

    drift_angle_rad = np.radians(drift_angle_deg)
    dx = np.cos(drift_angle_rad) * projection_dist
    dy = np.sin(drift_angle_rad) * projection_dist

    try:
        obstacle_valid = make_valid(obstacle)
        if obstacle_valid.is_empty:
            return Polygon()

        # Create shadow by sweeping the obstacle in the drift direction
        # Use multiple intermediate positions to create a smooth sweep
        # that preserves the obstacle's shape (no convex hull!)
        num_steps = 10
        step_dx = dx / num_steps
        step_dy = dy / num_steps

        sweep_parts = [obstacle_valid]
        for i in range(1, num_steps + 1):
            shifted = translate(obstacle_valid, xoff=step_dx * i, yoff=step_dy * i)
            sweep_parts.append(shifted)

        # Union all the swept positions - this creates a shadow that
        # follows the obstacle's shape, not a convex hull
        shadow = unary_union(sweep_parts)
        shadow = make_valid(shadow)

        if not include_obstacle:
            # Remove the obstacle itself from the shadow
            shadow = shadow.difference(obstacle_valid)
            shadow = make_valid(shadow)

        return shadow

    except Exception:
        try:
            return make_valid(obstacle)
        except Exception:
            return obstacle


def clip_corridor_at_obstacles(corridor: Polygon, obstacles: list,
                               drift_angle_deg: float, leg_centroid: tuple,
                               log_prefix: str = "") -> Polygon:
    """
    Clip the corridor at obstacles using a sweep/shadow approach.

    For each obstacle that intersects the corridor, create a shadow that extends
    from the obstacle in the drift direction. The corridor is then clipped by
    subtracting these shadows.

    The shadow includes the obstacle itself (where ships ground) and the area
    behind it (unreachable because ships would hit the obstacle first).

    Args:
        corridor: The corridor polygon
        obstacles: List of (polygon, value) tuples for obstacles
        drift_angle_deg: Drift direction in degrees
        leg_centroid: (x, y) centroid of the leg (origin point for drift)
        log_prefix: Prefix for log messages

    Returns:
        Corridor with area behind obstacles blocked
    """
    from qgis.core import QgsMessageLog, Qgis
    from shapely.geometry import box

    if not obstacles:
        if log_prefix:
            QgsMessageLog.logMessage(
                f"{log_prefix}No obstacles provided - returning original corridor",
                "OMRAT", Qgis.Warning
            )
        return corridor

    corridor_valid = make_valid(corridor)
    if corridor_valid.is_empty:
        return corridor

    # Get corridor bounds
    corr_minx, corr_miny, corr_maxx, corr_maxy = corridor_valid.bounds
    corridor_diagonal = np.sqrt((corr_maxx - corr_minx)**2 + (corr_maxy - corr_miny)**2)

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}Starting clip: {len(obstacles)} obstacles, corridor bounds=[{corr_minx:.0f}, {corr_miny:.0f}, {corr_maxx:.0f}, {corr_maxy:.0f}]",
            "OMRAT", Qgis.Info
        )

    # Collect all obstacle parts that intersect the corridor
    obstacle_parts = []
    intersecting_count = 0

    for poly, value in obstacles:
        try:
            poly_valid = make_valid(poly)
            if poly_valid.is_empty:
                continue

            # Check if obstacle intersects corridor
            obstacle_in_corridor = corridor_valid.intersection(poly_valid)
            obstacle_in_corridor = make_valid(obstacle_in_corridor)

            if obstacle_in_corridor.is_empty:
                continue

            intersecting_count += 1

            # Extract polygon parts from intersection (may be MultiPolygon)
            if isinstance(obstacle_in_corridor, Polygon):
                if not obstacle_in_corridor.is_empty:
                    obstacle_parts.append(obstacle_in_corridor)
            elif isinstance(obstacle_in_corridor, MultiPolygon):
                for p in obstacle_in_corridor.geoms:
                    if not p.is_empty:
                        obstacle_parts.append(p)
            elif isinstance(obstacle_in_corridor, GeometryCollection):
                for geom in obstacle_in_corridor.geoms:
                    if isinstance(geom, Polygon) and not geom.is_empty:
                        obstacle_parts.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        for p in geom.geoms:
                            if not p.is_empty:
                                obstacle_parts.append(p)

        except Exception as e:
            if log_prefix:
                QgsMessageLog.logMessage(
                    f"{log_prefix}Error processing obstacle: {e}",
                    "OMRAT", Qgis.Warning
                )
            continue

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}{intersecting_count}/{len(obstacles)} obstacles intersect corridor, {len(obstacle_parts)} parts total",
            "OMRAT", Qgis.Info
        )

    if not obstacle_parts:
        return corridor

    # Union all obstacles
    try:
        all_obstacles = unary_union(obstacle_parts)
        all_obstacles = make_valid(all_obstacles)
    except Exception as e:
        if log_prefix:
            QgsMessageLog.logMessage(f"{log_prefix}Error unioning obstacles: {e}", "OMRAT", Qgis.Warning)
        return corridor

    if all_obstacles.is_empty:
        return corridor

    if log_prefix:
        obs_bounds = all_obstacles.bounds
        QgsMessageLog.logMessage(
            f"{log_prefix}Combined obstacles bounds=({obs_bounds[0]:.0f}, {obs_bounds[1]:.0f}, {obs_bounds[2]:.0f}, {obs_bounds[3]:.0f}), area={all_obstacles.area:.0f}m²",
            "OMRAT", Qgis.Info
        )

    # Create shadow by sweeping obstacles in drift direction
    shadow = create_shadow_behind_obstacle_sweep(all_obstacles, drift_angle_deg, corridor_diagonal)

    if shadow.is_empty:
        if log_prefix:
            QgsMessageLog.logMessage(f"{log_prefix}Shadow is empty - returning original corridor", "OMRAT", Qgis.Info)
        return corridor

    # Clip shadow to corridor
    shadow_clipped = shadow.intersection(corridor_valid)
    shadow_clipped = make_valid(shadow_clipped)

    if shadow_clipped.is_empty:
        return corridor

    # Create blocking lines for debug output
    blocking_lines = _create_blocking_lines_from_obstacle_boundary(
        all_obstacles, drift_angle_deg, corridor_valid.bounds, log_prefix
    )

    # Write debug output
    if log_prefix:
        from shapely import wkt as shapely_wkt
        import tempfile
        import os

        direction_name = log_prefix.strip().replace(" ", "_").replace(":", "")

        # Write obstacle intersection
        if not all_obstacles.is_empty:
            obs_wkt = shapely_wkt.dumps(all_obstacles, rounding_precision=2)
            obs_file = os.path.join(tempfile.gettempdir(), f"obstacle_intersection_{direction_name}.wkt")
            with open(obs_file, 'w') as f:
                f.write(obs_wkt)
            QgsMessageLog.logMessage(
                f"{log_prefix}OBSTACLE INTERSECTION WKT written to: {obs_file}",
                "OMRAT", Qgis.Info
            )

        # Write blocking lines
        if blocking_lines and not blocking_lines.is_empty:
            bl_wkt = shapely_wkt.dumps(blocking_lines, rounding_precision=2)
            bl_file = os.path.join(tempfile.gettempdir(), f"blocking_line_{direction_name}.wkt")
            with open(bl_file, 'w') as f:
                f.write(bl_wkt)
            QgsMessageLog.logMessage(
                f"{log_prefix}BLOCKING LINE WKT written to: {bl_file}",
                "OMRAT", Qgis.Info
            )

        # Write shadow for debugging
        if not shadow_clipped.is_empty:
            shadow_wkt = shapely_wkt.dumps(shadow_clipped, rounding_precision=2)
            shadow_file = os.path.join(tempfile.gettempdir(), f"shadow_{direction_name}.wkt")
            with open(shadow_file, 'w') as f:
                f.write(shadow_wkt)
            QgsMessageLog.logMessage(
                f"{log_prefix}SHADOW WKT written to: {shadow_file}",
                "OMRAT", Qgis.Info
            )

    # Subtract shadow from corridor
    try:
        result = corridor_valid.difference(shadow_clipped)
        result = make_valid(result)

        if log_prefix:
            if not result.is_empty:
                result_bounds = result.bounds
                reduction = (corridor_valid.area - result.area) / corridor_valid.area * 100
                QgsMessageLog.logMessage(
                    f"{log_prefix}Clipped corridor: {reduction:.1f}% reduction, result bounds=[{result_bounds[0]:.0f}, {result_bounds[1]:.0f}, {result_bounds[2]:.0f}, {result_bounds[3]:.0f}]",
                    "OMRAT", Qgis.Info
                )
            else:
                QgsMessageLog.logMessage(
                    f"{log_prefix}Corridor completely blocked!",
                    "OMRAT", Qgis.Warning
                )

        # If result is MultiPolygon, keep only the part containing/closest to leg
        if isinstance(result, (MultiPolygon, GeometryCollection)):
            from shapely.geometry import Point
            leg_point = Point(leg_centroid[0], leg_centroid[1])

            polygons = []
            for geom in result.geoms:
                if isinstance(geom, Polygon) and not geom.is_empty:
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    for p in geom.geoms:
                        if isinstance(p, Polygon) and not p.is_empty:
                            polygons.append(p)

            if not polygons:
                return Polygon()

            # Find polygon containing or closest to leg
            containing = None
            closest = None
            closest_dist = float('inf')

            for poly in polygons:
                if poly.contains(leg_point):
                    containing = poly
                    break
                dist = poly.distance(leg_point)
                if dist < closest_dist:
                    closest_dist = dist
                    closest = poly

            result = containing if containing else closest
            if result is None:
                return Polygon()

        return result

    except Exception as e:
        if log_prefix:
            QgsMessageLog.logMessage(f"{log_prefix}Error clipping corridor: {e}", "OMRAT", Qgis.Warning)
        return corridor


def _create_blocking_lines_from_obstacle_boundary(
    obstacles: Polygon | MultiPolygon,
    drift_angle_deg: float,
    corridor_bounds: tuple,
    log_prefix: str = ""
) -> MultiLineString | None:
    """
    Create blocking lines at the boundary edges where obstacles meet open corridor.

    The key insight: A blocking line should be created where the corridor transitions
    from "open" (no obstacle) to "blocked" (obstacle present). For a drifting ship,
    this is the point where part of the corridor becomes unreachable because ships
    at that lateral position would hit the obstacle.

    For N drift with obstacles partially covering the corridor:
    - If obstacle covers the WEST half, ships on the west hit it and stop
    - Ships on the east can drift past (around) the obstacle
    - The blocking line is at the EAST edge of the obstacle - marking where
      the "blocked" zone ends and "open" zone begins

    This produces blocking lines only at lateral boundaries where coverage changes.
    """
    from qgis.core import QgsMessageLog, Qgis

    if obstacles.is_empty:
        return None

    corr_minx, corr_miny, corr_maxx, corr_maxy = corridor_bounds
    angle = drift_angle_deg % 360

    # Get all polygons from obstacles
    if isinstance(obstacles, Polygon):
        polys = [obstacles]
    elif isinstance(obstacles, MultiPolygon):
        polys = list(obstacles.geoms)
    else:
        return None

    # Filter to valid polygons
    polys = [p for p in polys if not p.is_empty and hasattr(p, 'exterior')]
    if not polys:
        return None

    lines = []

    # For N drift (90 degrees): blocking lines are vertical
    # We need to find lateral (X) positions where obstacle coverage CHANGES
    if 67.5 <= angle < 112.5:  # N drift
        # For each polygon, get its lateral extent (min_x, max_x)
        # A blocking line goes at each unique lateral boundary

        # Collect all lateral boundaries with their front Y position
        boundaries = []  # [(x, front_y, 'west'|'east'), ...]

        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            # West edge of this polygon - where it starts
            boundaries.append((poly_minx, poly_miny, 'west'))
            # East edge of this polygon - where it ends
            boundaries.append((poly_maxx, poly_miny, 'east'))

        # Group by X position and keep the one with the frontmost (lowest) Y
        x_to_best = {}
        for x, front_y, side in boundaries:
            key = round(x, 1)  # Group nearby X values
            if key not in x_to_best or front_y < x_to_best[key][1]:
                x_to_best[key] = (x, front_y, side)

        # Create vertical blocking lines from front_y to corridor top
        for x, front_y, side in x_to_best.values():
            line = LineString([(x, front_y), (x, corr_maxy)])
            lines.append(line)

    elif 247.5 <= angle < 292.5:  # S drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            boundaries.append((poly_minx, poly_maxy, 'west'))
            boundaries.append((poly_maxx, poly_maxy, 'east'))

        x_to_best = {}
        for x, front_y, side in boundaries:
            key = round(x, 1)
            if key not in x_to_best or front_y > x_to_best[key][1]:
                x_to_best[key] = (x, front_y, side)

        for x, front_y, side in x_to_best.values():
            line = LineString([(x, front_y), (x, corr_miny)])
            lines.append(line)

    elif angle >= 337.5 or angle < 22.5:  # E drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            boundaries.append((poly_miny, poly_minx, 'south'))
            boundaries.append((poly_maxy, poly_minx, 'north'))

        y_to_best = {}
        for y, front_x, side in boundaries:
            key = round(y, 1)
            if key not in y_to_best or front_x < y_to_best[key][1]:
                y_to_best[key] = (y, front_x, side)

        for y, front_x, side in y_to_best.values():
            line = LineString([(front_x, y), (corr_maxx, y)])
            lines.append(line)

    elif 157.5 <= angle < 202.5:  # W drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            boundaries.append((poly_miny, poly_maxx, 'south'))
            boundaries.append((poly_maxy, poly_maxx, 'north'))

        y_to_best = {}
        for y, front_x, side in boundaries:
            key = round(y, 1)
            if key not in y_to_best or front_x > y_to_best[key][1]:
                y_to_best[key] = (y, front_x, side)

        for y, front_x, side in y_to_best.values():
            line = LineString([(front_x, y), (corr_minx, y)])
            lines.append(line)

    elif 22.5 <= angle < 67.5:  # NE drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            # For NE, perpendicular is along (x - y) direction
            boundaries.append((poly_minx - poly_miny, poly_minx, poly_miny, 'sw'))
            boundaries.append((poly_maxx - poly_miny, poly_maxx, poly_miny, 'se'))

        key_to_best = {}
        for perp, x, y, side in boundaries:
            key = round(perp, 1)
            front_dist = x + y  # Distance along drift direction
            if key not in key_to_best or front_dist < key_to_best[key][1]:
                key_to_best[key] = (perp, front_dist, x, y, side)

        for perp, front_dist, x, y, side in key_to_best.values():
            dist = corr_maxy - y
            line = LineString([(x, y), (x + dist, corr_maxy)])
            lines.append(line)

    elif 112.5 <= angle < 157.5:  # NW drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            # For NW, perpendicular is along (x + y) direction
            boundaries.append((poly_minx + poly_miny, poly_minx, poly_miny, 'sw'))
            boundaries.append((poly_maxx + poly_miny, poly_maxx, poly_miny, 'se'))

        key_to_best = {}
        for perp, x, y, side in boundaries:
            key = round(perp, 1)
            front_dist = -x + y  # Distance along NW drift direction
            if key not in key_to_best or front_dist < key_to_best[key][1]:
                key_to_best[key] = (perp, front_dist, x, y, side)

        for perp, front_dist, x, y, side in key_to_best.values():
            dist = corr_maxy - y
            line = LineString([(x, y), (x - dist, corr_maxy)])
            lines.append(line)

    elif 202.5 <= angle < 247.5:  # SW drift
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            boundaries.append((poly_minx - poly_maxy, poly_minx, poly_maxy, 'nw'))
            boundaries.append((poly_maxx - poly_maxy, poly_maxx, poly_maxy, 'ne'))

        key_to_best = {}
        for perp, x, y, side in boundaries:
            key = round(perp, 1)
            front_dist = -(x + y)  # Distance along SW drift direction
            if key not in key_to_best or front_dist < key_to_best[key][1]:
                key_to_best[key] = (perp, front_dist, x, y, side)

        for perp, front_dist, x, y, side in key_to_best.values():
            dist = y - corr_miny
            line = LineString([(x, y), (x - dist, corr_miny)])
            lines.append(line)

    else:  # SE drift (292.5 <= angle < 337.5)
        boundaries = []
        for poly in polys:
            poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds
            boundaries.append((poly_minx + poly_maxy, poly_minx, poly_maxy, 'nw'))
            boundaries.append((poly_maxx + poly_maxy, poly_maxx, poly_maxy, 'ne'))

        key_to_best = {}
        for perp, x, y, side in boundaries:
            key = round(perp, 1)
            front_dist = x - y  # Distance along SE drift direction
            if key not in key_to_best or front_dist < key_to_best[key][1]:
                key_to_best[key] = (perp, front_dist, x, y, side)

        for perp, front_dist, x, y, side in key_to_best.values():
            dist = y - corr_miny
            line = LineString([(x, y), (x + dist, corr_miny)])
            lines.append(line)

    if not lines:
        return None

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}Created {len(lines)} blocking lines at obstacle boundaries",
            "OMRAT", Qgis.Info
        )

    return MultiLineString(lines)


def apply_obstacle_shadows(corridor: Polygon, obstacles: list,
                           drift_angle_deg: float, projection_dist: float,
                           log_prefix: str = "") -> Polygon:
    """
    Apply obstacle shadows to the corridor.

    The shadow represents the area "behind" an obstacle in the drift direction.
    Ships drifting toward the obstacle will ground/stop, so areas behind the
    obstacle (in the drift direction) are unreachable.

    Args:
        corridor: The corridor polygon
        obstacles: List of (polygon, value) tuples for obstacles that block
        drift_angle_deg: Drift direction in degrees
        projection_dist: Projection distance in meters
        log_prefix: Prefix for log messages (for debugging)

    Returns:
        Corridor with shadows subtracted (may have holes or be MultiPolygon)
    """
    from qgis.core import QgsMessageLog, Qgis

    shadows = []
    intersecting_count = 0
    corridor_valid = make_valid(corridor)
    # Use buffer(0) to fix any topology issues
    corridor_valid = corridor_valid.buffer(0)

    if log_prefix:
        corridor_bounds = corridor_valid.bounds
        QgsMessageLog.logMessage(
            f"{log_prefix}Corridor bounds: ({corridor_bounds[0]:.1f}, {corridor_bounds[1]:.1f}, {corridor_bounds[2]:.1f}, {corridor_bounds[3]:.1f}), area={corridor_valid.area:.1f}m²",
            "OMRAT", Qgis.Info
        )

    for poly, value in obstacles:
        try:
            # Fix potential topology issues in obstacle
            poly_valid = make_valid(poly)
            if poly_valid.is_empty:
                if log_prefix:
                    QgsMessageLog.logMessage(
                        f"{log_prefix}Obstacle value={value} is empty after make_valid, skipping",
                        "OMRAT", Qgis.Warning
                    )
                continue
            poly_valid = poly_valid.buffer(0)

            # Log obstacle bounds for debugging
            if log_prefix:
                obs_bounds = poly_valid.bounds
                QgsMessageLog.logMessage(
                    f"{log_prefix}Checking obstacle value={value}, bounds=({obs_bounds[0]:.1f}, {obs_bounds[1]:.1f}, {obs_bounds[2]:.1f}, {obs_bounds[3]:.1f}), area={poly_valid.area:.1f}m²",
                    "OMRAT", Qgis.Info
                )

            # First, get the intersection of obstacle with corridor
            # This is the actual obstacle area that affects this corridor
            obstacle_in_corridor = corridor_valid.intersection(poly_valid)
            obstacle_in_corridor = make_valid(obstacle_in_corridor)

            if obstacle_in_corridor.is_empty:
                if log_prefix:
                    QgsMessageLog.logMessage(
                        f"{log_prefix}Obstacle value={value} does NOT intersect corridor",
                        "OMRAT", Qgis.Info
                    )
                continue

            intersecting_count += 1

            # Extract polygons from the intersection result (may be Polygon, MultiPolygon, or GeometryCollection)
            obstacle_polys: list[Polygon] = []
            if isinstance(obstacle_in_corridor, Polygon):
                obstacle_polys = [obstacle_in_corridor]
            elif isinstance(obstacle_in_corridor, MultiPolygon):
                obstacle_polys = list(obstacle_in_corridor.geoms)
            elif isinstance(obstacle_in_corridor, GeometryCollection):
                for geom in obstacle_in_corridor.geoms:
                    if isinstance(geom, Polygon) and not geom.is_empty:
                        obstacle_polys.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        obstacle_polys.extend(g for g in geom.geoms if not g.is_empty)

            if not obstacle_polys:
                if log_prefix:
                    QgsMessageLog.logMessage(
                        f"{log_prefix}Obstacle value={value} intersection has no polygons (type={type(obstacle_in_corridor).__name__})",
                        "OMRAT", Qgis.Warning
                    )
                continue

            # Create shadow from ONLY the portion of obstacle that's in the corridor
            # This ensures shadows only extend from where the obstacle actually blocks
            # Union all obstacle polygons and create a single shadow
            obstacle_union = unary_union(obstacle_polys) if len(obstacle_polys) > 1 else obstacle_polys[0]
            obstacle_union = make_valid(obstacle_union)

            # Log the intersection details for debugging
            if log_prefix:
                intersection_bounds = obstacle_union.bounds
                QgsMessageLog.logMessage(
                    f"{log_prefix}Obstacle value={value} intersection bounds=({intersection_bounds[0]:.1f}, {intersection_bounds[1]:.1f}, {intersection_bounds[2]:.1f}, {intersection_bounds[3]:.1f}), area={obstacle_union.area:.1f}m²",
                    "OMRAT", Qgis.Info
                )

            # Ensure we have a Polygon for shadow creation (buffer(0) normalizes geometry)
            if not isinstance(obstacle_union, Polygon):
                obstacle_union = obstacle_union.buffer(0)
                if isinstance(obstacle_union, MultiPolygon):
                    # Use convex hull as approximation for shadow
                    obstacle_union = obstacle_union.convex_hull
            if not isinstance(obstacle_union, Polygon) or obstacle_union.is_empty:
                continue
            shadow = create_shadow_behind_obstacle(obstacle_union, drift_angle_deg, projection_dist)
            shadow = make_valid(shadow)

            # Clip shadow to corridor bounds
            shadow = shadow.intersection(corridor_valid)
            if not shadow.is_empty:
                shadows.append(make_valid(shadow))
                QgsMessageLog.logMessage(
                    f"{log_prefix}Obstacle value={value} INTERSECTS corridor (intersection area={obstacle_in_corridor.area:.1f}m²), shadow area={shadow.area:.1f}m²",
                    "OMRAT", Qgis.Info
                )
            else:
                if log_prefix:
                    QgsMessageLog.logMessage(
                        f"{log_prefix}Obstacle value={value} intersects but shadow is empty after clipping",
                        "OMRAT", Qgis.Warning
                    )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"{log_prefix}Error processing obstacle value={value}: {e}",
                "OMRAT", Qgis.Warning
            )

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}{intersecting_count}/{len(obstacles)} obstacles intersect corridor, {len(shadows)} shadows created",
            "OMRAT", Qgis.Info
        )

    if not shadows:
        return corridor

    corridor = make_valid(corridor)

    try:
        all_shadows = unary_union(shadows)
        all_shadows = make_valid(all_shadows)
        result = corridor.difference(all_shadows)

        if result.is_empty:
            return Polygon()

        return make_valid(result)
    except Exception:
        return corridor


class DriftCorridorGenerator:
    """Generates drift corridors for QGIS layers."""

    def __init__(self, plugin: 'OMRAT'):
        self.plugin = plugin
        self._progress_callback = None
        self._cancelled = False
        # Pre-collected data for background thread execution
        self._precollected_data: dict | None = None

    def clear_cache(self) -> None:
        """Clear any cached data.

        Call this when the project is unloaded or data changes to ensure
        fresh data is used for the next corridor generation.
        """
        self._precollected_data = None
        self._cancelled = False
        self._progress_callback = None

    def diagnose_data(self, depth_threshold: float, height_threshold: float) -> str:
        """Diagnose the current data state and return a report.

        This method helps identify why obstacles might not be affecting corridors.
        Call this from main thread before running analysis.
        """
        from qgis.core import QgsMessageLog, Qgis
        from shapely import wkt as shapely_wkt

        report = []
        report.append("=== DRIFT CORRIDOR DIAGNOSTIC REPORT ===")
        report.append(f"Depth threshold: {depth_threshold}m")
        report.append(f"Height threshold: {height_threshold}m")

        # Check legs
        legs = self.get_legs_from_routes()
        report.append(f"\n--- LEGS ({len(legs)} found) ---")
        for i, leg in enumerate(legs):
            bounds = leg.bounds
            report.append(f"  Leg {i}: bounds=({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f})")

        # Check depth obstacles
        depth_obstacles = self.get_depth_obstacles(depth_threshold)
        report.append(f"\n--- DEPTH OBSTACLES ({len(depth_obstacles)} found with threshold {depth_threshold}m) ---")

        # Check table directly with detailed parsing
        table = self.plugin.main_widget.twDepthList
        report.append(f"  Table has {table.rowCount()} rows")
        for row in range(table.rowCount()):
            depth_item = table.item(row, 1)
            wkt_item = table.item(row, 2)
            depth_text = depth_item.text() if depth_item else "None"

            wkt_info = "None"
            if wkt_item and wkt_item.text():
                wkt_text = wkt_item.text()
                wkt_len = len(wkt_text)
                try:
                    parsed = shapely_wkt.loads(wkt_text)
                    wkt_info = f"len={wkt_len}, type={parsed.geom_type}, valid={parsed.is_valid}, area={parsed.area:.6f}"
                except Exception as e:
                    wkt_info = f"len={wkt_len}, PARSE ERROR: {e}"

            report.append(f"  Row {row}: depth='{depth_text}', wkt={wkt_info}")

        for i, (poly, depth) in enumerate(depth_obstacles):
            bounds = poly.bounds
            report.append(f"  Obstacle {i}: depth={depth}m, bounds=({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}), area={poly.area:.6f}")

        # Check structure obstacles
        struct_obstacles = self.get_structure_obstacles(height_threshold)
        report.append(f"\n--- STRUCTURE OBSTACLES ({len(struct_obstacles)} found) ---")
        for i, (poly, height) in enumerate(struct_obstacles):
            bounds = poly.bounds
            report.append(f"  Obstacle {i}: height={height}m, bounds=({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f})")

        # Check if obstacles overlap with legs (in WGS84)
        report.append("\n--- OVERLAP CHECK (WGS84) ---")
        all_obstacles = depth_obstacles + struct_obstacles
        for leg_idx, leg in enumerate(legs):
            leg_buffer = leg.buffer(0.01)  # ~1km buffer in degrees

            overlapping = 0
            for poly, _ in all_obstacles:
                if leg_buffer.intersects(poly):
                    overlapping += 1

            report.append(f"  Leg {leg_idx}: {overlapping}/{len(all_obstacles)} obstacles within ~1km")

        report.append("\n=== END DIAGNOSTIC REPORT ===")

        report_text = "\n".join(report)
        QgsMessageLog.logMessage(report_text, "OMRAT", Qgis.Info)
        return report_text

    def set_progress_callback(self, callback) -> None:
        """Set a callback function for progress updates.

        Args:
            callback: Function that takes (completed, total, message) and returns bool.
                      Return False to cancel the operation.
        """
        self._progress_callback = callback

    def _report_progress(self, completed: int, total: int, message: str) -> bool:
        """Report progress and check for cancellation.

        Returns:
            True to continue, False to cancel
        """
        if self._progress_callback:
            result = self._progress_callback(completed, total, message)
            if result is False:
                self._cancelled = True
                return False
        return True

    def precollect_data(self, depth_threshold: float, height_threshold: float) -> None:
        """Pre-collect all data from Qt widgets (must be called from main thread).

        This method reads all necessary data from UI widgets and stores it
        for later use in generate_corridors_from_data() which can run in a background thread.
        """
        from qgis.core import QgsMessageLog, Qgis

        self._precollected_data = {
            'legs': self.get_legs_from_routes(),
            'depth_obstacles': self.get_depth_obstacles(depth_threshold),
            'structure_obstacles': self.get_structure_obstacles(height_threshold),
            'lateral_std': self.get_distribution_std(),
            'repair_params': self.get_repair_params(),
            'drift_speed': self.get_drift_speed_ms(),
        }

        QgsMessageLog.logMessage(
            f"Pre-collected data: {len(self._precollected_data['legs'])} legs, "
            f"{len(self._precollected_data['depth_obstacles'])} depth obstacles, "
            f"{len(self._precollected_data['structure_obstacles'])} structure obstacles",
            "OMRAT", Qgis.Info
        )

    def get_legs_from_routes(self) -> list[LineString]:
        """Extract LineString geometries from route layers."""
        legs = []
        for layer in self.plugin.qgis_geoms.vector_layers:
            for feature in layer.getFeatures():
                geom = feature.geometry()
                if geom and not geom.isNull():
                    # Convert QgsGeometry to Shapely LineString
                    wkt = geom.asWkt()
                    try:
                        from shapely import wkt as shapely_wkt
                        shapely_geom = shapely_wkt.loads(wkt)
                        if isinstance(shapely_geom, LineString):
                            legs.append(shapely_geom)
                    except Exception:
                        pass
        return legs

    def get_depth_obstacles(self, depth_threshold: float) -> list[tuple[Polygon, float]]:
        """Get depth polygons that are shallower than threshold.

        Depth values are read from the twDepthList table (column 1),
        WKT geometries from column 2.

        For depth intervals like "0-10", we use the UPPER bound (10m) as the max depth.
        Areas with max depth <= threshold are considered obstacles (grounding risk).

        For single values that appear to be bin labels (e.g., 0.0, 3.0, 6.0 at 3m intervals),
        we detect the bin width and use (value + bin_width) as the max depth.

        Example: threshold=10m
          - "0-10" interval → max depth 10m ≤ 10m → INCLUDED (grounding risk)
          - "10-20" interval → max depth 20m > 10m → EXCLUDED (deep enough)
          - "9.0" single value with 3m bins → max depth 12m > 10m → EXCLUDED
        """
        from qgis.core import QgsMessageLog, Qgis

        obstacles = []
        table = self.plugin.main_widget.twDepthList

        # First pass: collect all depth values to detect bin width
        depth_values = []
        for row in range(table.rowCount()):
            depth_item = table.item(row, 1)
            if depth_item is None:
                continue
            depth_text = depth_item.text().strip()
            try:
                if '-' in depth_text and not depth_text.startswith('-'):
                    # Interval format - skip for bin detection
                    continue
                elif depth_text.startswith('-'):
                    if '--' in depth_text:
                        continue
                    val = abs(float(depth_text))
                else:
                    val = float(depth_text)
                depth_values.append(val)
            except ValueError:
                continue

        # Detect bin width from sorted unique values
        bin_width = 0.0
        if len(depth_values) >= 2:
            sorted_depths = sorted(set(depth_values))
            if len(sorted_depths) >= 2:
                # Calculate differences between consecutive values
                diffs = [sorted_depths[i+1] - sorted_depths[i] for i in range(len(sorted_depths)-1)]
                # Use the most common difference as bin width
                if diffs:
                    from collections import Counter
                    diff_counts = Counter(round(d, 1) for d in diffs)
                    most_common_diff = diff_counts.most_common(1)[0][0]
                    if most_common_diff > 0:
                        bin_width = most_common_diff
                        QgsMessageLog.logMessage(
                            f"Detected depth bin width: {bin_width}m from values {sorted_depths[:5]}...",
                            "OMRAT", Qgis.Info
                        )

        for row in range(table.rowCount()):
            try:
                # Get depth value from table column 1
                depth_item = table.item(row, 1)
                if depth_item is None:
                    continue
                depth_text = depth_item.text().strip()

                # Handle interval format like "0-10" or "0.0-10.0" or single value "10"
                # Also handle negative values (GEBCO stores depths as negative)
                if '-' in depth_text and not depth_text.startswith('-'):
                    # Interval format: "0-10" or "10-20"
                    parts = depth_text.split('-')
                    # Use the UPPER bound - the maximum depth in this area
                    depth = float(parts[-1])
                elif depth_text.startswith('-'):
                    # Negative single value or negative interval like "-10" or "-20--10"
                    # Convert to positive depth
                    if '--' in depth_text:
                        # Format like "-20--10" means -20 to -10 (depths 10 to 20m)
                        parts = depth_text.split('--')
                        depth = abs(float(parts[0]))  # Use the shallower (less negative)
                    else:
                        # Single negative value - treat as lower bound of bin
                        depth = abs(float(depth_text)) + bin_width
                else:
                    # Single positive value - treat as lower bound of bin
                    # The max depth in this bin is value + bin_width
                    depth = float(depth_text) + bin_width

                QgsMessageLog.logMessage(
                    f"Depth row {row}: '{depth_text}' → max_depth={depth}m (bin_width={bin_width}m), threshold={depth_threshold}m, include={depth <= depth_threshold}",
                    "OMRAT", Qgis.Info
                )

                # Include this area if its max depth <= threshold
                if depth > depth_threshold:
                    continue

                # Get WKT from table column 2
                wkt_item = table.item(row, 2)
                if wkt_item is None:
                    QgsMessageLog.logMessage(
                        f"Depth row {row}: WKT item is None (column 2 empty)", "OMRAT", Qgis.Warning
                    )
                    continue
                wkt = wkt_item.text()
                if not wkt or not wkt.strip():
                    QgsMessageLog.logMessage(
                        f"Depth row {row}: WKT text is empty", "OMRAT", Qgis.Warning
                    )
                    continue

                QgsMessageLog.logMessage(
                    f"Depth row {row}: WKT starts with '{wkt[:50]}...' (len={len(wkt)})", "OMRAT", Qgis.Info
                )

                from shapely import wkt as shapely_wkt
                from shapely.geometry import MultiPolygon
                shapely_geom = shapely_wkt.loads(wkt)

                QgsMessageLog.logMessage(
                    f"Depth row {row}: Parsed geometry type={type(shapely_geom).__name__}, valid={shapely_geom.is_valid}, empty={shapely_geom.is_empty}",
                    "OMRAT", Qgis.Info
                )

                # Handle both Polygon and MultiPolygon
                if hasattr(shapely_geom, 'exterior'):
                    # Single Polygon
                    obstacles.append((shapely_geom, depth))
                    QgsMessageLog.logMessage(
                        f"Added depth obstacle (Polygon): {depth}m", "OMRAT", Qgis.Info
                    )
                elif isinstance(shapely_geom, MultiPolygon):
                    # MultiPolygon - add each polygon separately
                    for poly in shapely_geom.geoms:
                        if hasattr(poly, 'exterior') and not poly.is_empty:
                            obstacles.append((poly, depth))
                    QgsMessageLog.logMessage(
                        f"Added depth obstacle (MultiPolygon with {len(shapely_geom.geoms)} parts): {depth}m", "OMRAT", Qgis.Info
                    )
                else:
                    QgsMessageLog.logMessage(
                        f"Skipped depth row {row}: geometry type {type(shapely_geom).__name__} not supported", "OMRAT", Qgis.Warning
                    )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error parsing depth row {row}: {e}", "OMRAT", Qgis.Warning
                )

        QgsMessageLog.logMessage(
            f"Found {len(obstacles)} depth obstacles with threshold {depth_threshold}m",
            "OMRAT", Qgis.Info
        )
        return obstacles

    def get_structure_obstacles(self, height_threshold: float) -> list[tuple[Polygon, float]]:
        """Get structure polygons that are lower than threshold.

        Height values are read from the twObjectList table (column 1),
        WKT geometries from column 2.
        """
        from qgis.core import QgsMessageLog, Qgis

        obstacles = []
        table = self.plugin.main_widget.twObjectList

        for row in range(table.rowCount()):
            try:
                # Get height value from table column 1
                height_item = table.item(row, 1)
                if height_item is None:
                    continue
                height = float(height_item.text())

                QgsMessageLog.logMessage(
                    f"Structure row {row}: height={height}m, threshold={height_threshold}m, include={height <= height_threshold}",
                    "OMRAT", Qgis.Info
                )

                if height > height_threshold:
                    continue  # Skip if height is greater than threshold

                # Get WKT from table column 2
                wkt_item = table.item(row, 2)
                if wkt_item is None:
                    continue
                wkt = wkt_item.text()

                from shapely import wkt as shapely_wkt
                from shapely.geometry import MultiPolygon
                shapely_geom = shapely_wkt.loads(wkt)

                # Handle both Polygon and MultiPolygon
                if hasattr(shapely_geom, 'exterior'):
                    obstacles.append((shapely_geom, height))
                    QgsMessageLog.logMessage(
                        f"Added structure obstacle (Polygon): {height}m", "OMRAT", Qgis.Info
                    )
                elif isinstance(shapely_geom, MultiPolygon):
                    for poly in shapely_geom.geoms:
                        if hasattr(poly, 'exterior') and not poly.is_empty:
                            obstacles.append((poly, height))
                    QgsMessageLog.logMessage(
                        f"Added structure obstacle (MultiPolygon with {len(shapely_geom.geoms)} parts): {height}m", "OMRAT", Qgis.Info
                    )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error parsing structure row {row}: {e}", "OMRAT", Qgis.Warning
                )

        QgsMessageLog.logMessage(
            f"Found {len(obstacles)} structure obstacles with threshold {height_threshold}m",
            "OMRAT", Qgis.Info
        )
        return obstacles

    def get_distribution_std(self) -> float:
        """Get the lateral distribution standard deviation from routes."""
        # Try to get from distribution settings
        try:
            # Use the first non-zero std from direction 1
            std1 = float(self.plugin.main_widget.leNormStd1_1.text() or 0)
            if std1 > 0:
                return std1
        except (ValueError, AttributeError):
            pass
        # Default fallback
        return 100.0

    def get_repair_params(self) -> dict:
        """Get repair time distribution parameters from drift settings."""
        drift_values = self.plugin.drift_values
        return {
            'use_lognormal': drift_values.get('use_lognormal', 1),
            'std': drift_values.get('std', 0.95),
            'loc': drift_values.get('loc', 0.2),
            'scale': drift_values.get('scale', 0.85),
        }

    def get_drift_speed_ms(self) -> float:
        """Get drift speed in m/s from settings."""
        drift_values = self.plugin.drift_values
        speed_kts = float(drift_values.get('speed', 1.94))
        return speed_kts * 1852.0 / 3600.0

    def generate_corridors(self, depth_threshold: float, height_threshold: float,
                           target_prob: float = 1e-3) -> list[dict]:
        """
        Generate drift corridors for all legs in all 8 directions.

        Args:
            depth_threshold: Depths <= this value create shadows (grounding risk)
            height_threshold: Heights <= this value create shadows (allision risk)
            target_prob: Target probability for projection distance

        Returns:
            List of corridor dicts with keys: direction, leg_index, polygon (in EPSG:4326)
        """
        from qgis.core import QgsMessageLog, Qgis

        # Reset cancellation flag
        self._cancelled = False

        # Use pre-collected data if available (for background thread execution)
        if self._precollected_data is not None:
            legs = self._precollected_data['legs']
            depth_obstacles = self._precollected_data['depth_obstacles']
            structure_obstacles = self._precollected_data['structure_obstacles']
            lateral_std = self._precollected_data['lateral_std']
            repair_params = self._precollected_data['repair_params']
            drift_speed = self._precollected_data['drift_speed']
        else:
            # Collect data now (only safe in main thread)
            legs = self.get_legs_from_routes()
            depth_obstacles = self.get_depth_obstacles(depth_threshold)
            structure_obstacles = self.get_structure_obstacles(height_threshold)
            lateral_std = self.get_distribution_std()
            repair_params = self.get_repair_params()
            drift_speed = self.get_drift_speed_ms()

        if not legs:
            QgsMessageLog.logMessage("No legs found from routes", "OMRAT", Qgis.Warning)
            return []

        # Calculate total work units for progress (legs × 8 directions)
        total_work = len(legs) * len(DIRECTIONS)
        completed_work = 0

        # Report initial progress
        if not self._report_progress(0, total_work, "Initializing..."):
            return []

        # Calculate derived parameters
        half_width = get_distribution_width(lateral_std, 0.99) / 2
        projection_dist = get_projection_distance(repair_params, drift_speed, target_prob)

        # Safety bounds - max 50km projection
        projection_dist = min(projection_dist, 50000)

        QgsMessageLog.logMessage(
            f"Corridor params: half_width={half_width:.1f}m, projection_dist={projection_dist:.1f}m, "
            f"lateral_std={lateral_std:.1f}m, drift_speed={drift_speed:.2f}m/s",
            "OMRAT", Qgis.Info
        )

        QgsMessageLog.logMessage(
            f"Repair params: {repair_params}",
            "OMRAT", Qgis.Info
        )

        QgsMessageLog.logMessage(
            f"Total obstacles: {len(depth_obstacles)} depth + {len(structure_obstacles)} structure = {len(depth_obstacles) + len(structure_obstacles)}",
            "OMRAT", Qgis.Info
        )

        # Log details of depth obstacles to verify threshold filtering
        if depth_obstacles:
            depth_values = [d for _, d in depth_obstacles]
            QgsMessageLog.logMessage(
                f"Depth obstacle values (should all be <= threshold): {sorted(set(depth_values))}",
                "OMRAT", Qgis.Info
            )

        # WGS84 CRS for input/output
        wgs84 = CRS("EPSG:4326")

        corridors = []
        for leg_idx, leg in enumerate(legs):
            # Check for cancellation at start of each leg
            if self._cancelled:
                QgsMessageLog.logMessage("Corridor generation cancelled", "OMRAT", Qgis.Warning)
                return corridors

            # Validate leg coordinates are reasonable WGS84 values
            centroid = leg.centroid
            if not (-180 <= centroid.x <= 180 and -90 <= centroid.y <= 90):
                completed_work += len(DIRECTIONS)
                continue  # Skip invalid leg

            utm_crs = get_utm_crs(centroid.x, centroid.y)

            # Transform leg to UTM (meters)
            try:
                leg_utm = transform_geometry(leg, wgs84, utm_crs)
            except Exception:
                completed_work += len(DIRECTIONS)
                continue

            # Transform obstacles to UTM (once per leg)
            if not self._report_progress(completed_work, total_work,
                                         f"Transforming obstacles for leg {leg_idx + 1}/{len(legs)}..."):
                return corridors

            obstacles_utm = []
            all_obstacles = depth_obstacles + structure_obstacles

            # Log counts for debugging
            QgsMessageLog.logMessage(
                f"Leg {leg_idx}: Processing {len(depth_obstacles)} depth + {len(structure_obstacles)} structure = {len(all_obstacles)} total obstacles",
                "OMRAT", Qgis.Info
            )

            for poly, value in all_obstacles:
                try:
                    poly_utm = transform_geometry(poly, wgs84, utm_crs)
                    # Make the polygon valid if needed
                    if not poly_utm.is_valid:
                        poly_utm = make_valid(poly_utm)
                    if not poly_utm.is_empty:
                        # Handle case where make_valid returns a GeometryCollection
                        if poly_utm.geom_type == 'GeometryCollection':
                            from shapely.geometry import Polygon, MultiPolygon
                            for geom in poly_utm.geoms:
                                if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty:
                                    obstacles_utm.append((geom, value))
                        else:
                            obstacles_utm.append((poly_utm, value))
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Failed to transform obstacle value={value}: {e}", "OMRAT", Qgis.Warning
                    )

            QgsMessageLog.logMessage(
                f"Leg {leg_idx}: {len(obstacles_utm)} obstacles transformed to UTM",
                "OMRAT", Qgis.Info
            )

            # Log leg position for debugging
            leg_bounds = leg_utm.bounds
            QgsMessageLog.logMessage(
                f"Leg {leg_idx} UTM bounds: ({leg_bounds[0]:.1f}, {leg_bounds[1]:.1f}) to ({leg_bounds[2]:.1f}, {leg_bounds[3]:.1f})",
                "OMRAT", Qgis.Info
            )

            for dir_name, angle in DIRECTIONS.items():
                # Check for cancellation
                if self._cancelled:
                    return corridors

                # Report progress
                if not self._report_progress(completed_work, total_work,
                                             f"Leg {leg_idx + 1}/{len(legs)} - {dir_name}"):
                    return corridors

                # Create corridor in UTM (meters)
                corridor_utm = create_projected_corridor(leg_utm, half_width, angle, projection_dist)

                if corridor_utm.is_empty:
                    completed_work += 1
                    continue

                original_area = corridor_utm.area
                corridor_bounds_before = corridor_utm.bounds

                if obstacles_utm:
                    log_prefix = f"Leg {leg_idx} {dir_name}: "
                    # Get leg centroid for clipping reference
                    leg_centroid_utm = leg_utm.centroid
                    leg_centroid = (leg_centroid_utm.x, leg_centroid_utm.y)

                    # Use the new clipping approach instead of shadow-based approach
                    corridor_utm = clip_corridor_at_obstacles(
                        corridor_utm, obstacles_utm, angle, leg_centroid, log_prefix
                    )

                    # Log area change
                    if not corridor_utm.is_empty:
                        area_reduction = (original_area - corridor_utm.area) / original_area * 100
                        QgsMessageLog.logMessage(
                            f"Leg {leg_idx} {dir_name}: corridor area reduced by {area_reduction:.1f}% (from {original_area:.0f}m² to {corridor_utm.area:.0f}m²)",
                            "OMRAT", Qgis.Info
                        )
                        corridor_bounds_after = corridor_utm.bounds
                        QgsMessageLog.logMessage(
                            f"Leg {leg_idx} {dir_name}: bounds before=({corridor_bounds_before[0]:.1f}, {corridor_bounds_before[1]:.1f}, {corridor_bounds_before[2]:.1f}, {corridor_bounds_before[3]:.1f}), after=({corridor_bounds_after[0]:.1f}, {corridor_bounds_after[1]:.1f}, {corridor_bounds_after[2]:.1f}, {corridor_bounds_after[3]:.1f})",
                            "OMRAT", Qgis.Info
                        )

                if corridor_utm.is_empty:
                    completed_work += 1
                    continue

                # Transform back to WGS84
                try:
                    corridor_wgs84 = transform_geometry(corridor_utm, utm_crs, wgs84)
                    corridor_wgs84 = make_valid(corridor_wgs84)

                    # Validate the result is reasonable WGS84
                    if corridor_wgs84.is_empty:
                        completed_work += 1
                        continue

                    bounds = corridor_wgs84.bounds
                    min_lon, min_lat, max_lon, max_lat = bounds

                    # Check bounds are valid WGS84 coordinates
                    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and
                            -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                        completed_work += 1
                        continue  # Invalid coordinates

                    # Check corridor isn't unreasonably large (max ~2 degrees extent)
                    if (max_lon - min_lon) > 2 or (max_lat - min_lat) > 2:
                        completed_work += 1
                        continue  # Too large, likely transformation error

                    corridors.append({
                        'direction': dir_name,
                        'angle': angle,
                        'leg_index': leg_idx,
                        'polygon': corridor_wgs84,
                    })
                except Exception:
                    pass

                completed_work += 1

        # Final progress report
        self._report_progress(total_work, total_work, f"Complete: {len(corridors)} corridors generated")

        # Clear precollected data after use to ensure fresh data on next run
        self._precollected_data = None

        # Log summary of generated corridors
        if corridors:
            north_corridors = [c for c in corridors if c['direction'] == 'N']
            if north_corridors:
                for nc in north_corridors:
                    bounds = nc['polygon'].bounds
                    QgsMessageLog.logMessage(
                        f"Generated N corridor for leg {nc['leg_index']}: lat bounds [{bounds[1]:.4f}, {bounds[3]:.4f}]",
                        "OMRAT", Qgis.Info
                    )

        return corridors
