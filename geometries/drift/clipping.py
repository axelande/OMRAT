# -*- coding: utf-8 -*-
"""
Corridor clipping functions for obstacle intersection.

Clips corridors at obstacles by creating and subtracting blocking zones,
and filters to keep only reachable parts from the upwind edge.
"""

from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

from .shadow import create_obstacle_shadow, extract_polygons


def clip_corridor_at_obstacles(corridor: Polygon, obstacles: list,
                               drift_angle_deg: float,
                               log_prefix: str = "") -> Polygon:
    """
    Clip the corridor at obstacles using blocking zones.

    The approach:
    1. For each obstacle that intersects the corridor
    2. Create a "blocking zone" that extends from the obstacle to the corridor boundary
       in the drift direction (using quad-based sweep)
    3. Subtract all blocking zones from the corridor

    This implements the key principle: ships at lateral positions that hit an obstacle
    will ground there and cannot drift further. Ships at positions that miss the
    obstacle can drift past it.

    Args:
        corridor: The corridor polygon (in UTM)
        obstacles: List of (polygon, value) tuples
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)
        log_prefix: Prefix for log messages (empty string to disable logging)

    Returns:
        Clipped corridor polygon
    """
    if not obstacles:
        return corridor

    corridor = make_valid(corridor)
    if corridor.is_empty:
        return corridor

    corridor_bounds = corridor.bounds
    blocking_zones = []
    intersecting_count = 0

    # Import logging only if needed
    if log_prefix:
        from qgis.core import QgsMessageLog, Qgis

    for poly, value in obstacles:
        try:
            poly = make_valid(poly)
            if poly.is_empty:
                continue

            if corridor.intersects(poly):
                intersecting_count += 1
                # Get intersection with corridor
                intersection = corridor.intersection(poly)
                intersection = make_valid(intersection)

                if intersection.is_empty:
                    continue

                # Extract polygon parts and create blocking zones
                parts = extract_polygons(intersection)

                for part in parts:
                    if part.is_empty:
                        continue
                    # Create blocking zone from this obstacle part
                    blocking = create_obstacle_shadow(part, drift_angle_deg, corridor_bounds)
                    if not blocking.is_empty:
                        blocking_zones.append(blocking)

        except Exception as e:
            if log_prefix:
                QgsMessageLog.logMessage(
                    f"{log_prefix}Error processing obstacle: {e}",
                    "OMRAT", Qgis.Warning
                )

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}Found {intersecting_count} obstacles, created {len(blocking_zones)} blocking zones",
            "OMRAT", Qgis.Info
        )

    if not blocking_zones:
        return corridor

    # Subtract all blocking zones from corridor
    try:
        all_blocking = unary_union(blocking_zones)
        all_blocking = make_valid(all_blocking)

        result = corridor.difference(all_blocking)
        result = make_valid(result)

        if log_prefix and not result.is_empty:
            reduction = (corridor.area - result.area) / corridor.area * 100
            QgsMessageLog.logMessage(
                f"{log_prefix}Blocked {reduction:.1f}% of corridor area",
                "OMRAT", Qgis.Info
            )

        return result

    except Exception as e:
        if log_prefix:
            QgsMessageLog.logMessage(
                f"{log_prefix}Error in clip_corridor_at_obstacles: {e}",
                "OMRAT", Qgis.Warning
            )
        return corridor


def split_corridor_by_anchor_zone(
        clipped: Polygon,
        anchor_zone: Polygon,
        drift_angle_deg: float,
        corridor_bounds: tuple[float, float, float, float],
) -> tuple[Polygon, Polygon]:
    """
    Split a clipped corridor into blue (anchorable) and green (deep) zones.

    The corridor starts as green (deep water) on the upwind side.  Once
    the drift path enters an anchor zone the corridor turns blue, and
    everything behind that anchor zone (in the drift direction) stays
    blue — the ship has entered anchorable territory.

    Concretely:
      1. blue  = clipped ∩ anchor_zone  (actual anchorable cells)
      2. green = clipped − anchor_zone  (deep water cells)
      3. Create shadows from blue parts in the drift direction.
      4. Green areas inside the shadow (= behind blue) are converted
         to blue, not removed.

    Args:
        clipped: The corridor polygon after obstacle clipping (UTM).
        anchor_zone: Union of all depth cells where anchoring is
            possible (depth < anchor_threshold), in the same CRS.
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E).
        corridor_bounds: (minx, miny, maxx, maxy) of the *original*
            (unclipped) corridor – used to size shadows.

    Returns:
        (blue_zone, green_zone) polygons.
    """
    if clipped.is_empty:
        return Polygon(), Polygon()

    clipped = make_valid(clipped)
    anchor_zone = make_valid(anchor_zone)

    try:
        blue = make_valid(clipped.intersection(anchor_zone))
    except Exception:
        blue = Polygon()

    try:
        green = make_valid(clipped.difference(anchor_zone))
    except Exception:
        green = clipped

    # If there are no blue parts there is nothing to shadow.
    if blue.is_empty or green.is_empty:
        return blue, green

    # Build shadow zones behind each blue polygon part.
    blue_parts = extract_polygons(blue)
    shadow_zones = []
    for part in blue_parts:
        if part.is_empty:
            continue
        shadow = create_obstacle_shadow(part, drift_angle_deg, corridor_bounds)
        if not shadow.is_empty:
            shadow_zones.append(shadow)

    if not shadow_zones:
        return blue, green

    try:
        all_shadows = make_valid(unary_union(shadow_zones))

        # Green areas behind blue are converted to blue, not removed.
        green_behind_blue = make_valid(green.intersection(all_shadows))
        green = make_valid(green.difference(all_shadows))
        if not green_behind_blue.is_empty:
            blue = make_valid(unary_union([blue, green_behind_blue]))
    except Exception:
        pass  # keep blue/green as-is on failure

    return blue, green


def keep_reachable_part(clipped: Polygon, original_corridor: Polygon,
                        drift_angle_deg: float, log_prefix: str = "") -> Polygon:
    """
    Keep only the part of the clipped corridor that is reachable from the upwind edge.

    Ships start from the upwind edge and drift downwind. If obstacles create
    separate regions, we only want the region(s) connected to the starting edge.

    Args:
        clipped: The corridor after subtracting obstacles
        original_corridor: The original corridor before clipping
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)
        log_prefix: Prefix for log messages (empty string to disable logging)

    Returns:
        The reachable part of the corridor
    """
    if clipped.is_empty:
        return clipped

    # Extract all polygon parts
    parts = extract_polygons(clipped)
    if len(parts) <= 1:
        return clipped  # Single part, nothing to filter

    # Import logging only if needed
    if log_prefix:
        from qgis.core import QgsMessageLog, Qgis

    # Determine the "upwind" edge of the original corridor
    # Upwind is opposite to drift direction
    upwind_edge = _get_upwind_edge(original_corridor.bounds, drift_angle_deg)

    # Keep parts that touch the upwind edge
    reachable_parts = []
    for part in parts:
        if part.intersects(upwind_edge):
            reachable_parts.append(part)

    if not reachable_parts:
        # If no parts touch upwind edge, keep largest part as fallback
        if log_prefix:
            QgsMessageLog.logMessage(
                f"{log_prefix}No parts touch upwind edge, keeping largest",
                "OMRAT", Qgis.Warning
            )
        return max(parts, key=lambda p: p.area)

    if log_prefix and len(reachable_parts) < len(parts):
        QgsMessageLog.logMessage(
            f"{log_prefix}Kept {len(reachable_parts)}/{len(parts)} parts touching upwind edge",
            "OMRAT", Qgis.Info
        )

    return make_valid(unary_union(reachable_parts))


def _get_upwind_edge(bounds: tuple[float, float, float, float],
                     drift_angle_deg: float) -> Polygon:
    """
    Create a thin rectangle along the upwind edge of the corridor bounds.

    Args:
        bounds: (minx, miny, maxx, maxy) of the corridor
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)

    Returns:
        Thin box polygon along the upwind edge
    """
    minx, miny, maxx, maxy = bounds
    margin = max(maxx - minx, maxy - miny) * 0.01  # Small margin for intersection test

    angle = drift_angle_deg % 360

    # For each drift direction, upwind is the opposite edge
    if angle < 22.5 or angle >= 337.5:  # N drift -> upwind is South
        return box(minx - margin, miny - margin, maxx + margin, miny + margin)
    elif 22.5 <= angle < 67.5:  # NW drift -> upwind is SE
        return box(maxx - margin, miny - margin, maxx + margin, miny + margin)
    elif 67.5 <= angle < 112.5:  # W drift -> upwind is East
        return box(maxx - margin, miny - margin, maxx + margin, maxy + margin)
    elif 112.5 <= angle < 157.5:  # SW drift -> upwind is NE
        return box(maxx - margin, maxy - margin, maxx + margin, maxy + margin)
    elif 157.5 <= angle < 202.5:  # S drift -> upwind is North
        return box(minx - margin, maxy - margin, maxx + margin, maxy + margin)
    elif 202.5 <= angle < 247.5:  # SE drift -> upwind is NW
        return box(minx - margin, maxy - margin, minx + margin, maxy + margin)
    elif 247.5 <= angle < 292.5:  # E drift -> upwind is West
        return box(minx - margin, miny - margin, minx + margin, maxy + margin)
    else:  # NE drift (292.5 to 337.5) -> upwind is SW
        return box(minx - margin, miny - margin, minx + margin, miny + margin)
