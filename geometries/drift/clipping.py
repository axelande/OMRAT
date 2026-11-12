# -*- coding: utf-8 -*-
"""
Corridor clipping functions for obstacle intersection.

Clips corridors at obstacles by creating and subtracting blocking zones,
and splits them into the anchorable and the deep part.
"""

from shapely.geometry import Polygon
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
        drift_angle_deg: Nautical compass angle (0=N, 90=E, 180=S, 270=W)
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
                    "OMRAT", Qgis.MessageLevel.Warning
                )

    if log_prefix:
        QgsMessageLog.logMessage(
            f"{log_prefix}Found {intersecting_count} obstacles, created {len(blocking_zones)} blocking zones",
            "OMRAT", Qgis.MessageLevel.Info
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
                "OMRAT", Qgis.MessageLevel.Info
            )

        return result

    except Exception as e:
        if log_prefix:
            QgsMessageLog.logMessage(
                f"{log_prefix}Error in clip_corridor_at_obstacles: {e}",
                "OMRAT", Qgis.MessageLevel.Warning
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
        drift_angle_deg: Nautical compass angle (0=N, 90=E, 180=S, 270=W).
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
    except Exception:  # nosec B110 B112
        blue = Polygon()

    try:
        green = make_valid(clipped.difference(anchor_zone))
    except Exception:  # nosec B110 B112
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
    except Exception:  # nosec B110 B112
        pass  # keep blue/green as-is on failure

    return blue, green
