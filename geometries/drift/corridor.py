# -*- coding: utf-8 -*-
"""
Corridor geometry creation functions.

Creates the base distribution surface and projected drift corridors.
"""

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.validation import make_valid

from .coordinates import compass_to_vector


def create_base_surface(leg: LineString, half_width: float) -> Polygon:
    """
    Create the base distribution surface around the leg.

    This is a rectangle centered on the leg with width = 2 * half_width.
    Represents the area where ships might be located at the start of drift.

    Args:
        leg: Route segment as a LineString (in UTM coordinates)
        half_width: Half the distribution width in meters

    Returns:
        Polygon representing the base surface
    """
    coords = np.array(leg.coords)
    if len(coords) < 2:
        return Polygon()

    start, end = coords[0], coords[-1]
    leg_vec = end - start
    leg_length = np.linalg.norm(leg_vec)

    if leg_length == 0:
        return Polygon()

    # Calculate perpendicular direction
    leg_dir = leg_vec / leg_length
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    # Create rectangle corners
    p1 = start - half_width * perp_dir
    p2 = start + half_width * perp_dir
    p3 = end + half_width * perp_dir
    p4 = end - half_width * perp_dir

    return Polygon([p1, p2, p3, p4])


def create_projected_corridor(leg: LineString, half_width: float,
                              drift_angle_deg: float, projection_dist: float) -> Polygon:
    """
    Project the base surface in the drift direction to create a corridor.

    The corridor represents the area a drifting ship might pass through,
    from initial position (base surface) to maximum drift distance.

    Args:
        leg: The route segment as a LineString (in UTM)
        half_width: Half the distribution width in meters
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)
        projection_dist: How far to project in meters

    Returns:
        Polygon representing the corridor (base + projected area)
    """
    coords = np.array(leg.coords)
    if len(coords) < 2:
        return Polygon()

    start, end = coords[0], coords[-1]
    leg_vec = end - start
    leg_length = np.linalg.norm(leg_vec)

    if leg_length == 0:
        return Polygon()

    leg_dir = leg_vec / leg_length
    perp_to_leg = np.array([-leg_dir[1], leg_dir[0]])

    # Get drift direction vector
    dx, dy = compass_to_vector(drift_angle_deg, projection_dist)
    drift_vec = np.array([dx, dy])

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

    # Create union and convex hull for clean corridor shape
    base_poly = Polygon([b1, b2, b3, b4])
    projected_poly = Polygon([p1, p2, p3, p4])

    corridor = unary_union([base_poly, projected_poly]).convex_hull
    return make_valid(corridor)
