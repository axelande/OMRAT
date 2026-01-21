# -*- coding: utf-8 -*-
"""
Obstacle shadow/blocking zone creation using quad-based sweep algorithm.

The quad-based sweep preserves the exact contour of obstacles when creating
shadows, unlike bounding-box or convex-hull approaches that fill in gaps.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.affinity import translate

from .coordinates import compass_to_vector


def create_obstacle_shadow(obstacle: Polygon, drift_angle_deg: float,
                           corridor_bounds: tuple[float, float, float, float]) -> Polygon:
    """
    Create a shadow/blocking zone behind an obstacle in the drift direction.

    Uses a quad-based sweep approach that preserves the obstacle's exact contour:
    1. Translate the obstacle in the drift direction (far enough to reach corridor boundary)
    2. Create quads connecting each edge of the original obstacle to the translated obstacle
    3. Union all parts (original + quads + translated) to create the sweep shape

    This creates a shadow that:
    - Has the EXACT obstacle contour on the front edge (facing against drift)
    - Extends properly in the drift direction
    - Works correctly for concave obstacles (gaps in concave parts remain open)

    Args:
        obstacle: The obstacle polygon (already intersected with corridor)
        drift_angle_deg: Compass angle (0=N, 90=W, 180=S, 270=E)
        corridor_bounds: (minx, miny, maxx, maxy) of the corridor

    Returns:
        Polygon representing the blocked area (obstacle + shadow behind it)
    """
    if obstacle.is_empty:
        return Polygon()

    corr_minx, corr_miny, corr_maxx, corr_maxy = corridor_bounds

    # Calculate extrusion distance (enough to reach corridor boundary and beyond)
    corridor_diagonal = np.sqrt((corr_maxx - corr_minx)**2 + (corr_maxy - corr_miny)**2)
    extrude_dist = corridor_diagonal * 2

    # Get drift vector (direction ships are moving)
    dx, dy = compass_to_vector(drift_angle_deg, extrude_dist)

    try:
        # Translate the obstacle in the drift direction
        far_obstacle = translate(obstacle, xoff=dx, yoff=dy)

        # Get coordinates for quad construction
        original_coords = list(obstacle.exterior.coords)[:-1]  # Exclude closing point
        translated_coords = list(far_obstacle.exterior.coords)[:-1]

        n = len(original_coords)
        if n < 3:
            # Fallback for degenerate obstacles
            return unary_union([obstacle, far_obstacle]).convex_hull

        # Create quads connecting each edge of original to corresponding edge of translated
        quads = _create_edge_quads(original_coords, translated_coords)

        # Union the original obstacle, all quads, and far obstacle
        all_parts = [obstacle, far_obstacle] + quads
        shadow = unary_union(all_parts)

        if not shadow.is_valid:
            shadow = make_valid(shadow)

        # If result is a collection, extract and union the polygons
        if hasattr(shadow, 'geoms'):
            polys = [g for g in shadow.geoms if isinstance(g, Polygon) and g.area > 0]
            if polys:
                shadow = unary_union(polys)
            else:
                # Fallback to convex hull
                shadow = unary_union([obstacle, far_obstacle]).convex_hull

        return shadow

    except Exception:
        # Fallback to convex hull approach
        try:
            far_obstacle = translate(obstacle, xoff=dx, yoff=dy)
            return unary_union([obstacle, far_obstacle]).convex_hull
        except Exception:
            return obstacle


def _create_edge_quads(original_coords: list, translated_coords: list) -> list[Polygon]:
    """
    Create quad polygons connecting edges of original and translated obstacles.

    Each quad connects one edge of the original obstacle to the corresponding
    edge of the translated obstacle, forming a tube-like structure.

    Args:
        original_coords: List of (x, y) tuples for original obstacle vertices
        translated_coords: List of (x, y) tuples for translated obstacle vertices

    Returns:
        List of valid quad polygons
    """
    n = len(original_coords)
    quads = []

    for i in range(n):
        j = (i + 1) % n
        # Quad from edge i-j of original to edge i-j of translated
        quad = Polygon([
            original_coords[i],
            original_coords[j],
            translated_coords[j],
            translated_coords[i]
        ])
        if quad.is_valid and quad.area > 0:
            quads.append(quad)
        elif not quad.is_valid:
            valid_quad = make_valid(quad)
            if not valid_quad.is_empty:
                quads.append(valid_quad)

    return quads


def extract_polygons(geom) -> list[Polygon]:
    """
    Extract all Polygon geometries from any geometry type.

    Handles Polygon, MultiPolygon, and GeometryCollection inputs.

    Args:
        geom: Any Shapely geometry object

    Returns:
        List of non-empty Polygon objects
    """
    polygons = []

    if geom is None or geom.is_empty:
        return polygons

    if isinstance(geom, Polygon):
        polygons.append(geom)
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            if not p.is_empty:
                polygons.append(p)
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            polygons.extend(extract_polygons(g))

    return polygons
