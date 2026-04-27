# -*- coding: utf-8 -*-
"""
Coordinate system utilities for drift corridor calculations.

Handles CRS transformations and compass-to-cartesian conversions.
"""

import numpy as np
from pyproj import Transformer, CRS


def get_utm_crs(lon: float, lat: float) -> CRS:
    """
    Get the appropriate UTM CRS for a given lon/lat coordinate.

    Args:
        lon: Longitude in degrees (-180 to 180)
        lat: Latitude in degrees (-90 to 90)

    Returns:
        pyproj CRS object for the appropriate UTM zone
    """
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    return CRS(f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84")


def transform_geometry(geom, from_crs: CRS, to_crs: CRS):
    """
    Transform a Shapely geometry between coordinate systems.

    Args:
        geom: Shapely geometry object
        from_crs: Source coordinate reference system
        to_crs: Target coordinate reference system

    Returns:
        Transformed Shapely geometry
    """
    from shapely.ops import transform
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return transform(transformer.transform, geom)


def compass_to_vector(angle_deg: float, distance: float) -> tuple[float, float]:
    """
    Convert nautical-compass angle + distance to a (dx, dy) vector in UTM.

    Uses the standard nautical compass convention (clockwise from North):
        0° = North = +Y
        45° = NorthEast = +X, +Y
        90° = East = +X
        135° = SouthEast = +X, -Y
        180° = South = -Y
        225° = SouthWest = -X, -Y
        270° = West = -X
        315° = NorthWest = -X, +Y

    UTM convention: +X = East, +Y = North.

    Matches :func:`drifting.engine.compass_to_math_deg`
    (``math_angle = (90 - compass_deg) % 360``), which is the canonical
    compass/math conversion used throughout the drifting model.

    Args:
        angle_deg: Compass angle in degrees (0=N, 90=E, 180=S, 270=W).
        distance: Distance in metres.

    Returns:
        ``(dx, dy)`` tuple in UTM coordinates.
    """
    # Spec 0° (N) -> Math 90° (+Y)
    # Spec 90° (E) -> Math 0° (+X)
    # Spec 180° (S) -> Math 270° (-Y)
    # Spec 270° (W) -> Math 180° (-X)
    math_angle = 90.0 - angle_deg
    rad = np.radians(math_angle)
    dx = np.cos(rad) * distance
    dy = np.sin(rad) * distance
    return (dx, dy)
