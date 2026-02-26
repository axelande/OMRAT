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
    Convert compass angle and distance to (dx, dy) vector.

    Uses the spec/nautical convention (counter-clockwise from North):
        0° = North = +Y
        45° = NorthWest = -X, +Y
        90° = West = -X
        135° = SouthWest = -X, -Y
        180° = South = -Y
        225° = SouthEast = +X, -Y
        270° = East = +X
        315° = NorthEast = +X, +Y

    UTM convention: +X=East, +Y=North

    Args:
        angle_deg: Compass angle in degrees (0=N, 90=W, 180=S, 270=E)
        distance: Distance in meters

    Returns:
        (dx, dy) tuple in UTM coordinates
    """
    # Convert spec convention to math angle:
    # Spec 0° (N) = Math 90° (+Y)
    # Spec 90° (W) = Math 180° (-X)
    # Spec 180° (S) = Math 270° (-Y)
    # Spec 270° (E) = Math 0° (+X)
    # Formula: math_angle = 90 + angle_deg (counter-clockwise rotation)
    math_angle = 90 + angle_deg
    rad = np.radians(math_angle)
    dx = np.cos(rad) * distance
    dy = np.sin(rad) * distance
    return (dx, dy)
