# -*- coding: utf-8 -*-
"""
Drift Corridor Generation for QGIS - Version 2

This module re-exports from the refactored `geometries.drift` package for
backward compatibility. New code should import from `geometries.drift` directly.

Creates drift corridors for shipping legs based on:
- Base surface (leg x distribution width)
- 8 wind directions using nautical/compass convention (N=0, NW=45, W=90, etc.)
- Projection distance based on repair time probability
- Shadows from depth and structure obstacles using quad-based sweep algorithm

Key features:
- Uses nautical/compass convention for angles (0=North, not East)
- Quad-based sweep algorithm that preserves exact obstacle contours
- Properly handles scattered MultiPolygon obstacles without convex hull
"""

# Re-export all public API from the drift package
from geometries.drift.constants import DIRECTIONS
from geometries.drift.coordinates import (
    get_utm_crs,
    transform_geometry,
    compass_to_vector,
)
from geometries.drift.distribution import (
    get_projection_distance,
    get_distribution_width,
)
from geometries.drift.corridor import (
    create_base_surface,
    create_projected_corridor,
)
from geometries.drift.shadow import (
    create_obstacle_shadow,
    extract_polygons,
)
from geometries.drift.clipping import (
    clip_corridor_at_obstacles,
    keep_reachable_part,
)
from geometries.drift.generator import DriftCorridorGenerator

__all__ = [
    # Constants
    'DIRECTIONS',
    # Coordinates
    'get_utm_crs',
    'transform_geometry',
    'compass_to_vector',
    # Distribution
    'get_projection_distance',
    'get_distribution_width',
    # Corridor
    'create_base_surface',
    'create_projected_corridor',
    # Shadow/Blocking
    'create_obstacle_shadow',
    'extract_polygons',
    # Clipping
    'clip_corridor_at_obstacles',
    'keep_reachable_part',
    # Generator
    'DriftCorridorGenerator',
]
