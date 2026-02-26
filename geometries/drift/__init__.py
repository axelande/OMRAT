# -*- coding: utf-8 -*-
"""
Drift Corridor Package

Provides drift corridor generation for ship grounding risk assessment.
Uses nautical/compass convention (0°=North, 90°=West) and quad-based sweep
algorithm for accurate obstacle shadow calculation.
"""

from .constants import DIRECTIONS
from .coordinates import (
    get_utm_crs,
    transform_geometry,
    compass_to_vector,
)
from .distribution import (
    get_projection_distance,
    get_distribution_width,
)
from .corridor import (
    create_base_surface,
    create_projected_corridor,
)
from .shadow import (
    create_obstacle_shadow,
    extract_polygons,
)
from .clipping import (
    clip_corridor_at_obstacles,
    keep_reachable_part,
)
from .generator import DriftCorridorGenerator
from .probability_integration import (
    compute_shadow_adjusted_holes,
    separate_obstacles_by_type,
    blend_with_pdf_holes,
    get_direction_index,
    direction_index_to_angle,
)

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
    # Shadow
    'create_obstacle_shadow',
    'extract_polygons',
    # Clipping
    'clip_corridor_at_obstacles',
    'keep_reachable_part',
    # Generator
    'DriftCorridorGenerator',
    # Probability Integration
    'compute_shadow_adjusted_holes',
    'separate_obstacles_by_type',
    'blend_with_pdf_holes',
    'get_direction_index',
    'direction_index_to_angle',
]
