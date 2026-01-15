"""
Smart hybrid: Use FAST method for depths, ACCURATE method for structures.

Rationale:
- Structures (2-10 objects): Use accurate dblquad for precision
- Depths (100+ polygons after MultiPolygon split): Use fast geometric method for speed
- The cascade already filters by draught per vessel, so fast method is sufficient for depths
"""
from typing import Any, Callable, Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
import logging

from geometries.calculate_probability_holes import compute_probability_holes
from geometries.pdf_corrected_fast_probability_holes import compute_probability_holes_pdf_corrected

logger = logging.getLogger(__name__)


def compute_probability_holes_smart_hybrid(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
    lateral_sigma_range: float = 5.0,
    use_fast: bool = False,  # Set True for depths, False for structures
    is_structure: bool = False,  # Set True when calculating allision (structures)
) -> list[list[list[float]]]:
    """
    Smart hybrid: Choose method based on whether objects are structures or depths.

    Args:
        lines: List of LineString geometries representing traffic legs
        distributions: List of distribution lists (one per leg)
        weights: List of weight lists (one per leg)
        objs_gdf_list: List of GeoDataFrames containing object geometries
        distance: Maximum drift distance in meters
        progress_callback: Optional callback(completed, total, message) -> bool
        lateral_sigma_range: Lateral range in standard deviations
        use_fast: If True, use fast geometric method (for depths with many polygons)
                 If False, use accurate dblquad method (for structures with few objects)
        is_structure: If True, calculating for structures (allision), else depths (grounding)

    Returns:
        3-level nested list: [leg_idx][direction_idx][object_idx] = probability (0-1)
    """
    if use_fast:
        # Fast geometric method - no calibration factors
        # Pure geometric probability calculation
        pdf_factor = 1.0
        if is_structure:
            logger.info("Using fast geometric method for structures (no calibration)")
        else:
            logger.info("Using fast geometric method for depths (no calibration)")

        return compute_probability_holes_pdf_corrected(
            lines, distributions, weights, objs_gdf_list, distance,
            drift_directions=None,  # Use all 8 directions
            lateral_sigma_range=lateral_sigma_range,
            progress_callback=progress_callback,
            pdf_correction_factor=pdf_factor
        )
    else:
        # Accurate dblquad method for structures (few objects)
        logger.info("Using ACCURATE dblquad method (for structures)")
        return compute_probability_holes(
            lines, distributions, weights, objs_gdf_list, distance,
            progress_callback=progress_callback
        )
