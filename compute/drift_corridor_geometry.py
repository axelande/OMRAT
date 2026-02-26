"""
Drift corridor geometry functions.

Pure geometric functions for drift corridor construction and intersection
testing. No QGIS dependency - only numpy and shapely.
"""
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _compass_idx_to_math_idx(compass_d_idx: int) -> int:
    """
    Convert compass direction index to math convention index.

    The wind rose uses compass convention (d_idx * 45):
    - d_idx=0 -> compass 0 deg = North
    - d_idx=1 -> compass 45 deg = NE
    - d_idx=2 -> compass 90 deg = East
    - etc.

    The probability_holes arrays use math convention indices (index * 45):
    - index=0 -> math 0 deg = East
    - index=1 -> math 45 deg = NE
    - index=2 -> math 90 deg = North
    - etc.

    Conversion: math_angle = (90 - compass_angle) % 360
                math_index = math_angle // 45
    """
    compass_angle = compass_d_idx * 45
    math_angle = (90 - compass_angle) % 360
    return math_angle // 45


def _extract_obstacle_segments(geom: BaseGeometry) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Extract individual line segments from a polygon boundary.

    IMPORTANT: This function normalizes polygon orientation to CCW (counter-clockwise)
    before extracting segments. This ensures consistent outward normal calculation
    in _segment_intersects_corridor().

    For CCW polygons:
    - Exterior ring goes counter-clockwise
    - Interior (hole) rings go clockwise
    - Outward normal = rotate segment vector 90 deg clockwise (right-hand rule)

    Args:
        geom: A shapely geometry (Polygon, MultiPolygon, etc.)

    Returns:
        List of ((x1, y1), (x2, y2)) tuples representing line segments
    """
    from shapely.geometry import polygon as shapely_polygon

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def extract_from_ring(ring_coords):
        coords = list(ring_coords)
        for i in range(len(coords) - 1):
            p1 = (float(coords[i][0]), float(coords[i][1]))
            p2 = (float(coords[i + 1][0]), float(coords[i + 1][1]))
            if p1 != p2:  # Skip zero-length segments
                segments.append((p1, p2))

    if isinstance(geom, Polygon):
        # Normalize polygon to CCW exterior, CW holes using shapely's orient()
        # This ensures consistent outward normal calculation
        oriented_geom = shapely_polygon.orient(geom, sign=1.0)  # 1.0 = CCW exterior
        extract_from_ring(oriented_geom.exterior.coords)
        for interior in oriented_geom.interiors:
            extract_from_ring(interior.coords)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            segments.extend(_extract_obstacle_segments(poly))
    elif hasattr(geom, 'boundary'):
        boundary = geom.boundary
        if hasattr(boundary, 'coords'):
            extract_from_ring(boundary.coords)
        elif hasattr(boundary, 'geoms'):
            for line in boundary.geoms:
                extract_from_ring(line.coords)

    return segments


def _create_drift_corridor(
    leg: LineString,
    drift_angle: float,
    distance: float,
    lateral_spread: float,
) -> Polygon | None:
    """
    Create the drift corridor polygon for a given leg and drift direction.

    Creates a polygon representing the area a ship could drift through,
    from the leg starting position to the maximum drift distance.

    This matches the approach in pdf_corrected_fast_probability_holes.py
    but uses convex hull to handle self-intersection cases.

    Args:
        leg: The traffic leg LineString
        drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
                     This matches pdf_corrected_fast_probability_holes.py
        distance: Maximum drift distance in meters
        lateral_spread: Half-width of corridor (in meters)

    Returns:
        Polygon representing the drift corridor, or None if invalid
    """
    leg_coords = np.array(leg.coords)
    if len(leg_coords) < 2:
        return None

    leg_start = leg_coords[0]
    leg_end = leg_coords[-1]
    leg_vec = leg_end - leg_start
    leg_length = np.linalg.norm(leg_vec)

    if leg_length == 0:
        return None

    leg_dir = leg_vec / leg_length
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    # Drift direction vector (math convention: 0=East, 90=North)
    drift_angle_rad = np.radians(drift_angle)
    drift_vec = np.array([np.cos(drift_angle_rad), np.sin(drift_angle_rad)]) * distance

    # Create leg rectangle corners (CCW order)
    p1 = leg_start - lateral_spread * perp_dir
    p2 = leg_start + lateral_spread * perp_dir
    p3 = leg_end + lateral_spread * perp_dir
    p4 = leg_end - lateral_spread * perp_dir

    # Create drifted rectangle corners (CCW order)
    p1_drift = p1 + drift_vec
    p2_drift = p2 + drift_vec
    p3_drift = p3 + drift_vec
    p4_drift = p4 + drift_vec

    # Create the two rectangles as separate polygons and union them
    # This avoids self-intersection issues when drift is along the leg direction
    leg_rect = Polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
    drift_rect = Polygon([tuple(p1_drift), tuple(p2_drift), tuple(p3_drift), tuple(p4_drift)])

    corridor = unary_union([leg_rect, drift_rect])

    # If union creates MultiPolygon (shouldn't happen but handle it), take convex hull
    if isinstance(corridor, MultiPolygon):
        corridor = corridor.convex_hull

    if corridor.is_empty or corridor.area == 0:
        return None

    return corridor


def _segment_intersects_corridor(
    segment: tuple[tuple[float, float], tuple[float, float]],
    corridor: Polygon,
    drift_angle: float | None = None,
    leg_centroid: tuple[float, float] | None = None,
    leg_line: LineString | None = None,
) -> bool:
    """
    Check if a line segment would be hit by ships drifting from the leg.

    A segment is hit if:
    1. The corridor geometrically intersects the segment (substantially, not just a point touch)
    2. The segment is ahead of the leg in the drift direction
    3. The drift direction "faces into" the segment's outward normal
       (ships must be moving toward the segment's blocking face)

    The key insight for obstacle polygons (assumed CCW): each edge has an outward normal
    pointing to the right of the edge vector. For a ship to hit an edge, it must be
    drifting INTO that outward normal (positive dot product).

    Args:
        segment: ((x1, y1), (x2, y2)) tuple
        corridor: Drift corridor polygon
        drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
        leg_centroid: (x, y) centroid of the leg
        leg_line: Optional LineString of the leg

    Returns:
        True if segment would be hit by drift
    """
    p1, p2 = segment
    seg_line = LineString([p1, p2])

    # Basic intersection check
    if not corridor.intersects(seg_line):
        return False

    # Check if the intersection is substantial (not just a point touch)
    intersection = corridor.intersection(seg_line)

    if intersection.is_empty:
        return False
    if intersection.geom_type == 'Point':
        t = 0.01
        interior_p1 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        interior_p2 = (p1[0] + (1-t) * (p2[0] - p1[0]), p1[1] + (1-t) * (p2[1] - p1[1]))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        if not (corridor.contains(Point(interior_p1)) or
                corridor.contains(Point(interior_p2)) or
                corridor.contains(Point(mid))):
            return False

    if drift_angle is None or leg_centroid is None:
        return True

    # Drift direction vector (unit vector)
    drift_angle_rad = np.radians(drift_angle)
    drift_dir = np.array([np.cos(drift_angle_rad), np.sin(drift_angle_rad)])

    # Calculate segment vector and normal
    seg_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    seg_len = np.linalg.norm(seg_vec)
    if seg_len == 0:
        return False

    # Outward normal for CCW polygon: rotate segment vector 90 deg clockwise
    # For segment (p1 -> p2), outward normal points to the RIGHT of the direction
    # Rotate (dx, dy) by -90 deg: (dy, -dx)
    seg_outward_normal = np.array([seg_vec[1], -seg_vec[0]]) / seg_len

    # Check if drift is parallel to segment (can't hit a parallel segment)
    drift_into_segment = np.dot(drift_dir, seg_outward_normal)
    if abs(drift_into_segment) < 0.17:  # Nearly parallel (< ~10 deg from parallel)
        return False

    # KEY CHECK: For a ship to hit this segment (enter the polygon through this face),
    # the drift direction must oppose the outward normal (negative dot product).
    # If drift_into_segment > 0, ships are moving in the same direction as the
    # outward normal, meaning they would EXIT through this face, not enter.
    if drift_into_segment > 0:
        return False

    # Check that the segment is not significantly behind the leg in the drift direction.
    # This prevents false positives where a wide corridor intersects a segment that is
    # BEHIND the leg in the drift direction (e.g., Leg 2 south of structure cannot hit
    # the top edge via S/SW/SE drift because those drift directions go away from structure).
    #
    # We check if the segment midpoint is ahead of the leg centroid in drift direction.
    # "Ahead" means the dot product of (segment_mid - leg_centroid) with drift_dir is positive.
    seg_mid = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    leg_center = np.array(leg_centroid)
    vec_to_segment = seg_mid - leg_center
    dist_to_segment = np.linalg.norm(vec_to_segment)

    # Dot product: positive means segment is in front of leg in drift direction
    distance_ahead = np.dot(vec_to_segment, drift_dir)

    # Allow significant tolerance because the corridor has lateral spread.
    # A segment that is slightly behind in the drift direction can still be
    # reachable by ships that start from the lateral edges of the leg.
    # Only reject if the segment is more than 50% of the way "behind" the leg.
    # This catches cases like Leg 2 (south) trying to hit Segment 2 (north top edge)
    # via S/SE/SW drift where the segment is very far behind.
    if distance_ahead < -0.5 * dist_to_segment:
        # Segment is substantially behind the leg in the drift direction
        return False

    return True
