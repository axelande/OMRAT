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
    # Any other geometry type (Point, LineString, GeometryCollection, ...) has
    # no useful "obstacle edge" for drift-hit detection and is ignored.

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
        interior_p2 = (p1[0] + (1 - t) * (p2[0] - p1[0]), p1[1] + (1 - t) * (p2[1] - p1[1]))
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

    # Calculate segment vector and normal.  Any zero-length segment would have
    # failed the earlier ``intersection.is_empty`` check so seg_len > 0 here.
    seg_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    seg_len = np.linalg.norm(seg_vec)

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


def segment_corridor_overlap_length(
    segment: tuple[tuple[float, float], tuple[float, float]],
    corridor: "Polygon",
    drift_angle: float | None = None,
    leg_centroid: tuple[float, float] | None = None,
) -> float:
    """Return the corridor-intersection length for a segment, or 0 if missed.

    Combines the shapely work of :func:`_segment_intersects_corridor` with
    the overlap-length measurement that immediately followed it at every
    caller in :mod:`compute.drifting_model`.  The original pair performed
    ``corridor.intersection(seg_line)`` twice -- once to decide "hit?" and
    a second time to measure -- which cost ~300k duplicate shapely
    intersections on proj.omrat.  Using this helper runs the intersection
    exactly once.

    Returns ``0.0`` for segments that don't hit, that face the wrong way,
    or whose overlap is degenerate (point touch).  The drift-direction
    filter runs before any shapely work so most misses (~74 % on
    proj.omrat) never allocate a LineString.
    """
    p1, p2 = segment

    # Drift-direction pre-filter.  Cheap arithmetic that rejects segments
    # that can never be hit (facing away, nearly parallel, far behind the
    # leg).  Running this before the shapely work saves ~1.4M shapely
    # ops on proj.omrat.
    if drift_angle is not None and leg_centroid is not None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 0.0:
            return 0.0
        inv_len = seg_len_sq ** -0.5

        # Outward normal for CCW polygon = (dy, -dx) / len
        nx = dy * inv_len
        ny = -dx * inv_len

        drift_rad = np.radians(drift_angle)
        drift_ux = float(np.cos(drift_rad))
        drift_uy = float(np.sin(drift_rad))

        drift_into_segment = drift_ux * nx + drift_uy * ny
        if abs(drift_into_segment) < 0.17 or drift_into_segment > 0:
            return 0.0

        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        vx = mx - leg_centroid[0]
        vy = my - leg_centroid[1]
        dist_to_segment_sq = vx * vx + vy * vy
        if dist_to_segment_sq > 0.0:
            distance_ahead = vx * drift_ux + vy * drift_uy
            if distance_ahead < -0.5 * dist_to_segment_sq ** 0.5:
                return 0.0

    seg_line = LineString([p1, p2])

    if not corridor.intersects(seg_line):
        return 0.0
    intersection = corridor.intersection(seg_line)
    if intersection.is_empty:
        return 0.0
    if intersection.geom_type == 'Point':
        t = 0.01
        interior_p1 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        interior_p2 = (p1[0] + (1 - t) * (p2[0] - p1[0]), p1[1] + (1 - t) * (p2[1] - p1[1]))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if not (corridor.contains(Point(interior_p1)) or
                corridor.contains(Point(interior_p2)) or
                corridor.contains(Point(mid))):
            return 0.0

    return float(getattr(intersection, 'length', 0.0))


def compute_edge_reachable_widths_1d(
    edges_info: list[dict],
    drift_angle: float,
    leg_perp_lo: float | None = None,
    leg_perp_hi: float | None = None,
) -> list[dict]:
    """1D shadow-carve facing edges within a polygon (or set of polygons).

    Each input dict must have keys ``p1`` and ``p2`` (two ``(x, y)`` tuples in
    UTM), and optionally ``dist`` (along-drift distance from the leg -- if
    absent, computed here from the segment midpoint's dot product with the
    drift direction, minus a zero reference).  The function projects each
    segment's endpoints onto the perpendicular-to-drift axis (u_perp),
    sorts edges by along-drift distance ascending, and for each edge
    subtracts the perpendicular ranges already claimed by closer edges.
    The remaining ranges are the edge's *reachable* perpendicular
    intervals -- their combined length is stored as ``reachable_width``.

    If ``leg_perp_lo`` / ``leg_perp_hi`` are provided, every edge's perp
    interval is first clipped to that range.  This is essential when
    ships are assumed uniform along the leg (IWRAP model): a polygon
    edge whose perp-drift projection falls outside the leg's own
    projection cannot be reached by any ship on the leg, so it must
    contribute zero.  Without the clip a polygon that overhangs the leg
    can produce sum(edge_h_eff) > 1, which is unphysical.

    Ships drifting from the leg first encounter the closest facing edge
    at any given perpendicular position, and the perpendicular width of
    ships that can still reach a farther edge is *(edge's perp range) -
    (union of all closer edges' perp ranges)*.  This mirrors the
    "a ship grounds once" physics without needing shadow polygons.
    """
    if not edges_info:
        return []
    drift_rad = np.radians(drift_angle)
    drift_ux = float(np.cos(drift_rad))
    drift_uy = float(np.sin(drift_rad))
    # Perpendicular direction (90° CCW from drift)
    perp_ux = -drift_uy
    perp_uy = drift_ux

    have_leg_clip = leg_perp_lo is not None and leg_perp_hi is not None
    if have_leg_clip and leg_perp_lo > leg_perp_hi:
        leg_perp_lo, leg_perp_hi = leg_perp_hi, leg_perp_lo

    prepared: list[dict] = []
    for e in edges_info:
        p1 = e['p1']
        p2 = e['p2']
        x1 = p1[0] * perp_ux + p1[1] * perp_uy
        x2 = p2[0] * perp_ux + p2[1] * perp_uy
        x_lo = min(x1, x2)
        x_hi = max(x1, x2)
        if have_leg_clip:
            # Ships uniform along the leg only exist in [leg_perp_lo, leg_perp_hi].
            # Portions of the edge outside that range get zero ships.
            x_lo = max(x_lo, float(leg_perp_lo))
            x_hi = min(x_hi, float(leg_perp_hi))
            if x_hi <= x_lo:
                continue
        if 'dist' in e and e['dist'] is not None:
            dist = float(e['dist'])
        else:
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])
            dist = mx * drift_ux + my * drift_uy
        prepared.append({**e, 'x_lo': x_lo, 'x_hi': x_hi, 'dist_sort': dist})

    order = sorted(range(len(prepared)), key=lambda i: prepared[i]['dist_sort'])

    covered: list[tuple[float, float]] = []
    result: list[dict] = [dict(prepared[i]) for i in range(len(prepared))]

    def _subtract(interval, cov):
        lo, hi = interval
        remaining = [(lo, hi)]
        for c_lo, c_hi in cov:
            new_r = []
            for r_lo, r_hi in remaining:
                if c_hi <= r_lo or c_lo >= r_hi:
                    new_r.append((r_lo, r_hi))
                elif c_lo <= r_lo and c_hi >= r_hi:
                    pass  # fully covered
                elif c_lo <= r_lo:
                    new_r.append((c_hi, r_hi))
                elif c_hi >= r_hi:
                    new_r.append((r_lo, c_lo))
                else:
                    new_r.append((r_lo, c_lo))
                    new_r.append((c_hi, r_hi))
            remaining = new_r
        return remaining

    def _merge(cov, new_iv):
        all_iv = sorted(cov + [new_iv])
        merged = [all_iv[0]]
        for lo, hi in all_iv[1:]:
            m_lo, m_hi = merged[-1]
            if lo <= m_hi:
                merged[-1] = (m_lo, max(m_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    for i in order:
        prep = prepared[i]
        reach = _subtract((prep['x_lo'], prep['x_hi']), covered)
        result[i]['reachable_intervals'] = reach
        result[i]['reachable_width'] = sum(hi - lo for lo, hi in reach)
        covered = _merge(covered, (prep['x_lo'], prep['x_hi']))
    return result
