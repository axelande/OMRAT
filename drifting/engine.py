from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


DIRECTIONS_COMPASS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)


@dataclass(frozen=True)
class DepthTarget:
    target_id: str
    depth_m: float
    geometry: BaseGeometry


@dataclass(frozen=True)
class StructureTarget:
    target_id: str
    top_height_m: float
    geometry: BaseGeometry


@dataclass(frozen=True)
class ShipState:
    draught_m: float
    anchor_d: float
    ship_height_m: float | None = None
    respect_structure_height: bool = False


@dataclass(frozen=True)
class LegState:
    leg_id: str
    line: LineString
    mean_offset_m: float
    lateral_sigma_m: float


@dataclass(frozen=True)
class DriftConfig:
    reach_distance_m: float
    corridor_sigma_multiplier: float = 3.0
    use_leg_offset_for_distance: bool = False


@dataclass(frozen=True)
class TargetHit:
    leg_id: str
    direction_deg: int
    role: str
    target_id: str
    distance_m: float
    coverage_percent: float


def available_targets(depths: Iterable[DepthTarget], structures: Iterable[StructureTarget]) -> dict[str, int]:
    return {
        "depth_count": sum(1 for _ in depths),
        "structure_count": sum(1 for _ in structures),
    }


def interesting_targets(ship: ShipState, depths: Iterable[DepthTarget], structures: Iterable[StructureTarget]) -> dict[str, list]:
    anchoring_limit = ship.draught_m * ship.anchor_d
    grounding_depths = [d for d in depths if d.depth_m < ship.draught_m]
    anchoring_depths = [d for d in depths if d.depth_m < anchoring_limit]
    if ship.respect_structure_height and ship.ship_height_m is not None:
        structure_targets = [s for s in structures if s.top_height_m <= ship.ship_height_m]
    else:
        structure_targets = list(structures)
    return {
        "grounding_depths": grounding_depths,
        "anchoring_depths": anchoring_depths,
        "structures": structure_targets,
    }


def compass_to_math_deg(compass_deg: int) -> float:
    # Compass: 0=N, 90=E, clockwise. Math: 0=E, 90=N, counterclockwise.
    return (90.0 - float(compass_deg)) % 360.0


def _offset_line_perpendicular(line: LineString, offset_m: float) -> LineString:
    if abs(offset_m) < 1e-12:
        return line
    # Parallel offset can fail for sharp/short lines. Fall back to original line.
    try:
        side = "left" if offset_m >= 0 else "right"
        shifted = line.parallel_offset(abs(offset_m), side=side, join_style=2)
        if shifted.is_empty:
            return line
        if shifted.geom_type == "MultiLineString":
            # Pick the longest segment for stability.
            shifted = max(list(shifted.geoms), key=lambda g: g.length)
        if isinstance(shifted, LineString):
            return shifted
    except Exception:
        pass
    return line


def build_directional_corridor(leg: LegState, direction_deg: int, cfg: DriftConfig) -> Polygon:
    shifted_line = _offset_line_perpendicular(leg.line, leg.mean_offset_m)
    spread = max(1.0, cfg.corridor_sigma_multiplier * max(0.0, leg.lateral_sigma_m))
    base = shifted_line.buffer(spread)

    math_deg = compass_to_math_deg(direction_deg)
    rad = math.radians(math_deg)
    dx = cfg.reach_distance_m * math.cos(rad)
    dy = cfg.reach_distance_m * math.sin(rad)
    moved = translate(base, xoff=dx, yoff=dy)

    return base.union(moved).convex_hull


def corridor_width_m(leg: LegState, cfg: DriftConfig, direction_deg: int | None = None) -> float:
    """Return effective corridor width used for edge-length normalization.

    Width is evaluated on the cross-drift axis using:
    - lateral spread width: +/- (k * sigma)
    - leg projection width: leg extent projected onto cross-drift axis

    Special cases:
    - drift parallel to leg -> width ~= 2 * k * sigma
    - drift perpendicular to leg -> width ~= leg_length
    """
    spread = max(1.0, cfg.corridor_sigma_multiplier * max(0.0, leg.lateral_sigma_m))
    base_width = 2.0 * spread

    if direction_deg is None:
        return max(1.0, base_width)

    # Cross-drift axis is perpendicular to drift direction.
    math_deg = compass_to_math_deg(direction_deg)
    rad = math.radians(math_deg)
    ux = math.cos(rad)
    uy = math.sin(rad)
    nx = -uy
    ny = ux

    coords = list(leg.line.coords)
    if len(coords) < 2:
        return max(1.0, base_width)

    projs = [float(x) * nx + float(y) * ny for x, y in coords]
    leg_cross_extent = max(projs) - min(projs) if projs else 0.0
    return max(1.0, max(base_width, max(0.0, leg_cross_extent)))


def directional_distance_m(
    leg: LegState,
    direction_deg: int,
    target_geom: BaseGeometry,
    use_leg_offset: bool = False,
) -> float | None:
    """Minimum directional distance from the leg to the target geometry.

    Extracts boundary coordinates of *target_geom*, shoots a reverse ray
    from each back to the leg reference line (same logic as
    ``directional_distance_to_point_from_offset_leg``), and returns the
    minimum distance found.  This ensures the distance is measured in the
    drift direction from the nearest point on the leg line to the closest
    vertex of the target -- consistent with the per-edge calculations.
    """
    if target_geom is None or target_geom.is_empty:
        return None

    # Collect all boundary coordinates from the target geometry.
    coords: list[tuple[float, float]] = []
    gt = target_geom.geom_type
    if gt == "Point":
        coords = [(target_geom.x, target_geom.y)]
    elif gt == "MultiPoint":
        coords = [(p.x, p.y) for p in target_geom.geoms]
    elif gt in ("LineString", "LinearRing"):
        coords = list(target_geom.coords)
    elif gt == "MultiLineString":
        for g in target_geom.geoms:
            coords.extend(g.coords)
    elif gt == "Polygon":
        coords = list(target_geom.exterior.coords)
    elif gt == "MultiPolygon":
        for g in target_geom.geoms:
            coords.extend(g.exterior.coords)
    elif gt == "GeometryCollection":
        for g in target_geom.geoms:
            if hasattr(g, "coords"):
                coords.extend(g.coords)

    if not coords:
        return None

    # Use reverse-ray from each target vertex back to the leg line.
    values: list[float] = []
    for cx, cy in coords:
        d = directional_distance_to_point_from_offset_leg(
            leg, direction_deg, Point(cx, cy),
            use_leg_offset=use_leg_offset,
        )
        if d is not None:
            values.append(d)

    if not values:
        return None
    return min(values)


def directional_distance_to_point_from_offset_leg(
    leg: LegState,
    direction_deg: int,
    point: Point,
    use_leg_offset: bool = False,
) -> float | None:
    """Measure directional distance from the leg reference line to a point.

    By default, distance is measured from the leg centerline.
    Set ``use_leg_offset=True`` to measure from the mean lateral-offset leg.

    Distance is measured strictly in the drift direction by tracing a reverse
    ray from the point back to the selected reference line and taking the
    first intersection along that direction.
    """
    if point is None or point.is_empty:
        return None

    start_line = (
        _offset_line_perpendicular(leg.line, leg.mean_offset_m)
        if use_leg_offset
        else leg.line
    )

    math_deg = compass_to_math_deg(direction_deg)
    rad = math.radians(math_deg)
    ux = math.cos(rad)
    uy = math.sin(rad)

    reverse_ray = LineString([
        (point.x, point.y),
        (point.x - ux * 1e6, point.y - uy * 1e6),
    ])
    inter = reverse_ray.intersection(start_line)

    pts: list[Point] = []
    gt = inter.geom_type
    if gt == "Point":
        pts = [inter]
    elif gt == "MultiPoint":
        pts = list(inter.geoms)
    elif gt in ("LineString", "LinearRing"):
        pts = [Point(c) for c in inter.coords]
    elif gt == "MultiLineString":
        for g in inter.geoms:
            pts.extend(Point(c) for c in g.coords)
    elif gt == "GeometryCollection":
        for g in inter.geoms:
            if hasattr(g, "coords"):
                pts.extend(Point(c) for c in g.coords)

    if not pts:
        p_leg, p_target = nearest_points(start_line, point)
        vx = p_target.x - p_leg.x
        vy = p_target.y - p_leg.y
        dot = vx * ux + vy * uy
        if dot < 0:
            return None
        return float(dot)

    values: list[float] = []
    for p_leg in pts:
        vx = point.x - p_leg.x
        vy = point.y - p_leg.y
        dot = vx * ux + vy * uy
        if dot >= 0:
            values.append(float(dot))

    if not values:
        return None
    return min(values)


def edge_average_distance_m(
    leg: LegState,
    direction_deg: int,
    edge: tuple[tuple[float, float], tuple[float, float]],
    use_leg_offset: bool = False,
) -> float | None:
    """Average directional distance for an edge using both edge endpoints."""
    start_distance = directional_distance_to_point_from_offset_leg(
        leg,
        direction_deg,
        Point(edge[0]),
        use_leg_offset=use_leg_offset,
    )
    end_distance = directional_distance_to_point_from_offset_leg(
        leg,
        direction_deg,
        Point(edge[1]),
        use_leg_offset=use_leg_offset,
    )
    values = [value for value in (start_distance, end_distance) if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def coverage_percent(corridor: Polygon, target_geom: BaseGeometry) -> float:
    if target_geom is None or target_geom.is_empty:
        return 0.0
    area = float(getattr(target_geom, "area", 0.0))
    if area <= 0.0:
        return 0.0
    inter = corridor.intersection(target_geom)
    return max(0.0, min(100.0, (float(inter.area) / area) * 100.0))


def edge_hit_percent(corridor: Polygon, target_geom: BaseGeometry, width_m: float) -> float:
    """
    Estimate hit probability from boundary overlap length.

    Probability proxy is overlap_length / corridor_width, clamped to [0, 1].
    Returned as percentage in [0, 100].
    """
    if target_geom is None or target_geom.is_empty:
        return 0.0
    if width_m <= 0.0:
        return 0.0

    try:
        boundary = target_geom.boundary
        inter = corridor.intersection(boundary)
        overlap_len = float(getattr(inter, "length", 0.0))
        if overlap_len <= 0.0:
            return 0.0
        frac = max(0.0, min(1.0, overlap_len / float(width_m)))
        return frac * 100.0
    except Exception:
        # Fallback to area overlap metric when boundary operations fail.
        return coverage_percent(corridor, target_geom)


def evaluate_leg_direction(
    leg: LegState,
    ship: ShipState,
    direction_deg: int,
    depths: Iterable[DepthTarget],
    structures: Iterable[StructureTarget],
    cfg: DriftConfig,
) -> list[TargetHit]:
    selected = interesting_targets(ship, depths, structures)
    corridor = build_directional_corridor(leg, direction_deg, cfg)
    width_m = corridor_width_m(leg, cfg, direction_deg)

    hits: list[TargetHit] = []

    # Grounding depths
    for d in selected["grounding_depths"]:
        if not corridor.intersects(d.geometry):
            continue
        dist = directional_distance_m(
            leg,
            direction_deg,
            d.geometry,
            use_leg_offset=cfg.use_leg_offset_for_distance,
        )
        if dist is None:
            continue
        hits.append(TargetHit(
            leg_id=leg.leg_id,
            direction_deg=direction_deg,
            role="grounding",
            target_id=d.target_id,
            distance_m=dist,
            coverage_percent=edge_hit_percent(corridor, d.geometry, width_m),
        ))

    # Anchoring depths
    for d in selected["anchoring_depths"]:
        if not corridor.intersects(d.geometry):
            continue
        dist = directional_distance_m(
            leg,
            direction_deg,
            d.geometry,
            use_leg_offset=cfg.use_leg_offset_for_distance,
        )
        if dist is None:
            continue
        hits.append(TargetHit(
            leg_id=leg.leg_id,
            direction_deg=direction_deg,
            role="anchoring",
            target_id=d.target_id,
            distance_m=dist,
            coverage_percent=edge_hit_percent(corridor, d.geometry, width_m),
        ))

    # Structures
    for s in selected["structures"]:
        if not corridor.intersects(s.geometry):
            continue
        dist = directional_distance_m(
            leg,
            direction_deg,
            s.geometry,
            use_leg_offset=cfg.use_leg_offset_for_distance,
        )
        if dist is None:
            continue
        hits.append(TargetHit(
            leg_id=leg.leg_id,
            direction_deg=direction_deg,
            role="structure",
            target_id=s.target_id,
            distance_m=dist,
            coverage_percent=edge_hit_percent(corridor, s.geometry, width_m),
        ))

    return hits
