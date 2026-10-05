"""
Powered grounding/allision geometry -- IWRAP Category I and II with shadow effects.

Category I: Obstacle already inside the lane.
  Ships follow the leg on their lateral offset; any obstacle that lies
  between the leg's start and end within the lateral spread is hit unless
  the navigator reacts.
  P(hit obstacle) = mass
    - mass: fraction of lateral distribution intercepted by the obstacle
      along the leg itself (first hit wins, so an obstacle shadowed by a
      nearer one along the leg receives only the uncovered rays)
    - No distance decay: the ship is already on a collision course.

Category II: Ships failing to turn at a bend.
  Ships continue past the turning point on the previous heading.
  P(hit obstacle) = mass * exp(-d_mean / (ai * V))
    - mass: fraction of lateral distribution intercepted by the obstacle
    - d_mean: weighted mean along-track distance to the obstacle
    - Shadow: closer obstacles block the distribution for obstacles behind them

Provides ``PoweredOverlapVisualizer`` which embeds matplotlib plots inside the
existing ``ShowGeomRes`` dialog, following the same pattern as
``DriftingOverlapVisualizer`` in ``geometries/get_drifting_overlap.py``.
"""

from __future__ import annotations

from collections import defaultdict
from math import cos, exp, radians
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.stats import norm
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shapely_transform
import shapely.wkt as sw

from compute.cat1_start_zone import start_zone_fraction
from compute.powered_breakdown import (
    allision_report_from_computations, build_sidebar_rows, format_frequency, format_share,
    geometric_probability, leg_title as _leg_title,
)
from geometries.boundary_edges import boundary_edges

from ui.show_geom_res import ShowGeomRes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIGMA = 3          # standard deviations for the distribution band
MAX_RANGE = 50_000   # max ray length in meters (50 km)
DEFAULT_MAX_DRAFT = 15.0  # default max ship draft (m)
# Ray count and span live in compute/ray_sampling.py (6 sigma, 1000 rays)
# so the report and the tangent-line markers use the same figures.
from compute.ray_sampling import N_RAYS, RAY_SPAN_SIGMA, ray_offsets  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class SimpleProjector:
    """Equirectangular projection (lon/lat -> metres) around a reference point."""

    def __init__(self, lon_ref: float, lat_ref: float) -> None:
        self.lon_ref = lon_ref
        self.lat_ref = lat_ref
        self.mx = 111_320.0 * cos(radians(lat_ref))
        self.my = 110_540.0

    def transform(self, lon: float, lat: float) -> tuple[float, float]:
        return ((lon - self.lon_ref) * self.mx,
                (lat - self.lat_ref) * self.my)


def _parse_point(coord_str: str) -> tuple[float, float]:
    parts = coord_str.strip().split()
    return float(parts[0]), float(parts[1])


def _project_wkt_geom(geom, proj: SimpleProjector):
    """Project a Shapely geometry using *proj*."""
    return shapely_transform(lambda x, y, z=None: proj.transform(x, y), geom)


def _weighted_avg_speed_knots(traffic_dir: dict) -> float:
    freq = traffic_dir.get("Frequency (ships/year)", [])
    spd = traffic_dir.get("Speed (knots)", [])
    tot_f, tot_fs = 0.0, 0.0
    for row_f, row_s in zip(freq, spd):
        for f, s in zip(row_f, row_s):
            try:
                fv = float(f) if f != '' else 0.0
                sv = float(s) if s != '' else 0.0
            except (ValueError, TypeError):
                continue
            if 0 < fv < float("inf") and 0 < sv < float("inf"):
                tot_f += fv
                tot_fs += fv * sv
    return tot_fs / tot_f if tot_f > 0 else 0.0


def _leg_vectors(start: np.ndarray, end: np.ndarray):
    """Return (unit_direction, perpendicular, length) for a leg."""
    d = end - start
    length = float(np.linalg.norm(d))
    if length == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0
    u = d / length
    n = np.array([-u[1], u[0]])
    return u, n, length


def _make_band_polygon(
    start: np.ndarray,
    end: np.ndarray,
    mean_offset: float,
    sigma: float,
    n_sigma: int = N_SIGMA,
) -> Polygon:
    """Create a polygon representing the distribution band around a leg."""
    u, n, _ = _leg_vectors(start, end)
    lo = mean_offset - n_sigma * sigma
    hi = mean_offset + n_sigma * sigma
    return Polygon([
        start + lo * n, start + hi * n,
        end + hi * n, end + lo * n,
    ])


def _get_all_coords(geom) -> list[tuple[float, float]]:
    """Extract all coordinate tuples from any Shapely geometry."""
    if geom.is_empty:
        return []
    if geom.geom_type == 'Point':
        return [(geom.x, geom.y)]
    elif geom.geom_type in ('LineString', 'LinearRing'):
        return list(geom.coords)
    elif geom.geom_type == 'Polygon':
        return list(geom.exterior.coords)
    elif geom.geom_type in ('MultiPoint', 'MultiLineString',
                            'MultiPolygon', 'GeometryCollection'):
        coords = []
        for part in geom.geoms:
            coords.extend(_get_all_coords(part))
        return coords
    return []


def _ray_hit_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    max_range: float,
    obstacle_geom,
) -> float | None:
    """Cast a ray and find along-track distance to first intersection."""
    ray_end = origin + max_range * direction
    ray = LineString([origin, ray_end])
    if not ray.intersects(obstacle_geom):
        return None
    try:
        intersection = ray.intersection(obstacle_geom)
    except Exception:  # nosec B110 B112
        return None
    coords = _get_all_coords(intersection)
    if not coords:
        return None
    along_dists = []
    for px, py in coords:
        d = np.dot(np.array([px, py]) - origin, direction)
        if d > 0:
            along_dists.append(d)
    return min(along_dists) if along_dists else None


def _project_to_local(geom, origin: np.ndarray, along_dir: np.ndarray, perp_dir: np.ndarray):
    """Project geometry into local (along-track, lateral) coordinates."""
    ox, oy = float(origin[0]), float(origin[1])
    ax_d, ay = float(along_dir[0]), float(along_dir[1])
    px, py = float(perp_dir[0]), float(perp_dir[1])

    def transform_fn(x, y, z=None):
        dx, dy = x - ox, y - oy
        along = dx * ax_d + dy * ay
        lateral = dx * px + dy * py
        return along, lateral

    return shapely_transform(transform_fn, geom)


def _plot_geom(ax: Axes, geom, **kwargs) -> None:
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
    elif geom.geom_type == "MultiPolygon":
        for i, part in enumerate(geom.geoms):
            kw = {**kwargs}
            if i > 0:
                kw.pop("label", None)
            x, y = part.exterior.xy
            ax.fill(x, y, **kw)


# ---------------------------------------------------------------------------
# Shadow-aware Cat II computation
# ---------------------------------------------------------------------------

def _powered_na(distance: float, mean_time: float, ship_speed: float) -> float:
    """Cat II: P(not recovered) = exp(-d / (ai * V))."""
    ai = mean_time * ship_speed
    if ai <= 0:
        return 0.0
    return exp(-distance / ai)


def _extract_edges_local(
    geom,
    turn_pt: np.ndarray,
    along_dir: np.ndarray,
    perp_dir: np.ndarray,
) -> np.ndarray | None:
    """Extract polygon/line edges and transform into a local (along, lateral) frame.

    In the frame returned here, every ray in the Cat II sweep becomes a
    horizontal line ``y = offset`` travelling in +x, so ray/polygon hit
    distance reduces to edge-crossing math that vectorises cleanly over
    (rays x edges).

    Returns an ``(M, 2, 2)`` array of edge endpoints ``[[along, lateral], ...]``
    or ``None`` if no edges were found (e.g. a Point obstacle, which a
    zero-width ray cannot hit).

    Edge ``k`` of the returned array is boundary edge ``k`` of
    :func:`geometries.boundary_edges.boundary_edges`, the same numbering
    the result layers use for ``segment_idx`` -- that is what lets the
    caster report *which* edge a ray hit.
    """
    if geom is None or getattr(geom, 'is_empty', True):
        return None
    edges = boundary_edges(geom)
    if not edges:
        return None
    arr = np.asarray(edges, dtype=float).reshape(-1, 2, 2)      # (M, 2 endpoints, xy)
    diff = arr - turn_pt
    along = diff @ along_dir
    lateral = diff @ perp_dir
    return np.stack([along, lateral], axis=2)


def _build_hit_matrix(
    offsets: np.ndarray,
    obstacles: list[tuple[dict, str]],
    turn_pt: np.ndarray,
    ext_dir: np.ndarray,
    perp: np.ndarray,
    max_range: float = MAX_RANGE,
) -> np.ndarray:
    """Along-track distance from ``turn_pt`` to each obstacle, per ray.

    Thin wrapper over :func:`_build_hit_and_edge_matrices` for callers
    that only need the distances.
    """
    return _build_hit_and_edge_matrices(offsets, obstacles, turn_pt, ext_dir, perp, max_range)[0]


def _build_hit_and_edge_matrices(
    offsets: np.ndarray,
    obstacles: list[tuple[dict, str]],
    turn_pt: np.ndarray,
    ext_dir: np.ndarray,
    perp: np.ndarray,
    max_range: float = MAX_RANGE,
    min_range: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Along-track distance *and* hit edge index per (ray, obstacle).

    Rays start at ``turn_pt + offset * perp`` and travel along ``ext_dir``.
    Hits closer than ``min_range`` (default ``0``) or at/after ``max_range``
    are ignored; Cat II passes the default 50 km, Cat I passes the leg
    length so only the stretch between the two waypoints counts, and the
    Cat I *start zone* setting raises ``min_range`` so edge crossings right
    after the origin waypoint are skipped (``compute/cat1_start_zone.py``).

    Returns ``(hit_matrix, edge_matrix)``: ``hit_matrix[i, j]`` is the
    distance from ray ``i`` to obstacle ``j`` (``inf`` for a miss) and
    ``edge_matrix[i, j]`` the canonical boundary-edge index that was hit
    (``-1`` for a miss).
    """
    n_rays = len(offsets)
    hit_matrix = np.full((n_rays, len(obstacles)), np.inf, dtype=float)
    edge_matrix = np.full((n_rays, len(obstacles)), -1, dtype=int)
    ray_ys = offsets[:, None]
    for obs_idx, (obs, _kind) in enumerate(obstacles):
        # Bounding-box pre-filter: skip obstacles entirely behind the origin
        # or beyond max_range to keep performance with many split sub-polygons.
        geom_b = obs["geom"]
        bx0, by0, bx1, by1 = geom_b.bounds
        corners = np.array([[bx0, by0], [bx0, by1], [bx1, by0], [bx1, by1]])
        alongs = (corners - turn_pt) @ ext_dir
        if alongs.max() <= min_range or alongs.min() >= max_range:
            continue
        edges = _extract_edges_local(obs["geom"], turn_pt, ext_dir, perp)
        if edges is None or edges.shape[0] == 0:
            continue
        x0, y0 = edges[:, 0, 0], edges[:, 0, 1]
        x1, y1 = edges[:, 1, 0], edges[:, 1, 1]
        y_min = np.minimum(y0, y1)[None, :]
        y_max = np.maximum(y0, y1)[None, :]
        dy = (y1 - y0)[None, :]
        crosses = (ray_ys >= y_min) & (ray_ys < y_max) & (dy != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (ray_ys - y0[None, :]) / dy
            along = x0[None, :] + t * (x1 - x0)[None, :]
        valid = crosses & (along > min_range) & (along < max_range)
        along_valid = np.where(valid, along, np.inf)
        best_edge = np.argmin(along_valid, axis=1)
        best_along = along_valid[np.arange(n_rays), best_edge]
        hit_matrix[:, obs_idx] = best_along
        edge_matrix[:, obs_idx] = np.where(np.isfinite(best_along), best_edge, -1)
    return hit_matrix, edge_matrix


def _accumulate_obs_hits(
    offsets: np.ndarray, masses: np.ndarray,
    hit_matrix: np.ndarray, obstacles: list, recovery: float,
    edge_matrix: np.ndarray | None = None,
) -> tuple[list, dict]:
    """Sum ray masses per first-hit obstacle (and per hit edge when given).

    ``edge_matrix`` is the second value of :func:`_build_hit_and_edge_matrices`;
    with it every accumulator entry also carries ``edges`` -- a
    ``{edge_idx: {mass, weighted_dist, n_rays}}`` split of the same rays,
    which is what the result layers need to colour boundary edges
    individually.
    """
    n_rays = len(offsets)
    best_obs_idx = np.argmin(hit_matrix, axis=1)
    best_dists = hit_matrix[np.arange(n_rays), best_obs_idx]
    hit_mask = np.isfinite(best_dists)
    ray_data: list = []
    obs_accum: dict = defaultdict(lambda: {
        "mass": 0.0, "weighted_dist": 0.0, "p_integral": 0.0,
        "n_rays": 0, "ray_offsets": [], "ray_dists": [], "obs": None, "kind": None,
        "edges": {},
    })
    for i in range(n_rays):
        off, m_i = float(offsets[i]), float(masses[i])
        if not hit_mask[i]:
            ray_data.append((off, m_i, None, None))
            continue
        obs, kind = obstacles[int(best_obs_idx[i])]
        best_key = (kind, obs["id"])
        best_d = float(best_dists[i])
        oa = obs_accum[best_key]
        oa["mass"] += m_i
        oa["weighted_dist"] += m_i * best_d
        if recovery > 0:
            oa["p_integral"] += m_i * exp(-best_d / recovery)
        oa["n_rays"] += 1
        oa["ray_offsets"].append(off)
        oa["ray_dists"].append(best_d)
        oa["obs"], oa["kind"] = obs, kind
        if edge_matrix is not None:
            e_idx = int(edge_matrix[i, int(best_obs_idx[i])])
            if e_idx >= 0:
                ea = oa["edges"].setdefault(e_idx, {"mass": 0.0, "weighted_dist": 0.0, "n_rays": 0})
                ea["mass"] += m_i
                ea["weighted_dist"] += m_i * best_d
                ea["n_rays"] += 1
        ray_data.append((off, m_i, best_key, best_d))
    return ray_data, obs_accum


def _edge_summaries(edge_accum: dict) -> dict[int, dict]:
    """``{edge_idx: {mass, mean_dist, n_rays}}`` from the per-edge accumulator."""
    out: dict[int, dict] = {}
    for e_idx, ea in edge_accum.items():
        mass = ea["mass"]
        out[int(e_idx)] = {
            "mass": mass,
            "mean_dist": ea["weighted_dist"] / mass if mass > 0 else 0.0,
            "n_rays": ea["n_rays"],
        }
    return out


def _build_summaries(obs_accum: dict, ai: float, speed_ms: float) -> dict:
    summaries: dict = {}
    for key, oa in obs_accum.items():
        mean_dist = oa["weighted_dist"] / oa["mass"] if oa["mass"] > 0 else 0.0
        summaries[key] = {
            "mass": oa["mass"],
            "mean_dist": mean_dist,
            "p_integral": oa["p_integral"],
            "p_approx": oa["mass"] * _powered_na(mean_dist, ai, speed_ms),
            "n_rays": oa["n_rays"],
            "ray_offsets": oa["ray_offsets"],
            "ray_dists": oa["ray_dists"],
            "obs": oa["obs"],
            "kind": oa["kind"],
            "edges": _edge_summaries(oa.get("edges", {})),
        }
    return summaries


def _compute_cat2_with_shadows(
    turn_pt: np.ndarray,
    ext_dir: np.ndarray,
    perp: np.ndarray,
    mean_offset: float,
    sigma: float,
    ai: float,
    speed_ms: float,
    obstacles: list[tuple[dict, str]],
) -> tuple[dict, list, np.ndarray, np.ndarray]:
    """Compute Cat II probabilities with shadow effects (vectorised).

    Casts ``N_RAYS`` parallel rays over ``mean +/- RAY_SPAN_SIGMA * sigma``
    (``compute/ray_sampling.py``); per ray keeps the FIRST obstacle hit.
    """
    offsets = ray_offsets(mean_offset, sigma)
    dx = offsets[1] - offsets[0]
    pdf_vals = norm.pdf(offsets, mean_offset, sigma)
    masses = pdf_vals * dx
    n_rays = len(offsets)
    if not obstacles:
        return ({}, [(float(offsets[i]), float(masses[i]), None, None) for i in range(n_rays)],
                offsets, pdf_vals)
    hit_matrix, edge_matrix = _build_hit_and_edge_matrices(offsets, obstacles, turn_pt, ext_dir, perp)
    ray_data, obs_accum = _accumulate_obs_hits(offsets, masses, hit_matrix, obstacles,
                                               ai * speed_ms, edge_matrix=edge_matrix)
    return _build_summaries(obs_accum, ai, speed_ms), ray_data, offsets, pdf_vals


# ---------------------------------------------------------------------------
# Shadow-aware Cat I computation (obstacle inside the lane)
# ---------------------------------------------------------------------------

def _build_cat1_summaries(obs_accum: dict) -> dict:
    """Per-obstacle Cat I summary.

    Same keys as :func:`_build_summaries` so the two categories can be
    consumed by one loop; ``p_approx`` is the intercepted mass itself
    because Cat I has no distance decay (Hansen eq. 4.15,
    ``N_I = Pc * Q * mass``).  ``p_integral`` equals ``mass`` for the
    same reason.
    """
    summaries: dict = {}
    for key, oa in obs_accum.items():
        mean_dist = oa["weighted_dist"] / oa["mass"] if oa["mass"] > 0 else 0.0
        summaries[key] = {
            "mass": oa["mass"],
            "mean_dist": mean_dist,
            "p_integral": oa["mass"],
            "p_approx": oa["mass"],
            "n_rays": oa["n_rays"],
            "ray_offsets": oa["ray_offsets"],
            "ray_dists": oa["ray_dists"],
            "obs": oa["obs"],
            "kind": oa["kind"],
            "edges": _edge_summaries(oa.get("edges", {})),
        }
    return summaries


def _compute_cat1_in_lane(
    origin: np.ndarray,
    along_dir: np.ndarray,
    perp: np.ndarray,
    mean_offset: float,
    sigma: float,
    leg_length: float,
    obstacles: list[tuple[dict, str]],
    start_zone_frac: float = 0.0,
) -> tuple[dict, list, np.ndarray, np.ndarray]:
    """Compute Cat I (in-lane) intercepted mass per obstacle (vectorised).

    Casts ``N_RAYS`` rays parallel to the leg from ``origin`` (the
    waypoint the ships *come from*) and keeps, per ray, the first
    obstacle hit within ``leg_length``.  Obstacles beyond the far
    waypoint belong to Cat II of this leg and are ignored here, so the
    two categories never count the same water twice.

    ``start_zone_frac`` (0..1) is the Cat I start-zone setting: edge
    crossings within ``start_zone_frac * leg_length`` of the origin are
    ignored, so an obstacle lying right where the ships enter the leg is
    not hit by this direction's rays.

    The intercepted mass per obstacle is the numerical form of
    ``Phi((z_max - mu) / sigma) - Phi((z_min - mu) / sigma)`` from the
    Cat I formula, evaluated with shadowing along the leg.

    Returns ``(summaries, ray_data, offsets, pdf_vals)`` with the same
    layout as :func:`_compute_cat2_with_shadows`; ``ray_data`` holds
    ``(offset, mass, hit_key | None, along_dist | None)`` tuples where
    the distance is measured from ``origin``.
    """
    offsets = ray_offsets(mean_offset, sigma)
    dx = offsets[1] - offsets[0]
    pdf_vals = norm.pdf(offsets, mean_offset, sigma)
    masses = pdf_vals * dx
    n_rays = len(offsets)
    if not obstacles or leg_length <= 0:
        return ({}, [(float(offsets[i]), float(masses[i]), None, None) for i in range(n_rays)],
                offsets, pdf_vals)
    min_range = max(0.0, min(1.0, float(start_zone_frac))) * leg_length
    hit_matrix, edge_matrix = _build_hit_and_edge_matrices(
        offsets, obstacles, origin, along_dir, perp, max_range=leg_length, min_range=min_range)
    ray_data, obs_accum = _accumulate_obs_hits(offsets, masses, hit_matrix, obstacles, 0.0,
                                               edge_matrix=edge_matrix)
    return _build_cat1_summaries(obs_accum), ray_data, offsets, pdf_vals


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _total_p_for_comp(comp: dict) -> float:
    """Sum the per-obstacle ``p_approx`` values for a computation.

    Covers both the Cat II summaries and the Cat I block (when
    present) so the sidebar ranks (Leg, Direction) pairs by their full
    powered contribution.
    """
    total = 0.0
    blocks = [comp.get("summaries") or {}]
    cat1 = comp.get("cat1") or {}
    blocks.append(cat1.get("summaries") or {})
    for block in blocks:
        for s in block.values():
            try:
                total += float(s.get("p_approx", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _load_depth_obstacles(data: dict, proj: "SimpleProjector") -> list[dict]:
    """Load depth obstacles, splitting MultiPolygons into individual Polygons.

    Each sub-polygon becomes its own obstacle with ID ``'<did>_<i>'``
    so the ray caster assigns separate mass/d_mean to each patch.
    """
    result: list[dict] = []
    for dep in data.get("depths", []):
        try:
            did, depth_val, wkt_str = dep
            geom = sw.loads(wkt_str)
            proj_geom = _project_wkt_geom(geom, proj)
            depth_f = float(depth_val)
        except Exception:  # nosec B110 B112
            continue
        if proj_geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(proj_geom.geoms):
                result.append({"id": f"{did}_{i}", "depth": depth_f, "geom": poly})
        else:
            result.append({"id": str(did), "depth": depth_f, "geom": proj_geom})
    return result


def _load_object_obstacles(data: dict, proj: "SimpleProjector") -> list[dict]:
    """Load structure obstacles, splitting MultiPolygons into individual Polygons.

    Each sub-polygon becomes its own obstacle with ID ``'<oid>_<i>'``
    so the ray caster assigns separate mass/d_mean to each structure patch.
    """
    result: list[dict] = []
    for obj in data.get("objects", []):
        try:
            oid, height, wkt_str = obj
            geom = sw.loads(wkt_str)
            proj_geom = _project_wkt_geom(geom, proj)
            height_f = float(height) if height else 0.0
        except Exception:  # nosec B110 B112
            continue
        if proj_geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(proj_geom.geoms):
                result.append({"id": f"{oid}_{i}", "height": height_f, "geom": poly})
        else:
            result.append({"id": str(oid), "height": height_f, "geom": proj_geom})
    return result


def _build_leg_entry(seg_id: str, seg: dict, traffic: dict, proj: "SimpleProjector") -> dict:
    lon_s, lat_s = _parse_point(seg["Start_Point"])
    lon_e, lat_e = _parse_point(seg["End_Point"])
    xs, ys = proj.transform(lon_s, lat_s)
    xe, ye = proj.transform(lon_e, lat_e)
    dirs = seg.get("Dirs", ["Dir 1", "Dir 2"])
    ai_values = [float(seg.get("ai1", 180)), float(seg.get("ai2", 180))]
    seg_traffic = traffic.get(seg_id, {})
    dir_info: list[dict] = []
    for i, d_name in enumerate(dirs):
        spd_kn = _weighted_avg_speed_knots(seg_traffic[d_name]) if d_name in seg_traffic else 0
        dir_info.append({
            "name": d_name, "speed_kn": spd_kn, "speed_ms": spd_kn * 1852.0 / 3600.0,
            "ai": ai_values[min(i, 1)],
            "mean": float(seg.get(f"mean{i + 1}_1", 0)),
            "std": float(seg.get(f"std{i + 1}_1", 100)),
        })
    return {"start": np.array([xs, ys]), "end": np.array([xe, ye]),
            "name": seg.get("Leg_name", ""), "start_wkt": seg["Start_Point"],
            "end_wkt": seg["End_Point"], "dirs": dir_info}


def _build_legs_and_obstacles(
    data: dict[str, Any],
    proj: SimpleProjector,
    mode: str,
    max_draft: float,
) -> tuple[dict[str, dict], list[tuple[dict, str]], list[dict], list[dict], list[dict]]:
    """Extract projected legs, obstacle list, and raw geometry lists from *data*."""
    depth_geoms_all = _load_depth_obstacles(data, proj)
    depth_geoms = [d for d in depth_geoms_all if d["depth"] <= max_draft]
    depth_geoms_deep = [d for d in depth_geoms_all if d["depth"] > max_draft]
    object_geoms = _load_object_obstacles(data, proj)
    all_obstacles: list[tuple[dict, str]] = []
    if mode in ("grounding", "both"):
        all_obstacles.extend([(dg, "depth") for dg in depth_geoms])
    if mode in ("allision", "both"):
        all_obstacles.extend([(og, "object") for og in object_geoms])
    legs: dict[str, dict] = {}
    traffic = data.get("traffic_data", {})
    for seg_id, seg in data.get("segment_data", {}).items():
        try:
            legs[seg_id] = _build_leg_entry(seg_id, seg, traffic, proj)
        except Exception:  # nosec B110 B112
            continue
    return legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms


def _run_all_computations(
    legs: dict[str, dict],
    all_obstacles: list[tuple[dict, str]],
    cat1_start_zone_frac: float = 0.0,
) -> list[dict]:
    """Run the shadow-aware Cat I and Cat II computations for every leg/direction.

    Each returned computation carries the Cat II result in ``summaries`` /
    ``ray_data`` (rays from the turning point onwards) and the Cat I
    result under ``cat1`` (rays from the origin waypoint along the leg,
    clipped at the turning point).  A (leg, direction) pair is included
    when either category hits at least one obstacle.

    ``cat1_start_zone_frac`` is forwarded to :func:`_compute_cat1_in_lane`
    (see ``compute/cat1_start_zone.py``); Cat II never uses it.
    """
    computations: list[dict] = []
    for seg_id, leg in legs.items():
        start, end = leg["start"], leg["end"]
        u, n, L = _leg_vectors(start, end)

        for di, d in enumerate(leg["dirs"]):
            if d["speed_ms"] <= 0:
                continue

            if di == 0:
                origin = start.copy()
                turn_pt = end.copy()
                ext_dir = u.copy()
            else:
                origin = end.copy()
                turn_pt = start.copy()
                ext_dir = (-u).copy()

            summaries, ray_data, offsets, pdf_vals = _compute_cat2_with_shadows(
                turn_pt, ext_dir, n, d["mean"], d["std"],
                d["ai"], d["speed_ms"], all_obstacles,
            )
            cat1_summaries, cat1_ray_data, _, _ = _compute_cat1_in_lane(
                origin, ext_dir, n, d["mean"], d["std"], L, all_obstacles,
                start_zone_frac=cat1_start_zone_frac,
            )

            if summaries or cat1_summaries:
                computations.append({
                    "seg_id": seg_id,
                    "leg": leg,
                    "dir_idx": di,
                    "dir_info": d,
                    "turn_pt": turn_pt,
                    "ext_dir": ext_dir,
                    "perp": n.copy(),
                    "summaries": summaries,
                    "ray_data": ray_data,
                    "offsets": offsets,
                    "pdf_vals": pdf_vals,
                    "start": start,
                    "end": end,
                    "cat1": {
                        "origin": origin,
                        "along_dir": ext_dir,
                        "length": L,
                        "start_zone_m": float(cat1_start_zone_frac) * L,
                        "summaries": cat1_summaries,
                        "ray_data": cat1_ray_data,
                    },
                })
    return computations


# ---------------------------------------------------------------------------
# PoweredOverlapVisualizer
# ---------------------------------------------------------------------------

LEG_GREY = "#8c8c8c"
RAYS_PER_OBSTACLE = 18      # rays drawn per hit obstacle and category
P_FLOOR = 1e-15             # lower clip of the log-scale Cat II probability panel
MAX_LABELS = 6              # obstacles named per panel (largest first)
INSET_FRAC = 0.16           # left part of a local panel holding the lateral-distribution inset
MAX_STRETCH = 2.5           # a fitted window may be at most this many times taller than what it must show
LANE_SIGMAS = 2.0           # a local window always shows the lane mean +/- this many sigma
MIN_VIEW_SPAN = 500.0       # m -- smallest window height
VIEW_HINT = (
    "How to use this view\n"
    "\n"
    "- Pick a leg and direction in the table\n"
    "  on the right; the map highlights it and\n"
    "  the panels below follow.\n"
    "- The Cat I / Cat II panels are true to\n"
    "  scale and zoomed to the obstacles hit;\n"
    "  the dashed boxes on the map show where.\n"
    "- To look closer, press the magnifier in\n"
    "  the toolbar above and drag a box; the\n"
    "  arrows button pans and the house button\n"
    "  resets the panels."
)


def _obs_tag(key: tuple) -> str:
    kind, obs_id = key
    return f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"


def _left_normal(direction: np.ndarray) -> np.ndarray:
    return np.array([-float(direction[1]), float(direction[0])])


def _side_sign(along: np.ndarray, perp: np.ndarray) -> float:
    """``+1`` when ``perp`` is the left normal of ``along``, else ``-1``.

    Ray offsets are measured along the leg's ``perp`` (left of the *drawn*
    direction).  Direction 2 travels the other way, so its local panels
    flip the sign to keep "+ = left of the heading": the panels are the
    map rotated, never mirrored.
    """
    return 1.0 if float(np.dot(_left_normal(along), perp)) >= 0.0 else -1.0


def _sample_hit_rays(ray_data: list | None, per_obstacle: int = RAYS_PER_OBSTACLE) -> list:
    """Evenly thinned hit rays: about ``per_obstacle`` and at least one per hit obstacle."""
    by_key: dict = defaultdict(list)
    for ray in ray_data or []:
        if ray[2] is not None:
            by_key[ray[2]].append(ray)
    out: list = []
    for rays in by_key.values():
        out.extend(rays[::max(1, len(rays) // per_obstacle)])
    return out


def _bounds_array(geoms: list[dict]) -> np.ndarray:
    """``(N, 4)`` bounding boxes of ``[{"geom": ...}, ...]``."""
    if not geoms:
        return np.zeros((0, 4))
    return np.array([g["geom"].bounds for g in geoms], dtype=float)


def _bounds_in_view(bounds: np.ndarray, origin, along, left, view) -> np.ndarray:
    """Mask of the bounding boxes that overlap ``view = (x0, x1, y0, y1)`` of a local frame.

    A cheap pre-filter, so a project with thousands of depth polygons only
    projects and draws the few that can be seen in a panel.
    """
    if bounds.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    corners = np.stack(
        [bounds[:, [0, 1]], bounds[:, [0, 3]], bounds[:, [2, 1]], bounds[:, [2, 3]]], axis=1,
    ) - np.asarray(origin, dtype=float)
    xs = corners @ np.asarray(along, dtype=float)
    ys = corners @ np.asarray(left, dtype=float)
    x0, x1, y0, y1 = view
    return ((xs.max(axis=1) >= x0) & (xs.min(axis=1) <= x1)
            & (ys.max(axis=1) >= y0) & (ys.min(axis=1) <= y1))


def _grown(view: tuple, factor: float = 0.5) -> tuple:
    """``view`` grown by ``factor`` of its size on every side (obstacle pre-filter margin)."""
    x0, x1, y0, y1 = view
    dx, dy = factor * (x1 - x0), factor * (y1 - y0)
    return x0 - dx, x1 + dx, y0 - dy, y1 + dy


def _label_box(color) -> dict:
    return dict(fc="white", alpha=0.92, ec=color, pad=2, boxstyle="round,pad=0.3")


def local_view(
    lane: tuple[float, float],
    hits: list[tuple[float, float, float, float]],
    anchors: tuple[float, ...] = (),
    box_ratio: float = 0.6,
) -> tuple[float, float, float, float]:
    """True-scale window ``(x0, x1, y0, y1)`` of a Cat I / Cat II panel.

    ``lane`` is the lateral range always shown (the lane mean +/- a few
    sigma), ``hits`` one ``(x_min, x_max, y_min, y_max)`` box of hit points
    per obstacle, largest contributor first.  The first box is always in
    the window; the next ones and the ``anchors`` (x positions such as the
    ray start or the turning point) join only while the window, once
    fitted to the panel, stays at most ``MAX_STRETCH`` times taller than
    the lane and hits it has to show -- with an equal aspect ratio a
    longer window would squash the lane to a line, so farther hits are
    left to the toolbar's pan / zoom.  Without hits the window starts at
    the first anchor.

    The left ``INSET_FRAC`` is reserved for the lateral-distribution inset
    and the result has the panel's height / width ``box_ratio``, so an
    equal aspect ratio shows all of it.
    """
    y0, y1 = float(min(lane)), float(max(lane))
    ratio = box_ratio if np.isfinite(box_ratio) and box_ratio > 0 else 0.6

    def _fits(nx0: float, nx1: float, ny0: float, ny1: float) -> bool:
        height = max(ny1 - ny0, MIN_VIEW_SPAN) * 1.1
        width = (nx1 - nx0) * 1.1 / (1.0 - INSET_FRAC)
        return width * ratio <= MAX_STRETCH * height

    if hits:
        x0, x1 = float(hits[0][0]), float(hits[0][1])
        y0, y1 = min(y0, float(hits[0][2])), max(y1, float(hits[0][3]))
        for hx0, hx1, hy0, hy1 in hits[1:]:
            box = (min(x0, hx0), max(x1, hx1), min(y0, hy0), max(y1, hy1))
            if _fits(*box):
                x0, x1, y0, y1 = (float(v) for v in box)
    else:
        x0 = float(anchors[0]) if anchors else 0.0
        x1 = x0 + 0.8 * max(y1 - y0, MIN_VIEW_SPAN) / ratio
    for a in anchors:
        if _fits(min(x0, a), max(x1, a), y0, y1):
            x0, x1 = min(x0, float(a)), max(x1, float(a))

    h = max(y1 - y0, MIN_VIEW_SPAN) * 1.1
    cy = (y0 + y1) / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    w = max(x1 - x0, 1.0)
    x0, x1 = x0 - 0.05 * w, x1 + 0.05 * w
    # The inset covers the left INSET_FRAC of the panel, over the padding.
    w = (x1 - x0) / (1.0 - INSET_FRAC)
    x0 = x1 - w
    if h < w * ratio:
        h = w * ratio
        y0, y1 = cy - h / 2.0, cy + h / 2.0
    else:
        grow = h / ratio - w
        x0, x1 = x0 - grow / 2.0, x1 + grow / 2.0
    return x0, x1, y0, y1


class PoweredOverlapVisualizer:
    """Interactive visualizer for powered (Cat I + Cat II) allision/grounding geometry.

    Panels, by ``axes`` key (``info`` and ``cat2`` are optional, so a bare
    three-axes figure still works):

    * ``overview`` -- every leg and obstacle; the selected (leg, direction)
      is highlighted with its band and travel-direction arrow, and the
      windows of the two local panels are outlined.  ``info`` beside it
      holds the legend and how to zoom with the toolbar.
    * ``detail`` -- **Cat I**: the leg in its own frame (ships enter at
      ``x = 0``, the turning point is at ``x = L``) with the rays that hit
      an obstacle inside the lane.  ``P_I = mass``.
    * ``cat2`` -- **Cat II** geometry: the rays of the ships that miss the
      turn and continue straight past the turning point.
    * ``waterfall`` -- **Cat II** probability on the same x-axis as
      ``cat2``: the survival curve ``exp(-d / (ai V))`` and
      ``P_II = mass x survival`` per obstacle, on a log scale.

    The local panels are the map rotated so the ships travel left to right
    (+y = left of the heading), drawn to true scale (equal aspect) and
    zoomed by :func:`local_view` to the obstacles hit.  The lateral
    distribution is an inset at their left edge.  The (leg, direction) is
    picked in the dialog's table only.

    Parameters
    ----------
    fig, axes : Figure, dict
        The figure and its panels (keys above).
    legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms, computations :
        Output of ``_build_legs_and_obstacles`` and ``_run_all_computations``.
    mode : str
        ``"allision"`` or ``"grounding"``.
    segment_data : dict, optional
        Names the legs as the route table does (``compute/leg_labels.py``).
    breakdown_rows : list, optional
        ``compute.powered_breakdown.build_sidebar_rows`` output: the first
        row is selected initially and the overview title shows the leg's
        annual frequency and share of the total.
    """

    def __init__(
        self,
        fig: Figure,
        axes: dict[str, Axes],
        legs: dict[str, dict],
        all_obstacles: list[tuple[dict, str]],
        depth_geoms: list[dict],
        depth_geoms_deep: list[dict],
        object_geoms: list[dict],
        computations: list[dict],
        mode: str,
        segment_data: dict[str, Any] | None = None,
        breakdown_rows: list[dict] | None = None,
    ) -> None:
        self.fig = fig
        self.axes = axes
        self.legs = legs
        self.all_obstacles = all_obstacles
        self.depth_geoms = depth_geoms
        self.depth_geoms_deep = depth_geoms_deep
        self.object_geoms = object_geoms
        self.computations = computations
        self.mode = mode
        self.segment_data = segment_data or {}
        self.breakdown_rows = list(breakdown_rows or [])
        self._rows_by_comp = {r["comp_idx"]: r for r in self.breakdown_rows}

        self.dir_colors = ["#d62728", "#2ca02c"]
        all_obs_keys: set[tuple] = set()
        for comp in computations:
            all_obs_keys.update(comp["summaries"].keys())
            all_obs_keys.update(((comp.get("cat1") or {}).get("summaries") or {}).keys())
        cmap = plt.cm.tab10
        self.obs_color_map: dict[tuple, Any] = {
            key: cmap(i % 10)
            for i, key in enumerate(sorted(all_obs_keys))
        }
        self._obstacle_bounds = _bounds_array([obs for obs, _kind in all_obstacles])
        self._layer_bounds = {
            "deep": _bounds_array(depth_geoms_deep),
            "depth": _bounds_array(depth_geoms),
            "object": _bounds_array(object_geoms),
        }

        self._selected_comp_idx: int | None = None
        self._overview_highlight: list[Any] = []
        self.views: dict[str, tuple | None] = {"cat1": None, "cat2": None}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run_visualization(self) -> None:
        """Draw the overview, select the largest contributor and draw its panels."""
        self._draw_overview()
        self._refresh_layout()
        if self.computations:
            self._select_computation(self.default_computation_index())
        self._connect_resize()

    def _connect_resize(self) -> None:
        """Refit the true-scale windows after the canvas changes size.

        The windows follow the panels' shape, which changes when the dialog
        opens or is resized.  A single-shot timer waits until the resize
        settles instead of redrawing on every intermediate size.
        """
        canvas = self.fig.canvas
        try:
            timer = canvas.new_timer(interval=250)
            timer.single_shot = True
            timer.add_callback(self._refit_after_resize)
            self._resize_timer = timer
            canvas.mpl_connect("resize_event", self._schedule_refit)
        except Exception:  # nosec B110 B112
            self._resize_timer = None

    def _schedule_refit(self, _event=None) -> None:
        timer = getattr(self, "_resize_timer", None)
        if timer is not None:
            timer.stop()
            timer.start()

    def _refit_after_resize(self) -> None:
        if self._selected_comp_idx is not None:
            self._select_computation(self._selected_comp_idx)

    def default_computation_index(self) -> int:
        """The first breakdown row (largest contributor), else computation 0."""
        if self.breakdown_rows:
            return int(self.breakdown_rows[0]["comp_idx"])
        return 0

    def leg_title(self, seg_id: Any) -> str:
        """Route-table name of a leg (``LEG_6_1_b``)."""
        return _leg_title(self.segment_data, seg_id, (self.legs.get(str(seg_id)) or {}).get("name"))

    def comp_title(self, comp: dict) -> str:
        """``"LEG_6_1_b East going"`` for a computation."""
        return f"{self.leg_title(comp['seg_id'])} {comp['dir_info']['name']}"

    # -----------------------------------------------------------------
    # Frames and windows
    # -----------------------------------------------------------------

    @staticmethod
    def _origin(comp: dict) -> np.ndarray:
        """Waypoint where the ships of ``comp`` enter the leg."""
        cat1 = comp.get("cat1") or {}
        if "origin" in cat1:
            return np.asarray(cat1["origin"], dtype=float)
        return np.asarray(comp["start"] if comp["dir_idx"] == 0 else comp["end"], dtype=float)

    def _frame(self, comp: dict, category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """``(base, along, left, sign)`` of a local panel; ``sign`` maps ray offsets to +y = left."""
        perp = np.asarray(comp["perp"], dtype=float)
        if category == "cat1":
            along = np.asarray((comp.get("cat1") or {}).get("along_dir", comp["ext_dir"]), dtype=float)
            base = self._origin(comp)
        else:
            along = np.asarray(comp["ext_dir"], dtype=float)
            base = np.asarray(comp["turn_pt"], dtype=float)
        sign = _side_sign(along, perp)
        return base, along, sign * perp, sign

    @staticmethod
    def _summaries(comp: dict, category: str) -> dict:
        if category == "cat1":
            return (comp.get("cat1") or {}).get("summaries") or {}
        return comp.get("summaries") or {}

    def _box_ratio(self, ax: Axes | None) -> float:
        """Height / width of ``ax`` on the page (inches)."""
        if ax is None:
            return 0.6
        try:
            pos = ax.get_position(original=True)
            ratio = (pos.height * self.fig.get_figheight()) / (pos.width * self.fig.get_figwidth())
        except Exception:  # nosec B110 B112
            return 0.6
        return float(ratio) if np.isfinite(ratio) and ratio > 0 else 0.6

    def _local_view(self, comp: dict, category: str, ax: Axes | None) -> tuple[float, float, float, float]:
        """The :func:`local_view` window of a category's panel, for the current layout."""
        d = comp["dir_info"]
        sign = self._frame(comp, category)[3]
        lane = tuple(sorted((sign * (d["mean"] - LANE_SIGMAS * d["std"]),
                             sign * (d["mean"] + LANE_SIGMAS * d["std"]))))
        ranked = sorted(self._summaries(comp, category).items(),
                        key=lambda kv: -float(kv[1].get("p_approx", 0.0) or 0.0))
        hits = []
        for _key, s in ranked:
            dists, offs = s.get("ray_dists") or [], s.get("ray_offsets") or []
            if dists and offs:
                ys = [sign * o for o in offs]
                hits.append((min(dists), max(dists), min(ys), max(ys)))
        if category == "cat1":
            leg_len = float((comp.get("cat1") or {}).get("length", 0.0) or 0.0)
            anchors: tuple[float, ...] = (0.0, leg_len) if leg_len > 0 else (0.0,)
        else:
            anchors = (0.0,)
        return local_view(lane, hits, anchors, self._box_ratio(ax))

    def _refresh_layout(self) -> None:
        """Run the figure's layout engine now, so the panel boxes are final before sizing windows."""
        try:
            engine = self.fig.get_layout_engine()
            if engine is not None:
                engine.execute(self.fig)
        except Exception:  # nosec B110 B112
            pass

    # -----------------------------------------------------------------
    # Shared drawing helpers
    # -----------------------------------------------------------------

    def _dir_color(self, comp: dict):
        return self.dir_colors[int(comp.get("dir_idx", 0)) % 2]

    def _plot_obstacle_layers(self, ax: Axes) -> None:
        """Deep (ignored), shallow and structure polygons."""
        styles = (
            (self.depth_geoms_deep, dict(facecolor="#e0e0e0", alpha=0.3, edgecolor="#aaaaaa", linewidth=0.3)),
            (self.depth_geoms, dict(facecolor="lightblue", alpha=0.5, edgecolor="steelblue", linewidth=0.6)),
            (self.object_geoms, dict(facecolor="salmon", alpha=0.5, edgecolor="darkred", linewidth=0.6)),
        )
        for geoms, style in styles:
            for g in geoms:
                _plot_geom(ax, g["geom"], **style)

    def _plot_local_obstacles(self, ax: Axes, origin, along, left, view, hit_keys: set) -> None:
        """Obstacles inside ``view`` of a local frame; those in ``hit_keys`` outlined in their colour."""
        mask = _bounds_in_view(self._obstacle_bounds, origin, along, left, view)
        for (obs, kind), keep in zip(self.all_obstacles, mask):
            if not keep:
                continue
            local = _project_to_local(obs["geom"], origin, along, left)
            key = (kind, obs["id"])
            face = "lightblue" if kind == "depth" else "salmon"
            if key in hit_keys:
                _plot_geom(ax, local, facecolor=face, alpha=0.5,
                           edgecolor=self.obs_color_map.get(key, "black"), linewidth=1.5, zorder=1)
            else:
                _plot_geom(ax, local, facecolor=face, alpha=0.3,
                           edgecolor="steelblue" if kind == "depth" else "darkred", linewidth=0.5, zorder=1)

    def _plot_pdf_inset(self, ax: Axes, y: np.ndarray, pdf_vals, ray_data: list) -> None:
        """Lateral distribution in the left ``INSET_FRAC`` of the panel, filled per hit obstacle.

        x is in axes fraction and y in data, so the curve lines up with the
        rays whatever part of the leg the window shows.
        """
        pdf = np.asarray(pdf_vals, dtype=float)
        peak = float(pdf.max()) if pdf.size else 0.0
        if peak <= 0.0:
            return
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.add_patch(Rectangle((0.0, 0.0), INSET_FRAC, 1.0, transform=ax.transAxes,
                               facecolor="white", edgecolor="none", alpha=0.95, zorder=6))
        ax.plot([INSET_FRAC, INSET_FRAC], [0.0, 1.0], transform=ax.transAxes, color="black",
                linewidth=0.8, zorder=7)
        right = 0.96 * INSET_FRAC
        curve = right - pdf / peak * 0.85 * INSET_FRAC
        ax.fill_betweenx(y, curve, right, transform=trans, color="#e6e6e6", linewidth=0, zorder=7)
        keys = [r[2] for r in ray_data] if len(ray_data) == len(y) else []
        for key in {k for k in keys if k is not None}:
            mask = np.array([k == key for k in keys])
            ax.fill_betweenx(y, curve, right, where=mask, transform=trans,
                             color=self.obs_color_map.get(key, "#888888"), alpha=0.85, linewidth=0, zorder=7)
        ax.plot(curve, y, transform=trans, color="black", linewidth=1.0, zorder=8)
        ax.text(INSET_FRAC / 2.0, 0.99, "lateral\ndistribution", transform=ax.transAxes,
                ha="center", va="top", fontsize=6, zorder=8)

    def _plot_local_rays(self, ax: Axes, ray_data: list | None, sign: float) -> None:
        for off, _m, key, dist in _sample_hit_rays(ray_data):
            ax.plot([0.0, dist], [sign * off, sign * off], color=self.obs_color_map.get(key, "#888888"),
                    alpha=0.55, linewidth=0.7, zorder=3)

    def _obstacle_legend(self, ax: Axes, summaries: dict, fmt) -> None:
        """Legend naming the ``MAX_LABELS`` largest hit obstacles in their colours."""
        if not summaries:
            return
        ranked = sorted(summaries.items(), key=lambda kv: -float(kv[1].get("p_approx", 0.0) or 0.0))
        handles = [
            Patch(facecolor=self.obs_color_map.get(key, "#888888"), alpha=0.7,
                  edgecolor=self.obs_color_map.get(key, "#888888"), label=fmt(key, s))
            for key, s in ranked[:MAX_LABELS]
        ]
        if len(ranked) > MAX_LABELS:
            handles.append(Patch(facecolor="none", edgecolor="none",
                                 label=f"+{len(ranked) - MAX_LABELS} smaller obstacles"))
        ax.legend(handles=handles, fontsize=6, loc="best", framealpha=0.9)

    def _annotate_column(self, ax: Axes, summaries: dict, fmt, y_of) -> None:
        """Label the hit obstacles in a column at the right edge, with a leader line to ``y_of(s)``.

        Evenly spaced so the boxes never overlap; only the ``MAX_LABELS``
        largest (by ``p_approx``) are named.
        """
        if not summaries:
            return
        ranked = sorted(summaries.items(), key=lambda kv: -float(kv[1].get("p_approx", 0.0) or 0.0))
        shown = sorted(ranked[:MAX_LABELS], key=lambda kv: -y_of(kv[1]))
        n = len(shown)
        for i, (key, s) in enumerate(shown):
            y_frac = 0.88 - 0.76 * (i / (n - 1)) if n > 1 else 0.5
            c = self.obs_color_map.get(key, "#888888")
            ax.annotate(
                fmt(key, s), xy=(float(s["mean_dist"]), y_of(s)),
                xytext=(0.985, y_frac), textcoords="axes fraction",
                ha="right", va="center", fontsize=6, color=c, fontweight="bold",
                bbox=_label_box(c), arrowprops=dict(arrowstyle="-", color=c, alpha=0.6), zorder=8,
            )
        if len(ranked) > MAX_LABELS:
            ax.text(0.985, 0.98, f"+{len(ranked) - MAX_LABELS} smaller obstacles", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color="dimgray")

    def _set_local_view(self, ax: Axes | None, view: tuple, start_label: str) -> None:
        """Apply a true-scale window; say how far the ray start is when it is out of view."""
        if ax is None:
            return
        x0, x1, y0, y1 = view
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        visible_left = x0 + INSET_FRAC * (x1 - x0)
        if visible_left > 0.0:
            ax.text(INSET_FRAC / 2.0, 0.01, f"{start_label}\n{visible_left / 1000:.1f} km\nto the left",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color="dimgray", zorder=8)

    def _frequency_text(self, comp_idx: int | None, comp: dict) -> str:
        row = self._rows_by_comp.get(comp_idx) if comp_idx is not None else None
        if row and row.get("frequency") is not None:
            return (f"{format_frequency(row['frequency'])} per year = "
                    f"{format_share(row['share'])} of the powered {self.mode} total")
        return f"P(hit) per ship = {geometric_probability(comp):.3e} (no annual frequency for this run)"

    # -----------------------------------------------------------------
    # Overview panel
    # -----------------------------------------------------------------

    def _draw_overview(self) -> None:
        ax = self.axes["overview"]
        ax.set_title(f"All legs -- powered {self.mode}", fontsize=9)
        self._plot_obstacle_layers(ax)
        for leg in self.legs.values():
            start, end = leg["start"], leg["end"]
            ax.plot([start[0], end[0]], [start[1], end[1]], "-", color=LEG_GREY, linewidth=1.2, zorder=4)
        info = self.axes.get("info")
        if info is not None:
            info.axis("off")
            info.legend(handles=self._overview_legend(), fontsize=7, loc="upper left", frameon=False)
            info.text(0.0, 0.0, VIEW_HINT, transform=info.transAxes, ha="left", va="bottom", fontsize=7,
                      bbox=dict(fc="#f7f7f7", ec="#cccccc", pad=4))
        else:
            ax.legend(handles=self._overview_legend(), fontsize=6, loc="upper left")
        ax.set_xlabel("Easting (m)", fontsize=7)
        ax.set_ylabel("Northing (m)", fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        # Freeze the extent: the selection highlight must not rescale the map.
        ax.autoscale_view()
        ax.set_autoscale_on(False)

    def _overview_legend(self) -> list:
        items: list = []
        if self.depth_geoms:
            items.append(Patch(facecolor="lightblue", edgecolor="steelblue", label="Depth areas (grounding)"))
        if self.object_geoms:
            items.append(Patch(facecolor="salmon", edgecolor="darkred", label="Objects (allision)"))
        if self.depth_geoms_deep:
            items.append(Patch(facecolor="#e0e0e0", edgecolor="#aaaaaa", label="Deep (ignored)"))
        items.append(Line2D([0], [0], color=LEG_GREY, lw=1.2, label="Route legs"))
        for di, dc in enumerate(self.dir_colors):
            items.append(Patch(facecolor=dc, alpha=0.4,
                               label=f"Selected leg, dir {di + 1} ({N_SIGMA} sigma band, arrow = heading)"))
        items.append(Line2D([0], [0], color="black", lw=0.9, ls="--", label="Area of the Cat I / Cat II panels"))
        return items

    def _highlight_selected(self, comp: dict) -> None:
        """Mark the selected (leg, direction) on the overview and outline the local panels' windows."""
        ax = self.axes["overview"]
        for artist in self._overview_highlight:
            try:
                artist.remove()
            except Exception:  # nosec B110 B112
                pass
        dc = self._dir_color(comp)
        d = comp["dir_info"]
        origin, tp = self._origin(comp), np.asarray(comp["turn_pt"], dtype=float)
        band = _make_band_polygon(comp["start"], comp["end"], d["mean"], d["std"])
        bx, by = band.exterior.xy
        artists: list = list(ax.fill(bx, by, facecolor=dc, alpha=0.35, edgecolor=dc, linewidth=0.8, zorder=7))
        artists += ax.plot([origin[0], tp[0]], [origin[1], tp[1]], "-", color=dc, linewidth=3.0, zorder=8)
        artists.append(ax.annotate(
            "", xy=(tp[0], tp[1]), xytext=(origin[0], origin[1]),
            arrowprops=dict(arrowstyle="-|>", color=dc, lw=1.2, mutation_scale=16), zorder=9,
        ))
        for category, label in (("cat1", "Cat I"), ("cat2", "Cat II")):
            view = self.views.get(category)
            if view is None:
                continue
            base, along, left, _sign = self._frame(comp, category)
            x0, x1, y0, y1 = view
            local = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
            world = base + local[:, :1] * along + local[:, 1:] * left
            artists += ax.plot(world[:, 0], world[:, 1], "--", color="black", linewidth=0.9, zorder=9)
            artists.append(ax.annotate(label, xy=(world[2, 0], world[2, 1]), xytext=(3, 3),
                                       textcoords="offset points", fontsize=6, zorder=10))
        mid = (origin + tp) / 2.0
        artists.append(ax.annotate(
            self.comp_title(comp), xy=(mid[0], mid[1]), xytext=(6, 6), textcoords="offset points",
            fontsize=7, fontweight="bold", color=dc, bbox=dict(fc="white", alpha=0.85, ec=dc, pad=1.5),
            zorder=10,
        ))
        self._overview_highlight = artists
        ax.set_title(
            f"Selected: {self.comp_title(comp)}\n{self._frequency_text(self._selected_comp_idx, comp)}",
            fontsize=9, fontweight="bold",
        )

    # -----------------------------------------------------------------
    # Cat I panel (bottom left)
    # -----------------------------------------------------------------

    def _draw_detail(self, comp: dict, view: tuple) -> None:
        """Bottom-left: Cat I -- obstacles inside the lane, in the leg's own frame, true scale."""
        ax = self.axes["detail"]
        ax.clear()
        cat1 = comp.get("cat1") or {}
        summaries = cat1.get("summaries") or {}
        leg_len = float(cat1.get("length", 0.0) or 0.0)
        origin, along, left, sign = self._frame(comp, "cat1")
        dc = self._dir_color(comp)

        self._plot_local_obstacles(ax, origin, along, left, _grown(view), set(summaries))
        start_zone = float(cat1.get("start_zone_m", 0.0) or 0.0)
        if start_zone > 0.0:
            ax.axvspan(0.0, start_zone, color="#999999", alpha=0.2, linewidth=0, zorder=1)
            ax.text(start_zone / 2.0, 0.02, "start zone\n(not counted)", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color="dimgray", clip_on=True)
        if leg_len > 0.0:
            ax.axvspan(leg_len, leg_len + MAX_RANGE, color="#f2f2f2", linewidth=0, zorder=0)
            ax.axvline(leg_len, color="black", linewidth=1.0, linestyle="--")
            ax.text(leg_len, 0.02, " turning point\n Cat II ->", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=6, clip_on=True)
        self._plot_local_rays(ax, cat1.get("ray_data"), sign)
        ax.axhline(sign * comp["dir_info"]["mean"], color=dc, linestyle=":", linewidth=0.9)
        ax.axvline(0.0, color="black", linewidth=0.8)
        self._plot_pdf_inset(ax, sign * np.asarray(comp["offsets"], dtype=float), comp["pdf_vals"],
                             cat1.get("ray_data") or [])
        self._obstacle_legend(ax, summaries, lambda key, s: f"{_obs_tag(key)}: mass = {float(s['mass']):.3g}")
        total_mass = sum(float(s.get("mass", 0.0) or 0.0) for s in summaries.values())
        if not summaries:
            ax.text(0.58, 0.5, "No obstacle inside the lane of this leg:\nCat I adds nothing here",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8, color="dimgray",
                    bbox=dict(fc="white", alpha=0.9, ec="#cccccc"), zorder=9)
        ax.set_title(
            f"Cat I -- obstacle inside the lane: {self.comp_title(comp)}\n"
            f"P_I = intercepted mass = {total_mass:.3g}  (no distance decay; true scale)",
            fontsize=8,
        )
        ax.set_xlabel("Along-track distance from where the ships enter the leg (m)", fontsize=7)
        ax.set_ylabel("Lateral offset (m), + = left of heading", fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # -----------------------------------------------------------------
    # Cat II panels (bottom right: geometry over probability)
    # -----------------------------------------------------------------

    def _draw_cat2(self, comp: dict, view: tuple) -> None:
        """Bottom-right, upper: Cat II geometry past the turning point, true scale."""
        ax = self.axes.get("cat2")
        if ax is None:
            return
        ax.clear()
        summaries = comp.get("summaries") or {}
        tp, ed, left, sign = self._frame(comp, "cat2")

        self._plot_local_obstacles(ax, tp, ed, left, _grown(view), set(summaries))
        self._plot_local_rays(ax, comp.get("ray_data"), sign)
        for key, s in summaries.items():
            # Continued in the probability panel below (shared x-axis).
            ax.axvline(float(s["mean_dist"]), color=self.obs_color_map.get(key, "#888888"),
                       linestyle=":", linewidth=0.9, alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.text(0.0, 0.02, " turning point", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=6, clip_on=True)
        ax.axhline(sign * comp["dir_info"]["mean"], color=self._dir_color(comp), linestyle=":", linewidth=0.9)
        self._plot_pdf_inset(ax, sign * np.asarray(comp["offsets"], dtype=float), comp["pdf_vals"],
                             comp.get("ray_data") or [])
        self._obstacle_legend(
            ax, summaries,
            lambda key, s: f"{_obs_tag(key)}: mass = {float(s['mass']):.3g}, d = {float(s['mean_dist']) / 1000:.1f} km",
        )
        if not summaries:
            ax.text(0.58, 0.5, f"No obstacle within {MAX_RANGE / 1000:.0f} km past the turning point:\n"
                    "Cat II adds nothing here", transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="dimgray", bbox=dict(fc="white", alpha=0.9, ec="#cccccc"), zorder=9)
        ax.set_title(
            f"Cat II -- ships that miss the turn at the end of {self.comp_title(comp)}\n"
            "they continue straight on; each ray stops at the first obstacle (true scale)",
            fontsize=8,
        )
        ax.set_ylabel("Lateral offset (m),\n+ = left of heading", fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6, labelbottom=False)

    def _draw_waterfall(self, comp: dict, view: tuple | None = None) -> None:
        """Bottom-right, lower: the Cat II probability ``mass x exp(-d / (ai V))`` per obstacle.

        Shares the x-axis with the ``cat2`` geometry above it: for every hit
        obstacle the survival ``exp(-d_mean / (ai V))`` (open marker) is
        multiplied by its mass, giving ``P_II`` (filled marker).  Log scale,
        because ``P_II`` is often many decades below the curve.
        """
        ax = self.axes["waterfall"]
        ax.clear()
        d_info = comp["dir_info"]
        summaries = comp.get("summaries") or {}
        x0, x1 = (view[0], view[1]) if view is not None else (0.0, 10_000.0)
        recovery = float(d_info["ai"]) * float(d_info["speed_ms"])
        ax.set_xlabel("Along-track distance past the turning point (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(x0, x1)
        if recovery <= 0.0:
            ax.text(0.5, 0.5, "Cat II is off for this direction\n(Mean time between checks <= 0)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8, color="dimgray")
            ax.set_title("Cat II probability", fontsize=8)
            return

        d_grid = np.linspace(0.0, max(x1, 1.0), 400)
        survival = np.maximum(np.exp(-d_grid / recovery), P_FLOOR)
        ax.plot(d_grid, survival, color="black", linewidth=1.4, zorder=3)
        ax.set_yscale("log")

        def _p_of(s: dict) -> float:
            return max(float(s.get("mass", 0.0) or 0.0) * float(np.exp(-float(s["mean_dist"]) / recovery)), P_FLOOR)

        p_vals: list[float] = []
        for key, s in summaries.items():
            c = self.obs_color_map.get(key, "#888888")
            d_mean = float(s["mean_dist"])
            surv = max(float(np.exp(-d_mean / recovery)), P_FLOOR)
            p = _p_of(s)
            p_vals.append(p)
            ax.axvline(d_mean, color=c, linestyle=":", linewidth=0.9, alpha=0.8)
            ax.vlines(d_mean, p, surv, colors=c, linewidth=1.6, zorder=4)
            ax.plot(d_mean, surv, "o", markerfacecolor="white", markeredgecolor=c, markersize=5, zorder=5)
            ax.plot(d_mean, p, "o", color=c, markeredgecolor="white", markersize=6, zorder=5)
        self._annotate_column(
            ax, summaries,
            lambda key, s: (f"{_obs_tag(key)}: P = {float(s['mass']):.3g} x "
                            f"{float(np.exp(-float(s['mean_dist']) / recovery)):.2e} = {_p_of(s):.2e}"),
            y_of=_p_of,
        )
        # Fit the visible stretch: the curve over the window plus the markers.
        s_hi = max(float(np.exp(-max(x0, 0.0) / recovery)), P_FLOOR)
        lows = [p for p in p_vals if p > P_FLOOR] or [max(float(np.exp(-x1 / recovery)), P_FLOOR)]
        y_lo = max(P_FLOOR, min(lows) / 30.0)
        y_hi = min(2.0, 3.0 * max([s_hi] + p_vals))
        ax.set_ylim(y_lo, max(y_hi, 10.0 * y_lo))
        handles = [Line2D([0], [0], color="black", lw=1.4, label="survival exp(-d / (ai V))")]
        if summaries:
            handles += [
                Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", ls="none",
                       label="survival at the obstacle"),
                Line2D([0], [0], marker="o", color="black", ls="none", label="P_II = mass x survival"),
            ]
        ax.legend(handles=handles, fontsize=6, loc="lower left")
        speed_kn = float(d_info.get("speed_kn") or float(d_info["speed_ms"]) * 3600.0 / 1852.0)
        ax.set_title(
            f"Cat II probability  P_II = mass x exp(-d / (ai V)),   "
            f"ai V = {float(d_info['ai']):.0f} s x {speed_kn:.1f} kn = {recovery:.0f} m",
            fontsize=8,
        )
        ax.set_ylabel("Probability (log scale)", fontsize=7)
        ax.grid(True, which="major", alpha=0.25)

    # -----------------------------------------------------------------
    # Selection (driven by the dialog's table)
    # -----------------------------------------------------------------

    def _select_computation(self, idx: int) -> None:
        """Select a computation and redraw the overview highlight and the Cat I / Cat II panels.

        Windows are sized twice: provisionally for the obstacle pre-filter,
        then again after the layout engine has placed the new titles, so
        the true-scale windows match the final panel boxes.
        """
        if idx < 0 or idx >= len(self.computations):
            return
        self._selected_comp_idx = idx
        comp = self.computations[idx]
        ax1, ax2 = self.axes.get("detail"), self.axes.get("cat2")
        provisional = {"cat1": self._local_view(comp, "cat1", ax1),
                       "cat2": self._local_view(comp, "cat2", ax2)}
        self._draw_detail(comp, provisional["cat1"])
        self._draw_cat2(comp, provisional["cat2"])
        self._draw_waterfall(comp, provisional["cat2"])
        self._refresh_layout()
        self.views = {"cat1": self._local_view(comp, "cat1", ax1),
                      "cat2": self._local_view(comp, "cat2", ax2) if ax2 is not None else None}
        self._set_local_view(ax1, self.views["cat1"], "ships enter")
        if self.views["cat2"] is not None:
            self._set_local_view(ax2, self.views["cat2"], "turning point")
            self._draw_waterfall(comp, self.views["cat2"])
        self._highlight_selected(comp)
        # A new selection starts a new history for the toolbar's Home / Back buttons.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and hasattr(toolbar, "update"):
            try:
                toolbar.update()
            except Exception:  # nosec B110 B112
                pass
        self.fig.canvas.draw()
        # Reflect the change on the sidebar so the user can see which
        # row corresponds to the panels currently shown.  Block
        # signals to avoid recursion through ``itemSelectionChanged``.
        sidebar = getattr(self, 'sidebar', None)
        idx_map = getattr(self, 'sidebar_index_map', None)
        if sidebar is not None and idx_map is not None:
            try:
                row = idx_map.index(idx)
                blocked = sidebar.blockSignals(True)
                try:
                    sidebar.selectRow(row)
                finally:
                    sidebar.blockSignals(blocked)
            except ValueError:
                pass

    # -----------------------------------------------------------------
    # Figure, sidebar and dialog
    # -----------------------------------------------------------------

    @staticmethod
    def build_figure() -> tuple[Figure, dict[str, Axes]]:
        """Overview + info on top; Cat I | Cat II (geometry over probability) below."""
        fig = Figure(figsize=(14, 12), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
        top = gs[0].subgridspec(1, 2, width_ratios=[3.0, 1.0])
        bottom = gs[1].subgridspec(1, 2)
        right = bottom[0, 1].subgridspec(2, 1, height_ratios=[1.25, 1.0])
        ax_cat2 = fig.add_subplot(right[0])
        axes = {
            "overview": fig.add_subplot(top[0, 0]),
            "info": fig.add_subplot(top[0, 1]),
            "detail": fig.add_subplot(bottom[0, 0]),
            "cat2": ax_cat2,
            "waterfall": fig.add_subplot(right[1], sharex=ax_cat2),
        }
        for a in axes.values():
            a.tick_params(labelsize=6)
        return fig, axes

    @staticmethod
    def _build_sidebar(rows: list[dict], has_report: bool, mode: str):
        """``QTableWidget`` with Leg | Direction | Frequency | % of total, one row per computation."""
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import (
            QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
        )
        value_header = 'Frequency (/yr)' if has_report else 'P(hit) per ship'
        sidebar = QTableWidget(len(rows), 4)
        sidebar.setHorizontalHeaderLabels(['Leg', 'Direction', value_header, '% of total'])
        sidebar.verticalHeader().setVisible(False)
        sidebar.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sidebar.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        sidebar.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for r_i, row in enumerate(rows):
            value = format_frequency(row['frequency']) if has_report else f"{row['p_geom']:.3e}"
            cells = (row['leg'], row['direction'], value, format_share(row['share']))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 0:
                    item.setToolTip(f"{row['leg']} {row['direction']} (segment id {row['key']})")
                sidebar.setItem(r_i, col, item)
        if not has_report:
            tip = (f"This run has no powered-{mode} report (it was saved before OMRAT stored one). "
                   "Re-run the model to see each leg's annual frequency and share of the total; "
                   "until then the legs are ranked by their per-ship hit probability.")
            for col in (2, 3):
                header_item = sidebar.horizontalHeaderItem(col)
                if header_item is not None:
                    header_item.setToolTip(tip)
        header = sidebar.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(520)
        return sidebar

    @classmethod
    def show_in_dialog(
        cls,
        dialog: ShowGeomRes,
        data: dict[str, Any],
        mode: str = "grounding",
        max_draft: float = DEFAULT_MAX_DRAFT,
        report: dict[str, Any] | None = None,
    ) -> "PoweredOverlapVisualizer":
        """Build the visualizer and embed it in *dialog*.

        Parameters
        ----------
        dialog : ShowGeomRes
            The Qt dialog that contains ``result_layout``.
        data : dict
            Full project data (as returned by ``GatherData.get_all_for_save``).
        mode : str
            ``"allision"`` or ``"grounding"``.
        max_draft : float
            Maximum ship draft. Depths deeper than this are ignored for
            grounding.
        report : dict, optional
            The run's powered report for ``mode`` (model report or the
            ``<stem>.powered.json`` block) -- source of the Frequency and
            % of total columns.  Without one, allision is rebuilt from the
            ray casts (``compute/powered_breakdown.py``); grounding shows
            the per-ship hit probability and ``--``.

        Returns
        -------
        PoweredOverlapVisualizer
        """
        # Remove any existing canvas
        layout = dialog.result_layout
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                layout.removeWidget(widget)
                widget.deleteLater()

        # Build projector from first segment
        segments = data.get("segment_data", {})
        if not segments:
            return None  # type: ignore[return-value]
        first_seg = segments[list(segments.keys())[0]]
        try:
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
        except Exception:  # nosec B110 B112
            return None  # type: ignore[return-value]
        proj = SimpleProjector(lon0, lat0)

        # Extract data
        legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms = \
            _build_legs_and_obstacles(data, proj, mode, max_draft)

        if not legs:
            return None  # type: ignore[return-value]

        # Same ray casts as the model, Cat I start zone included.
        computations = _run_all_computations(
            legs, all_obstacles, cat1_start_zone_frac=start_zone_fraction(data.get("pc")),
        )
        if report is None and mode == "allision":
            try:
                report = allision_report_from_computations(data, computations, object_geoms)
            except Exception:  # nosec B110 B112
                report = None
        rows = build_sidebar_rows(computations, report, segments, mode)

        fig, axes = cls.build_figure()
        canvas = FigureCanvas(fig)

        # Toolbar + canvas on the left, the table on the right, in a
        # splitter the user can drag.  The table is the only way to pick
        # a (leg, direction): both directions of a leg sit on the same
        # line of the map, so clicking there could not tell them apart.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        plot_box = QWidget()
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(NavigationToolbar2QT(canvas, plot_box))
        plot_layout.addWidget(canvas)
        sidebar = cls._build_sidebar(rows, report is not None, mode)
        table_box = QWidget()
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Pick a leg and direction to show it in the figure:")
        hint.setWordWrap(True)
        table_layout.addWidget(hint)
        table_layout.addWidget(sidebar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(plot_box)
        splitter.addWidget(table_box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        viz = cls(
            fig, axes, legs, all_obstacles,
            depth_geoms, depth_geoms_deep, object_geoms,
            computations, mode,
            segment_data=segments, breakdown_rows=rows,
        )
        viz.canvas = canvas
        viz.sidebar = sidebar
        viz.sidebar_index_map = [r["comp_idx"] for r in rows]
        viz.run_visualization()

        # Wire row selection to ``_select_computation``.  Use
        # ``itemSelectionChanged`` which fires both for mouse clicks
        # and for keyboard navigation.
        def _on_sidebar_selection_changed():
            selected = {idx.row() for idx in sidebar.selectedIndexes()}
            if not selected:
                return
            row = next(iter(selected))
            if 0 <= row < len(viz.sidebar_index_map):
                viz._select_computation(viz.sidebar_index_map[row])

        sidebar.itemSelectionChanged.connect(_on_sidebar_selection_changed)

        mode_label = "Powered Allision" if mode == "allision" else "Powered Grounding"
        dialog.setWindowTitle(f"OMRAT - {mode_label} Cat I + Cat II Visualization")
        canvas.draw()
        return viz
