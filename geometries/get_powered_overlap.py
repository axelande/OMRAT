"""
Powered grounding/allision visualization -- IWRAP Category II with shadow effects.

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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.stats import norm
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shapely_transform
import shapely.wkt as sw

from ui.show_geom_res import ShowGeomRes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIGMA = 3          # standard deviations for the distribution band
MAX_RANGE = 50_000   # max ray length in meters (50 km)
DEFAULT_MAX_DRAFT = 15.0  # default max ship draft (m)
N_RAYS = 500         # number of rays across the lateral distribution


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
    except Exception:
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
    """
    if geom is None or getattr(geom, 'is_empty', True):
        return None

    rings: list[np.ndarray] = []

    def _collect(g):
        gt = g.geom_type
        if gt == 'Polygon':
            if g.exterior is not None:
                rings.append(np.asarray(g.exterior.coords, dtype=float))
            for interior in g.interiors:
                rings.append(np.asarray(interior.coords, dtype=float))
        elif gt in ('LineString', 'LinearRing'):
            rings.append(np.asarray(g.coords, dtype=float))
        elif gt in ('MultiPolygon', 'MultiLineString', 'GeometryCollection'):
            for sub in g.geoms:
                _collect(sub)
        # Points / MultiPoints: measure-zero; skip.

    _collect(geom)
    if not rings:
        return None

    edges_list: list[np.ndarray] = []
    for ring in rings:
        if ring.shape[0] < 2:
            continue
        # Transform into local frame: along = (p - origin) . along_dir,
        # lateral = (p - origin) . perp_dir.
        diff = ring - turn_pt
        along = diff @ along_dir
        lateral = diff @ perp_dir
        local = np.stack([along, lateral], axis=1)
        edges_list.append(np.stack([local[:-1], local[1:]], axis=1))

    if not edges_list:
        return None
    return np.concatenate(edges_list, axis=0)


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

    Casts ``N_RAYS`` parallel rays across the lateral distribution.  Per ray,
    keeps the FIRST obstacle hit -- this is what creates shadows.

    Implementation
    --------------
    Every ray has the same ``ext_dir``, only its lateral offset differs.  In
    the local frame ``(along, lateral)`` anchored at ``turn_pt``, each ray is
    the line ``y = offset`` moving in +x.  For each polygon edge
    ``(v0, v1)`` in that frame, a ray at lateral ``y`` crosses the edge iff
    ``y`` lies strictly between ``y0`` and ``y1``, and the along-crossing is
    ``x0 + (y - y0) / (y1 - y0) * (x1 - x0)``.  Taking the minimum positive
    crossing per obstacle gives the ray's first-hit distance to that
    obstacle; argmin across obstacles gives the first-hit obstacle.  Both
    reductions are a single broadcasted numpy op.

    Returns
    -------
    summaries : dict
        ``{(kind, obs_id): {mass, mean_dist, p_integral, p_approx, ...}}``
    ray_data : list
        ``[(offset, mass_i, hit_key, hit_dist), ...]``
    offsets : ndarray
    pdf_vals : ndarray
    """
    offsets = np.linspace(mean_offset - 4 * sigma, mean_offset + 4 * sigma, N_RAYS)
    dx = offsets[1] - offsets[0]
    pdf_vals = norm.pdf(offsets, mean_offset, sigma)
    masses = pdf_vals * dx
    recovery = ai * speed_ms
    n_rays = len(offsets)

    # Empty-obstacle shortcut: every ray misses.
    if not obstacles:
        return (
            {},
            [(float(offsets[i]), float(masses[i]), None, None) for i in range(n_rays)],
            offsets,
            pdf_vals,
        )

    # Per-obstacle first-hit distance for every ray.
    hit_matrix = np.full((n_rays, len(obstacles)), np.inf, dtype=float)
    ray_ys = offsets[:, None]  # (n_rays, 1)

    for obs_idx, (obs, _kind) in enumerate(obstacles):
        edges = _extract_edges_local(obs["geom"], turn_pt, ext_dir, perp)
        if edges is None or edges.shape[0] == 0:
            continue

        # edges: (M, 2, 2)  where axis-1 is endpoint index, axis-2 is (along, lateral).
        x0 = edges[:, 0, 0]
        y0 = edges[:, 0, 1]
        x1 = edges[:, 1, 0]
        y1 = edges[:, 1, 1]

        y_min = np.minimum(y0, y1)[None, :]
        y_max = np.maximum(y0, y1)[None, :]
        dy = (y1 - y0)[None, :]

        crosses = (ray_ys >= y_min) & (ray_ys < y_max) & (dy != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (ray_ys - y0[None, :]) / dy
            along = x0[None, :] + t * (x1 - x0)[None, :]

        valid = crosses & (along > 0) & (along < MAX_RANGE)
        along = np.where(valid, along, np.inf)
        hit_matrix[:, obs_idx] = np.min(along, axis=1)

    best_obs_idx = np.argmin(hit_matrix, axis=1)
    best_dists = hit_matrix[np.arange(n_rays), best_obs_idx]
    hit_mask = np.isfinite(best_dists)

    # Accumulate per-obstacle summaries.  The outer loop is only N_RAYS (500)
    # so the remaining pure-Python cost is negligible relative to the numpy
    # reduction above.
    ray_data: list[tuple[float, float, tuple[str, Any] | None, float | None]] = []
    obs_accum: dict[tuple[str, Any], dict] = defaultdict(lambda: {
        "mass": 0.0, "weighted_dist": 0.0, "p_integral": 0.0,
        "n_rays": 0, "ray_offsets": [], "ray_dists": [],
        "obs": None, "kind": None,
    })

    for i in range(n_rays):
        off = float(offsets[i])
        m_i = float(masses[i])
        if not hit_mask[i]:
            ray_data.append((off, m_i, None, None))
            continue
        oi = int(best_obs_idx[i])
        obs, kind = obstacles[oi]
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
        oa["obs"] = obs
        oa["kind"] = kind
        ray_data.append((off, m_i, best_key, best_d))

    summaries: dict[tuple[str, Any], dict] = {}
    for key, oa in obs_accum.items():
        mean_dist = oa["weighted_dist"] / oa["mass"] if oa["mass"] > 0 else 0
        p_approx = oa["mass"] * _powered_na(mean_dist, ai, speed_ms)
        summaries[key] = {
            "mass": oa["mass"],
            "mean_dist": mean_dist,
            "p_integral": oa["p_integral"],
            "p_approx": p_approx,
            "n_rays": oa["n_rays"],
            "ray_offsets": oa["ray_offsets"],
            "ray_dists": oa["ray_dists"],
            "obs": oa["obs"],
            "kind": oa["kind"],
        }

    return summaries, ray_data, offsets, pdf_vals


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _total_p_for_comp(comp: dict) -> float:
    """Sum the per-obstacle ``p_approx`` values for a computation.

    Used to rank the sidebar list so the highest-probability
    (Leg, Direction) pairs float to the top.
    """
    total = 0.0
    for s in (comp.get("summaries") or {}).values():
        try:
            total += float(s.get("p_approx", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def find_closest_computation_index(
    click_xy: tuple[float, float],
    computations: list[dict],
    threshold: float,
) -> int | None:
    """Return the index of the computation whose turn-point is closest to
    ``click_xy``, or None if none are within ``threshold`` distance.

    Pure-data extract of the click-to-computation routing inside
    ``PoweredOverlapVisualizer._on_overview_click``, so the math can be
    tested without a Qt event.
    """
    best_idx: int | None = None
    best_dist = float('inf')
    click = np.array(click_xy)
    for ci, comp in enumerate(computations):
        tp = comp["turn_pt"]
        d = float(np.linalg.norm(tp - click))
        if d < best_dist:
            best_dist = d
            best_idx = ci
    if best_idx is not None and best_dist < threshold:
        return best_idx
    return None


def _build_legs_and_obstacles(
    data: dict[str, Any],
    proj: SimpleProjector,
    mode: str,
    max_draft: float,
) -> tuple[dict[str, dict], list[tuple[dict, str]], list[dict], list[dict], list[dict]]:
    """Extract projected legs, obstacle list, and raw geometry lists from *data*.

    Parameters
    ----------
    mode : str
        ``"allision"`` -> only objects;  ``"grounding"`` -> only depths.
        ``"both"`` -> all obstacles.

    Returns
    -------
    legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms
    """
    segments = data.get("segment_data", {})
    traffic = data.get("traffic_data", {})

    # Project obstacles
    depth_geoms_all = []
    for dep in data.get("depths", []):
        try:
            did, depth_val, wkt_str = dep
            geom = sw.loads(wkt_str)
            depth_geoms_all.append({
                "id": did,
                "depth": float(depth_val),
                "geom": _project_wkt_geom(geom, proj),
            })
        except Exception:
            continue

    depth_geoms = [d for d in depth_geoms_all if d["depth"] <= max_draft]
    depth_geoms_deep = [d for d in depth_geoms_all if d["depth"] > max_draft]

    object_geoms: list[dict] = []
    for obj in data.get("objects", []):
        try:
            oid, height, wkt_str = obj
            geom = sw.loads(wkt_str)
            object_geoms.append({
                "id": oid,
                "height": height,
                "geom": _project_wkt_geom(geom, proj),
            })
        except Exception:
            continue

    # Build obstacle list based on mode
    all_obstacles: list[tuple[dict, str]] = []
    if mode in ("grounding", "both"):
        all_obstacles.extend([(dg, "depth") for dg in depth_geoms])
    if mode in ("allision", "both"):
        all_obstacles.extend([(og, "object") for og in object_geoms])

    # Build legs
    legs: dict[str, dict] = {}
    for seg_id, seg in segments.items():
        try:
            lon_s, lat_s = _parse_point(seg["Start_Point"])
            lon_e, lat_e = _parse_point(seg["End_Point"])
        except Exception:
            continue
        xs, ys = proj.transform(lon_s, lat_s)
        xe, ye = proj.transform(lon_e, lat_e)
        start = np.array([xs, ys])
        end = np.array([xe, ye])

        dirs = seg.get("Dirs", ["Dir 1", "Dir 2"])
        ai_values = [float(seg.get("ai1", 180)), float(seg.get("ai2", 180))]

        seg_traffic = traffic.get(seg_id, {})
        dir_info: list[dict] = []
        for i, d_name in enumerate(dirs):
            spd_kn = (_weighted_avg_speed_knots(seg_traffic[d_name])
                      if d_name in seg_traffic else 0)
            mean_key = f"mean{i + 1}_1"
            std_key = f"std{i + 1}_1"
            dir_info.append({
                "name": d_name,
                "speed_kn": spd_kn,
                "speed_ms": spd_kn * 1852.0 / 3600.0,
                "ai": ai_values[min(i, 1)],
                "mean": float(seg.get(mean_key, 0)),
                "std": float(seg.get(std_key, 100)),
            })

        legs[seg_id] = {
            "start": start,
            "end": end,
            "name": seg.get("Leg_name", ""),
            "start_wkt": seg["Start_Point"],
            "end_wkt": seg["End_Point"],
            "dirs": dir_info,
        }

    return legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms


def _run_all_computations(
    legs: dict[str, dict],
    all_obstacles: list[tuple[dict, str]],
) -> list[dict]:
    """Run the shadow-aware Cat II computation for every leg/direction."""
    computations: list[dict] = []
    for seg_id, leg in legs.items():
        start, end = leg["start"], leg["end"]
        u, n, L = _leg_vectors(start, end)

        for di, d in enumerate(leg["dirs"]):
            if d["speed_ms"] <= 0:
                continue

            if di == 0:
                turn_pt = end.copy()
                ext_dir = u.copy()
            else:
                turn_pt = start.copy()
                ext_dir = (-u).copy()

            summaries, ray_data, offsets, pdf_vals = _compute_cat2_with_shadows(
                turn_pt, ext_dir, n, d["mean"], d["std"],
                d["ai"], d["speed_ms"], all_obstacles,
            )

            if summaries:
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
                })
    return computations


# ---------------------------------------------------------------------------
# PoweredOverlapVisualizer
# ---------------------------------------------------------------------------

class PoweredOverlapVisualizer:
    """Interactive visualizer for powered (Cat II) allision/grounding geometry.

    Embeds three matplotlib panels inside a ``ShowGeomRes`` dialog:

    1. **Overview map** -- route legs with distribution bands, obstacles, ray fans.
    2. **Detailed view** -- local-coordinate plot for a selected turning point
       showing lateral distribution coloured by obstacle, shadow regions.
    3. **Waterfall view** -- how the distribution loses mass at each obstacle
       for the selected computation.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure.
    axes : dict
        Mapping of panel name to ``Axes`` (``overview``, ``detail``, ``waterfall``).
    legs, all_obstacles, ... : various
        Pre-computed data produced by ``_build_legs_and_obstacles``
        and ``_run_all_computations``.
    mode : str
        ``"allision"`` or ``"grounding"`` -- controls title text.
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

        # Colour maps
        self.dir_colors = ["#d62728", "#2ca02c"]
        self.leg_colors: dict[str, Any] = dict(
            zip(legs.keys(), plt.cm.Set1(np.linspace(0, 0.8, max(len(legs), 1))))
        )
        all_obs_keys: set[tuple] = set()
        for comp in computations:
            all_obs_keys.update(comp["summaries"].keys())
        cmap = plt.cm.tab10
        self.obs_color_map: dict[tuple, Any] = {
            key: cmap(i % 10)
            for i, key in enumerate(sorted(all_obs_keys))
        }

        # State: which computation is selected for detail/waterfall
        self._selected_comp_idx: int | None = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run_visualization(self) -> None:
        """Draw the overview, pick a default computation, and draw detail views."""
        self._draw_overview()
        if self.computations:
            self._select_computation(0)
        self._connect_events()

    # -----------------------------------------------------------------
    # Overview panel
    # -----------------------------------------------------------------

    def _draw_overview(self) -> None:
        ax = self.axes["overview"]
        mode_label = "Allision" if self.mode == "allision" else "Grounding"
        ax.set_title(
            f"Powered {mode_label} -- Cat II Overview\n"
            "Rays from turning points (shadow-aware)",
            fontsize=11,
        )

        # Depth areas (faint deep, highlighted shallow)
        for dg in self.depth_geoms_deep:
            _plot_geom(ax, dg["geom"], color="#e0e0e0", alpha=0.3,
                       edgecolor="#aaaaaa", linewidth=0.3)
        for dg in self.depth_geoms:
            _plot_geom(ax, dg["geom"], color="lightblue", alpha=0.5,
                       edgecolor="steelblue", linewidth=0.6)
        for og in self.object_geoms:
            _plot_geom(ax, og["geom"], color="salmon", alpha=0.5,
                       edgecolor="darkred", linewidth=0.6)

        # Legs and bands
        for seg_id, leg in self.legs.items():
            start, end = leg["start"], leg["end"]
            u, n, L = _leg_vectors(start, end)
            c = self.leg_colors[seg_id]
            ax.plot([start[0], end[0]], [start[1], end[1]], "-",
                    color=c, linewidth=2.5, zorder=4, label=f"Leg {seg_id}")
            ax.plot(*start, "o", color=c, markersize=5, zorder=5)
            ax.plot(*end, "s", color=c, markersize=5, zorder=5)

            for di, d in enumerate(leg["dirs"]):
                dc = self.dir_colors[di]
                mean, sigma = d["mean"], d["std"]
                band = _make_band_polygon(start, end, mean, sigma)
                bx, by = band.exterior.xy
                ax.fill(bx, by, color=dc, alpha=0.12, edgecolor=dc, linewidth=0.8)
                cl_start = start + mean * n
                cl_end = end + mean * n
                ax.plot([cl_start[0], cl_end[0]], [cl_start[1], cl_end[1]],
                        "--", color=dc, linewidth=1.0, alpha=0.7)

        # Ray fans for computations with hits
        for comp in self.computations:
            tp = comp["turn_pt"]
            ed = comp["ext_dir"]
            n = comp["perp"]
            dc = self.dir_colors[comp["dir_idx"]]
            for off, m_i, hit_key, hit_dist in comp["ray_data"][::20]:
                origin = tp + off * n
                if hit_key is not None:
                    end_pt = origin + hit_dist * ed
                    obs_c = "steelblue" if hit_key[0] == "depth" else "darkred"
                    ax.plot([origin[0], end_pt[0]], [origin[1], end_pt[1]],
                            "-", color=obs_c, alpha=0.15, linewidth=0.5)
                else:
                    end_pt = origin + MAX_RANGE * 0.1 * ed
                    ax.plot([origin[0], end_pt[0]], [origin[1], end_pt[1]],
                            ":", color=dc, alpha=0.08, linewidth=0.3)

        # Mark computations as clickable circles -- we keep the
        # selected computation's marker around for ``_highlight_selected``
        # to override its style when the user clicks elsewhere.
        self._comp_markers: list[Any] = []
        for ci, comp in enumerate(self.computations):
            tp = comp["turn_pt"]
            marker_style = "D" if ci == 0 else "o"
            ax.plot(tp[0], tp[1], marker_style, color="black",
                    markersize=8, zorder=10, picker=True, pickradius=10)
            ax.annotate(
                f"L{comp['seg_id']} {comp['dir_info']['name'][:5]}",
                xy=(tp[0], tp[1]), fontsize=6,
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(fc="white", alpha=0.7, pad=1, ec="none"),
            )

        # Legend
        legend_items = []
        if self.depth_geoms:
            legend_items.append(Patch(facecolor="lightblue", edgecolor="steelblue",
                                      label="Depth areas (grounding)"))
        if self.object_geoms:
            legend_items.append(Patch(facecolor="salmon", edgecolor="darkred",
                                      label="Objects (allision)"))
        if self.depth_geoms_deep:
            legend_items.append(Patch(facecolor="#e0e0e0", edgecolor="#aaa",
                                      label="Deep (ignored)"))
        for di, dc in enumerate(self.dir_colors[:2]):
            legend_items.append(Patch(facecolor=dc, alpha=0.3,
                                      label=f"Dir {di + 1} band ({N_SIGMA}s)"))
        for seg_id in self.legs:
            legend_items.append(plt.Line2D([0], [0], color=self.leg_colors[seg_id],
                                            lw=2, label=f"Leg {seg_id}"))
        if legend_items:
            ax.legend(handles=legend_items, fontsize=6, loc="upper left", ncol=2)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------
    # Detail panel
    # -----------------------------------------------------------------

    def _draw_detail(self, comp: dict) -> None:
        ax = self.axes["detail"]
        ax.clear()

        d_info = comp["dir_info"]
        tp = comp["turn_pt"]
        ed = comp["ext_dir"]
        perp = comp["perp"]
        offsets = comp["offsets"]
        pdf_vals = comp["pdf_vals"]
        ray_data = comp["ray_data"]
        summaries = comp["summaries"]
        recovery = d_info["ai"] * d_info["speed_ms"]

        ax.set_title(
            f"LEG {comp['seg_id']} {d_info['name']}   "
            f"mean={d_info['mean']:+.0f}m  std={d_info['std']:.0f}m  "
            f"recovery={recovery:.0f}m\n"
            "Local frame: leg rotated so along-track = +x, perpendicular = y",
            fontsize=8, fontweight="bold",
        )

        # Project obstacles to local coordinates
        for obs, kind in self.all_obstacles:
            obs_local = _project_to_local(obs["geom"], tp, ed, perp)
            color = "lightblue" if kind == "depth" else "salmon"
            edge = "steelblue" if kind == "depth" else "darkred"
            _plot_geom(ax, obs_local, color=color, alpha=0.35,
                       edgecolor=edge, linewidth=0.5)

        # Distribution curve at x=0, coloured by obstacle hit
        pdf_scale = MAX_RANGE * 0.04 / max(pdf_vals) if max(pdf_vals) > 0 else 1
        dist_x = -pdf_vals * pdf_scale
        dx = offsets[1] - offsets[0]

        for off, m_i, hit_key, hit_dist in ray_data:
            c = self.obs_color_map.get(hit_key, "#dddddd") if hit_key else "#dddddd"
            i_off = int(np.argmin(np.abs(offsets - off)))
            ax.barh(off, dist_x[i_off], height=dx,
                    color=c, alpha=0.6, edgecolor="none")

        ax.plot(dist_x, offsets, "k-", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.8)

        # Sample rays
        step = max(1, N_RAYS // 80)
        for off, m_i, hit_key, hit_dist in ray_data[::step]:
            if hit_key is not None:
                c = self.obs_color_map[hit_key]
                ax.plot([0, hit_dist], [off, off],
                        color=c, alpha=0.25, linewidth=0.5)

        # Shadow regions
        for key, s in sorted(summaries.items(), key=lambda x: x[1]["mean_dist"]):
            ray_offs = np.array(s["ray_offsets"])
            ray_ds = np.array(s["ray_dists"])
            if len(ray_offs) < 2:
                continue
            c = self.obs_color_map[key]
            lat_min, lat_max = ray_offs.min(), ray_offs.max()
            d_max_obs = ray_ds.max()
            shadow_x = [d_max_obs, MAX_RANGE, MAX_RANGE, d_max_obs]
            shadow_y = [lat_min, lat_min, lat_max, lat_max]
            ax.fill(shadow_x, shadow_y, color=c, alpha=0.06,
                    hatch="//", edgecolor=c, linewidth=0)

        # Annotate obstacles -- evenly distribute the labels along the
        # full y-axis range so two boxes never overlap regardless of
        # font / DPI.  Sorted by along-track distance so the visual
        # order matches the curve to the right.
        sorted_obs_for_labels = sorted(
            summaries.items(), key=lambda x: x[1]["mean_dist"],
        )
        if sorted_obs_for_labels:
            max_obs_dist_for_labels = max(
                s["mean_dist"] for _, s in sorted_obs_for_labels
            )
            label_x = max_obs_dist_for_labels + MAX_RANGE * 0.04
            n = len(sorted_obs_for_labels)
            y_top = offsets[-1] - (offsets[-1] - offsets[0]) * 0.05
            y_bot = offsets[0] + (offsets[-1] - offsets[0]) * 0.05
            if n == 1:
                ys = [(y_top + y_bot) / 2]
            else:
                # Evenly spaced positions from top to bottom; with 2
                # labels they land at 5 % and 95 % of the y-range.
                ys = [
                    y_top + (y_bot - y_top) * (i / (n - 1))
                    for i in range(n)
                ]
            for li, (key, s) in enumerate(sorted_obs_for_labels):
                kind, obs_id = key
                tag = f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"
                c = self.obs_color_map[key]
                ray_offs = s["ray_offsets"]
                mid_lat = (min(ray_offs) + max(ray_offs)) / 2
                ax.annotate(
                    f"{tag}\nmass={s['mass']:.4f}\n"
                    f"d={s['mean_dist'] / 1000:.1f} km\n"
                    f"P={s['p_approx']:.2e}",
                    xy=(s["mean_dist"], mid_lat),
                    xytext=(label_x, ys[li]),
                    fontsize=6, color=c, fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(
                        arrowstyle="-", color=c, alpha=0.6,
                        connectionstyle="arc3,rad=0.0",
                    ),
                    bbox=dict(fc="white", alpha=0.95, ec=c, pad=2,
                              boxstyle="round,pad=0.3"),
                )

        ax.axhline(d_info["mean"], color="red", linewidth=0.6,
                    linestyle="--", alpha=0.5, label="dist. mean")
        # Indicate the leg direction in the local frame: a thick black
        # arrow from x=0 toward +x at lateral offset 0 makes it
        # immediately obvious that "along-track" goes left -> right
        # regardless of the leg's actual orientation in the top map.
        try:
            arrow_len = MAX_RANGE * 0.04
            ax.annotate(
                "", xy=(arrow_len, 0), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black", lw=1.5,
                    shrinkA=0, shrinkB=0,
                ),
                annotation_clip=False,
            )
            ax.text(
                arrow_len * 1.05, 0, " leg direction",
                fontsize=6, color="black", va="center",
            )
        except Exception:
            pass
        ax.set_xlabel("Along-track distance from turning point (m)", fontsize=7)
        ax.set_ylabel("Lateral offset from centreline (m)", fontsize=7)
        max_obs_dist = max((s["mean_dist"] for s in summaries.values()), default=10000)
        # Extra room on the right so the staggered labels fit.
        ax.set_xlim(-MAX_RANGE * 0.06, max(30000, max_obs_dist * 1.4))
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # -----------------------------------------------------------------
    # Waterfall panel
    # -----------------------------------------------------------------

    def _draw_waterfall(self, comp: dict) -> None:
        """Probability decay vs distance.

        x-axis: along-track distance from the turning point.
        y-axis: probability that the ship is still off-course (has not
                recovered yet) -- the IWRAP Cat II ``exp(-d / (ai*V))``
                survival curve.

        For each obstacle a marker is plotted at
        ``(d_mean, mass × exp(-d_mean / recovery))`` so the user can see
        the per-obstacle probability contribution dropping with distance.
        """
        ax = self.axes["waterfall"]
        ax.clear()

        d_info = comp["dir_info"]
        summaries = comp["summaries"]
        recovery = max(d_info["ai"] * d_info["speed_ms"], 1.0)

        ax.set_title(
            f"Probability vs distance: LEG {comp['seg_id']} {d_info['name']}\n"
            f"Survival curve  exp(-d / {recovery:.0f} m)",
            fontsize=9,
        )

        sorted_obs = sorted(summaries.items(), key=lambda x: x[1]["mean_dist"])
        if sorted_obs:
            max_d = max(s["mean_dist"] for _, s in sorted_obs) * 1.4
        else:
            max_d = max(recovery * 5, 10000.0)
        max_d = max(max_d, recovery * 2)

        # Survival curve: probability the ship hasn't recovered by d.
        d_grid = np.linspace(0, max_d, 400)
        survival = np.exp(-d_grid / recovery)
        ax.plot(d_grid, survival, color="black", linewidth=1.5,
                label="Survival exp(-d / recovery)")
        ax.fill_between(d_grid, 0, survival, color="black", alpha=0.05)

        # Per-obstacle probability contribution.
        max_p = 0.0
        for key, s in sorted_obs:
            kind, obs_id = key
            tag = f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"
            c = self.obs_color_map[key]
            d_mean = s["mean_dist"]
            mass = float(s.get("mass", 0.0) or 0.0)
            p_contrib = mass * float(np.exp(-d_mean / recovery))
            max_p = max(max_p, p_contrib)
            # Vertical drop line + marker.
            ax.vlines(d_mean, 0, p_contrib, colors=c, linewidth=1.2,
                      alpha=0.6)
            ax.plot(d_mean, p_contrib, marker="o", color=c, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(
                f"{tag}\nmass={mass:.3f}\nd={d_mean/1000:.1f}km\n"
                f"P={s.get('p_approx', p_contrib):.2e}",
                xy=(d_mean, p_contrib),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6, color=c, fontweight="bold",
                bbox=dict(fc="white", alpha=0.95, ec=c, pad=2,
                          boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="-", color=c, alpha=0.5),
            )

        ax.set_xlabel("Along-track distance from turning point (m)", fontsize=7)
        ax.set_ylabel("Probability (mass × survival)", fontsize=7)
        ax.set_xlim(0, max_d)
        # y-limit accommodates the survival curve (max=1) and the
        # tallest obstacle marker.
        ax.set_ylim(0, max(1.05, max_p * 1.2))
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")

    # -----------------------------------------------------------------
    # Interaction
    # -----------------------------------------------------------------

    def _select_computation(self, idx: int) -> None:
        """Select a computation and redraw the detail and waterfall panels."""
        if idx < 0 or idx >= len(self.computations):
            return
        self._selected_comp_idx = idx
        comp = self.computations[idx]
        self._draw_detail(comp)
        self._draw_waterfall(comp)
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

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect(
            'button_press_event', self._on_overview_click,
        )

    def _on_overview_click(self, event) -> None:
        """Handle click on the overview panel to select a turning point.

        Two-stage routing:

        1. If the click lands close to a computation's turn-point
           marker (8 % of the axis range), select that computation.
        2. Otherwise project the click onto every leg line; the
           computation is the one whose ``turn_pt`` is closest to the
           projection on the nearest leg.  This way clicking *anywhere*
           on a leg switches the bottom panels to a computation on
           that leg.

        Note: clicks may not always reach this handler depending on
        the QGIS / Qt build.  The sidebar QTableWidget is the
        guaranteed-working interaction; this stays as a convenience.
        """
        if event.inaxes != self.axes["overview"]:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = self.axes["overview"]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        axis_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

        # Stage 1: snap to a turn-point marker.
        idx = find_closest_computation_index(
            (event.xdata, event.ydata), self.computations,
            threshold=0.08 * axis_range,
        )
        if idx is not None:
            self._select_computation(idx)
            return

        # Stage 2: project the click onto each leg, find the closest
        # leg, then pick the computation on that leg whose turn-point
        # is closest to the projection.
        click = np.array([event.xdata, event.ydata])
        best_leg: str | None = None
        best_proj: np.ndarray | None = None
        best_perp_dist = float("inf")
        for seg_id, leg in self.legs.items():
            start = np.asarray(leg["start"], dtype=float)
            end = np.asarray(leg["end"], dtype=float)
            seg = end - start
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq <= 0:
                continue
            t = float(np.dot(click - start, seg)) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj = start + t * seg
            d = float(np.linalg.norm(click - proj))
            if d < best_perp_dist:
                best_perp_dist = d
                best_leg = str(seg_id)
                best_proj = proj
        if best_leg is None or best_proj is None:
            return
        if best_perp_dist > 0.50 * axis_range:
            return

        candidates = [
            (i, c) for i, c in enumerate(self.computations)
            if str(c.get("seg_id")) == best_leg
        ]
        if not candidates:
            idx = find_closest_computation_index(
                (event.xdata, event.ydata), self.computations,
                threshold=axis_range,
            )
            if idx is not None:
                self._select_computation(idx)
            return
        chosen_idx = min(
            candidates,
            key=lambda ic: float(np.linalg.norm(
                np.asarray(ic[1]["turn_pt"]) - best_proj,
            )),
        )[0]
        self._select_computation(chosen_idx)

    # -----------------------------------------------------------------
    # Class method: embed in dialog
    # -----------------------------------------------------------------

    @classmethod
    def show_in_dialog(
        cls,
        dialog: ShowGeomRes,
        data: dict[str, Any],
        mode: str = "grounding",
        max_draft: float = DEFAULT_MAX_DRAFT,
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
        except Exception:
            return None  # type: ignore[return-value]
        proj = SimpleProjector(lon0, lat0)

        # Extract data
        legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms = \
            _build_legs_and_obstacles(data, proj, mode, max_draft)

        if not legs:
            return None  # type: ignore[return-value]

        # Run computations
        computations = _run_all_computations(legs, all_obstacles)

        # Create figure with three panels
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax_overview = fig.add_subplot(gs[0, :])
        ax_detail = fig.add_subplot(gs[1, 0])
        ax_waterfall = fig.add_subplot(gs[1, 1])
        axes = {
            "overview": ax_overview,
            "detail": ax_detail,
            "waterfall": ax_waterfall,
        }

        # Clean axes
        for a in axes.values():
            a.tick_params(labelsize=6)

        canvas = FigureCanvas(fig)

        # Wrap canvas + sidebar table in a horizontal splitter so the
        # user can drag the boundary.  The sidebar lists every
        # computation as a (Leg | Direction | Probability) row -- clicking
        # a row swaps the bottom panels.  This is more reliable than
        # clicking on legs in the matplotlib canvas itself, which
        # sometimes fails to receive events on certain Qt+QGIS combos.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import (
            QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView,
            QHeaderView,
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(canvas)

        sidebar = QTableWidget(len(computations), 3)
        sidebar.setHorizontalHeaderLabels(['Leg', 'Direction', 'Probability'])
        sidebar.verticalHeader().setVisible(False)
        sidebar.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sidebar.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows,
        )
        sidebar.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection,
        )
        # Sort rows by descending probability so the dominant
        # computations are at the top of the list.
        ordered = sorted(
            enumerate(computations),
            key=lambda ic: -_total_p_for_comp(ic[1]),
        )
        sidebar_index_map: list[int] = []  # row -> original computation idx
        for row, (orig_idx, comp) in enumerate(ordered):
            sidebar_index_map.append(orig_idx)
            seg_id = str(comp.get('seg_id', ''))
            d_name = comp.get('dir_info', {}).get('name', '')
            p_total = _total_p_for_comp(comp)
            for col, val in enumerate(
                (seg_id, d_name, f"{p_total:.3e}"),
            ):
                item = QTableWidgetItem(val)
                if col == 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter,
                    )
                sidebar.setItem(row, col, item)
        sidebar.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        sidebar.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch,
        )
        sidebar.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents,
        )
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(360)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Build and run visualizer
        viz = cls(
            fig, axes, legs, all_obstacles,
            depth_geoms, depth_geoms_deep, object_geoms,
            computations, mode,
        )
        viz.canvas = canvas
        viz.sidebar = sidebar
        viz.sidebar_index_map = sidebar_index_map
        viz.run_visualization()

        # Wire row selection to ``_select_computation``.  Use
        # ``itemSelectionChanged`` which fires both for mouse clicks
        # and for keyboard navigation.
        def _on_sidebar_selection_changed():
            rows = {
                idx.row() for idx in sidebar.selectedIndexes()
            }
            if not rows:
                return
            row = next(iter(rows))
            if 0 <= row < len(sidebar_index_map):
                viz._select_computation(sidebar_index_map[row])

        sidebar.itemSelectionChanged.connect(_on_sidebar_selection_changed)
        # Highlight the default-selected row (the one auto-picked by
        # ``run_visualization``) in the sidebar.
        try:
            default_row = sidebar_index_map.index(viz._selected_comp_idx)
            sidebar.selectRow(default_row)
        except (ValueError, AttributeError):
            sidebar.selectRow(0)

        mode_label = "Powered Allision" if mode == "allision" else "Powered Grounding"
        dialog.setWindowTitle(f"OMRAT - {mode_label} Cat II Visualization")

        fig.tight_layout()
        canvas.draw()
        return viz
