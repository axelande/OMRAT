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
    """Compute Cat II probabilities with shadow effects.

    Casts ``N_RAYS`` rays across the lateral distribution. Each ray finds the
    FIRST obstacle it hits -- this naturally creates shadows.

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

    ray_data: list[tuple[float, float, tuple[str, Any] | None, float | None]] = []
    obs_accum: dict[tuple[str, Any], dict] = defaultdict(lambda: {
        "mass": 0.0, "weighted_dist": 0.0, "p_integral": 0.0,
        "n_rays": 0, "ray_offsets": [], "ray_dists": [],
        "obs": None, "kind": None,
    })

    for off, m_i in zip(offsets, masses):
        ray_origin = turn_pt + off * perp

        best_d = float('inf')
        best_key: tuple[str, Any] | None = None
        best_obs = None
        best_kind: str | None = None

        for obs, kind in obstacles:
            d = _ray_hit_distance(ray_origin, ext_dir, MAX_RANGE, obs["geom"])
            if d is not None and 0 < d < best_d:
                best_d = d
                best_key = (kind, obs["id"])
                best_obs = obs
                best_kind = kind

        if best_key is not None:
            oa = obs_accum[best_key]
            oa["mass"] += m_i
            oa["weighted_dist"] += m_i * best_d
            if recovery > 0:
                oa["p_integral"] += m_i * exp(-best_d / recovery)
            oa["n_rays"] += 1
            oa["ray_offsets"].append(off)
            oa["ray_dists"].append(best_d)
            oa["obs"] = best_obs
            oa["kind"] = best_kind
            ray_data.append((off, m_i, best_key, best_d))
        else:
            ray_data.append((off, m_i, None, None))

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

        # Mark computations as clickable circles
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
            f"recovery={recovery:.0f}m",
            fontsize=9, fontweight="bold",
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

        # Annotate obstacles
        for key, s in sorted(summaries.items(), key=lambda x: x[1]["mean_dist"]):
            kind, obs_id = key
            tag = f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"
            c = self.obs_color_map[key]
            ray_offs = s["ray_offsets"]
            mid_lat = (min(ray_offs) + max(ray_offs)) / 2
            ax.annotate(
                f"{tag}\nmass={s['mass']:.4f}\nd={s['mean_dist'] / 1000:.1f} km\n"
                f"P={s['p_approx']:.2e}",
                xy=(s["mean_dist"], mid_lat),
                fontsize=6, color=c, fontweight="bold", ha="center", va="center",
                bbox=dict(fc="white", alpha=0.9, ec=c, pad=2,
                          boxstyle="round,pad=0.3"),
            )

        ax.axhline(d_info["mean"], color="red", linewidth=0.6,
                    linestyle="--", alpha=0.5, label="dist. mean")
        ax.set_xlabel("Along-track distance from turning point (m)", fontsize=7)
        ax.set_ylabel("Lateral offset from centreline (m)", fontsize=7)
        max_obs_dist = max((s["mean_dist"] for s in summaries.values()), default=10000)
        ax.set_xlim(-MAX_RANGE * 0.06, max(30000, max_obs_dist * 1.2))
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # -----------------------------------------------------------------
    # Waterfall panel
    # -----------------------------------------------------------------

    def _draw_waterfall(self, comp: dict) -> None:
        ax = self.axes["waterfall"]
        ax.clear()

        d_info = comp["dir_info"]
        offsets = comp["offsets"]
        pdf_vals = comp["pdf_vals"]
        summaries = comp["summaries"]
        dx = offsets[1] - offsets[0]

        ax.set_title(
            f"Distribution evolution: LEG {comp['seg_id']} {d_info['name']}\n"
            "Mass lost at each obstacle (shadow effect)",
            fontsize=9,
        )

        sorted_obs = sorted(summaries.items(), key=lambda x: x[1]["mean_dist"])
        remaining = pdf_vals.copy() * dx

        # Scale factor for visibility
        scale_factor = 3000
        orig_curve = pdf_vals * dx * scale_factor
        ax.fill_betweenx(offsets, 0, -orig_curve, color="gray", alpha=0.2,
                         label="Original distribution")
        ax.plot(-orig_curve, offsets, "k-", linewidth=1.5)

        for obs_idx, (key, s) in enumerate(sorted_obs):
            kind, obs_id = key
            tag = f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"
            d_mean = s["mean_dist"]
            c = self.obs_color_map[key]

            rem_curve = remaining * scale_factor
            ax.fill_betweenx(offsets, d_mean, d_mean - rem_curve, color=c, alpha=0.15)
            ax.plot(d_mean - rem_curve, offsets, color=c, linewidth=1.0)

            # Highlight consumed portion
            hit_offsets_set = set(s["ray_offsets"])
            consumed = np.zeros_like(remaining)
            for i, off in enumerate(offsets):
                if off in hit_offsets_set:
                    consumed[i] = remaining[i]
            consumed_curve = consumed * scale_factor
            ax.fill_betweenx(offsets, d_mean, d_mean - consumed_curve, color=c, alpha=0.5)

            ax.axvline(d_mean, color=c, linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(
                d_mean, offsets[-1] + (offsets[-1] - offsets[0]) * 0.03,
                f"{tag}\nd={d_mean / 1000:.1f}km\nmass={s['mass']:.4f}\n"
                f"P={s['p_approx']:.2e}",
                fontsize=6, color=c, ha="center", va="bottom", fontweight="bold",
                bbox=dict(fc="white", alpha=0.9, ec=c, pad=2),
            )

            # Remove consumed rays
            for off in s["ray_offsets"]:
                i = int(np.argmin(np.abs(offsets - off)))
                remaining[i] = 0.0

        # Final remaining
        if sorted_obs:
            final_x = max(s["mean_dist"] for _, s in sorted_obs) * 1.15
        else:
            final_x = 10000
        rem_curve = remaining * scale_factor
        ax.fill_betweenx(offsets, final_x, final_x - rem_curve,
                         color="gray", alpha=0.15)
        ax.plot(final_x - rem_curve, offsets, "k--", linewidth=0.8, alpha=0.5)
        remaining_mass = remaining.sum()
        ax.text(
            final_x, offsets[0] - (offsets[-1] - offsets[0]) * 0.05,
            f"Remaining mass:\n{remaining_mass:.4f}",
            fontsize=7, ha="center", va="top",
            bbox=dict(fc="white", ec="gray", pad=3),
        )

        ax.set_xlabel("Along-track distance from turning point (m)", fontsize=7)
        ax.set_ylabel("Lateral offset from centreline (m)", fontsize=7)
        ax.axhline(d_info["mean"], color="red", linewidth=0.6,
                    linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        # Legend
        legend_h = [Patch(fc="gray", alpha=0.2, label="Original dist.")]
        for key, s in sorted_obs:
            kind, obs_id = key
            tag = f"D#{obs_id}" if kind == "depth" else f"O#{obs_id}"
            c = self.obs_color_map[key]
            legend_h.append(Patch(fc=c, alpha=0.5, label=f"{tag} consumed"))
        ax.legend(handles=legend_h, fontsize=6, loc="upper right")

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

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect('button_press_event', self._on_overview_click)

    def _on_overview_click(self, event) -> None:
        """Handle click on the overview panel to select a turning point."""
        if event.inaxes != self.axes["overview"]:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Find closest computation turning point
        best_idx: int | None = None
        best_dist = float('inf')
        click = np.array([event.xdata, event.ydata])
        for ci, comp in enumerate(self.computations):
            tp = comp["turn_pt"]
            d = float(np.linalg.norm(tp - click))
            if d < best_dist:
                best_dist = d
                best_idx = ci

        # Only select if click is reasonably close (within 5% of axis range)
        ax = self.axes["overview"]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        threshold = 0.05 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if best_idx is not None and best_dist < threshold:
            self._select_computation(best_idx)

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
        layout.addWidget(canvas)

        # Build and run visualizer
        viz = cls(
            fig, axes, legs, all_obstacles,
            depth_geoms, depth_geoms_deep, object_geoms,
            computations, mode,
        )
        viz.canvas = canvas
        viz.run_visualization()

        mode_label = "Powered Allision" if mode == "allision" else "Powered Grounding"
        dialog.setWindowTitle(f"OMRAT - {mode_label} Cat II Visualization")

        fig.tight_layout()
        canvas.draw()
        return viz
