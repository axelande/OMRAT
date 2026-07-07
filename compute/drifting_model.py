"""
Drifting model mixin -- extracted from run_calculations.Calculation.

Contains all methods related to the drifting allision / grounding / anchoring
cascade, spatial pre-computation, report generation, and the top-level
``run_drifting_model`` entry point.

The class ``DriftingModelMixin`` is designed to be composed into the
``Calculation`` class at runtime via multiple inheritance.
"""

from typing import Any, Callable
import os
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
from pathlib import Path

import geopandas as gpd
from scipy import stats
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:
    from shapely import make_valid as shp_make_valid
except Exception:
    shp_make_valid = None

from compute.basic_equations import get_not_repaired
from compute.drift_corridor_geometry import (
    _compass_idx_to_math_idx,
    _extract_obstacle_segments,
    _create_drift_corridor,
    segment_corridor_overlap_length,
)
from compute.data_preparation import (
    clean_traffic,
    split_structures_and_depths,
    transform_to_utm,
    prepare_traffic_lists,
)
from geometries.get_drifting_overlap import (
    compute_min_distance_by_object,
    directional_distances_to_points,
)
from geometries.calculate_probability_holes import compute_probability_holes
from geometries.analytical_probability import (
    compute_probability_holes_analytical,
    compute_probability_analytical,
    _extract_polygon_rings,
)
from geometries.drift.shadow import create_obstacle_shadow, extract_polygons
from geometries.result_layers import create_result_layers
from compute.drifting_report_builder import DriftingReportBuilderMixin
from drifting.engine import LegState, compass_to_math_deg


class DriftingModelMixin(DriftingReportBuilderMixin):
    """Mixin providing the full drifting-model calculation pipeline.

    Expects the host class to provide:
      - ``self.p``  (parent OMRAT plugin reference)
      - ``self._report_progress(phase, progress, message)``
      - ``self._progress_callback``
      - ``self.drifting_allision_prob``
      - ``self.drifting_grounding_prob``
      - ``self.drifting_report``
      - ``self._last_structures``
      - ``self._last_depths``
      - ``self.allision_result_layer``
      - ``self.grounding_result_layer``
      - ``self.write_drifting_report_markdown(path, data)``  (from DriftingReportMixin)
    """

    # --- Drifting model helpers ---
    def _compute_reach_distance(self, data: dict[str, Any], longest_length: float) -> float:
            reach_distance = longest_length * 10.0
            try:
                rep = data.get('drift', {}).get('repair', {})
                use_ln = rep.get('use_lognormal', False)
                dist_type = rep.get('dist_type', '')
                t99_h = None

                if dist_type == 'weibull':
                    wb_shape = float(rep.get('wb_shape', 1.0))
                    wb_loc = float(rep.get('wb_loc', 0.0))
                    wb_scale = float(rep.get('wb_scale', 1.0))
                    t99_h = float(stats.weibull_min(c=wb_shape, loc=wb_loc, scale=wb_scale).ppf(0.99))
                elif use_ln:
                    s = float(rep.get('std', 0.0))
                    loc = float(rep.get('loc', 0.0))
                    scale = float(rep.get('scale', 1.0))
                    t99_h = float(stats.lognorm(s, loc=loc, scale=scale).ppf(0.99))

                if t99_h is not None and t99_h > 0:
                    drift_speed_kts = float(data.get('drift', {}).get('speed', 0.0))
                    drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s
                    if drift_speed > 0:
                        reach_distance = drift_speed * 3600.0 * t99_h
                        reach_distance = min(reach_distance, longest_length * 10.0)
            except Exception:
                pass
            return reach_distance

    def _distribution_centerline_stats(
            self,
            leg_distributions: list[Any],
            leg_weights: list[float],
        ) -> tuple[float, float]:
            """Approximate a mixed lateral distribution by mean offset and sigma."""
            if not leg_distributions or not leg_weights:
                return 0.0, 1.0

            weighted_entries: list[tuple[float, float, float]] = []
            for dist, weight in zip(leg_distributions, leg_weights):
                try:
                    w = float(weight)
                    if w <= 0:
                        continue
                    mean_val = float(dist.mean())
                    std_val = float(dist.std())
                    if not np.isfinite(mean_val) or not np.isfinite(std_val):
                        continue
                    weighted_entries.append((w, mean_val, max(0.0, std_val)))
                except Exception:
                    continue

            if not weighted_entries:
                return 0.0, 1.0

            total_weight = sum(w for w, _, _ in weighted_entries)
            if total_weight <= 0:
                return 0.0, 1.0

            mean_offset = sum(w * mean_val for w, mean_val, _ in weighted_entries) / total_weight
            variance = sum(
                w * (std_val ** 2 + (mean_val - mean_offset) ** 2)
                for w, mean_val, std_val in weighted_entries
            ) / total_weight
            return float(mean_offset), float(np.sqrt(max(variance, 1.0)))

    # ------------------------------------------------------------------
    # Shadow-coverage helpers (used by the cascade)
    # ------------------------------------------------------------------
    def _build_blocker_shadow(
            self,
            geom: BaseGeometry | None,
            compass_angle: float,
            corridor_bounds: tuple[float, float, float, float] | None,
            shadow_cache: dict[tuple[int, float], BaseGeometry] | None = None,
        ) -> BaseGeometry:
            """Quad-sweep shadow of a Polygon/MultiPolygon obstacle.

            Returns an empty Polygon if the input is empty or corridor_bounds is
            None.  MultiPolygons are handled by shadowing each component polygon
            and unioning the results.

            When ``shadow_cache`` is provided, the *full* obstacle shadow (union
            over all MultiPolygon components) is memoised by
            ``(id(geom), compass_angle)``.  ``geom`` is the obstacle's stored
            ``wkt`` field -- the same Python object across every (leg, dir)
            call -- so the cache hits across legs.  Caching at the component
            level doesn't work because ``shapely.MultiPolygon.geoms`` yields
            fresh Polygon objects per iteration, so component ``id(p)`` is
            different on every call.

            The caller must guarantee that ``corridor_bounds`` is the same for
            every cache hit (e.g. a global bound covering all legs) --
            ``create_obstacle_shadow`` computes an extrude distance from those
            bounds, so reusing shadows is only correct when they were built
            against the same bound.
            """
            if geom is None or geom.is_empty or corridor_bounds is None:
                return Polygon()

            if shadow_cache is not None:
                geom_key = (id(geom), float(compass_angle))
                cached = shadow_cache.get(geom_key)
                if cached is not None:
                    return cached

            try:
                polys = extract_polygons(geom)
            except Exception:
                polys = []
            if not polys:
                result: BaseGeometry = Polygon()
            else:
                shadows: list[BaseGeometry] = []
                for p in polys:
                    try:
                        s = create_obstacle_shadow(p, compass_angle, corridor_bounds)
                    except Exception:
                        s = Polygon()
                    if s is not None and not s.is_empty:
                        shadows.append(s)
                if not shadows:
                    result = Polygon()
                elif len(shadows) == 1:
                    result = shadows[0]
                else:
                    try:
                        result = unary_union(shadows)
                    except Exception:
                        result = shadows[0]

            if shadow_cache is not None:
                shadow_cache[(id(geom), float(compass_angle))] = result
            return result

    def _analytical_hole_for_geom(
            self,
            geom: BaseGeometry | None,
            leg: LineString,
            compass_angle: float,
            dists_list: list,
            weights_arr: np.ndarray,
            reach_distance: float,
            lateral_range: float,
            n_slices: int = 200,
        ) -> float:
            """Compute the analytical probability hole of a (possibly carved) geom.

            Mirrors ``compute_hole`` from the ``drifting/debug`` scripts.  Extracts
            all exterior + interior rings across Polygon / MultiPolygon /
            GeometryCollection inputs and passes them to
            :func:`compute_probability_analytical`.
            """
            if geom is None or geom.is_empty or reach_distance <= 0.0:
                return 0.0
            try:
                polys = extract_polygons(geom)
            except Exception:
                polys = []
            if not polys:
                return 0.0
            rings: list[np.ndarray] = []
            for p in polys:
                try:
                    rings.extend(_extract_polygon_rings(p))
                except Exception:
                    continue
            if not rings:
                return 0.0
            try:
                coords = np.array(leg.coords)
                if len(coords) < 2:
                    return 0.0
                leg_start = coords[0]
                leg_vec = coords[-1] - coords[0]
                leg_len = float(leg.length)
                leg_dir = leg_vec / leg_len if leg_len > 0 else np.array([1.0, 0.0])
                perp_dir = np.array([-leg_dir[1], leg_dir[0]])
                math_angle = compass_to_math_deg(compass_angle)
                rad = np.radians(math_angle)
                drift_vec = np.array([np.cos(rad), np.sin(rad)])
                h = compute_probability_analytical(
                    leg_start=leg_start,
                    leg_vec=leg_vec,
                    perp_dir=perp_dir,
                    drift_vec=drift_vec,
                    distance=float(reach_distance),
                    lateral_range=float(lateral_range),
                    polygon_rings=rings,
                    dists=dists_list,
                    weights=weights_arr,
                    n_slices=n_slices,
                )
                return max(0.0, float(h))
            except Exception:
                return 0.0

    def _edge_weighted_holes(
            self,
            obs_geom: BaseGeometry | None,
            drift_corridor: Polygon | None,
            drift_angle: float,
            leg: LineString | None,
            hole_pct: float,
            width_m: float | None = None,
        ) -> list[tuple[int | None, float]]:
            """Split obstacle hole percentage into edge-level fractions by overlap length.

            Distributes the analytically-computed *hole_pct* across individual
            edges in proportion to each edge's overlap length with the drift
            corridor.  The sum of all edge fractions equals *hole_pct*, which
            preserves the correct total probability while allowing per-edge
            distance weighting downstream.
            """
            if obs_geom is None or drift_corridor is None:
                return [(None, hole_pct)]

            try:
                segments = _extract_obstacle_segments(obs_geom)
                if not segments:
                    return [(None, hole_pct)]

                leg_centroid = None
                if leg is not None:
                    c = leg.centroid
                    leg_centroid = (c.x, c.y)

                # Batched drift-direction pre-filter.  ``segment_corridor_overlap_length``
                # runs this same test per-call, but at ~1.9M total calls for
                # proj.omrat the per-call ``np.radians``/``cos``/``sin`` and
                # vector allocations cost real time.  Here we do it once per
                # polygon in numpy and pass the surviving segment indices to
                # the shapely-heavy helper.
                skip = [False] * len(segments)
                if drift_angle is not None and leg_centroid is not None:
                    seg_arr = np.asarray(segments, dtype=float)  # (N, 2, 2)
                    p1 = seg_arr[:, 0, :]
                    p2 = seg_arr[:, 1, :]
                    dx = p2[:, 0] - p1[:, 0]
                    dy = p2[:, 1] - p1[:, 1]
                    seg_len_sq = dx * dx + dy * dy
                    ok_len = seg_len_sq > 0.0
                    inv_len = np.where(ok_len, seg_len_sq ** -0.5, 0.0)
                    # Outward normal for CCW polygons
                    nx = dy * inv_len
                    ny = -dx * inv_len

                    drift_rad = np.radians(drift_angle)
                    drift_ux = float(np.cos(drift_rad))
                    drift_uy = float(np.sin(drift_rad))

                    drift_into_segment = drift_ux * nx + drift_uy * ny
                    facing_ok = (np.abs(drift_into_segment) >= 0.17) & (drift_into_segment <= 0)

                    mx = 0.5 * (p1[:, 0] + p2[:, 0])
                    my = 0.5 * (p1[:, 1] + p2[:, 1])
                    vx = mx - leg_centroid[0]
                    vy = my - leg_centroid[1]
                    dist_to_segment = np.sqrt(vx * vx + vy * vy)
                    distance_ahead = vx * drift_ux + vy * drift_uy
                    ahead_ok = distance_ahead >= -0.5 * dist_to_segment

                    pass_mask = ok_len & facing_ok & ahead_ok
                    skip = (~pass_mask).tolist()

                weighted: list[tuple[int, float]] = []
                for seg_idx, segment in enumerate(segments):
                    if skip[seg_idx]:
                        continue
                    overlap_len = segment_corridor_overlap_length(
                        segment, drift_corridor, drift_angle, leg_centroid,
                    )
                    if overlap_len > 0.0:
                        weighted.append((seg_idx, overlap_len))

                if not weighted:
                    return [(None, hole_pct)]

                total_overlap = sum(v for _, v in weighted)
                if total_overlap <= 0.0:
                    return [(None, hole_pct)]

                return [
                    (seg_idx, hole_pct * (overlap_len / total_overlap))
                    for seg_idx, overlap_len in weighted
                ]
            except Exception:
                return [(None, hole_pct)]

    # ------------------------------------------------------------------
    # Shadow + edge-geometry precompute (ship-independent)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_global_shadow_bounds(
        transformed_lines: list[LineString],
        structures: list[dict[str, Any]],
        depths: list[dict[str, Any]],
        reach_distance: float,
    ) -> tuple[float, float, float, float] | None:
        """Union of all leg + obstacle bboxes, padded by ``reach_distance``.

        Used as a uniform extrude bound so the shadow memo
        (keyed by ``(polygon, compass_angle)``) hits across legs.
        Returns ``None`` when nothing has a finite bounding box.
        """
        xs: list[float] = []
        ys: list[float] = []
        for line in transformed_lines:
            try:
                b = line.bounds
                xs.extend([b[0], b[2]])
                ys.extend([b[1], b[3]])
            except Exception:
                pass
        for source in (structures, depths):
            for item in source:
                g = item.get('wkt')
                if g is None or g.is_empty:
                    continue
                b = g.bounds
                xs.extend([b[0], b[2]])
                ys.extend([b[1], b[3]])
        if not (xs and ys):
            return None
        reach_pad = max(1000.0, float(reach_distance))
        return (
            min(xs) - reach_pad, min(ys) - reach_pad,
            max(xs) + reach_pad, max(ys) + reach_pad,
        )

    @staticmethod
    def _precompute_leg_lateral_params(
        transformed_lines: list[LineString],
        distributions: list[list[Any]],
        weights: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Per-leg lateral-distribution scalars + ``LegState`` for each leg.

        Computed once so every ``(leg, direction)`` worker shares the
        same lateral-distribution parameters; the result is consumed
        by ``_shadow_task`` inside ``_precompute_shadow_layer``.
        """
        out: list[dict[str, Any]] = []
        for leg_idx, line in enumerate(transformed_lines):
            try:
                dists_dir = (
                    distributions[leg_idx]
                    if leg_idx < len(distributions) else []
                )
                wgts = weights[leg_idx] if leg_idx < len(weights) else []
                w_dir: np.ndarray | None = None
                lateral_spread = 0.0
                if dists_dir and wgts:
                    w_dir = np.array(wgts)
                    if w_dir.sum() > 0:
                        w_dir = w_dir / w_dir.sum()
                        weighted_std = float(np.sqrt(sum(
                            wt * (dist.std() ** 2)
                            for dist, wt in zip(dists_dir, w_dir) if wt > 0
                        )))
                        lateral_spread = 5.0 * weighted_std
            except Exception:
                dists_dir = []
                w_dir = None
                lateral_spread = 0.0
            try:
                coords = list(line.coords)
                if len(coords) >= 2:
                    leg_state = LegState(
                        leg_id=str(leg_idx),
                        line=line,
                        mean_offset_m=0.0,
                        lateral_sigma_m=max(1.0, lateral_spread / 5.0),
                    )
                else:
                    leg_state = None
            except Exception:
                leg_state = None
            out.append({
                'dists_dir': dists_dir,
                'w_dir': w_dir,
                'lateral_spread': lateral_spread,
                'leg_state': leg_state,
                'line': line,
            })
        return out

    def _build_edge_geom_for_poly(
        self,
        poly, drift_corridor, math_angle: float, line, leg_state,
        drift_repair: dict, drift_speed: float, use_leg_offset: bool,
        compass_angle: float,
    ) -> list[dict[str, Any]]:
        if poly is None or poly.is_empty or drift_corridor is None:
            return []
        try:
            segments = _extract_obstacle_segments(poly)
            raw = self._edge_weighted_holes(poly, drift_corridor, math_angle, line, 1.0, None)
            total_frac = sum(frac for _, frac in raw if frac > 0.0)
            selected = [(si, frac) for si, frac in raw
                        if frac > 0.0 and si is not None and 0 <= si < len(segments)]
            if not selected or leg_state is None:
                return []
            endpoints = np.empty((2 * len(selected), 2), dtype=float)
            for i, (si, _) in enumerate(selected):
                e = segments[si]
                endpoints[2 * i, 0] = e[0][0]; endpoints[2 * i, 1] = e[0][1]
                endpoints[2 * i + 1, 0] = e[1][0]; endpoints[2 * i + 1, 1] = e[1][1]
            dists = directional_distances_to_points(
                endpoints, line, compass_angle, use_leg_offset=use_leg_offset)
            items: list[dict[str, Any]] = []
            for i, (si, frac) in enumerate(selected):
                valid = [float(d) for d in [dists[2*i], dists[2*i+1]] if np.isfinite(d)]
                if not valid:
                    continue
                edge_dist = sum(valid) / len(valid)
                items.append({
                    'seg_idx': si,
                    'len_frac': frac / total_frac if total_frac > 0 else 0.0,
                    'edge_dist': edge_dist,
                    'edge_p_nr': get_not_repaired(drift_repair, drift_speed, edge_dist),
                })
            return items
        except Exception:
            return []

    def _build_shadow_entry(
        self,
        leg_idx: int, d_idx: int,
        leg_precomputed: list, structures: list, depths: list,
        struct_min_dists, depth_min_dists, reach_distance: float,
        drift_repair: dict, drift_speed: float, use_leg_offset: bool,
        shadow_memo: dict, global_shadow_bounds,
    ) -> tuple[tuple[int, int], dict[str, Any]]:
        lp = leg_precomputed[leg_idx]
        line, dists_dir, w_dir = lp['line'], lp['dists_dir'], lp['w_dir']
        lateral_spread, leg_state = lp['lateral_spread'], lp['leg_state']
        compass_angle = d_idx * 45
        math_angle = (90 - compass_angle) % 360
        math_dir_idx = _compass_idx_to_math_idx(d_idx)
        drift_corridor, bounds = self._compute_drift_corridor_and_bounds(
            line, math_angle, reach_distance, lateral_spread, dists_dir, w_dir, structures, depths)
        shadow_bounds = global_shadow_bounds if global_shadow_bounds is not None else bounds
        shadows: dict = {}
        edge_geom: dict = {}
        for s_idx, s in enumerate(structures):
            poly = s.get('wkt')
            if poly is None or poly.is_empty:
                continue
            if struct_min_dists is not None:
                try:
                    d = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                    if d is None or (reach_distance > 0 and d > reach_distance * 1.01):
                        continue
                except (IndexError, TypeError):
                    pass
            try:
                sh = self._build_blocker_shadow(poly, compass_angle, shadow_bounds, shadow_memo)
            except Exception:
                sh = Polygon()
            shadows[('allision', s_idx)] = sh
            edge_geom[('allision', s_idx)] = self._build_edge_geom_for_poly(
                poly, drift_corridor, math_angle, line, leg_state,
                drift_repair, drift_speed, use_leg_offset, compass_angle)
        for d_idx2, dep in enumerate(depths):
            poly = dep.get('wkt')
            if poly is None or poly.is_empty:
                continue
            if depth_min_dists is not None:
                try:
                    d = depth_min_dists[leg_idx][math_dir_idx][d_idx2]
                    if d is None or (reach_distance > 0 and d > reach_distance * 1.01):
                        continue
                except (IndexError, TypeError):
                    pass
            try:
                sh = self._build_blocker_shadow(poly, compass_angle, shadow_bounds, shadow_memo)
            except Exception:
                sh = Polygon()
            shadows[('depth', d_idx2)] = sh
            edge_geom[('depth', d_idx2)] = self._build_edge_geom_for_poly(
                poly, drift_corridor, math_angle, line, leg_state,
                drift_repair, drift_speed, use_leg_offset, compass_angle)
        return (leg_idx, d_idx), {
            'corridor': drift_corridor, 'bounds': bounds,
            'dists_list': dists_dir, 'weights_arr': w_dir,
            'lateral_spread': lateral_spread, 'leg_state_tmp': leg_state,
            'shadow': shadows, 'edge_geom': edge_geom,
        }

    def _compute_drift_corridor_and_bounds(
        self,
        line, math_angle: float, reach_distance: float, lateral_spread: float,
        dists_dir: list, w_dir, structures: list, depths: list,
    ) -> tuple:
        drift_corridor = None
        if dists_dir and w_dir is not None and lateral_spread > 0.0 and reach_distance > 0:
            try:
                drift_corridor = _create_drift_corridor(line, math_angle, reach_distance, lateral_spread)
            except Exception:
                drift_corridor = None
        if drift_corridor is not None and not drift_corridor.is_empty:
            return drift_corridor, drift_corridor.bounds
        try:
            xs = [line.bounds[0], line.bounds[2]]
            ys = [line.bounds[1], line.bounds[3]]
            for s in structures:
                g = s.get('wkt')
                if g is not None and not g.is_empty:
                    xs.extend([g.bounds[0], g.bounds[2]]); ys.extend([g.bounds[1], g.bounds[3]])
            for dep in depths:
                g = dep.get('wkt')
                if g is not None and not g.is_empty:
                    xs.extend([g.bounds[0], g.bounds[2]]); ys.extend([g.bounds[1], g.bounds[3]])
            pad = max(1000.0, (max(xs) - min(xs)) * 0.1)
            return drift_corridor, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        except Exception:
            return drift_corridor, None

    def _precompute_shadow_layer(
            self,
            transformed_lines: list[LineString],
            distributions: list[list[Any]],
            weights: list[list[float]],
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists,
            depth_min_dists,
            reach_distance: float,
            drift_repair: dict[str, Any],
            drift_speed: float,
            use_leg_offset_for_distance: bool,
            progress_base: float = 0.0,
            progress_span: float = 1.0,
        ) -> dict[tuple[int, int], dict[str, Any]]:
        cache: dict[tuple[int, int], dict[str, Any]] = {}
        n_legs = len(transformed_lines)
        total_units = max(1, n_legs * 8)
        # shadow_memo is shared so cache hits across legs/directions for the same polygon.
        # Profile: 26,545 create_obstacle_shadow calls (~234 s) before this cache.
        shadow_memo: dict[tuple[int, float], BaseGeometry] = {}
        global_shadow_bounds = self._compute_global_shadow_bounds(
            transformed_lines, structures, depths, reach_distance)
        leg_precomputed = self._precompute_leg_lateral_params(
            transformed_lines, distributions, weights)

        def _shadow_task(leg_idx: int, d_idx: int) -> tuple[tuple[int, int], dict[str, Any]]:
            return self._build_shadow_entry(
                leg_idx, d_idx, leg_precomputed, structures, depths,
                struct_min_dists, depth_min_dists, reach_distance,
                drift_repair, drift_speed, use_leg_offset_for_distance,
                shadow_memo, global_shadow_bounds,
            )

        return self._run_shadow_pool(
            _shadow_task, cache, n_legs, total_units, progress_base, progress_span,
        )

    def _run_shadow_pool(
        self,
        shadow_task: Callable[[int, int], tuple[tuple[int, int], dict[str, Any]]],
        cache: dict[tuple[int, int], dict[str, Any]],
        n_legs: int,
        total_units: int,
        progress_base: float,
        progress_span: float,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Dispatch ``shadow_task`` across (leg, direction) tuples.

        Parallelises with :class:`ThreadPoolExecutor` when there's
        enough work; falls back to a sequential loop for tiny /
        degenerate inputs.  Reports progress every 5% and propagates
        cancellation by stamping ``cache['__cancelled__']`` and
        returning early.
        """
        max_workers = max(1, min(8, cpu_count() - 1))
        completed = 0
        cancelled = False

        def _report(msg: str) -> bool:
            phase_progress = completed / total_units
            overall = progress_base + progress_span * min(1.0, phase_progress)
            return self._report_progress('shadow', overall, msg)

        report_step = max(1, total_units // 20)

        if n_legs <= 1 or max_workers <= 1:
            for leg_idx in range(n_legs):
                for d_idx in range(8):
                    (key, entry) = shadow_task(leg_idx, d_idx)
                    cache[key] = entry
                    completed += 1
                    if (
                        completed % report_step == 0
                        or completed == total_units
                    ):
                        if not _report(
                            f"Drifting - shadows ({completed}/{total_units})"
                        ):
                            cache['__cancelled__'] = True  # type: ignore[index]
                            return cache
            _report("Drifting - shadows done")
            return cache

        # Shapely + numpy release the GIL during geometry / linear-algebra
        # operations, so Python threads give real parallelism here.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(shadow_task, leg_idx, d_idx): (leg_idx, d_idx)
                for leg_idx in range(n_legs)
                for d_idx in range(8)
            }
            try:
                for fut in as_completed(futures):
                    try:
                        (key, entry) = fut.result()
                        cache[key] = entry
                    except Exception:
                        # Swallow per-task failures -- an empty cache
                        # entry simply falls back to precomputed h_X in
                        # the cascade.
                        pass
                    completed += 1
                    if (
                        completed % report_step == 0
                        or completed == total_units
                    ):
                        if not _report(
                            f"Drifting - shadows ({completed}/{total_units})"
                        ):
                            cancelled = True
                            for f in futures:
                                f.cancel()
                            break
            except Exception:
                pass
        if cancelled:
            cache['__cancelled__'] = True  # type: ignore[index]
            return cache

        _report("Drifting - shadows done")
        return cache


    def _build_obstacle_list_for_bucket(
        self, leg_idx: int, d_idx: int, cell: dict,
        anchor_d: float, structures: list, depths: list,
        struct_min_dists, depth_min_dists,
        struct_probability_holes, depth_probability_holes, threshold_to_idx,
    ) -> list[tuple[str, int, float, float]]:
        draught = float(cell.get('draught', 0.0))
        return self._collect_cell_obstacles(
            leg_idx, d_idx, draught, anchor_d, structures, depths,
            struct_min_dists, depth_min_dists,
            struct_probability_holes, depth_probability_holes,
            threshold_to_idx,
        )

    def _compute_bucket_entries(
        self,
        key: tuple[int, int, tuple],
        obstacles: list[tuple[str, int, float, float]],
        shadow_cache: dict, transformed_lines: list,
        structures: list, depths: list, reach_distance: float,
    ) -> tuple[tuple, list[dict] | None]:
        leg_idx, d_idx, _bk = key
        ld_entry = shadow_cache.get((leg_idx, d_idx))
        if ld_entry is None:
            return key, None
        shadows_map = ld_entry.get('shadow', {})
        dists_dir = ld_entry.get('dists_list', [])
        w_dir = ld_entry.get('weights_arr', None)
        lateral_spread = ld_entry.get('lateral_spread', 0.0)
        compass_angle = d_idx * 45
        _have_integrator = w_dir is not None and dists_dir and lateral_spread > 0.0
        sorted_obs = sorted(obstacles, key=lambda x: float(x[2]))
        blocker_union: BaseGeometry | None = None
        anchor_union: BaseGeometry | None = None
        entries: list[dict[str, Any]] = []
        for obs_type, obs_idx, dist, hole_pct in sorted_obs:
            if obs_type == 'allision':
                s = structures[obs_idx] if obs_idx < len(structures) else None
                geom_X = s.get('wkt') if s is not None else None
            else:
                d_obj = depths[obs_idx] if obs_idx < len(depths) else None
                geom_X = d_obj.get('wkt') if d_obj is not None else None
            if geom_X is None or geom_X.is_empty:
                entries.append({'obs_type': obs_type, 'obs_idx': obs_idx,
                    'dist': dist, 'hole_pct': hole_pct, 'h_reach': float(hole_pct), 'h_in_anchor': 0.0})
                continue
            carve = blocker_union is not None and not blocker_union.is_empty
            # Short-circuit when blocker shadow doesn't touch this obstacle --
            # ``intersects`` is orders of magnitude cheaper than the full
            # ``difference`` + analytical integration.
            if carve and not blocker_union.intersects(geom_X):
                carve = False
            if carve:
                try:
                    reach = geom_X.difference(blocker_union)
                except Exception:
                    reach = geom_X
                if reach.is_empty:
                    h_reach = 0.0
                elif _have_integrator:
                    h_reach = self._analytical_hole_for_geom(
                        reach, transformed_lines[leg_idx], compass_angle,
                        dists_dir, w_dir, reach_distance, lateral_spread)
                else:
                    h_reach = float(hole_pct)
            else:
                reach = geom_X
                h_reach = float(hole_pct)
            h_in_anchor = 0.0
            if (obs_type != 'anchoring' and anchor_union is not None
                    and not anchor_union.is_empty and not reach.is_empty
                    and _have_integrator and anchor_union.intersects(reach)):
                try:
                    _in = reach.intersection(anchor_union)
                    if _in is not None and not _in.is_empty:
                        h_in_anchor = self._analytical_hole_for_geom(
                            _in, transformed_lines[leg_idx], compass_angle,
                            dists_dir, w_dir, reach_distance, lateral_spread)
                except Exception:
                    h_in_anchor = 0.0
            entries.append({'obs_type': obs_type, 'obs_idx': obs_idx,
                'dist': dist, 'hole_pct': hole_pct, 'h_reach': h_reach, 'h_in_anchor': h_in_anchor})
            # Anchoring obstacles reference depth polygons; map 'anchoring' -> 'depth'
            # so the shadow lookup populates anchor_union correctly.
            lookup_type = 'depth' if obs_type == 'anchoring' else obs_type
            _s = shadows_map.get((lookup_type, obs_idx))
            if _s is None or _s.is_empty:
                continue
            if obs_type in ('allision', 'grounding'):
                blocker_union = _s if blocker_union is None else unary_union([blocker_union, _s])
            elif obs_type == 'anchoring':
                anchor_union = _s if anchor_union is None else unary_union([anchor_union, _s])
        return key, entries

    def _collect_bucket_obs(
        self,
        transformed_lines: list, traffic_by_leg: list, anchor_d: float,
        structures: list, depths: list, struct_min_dists, depth_min_dists,
        struct_probability_holes, depth_probability_holes, threshold_to_idx,
    ) -> dict[tuple[int, int, tuple], list[tuple[str, int, float, float]]]:
        bucket_obs: dict[tuple[int, int, tuple], list] = {}
        for leg_idx in range(len(transformed_lines)):
            cells = traffic_by_leg[leg_idx] if leg_idx < len(traffic_by_leg) else []
            for cell in cells:
                if float(cell.get('speed', 0.0)) <= 0.0 or float(cell.get('freq', 0.0)) <= 0.0:
                    continue
                for d_idx in range(8):
                    obstacles = self._build_obstacle_list_for_bucket(
                        leg_idx, d_idx, cell, anchor_d, structures, depths,
                        struct_min_dists, depth_min_dists,
                        struct_probability_holes, depth_probability_holes, threshold_to_idx,
                    )
                    if not obstacles:
                        continue
                    bucket_key = tuple(sorted((ot, oi) for ot, oi, _d, _h in obstacles))
                    key = (leg_idx, d_idx, bucket_key)
                    if key not in bucket_obs:
                        bucket_obs[key] = obstacles
        return bucket_obs

    def _run_bucket_parallel(
        self,
        bucket_obs: dict, memo: dict,
        progress_base: float, progress_span: float,
        shadow_cache: dict, transformed_lines: list,
        structures: list, depths: list, reach_distance: float,
    ) -> None:
        total_units = max(1, len(bucket_obs))
        completed = 0
        cancelled = False

        def _report(msg: str) -> bool:
            phase = progress_base + progress_span * (completed / total_units)
            return self._report_progress('shadow', min(1.0, phase), msg)

        def _compute(item):
            return self._compute_bucket_entries(
                item[0], item[1], shadow_cache, transformed_lines,
                structures, depths, reach_distance)

        max_workers = max(1, min(8, cpu_count() - 1))
        if total_units <= 1 or max_workers <= 1:
            for item in bucket_obs.items():
                key, entries = _compute(item)
                if entries is not None:
                    memo[key] = entries
                completed += 1
                if completed % max(1, total_units // 50) == 0 or completed == total_units:
                    if not _report(f"Drifting - bucket memo ({completed}/{total_units})"):
                        memo['__cancelled__'] = True  # type: ignore[index]
                        return
            return
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute, item): item[0] for item in bucket_obs.items()}
            try:
                for fut in as_completed(futures):
                    try:
                        key, entries = fut.result()
                        if entries is not None:
                            memo[key] = entries
                    except Exception:
                        pass
                    completed += 1
                    if completed % max(1, total_units // 50) == 0 or completed == total_units:
                        if not _report(f"Drifting - bucket memo ({completed}/{total_units})"):
                            cancelled = True
                            for f in futures:
                                f.cancel()
                            break
            except Exception:
                pass
        if cancelled:
            memo['__cancelled__'] = True  # type: ignore[index]

    def _precompute_bucket_memo(
            self,
            data: dict[str, Any],
            transformed_lines: list[LineString],
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists,
            depth_min_dists,
            struct_probability_holes,
            depth_probability_holes,
            shadow_cache: dict[tuple[int, int], dict[str, Any]],
            threshold_to_idx: dict[float, int] | None,
            reach_distance: float,
            progress_base: float = 0.5,
            progress_span: float = 0.5,
        ) -> dict[tuple[int, int, tuple], list[dict[str, Any]]]:
        drift = data['drift']
        anchor_d = float(drift.get('anchor_d', 0.0))
        traffic_by_leg: list[list[dict[str, float]]] = [
            leg_traffic for _, _, _, leg_traffic, _ in clean_traffic(data)
        ]
        bucket_obs = self._collect_bucket_obs(
            transformed_lines, traffic_by_leg, anchor_d, structures, depths,
            struct_min_dists, depth_min_dists,
            struct_probability_holes, depth_probability_holes, threshold_to_idx,
        )
        memo: dict[tuple[int, int, tuple], list[dict[str, Any]]] = {}
        self._run_bucket_parallel(
            bucket_obs, memo, progress_base, progress_span,
            shadow_cache, transformed_lines, structures, depths, reach_distance,
        )
        return memo

    def _build_transformed(self, data: dict[str, Any]) -> tuple[
            list[LineString], list[list[Any]], list[list[float]], list[str],
            list[dict[str, Any]], list[dict[str, Any]],
            list[gpd.GeoDataFrame], list[gpd.GeoDataFrame],
            list[LineString]
        ]:
            from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
            from shapely.ops import transform
            from compute.data_preparation import _is_qgis_available

            lines, distributions, weights, line_names = prepare_traffic_lists(data)
            structures, depths = split_structures_and_depths(data)
            structure_geoms = [s['wkt'] for s in structures]
            depth_geoms = [d['wkt'] for d in depths]
            transformed_lines, transformed_objs_all, utm_epsg = transform_to_utm(lines, structure_geoms + depth_geoms)
            # Persist CRS info for downstream runtime-debug geometry export
            self._last_utm_epsg = utm_epsg
            n_struct = len(structure_geoms)
            transformed_structs = transformed_objs_all[:n_struct]
            transformed_depths = transformed_objs_all[n_struct:]

            # Create reverse transform (UTM -> WGS84) for converting fixed geometries back
            # This ensures wkt_wgs84 has the same vertex order as wkt (UTM)
            if _is_qgis_available():
                wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
                utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")
                transform_context = QgsProject.instance().transformContext()
                reverse_transform = QgsCoordinateTransform(utm_crs, wgs84_crs, transform_context)

                def transform_utm_to_wgs84(geom):
                    """Transform a shapely geometry from UTM back to WGS84."""
                    from qgis.core import QgsPointXY
                    def reverse_coords(x, y):
                        point = reverse_transform.transform(QgsPointXY(x, y))
                        return point.x(), point.y()
                    return transform(reverse_coords, geom)
            else:
                from pyproj import Transformer as _RevTransformer
                _rev_proj = _RevTransformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

                def transform_utm_to_wgs84(geom):
                    return transform(lambda x, y: _rev_proj.transform(x, y), geom)

            # Cache converter for runtime segment-level debug metadata
            self._segment_utm_to_wgs84 = transform_utm_to_wgs84

            # Fix invalid geometries and split any MultiPolygons that may arise from make_valid
            # Note: split_structures_and_depths already splits MultiPolygons, but make_valid
            # can sometimes create new MultiPolygons from invalid geometries
            fixed_structs = []
            fixed_structs_meta = []  # Track original structure metadata
            for i, g in enumerate(transformed_structs):
                try:
                    fixed = shp_make_valid(g) if shp_make_valid is not None else g.buffer(0)
                except Exception:
                    fixed = g

                # Split MultiPolygons into individual Polygons (safety for make_valid results)
                orig = structures[i] if i < len(structures) else {'id': f'struct_{i}', 'height': 0.0}
                if fixed.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(fixed.geoms):
                        fixed_structs.append(poly)
                        # Transform the UTM polygon back to WGS84 so segment indices match
                        poly_wgs84 = transform_utm_to_wgs84(poly)
                        fixed_structs_meta.append({
                            'id': f"{orig['id']}_{j}" if len(fixed.geoms) > 1 else orig['id'],
                            'height': orig['height'],
                            'wkt': poly,
                            'wkt_wgs84': poly_wgs84,  # Transformed back from UTM for consistent segment indices
                        })
                else:
                    fixed_structs.append(fixed)
                    # Transform the UTM geometry back to WGS84 so segment indices match
                    fixed_wgs84 = transform_utm_to_wgs84(fixed)
                    fixed_structs_meta.append({
                        'id': orig['id'],
                        'height': orig['height'],
                        'wkt': fixed,
                        'wkt_wgs84': fixed_wgs84,  # Transformed back from UTM for consistent segment indices
                    })

            fixed_depths = []
            fixed_depths_meta = []  # Track original depth metadata
            for i, g in enumerate(transformed_depths):
                try:
                    fixed = shp_make_valid(g) if shp_make_valid is not None else g.buffer(0)
                except Exception:
                    fixed = g

                # Get the depth value for this geometry
                depth_val = depths[i]['depth'] if i < len(depths) else 0.0
                depth_id = depths[i]['id'] if i < len(depths) else f'depth_{i}'

                # Split MultiPolygons into individual Polygons (safety for make_valid results)
                if fixed.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(fixed.geoms):
                        fixed_depths.append(poly)
                        # Transform the UTM polygon back to WGS84 so segment indices match
                        poly_wgs84 = transform_utm_to_wgs84(poly)
                        fixed_depths_meta.append({
                            'id': f"{depth_id}_{j}" if len(fixed.geoms) > 1 else depth_id,
                            'depth': depth_val,
                            'wkt': poly,
                            'wkt_wgs84': poly_wgs84,  # Transformed back from UTM for consistent segment indices
                        })
                else:
                    fixed_depths.append(fixed)
                    # Transform the UTM geometry back to WGS84 so segment indices match
                    fixed_wgs84 = transform_utm_to_wgs84(fixed)
                    fixed_depths_meta.append({
                        'id': depth_id,
                        'depth': depth_val,
                        'wkt': fixed,
                        'wkt_wgs84': fixed_wgs84,  # Transformed back from UTM for consistent segment indices
                    })

            structs_gdfs = [gpd.GeoDataFrame(geometry=[g]) for g in fixed_structs]
            # Include depth values in the GeoDataFrame
            depths_gdfs = [gpd.GeoDataFrame({'depth': [fixed_depths_meta[i]['depth']], 'geometry': [g]})
                          for i, g in enumerate(fixed_depths)]
            return (
                lines, distributions, weights, line_names,
                fixed_structs_meta, fixed_depths_meta,
                structs_gdfs, depths_gdfs,
                transformed_lines,
            )

    def _precompute_spatial(self,
            transformed_lines: list[LineString],
            distributions: list[list[Any]],
            weights: list[list[float]],
            structs_gdfs: list[gpd.GeoDataFrame],
            depths_gdfs: list[gpd.GeoDataFrame],
            reach_distance: float,
            data: dict[str, Any] | None = None,
        ) -> tuple[list, list, list, list]:
            struct_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            depth_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            # Calculate probability holes using FAST Monte Carlo method
            # Unified progress tracking across structures AND depths
            # Count actual objects for progress estimation
            def count_objects(gdf_list):
                return sum(len(gdf) for gdf in gdf_list)

            struct_obj_count = count_objects(structs_gdfs) if len(structs_gdfs) > 0 else 0
            depth_obj_count = count_objects(depths_gdfs) if len(depths_gdfs) > 0 else 0

            # Estimate total work (8 directions x objects per leg)
            # Structures use dblquad (~slow), depths use fast method (~quick)
            # Weight: 1 structure ~ 100 depth objects in terms of computation time
            weighted_struct = struct_obj_count * 100
            weighted_depth = depth_obj_count * 1
            total_weighted_work = max(1, weighted_struct + weighted_depth)

            # Track progress across BOTH calculations within the 'spatial' phase
            struct_done = False

            def spatial_progress_callback(completed: int, total: int, msg: str) -> bool:
                """Report progress within the spatial phase (0-60% of overall)"""
                # Calculate weighted progress within spatial phase
                if not struct_done:
                    # Currently calculating structures (first half of spatial)
                    weighted_progress = (completed / max(total, 1)) * weighted_struct
                    label = f"Drifting - structure probabilities ({completed}/{total})"
                else:
                    # Currently calculating depths (second half of spatial)
                    weighted_progress = weighted_struct + (completed / max(total, 1)) * weighted_depth
                    label = f"Drifting - depth probabilities ({completed}/{total})"

                # Convert to fraction of spatial phase (0.0 to 1.0)
                phase_progress = weighted_progress / total_weighted_work
                return self._report_progress('spatial', phase_progress, label)

            # Choose probability hole computation method
            use_analytical = data.get('use_analytical', True) if data else True
            compute_holes_fn = (
                compute_probability_holes_analytical if use_analytical
                else compute_probability_holes
            )
            method_name = "analytical cross-section CDF" if use_analytical else "Monte Carlo"
            logger.info(f"Probability holes: using {method_name} method")

            # Calculate structures (allision)
            struct_probability_holes = compute_holes_fn(
                transformed_lines, distributions, weights, structs_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(structs_gdfs) > 0 else []

            struct_done = True  # Switch to depths

            # Calculate depths (grounding)
            depth_probability_holes = compute_holes_fn(
                transformed_lines, distributions, weights, depths_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(depths_gdfs) > 0 else []
            return (
                struct_min_dists, depth_min_dists,
                struct_probability_holes,
                depth_probability_holes,
            )


    def _debug_add_trace(
        self,
        report_dict: dict[str, Any],
        leg_dir_key: str, obs_key: str, obs_type: str,
        contrib: float, dist: float, hole_pct: float, remaining_before: float,
        p_nr: float | None = None, anchor_effect: float | None = None,
        exposure_factor: float | None = None, rp: float | None = None,
        base: float | None = None, freq: float | None = None,
    ) -> None:
        dbg = report_dict.setdefault('debug_obstacles', {})
        key = f"{leg_dir_key}|{obs_key}|{obs_type}"
        rec = dbg.setdefault(key, {
            'leg_dir_key': leg_dir_key, 'obstacle': obs_key, 'type': obs_type,
            'contrib': 0.0, 'weight': 0.0, 'dist_sum': 0.0, 'hole_sum': 0.0,
            'remaining_before_sum': 0.0, 'p_nr_sum': 0.0, 'p_nr_weight': 0.0,
            'anchor_effect_sum': 0.0, 'anchor_effect_weight': 0.0,
            'exposure_sum': 0.0, 'exposure_weight': 0.0, 'rp': 0.0,
            'base_sum': 0.0, 'base_weight': 0.0, 'freq_sum': 0.0,
            'freq_weight': 0.0, 'count': 0,
        })
        w = max(float(contrib), 0.0)
        rec['contrib'] += float(contrib)
        rec['weight'] += w
        rec['dist_sum'] += float(dist) * w
        rec['hole_sum'] += float(hole_pct) * w
        rec['remaining_before_sum'] += float(remaining_before) * w
        if p_nr is not None:
            rec['p_nr_sum'] += float(p_nr) * w; rec['p_nr_weight'] += w
        if anchor_effect is not None:
            rec['anchor_effect_sum'] += float(anchor_effect) * w; rec['anchor_effect_weight'] += w
        if exposure_factor is not None:
            rec['exposure_sum'] += float(exposure_factor) * w; rec['exposure_weight'] += w
        if rp is not None and rec['rp'] == 0.0:
            rec['rp'] = float(rp)
        if base is not None:
            rec['base_sum'] += float(base) * w; rec['base_weight'] += w
        if freq is not None:
            rec['freq_sum'] += float(freq) * w; rec['freq_weight'] += w
        rec['count'] += 1

    def _init_drift_report(self, debug_trace: bool) -> dict[str, Any]:
        report: dict[str, Any] = {
            'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0},
            'by_leg_direction': {}, 'by_object': {},
            'by_structure_legdir': {}, 'by_depth_legdir': {}, 'by_anchoring_legdir': {},
            'by_structure_segment_legdir': {}, 'by_depth_segment_legdir': {},
            'by_anchoring_segment_legdir': {},
            'by_cell_allision': {}, 'by_cell_grounding': {},
        }
        if debug_trace:
            report['debug_obstacles'] = {}
        return report

    def _process_leg_cells(
        self,
        leg_idx: int, line, seg_id: str, line_length: float,
        ship_cells: list, report: dict,
        drift: dict, drift_speed: float, anchor_p: float, anchor_d: float,
        structures: list, depths: list,
        struct_min_dists, depth_min_dists,
        struct_probability_holes, depth_probability_holes,
        threshold_to_idx, shadow_cache, bucket_memo,
        blackout_rate_by_type: dict, drift_p: float,
        rose_vals: dict, rose_total: float,
        debug_add_fn, total_cascade_work: int,
        cascade_progress: int, n_legs: int,
    ) -> tuple[float, float, float, int, bool]:
        ta = tg = tan = 0.0
        cp = cascade_progress
        for cell in ship_cells:
            freq = float(cell.get('freq', 0.0))
            speed_kts = float(cell.get('speed', 0.0))
            draught = float(cell.get('draught', 0.0))
            ship_type = int(cell.get('ship_type', -1))
            ship_size = int(cell.get('ship_size', -1))
            if speed_kts <= 0.0 or freq <= 0.0:
                continue
            hours_present = (line_length / (speed_kts * 1852.0)) * freq
            bph = blackout_rate_by_type.get(ship_type, drift_p) / (365.0 * 24.0)
            base = hours_present * bph
            cell_a = cell_g = 0.0
            for d_idx in range(8):
                angle = d_idx * 45
                rv = rose_vals.get(angle, 0.0)
                rp = (rv / rose_total) if rose_total > 0 else 0.0
                if rp <= 0.0:
                    continue
                a_d, g_d, an_d = self._process_cell_direction(
                    leg_idx=leg_idx, d_idx=d_idx, line=line, seg_id=seg_id,
                    cell=cell, base=base, rp=rp, freq=freq, draught=draught,
                    ship_type=ship_type, ship_size=ship_size, drift=drift,
                    drift_speed=drift_speed, anchor_p=anchor_p, anchor_d=anchor_d,
                    structures=structures, depths=depths,
                    struct_min_dists=struct_min_dists, depth_min_dists=depth_min_dists,
                    struct_probability_holes=struct_probability_holes,
                    depth_probability_holes=depth_probability_holes,
                    threshold_to_idx=threshold_to_idx, shadow_cache=shadow_cache,
                    bucket_memo=bucket_memo, debug_add=debug_add_fn, report=report,
                )
                ta += a_d; tg += g_d; tan += an_d
                cell_a += a_d; cell_g += g_d
                cp += 1
                if total_cascade_work > 0 and cp % max(1, total_cascade_work // 100) == 0:
                    if not self._report_progress('cascade', cp / total_cascade_work,
                            f"Drifting - traffic cascade (leg {leg_idx + 1}/{n_legs})"):
                        return ta, tg, tan, cp, True
            if ship_type >= 0 and ship_size >= 0:
                ck = f"{ship_type}_{ship_size}"
                if cell_a > 0.0:
                    report['by_cell_allision'][ck] = report['by_cell_allision'].get(ck, 0.0) + cell_a
                if cell_g > 0.0:
                    report['by_cell_grounding'][ck] = report['by_cell_grounding'].get(ck, 0.0) + cell_g
        return ta, tg, tan, cp, False

    def _iterate_traffic_and_sum(self,
            data: dict[str, Any],
            line_names: list[str],
            transformed_lines: list[LineString],
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            struct_probability_holes: list,
            depth_probability_holes: list,
            distributions: list[list[Any]] | None = None,
            weights: list[list[float]] | None = None,
            reach_distance: float = 0.0,
            threshold_to_idx: dict[float, int] | None = None,
            shadow_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
            bucket_memo: dict[tuple[int, int, tuple], list[dict[str, Any]]] | None = None,
        ) -> tuple[float, float, dict[str, Any]]:
        drift = data['drift']
        debug_trace = bool(drift.get('debug_trace', False))
        drift_p = float(drift.get('drift_p', 1.0))
        _raw_by_type = drift.get('blackout_by_ship_type') or {}
        blackout_rate_by_type: dict[int, float] = {}
        for k, v in _raw_by_type.items():
            try:
                blackout_rate_by_type[int(k)] = float(v)
            except Exception:
                continue
        anchor_p = float(drift.get('anchor_p', 0.7))
        anchor_d = float(drift.get('anchor_d', 7.0))
        drift_speed = float(drift.get('speed', 1.0)) * 1852.0 / 3600.0
        rose_vals = {int(k): float(v) for k, v in drift.get('rose', {}).items()}
        rose_total = sum(rose_vals.values())

        def _debug_add(rd, ldk, ok, ot, c, d, h, rb, p_nr=None, anchor_effect=None,
                       exposure_factor=None, rp=None, base=None, freq=None) -> None:
            if debug_trace:
                self._debug_add_trace(rd, ldk, ok, ot, c, d, h, rb, p_nr,
                    anchor_effect, exposure_factor, rp, base, freq)

        traffic_by_leg = [lt for _, _, _, lt, _ in clean_traffic(data)]
        report = self._init_drift_report(debug_trace)
        total_allision = total_grounding = total_anchoring = 0.0
        total_cascade_work = sum(
            len(traffic_by_leg[i]) * 8 if i < len(traffic_by_leg) else 0
            for i in range(len(transformed_lines))
        )
        cascade_progress = 0
        for leg_idx, line in enumerate(transformed_lines):
            try:
                nm = line_names[leg_idx]
                seg_id = nm.split('Leg ')[1].split('-')[0].strip()
            except Exception:
                seg_id = str(leg_idx)
            line_length = float(data.get('segment_data', {}).get(seg_id, {}).get('line_length', line.length))
            ship_cells = traffic_by_leg[leg_idx] if leg_idx < len(traffic_by_leg) else []
            a, g, an, cascade_progress, cancelled = self._process_leg_cells(
                leg_idx, line, seg_id, line_length, ship_cells, report,
                drift, drift_speed, anchor_p, anchor_d, structures, depths,
                struct_min_dists, depth_min_dists, struct_probability_holes,
                depth_probability_holes, threshold_to_idx, shadow_cache, bucket_memo,
                blackout_rate_by_type, drift_p, rose_vals, rose_total, _debug_add,
                total_cascade_work, cascade_progress, len(transformed_lines),
            )
            total_allision += a; total_grounding += g; total_anchoring += an
            if cancelled:
                report['totals']['allision'] = total_allision
                report['totals']['grounding'] = total_grounding
                report['totals']['anchoring'] = total_anchoring
                return total_allision, total_grounding, report
        report['totals']['allision'] = total_allision
        report['totals']['grounding'] = total_grounding
        report['totals']['anchoring'] = total_anchoring
        return total_allision, total_grounding, report

    def _collect_cell_obstacles(
        self,
        leg_idx: int, d_idx: int, draught: float, anchor_d: float,
        structures: list, depths: list,
        struct_min_dists, depth_min_dists,
        struct_probability_holes, depth_probability_holes,
        threshold_to_idx,
    ) -> list[tuple[str, int, float, float]]:
        math_dir_idx = _compass_idx_to_math_idx(d_idx)
        obstacles: list[tuple[str, int, float, float]] = []
        if struct_min_dists and struct_probability_holes:
            for s_idx in range(len(structures)):
                try:
                    dist = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                    hole_pct = struct_probability_holes[leg_idx][math_dir_idx][s_idx]
                    if dist is not None and hole_pct > 0.0:
                        obstacles.append(('allision', s_idx, dist, hole_pct))
                except (IndexError, TypeError):
                    pass
        anchor_threshold = anchor_d * draught if anchor_d > 0.0 else 0.0
        if depth_min_dists and depth_probability_holes and threshold_to_idx:
            grounding_idx = threshold_to_idx.get(round(draught, 2))
            if grounding_idx is not None:
                try:
                    dist = depth_min_dists[leg_idx][math_dir_idx][grounding_idx]
                    hole_pct = depth_probability_holes[leg_idx][math_dir_idx][grounding_idx]
                    if dist is not None and hole_pct > 0.0:
                        obstacles.append(('grounding', grounding_idx, dist, hole_pct))
                except (IndexError, TypeError):
                    pass
            if anchor_threshold > 0.0:
                anchoring_idx = threshold_to_idx.get(round(anchor_threshold, 2))
                if anchoring_idx is not None:
                    try:
                        dist = depth_min_dists[leg_idx][math_dir_idx][anchoring_idx]
                        hole_pct = depth_probability_holes[leg_idx][math_dir_idx][anchoring_idx]
                        if dist is not None and hole_pct > 0.0:
                            obstacles.append(('anchoring', anchoring_idx, dist, hole_pct))
                    except (IndexError, TypeError):
                        pass
        elif depth_min_dists and depth_probability_holes:
            for dep_idx, dep in enumerate(depths):
                try:
                    dist = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                    hole_pct = depth_probability_holes[leg_idx][math_dir_idx][dep_idx]
                    if dist is None or hole_pct <= 0.0:
                        continue
                    if anchor_threshold > 0.0 and dep['depth'] < anchor_threshold:
                        obstacles.append(('anchoring', dep_idx, dist, hole_pct))
                    if dep['depth'] < draught:
                        obstacles.append(('grounding', dep_idx, dist, hole_pct))
                except (IndexError, TypeError):
                    pass
        return obstacles

    def _lookup_bucket_entries(
        self,
        leg_idx: int, d_idx: int,
        obstacles: list[tuple[str, int, float, float]],
        bucket_memo,
    ) -> list[dict]:
        bucket_key = tuple(sorted((ot, oi) for ot, oi, _d, _h in obstacles))
        entries = bucket_memo.get((leg_idx, d_idx, bucket_key)) if bucket_memo else None
        if entries is None:
            entries = [
                {
                    'obs_type': ot, 'obs_idx': oi,
                    'dist': float(d_val), 'hole_pct': float(h_val),
                    'h_reach': float(h_val), 'h_in_anchor': 0.0,
                }
                for ot, oi, d_val, h_val in obstacles
            ]
        return entries

    def _apply_anchoring_entry(
        self, *,
        entry: dict, base: float, rp: float, anchor_p: float, h_eff: float,
        depths: list, seg_id: str, d_idx: int, dist: float,
        leg_dir_key: str, precomputed_edges: list,
        debug_add, report: dict, freq: float, line,
    ) -> float:
        obs_idx = entry['obs_idx']
        try:
            dep = depths[obs_idx]
            obs_key = f"Anchoring - {dep.get('id', str(obs_idx))}"
        except Exception:
            obs_key = f"Anchoring - {obs_idx}"
        contrib_total = base * rp * anchor_p * h_eff
        if precomputed_edges:
            for eg in precomputed_edges:
                edge_hole = h_eff * eg['len_frac']
                if edge_hole <= 0.0:
                    continue
                per_edge = base * rp * anchor_p * edge_hole
                self._update_anchoring_report(
                    report, per_edge, obs_idx, depths, seg_id,
                    d_idx, dist, edge_hole, None, line,
                )
                self._add_direct_segment_contrib(
                    report, 'by_anchoring_segment_legdir', obs_key,
                    eg['seg_idx'], leg_dir_key, per_edge,
                )
        else:
            self._update_anchoring_report(
                report, contrib_total, obs_idx, depths, seg_id,
                d_idx, dist, h_eff, None, line,
            )
        debug_add(
            report, leg_dir_key, obs_key, 'anchoring',
            contrib_total, dist, h_eff, 1.0,
            p_nr=None, anchor_effect=anchor_p,
            exposure_factor=base * rp, rp=rp, base=base, freq=freq,
        )
        return contrib_total

    def _apply_hit_entry(
        self, *,
        obs_type: str, obs_idx: int, dist: float,
        hole_pct: float, h_eff: float, h_reach: float,
        base: float, rp: float,
        structures: list, depths: list,
        seg_id: str, cell: dict, d_idx: int, leg_dir_key: str,
        precomputed_edges: list, drift: dict, drift_speed: float,
        freq: float, ship_type: int, ship_size: int,
        line, debug_add, report: dict,
    ) -> tuple[float, float]:
        if obs_type == 'allision':
            s = structures[obs_idx] if obs_idx < len(structures) else None
            key_name = f"Structure - {s.get('id', str(obs_idx))}" if s is not None else f"Structure - {obs_idx}"
            direct_key = 'by_structure_segment_legdir'
        else:
            dep = depths[obs_idx] if obs_idx < len(depths) else None
            key_name = f"Depth - {dep.get('id', str(obs_idx))}" if dep is not None else f"Depth - {obs_idx}"
            direct_key = 'by_depth_segment_legdir'
        allision_d = 0.0
        grounding_d = 0.0
        obs_total = 0.0
        shadow_loss = max(0.0, 1.0 - (h_reach / hole_pct)) if hole_pct > 0 else 0.0
        if not precomputed_edges:
            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
            c = base * rp * h_eff * p_nr
            allision_d += c if obs_type == 'allision' else 0.0
            grounding_d += c if obs_type != 'allision' else 0.0
            obs_total += c
            self._update_report(report, obs_type, c, obs_idx, structures, depths,
                seg_id, cell, d_idx, dist, base, rp, shadow_loss, p_nr, h_eff,
                freq, ship_type, ship_size, None, line)
        else:
            for eg in precomputed_edges:
                edge_hole = h_eff * eg['len_frac']
                if edge_hole <= 0.0:
                    continue
                p_nr = eg['edge_p_nr']
                c = base * rp * edge_hole * p_nr
                allision_d += c if obs_type == 'allision' else 0.0
                grounding_d += c if obs_type != 'allision' else 0.0
                obs_total += c
                self._update_report(report, obs_type, c, obs_idx, structures, depths,
                    seg_id, cell, d_idx, eg['edge_dist'], base, rp, shadow_loss,
                    p_nr, edge_hole, freq, ship_type, ship_size, None, line)
                self._add_direct_segment_contrib(
                    report, direct_key, key_name, eg['seg_idx'], leg_dir_key, c)
        debug_add(report, leg_dir_key, key_name, obs_type, obs_total, dist, h_eff, 1.0,
            p_nr=None, anchor_effect=None, exposure_factor=base * rp,
            rp=rp, base=base, freq=freq)
        return allision_d, grounding_d

    def _process_cell_direction(self,
            *,
            leg_idx: int,
            d_idx: int,
            line: LineString,
            seg_id: str,
            cell: dict[str, Any],
            base: float,
            rp: float,
            freq: float,
            draught: float,
            ship_type: int,
            ship_size: int,
            drift: dict[str, Any],
            drift_speed: float,
            anchor_p: float,
            anchor_d: float,
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            struct_probability_holes: list,
            depth_probability_holes: list,
            threshold_to_idx: dict[float, int] | None,
            shadow_cache: dict[tuple[int, int], dict[str, Any]] | None,
            bucket_memo: dict[tuple[int, int, tuple], list[dict[str, Any]]] | None,
            debug_add: Callable[..., None],
            report: dict[str, Any],
        ) -> tuple[float, float, float]:
        obstacles = self._collect_cell_obstacles(
            leg_idx, d_idx, draught, anchor_d, structures, depths,
            struct_min_dists, depth_min_dists,
            struct_probability_holes, depth_probability_holes,
            threshold_to_idx,
        )
        if not obstacles:
            return 0.0, 0.0, 0.0

        ld_entry = shadow_cache.get((leg_idx, d_idx)) if shadow_cache else None
        edge_geom_map = ld_entry['edge_geom'] if ld_entry else {}
        leg_dir_key = f"{seg_id}:{str(cell.get('direction', '')).strip()}:{d_idx*45}"
        entries = self._lookup_bucket_entries(leg_idx, d_idx, obstacles, bucket_memo)

        total_allision = 0.0
        total_grounding = 0.0
        total_anchoring = 0.0

        for entry in entries:
            obs_type = entry['obs_type']
            obs_idx = entry['obs_idx']
            dist = entry['dist']
            hole_pct = entry['hole_pct']
            h_reach = entry['h_reach']
            h_in_anchor = entry['h_in_anchor']
            if obs_type == 'anchoring':
                h_eff = max(0.0, h_reach)
            else:
                h_eff = max(0.0, h_reach - anchor_p * h_in_anchor)
            if h_eff <= 0.0:
                continue
            obs_geom_key = 'allision' if obs_type == 'allision' else 'depth'
            precomputed_edges = edge_geom_map.get((obs_geom_key, obs_idx), []) if edge_geom_map else []
            if obs_type == 'anchoring':
                total_anchoring += self._apply_anchoring_entry(
                    entry=entry, base=base, rp=rp, anchor_p=anchor_p, h_eff=h_eff,
                    depths=depths, seg_id=seg_id, d_idx=d_idx, dist=dist,
                    leg_dir_key=leg_dir_key, precomputed_edges=precomputed_edges,
                    debug_add=debug_add, report=report, freq=freq, line=line,
                )
            else:
                a_d, g_d = self._apply_hit_entry(
                    obs_type=obs_type, obs_idx=obs_idx, dist=dist,
                    hole_pct=hole_pct, h_eff=h_eff, h_reach=h_reach,
                    base=base, rp=rp, structures=structures, depths=depths,
                    seg_id=seg_id, cell=cell, d_idx=d_idx, leg_dir_key=leg_dir_key,
                    precomputed_edges=precomputed_edges, drift=drift, drift_speed=drift_speed,
                    freq=freq, ship_type=ship_type, ship_size=ship_size,
                    line=line, debug_add=debug_add, report=report,
                )
                total_allision += a_d
                total_grounding += g_d

        return total_allision, total_grounding, total_anchoring


    def _auto_generate_drifting_report(self, data: dict[str, Any]) -> str | None:
        """Auto-generate the drifting Markdown report to disk.

        Path resolution priority:
        - If ``LEReportPath`` points to a folder (the post-Quick-Start
          convention) skip — the combined report is written by the
          run-history path with the model name baked into the filename.
        - If ``LEReportPath`` points to a file, write to that path.
        - Otherwise, write to '<cwd>/drifting_report.md'.

        Returns the written content on success, else None.
        """
        try:
            ui_path = None
            try:
                if hasattr(self.p.main_widget, 'LEReportPath') and self.p.main_widget.LEReportPath is not None:
                    t = self.p.main_widget.LEReportPath.text()
                    if isinstance(t, str) and t.strip():
                        ui_path = t.strip()
            except Exception:
                ui_path = None

            if ui_path and Path(ui_path).is_dir():
                # The folder path is owned by the run-history /
                # _auto_save_run flow which writes the combined report.
                return None

            path = ui_path or str(Path(os.getcwd()) / 'drifting_report.md')
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return self.write_drifting_report_markdown(path, data)
        except Exception:
            # Silent failure: do not interrupt calculations/UI/tests
            return None

    def _merge_depths_by_threshold(
        self,
        data: dict[str, Any],
        depths: list[dict[str, Any]],
        drift: dict[str, Any],
    ) -> tuple[
        bool,
        list[gpd.GeoDataFrame],
        list[dict[str, Any]],
        dict[float, int],
    ]:
        """Merge depth polygons by unique depth value, when worthwhile.

        With many depth polygons but few unique depth VALUES, several
        thresholds (draughts) map to the same merged polygon -- e.g.
        depth values ``[0, 3, 6, 9, 12]`` mean any threshold in
        ``(6, 9]`` includes the same set of polygons (those with
        ``depth <= 6``).  Merging once and indexing by threshold keeps
        the cascade arithmetic.

        Returns ``(use_merged, merged_gdfs, merged_meta, threshold_to_idx)``.
        ``use_merged`` is ``False`` when the projection wouldn't reduce
        the work, in which case the other three are empty placeholders.
        """
        unique_depth_vals = (
            sorted(set(d['depth'] for d in depths)) if depths else []
        )
        use_merged = (
            len(depths) > len(unique_depth_vals) + 1
            and len(unique_depth_vals) > 0
        )
        merged_depths_gdfs: list[gpd.GeoDataFrame] = []
        merged_depths_meta: list[dict[str, Any]] = []
        threshold_to_idx: dict[float, int] = {}

        if not (use_merged and depths):
            return use_merged, merged_depths_gdfs, merged_depths_meta, threshold_to_idx

        # ``_build_transformed`` stashes a UTM->WGS84 transformer on
        # ``self`` so we can attach a WGS84 copy of the merged geometry.
        # Without it, ``create_result_layers`` falls back to the UTM
        # geometry on a WGS84 layer and the features land on the
        # equator off the coast of Africa.
        _to_wgs84 = getattr(self, '_segment_utm_to_wgs84', None)
        cumulative_geoms: list[tuple[float, int]] = []
        for boundary in unique_depth_vals:
            qualifying = [d['wkt'] for d in depths if d['depth'] <= boundary]
            if not qualifying:
                continue
            merged_geom = unary_union(qualifying)
            merged_geom_wgs84 = merged_geom
            if _to_wgs84 is not None:
                try:
                    merged_geom_wgs84 = _to_wgs84(merged_geom)
                except Exception:
                    merged_geom_wgs84 = merged_geom
            idx = len(merged_depths_gdfs)
            merged_depths_gdfs.append(
                gpd.GeoDataFrame(geometry=[merged_geom]),
            )
            merged_depths_meta.append({
                'id': f'merged_depth_le_{boundary}',
                'depth': boundary,
                'wkt': merged_geom,
                'wkt_wgs84': merged_geom_wgs84,
            })
            cumulative_geoms.append((boundary, idx))

        # Collect all thresholds from traffic draughts and anchor draughts.
        draughts: set[float] = set()
        for _, _, _, leg_traffic, _ in clean_traffic(data):
            for cell in leg_traffic:
                d = float(cell.get('draught', 0.0))
                if d > 0:
                    draughts.add(round(d, 2))
        anchor_d_val = float(drift.get('anchor_d', 0.0))
        all_thresholds: set[float] = set()
        for d in draughts:
            all_thresholds.add(d)
            if anchor_d_val > 0:
                all_thresholds.add(round(anchor_d_val * d, 2))

        # For threshold T, the merged polygon includes all depths
        # strictly less than T.  Pick the highest boundary < T.
        for threshold in all_thresholds:
            best_idx = None
            for boundary, idx in cumulative_geoms:
                if boundary < threshold:
                    best_idx = idx
            if best_idx is not None:
                threshold_to_idx[round(threshold, 2)] = best_idx

        return use_merged, merged_depths_gdfs, merged_depths_meta, threshold_to_idx

    def _emit_zero_drifting(self) -> tuple[float, float]:
        self.p.main_widget.LEPDriftAllision.setText(f"{0.0:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{0.0:.3e}")
        except Exception:
            pass
        self.drifting_allision_prob = 0.0
        self.drifting_grounding_prob = 0.0
        return 0.0, 0.0

    def _apply_drifting_risk_factors(
        self, report: dict, total_allision: float, total_grounding: float,
        allision_rf: float, grounding_rf: float,
    ) -> tuple[float, float]:
        prob_a = float(total_allision * allision_rf)
        prob_g = float(total_grounding * grounding_rf)
        if allision_rf != 1.0:
            for k, v in list(report.get('by_cell_allision', {}).items()):
                report['by_cell_allision'][k] = float(v) * allision_rf
        if grounding_rf != 1.0:
            for k, v in list(report.get('by_cell_grounding', {}).items()):
                report['by_cell_grounding'][k] = float(v) * grounding_rf
        return prob_a, prob_g

    def _store_depth_meta(self, data: dict, effective_depths_meta: list) -> None:
        self._last_depths = effective_depths_meta
        try:
            from shapely import wkt as _sw
            original: list[dict[str, Any]] = []
            for row in data.get('depths', []) or []:
                try:
                    did, depth_val, wkt_str = row
                except Exception:
                    continue
                try:
                    geom = _sw.loads(wkt_str) if isinstance(wkt_str, str) else wkt_str
                except Exception:
                    continue
                original.append({'id': str(did), 'depth': float(depth_val) if depth_val else 0.0,
                    'wkt': geom, 'wkt_wgs84': geom})
            self._last_depths_original = original
        except Exception:
            self._last_depths_original = []

    def _finalize_drifting(
        self, report: dict, structures: list, effective_depths_meta: list, data: dict,
    ) -> None:
        self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")
        except Exception:
            pass
        self._report_progress('layers', 0.0, "Drifting - generating report...")
        self._auto_generate_drifting_report(data)
        self._report_progress('layers', 0.3, "Drifting - creating result layers...")
        try:
            self.allision_result_layer, self.grounding_result_layer = create_result_layers(
                report, structures, effective_depths_meta, add_to_project=False
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create result layers: {e}")
        self._report_progress('layers', 1.0, "Drifting model complete")

    def run_drifting_model(self, data: dict[str, Any]) -> tuple[float, float]:
        """Compute drifting allision and grounding, and store a breakdown report."""
        if not data.get('traffic_data') or not data.get('segment_data'):
            return self._emit_zero_drifting()
        _, distributions, weights, line_names, structures, depths, structs_gdfs, depths_gdfs, transformed_lines = (
            self._build_transformed(data)
        )
        if len(structs_gdfs) == 0 and len(depths_gdfs) == 0:
            return self._emit_zero_drifting()
        longest = max(line.length for line in transformed_lines) if transformed_lines else 0.0
        reach_distance = self._compute_reach_distance(data, longest)
        drift = data['drift']
        use_merged, merged_depths_gdfs, merged_depths_meta, threshold_to_idx = (
            self._merge_depths_by_threshold(data, depths, drift)
        )
        effective_depths_gdfs = merged_depths_gdfs if use_merged else depths_gdfs
        effective_depths_meta = merged_depths_meta if use_merged else depths
        struct_min_dists, depth_min_dists, struct_probability_holes, depth_probability_holes = (
            self._precompute_spatial(transformed_lines, distributions, weights,
                structs_gdfs, effective_depths_gdfs, reach_distance, data)
        )
        shadow_cache = self._precompute_shadow_layer(
            transformed_lines, distributions, weights, structures, effective_depths_meta,
            struct_min_dists, depth_min_dists, reach_distance,
            drift.get('repair', {}), float(drift.get('speed', 0.0)) * 1852.0 / 3600.0,
            bool(drift.get('use_leg_offset_for_distance', False)),
            progress_base=0.0, progress_span=0.5,
        )
        if shadow_cache.get('__cancelled__'):
            self.drifting_report = {'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0}}
            return self._emit_zero_drifting()
        bucket_memo = self._precompute_bucket_memo(
            data, transformed_lines, structures, effective_depths_meta,
            struct_min_dists, depth_min_dists, struct_probability_holes, depth_probability_holes,
            shadow_cache, threshold_to_idx if use_merged else None, reach_distance,
            progress_base=0.5, progress_span=0.5,
        )
        if bucket_memo.get('__cancelled__'):
            self.drifting_report = {'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0}}
            return self._emit_zero_drifting()
        total_allision, total_grounding, report = self._iterate_traffic_and_sum(
            data, line_names, transformed_lines, structures, effective_depths_meta,
            struct_min_dists, depth_min_dists, struct_probability_holes, depth_probability_holes,
            distributions, weights, reach_distance,
            threshold_to_idx=threshold_to_idx if use_merged else None,
            shadow_cache=shadow_cache, bucket_memo=bucket_memo,
        )
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))
        self.drifting_allision_prob, self.drifting_grounding_prob = self._apply_drifting_risk_factors(
            report, total_allision, total_grounding, allision_rf, grounding_rf)
        self.drifting_report = report
        self._last_structures = structures
        self._store_depth_meta(data, effective_depths_meta)
        self._finalize_drifting(report, structures, effective_depths_meta, data)
        return self.drifting_allision_prob, self.drifting_grounding_prob
