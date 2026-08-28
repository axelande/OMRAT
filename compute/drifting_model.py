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
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
from scipy import stats  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402
from shapely.geometry.base import BaseGeometry  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

try:
    from shapely import make_valid as shp_make_valid
except Exception:  # nosec B110 B112
    shp_make_valid = None

from compute.basic_equations import get_not_repaired  # noqa: E402
from compute.drift_corridor_geometry import (  # noqa: E402
    _compass_idx_to_math_idx,
    _extract_obstacle_segments,
    _create_drift_corridor,
    segment_corridor_overlap_length,
    compute_edge_reachable_widths_1d,
)
from compute.data_preparation import (  # noqa: E402
    clean_traffic,
    split_structures_and_depths,
    transform_to_utm,
    prepare_traffic_lists,
)
from geometries.get_drifting_overlap import (  # noqa: E402
    compute_min_distance_by_object,
    directional_distances_to_points,
)
from geometries.analytical_probability import (  # noqa: E402
    compute_probability_holes_analytical,
    compute_probability_analytical,
    _extract_polygon_rings,
)
from geometries.drift.shadow import create_obstacle_shadow, extract_polygons  # noqa: E402
from geometries.result_layers import create_result_layers  # noqa: E402
from compute.drifting_report_builder import DriftingReportBuilderMixin  # noqa: E402
from drifting.engine import LegState, compass_to_math_deg  # noqa: E402


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
            elif dist_type == 'normal':
                n_mean = float(rep.get('norm_mean', 0.0))
                n_std = float(rep.get('norm_std', 1.0))
                t99_h = float(stats.norm(loc=n_mean, scale=n_std).ppf(0.99))
            elif use_ln:
                s = float(rep.get('std', 0.0))
                loc = float(rep.get('loc', 0.0))
                scale = float(rep.get('scale', 1.0))
                t99_h = float(stats.lognorm(s, loc=loc, scale=scale).ppf(0.99))
            elif rep.get('func'):
                # User-defined repair CDF: find t99 numerically by bisection
                # on the compiled expression (t such that cdf(t) >= 0.99).
                from compute.basic_equations import _safe_compile, _safe_eval
                code = _safe_compile(str(rep['func']))
                lo, hi = 0.0, 200.0
                if _safe_eval(code, hi) >= 0.99:
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if _safe_eval(code, mid) >= 0.99:
                            hi = mid
                        else:
                            lo = mid
                    t99_h = hi

            if t99_h is not None and t99_h > 0:
                drift_speed_kts = float(data.get('drift', {}).get('speed', 0.0))
                drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s
                if drift_speed > 0:
                    reach_distance = drift_speed * 3600.0 * t99_h
                    reach_distance = min(reach_distance, longest_length * 10.0)
        except Exception:  # nosec B110 B112
            pass
        return reach_distance

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
        except Exception:  # nosec B110 B112
            polys = []
        if not polys:
            result: BaseGeometry = Polygon()
        else:
            shadows: list[BaseGeometry] = []
            for p in polys:
                try:
                    s = create_obstacle_shadow(p, compass_angle, corridor_bounds)
                except Exception:  # nosec B110 B112
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
                except Exception:  # nosec B110 B112
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
        except Exception:  # nosec B110 B112
            polys = []
        if not polys:
            return 0.0
        rings: list[np.ndarray] = []
        for p in polys:
            try:
                rings.extend(_extract_polygon_rings(p))
            except Exception:  # nosec B110 B112
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
        except Exception:  # nosec B110 B112
            return 0.0

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
            except Exception:  # nosec B110 B112
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
                        if not np.isfinite(weighted_std):
                            weighted_std = 0.0
                        lateral_spread = 5.0 * weighted_std
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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
        """Per-edge geometry for the drifting cascade using 1D shadow-carve.

        For each facing edge we compute:
        - ``edge_dist``: along-drift distance from the leg to the edge midpoint
          (using :func:`directional_distances_to_points`).
        - ``edge_p_nr``: probability the ship has not been repaired at that
          distance.
        - ``reachable_width``: the width of the edge's perpendicular-to-drift
          projection that is NOT already covered by a closer edge of the same
          polygon.  Ships in that reachable perpendicular range strike this
          edge as their first-contact hazard; ships in the shadowed range
          strike a closer edge first.  Sum of ``reachable_width`` across
          edges = polygon's perpendicular-to-drift projection (its geometric
          E-W span for southward drift).  This gives the "a ship grounds
          once" physics without per-obstacle shadow polygons.
        - ``edge_h_eff = reachable_width / leg_drift_width``: this edge's
          share of ships on the leg (uniform-along-leg assumption).

        The polygon-level shadow reduction from ``h_eff / hole_pct`` is
        applied later in :meth:`_apply_hit_entry`.
        """
        if poly is None or poly.is_empty:
            return []
        try:
            segments = _extract_obstacle_segments(poly)
            if not segments or leg_state is None:
                return []
            # Effective reference line for distance measurement.  Default is the
            # leg centerline; if ``use_leg_offset`` is set the caller wants the
            # distance measured from the leg shifted by the mean lateral offset
            # (traffic distribution mean).  The lateral std is ignored -- we
            # only shift by the mean.  This affects ``edge_dist`` (and hence
            # ``edge_p_nr``); it does NOT change edge selection or the 1D
            # shadow-carve, which are geometric relative to the leg centerline.
            effective_line = line
            if use_leg_offset:
                offset_m = float(getattr(leg_state, 'mean_offset_m', 0.0) or 0.0)
                if abs(offset_m) > 1e-9:
                    from drifting.engine import _offset_line_perpendicular
                    effective_line = _offset_line_perpendicular(line, offset_m)
            # Perpendicular-to-drift leg width -- denominator for edge_h_eff.
            # Also compute the leg's perp-drift interval so we can clip each
            # facing edge to it: under the IWRAP-uniform-ships model, ships
            # exist only within the leg's own perp-drift footprint, so any
            # polygon overhang beyond the leg contributes zero.
            rad = np.radians(math_angle)
            perp_drift = np.array([-np.sin(rad), np.cos(rad)])
            leg_coords = np.array(line.coords)
            leg_vec = leg_coords[-1] - leg_coords[0]
            leg_drift_width = abs(float(np.dot(leg_vec, perp_drift)))
            if leg_drift_width <= 0.0:
                return []
            leg_perps = np.dot(leg_coords, perp_drift)
            leg_perp_lo = float(leg_perps.min())
            leg_perp_hi = float(leg_perps.max())
            leg_centroid_pt = line.centroid
            leg_centroid = (leg_centroid_pt.x, leg_centroid_pt.y)
            # First pass: filter facing edges + keep along-drift distance so
            # the 1D shadow-carve can sort them closest-first.
            drift_rad = np.radians(math_angle)
            drift_ux = float(np.cos(drift_rad))
            drift_uy = float(np.sin(drift_rad))

            def _facing_ahead(seg) -> bool:
                """Direction-only pre-filter, mirroring the checks inside
                :func:`segment_corridor_overlap_length` but without the
                shapely corridor-intersection test."""
                p1s, p2s = seg
                dx = p2s[0] - p1s[0]
                dy = p2s[1] - p1s[1]
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq <= 0.0:
                    return False
                inv_len = seg_len_sq ** -0.5
                # Outward normal for CCW polygon = (dy, -dx) / len
                drift_into = drift_ux * dy * inv_len - drift_uy * dx * inv_len
                if abs(drift_into) < 0.17 or drift_into > 0:
                    return False
                mx_ = 0.5 * (p1s[0] + p2s[0])
                my_ = 0.5 * (p1s[1] + p2s[1])
                vx = mx_ - leg_centroid[0]
                vy = my_ - leg_centroid[1]
                d2 = vx * vx + vy * vy
                if d2 > 0.0:
                    ahead = vx * drift_ux + vy * drift_uy
                    if ahead < -0.5 * d2 ** 0.5:
                        return False
                return True

            def _collect(use_corridor_gate: bool) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                for seg_idx, seg in enumerate(segments):
                    if use_corridor_gate:
                        if segment_corridor_overlap_length(
                                seg, drift_corridor, math_angle, leg_centroid) <= 0.0:
                            continue
                    elif not _facing_ahead(seg):
                        continue
                    p1 = seg[0]
                    p2 = seg[1]
                    mx = 0.5 * (p1[0] + p2[0])
                    my = 0.5 * (p1[1] + p2[1])
                    # Along-drift distance from leg centroid to segment midpoint
                    # (positive = downstream of leg in drift direction).
                    dist_along = (mx - leg_centroid[0]) * drift_ux + (my - leg_centroid[1]) * drift_uy
                    out.append({
                        'seg_idx': seg_idx,
                        'p1': p1, 'p2': p2,
                        'dist': dist_along,
                    })
                return out

            candidates: list[dict[str, Any]] = []
            if drift_corridor is not None:
                candidates = _collect(use_corridor_gate=True)
            if not candidates:
                # FALLBACK: the drift corridor missed every facing edge (short
                # reach, odd drift angle, or corridor construction failure)
                # even though the obstacle has a nonzero analytical hole.
                # Without edges the cascade would fall back to the legacy
                # simple branch -- whole-polygon h_eff times p_nr at the
                # MINIMUM VERTEX distance -- which grossly overstates the
                # probability when the facing edges are much farther away
                # than the nearest corner.  Build the edges with the
                # direction-only filter instead; the leg-perp clip and the
                # 1D shadow-carve still bound the geometry, and per-edge
                # p_nr(edge_dist) handles distance correctly.
                candidates = _collect(use_corridor_gate=False)
            if not candidates:
                return []
            carved = compute_edge_reachable_widths_1d(
                candidates, math_angle,
                leg_perp_lo=leg_perp_lo, leg_perp_hi=leg_perp_hi,
            )
            positive = [c for c in carved if c.get('reachable_width', 0.0) > 0.0]
            if not positive:
                return []
            # Total reachable width across facing edges of this polygon.  Used
            # to normalise per-edge shares so the polygon total remains equal
            # to the analytical ``h_eff`` (which correctly accounts for the
            # AIS lateral ship distribution).  Without this normalisation, a
            # polygon whose ship density is far from the leg's uniform
            # assumption over-contributes -- see the ESCOW test10 inflation.
            total_reach = float(sum(float(c['reachable_width']) for c in positive))
            if total_reach <= 0.0:
                return []
            # Vectorised along-drift edge distances for p_nr; the value used
            # here (mid-endpoint average) matches the pre-existing formula.
            endpoints = np.empty((2 * len(positive), 2), dtype=float)
            for i, eg in enumerate(positive):
                endpoints[2 * i, 0] = eg['p1'][0]
                endpoints[2 * i, 1] = eg['p1'][1]
                endpoints[2 * i + 1, 0] = eg['p2'][0]
                endpoints[2 * i + 1, 1] = eg['p2'][1]
            # Distance is measured to ``effective_line`` (leg centerline by
            # default, offset by the traffic-distribution mean if the user
            # opted in via ``use_leg_offset``).
            dists = directional_distances_to_points(
                endpoints, effective_line, compass_angle,
                use_leg_offset=False,
            )
            items: list[dict[str, Any]] = []
            for i, eg in enumerate(positive):
                valid = [float(d) for d in [dists[2 * i], dists[2 * i + 1]] if np.isfinite(d)]
                if not valid:
                    continue
                edge_dist = sum(valid) / len(valid)
                reachable_width = float(eg['reachable_width'])
                # ``edge_h_eff`` is the pure-geometric per-edge share
                # (uniform-ship assumption) -- kept for backtracing.  The
                # ``len_frac`` is what actually multiplies ``h_eff`` in the
                # cascade so the polygon total matches the analytical value.
                edge_h_eff = reachable_width / leg_drift_width
                len_frac = reachable_width / total_reach
                items.append({
                    'seg_idx': eg['seg_idx'],
                    'edge_h_eff': edge_h_eff,
                    'len_frac': len_frac,
                    'reachable_width_m': reachable_width,
                    # Perp-drift intervals used for INTER-OBSTACLE shadow-carve
                    # at bucket time (a closer obstacle's occupied perp
                    # intervals subtract from this edge's reach).  Empty list
                    # if all intervals were absorbed by intra-polygon carving.
                    'reachable_intervals': list(eg.get('reachable_intervals', [])),
                    'edge_dist': edge_dist,
                    'edge_p_nr': get_not_repaired(drift_repair, drift_speed, edge_dist),
                })
            return items
        except Exception:  # nosec B110 B112
            return []

    @staticmethod
    def _merge_perp_intervals(
        cov: list[tuple[float, float]],
        new_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Merge ``new_intervals`` into ``cov`` and return a sorted, merged list."""
        if not new_intervals:
            return list(cov)
        all_iv = sorted(list(cov) + list(new_intervals))
        merged: list[tuple[float, float]] = [tuple(all_iv[0])]
        for lo, hi in all_iv[1:]:
            m_lo, m_hi = merged[-1]
            if lo <= m_hi:
                merged[-1] = (m_lo, max(m_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    def _recarve_edges_inter_obstacle(
        self,
        edges: list[dict[str, Any]],
        covered_perp: list[tuple[float, float]],
    ) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
        """Extend the "ship grounds once" rule to run ACROSS obstacles.

        Each edge's perp-drift intervals (``reachable_intervals``) are
        subtracted against the closer obstacles' occupied perp intervals
        (``covered_perp``).  Edges whose reach collapses to zero are dropped
        from the returned edge list.  Surviving edges get updated
        ``reachable_width_m``, ``edge_h_eff`` and ``len_frac``.  Also returns
        the union of the OBSTACLE's own perp intervals (before this carve --
        i.e. what this obstacle claims for downstream carving of farther
        obstacles).
        """
        if not edges:
            return [], []
        # Collect the obstacle's original perp intervals (union of ALL edges'
        # reachable_intervals) so a farther obstacle can carve against this
        # obstacle's full footprint, not just the surviving portions.
        obs_intervals: list[tuple[float, float]] = []
        for e in edges:
            for lo, hi in e.get('reachable_intervals', []) or []:
                if hi > lo:
                    obs_intervals.append((float(lo), float(hi)))
        obs_intervals_merged = self._merge_perp_intervals([], obs_intervals)
        if not covered_perp:
            # No inter-obstacle shadow yet -- keep edges as-is.  We still
            # need to report our own footprint back so the next obstacle
            # can carve against it.
            return list(edges), obs_intervals_merged
        # Re-carve each edge's intervals against ``covered_perp``.
        survivors: list[dict[str, Any]] = []
        total_reach_new = 0.0
        for e in edges:
            new_intervals: list[tuple[float, float]] = []
            for lo, hi in e.get('reachable_intervals', []) or []:
                remaining = [(lo, hi)]
                for c_lo, c_hi in covered_perp:
                    next_remaining = []
                    for r_lo, r_hi in remaining:
                        if c_hi <= r_lo or c_lo >= r_hi:
                            next_remaining.append((r_lo, r_hi))
                        elif c_lo <= r_lo and c_hi >= r_hi:
                            pass
                        elif c_lo <= r_lo:
                            next_remaining.append((c_hi, r_hi))
                        elif c_hi >= r_hi:
                            next_remaining.append((r_lo, c_lo))
                        else:
                            next_remaining.append((r_lo, c_lo))
                            next_remaining.append((c_hi, r_hi))
                    remaining = next_remaining
                for r_lo, r_hi in remaining:
                    if r_hi > r_lo:
                        new_intervals.append((r_lo, r_hi))
            new_reach = sum(hi - lo for lo, hi in new_intervals)
            if new_reach <= 0.0:
                continue
            # Preserve original leg_drift_width via edge_h_eff / reach ratio
            old_reach = float(e.get('reachable_width_m', 0.0))
            if old_reach > 0.0:
                leg_dw = old_reach / float(e['edge_h_eff']) if e.get('edge_h_eff', 0.0) > 0 else 0.0
            else:
                leg_dw = 0.0
            new_e = dict(e)
            new_e['reachable_width_m'] = new_reach
            new_e['reachable_intervals'] = new_intervals
            new_e['edge_h_eff'] = (new_reach / leg_dw) if leg_dw > 0.0 else float(e['edge_h_eff'])
            survivors.append(new_e)
            total_reach_new += new_reach
        # Renormalise len_frac so the polygon's total share still sums to 1.
        if total_reach_new > 0.0:
            for e in survivors:
                e['len_frac'] = float(e['reachable_width_m']) / total_reach_new
        return survivors, obs_intervals_merged

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
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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
        # Always build a corridor so per-edge p_nr weighting applies even when
        # no AIS lateral distributions are available (lateral_spread==0).
        # A 2500 m fallback half-width gives a corridor that fully encloses any
        # obstacle that is within the leg's own projected width.
        effective_spread = lateral_spread if lateral_spread > 0.0 else 2500.0
        if reach_distance > 0:
            try:
                drift_corridor = _create_drift_corridor(line, math_angle, reach_distance, effective_spread)
            except Exception:  # nosec B110 B112
                drift_corridor = None
        if drift_corridor is not None and not drift_corridor.is_empty:
            return drift_corridor, drift_corridor.bounds
        try:
            xs = [line.bounds[0], line.bounds[2]]
            ys = [line.bounds[1], line.bounds[3]]
            for s in structures:
                g = s.get('wkt')
                if g is not None and not g.is_empty:
                    xs.extend([g.bounds[0], g.bounds[2]])
                    ys.extend([g.bounds[1], g.bounds[3]])
            for dep in depths:
                g = dep.get('wkt')
                if g is not None and not g.is_empty:
                    xs.extend([g.bounds[0], g.bounds[2]])
                    ys.extend([g.bounds[1], g.bounds[3]])
            pad = max(1000.0, (max(xs) - min(xs)) * 0.1)
            return drift_corridor, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        except Exception:  # nosec B110 B112
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
                    except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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
        # Union of ACTUAL anchor polygon geometries (not their swept shadows).
        # Used to detect the common physical case where a grounding polygon
        # is nested inside its own anchor polygon (a fact of bathymetry -- a
        # depth <= D contour always sits inside a depth <= anchor_d * D
        # contour).  In that case every ship reaching the grounding polygon
        # must first drift through the anchor zone, so h_in_anchor should
        # equal h_reach and anchor_p * h_in_anchor should reduce grounding.
        anchor_polygon_union: BaseGeometry | None = None
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
                except Exception:  # nosec B110 B112
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
            if (obs_type != 'anchoring' and not reach.is_empty
                    and _have_integrator):
                # Union of the anchor's swept-shadow with the anchor's actual
                # polygon.  The swept shadow captures ships that pass DOWNSTREAM
                # of the anchor polygon; the polygon geometry itself captures
                # the far more common case where a grounding polygon is
                # nested INSIDE the anchor polygon (deeper contour inside
                # shallower contour).  Without the polygon-geometry term, the
                # nested case gets h_in_anchor = 0 and anchor never reduces
                # grounding for those ships -- a systematic under-application
                # of the anchor probability that inflates grounding by ~3-5x.
                anchor_capture: BaseGeometry | None = None
                if anchor_union is not None and not anchor_union.is_empty:
                    anchor_capture = anchor_union
                if anchor_polygon_union is not None and not anchor_polygon_union.is_empty:
                    if anchor_capture is None:
                        anchor_capture = anchor_polygon_union
                    else:
                        try:
                            anchor_capture = unary_union([anchor_capture, anchor_polygon_union])
                        except Exception:  # nosec B110 B112
                            pass
                if (anchor_capture is not None
                        and not anchor_capture.is_empty
                        and anchor_capture.intersects(reach)):
                    try:
                        _in = reach.intersection(anchor_capture)
                        if _in is not None and not _in.is_empty:
                            h_in_anchor = self._analytical_hole_for_geom(
                                _in, transformed_lines[leg_idx], compass_angle,
                                dists_dir, w_dir, reach_distance, lateral_spread)
                    except Exception:  # nosec B110 B112
                        h_in_anchor = 0.0
            entries.append({'obs_type': obs_type, 'obs_idx': obs_idx,
                            'dist': dist, 'hole_pct': hole_pct, 'h_reach': h_reach, 'h_in_anchor': h_in_anchor})
            # Anchoring obstacles reference depth polygons; map 'anchoring' -> 'depth'
            # so the shadow lookup populates anchor_union correctly.
            lookup_type = 'depth' if obs_type == 'anchoring' else obs_type
            _s = shadows_map.get((lookup_type, obs_idx))
            if obs_type in ('allision', 'grounding'):
                if _s is not None and not _s.is_empty:
                    blocker_union = _s if blocker_union is None else unary_union([blocker_union, _s])
            elif obs_type == 'anchoring':
                if _s is not None and not _s.is_empty:
                    anchor_union = _s if anchor_union is None else unary_union([anchor_union, _s])
                # Also accumulate the anchor polygon geometry itself (in
                # addition to its downstream shadow) so nested grounding
                # polygons trigger the anchor reduction.
                if geom_X is not None and not geom_X.is_empty:
                    if anchor_polygon_union is None:
                        anchor_polygon_union = geom_X
                    else:
                        try:
                            anchor_polygon_union = unary_union([anchor_polygon_union, geom_X])
                        except Exception:  # nosec B110 B112
                            pass
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
                    except Exception:  # nosec B110 B112
                        pass
                    completed += 1
                    if completed % max(1, total_units // 50) == 0 or completed == total_units:
                        if not _report(f"Drifting - bucket memo ({completed}/{total_units})"):
                            cancelled = True
                            for f in futures:
                                f.cancel()
                            break
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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

        # Probability holes always use the analytical cross-section CDF
        # integration.  The Monte-Carlo estimator in
        # geometries/calculate_probability_holes.py is retained ONLY as an
        # independent cross-check for tests/examples -- it is not selectable
        # from the UI or the .omrat data (the old 'use_analytical' flag was
        # never exposed anywhere and defaulted to True).
        compute_holes_fn = compute_probability_holes_analytical
        logger.info("Probability holes: using analytical cross-section CDF method")

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
            rec['p_nr_sum'] += float(p_nr) * w
            rec['p_nr_weight'] += w
        if anchor_effect is not None:
            rec['anchor_effect_sum'] += float(anchor_effect) * w
            rec['anchor_effect_weight'] += w
        if exposure_factor is not None:
            rec['exposure_sum'] += float(exposure_factor) * w
            rec['exposure_weight'] += w
        if rp is not None and rec['rp'] == 0.0:
            rec['rp'] = float(rp)
        if base is not None:
            rec['base_sum'] += float(base) * w
            rec['base_weight'] += w
        if freq is not None:
            rec['freq_sum'] += float(freq) * w
            rec['freq_weight'] += w
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
                ta += a_d
                tg += g_d
                tan += an_d
                cell_a += a_d
                cell_g += g_d
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
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
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
            total_allision += a
            total_grounding += g
            total_anchoring += an
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
                # The threshold_to_idx keys were built in
                # _merge_depths_by_threshold from the ROUNDED draughts:
                # round(anchor_d * round(d, 2), 2).  Look up the same way --
                # round(anchor_d * d, 2) differs in the 2nd decimal for
                # full-precision AIS draughts (e.g. 7 x 5.87714 -> 41.14 vs
                # 7 x 5.88 -> 41.16), which silently dropped the anchoring
                # obstacle for most AIS-derived ship cells.
                anchoring_idx = threshold_to_idx.get(
                    round(anchor_d * round(draught, 2), 2))
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
        entry: dict, base: float, rp: float, anchor_p: float,
        hole_pct: float, h_eff: float,
        depths: list, seg_id: str, d_idx: int, dist: float,
        leg_dir_key: str, precomputed_edges: list,
        debug_add, report: dict, freq: float, line,
    ) -> float:
        obs_idx = entry['obs_idx']
        try:
            dep = depths[obs_idx]
            obs_key = f"Anchoring - {dep.get('id', str(obs_idx))}"
        except Exception:  # nosec B110 B112
            obs_key = f"Anchoring - {obs_idx}"
        if precomputed_edges:
            # Anchoring uses the ANALYTICAL h_eff (AIS-weighted), not the
            # IWRAP-uniform geometric edge_h_eff that grounding/allision use.
            # Rationale:
            #   * Anchoring doesn't include ``edge_p_nr``, so it has no need
            #     for per-edge distance weighting (unlike grounding).
            #   * Anchor polygons for typical draughts cover most of the
            #     seabed shallower than ~40--50 m; under uniform-along-leg
            #     the geometric baseline says nearly every ship enters the
            #     anchor zone, which grossly overestimates the number of
            #     ships that would actually anchor -- ships DO follow the
            #     AIS distribution, and the analytical h_anchor already
            #     captures how many of them realistically drift into that
            #     zone.
            # We still report per edge for traceability, distributing the
            # analytical h_eff across the polygon's edges by the same
            # 1D-shadow-carved reach share (``len_frac``) that grounding
            # uses for its geometry.  Sum of edge contributions equals
            # the obstacle-level ``base * rp * anchor_p * h_eff``.
            contrib_total = 0.0
            for eg in precomputed_edges:
                edge_hole = float(h_eff) * float(eg['len_frac'])
                if edge_hole <= 0.0:
                    continue
                per_edge = base * rp * anchor_p * edge_hole
                contrib_total += per_edge
                self._update_anchoring_report(
                    report, per_edge, obs_idx, depths, seg_id,
                    d_idx, dist, edge_hole, None, line,
                )
                self._add_direct_segment_contrib(
                    report, 'by_anchoring_segment_legdir', obs_key,
                    eg['seg_idx'], leg_dir_key, per_edge,
                )
        else:
            contrib_total = base * rp * anchor_p * h_eff
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
            # Per-edge branch -- IWRAP-style uniform ships along the reference
            # line (leg centerline by default, leg+mean_offset if the user
            # toggles ``use_leg_offset_for_distance``).  The lateral spread
            # (std) is NOT used in the drift model here; only the geometric
            # ``reachable_width`` matters.  The attenuation ratio combines
            # (a) between-polygon geometric shadow (h_reach / hole_pct) and
            # (b) anchor survival: ships that would successfully anchor in
            # the anchor zone on their way to this polygon never ground, so
            # only the fraction  (h_reach - anchor_p*h_in_anchor) / h_reach
            # continues.  Both together equal  h_eff / hole_pct  because
            # h_eff = h_reach - anchor_p * h_in_anchor.
            if hole_pct > 0.0:
                shadow_ratio = float(h_eff) / float(hole_pct)
            else:
                shadow_ratio = 1.0
            shadow_ratio = max(0.0, min(1.0, shadow_ratio))
            for eg in precomputed_edges:
                # Pure geometric per-edge share (uniform ships), attenuated
                # only by between-polygon shadow.
                edge_hole = float(eg['edge_h_eff']) * shadow_ratio
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
                self._record_edge_meta(
                    report=report, obs_key=key_name, seg_idx=eg['seg_idx'],
                    leg_dir_key=leg_dir_key, obs_type=obs_type,
                    hole_pct=hole_pct, h_eff=h_eff,
                    reachable_width=eg.get('reachable_width_m', 0.0),
                    edge_h_eff=eg['edge_h_eff'], len_frac=eg['len_frac'],
                    edge_dist=eg['edge_dist'], edge_p_nr=p_nr, edge_hole=edge_hole,
                    base=base, rp=rp, contrib=c,
                )
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
        leg_dir_key = f"{seg_id}:{str(cell.get('direction', '')).strip()}:{d_idx * 45}"
        entries = self._lookup_bucket_entries(leg_idx, d_idx, obstacles, bucket_memo)

        total_allision = 0.0
        total_grounding = 0.0
        total_anchoring = 0.0

        # Inter-obstacle along-drift shadow: walk entries closest-first
        # (already the sort order in ``entries``) and accumulate a running
        # list of covered perp-drift intervals.  Each farther obstacle's
        # edges are re-carved against the intervals claimed by closer
        # obstacles, so a ship that grounds on a near depth polygon
        # doesn't also allide on a structure behind it (or vice versa).
        covered_perp: list[tuple[float, float]] = []
        # Compute polygon-level ``total_reach`` for each obstacle from its
        # ORIGINAL edges once, so we can renormalise ``len_frac`` when
        # inter-obstacle carving trims individual edge reaches.
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
            # Re-carve edges against ``covered_perp`` before we consume them.
            # ``precomputed_edges`` are per-obstacle only (intra-polygon
            # 1D shadow-carve); this pass extends the "ship grounds once"
            # rule to run ACROSS obstacles in the ship's obstacle list.
            precomputed_edges, obs_perp_intervals = self._recarve_edges_inter_obstacle(
                precomputed_edges, covered_perp,
            )
            if precomputed_edges is None:
                # Empty list is fine (fall back to obstacle-level simple branch);
                # None means the helper decided to skip this obstacle entirely.
                precomputed_edges = []
            if obs_type == 'anchoring':
                # Anchoring is PROBABILISTIC (a ship in the anchor zone
                # anchors with probability anchor_p), not a solid wall: it
                # must NOT claim covered_perp, or the huge anchor polygon
                # (which usually encloses the whole corridor) would zero
                # every grounding/allision edge behind it.  The probabilistic
                # reduction of grounding is already handled separately via
                # ``h_eff = h_reach - anchor_p * h_in_anchor``.
                obs_perp_intervals = []
            # Update the cross-obstacle covered list with THIS obstacle's
            # occupied perp intervals for the next iteration.
            if obs_perp_intervals:
                covered_perp = self._merge_perp_intervals(covered_perp, obs_perp_intervals)
            if obs_type == 'anchoring':
                total_anchoring += self._apply_anchoring_entry(
                    entry=entry, base=base, rp=rp, anchor_p=anchor_p,
                    hole_pct=hole_pct, h_eff=h_eff,
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
            except Exception:  # nosec B110 B112
                ui_path = None

            if ui_path and Path(ui_path).is_dir():
                # The folder path is owned by the run-history /
                # _auto_save_run flow which writes the combined report.
                return None

            path = ui_path or str(Path(os.getcwd()) / 'drifting_report.md')
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return self.write_drifting_report_markdown(path, data)
        except Exception:  # nosec B110 B112
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
                except Exception:  # nosec B110 B112
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
        except Exception:  # nosec B110 B112
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
                except Exception:  # nosec B110 B112
                    continue
                try:
                    geom = _sw.loads(wkt_str) if isinstance(wkt_str, str) else wkt_str
                except Exception:  # nosec B110 B112
                    continue
                depth_f = float(depth_val) if depth_val else 0.0
                if geom.geom_type == 'MultiPolygon':
                    for i, poly in enumerate(geom.geoms):
                        original.append({'id': f"{did}_{i}", 'depth': depth_f,
                                         'wkt': poly, 'wkt_wgs84': poly})
                else:
                    original.append({'id': str(did), 'depth': depth_f,
                                     'wkt': geom, 'wkt_wgs84': geom})
            self._last_depths_original = original
        except Exception:  # nosec B110 B112
            self._last_depths_original = []

    def _finalize_drifting(
        self, report: dict, structures: list, effective_depths_meta: list, data: dict,
    ) -> None:
        self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")
        except Exception:  # nosec B110 B112
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
