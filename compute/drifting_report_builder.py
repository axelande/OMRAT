"""Drifting-model report builder mixin.

Extracts the per-object / per-segment / per-leg-direction bookkeeping
used by the drifting cascade.  These methods mutate a ``report`` dict in
place and are composed into :class:`compute.drifting_model.DriftingModelMixin`
via multiple inheritance.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from compute.drift_corridor_geometry import (
    _extract_obstacle_segments,
    _segment_intersects_corridor,
)


class DriftingReportBuilderMixin:
    """Mixin providing the report-assembly helpers for the drifting cascade.

    The host class is expected to optionally provide
    ``self._segment_utm_to_wgs84(LineString) -> LineString`` for recording
    WGS84 segment WKT in the runtime debug layer; if absent, the
    ``segment_wkt_wgs84`` field is left as ``None``.
    """

    def _add_direct_segment_contrib(
            self,
            report: dict[str, Any],
            report_key: str,
            obstacle_key: str,
            seg_idx: int | None,
            leg_dir_key: str,
            contrib: float,
        ) -> None:
            """Write segment contribution directly, bypassing equal split helper."""
            if seg_idx is None:
                return
            obs_seg_map = report.setdefault(report_key, {}).setdefault(obstacle_key, {})
            seg_key = f"seg_{seg_idx}"
            seg_data = obs_seg_map.setdefault(seg_key, {})
            seg_data[leg_dir_key] = seg_data.get(leg_dir_key, 0.0) + contrib

    def _update_report(self,
            report: dict[str, Any],
            event: str,
            contrib: float,
            idx: int,
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            seg_id: str,
            cell: dict[str, float],
            d_idx: int,
            dist: float,
            base: float,
            rp: float,
            anchor_factor: float,
            p_nr: float,
            ov_frac: float,
            freq: float,
            ship_type: int,
            ship_size: int,
            drift_corridor: Polygon | None = None,
            leg: LineString | None = None,
        ) -> None:
        """Update report dictionaries with contribution.

        Now also tracks per-segment contributions when drift_corridor is provided.
        """
        # Per-object accumulation.  This populates ``report['by_object']``
        # which is the input for the result-layer factory; without it the
        # drifting allision / grounding QGIS layers come up empty.
        try:
            if idx is not None:
                if event == 'allision':
                    o = structures[idx]
                    okey = f"Structure - {o.get('id', str(idx))}"
                elif event in ('grounding', 'anchoring'):
                    d = depths[idx]
                    okey = f"Depth - {d.get('id', str(idx))}"
                else:
                    okey = None
                if okey is not None:
                    ob = report.setdefault('by_object', {}).setdefault(
                        okey,
                        {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0},
                    )
                    ob[event] = ob.get(event, 0.0) + contrib
        except Exception:
            pass

        # Per leg-direction accumulation
        leg_dir_label = str(cell.get('direction', '')).strip()
        leg_dir_key = f"{seg_id}:{leg_dir_label}:{d_idx*45}"
        rec = report['by_leg_direction'].setdefault(leg_dir_key, {
            'base_hours': 0.0,
            'contrib_allision': 0.0,
            'contrib_grounding': 0.0,
            'ship_categories': {},
            'min_distance_allision': None,
            'min_distance_grounding': None,
            'anchor_factor_sum': 0.0,
            'not_repaired_sum': 0.0,
            'overlap_sum': 0.0,
            'weight_sum': 0.0,
        })
        rec['base_hours'] += base * rp
        if event == 'allision':
            rec['contrib_allision'] += contrib
            md = rec['min_distance_allision']
            rec['min_distance_allision'] = dist if md is None or dist < md else md
        else:
            rec['contrib_grounding'] += contrib
            md = rec['min_distance_grounding']
            rec['min_distance_grounding'] = dist if md is None or dist < md else md

        # Weighted diagnostics
        w = base * rp
        rec['anchor_factor_sum'] += anchor_factor * w
        rec['not_repaired_sum'] += p_nr * w
        rec['overlap_sum'] += ov_frac * w
        rec['weight_sum'] += w

        # Ship category accumulation
        cat_key = f"{ship_type}-{ship_size}"
        scat = rec['ship_categories'].setdefault(cat_key, {'allision': 0.0, 'grounding': 0.0, 'freq': 0.0})
        scat[event] += contrib
        scat['freq'] += freq

        # Per-structure per leg-direction accumulation (allision only)
        try:
            if event == 'allision' and idx is not None:
                s = structures[idx]
                skey = f"Structure - {s.get('id', str(idx))}"
                s_map = report['by_structure_legdir'].setdefault(skey, {})
                s_map[leg_dir_key] = s_map.get(leg_dir_key, 0.0) + contrib

                # Per-segment tracking: determine which segments of this structure
                # actually intersect with the drift corridor
                if drift_corridor is not None:
                    obs_geom = s.get('wkt')
                    if obs_geom is not None:
                        # Convert compass angle (d_idx*45) to math convention
                        compass_angle = d_idx * 45
                        math_drift_angle = (90 - compass_angle) % 360
                        self._update_segment_contributions(
                            report, 'by_structure_segment_legdir',
                            skey, leg_dir_key, contrib, obs_geom, drift_corridor,
                            math_drift_angle, leg
                        )
        except Exception:
            pass

        # Per-depth per leg-direction accumulation (grounding only)
        try:
            if event == 'grounding' and idx is not None:
                d = depths[idx]
                dkey = f"Depth - {d.get('id', str(idx))}"
                d_map = report['by_depth_legdir'].setdefault(dkey, {})
                d_map[leg_dir_key] = d_map.get(leg_dir_key, 0.0) + contrib

                # Per-segment tracking for depths
                if drift_corridor is not None:
                    obs_geom = d.get('wkt')
                    if obs_geom is not None:
                        # Convert compass angle (d_idx*45) to math convention
                        compass_angle = d_idx * 45
                        math_drift_angle = (90 - compass_angle) % 360
                        self._update_segment_contributions(
                            report, 'by_depth_segment_legdir',
                            dkey, leg_dir_key, contrib, obs_geom, drift_corridor,
                            math_drift_angle, leg
                        )
        except Exception:
            pass

    def _update_segment_contributions(
        self,
        report: dict[str, Any],
        report_key: str,
        obstacle_key: str,
        leg_dir_key: str,
        contrib: float,
        obs_geom: BaseGeometry,
        drift_corridor: Polygon,
        drift_angle: float | None = None,
        leg: LineString | None = None,
    ) -> None:
        """
        Track which segments of an obstacle are hit by a drift corridor.

        Distributes the contribution among segments that actually intersect
        with the drift corridor AND are in the drift direction from the leg.

        Args:
            report: The report dictionary to update
            report_key: 'by_structure_segment_legdir' or 'by_depth_segment_legdir'
            obstacle_key: Key for the obstacle (e.g., "Structure - id")
            leg_dir_key: Key for leg-direction (e.g., "1:North:0")
            contrib: Total contribution for this obstacle from this leg-direction
            obs_geom: Obstacle geometry (UTM)
            drift_corridor: Drift corridor polygon (UTM)
            drift_angle: Drift direction in degrees (math convention: 0=East, 90=North)
            leg: The traffic leg LineString for direction checking
        """
        try:
            # Extract obstacle segments
            segments = _extract_obstacle_segments(obs_geom)
            if not segments:
                return

            # Get leg centroid for direction checking
            leg_centroid = None
            if leg is not None:
                centroid = leg.centroid
                leg_centroid = (centroid.x, centroid.y)

            # Find which segments intersect with the corridor in the drift direction
            intersecting_indices: list[int] = []
            seg_intersection_len: dict[int, float] = {}
            for seg_idx, segment in enumerate(segments):
                if _segment_intersects_corridor(segment, drift_corridor, drift_angle, leg_centroid):
                    intersecting_indices.append(seg_idx)
                    try:
                        seg_line_tmp = LineString([segment[0], segment[1]])
                        seg_intersection_len[seg_idx] = float(seg_line_tmp.intersection(drift_corridor).length)
                    except Exception:
                        seg_intersection_len[seg_idx] = 0.0

            if not intersecting_indices:
                return

            # Distribute contribution by corridor-intersection length (fallback to equal split)
            total_inter_len = sum(max(seg_intersection_len.get(i, 0.0), 0.0) for i in intersecting_indices)

            # Initialize data structure if needed
            obs_seg_map = report.setdefault(report_key, {}).setdefault(obstacle_key, {})

            # Store contribution for each intersecting segment
            for seg_idx in intersecting_indices:
                if total_inter_len > 0.0:
                    w_seg = max(seg_intersection_len.get(seg_idx, 0.0), 0.0) / total_inter_len
                    contrib_seg = contrib * w_seg
                else:
                    contrib_seg = contrib / len(intersecting_indices)
                seg_key = f"seg_{seg_idx}"
                seg_data = obs_seg_map.setdefault(seg_key, {})
                seg_data[leg_dir_key] = seg_data.get(leg_dir_key, 0.0) + contrib_seg

                # Runtime segment debug metadata (exact segment geometry and hit metrics)
                try:
                    seg_line = LineString([segments[seg_idx][0], segments[seg_idx][1]])
                    inter_len = 0.0
                    try:
                        inter_len = float(seg_line.intersection(drift_corridor).length)
                    except Exception:
                        inter_len = 0.0

                    dist_to_leg = None
                    if leg is not None:
                        try:
                            dist_to_leg = float(seg_line.distance(leg))
                        except Exception:
                            dist_to_leg = None

                    runtime_map = report.setdefault('runtime_segment_hits', {}).setdefault(obstacle_key, {})
                    seg_meta = runtime_map.setdefault(seg_key, {
                        'segment_wkt_utm': seg_line.wkt,
                        'segment_wkt_wgs84': None,
                        'hits': {},
                    })

                    # Best-effort WGS84 conversion if converter is available
                    if seg_meta.get('segment_wkt_wgs84') is None:
                        conv = getattr(self, '_segment_utm_to_wgs84', None)
                        if conv is not None:
                            try:
                                seg_meta['segment_wkt_wgs84'] = conv(seg_line).wkt
                            except Exception:
                                pass

                    hit = seg_meta['hits'].setdefault(leg_dir_key, {
                        'count': 0,
                        'max_intersection_len_m': 0.0,
                        'min_distance_to_leg_m': None,
                        'contrib_sum': 0.0,
                    })
                    hit['count'] += 1
                    hit['contrib_sum'] += float(contrib_seg)
                    if inter_len > float(hit.get('max_intersection_len_m', 0.0)):
                        hit['max_intersection_len_m'] = inter_len
                    if dist_to_leg is not None:
                        cur = hit.get('min_distance_to_leg_m')
                        if cur is None or dist_to_leg < float(cur):
                            hit['min_distance_to_leg_m'] = dist_to_leg
                except Exception:
                    pass

        except Exception:
            pass

    def _update_anchoring_report(
        self,
        report: dict[str, Any],
        anchor_contrib: float,
        obs_idx: int,
        depths: list[dict[str, Any]],
        seg_id: str,
        d_idx: int,
        dist: float,
        hole_pct: float,
        drift_corridor: Polygon | None,
        leg: LineString,
    ) -> None:
        """
        Track anchoring contributions per depth and per segment.

        Anchoring is now tracked like grounding and allision - we record which
        segments of a depth obstacle would receive the anchoring shadow.

        Args:
            report: The report dictionary to update
            anchor_contrib: Anchoring contribution value
            obs_idx: Index of the depth obstacle
            depths: List of depth dictionaries
            seg_id: Segment (leg) ID
            d_idx: Direction index (compass convention: 0=N, 1=NE, ...)
            dist: Distance to obstacle
            hole_pct: Probability hole percentage
            drift_corridor: Drift corridor polygon (UTM)
            leg: The traffic leg LineString
        """
        try:
            # Direction names for reporting
            dir_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            dir_name = dir_names[d_idx % 8]
            compass_angle = d_idx * 45
            leg_dir_key = f"{seg_id}:{dir_name}:{compass_angle}"

            # Per-depth per leg-direction accumulation
            d = depths[obs_idx]
            dkey = f"Anchoring - {d.get('id', str(obs_idx))}"
            d_map = report['by_anchoring_legdir'].setdefault(dkey, {})
            d_map[leg_dir_key] = d_map.get(leg_dir_key, 0.0) + anchor_contrib

            # Per-segment tracking for anchoring
            if drift_corridor is not None:
                obs_geom = d.get('wkt')
                if obs_geom is not None:
                    # Convert compass angle to math convention
                    math_drift_angle = (90 - compass_angle) % 360
                    self._update_segment_contributions(
                        report, 'by_anchoring_segment_legdir',
                        dkey, leg_dir_key, anchor_contrib, obs_geom, drift_corridor,
                        math_drift_angle, leg
                    )
        except Exception:
            pass
