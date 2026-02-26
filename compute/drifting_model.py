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
import numpy as np
from numpy import exp, log
from pathlib import Path

import geopandas as gpd
from scipy import stats
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

try:
    from shapely import make_valid as shp_make_valid
except Exception:
    shp_make_valid = None

from compute.basic_equations import (
    get_drifting_prob,
    get_Fcoll,
    powered_na,
    get_not_repaired,
)
from compute.drift_corridor_geometry import (
    _compass_idx_to_math_idx,
    _extract_obstacle_segments,
    _create_drift_corridor,
    _segment_intersects_corridor,
)
from compute.data_preparation import (
    clean_traffic,
    split_structures_and_depths,
    transform_to_utm,
    prepare_traffic_lists,
    get_distribution,
)
from geometries.route import get_multiple_ed, get_multi_drift_distance
from geometries.get_drifting_overlap import (
    compute_min_distance_by_object,
    compute_leg_overlap_fraction,
    compute_dir_overlap_fraction_by_object,
    compute_dir_leg_overlap_fraction_by_object,
)
from geometries.calculate_probability_holes import compute_probability_holes
from geometries.result_layers import create_result_layers


class DriftingModelMixin:
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
                if use_ln:
                    s = float(rep.get('std', 0.0))
                    loc = float(rep.get('loc', 0.0))
                    scale = float(rep.get('scale', 1.0))
                    t99_h = float(stats.lognorm(s, loc=loc, scale=scale).ppf(0.99))
                    drift_speed_kts = float(data.get('drift', {}).get('speed', 0.0))
                    drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s
                    if t99_h > 0 and drift_speed > 0:
                        reach_distance = drift_speed * 3600.0 * t99_h
                        reach_distance = min(reach_distance, longest_length * 10.0)
            except Exception:
                pass
            return reach_distance

    def _build_transformed(self, data: dict[str, Any]) -> tuple[
            list[LineString], list[list[Any]], list[list[float]], list[str],
            list[dict[str, Any]], list[dict[str, Any]],
            list[gpd.GeoDataFrame], list[gpd.GeoDataFrame],
            list[LineString]
        ]:
            from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
            from shapely.ops import transform

            lines, distributions, weights, line_names = prepare_traffic_lists(data)
            structures, depths = split_structures_and_depths(data)
            structure_geoms = [s['wkt'] for s in structures]
            depth_geoms = [d['wkt'] for d in depths]
            transformed_lines, transformed_objs_all, utm_epsg = transform_to_utm(lines, structure_geoms + depth_geoms)
            n_struct = len(structure_geoms)
            transformed_structs = transformed_objs_all[:n_struct]
            transformed_depths = transformed_objs_all[n_struct:]

            # Create reverse transform (UTM -> WGS84) for converting fixed geometries back
            # This ensures wkt_wgs84 has the same vertex order as wkt (UTM)
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
        ) -> tuple[list, list, list, list, list, list, list, list, list]:
            struct_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            depth_min_dists = compute_min_distance_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            struct_overlap_fracs_dir = compute_dir_overlap_fraction_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            struct_overlap_fracs_dir_leg = compute_dir_leg_overlap_fraction_by_object(
                transformed_lines, distributions, weights, structs_gdfs, distance=reach_distance
            ) if len(structs_gdfs) > 0 else []
            depth_overlap_fracs_dir = compute_dir_overlap_fraction_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            depth_overlap_fracs_dir_leg = compute_dir_leg_overlap_fraction_by_object(
                transformed_lines, distributions, weights, depths_gdfs, distance=reach_distance
            ) if len(depths_gdfs) > 0 else []
            depth_overlap_fracs_leg = compute_leg_overlap_fraction(
                transformed_lines, distributions, weights, depths_gdfs
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

            # Calculate structures using accurate dblquad integration (allision)
            # This computes the true geometric probability that a drifting ship
            # hits the obstacle, integrating the lateral PDF along the leg.
            # Distance-dependent repair probability is handled separately in the
            # cascade via get_not_repaired().
            struct_probability_holes = compute_probability_holes(
                transformed_lines, distributions, weights, structs_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(structs_gdfs) > 0 else []

            struct_done = True  # Switch to depths

            # Calculate depths using accurate dblquad integration (grounding)
            # NOTE: No draught filtering here - the cascade calculation
            # filters by draught per vessel category
            depth_probability_holes = compute_probability_holes(
                transformed_lines, distributions, weights, depths_gdfs,
                distance=reach_distance,
                progress_callback=spatial_progress_callback
            ) if len(depths_gdfs) > 0 else []
            return (
                struct_min_dists, depth_min_dists,
                struct_overlap_fracs_dir, depth_overlap_fracs_dir,
                depth_overlap_fracs_leg,
                depth_overlap_fracs_dir_leg,
                struct_overlap_fracs_dir_leg,
                struct_probability_holes,
                depth_probability_holes,
            )

    def _find_nearest_targets(self,
            leg_idx: int,
            d_idx: int,
            height: float,
            draught: float,
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            anchor_d: float,
        ) -> tuple[float | None, int | None, float | None, int | None, float | None]:
        """Find nearest allision target, grounding target, and anchor depth.

        Returns:
            (allision_dist, allision_idx, grounding_dist, grounding_idx, anchor_dist)
        """
        # Convert compass d_idx to math index for array lookups
        # The min_dists arrays use math convention (index 0 = East, index 2 = North)
        math_dir_idx = _compass_idx_to_math_idx(d_idx)

        # Find nearest allision candidate (structure lower than ship height)
        allision_dist, allision_idx = None, None
        if struct_min_dists:
            for s_idx, s in enumerate(structures):
                if s['height'] < height:
                    md = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                    if md is not None and (allision_dist is None or md < allision_dist):
                        allision_dist, allision_idx = md, s_idx

        # Find nearest grounding candidate (depth shallower than draught)
        grounding_dist, grounding_idx = None, None
        if depth_min_dists:
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < draught:
                    md = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                    if md is not None and (grounding_dist is None or md < grounding_dist):
                        grounding_dist, grounding_idx = md, dep_idx

        # Find nearest anchor candidate
        anchor_dist = None
        if depth_min_dists and anchor_d > 0.0:
            thr = anchor_d * draught
            for dep_idx, dep in enumerate(depths):
                if dep['depth'] < thr:
                    md = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                    if md is not None and (anchor_dist is None or md < anchor_dist):
                        anchor_dist = md

        return allision_dist, allision_idx, grounding_dist, grounding_idx, anchor_dist

    def _compute_overlap_fractions(self,
            leg_idx: int,
            d_idx: int,
            allision_idx: int | None,
            grounding_idx: int | None,
            allision_dist: float | None,
            grounding_dist: float | None,
            struct_overlap_fracs_dir: list,
            struct_overlap_fracs_dir_leg: list,
            depth_overlap_fracs_leg: list,
            depth_overlap_fracs_dir_leg: list,
            depth_overlap_fracs_dir: list,
            struct_probability_holes: list,
        ) -> tuple[float, float, bool]:
        """Compute overlap fractions for allision and grounding.

        Returns:
            (ov_all, ov_gro, gro_has_true_overlap)
        """
        # Convert compass d_idx to math index for array lookups
        math_dir_idx = _compass_idx_to_math_idx(d_idx)

        # Allision overlap - use probability hole as primary metric
        ov_all = 0.0
        if allision_idx is not None and allision_dist is not None:
            # Primary: use probability hole (integrated probability mass)
            try:
                if struct_probability_holes:
                    ov_all = struct_probability_holes[leg_idx][math_dir_idx][allision_idx]
            except Exception:
                pass

            # Fallback: use traditional overlap metrics if hole calculation failed
            if ov_all <= 0.0 and struct_overlap_fracs_dir:
                ov_all = struct_overlap_fracs_dir[leg_idx][d_idx][allision_idx]
                # Use leg-based directional overlap if larger
                try:
                    if struct_overlap_fracs_dir_leg:
                        ov_all = max(ov_all, struct_overlap_fracs_dir_leg[leg_idx][d_idx][allision_idx])
                except Exception:
                    pass

        # Grounding overlap
        ov_gro = 0.0
        gro_has_true_overlap = False
        if grounding_idx is not None and grounding_dist is not None and depth_overlap_fracs_leg:
            # Primary: fraction of original leg overlapping shallow depth
            ov_gro = depth_overlap_fracs_leg[leg_idx][grounding_idx]
            if ov_gro > 0.0:
                gro_has_true_overlap = True
            # Fallback: if leg overlap is zero, use directional corridor measured along the leg
            if ov_gro <= 0.0 and depth_overlap_fracs_dir_leg:
                try:
                    ov_gro = depth_overlap_fracs_dir_leg[leg_idx][d_idx][grounding_idx]
                    if ov_gro > 0.0:
                        gro_has_true_overlap = True
                except Exception:
                    pass
            # Secondary fallback: use directional centre-line overlap
            if ov_gro <= 0.0 and depth_overlap_fracs_dir:
                try:
                    ov_gro = depth_overlap_fracs_dir[leg_idx][d_idx][grounding_idx]
                except Exception:
                    pass

        return ov_all, ov_gro, gro_has_true_overlap

    def _select_event_type(self,
            allision_dist: float | None,
            grounding_dist: float | None,
            allision_idx: int | None,
            grounding_idx: int | None,
            ov_all: float,
            ov_gro: float,
            gro_has_true_overlap: bool,
        ) -> tuple[str, float | None, int | None]:
        """Determine whether this is an allision or grounding event.

        Returns:
            (event_type, distance, target_idx)
        """
        # Initial choice based on distance
        choose_allision = False
        if allision_dist is not None and grounding_dist is not None:
            choose_allision = allision_dist <= grounding_dist
        elif allision_dist is not None:
            choose_allision = True
        else:
            choose_allision = False

        event = 'allision' if choose_allision else 'grounding'
        dist = allision_dist if choose_allision else grounding_dist
        idx = allision_idx if choose_allision else grounding_idx

        # If chosen event has zero overlap but the other has overlap, switch
        # Only switch to grounding if grounding has a real leg-based overlap
        if event == 'allision' and ov_all <= 0.0 and gro_has_true_overlap and ov_gro > 0.0:
            event = 'grounding'
            dist = grounding_dist
            idx = grounding_idx
        elif event == 'grounding' and ov_gro <= 0.0 and ov_all > 0.0:
            event = 'allision'
            dist = allision_dist
            idx = allision_idx

        return event, dist, idx

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
        # Per-object accumulation
        try:
            if event == 'allision' and idx is not None:
                o = structures[idx]
                okey = f"Structure - {o.get('id', str(idx))}"
                ob = report['by_object'].setdefault(okey, {'allision': 0.0, 'grounding': 0.0})
                ob['allision'] += contrib
            elif event == 'grounding' and idx is not None:
                o = depths[idx]
                okey = f"Depth - {o.get('id', str(idx))}"
                ob = report['by_object'].setdefault(okey, {'allision': 0.0, 'grounding': 0.0})
                ob['grounding'] += contrib
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
            for seg_idx, segment in enumerate(segments):
                if _segment_intersects_corridor(segment, drift_corridor, drift_angle, leg_centroid):
                    intersecting_indices.append(seg_idx)

            if not intersecting_indices:
                return

            # Distribute contribution equally among intersecting segments
            contrib_per_segment = contrib / len(intersecting_indices)

            # Initialize data structure if needed
            obs_seg_map = report.setdefault(report_key, {}).setdefault(obstacle_key, {})

            # Store contribution for each intersecting segment
            for seg_idx in intersecting_indices:
                seg_key = f"seg_{seg_idx}"
                seg_data = obs_seg_map.setdefault(seg_key, {})
                seg_data[leg_dir_key] = seg_data.get(leg_dir_key, 0.0) + contrib_per_segment

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

    def _iterate_traffic_and_sum(self,
            data: dict[str, Any],
            line_names: list[str],
            transformed_lines: list[LineString],
            structures: list[dict[str, Any]],
            depths: list[dict[str, Any]],
            struct_min_dists: list,
            depth_min_dists: list,
            struct_overlap_fracs_dir: list,
            depth_overlap_fracs_dir: list,
            depth_overlap_fracs_leg: list,
            depth_overlap_fracs_dir_leg: list,
            struct_overlap_fracs_dir_leg: list,
            struct_probability_holes: list,
            depth_probability_holes: list,
            distributions: list[list[Any]] | None = None,
            weights: list[list[float]] | None = None,
            reach_distance: float = 0.0,
        ) -> tuple[float, float, dict[str, Any]]:
        drift = data['drift']
        blackout_per_hour = float(drift.get('drift_p', 0.0)) / (365.0 * 24.0)
        anchor_p = float(drift.get('anchor_p', 0.0))
        anchor_d = float(drift.get('anchor_d', 0.0))
        drift_speed_kts = float(drift.get('speed', 0.0))
        drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert knots to m/s

        # Rose helper
        rose_vals = {int(k): float(v) for k, v in drift.get('rose', {}).items()}
        rose_total = sum(rose_vals.values())
        def rose_prob(idx: int) -> float:
            angle = idx * 45
            v = rose_vals.get(angle, 0.0)
            return (v / rose_total) if rose_total > 0 else 0.0

        # Compose traffic per leg
        traffic_by_leg: list[list[dict[str, float]]] = []
        for geom, _, _, leg_traffic, _ in clean_traffic(data):
            traffic_by_leg.append(leg_traffic)

        # Prepare report structure
        report: dict[str, Any] = {
            'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0},
            'by_leg_direction': {},
            'by_object': {},
            'by_structure_legdir': {},
            'by_depth_legdir': {},  # Per-depth per leg-direction contributions for grounding
            'by_anchoring_legdir': {},  # Per-depth per leg-direction contributions for anchoring
            'by_structure_segment_legdir': {},  # Per-segment per leg-direction for structures
            'by_depth_segment_legdir': {},  # Per-segment per leg-direction for depths
            'by_anchoring_segment_legdir': {},  # Per-segment per leg-direction for anchoring
        }

        total_allision = 0.0
        total_grounding = 0.0
        total_anchoring = 0.0

        # Count total cascade iterations for progress tracking
        # Each leg x each ship cell x 8 directions
        total_cascade_work = sum(
            len(traffic_by_leg[i]) * 8 if i < len(traffic_by_leg) else 0
            for i in range(len(transformed_lines))
        )
        cascade_progress = 0

        for leg_idx, line in enumerate(transformed_lines):
            # Segment id and length
            try:
                nm = line_names[leg_idx]
                seg_id = nm.split('Leg ')[1].split('-')[0].strip()
            except Exception:
                seg_id = str(leg_idx)
            line_length = float(data.get('segment_data', {}).get(seg_id, {}).get('line_length', line.length))

            ship_cells = traffic_by_leg[leg_idx] if leg_idx < len(traffic_by_leg) else []
            for cell in ship_cells:
                freq = float(cell.get('freq', 0.0))
                speed_kts = float(cell.get('speed', 0.0))
                draught = float(cell.get('draught', 0.0))
                height = float(cell.get('height', 0.0))
                ship_type = int(cell.get('ship_type', -1))
                ship_size = int(cell.get('ship_size', -1))
                if speed_kts <= 0.0 or freq <= 0.0:
                    continue
                hours_present = (line_length / (speed_kts * 1852.0)) * freq
                base = hours_present * blackout_per_hour

                for d_idx in range(8):
                    rp = rose_prob(d_idx)
                    if rp <= 0.0:
                        continue

                    # Create drift corridor for per-segment intersection checking
                    drift_corridor: Polygon | None = None
                    if distributions is not None and weights is not None and reach_distance > 0:
                        try:
                            # Calculate lateral spread from distributions
                            dists = distributions[leg_idx] if leg_idx < len(distributions) else []
                            wgts = weights[leg_idx] if leg_idx < len(weights) else []
                            if dists and wgts:
                                w = np.array(wgts)
                                if w.sum() > 0:
                                    w = w / w.sum()
                                    weighted_std = float(np.sqrt(sum(
                                        wt * (dist.std() ** 2) for dist, wt in zip(dists, w) if wt > 0
                                    )))
                                    lateral_spread = 5.0 * weighted_std  # 5 sigma range
                                    compass_angle = d_idx * 45  # Compass angle (0=N, 45=NE, 90=E, etc.)
                                    # Convert compass to math convention for _create_drift_corridor
                                    # Compass: 0=North (CW), Math: 0=East (CCW)
                                    math_angle = (90 - compass_angle) % 360
                                    drift_corridor = _create_drift_corridor(
                                        line, math_angle, reach_distance, lateral_spread
                                    )
                        except Exception:
                            drift_corridor = None

                    # Build list of all obstacles with their distances and holes
                    obstacles: list[tuple[str, int, float, float]] = []

                    # Convert compass d_idx to math index for array lookups
                    # The min_dists and probability_holes arrays use math convention
                    math_dir_idx = _compass_idx_to_math_idx(d_idx)

                    # Add all structures (allision targets)
                    if struct_min_dists and struct_probability_holes:
                        for s_idx, s in enumerate(structures):
                            if s['height'] < height:
                                try:
                                    dist = struct_min_dists[leg_idx][math_dir_idx][s_idx]
                                    hole_pct = struct_probability_holes[leg_idx][math_dir_idx][s_idx]
                                    if dist is not None and hole_pct > 0.0:
                                        obstacles.append(('allision', s_idx, dist, hole_pct))
                                except (IndexError, TypeError):
                                    pass

                    # Add all depths (anchoring or grounding)
                    if depth_min_dists and depth_probability_holes:
                        anchor_threshold = anchor_d * draught if anchor_d > 0.0 else 0.0
                        for dep_idx, dep in enumerate(depths):
                            try:
                                dist = depth_min_dists[leg_idx][math_dir_idx][dep_idx]
                                hole_pct = depth_probability_holes[leg_idx][math_dir_idx][dep_idx]
                                if dist is None or hole_pct <= 0.0:
                                    continue

                                # Determine if this depth is for anchoring or grounding
                                if anchor_threshold > 0.0 and dep['depth'] < anchor_threshold:
                                    obstacles.append(('anchoring', dep_idx, dist, hole_pct))
                                if dep['depth'] < draught:
                                    obstacles.append(('grounding', dep_idx, dist, hole_pct))
                            except (IndexError, TypeError):
                                pass

                    if not obstacles:
                        continue

                    # Sort obstacles by distance (closest first)
                    obstacles.sort(key=lambda x: x[2])

                    # Process cascade: track remaining probability
                    remaining_prob = 1.0

                    for obs_type, obs_idx, dist, hole_pct in obstacles:
                        if remaining_prob <= 0.0:
                            break

                        if obs_type == 'anchoring':
                            # Anchoring: calculate the probability reduction and track per-segment
                            # The "anchor contribution" is the probability of successfully anchoring
                            # at this depth, which shadows obstacles behind it.
                            anchor_contrib = base * rp * remaining_prob * anchor_p * hole_pct
                            total_anchoring += anchor_contrib

                            # Update report with per-segment anchoring tracking
                            self._update_anchoring_report(
                                report, anchor_contrib, obs_idx, depths, seg_id,
                                d_idx, dist, hole_pct, drift_corridor, line
                            )

                            # Anchoring reduces remaining probability
                            remaining_prob *= (1.0 - anchor_p * hole_pct)

                        elif obs_type == 'allision':
                            # Allision: calculate contribution from this structure
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_allision += contrib

                            # Update report with drift corridor for per-segment tracking
                            self._update_report(
                                report, 'allision', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size, drift_corridor, line
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

                        elif obs_type == 'grounding':
                            # Grounding: calculate contribution from this depth
                            p_nr = get_not_repaired(drift['repair'], drift_speed, dist)
                            contrib = base * rp * remaining_prob * hole_pct * p_nr

                            total_grounding += contrib

                            # Update report with drift corridor for per-segment tracking
                            self._update_report(
                                report, 'grounding', contrib, obs_idx,
                                structures, depths, seg_id, cell, d_idx, dist,
                                base, rp, 1.0 - remaining_prob, p_nr, hole_pct, freq,
                                ship_type, ship_size, drift_corridor, line
                            )

                            # Reduce remaining probability
                            remaining_prob *= (1.0 - hole_pct)

                    # Update cascade progress after each direction
                    cascade_progress += 1
                    if total_cascade_work > 0 and cascade_progress % max(1, total_cascade_work // 20) == 0:
                        phase_progress = cascade_progress / total_cascade_work
                        if not self._report_progress(
                            'cascade', phase_progress,
                            f"Drifting - traffic cascade (leg {leg_idx + 1}/{len(transformed_lines)})"
                        ):
                            # Cancelled - return early with partial results
                            report['totals']['allision'] = total_allision
                            report['totals']['grounding'] = total_grounding
                            report['totals']['anchoring'] = total_anchoring
                            return total_allision, total_grounding, report

        report['totals']['allision'] = total_allision
        report['totals']['grounding'] = total_grounding
        report['totals']['anchoring'] = total_anchoring
        return total_allision, total_grounding, report

    def _auto_generate_drifting_report(self, data: dict[str, Any]) -> str | None:
        """Auto-generate the drifting Markdown report to disk.

        Path resolution priority:
        - If the UI field LEReportPath has a value, write to that path
        - Otherwise, write to '<cwd>/drifting_report.md'

        Returns the written content on success, else None.
        """
        try:
            # Prefer UI-provided path if present
            ui_path = None
            try:
                if hasattr(self.p.main_widget, 'LEReportPath') and self.p.main_widget.LEReportPath is not None:
                    t = self.p.main_widget.LEReportPath.text()
                    if isinstance(t, str) and t.strip():
                        ui_path = t.strip()
            except Exception:
                ui_path = None

            path = ui_path or str(Path(os.getcwd()) / 'drifting_report.md')
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return self.write_drifting_report_markdown(path, data)
        except Exception:
            # Silent failure: do not interrupt calculations/UI/tests
            return None

    def run_drifting_model(self, data: dict[str, Any]) -> tuple[float, float]:
        """Compute drifting allision and grounding, and store a breakdown report."""
        if not data.get('traffic_data') or not data.get('segment_data'):
            self.p.main_widget.LEPDriftAllision.setText(f"{float(0):.3e}")
            try:
                self.p.main_widget.LEPDriftingGrounding.setText(f"{float(0):.3e}")
            except Exception:
                pass
            self.drifting_allision_prob = 0.0
            self.drifting_grounding_prob = 0.0
            return 0.0, 0.0

        # Build transformed inputs once
        (
            lines, distributions, weights, line_names,
            structures, depths,
            structs_gdfs, depths_gdfs,
            transformed_lines,
        ) = self._build_transformed(data)

        if len(structs_gdfs) == 0 and len(depths_gdfs) == 0:
            self.p.main_widget.LEPDriftAllision.setText(f"{float(0):.3e}")
            try:
                self.p.main_widget.LEPDriftingGrounding.setText(f"{float(0):.3e}")
            except Exception:
                pass
            self.drifting_allision_prob = 0.0
            self.drifting_grounding_prob = 0.0
            return 0.0, 0.0

        longest_length = max(line.length for line in transformed_lines) if transformed_lines else 0.0
        reach_distance = self._compute_reach_distance(data, longest_length)
        drift = data.get('drift', {})
        (
            struct_min_dists, depth_min_dists,
            struct_overlap_fracs_dir, depth_overlap_fracs_dir,
            depth_overlap_fracs_leg,
            depth_overlap_fracs_dir_leg,
            struct_overlap_fracs_dir_leg,
            struct_probability_holes,
            depth_probability_holes,
        ) = self._precompute_spatial(
            transformed_lines, distributions, weights,
            structs_gdfs, depths_gdfs, reach_distance, data
        )

        total_allision, total_grounding, report = self._iterate_traffic_and_sum(
            data, line_names, transformed_lines, structures, depths,
            struct_min_dists, depth_min_dists,
            struct_overlap_fracs_dir, depth_overlap_fracs_dir, depth_overlap_fracs_leg,
            depth_overlap_fracs_dir_leg, struct_overlap_fracs_dir_leg,
            struct_probability_holes, depth_probability_holes,
            distributions, weights, reach_distance
        )

        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))

        self.drifting_allision_prob = float(total_allision * allision_rf)
        self.drifting_grounding_prob = float(total_grounding * grounding_rf)
        self.drifting_report = report

        # Store structures and depths for result layer generation
        self._last_structures = structures
        self._last_depths = depths

        self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
        try:
            self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")
        except Exception:
            pass
        # Auto-generate Markdown report to disk (best-effort, non-blocking)
        self._report_progress('layers', 0.0, "Drifting - generating report...")
        self._auto_generate_drifting_report(data)

        # Create result layers showing where allisions/groundings occurred
        self._report_progress('layers', 0.3, "Drifting - creating result layers...")
        try:
            self.allision_result_layer, self.grounding_result_layer = create_result_layers(
                report, structures, depths, add_to_project=True
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create result layers: {e}")

        self._report_progress('layers', 1.0, "Drifting model complete")
        return self.drifting_allision_prob, self.drifting_grounding_prob
