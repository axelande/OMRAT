"""Powered grounding and allision model calculations.

Extracted from run_calculations.py -- Category II IWRAP models where ships
fail to turn at a bend and continue straight, using shadow-aware ray casting.
"""

import numpy as np
from numpy import exp
from typing import Any

from geometries.get_powered_overlap import (
    PoweredOverlapVisualizer,
    SimpleProjector as _PoweredProjector,
    _build_legs_and_obstacles,
    _parse_point,
    _run_all_computations,
)
from compute.data_preparation import get_distribution


class PoweredModelMixin:
    """Mixin providing powered grounding and allision model methods.

    Expects ``self.p.main_widget`` to expose the UI line-edit widgets
    ``LEPPoweredGrounding`` and ``LEPPoweredAllision``.
    """

    def run_powered_grounding_model(self, data: dict[str, Any]) -> float:
        """Calculate powered grounding probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially running aground on shallow depth areas.

        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))

        Shadow effect: closer depth areas block the distribution for areas
        behind them.  Only depths shallower than ship draught count.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        depths_list = data.get('depths', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not depths_list:
            self.powered_grounding_report = {
                'totals': {'grounding': 0.0}, 'by_obstacle': {},
                'by_obstacle_leg': {},
            }
            try:
                self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        pc_grounding = float(pc_vals.get('grounding', pc_vals.get('p_pc', 1.6e-4)))
        by_obstacle: dict[str, float] = {}
        # Per-(obstacle, leg) breakdown for the result-layer attributes.
        # Outer key = obstacle id, inner key = leg id.
        by_obstacle_leg: dict[str, dict[str, float]] = {}

        # Build projector
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
        except Exception:
            try:
                self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        # Collect all unique draughts from traffic data so we can compute
        # shadow-aware results once per draught bracket.
        draught_set: set[float] = set()
        for leg_key, leg_dirs in traffic_data.items():
            for dir_key, dir_data in leg_dirs.items():
                draught_array = dir_data.get('Draught (meters)', [])
                for row in draught_array:
                    if not hasattr(row, '__iter__'):
                        continue
                    for d_val in row:
                        try:
                            v = float(d_val) if d_val != '' else 0.0
                            if v > 0:
                                draught_set.add(v)
                        except (ValueError, TypeError):
                            pass
        if not draught_set:
            draught_set = {5.0}  # Default draught

        # Effective bins: obstacle set only changes when crossing a depth contour.
        # This avoids repeating heavy geometry/ray computations for many similar draughts.
        depth_values: list[float] = []
        for dep in depths_list:
            try:
                depth_values.append(float(dep[1]))
            except (IndexError, ValueError, TypeError):
                continue
        unique_depths = sorted(set(depth_values))

        def _depth_bin_key(draught: float) -> float | None:
            valid = [d for d in unique_depths if d <= draught]
            return valid[-1] if valid else None

        bins_needed: set[float | None] = {_depth_bin_key(d) for d in draught_set}

        # Cache: depth-bin key -> precomputed overlap results for that obstacle set.
        bin_results: dict[float | None, list[dict]] = {}
        sorted_bins = sorted([b for b in bins_needed if b is not None])
        if None in bins_needed:
            sorted_bins = [None] + sorted_bins

        total_bins = len(sorted_bins)
        for idx, bin_key in enumerate(sorted_bins, start=1):
            # Optional progress callback so long powered runs don't look stuck.
            if getattr(self, "_progress_callback", None):
                self._progress_callback(
                    idx - 1,
                    max(total_bins, 1),
                    f"Powered grounding: obstacle bin {idx}/{max(total_bins, 1)}",
                )

            try:
                max_draft = -1.0 if bin_key is None else float(bin_key)
                legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                    data, proj, mode="grounding", max_draft=max_draft)
                if all_obstacles:
                    bin_results[bin_key] = _run_all_computations(legs, all_obstacles)
                else:
                    bin_results[bin_key] = []
            except Exception:
                bin_results[bin_key] = []

        # Sum per-ship-type contributions using pre-computed shadow results
        for leg_key, leg_dirs in traffic_data.items():
            seg_info = segment_data.get(leg_key, {})
            ai_per_dir = [
                float(seg_info.get('ai1', 180.0)),
                float(seg_info.get('ai2', 180.0)),
            ]

            for dir_idx, (dir_key, dir_data) in enumerate(leg_dirs.items()):
                ai_seconds = ai_per_dir[min(dir_idx, 1)]
                freq_array = dir_data.get('Frequency (ships/year)', [])
                draught_array = dir_data.get('Draught (meters)', [])
                speed_array = dir_data.get('Speed (knots)', [])

                for loa_i, freq_row in enumerate(freq_array):
                    if not hasattr(freq_row, '__iter__'):
                        continue
                    for type_j, freq_val in enumerate(freq_row):
                        try:
                            q = float(freq_val) if freq_val != '' else 0.0
                        except (ValueError, TypeError):
                            q = 0.0
                        if q <= 0:
                            continue

                        # Get ship draught
                        draught = 5.0
                        try:
                            if loa_i < len(draught_array) and type_j < len(draught_array[loa_i]):
                                d_val = draught_array[loa_i][type_j]
                                if isinstance(d_val, (int, float)) and d_val > 0:
                                    draught = float(d_val)
                                elif isinstance(d_val, str) and d_val != '':
                                    draught = float(d_val)
                        except Exception:
                            pass

                        # Get ship speed
                        speed_kts = 10.0
                        try:
                            if loa_i < len(speed_array) and type_j < len(speed_array[loa_i]):
                                s_val = speed_array[loa_i][type_j]
                                if isinstance(s_val, (int, float)) and s_val > 0:
                                    speed_kts = float(s_val)
                                elif isinstance(s_val, str) and s_val != '':
                                    speed_kts = float(s_val)
                        except Exception:
                            pass
                        speed_ms = speed_kts * 1852.0 / 3600.0

                        # Map ship draught to effective depth bin results.
                        comps = bin_results.get(_depth_bin_key(draught), [])

                        # Sum over matching leg/direction computations
                        for comp in comps:
                            if comp["seg_id"] != leg_key:
                                continue
                            if comp["dir_idx"] != dir_idx:
                                continue
                            for key, s in comp["summaries"].items():
                                _kind, obs_id = key
                                mass = s["mass"]
                                d_mean = s["mean_dist"]
                                if mass <= 0 or d_mean <= 0:
                                    continue
                                recovery = ai_seconds * speed_ms
                                if recovery <= 0:
                                    continue
                                prob_not_rec = exp(-d_mean / recovery)
                                contrib = pc_grounding * q * mass * prob_not_rec
                                total += contrib
                                # Track per-depth-id for the result layer.
                                obs_id_str = str(obs_id)
                                by_obstacle[obs_id_str] = (
                                    by_obstacle.get(obs_id_str, 0.0) + contrib
                                )
                                # ...and per-(depth, leg) so the layer
                                # can show which leg dominates.
                                leg_str = str(leg_key)
                                obs_legs = by_obstacle_leg.setdefault(obs_id_str, {})
                                obs_legs[leg_str] = (
                                    obs_legs.get(leg_str, 0.0) + contrib
                                )

        # Persist per-obstacle breakdown for the result layer.
        self.powered_grounding_report = {
            'totals': {'grounding': float(total)},
            'by_obstacle': by_obstacle,
            'by_obstacle_leg': by_obstacle_leg,
            'causation_factor': pc_grounding,
        }

        # Build the polygon result layer if we have geometry from the
        # earlier drifting phase (which transforms + stores ``wkt_wgs84``).
        try:
            from geometries.result_layers import create_powered_grounding_layer
            depths_meta = getattr(self, '_last_depths', None) or []
            self.powered_grounding_layer = create_powered_grounding_layer(
                self.powered_grounding_report, depths_meta, add_to_project=False,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Failed to create powered-grounding layer: {e}"
            )

        try:
            self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
        except Exception:
            pass
        return total

    def run_powered_allision_model(self, data: dict[str, Any]) -> float:
        """Calculate powered allision probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially hitting structures (objects).

        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))

        Shadow effect: closer obstacles block the distribution for obstacles
        behind them.  Mass = fraction of lateral distribution intercepted.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        objects_list = data.get('objects', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not objects_list:
            self.powered_allision_report = {
                'totals': {'allision': 0.0}, 'by_obstacle': {},
                'by_obstacle_leg': {},
            }
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        pc_allision = float(pc_vals.get('allision', 1.9e-4))
        by_obstacle: dict[str, float] = {}
        by_obstacle_leg: dict[str, dict[str, float]] = {}

        # Build obstacle height lookup from objects list, keyed by ID
        obj_heights: dict[str, float] = {}
        for obj in objects_list:
            try:
                obj_heights[str(obj[0])] = float(obj[1])
            except (IndexError, ValueError, TypeError):
                pass

        # Build projector and geometry once
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
            legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                data, proj, mode="allision", max_draft=0)
        except Exception:
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        if not all_obstacles:
            try:
                self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
            except Exception:
                pass
            return total

        # Shadow-aware ray casting: compute mass & d_mean per obstacle
        # per leg/direction (geometry only, independent of ship type)
        computations = _run_all_computations(legs, all_obstacles)

        # For each computation (leg/dir with hits), sum per-ship-type
        for comp in computations:
            seg_id = comp["seg_id"]
            dir_idx = comp["dir_idx"]
            d_info = comp["dir_info"]
            ai_seconds = d_info["ai"]

            leg_dirs = traffic_data.get(seg_id, {})
            dir_keys = list(leg_dirs.keys())
            if dir_idx >= len(dir_keys):
                continue
            dir_data = leg_dirs[dir_keys[dir_idx]]

            freq_array = dir_data.get('Frequency (ships/year)', [])
            speed_array = dir_data.get('Speed (knots)', [])
            height_array = dir_data.get('Ship heights (meters)', [])

            for loa_i, freq_row in enumerate(freq_array):
                if not hasattr(freq_row, '__iter__'):
                    continue
                for type_j, freq_val in enumerate(freq_row):
                    try:
                        q = float(freq_val) if freq_val != '' else 0.0
                    except (ValueError, TypeError):
                        q = 0.0
                    if q <= 0:
                        continue

                    speed_kts = 10.0
                    try:
                        if loa_i < len(speed_array) and type_j < len(speed_array[loa_i]):
                            s_val = speed_array[loa_i][type_j]
                            if isinstance(s_val, (int, float)) and s_val > 0:
                                speed_kts = float(s_val)
                            elif isinstance(s_val, str) and s_val != '':
                                speed_kts = float(s_val)
                    except Exception:
                        pass
                    speed_ms = speed_kts * 1852.0 / 3600.0

                    # Get ship height for allision filtering
                    ship_height = 0.0
                    try:
                        if loa_i < len(height_array) and type_j < len(height_array[loa_i]):
                            h_val = height_array[loa_i][type_j]
                            if isinstance(h_val, (int, float)):
                                ship_height = float(h_val)
                    except Exception:
                        pass

                    # Sum over obstacles hit by this leg/direction
                    for key, s in comp["summaries"].items():
                        kind, obs_id = key
                        mass = s["mass"]
                        d_mean = s["mean_dist"]
                        if mass <= 0 or d_mean <= 0:
                            continue
                        # Height check: if ship is shorter than the structure,
                        # it passes under (e.g. ship under a bridge)
                        if kind == "object":
                            struct_height = obj_heights.get(str(obs_id), 0.0)
                            if ship_height < struct_height:
                                continue
                        recovery = ai_seconds * speed_ms
                        if recovery <= 0:
                            continue
                        prob_not_rec = exp(-d_mean / recovery)
                        contrib = pc_allision * q * mass * prob_not_rec
                        total += contrib
                        obs_id_str = str(obs_id)
                        by_obstacle[obs_id_str] = (
                            by_obstacle.get(obs_id_str, 0.0) + contrib
                        )
                        # Per-(structure, leg) breakdown for the layer.
                        leg_str = str(seg_id)
                        obs_legs = by_obstacle_leg.setdefault(obs_id_str, {})
                        obs_legs[leg_str] = (
                            obs_legs.get(leg_str, 0.0) + contrib
                        )

        # Persist per-obstacle breakdown for the result layer.
        self.powered_allision_report = {
            'totals': {'allision': float(total)},
            'by_obstacle': by_obstacle,
            'by_obstacle_leg': by_obstacle_leg,
            'causation_factor': pc_allision,
        }

        # Build the polygon result layer.
        try:
            from geometries.result_layers import create_powered_allision_layer
            structs_meta = getattr(self, '_last_structures', None) or []
            self.powered_allision_layer = create_powered_allision_layer(
                self.powered_allision_report, structs_meta, add_to_project=False,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Failed to create powered-allision layer: {e}"
            )

        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:
            pass
        return total
