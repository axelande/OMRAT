"""Powered grounding and allision model calculations.

Extracted from run_calculations.py -- Category II IWRAP models where ships
fail to turn at a bend and continue straight, using shadow-aware ray casting.
"""

from numpy import exp
from typing import Any

from geometries.get_powered_overlap import (
    SimpleProjector as _PoweredProjector,
    _build_legs_and_obstacles,
    _parse_point,
    _run_all_computations,
)


def _depth_bin_key(draught: float, unique_depths: list[float]) -> float | None:
    valid = [d for d in unique_depths if d <= draught]
    return valid[-1] if valid else None


def _extract_positive(array, row_i: int, col_j: int, default: float) -> float:
    """Return array[row_i][col_j] if positive numeric, else default."""
    try:
        if row_i < len(array) and col_j < len(array[row_i]):
            val = array[row_i][col_j]
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
            if isinstance(val, str) and val != '':
                return float(val)
    except Exception:
        pass
    return default


def _extract_nonneg(array, row_i: int, col_j: int, default: float) -> float:
    """Return array[row_i][col_j] as float if numeric, else default."""
    try:
        if row_i < len(array) and col_j < len(array[row_i]):
            val = array[row_i][col_j]
            if isinstance(val, (int, float)):
                return float(val)
    except Exception:
        pass
    return default


class PoweredModelMixin:
    """Mixin providing powered grounding and allision model methods.

    Expects ``self.p.main_widget`` to expose the UI line-edit widgets
    ``LEPPoweredGrounding`` and ``LEPPoweredAllision``.
    """

    # ------------------------------------------------------------------
    # Grounding helpers

    def _emit_empty_grounding(self, total: float) -> float:
        self.powered_grounding_report = {
            'totals': {'grounding': 0.0}, 'by_obstacle': {},
            'by_obstacle_leg': {}, 'by_cell': {},
        }
        try:
            self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
        except Exception:
            pass
        return total

    def _collect_draught_set(self, traffic_data: dict) -> set[float]:
        draught_set: set[float] = set()
        for leg_dirs in traffic_data.values():
            for dir_data in leg_dirs.values():
                for row in dir_data.get('Draught (meters)', []):
                    if not hasattr(row, '__iter__'):
                        continue
                    for d_val in row:
                        try:
                            v = float(d_val) if d_val != '' else 0.0
                            if v > 0:
                                draught_set.add(v)
                        except (ValueError, TypeError):
                            pass
        return draught_set or {5.0}

    def _build_unique_depths(self, depths_list: list) -> list[float]:
        depth_values: list[float] = []
        for dep in depths_list:
            try:
                depth_values.append(float(dep[1]))
            except (IndexError, ValueError, TypeError):
                continue
        return sorted(set(depth_values))

    def _precompute_bin_results(
        self,
        draught_set: set[float],
        unique_depths: list[float],
        data: dict,
        proj: Any,
    ) -> dict:
        bins_needed = {_depth_bin_key(d, unique_depths) for d in draught_set}
        sorted_bins = sorted([b for b in bins_needed if b is not None])
        if None in bins_needed:
            sorted_bins = [None] + sorted_bins
        bin_results: dict = {}
        total_bins = len(sorted_bins)
        for idx, bin_key in enumerate(sorted_bins, start=1):
            if getattr(self, "_progress_callback", None):
                self._progress_callback(
                    idx - 1, max(total_bins, 1),
                    f"Powered grounding: obstacle bin {idx}/{max(total_bins, 1)}",
                )
            try:
                max_draft = -1.0 if bin_key is None else float(bin_key)
                legs, all_obs, _, _, _ = _build_legs_and_obstacles(
                    data, proj, mode="grounding", max_draft=max_draft)
                bin_results[bin_key] = _run_all_computations(legs, all_obs) if all_obs else []
            except Exception:
                bin_results[bin_key] = []
        return bin_results

    def _sum_grounding_contribs(
        self, traffic_data: dict, segment_data: dict,
        bin_results: dict, unique_depths: list[float], pc_grounding: float,
    ) -> tuple[float, dict, dict, dict]:
        total = 0.0
        by_obstacle: dict[str, float] = {}
        by_obstacle_leg: dict[str, dict[str, float]] = {}
        by_cell: dict[str, float] = {}
        for leg_key, leg_dirs in traffic_data.items():
            seg_info = segment_data.get(leg_key, {})
            ai_per_dir = [float(seg_info.get('ai1', 180.0)), float(seg_info.get('ai2', 180.0))]
            for dir_idx, (_, dir_data) in enumerate(leg_dirs.items()):
                ai = ai_per_dir[min(dir_idx, 1)]
                freq_arr = dir_data.get('Frequency (ships/year)', [])
                drau_arr = dir_data.get('Draught (meters)', [])
                spd_arr = dir_data.get('Speed (knots)', [])
                for loa_i, freq_row in enumerate(freq_arr):
                    if not hasattr(freq_row, '__iter__'):
                        continue
                    for type_j, fv in enumerate(freq_row):
                        try:
                            q = float(fv) if fv != '' else 0.0
                        except (ValueError, TypeError):
                            q = 0.0
                        if q <= 0:
                            continue
                        draught = _extract_positive(drau_arr, loa_i, type_j, 5.0)
                        speed_ms = _extract_positive(spd_arr, loa_i, type_j, 10.0) * 1852.0 / 3600.0
                        for comp in bin_results.get(_depth_bin_key(draught, unique_depths), []):
                            if comp["seg_id"] != leg_key or comp["dir_idx"] != dir_idx:
                                continue
                            for (_, obs_id), s in comp["summaries"].items():
                                mass, d_mean = s["mass"], s["mean_dist"]
                                recovery = ai * speed_ms
                                if mass <= 0 or d_mean <= 0 or recovery <= 0:
                                    continue
                                c = pc_grounding * q * mass * exp(-d_mean / recovery)
                                total += c
                                k = str(obs_id)
                                by_obstacle[k] = by_obstacle.get(k, 0.0) + c
                                _leg_map = by_obstacle_leg.setdefault(k, {})
                                _leg_map[str(leg_key)] = _leg_map.get(str(leg_key), 0.0) + c
                                ck = f"{loa_i}_{type_j}"
                                by_cell[ck] = by_cell.get(ck, 0.0) + c
        return total, by_obstacle, by_obstacle_leg, by_cell

    def _finalize_grounding(
        self, total: float, by_obstacle: dict, by_obstacle_leg: dict,
        by_cell: dict, pc_grounding: float, segment_data: dict,
    ) -> None:
        self.powered_grounding_report = {
            'totals': {'grounding': float(total)},
            'by_obstacle': by_obstacle,
            'by_obstacle_leg': by_obstacle_leg,
            'by_cell': by_cell,
            'causation_factor': pc_grounding,
        }
        try:
            from geometries.result_layers import create_powered_grounding_layer
            depths_meta = (
                getattr(self, '_last_depths_original', None)
                or getattr(self, '_last_depths', None) or []
            )
            self.powered_grounding_layer = create_powered_grounding_layer(
                self.powered_grounding_report, depths_meta,
                add_to_project=False, segment_data=segment_data,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Failed to create powered-grounding layer: {e}")
        try:
            self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Allision helpers

    def _emit_empty_allision(self, total: float) -> float:
        self.powered_allision_report = {
            'totals': {'allision': 0.0}, 'by_obstacle': {},
            'by_obstacle_leg': {}, 'by_cell': {},
        }
        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:
            pass
        return total

    @staticmethod
    def _build_obj_heights(objects_list: list) -> dict[str, float]:
        obj_heights: dict[str, float] = {}
        for obj in objects_list:
            try:
                obj_heights[str(obj[0])] = float(obj[1])
            except (IndexError, ValueError, TypeError):
                pass
        return obj_heights

    def _sum_allision_contribs(
        self, computations: list, traffic_data: dict,
        obj_heights: dict, pc_allision: float,
    ) -> tuple[float, dict, dict, dict]:
        total = 0.0
        by_obstacle: dict[str, float] = {}
        by_obstacle_leg: dict[str, dict[str, float]] = {}
        by_cell: dict[str, float] = {}
        for comp in computations:
            seg_id, dir_idx = comp["seg_id"], comp["dir_idx"]
            ai = comp["dir_info"]["ai"]
            leg_dirs = traffic_data.get(seg_id, {})
            dir_keys = list(leg_dirs.keys())
            if dir_idx >= len(dir_keys):
                continue
            dir_data = leg_dirs[dir_keys[dir_idx]]
            freq_arr = dir_data.get('Frequency (ships/year)', [])
            spd_arr = dir_data.get('Speed (knots)', [])
            hgt_arr = dir_data.get('Ship heights (meters)', [])
            for loa_i, freq_row in enumerate(freq_arr):
                if not hasattr(freq_row, '__iter__'):
                    continue
                for type_j, fv in enumerate(freq_row):
                    try:
                        q = float(fv) if fv != '' else 0.0
                    except (ValueError, TypeError):
                        q = 0.0
                    if q <= 0:
                        continue
                    speed_ms = _extract_positive(spd_arr, loa_i, type_j, 10.0) * 1852.0 / 3600.0
                    ship_h = _extract_nonneg(hgt_arr, loa_i, type_j, 0.0)
                    for (kind, obs_id), s in comp["summaries"].items():
                        mass, d_mean = s["mass"], s["mean_dist"]
                        recovery = ai * speed_ms
                        if mass <= 0 or d_mean <= 0 or recovery <= 0:
                            continue
                        if kind == "object" and ship_h < obj_heights.get(str(obs_id), 0.0):
                            continue
                        c = pc_allision * q * mass * exp(-d_mean / recovery)
                        total += c
                        k = str(obs_id)
                        by_obstacle[k] = by_obstacle.get(k, 0.0) + c
                        _seg_map = by_obstacle_leg.setdefault(k, {})
                        _seg_map[str(seg_id)] = _seg_map.get(str(seg_id), 0.0) + c
                        ck = f"{loa_i}_{type_j}"
                        by_cell[ck] = by_cell.get(ck, 0.0) + c
        return total, by_obstacle, by_obstacle_leg, by_cell

    def _finalize_allision(
        self, total: float, by_obstacle: dict, by_obstacle_leg: dict,
        by_cell: dict, pc_allision: float, segment_data: dict,
    ) -> None:
        self.powered_allision_report = {
            'totals': {'allision': float(total)},
            'by_obstacle': by_obstacle,
            'by_obstacle_leg': by_obstacle_leg,
            'by_cell': by_cell,
            'causation_factor': pc_allision,
        }
        try:
            from geometries.result_layers import create_powered_allision_layer
            structs_meta = getattr(self, '_last_structures', None) or []
            self.powered_allision_layer = create_powered_allision_layer(
                self.powered_allision_report, structs_meta,
                add_to_project=False, segment_data=segment_data,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Failed to create powered-allision layer: {e}")
        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public entry points

    def run_powered_grounding_model(self, data: dict[str, Any]) -> float:
        """Calculate powered grounding probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially running aground on shallow depth areas.
        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        depths_list = data.get('depths', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not depths_list:
            return self._emit_empty_grounding(total)

        pc_grounding = float(pc_vals.get('grounding', pc_vals.get('p_pc', 1.6e-4)))
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
        except Exception:
            return self._emit_empty_grounding(total)

        draught_set = self._collect_draught_set(traffic_data)
        unique_depths = self._build_unique_depths(depths_list)
        bin_results = self._precompute_bin_results(draught_set, unique_depths, data, proj)
        total, by_obs, by_obs_leg, by_cell = self._sum_grounding_contribs(
            traffic_data, segment_data, bin_results, unique_depths, pc_grounding,
        )
        self._finalize_grounding(total, by_obs, by_obs_leg, by_cell, pc_grounding, segment_data)
        return total

    def run_powered_allision_model(self, data: dict[str, Any]) -> float:
        """Calculate powered allision probability using shadow-aware ray casting.

        Category II: ships fail to turn at a bend and continue straight,
        potentially hitting structures (objects).
        N_II = Pc * Q * mass * exp(-d_mean / (ai * V))
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        objects_list = data.get('objects', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not objects_list:
            return self._emit_empty_allision(total)

        pc_allision = float(pc_vals.get('allision', 1.9e-4))
        obj_heights = self._build_obj_heights(objects_list)
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
            legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                data, proj, mode="allision", max_draft=0)
        except Exception:
            return self._emit_empty_allision(total)

        if not all_obstacles:
            return self._emit_empty_allision(total)

        computations = _run_all_computations(legs, all_obstacles)
        total, by_obs, by_obs_leg, by_cell = self._sum_allision_contribs(
            computations, traffic_data, obj_heights, pc_allision,
        )
        self._finalize_allision(total, by_obs, by_obs_leg, by_cell, pc_allision, segment_data)
        return total
