"""Powered grounding and allision model calculations.

Extracted from run_calculations.py -- IWRAP Category I and Category II
powered models, both evaluated with shadow-aware ray casting:

* **Category I** (in-lane): the obstacle lies between the leg's two
  waypoints inside the lateral spread.  ``N_I = Pc_I * Q * mass`` where
  ``mass`` is the fraction of the lateral distribution the obstacle
  intercepts along the leg (Hansen eq. 4.15).  No distance decay.
* **Category II** (missed turn): ships fail to turn at the leg's end and
  continue straight.  ``N_II = Pc_II * Q * mass * exp(-d_mean / (ai * V))``
  (Hansen eq. 4.16-4.17).

The two categories cover disjoint stretches of water (the leg itself vs.
the extension past the turning point), so their sum never double counts.
Each has its own causation factor: ``pc['grounding_cat1']`` /
``pc['allision_cat1']`` for Category I and ``pc['grounding']`` /
``pc['allision']`` for Category II.
"""

from numpy import exp
from typing import Any, Iterator

from compute.iwrap_defaults import IWRAP_PC_DEFAULTS
from geometries.get_powered_overlap import (
    SimpleProjector as _PoweredProjector,
    _build_legs_and_obstacles,
    _parse_point,
    _run_all_computations,
)

CATEGORIES: tuple[str, str] = ('cat1', 'cat2')


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
    except Exception:  # nosec B110 B112
        pass
    return default


def _extract_nonneg(array, row_i: int, col_j: int, default: float) -> float:
    """Return array[row_i][col_j] as float if numeric, else default."""
    try:
        if row_i < len(array) and col_j < len(array[row_i]):
            val = array[row_i][col_j]
            if isinstance(val, (int, float)):
                return float(val)
    except Exception:  # nosec B110 B112
        pass
    return default


def _edge_shares(edges: dict | None, recovery: float | None) -> dict[int, float]:
    """Fraction of an obstacle's hit probability landing on each boundary edge.

    ``edges`` is the ``{edge_idx: {mass, mean_dist}}`` block the ray caster
    attaches to every obstacle summary.  Each edge is weighted like the
    obstacle itself -- ``mass * exp(-d / recovery)`` for Cat II, plain
    ``mass`` when *recovery* is ``None`` (Cat I) -- and the weights are
    normalised so the shares sum to one.  The per-edge values in the result
    layer therefore always add up to the obstacle's ``by_obstacle`` figure.
    """
    raw: dict[int, float] = {}
    for e_idx, e in (edges or {}).items():
        mass = e["mass"]
        if mass <= 0:
            continue
        raw[int(e_idx)] = mass if recovery is None else mass * exp(-e["mean_dist"] / recovery)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {e_idx: v / total for e_idx, v in raw.items()}


def _iter_hit_probs(comp: dict, recovery: float) -> Iterator[tuple[str, tuple, float, dict[int, float]]]:
    """Yield ``(category, (kind, obs_id), p_hit, edge_shares)`` per obstacle hit.

    * ``'cat2'``: ``p_hit = mass * exp(-d_mean / recovery)`` -- skipped when
      the recovery distance ``ai * V`` is not positive.
    * ``'cat1'``: ``p_hit = mass`` -- the ship is already on course for the
      obstacle, so there is no distance term.

    ``edge_shares`` splits ``p_hit`` over the obstacle's boundary edges
    (see :func:`_edge_shares`); it is empty when the caster has no edge
    information for that obstacle.
    """
    for obs_key, s in comp["summaries"].items():
        mass, d_mean = s["mass"], s["mean_dist"]
        if mass <= 0 or d_mean <= 0 or recovery <= 0:
            continue
        yield 'cat2', obs_key, mass * exp(-d_mean / recovery), _edge_shares(s.get("edges"), recovery)
    for obs_key, s in ((comp.get("cat1") or {}).get("summaries") or {}).items():
        mass = s["mass"]
        if mass <= 0:
            continue
        yield 'cat1', obs_key, mass, _edge_shares(s.get("edges"), None)


class _ContribAccumulator:
    """Per-obstacle / per-leg / per-edge / per-cell / per-category running totals.

    ``by_obstacle``, ``by_obstacle_leg``, ``by_obstacle_segment_legdir`` and
    ``by_cell`` sum both categories (they feed the result layers and the
    consequence module); ``by_category`` keeps the Cat I / Cat II split for
    the report and the IWRAP comparison.

    ``by_obstacle_segment_legdir[obs_id]["seg_<k>"][dir_key]`` is the share
    of the (obstacle, leg-direction) contribution that landed on boundary
    edge ``k`` (numbering of :func:`geometries.boundary_edges.boundary_edges`).
    Summed over edges and directions it equals ``by_obstacle[obs_id]``.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.by_obstacle: dict[str, float] = {}
        self.by_obstacle_leg: dict[str, dict[str, float]] = {}
        self.by_obstacle_segment_legdir: dict[str, dict[str, dict[str, float]]] = {}
        self.by_cell: dict[str, float] = {}
        self.by_category: dict[str, dict[str, Any]] = {
            cat: {'total': 0.0, 'by_obstacle': {}} for cat in CATEGORIES
        }

    def add(
        self, category: str, c: float, obs_id: Any, dir_key: str, cell_key: str,
        edge_shares: dict[int, float] | None = None,
    ) -> None:
        self.total += c
        k = str(obs_id)
        self.by_obstacle[k] = self.by_obstacle.get(k, 0.0) + c
        leg_map = self.by_obstacle_leg.setdefault(k, {})
        leg_map[dir_key] = leg_map.get(dir_key, 0.0) + c
        self.by_cell[cell_key] = self.by_cell.get(cell_key, 0.0) + c
        cat = self.by_category[category]
        cat['total'] += c
        cat['by_obstacle'][k] = cat['by_obstacle'].get(k, 0.0) + c
        if edge_shares:
            seg_root = self.by_obstacle_segment_legdir.setdefault(k, {})
            for e_idx, share in edge_shares.items():
                seg_map = seg_root.setdefault(f"seg_{e_idx}", {})
                seg_map[dir_key] = seg_map.get(dir_key, 0.0) + c * share

    def report(self, total_key: str, pc_cat2: float, pc_cat1: float) -> dict[str, Any]:
        return {
            'totals': {
                total_key: float(self.total),
                'cat1': float(self.by_category['cat1']['total']),
                'cat2': float(self.by_category['cat2']['total']),
            },
            'by_obstacle': self.by_obstacle,
            'by_obstacle_leg': self.by_obstacle_leg,
            'by_obstacle_segment_legdir': self.by_obstacle_segment_legdir,
            'by_cell': self.by_cell,
            'by_category': self.by_category,
            'causation_factor': pc_cat2,
            'causation_factor_cat1': pc_cat1,
        }


def _empty_report(total_key: str) -> dict[str, Any]:
    return {
        'totals': {total_key: 0.0, 'cat1': 0.0, 'cat2': 0.0},
        'by_obstacle': {}, 'by_obstacle_leg': {}, 'by_obstacle_segment_legdir': {}, 'by_cell': {},
        'by_category': {cat: {'total': 0.0, 'by_obstacle': {}} for cat in CATEGORIES},
    }


def _pc_pair(pc_vals: dict, kind: str) -> tuple[float, float]:
    """``(Cat II factor, Cat I factor)`` for ``kind`` in ``('grounding', 'allision')``.

    Category II keeps its historical keys (``grounding`` with the legacy
    ``p_pc`` alias, ``allision``); Category I reads ``<kind>_cat1`` and
    falls back to the IWRAP default when the project has no such key.
    """
    if kind == 'grounding':
        pc_cat2 = float(pc_vals.get('grounding', pc_vals.get('p_pc', IWRAP_PC_DEFAULTS['grounding'])))
    else:
        pc_cat2 = float(pc_vals.get('allision', IWRAP_PC_DEFAULTS['allision']))
    pc_cat1 = float(pc_vals.get(f'{kind}_cat1', IWRAP_PC_DEFAULTS[f'{kind}_cat1']))
    return pc_cat2, pc_cat1


class PoweredModelMixin:
    """Mixin providing powered grounding and allision model methods.

    Expects ``self.p.main_widget`` to expose the UI line-edit widgets
    ``LEPPoweredGrounding`` and ``LEPPoweredAllision``.
    """

    # ------------------------------------------------------------------
    # Grounding helpers

    def _emit_empty_grounding(self, total: float) -> float:
        self.powered_grounding_report = _empty_report('grounding')
        try:
            self.p.main_widget.LEPPoweredGrounding.setText(f"{total:.3e}")
        except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
                bin_results[bin_key] = []
        return bin_results

    def _sum_grounding_contribs(
        self, traffic_data: dict, segment_data: dict,
        bin_results: dict, unique_depths: list[float],
        pc_grounding: float, pc_grounding_cat1: float,
    ) -> _ContribAccumulator:
        pc_by_cat = {'cat2': pc_grounding, 'cat1': pc_grounding_cat1}
        acc = _ContribAccumulator()
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
                            dir_key = f"{leg_key}:{dir_idx}"
                            cell_key = f"{loa_i}_{type_j}"
                            for category, (_, obs_id), p_hit, shares in _iter_hit_probs(comp, ai * speed_ms):
                                acc.add(category, pc_by_cat[category] * q * p_hit,
                                        obs_id, dir_key, cell_key, edge_shares=shares)
        return acc

    def _finalize_grounding(
        self, acc: _ContribAccumulator, pc_grounding: float,
        pc_grounding_cat1: float, segment_data: dict,
    ) -> None:
        total = acc.total
        self.powered_grounding_report = acc.report('grounding', pc_grounding, pc_grounding_cat1)
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
        except Exception:  # nosec B110 B112
            pass

    # ------------------------------------------------------------------
    # Allision helpers

    def _emit_empty_allision(self, total: float) -> float:
        self.powered_allision_report = _empty_report('allision')
        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:  # nosec B110 B112
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
        obj_heights: dict, pc_allision: float, pc_allision_cat1: float,
    ) -> _ContribAccumulator:
        pc_by_cat = {'cat2': pc_allision, 'cat1': pc_allision_cat1}
        acc = _ContribAccumulator()
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
                    dir_key = f"{seg_id}:{dir_idx}"
                    cell_key = f"{loa_i}_{type_j}"
                    for category, (kind, obs_id), p_hit, shares in _iter_hit_probs(comp, ai * speed_ms):
                        # Clearance check applies to both categories: a ship
                        # lower than the structure passes under it whether it
                        # is in the lane or has missed the turn.
                        if kind == "object" and ship_h < obj_heights.get(str(obs_id), 0.0):
                            continue
                        acc.add(category, pc_by_cat[category] * q * p_hit,
                                obs_id, dir_key, cell_key, edge_shares=shares)
        return acc

    def _finalize_allision(
        self, acc: _ContribAccumulator, pc_allision: float,
        pc_allision_cat1: float, segment_data: dict,
    ) -> None:
        total = acc.total
        self.powered_allision_report = acc.report('allision', pc_allision, pc_allision_cat1)
        try:
            from geometries.result_layers import create_powered_allision_layer
            structs_meta = getattr(self, '_last_powered_allision_structs', None) or []
            self.powered_allision_layer = create_powered_allision_layer(
                self.powered_allision_report, structs_meta,
                add_to_project=False, segment_data=segment_data,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Failed to create powered-allision layer: {e}")
        try:
            self.p.main_widget.LEPPoweredAllision.setText(f"{total:.3e}")
        except Exception:  # nosec B110 B112
            pass

    # ------------------------------------------------------------------
    # Public entry points

    def run_powered_grounding_model(self, data: dict[str, Any]) -> float:
        """Calculate powered grounding probability using shadow-aware ray casting.

        Category I: shallow areas inside the leg's lateral spread.
        N_I = Pc_I * Q * mass
        Category II: ships fail to turn at a bend and continue straight,
        potentially running aground on shallow depth areas beyond it.
        N_II = Pc_II * Q * mass * exp(-d_mean / (ai * V))
        The returned total is N_I + N_II; the report keeps the split under
        ``totals['cat1']`` / ``totals['cat2']`` and ``by_category``.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        depths_list = data.get('depths', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not depths_list:
            return self._emit_empty_grounding(total)

        pc_grounding, pc_grounding_cat1 = _pc_pair(pc_vals, 'grounding')
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
        except Exception:  # nosec B110 B112
            return self._emit_empty_grounding(total)

        draught_set = self._collect_draught_set(traffic_data)
        unique_depths = self._build_unique_depths(depths_list)
        bin_results = self._precompute_bin_results(draught_set, unique_depths, data, proj)
        acc = self._sum_grounding_contribs(
            traffic_data, segment_data, bin_results, unique_depths,
            pc_grounding, pc_grounding_cat1,
        )
        self._finalize_grounding(acc, pc_grounding, pc_grounding_cat1, segment_data)
        return acc.total

    def run_powered_allision_model(self, data: dict[str, Any]) -> float:
        """Calculate powered allision probability using shadow-aware ray casting.

        Category I: structures inside the leg's lateral spread.
        N_I = Pc_I * Q * mass
        Category II: ships fail to turn at a bend and continue straight,
        potentially hitting structures (objects) beyond it.
        N_II = Pc_II * Q * mass * exp(-d_mean / (ai * V))
        The returned total is N_I + N_II; the report keeps the split under
        ``totals['cat1']`` / ``totals['cat2']`` and ``by_category``.
        """
        total = 0.0
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        objects_list = data.get('objects', [])
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data or not objects_list:
            return self._emit_empty_allision(total)

        pc_allision, pc_allision_cat1 = _pc_pair(pc_vals, 'allision')
        try:
            first_seg = segment_data[list(segment_data.keys())[0]]
            lon0, lat0 = _parse_point(first_seg["Start_Point"])
            proj = _PoweredProjector(lon0, lat0)
            legs, all_obstacles, _, _, _ = _build_legs_and_obstacles(
                data, proj, mode="allision", max_draft=0)
        except Exception:  # nosec B110 B112
            return self._emit_empty_allision(total)

        if not all_obstacles:
            return self._emit_empty_allision(total)

        # obj_heights and structs_meta must use the same split IDs as all_obstacles
        # (which splits MultiPolygons into sub-polygons with IDs like '1_0', '1_1').
        try:
            from shapely import wkt as _sw
        except Exception:  # nosec B110 B112
            _sw = None
        obj_heights: dict[str, float] = {}
        structs_meta_for_layer: list[dict] = []
        for obj in objects_list:
            try:
                oid, height, wkt_str = obj
                height_f = float(height) if height else 0.0
                if _sw is not None:
                    geom_wgs84 = _sw.loads(wkt_str)
                    if geom_wgs84.geom_type == 'MultiPolygon':
                        for i, poly in enumerate(geom_wgs84.geoms):
                            sub_id = f"{oid}_{i}"
                            obj_heights[sub_id] = height_f
                            structs_meta_for_layer.append(
                                {'id': sub_id, 'height': height_f, 'wkt_wgs84': poly.wkt}
                            )
                        continue
                obj_heights[str(oid)] = height_f
                structs_meta_for_layer.append(
                    {'id': str(oid), 'height': height_f, 'wkt_wgs84': wkt_str}
                )
            except Exception:  # nosec B110 B112
                pass
        self._last_powered_allision_structs = structs_meta_for_layer

        computations = _run_all_computations(legs, all_obstacles)
        acc = self._sum_allision_contribs(
            computations, traffic_data, obj_heights, pc_allision, pc_allision_cat1,
        )
        self._finalize_allision(acc, pc_allision, pc_allision_cat1, segment_data)
        return acc.total
