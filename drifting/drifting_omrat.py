"""
Drifting model adapter using drifting.engine geometry primitives.

This module provides a drop-in replacement for the DriftingModelMixin calculation
pipeline. It reuses the spatial precomputation and traffic iteration structure but
leverages drifting.engine for geometric operations, providing cleaner separation
of concerns between geometry and probability modeling.

The main entry point is `compute_drifting_probabilities()`, which takes the same
input dict as the current `run_drifting_model()` and produces identical output.
"""

from typing import Any, Callable
from scipy import stats
from shapely.geometry import LineString, Polygon

from drifting.engine import (
    DepthTarget,
    StructureTarget,
    ShipState,
    LegState,
    DriftConfig,
    evaluate_leg_direction,
    build_directional_corridor,
)

from compute.basic_equations import get_not_repaired


class DriftingEngineCalculator:
    """
    Orchestrates drifting probability calculation using drifting.engine geometry.

    This class maintains compatibility with OMRAT's existing DriftingModelMixin
    interface while leveraging drifting.engine for geometric operations.
    """

    def __init__(self, progress_callback: Callable[[int, int, str], bool] | None = None):
        """
        Initialize calculator with optional progress reporting.

        Args:
            progress_callback: Function(completed%, total%, message) -> bool to continue.
        """
        self._progress_callback = progress_callback

    def _report_progress(
        self, phase: str, phase_progress: float, message: str
    ) -> bool:
        """
        Report progress across phases:
          'spatial' (0-40%), 'shadow' (40-60%), 'cascade' (60-90%), 'layers' (90-100%).

        Args:
            phase: One of 'spatial', 'shadow', 'cascade', 'layers'
            phase_progress: Progress within phase (0.0 to 1.0)
            message: Status message

        Returns:
            True to continue, False to cancel
        """
        if not self._progress_callback:
            return True

        phase_weights = {
            "spatial": (0.0, 0.40),
            "shadow":  (0.40, 0.60),
            "cascade": (0.60, 0.90),
            "layers":  (0.90, 1.0),
        }

        start, end = phase_weights.get(phase, (0.0, 1.0))
        overall_progress = start + (end - start) * min(1.0, max(0.0, phase_progress))
        return self._progress_callback(
            int(overall_progress * 100), 100, message
        )

    def compute_drifting_probabilities(
        self,
        data: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Compute drifting allision and grounding probabilities.

        This is the main entry point, matching the interface of run_drifting_model().

        Args:
            data: Configuration dict with keys:
              - traffic_data: {leg_id: {direction: {Frequency, Speed, draught, height, ...}}}
              - segment_data: {leg_id: {line_length}}
              - drift: {speed, drift_p, rose: {0:0.125, 45:0.125, ...}, anchor_p, anchor_d, repair: {...}}
              - geometry: {segments: [...], structures: [...], depths: [...]}
              - pc: {allision_drifting_rf, grounding_drifting_rf}

        Returns:
            (allision_probability, grounding_probability, report_dict)
        """
        if not data.get("traffic_data") or not data.get("segment_data"):
            return 0.0, 0.0, {"totals": {"allision": 0.0, "grounding": 0.0, "anchoring": 0.0}}

        allision, grounding, report = self._compute_via_engine(data)

        # Apply risk factors
        pc_vals = data.get("pc", {}) if isinstance(data.get("pc", {}), dict) else {}
        allision_rf = float(pc_vals.get("allision_drifting_rf", 1.0))
        grounding_rf = float(pc_vals.get("grounding_drifting_rf", 1.0))

        return float(allision * allision_rf), float(grounding * grounding_rf), report

    def _compute_via_engine(self, data: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
        """
        Internal computation using drifting.engine primitives.

        Follows the same three-stage pipeline:
        1. Extract and validate inputs
        2. Build engine objects (LegState, DepthTarget, StructureTarget, ShipState)
        3. Iterate traffic and cascade
        """
        drift = data.get("drift", {})
        segment_data = data.get("segment_data", {})
        traffic_data = data.get("traffic_data", {})
        geometry_data = data.get("geometry", {})

        # --- Extract geometry ---
        try:
            segments = geometry_data.get("segments", [])
            structures_list = geometry_data.get("structures", [])
            depths_list = geometry_data.get("depths", [])
        except Exception:
            return 0.0, 0.0, {"totals": {"allision": 0.0, "grounding": 0.0, "anchoring": 0.0}}

        if not segments or (not structures_list and not depths_list):
            return 0.0, 0.0, {"totals": {"allision": 0.0, "grounding": 0.0, "anchoring": 0.0}}

        # --- Build engine objects ---
        self._report_progress("spatial", 0.0, "Drifting - building geometry objects...")

        legs: list[LegState] = []
        for seg in segments:
            try:
                seg_id = str(seg.get("id", ""))
                line_str = seg.get("wkt", "")
                mean_offset = float(seg.get("mean_offset_m", 0.0))
                lateral_sigma = float(seg.get("lateral_sigma_m", 100.0))

                # Parse WKT LineString
                from shapely import wkt as shapely_wkt
                line = shapely_wkt.loads(line_str)
                if not isinstance(line, LineString):
                    continue

                legs.append(
                    LegState(
                        leg_id=seg_id,
                        line=line,
                        mean_offset_m=mean_offset,
                        lateral_sigma_m=lateral_sigma,
                    )
                )
            except Exception:
                pass

        structures: list[StructureTarget] = []
        for struct in structures_list:
            try:
                struct_id = str(struct.get("id", ""))
                height_m = float(struct.get("top_height_m", 0.0))
                wkt_str = struct.get("wkt", "")

                from shapely import wkt as shapely_wkt
                geom = shapely_wkt.loads(wkt_str)
                structures.append(
                    StructureTarget(
                        target_id=struct_id,
                        top_height_m=height_m,
                        geometry=geom,
                    )
                )
            except Exception:
                pass

        depths: list[DepthTarget] = []
        for dep in depths_list:
            try:
                dep_id = str(dep.get("id", ""))
                depth_m = float(dep.get("depth", 0.0))
                wkt_str = dep.get("wkt", "")

                from shapely import wkt as shapely_wkt
                geom = shapely_wkt.loads(wkt_str)
                depths.append(
                    DepthTarget(
                        target_id=dep_id,
                        depth_m=depth_m,
                        geometry=geom,
                    )
                )
            except Exception:
                pass

        if not legs or (not structures and not depths):
            return 0.0, 0.0, {"totals": {"allision": 0.0, "grounding": 0.0, "anchoring": 0.0}}

        # --- Compute reach distance ---
        longest_length = max((leg.line.length for leg in legs), default=0.0)
        reach_distance = self._compute_reach_distance(data, longest_length)

        # --- Build drift config ---
        use_leg_offset_for_distance = bool(data.get("drift", {}).get("use_leg_offset_for_distance", False))
        cfg = DriftConfig(
            reach_distance_m=reach_distance,
            corridor_sigma_multiplier=3.0,
            use_leg_offset_for_distance=use_leg_offset_for_distance,
        )

        self._report_progress("spatial", 1.0, "Drifting - starting cascade...")

        # --- Traffic cascade ---
        allision_total, grounding_total, report = self._iterate_traffic_cascade(
            data, legs, structures, depths, cfg
        )

        return allision_total, grounding_total, report

    def _compute_reach_distance(self, data: dict[str, Any], longest_length: float) -> float:
        """Compute reach distance from repair model or default to 10× longest leg."""
        reach_distance = longest_length * 10.0
        try:
            rep = data.get("drift", {}).get("repair", {})
            use_ln = rep.get("use_lognormal", False)
            dist_type = rep.get("dist_type", "")
            t99_h = None

            if dist_type == "weibull":
                wb_shape = float(rep.get("wb_shape", 1.0))
                wb_loc = float(rep.get("wb_loc", 0.0))
                wb_scale = float(rep.get("wb_scale", 1.0))
                t99_h = float(stats.weibull_min(c=wb_shape, loc=wb_loc, scale=wb_scale).ppf(0.99))
            elif use_ln:
                s = float(rep.get("std", 0.0))
                loc = float(rep.get("loc", 0.0))
                scale = float(rep.get("scale", 1.0))
                t99_h = float(stats.lognorm(s, loc=loc, scale=scale).ppf(0.99))

            if t99_h is not None and t99_h > 0:
                drift_speed_kts = float(data.get("drift", {}).get("speed", 0.0))
                drift_speed = drift_speed_kts * 1852.0 / 3600.0  # Convert to m/s
                if drift_speed > 0:
                    reach_distance = drift_speed * 3600.0 * t99_h
                    reach_distance = min(reach_distance, longest_length * 10.0)
        except Exception:
            pass
        return reach_distance

    def _iterate_traffic_cascade(
        self,
        data: dict[str, Any],
        legs: list[LegState],
        structures: list[StructureTarget],
        depths: list[DepthTarget],
        cfg: DriftConfig,
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Main cascade loop: iterate over traffic cells and apply probability reduction.

        For each leg → ship cell → direction:
        1. Build list of reachable obstacles using drifting.engine
        2. Sort by distance (closest first)
        3. Apply cascade: remaining_prob starts at 1.0, reduced by each obstacle
        4. Accumulate probability contributions
        """
        drift = data["drift"]
        segment_data = data.get("segment_data", {})
        traffic_data = data.get("traffic_data", {})

        debug_trace = bool(drift.get("debug_trace", False))
        blackout_per_hour = float(drift.get("drift_p", 0.0)) / (365.0 * 24.0)
        anchor_p = float(drift.get("anchor_p", 0.0))
        anchor_d = float(drift.get("anchor_d", 0.0))
        drift_speed_kts = float(drift.get("speed", 0.0))
        drift_speed = drift_speed_kts * 1852.0 / 3600.0

        # Rose distribution
        rose_vals = {int(k): float(v) for k, v in drift.get("rose", {}).items()}
        rose_total = sum(rose_vals.values())

        def rose_prob(idx: int) -> float:
            angle = idx * 45
            v = rose_vals.get(angle, 0.0)
            return (v / rose_total) if rose_total > 0 else 0.0

        # Initialize report
        report: dict[str, Any] = {
            "totals": {"allision": 0.0, "grounding": 0.0, "anchoring": 0.0},
            "by_leg_direction": {},
            "by_object": {},
        }

        total_allision = 0.0
        total_grounding = 0.0
        total_anchoring = 0.0

        # Count total work for progress tracking
        total_work = 0
        for leg in legs:
            leg_id = leg.leg_id
            if leg_id in traffic_data:
                for direction_str in traffic_data[leg_id].keys():
                    cells = traffic_data[leg_id][direction_str]
                    if isinstance(cells, list):
                        total_work += len(cells) * 8
                    else:
                        total_work += 8

        work_done = 0

        # Main cascade loop
        for leg in legs:
            leg_id = leg.leg_id
            seg_data = segment_data.get(leg_id, {})
            line_length = float(seg_data.get("line_length", leg.line.length))

            if leg_id not in traffic_data:
                continue

            for direction_str, cells_or_list in traffic_data[leg_id].items():
                # Normalize traffic cell list
                if not isinstance(cells_or_list, list):
                    cells_or_list = [cells_or_list]

                for cell in cells_or_list:
                    freq = float(cell.get("freq", 0.0)) or float(
                        cell.get("Frequency (ships/year)", 0.0)
                    )
                    speed_kts = float(cell.get("speed", 0.0)) or float(
                        cell.get("Speed (knots)", 0.0)
                    )
                    draught = float(cell.get("draught", 0.0)) or float(
                        cell.get("Draught (m)", 0.0)
                    )
                    height = float(cell.get("height", 0.0)) or float(
                        cell.get("Height (m)", 0.0)
                    )

                    if speed_kts <= 0.0 or freq <= 0.0:
                        for d_idx in range(8):
                            work_done += 1
                        continue

                    hours_present = (line_length / (speed_kts * 1852.0)) * freq
                    base = hours_present * blackout_per_hour

                    # --- Iterate 8 compass directions ---
                    for d_idx in range(8):
                        direction_deg = d_idx * 45
                        rp = rose_prob(d_idx)

                        if rp <= 0.0:
                            work_done += 1
                            continue

                        # Use drifting.engine to evaluate this direction
                        ship = ShipState(
                            draught_m=draught,
                            ship_height_m=height,
                            anchor_d=anchor_d,
                        )

                        # Get geometric hits via drifting.engine
                        hits = evaluate_leg_direction(
                            leg, ship, direction_deg, depths, structures, cfg
                        )

                        if not hits:
                            work_done += 1
                            continue

                        # Build corridor for segment tracking
                        try:
                            drift_corridor = build_directional_corridor(leg, direction_deg, cfg)
                        except Exception:
                            drift_corridor = None

                        # Sort hits by distance (cascade order)
                        hits.sort(key=lambda h: h.distance_m)

                        # --- Apply cascade ---
                        remaining_prob = 1.0

                        for hit in hits:
                            if remaining_prob <= 0.0:
                                break

                            dist = hit.distance_m
                            hole_pct = hit.coverage_percent / 100.0  # Convert % to fraction
                            role = hit.role

                            if role == "anchoring":
                                # Anchoring: reduces remaining probability by anchor_p × hole_pct
                                contrib = base * rp * remaining_prob * anchor_p * hole_pct
                                total_anchoring += contrib
                                remaining_prob *= 1.0 - anchor_p * hole_pct

                            elif role == "structure":
                                # Allision: structure at water level
                                p_nr = get_not_repaired(drift.get("repair", {}), drift_speed, dist)
                                contrib = base * rp * remaining_prob * hole_pct * p_nr
                                total_allision += contrib
                                remaining_prob *= 1.0 - hole_pct

                            elif role == "grounding":
                                # Grounding: depth obstacle
                                p_nr = get_not_repaired(drift.get("repair", {}), drift_speed, dist)
                                contrib = base * rp * remaining_prob * hole_pct * p_nr
                                total_grounding += contrib
                                remaining_prob *= 1.0 - hole_pct

                        work_done += 1
                        if total_work > 0 and work_done % max(1, total_work // 20) == 0:
                            phase_progress = work_done / total_work
                            if not self._report_progress(
                                "cascade",
                                phase_progress,
                                f"Drifting - cascade ({work_done}/{total_work})",
                            ):
                                # Cancelled
                                report["totals"]["allision"] = total_allision
                                report["totals"]["grounding"] = total_grounding
                                report["totals"]["anchoring"] = total_anchoring
                                return total_allision, total_grounding, report

        report["totals"]["allision"] = total_allision
        report["totals"]["grounding"] = total_grounding
        report["totals"]["anchoring"] = total_anchoring

        return total_allision, total_grounding, report
