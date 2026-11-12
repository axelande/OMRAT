"""Ship-ship collision model mixin.

Extracted from ``run_calculations.py`` to improve maintainability.
The :class:`ShipCollisionModelMixin` provides :meth:`run_ship_collision_model`
decomposed into smaller, testable helper methods for each collision type:
head-on and overtaking along a leg; crossing, merging and bend per pair of
routes at a shared waypoint.
"""

import math
from typing import Any

import numpy as np

from compute.basic_equations import (
    get_crossing_collision_candidates,
    head_on_candidates_from_overlap,
    lateral_overlap_probability,
    overtaking_candidates_from_overlap,
)
from compute import passenger_reduction
from compute.encounter_angle import angle_counts, min_angle_rad
from compute.speed_restriction import speed_matrix
from geometries.distribution_curves import mixture as lateral_mixture
from geometries.model_projection import model_projector_or_none, projected_bearing
from geometries.junctions import (
    Junction,
    junction_id_for_point,
    refresh_junction_registry,
)


def _matrix_rows(traffic_data: dict[str, Any]) -> int | None:
    for leg in (traffic_data or {}).values():
        for block in (leg or {}).values() if isinstance(leg, dict) else []:
            freq = block.get('Frequency (ships/year)') if isinstance(block, dict) else None
            if freq:
                return len(freq)
    return None


def collision_type_scale(data: dict[str, Any], pc_vals: dict[str, Any]) -> list[float] | None:
    """Per ship-type row multiplier on the collision causation factor, or None.

    ``1 / sqrt(F)`` for passenger ships and fast ferries
    (``compute.passenger_reduction``); ``None`` when both factors are 1.
    """
    if not passenger_reduction.is_active(pc_vals):
        return None
    types = (data.get('ship_categories') or {}).get('types')
    return passenger_reduction.collision_scales(types, pc_vals, _matrix_rows(data.get('traffic_data') or {}))


class ShipCollisionModelMixin:
    """Mixin that adds the ship-ship collision model to a calculation runner.

    Expects the host class to provide:
    * ``self._report_progress(stage, fraction, message)``
    * ``self.p.main_widget`` with collision-result line-edit widgets
    * ``self.ship_collision_prob`` (written)
    * ``self.collision_report`` (written)
    """

    # ------------------------------------------------------------------
    # Static helpers (previously nested functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _overlap(cache: dict[float, float], comps1: list, comps2: list, b1: float, b2: float) -> float:
        """``P_G`` of the two mixtures for the mean half-breadth ``(b1 + b2) / 2``, cached per breadth."""
        half = round(0.5 * (float(b1) + float(b2)), 3)
        p = cache.get(half)
        if p is None:
            p = lateral_overlap_probability(comps1, comps2, half)
            cache[half] = p
        return p

    def _type_scales(self) -> list[float] | None:
        """Per ship-type multiplier on the collision causation factor
        (``compute.passenger_reduction.collision_scales``), set by
        ``run_ship_collision_model``; ``None`` = no reduction."""
        return getattr(self, '_collision_type_scale', None)

    @staticmethod
    def _pair_scale(scale: list[float] | None, row1: int, row2: int) -> float:
        if not scale:
            return 1.0
        s1 = scale[row1] if 0 <= row1 < len(scale) else 1.0
        s2 = scale[row2] if 0 <= row2 < len(scale) else 1.0
        return s1 * s2

    @staticmethod
    def get_loa_midpoint(loa_idx: int, length_intervals: list[dict]) -> float:
        """Get midpoint of LOA category for length estimates.

        ``loa_idx`` is the *length class* -- the column of the traffic
        matrices.  The collision loops passed the row (the ship type) until
        2026-11-09, so a pleasure craft got 162.5 m, a tug 262.5 m and every
        cargo ship 150 m in the crossing / merging / bend diameter and the
        beam estimate (KNOWN_ISSUES 29).
        """
        if loa_idx < len(length_intervals):
            interval = length_intervals[loa_idx]
            try:
                min_val = float(interval.get('min', 50))
                max_val = float(interval.get('max', 100))
                return (min_val + max_val) / 2.0
            except (ValueError, TypeError):
                pass
        default_midpoints = [25.0, 75.0, 150.0, 250.0, 350.0]
        if loa_idx < len(default_midpoints):
            return default_midpoints[loa_idx]
        return 150.0

    # No longer classifies anything: since 2026-11-09 the route pair decides
    # between crossing and merging (``_route_pair_kind``).  Kept because
    # ``compare_explain.MERGING_LIMIT_DEG`` mirrors it (a test pins them).
    MERGING_ANGLE_DEG: float = 30.0

    @staticmethod
    def estimate_beam(loa: float) -> float:
        """Estimate beam from LOA using typical ship ratios (L/B ~ 6-7)."""
        return loa / 6.5

    # ------------------------------------------------------------------
    # Private helpers for crossing calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_point(pt_str: str) -> tuple[float, float] | None:
        """Parse 'x y' coordinate string to (x, y) tuple."""
        if not pt_str:
            return None
        parts = str(pt_str).strip().split()
        if len(parts) >= 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _calc_bearing(
        start: tuple[float, float], end: tuple[float, float], projector: Any = None,
    ) -> float:
        """Bearing in degrees (0=N, CW) from start to end (lon/lat), measured in a projection.

        ``projector``: the model's :class:`~geometries.model_projection.ModelProjector`
        (the one the powered models use); ``None`` centres one on ``start``.
        Until 2026-11-12 this was ``atan2(dlon, dlat)`` on raw degrees, which
        at 57 N made a 44 degree turn 28 and a 50 degree one 80
        (KNOWN_ISSUES 36).
        """
        return projected_bearing(start, end, projector)

    @staticmethod
    def _points_match(
        p1: tuple[float, float] | None,
        p2: tuple[float, float] | None,
        tol: float = 1e-6,
    ) -> bool:
        """Check if two coordinate points are the same within tolerance."""
        if p1 is None or p2 is None:
            return False
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    @classmethod
    def _end_at(cls, seg: dict[str, Any], pt: tuple[float, float] | None) -> str | None:
        """``'start'`` / ``'end'`` when ``pt`` is that end of the leg, else ``None``.

        Picks the speed matrix at that waypoint (speed restriction zones,
        :func:`compute.speed_restriction.speed_matrix`).
        """
        if cls._points_match(cls._parse_point(seg.get('Start_Point', '')), pt):
            return 'start'
        if cls._points_match(cls._parse_point(seg.get('End_Point', '')), pt):
            return 'end'
        return None

    # ------------------------------------------------------------------
    # Junction-matrix lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _junction_at_point(
        junctions: dict[str, Junction] | None,
        pt: tuple[float, float] | None,
    ) -> Junction | None:
        """Return the junction registered at ``pt`` if any."""
        if not junctions or pt is None:
            return None
        j = junctions.get(junction_id_for_point(pt))
        if j is not None:
            return j
        # Tolerant fallback for stored coords that round differently.
        for cand in junctions.values():
            if abs(cand.point[0] - pt[0]) < 1e-6 and abs(cand.point[1] - pt[1]) < 1e-6:
                return cand
        return None

    @staticmethod
    def _coerce_junctions(
        raw: Any,
    ) -> dict[str, Junction] | None:
        """Accept either a live registry or a serialised junctions dict.

        Returns ``None`` when no junctions are supplied, so callers can
        keep the legacy (matrix-free) code path with a single
        ``if junctions is None`` check.
        """
        if not raw:
            return None
        # Already a registry of Junction objects.
        from geometries.junctions import Junction as _J
        sample = next(iter(raw.values()))
        if isinstance(sample, _J):
            return raw  # type: ignore[return-value]
        # Serialised form (dict of dicts).
        from geometries.junctions import deserialize_junctions as _de
        return _de(raw)

    @staticmethod
    def _junction_outward_bearing(
        junction: Junction,
        leg_id: str,
        segment_data: dict[str, Any],
        projector: Any = None,
    ) -> float | None:
        """Bearing from the junction toward leg's far endpoint, in the model's projection.

        ``projector`` defaults to ``model_projector(segment_data)``; pass it
        when taking several bearings of one model.
        """
        seg = (segment_data or {}).get(leg_id) or {}
        sp = ShipCollisionModelMixin._parse_point(seg.get('Start_Point', ''))
        ep = ShipCollisionModelMixin._parse_point(seg.get('End_Point', ''))
        if sp is None or ep is None:
            return None
        side = junction.legs.get(str(leg_id))
        far = ep if side == 'start' else sp if side == 'end' else None
        if far is None:
            return None
        if projector is None:
            projector = model_projector_or_none(segment_data)
        return ShipCollisionModelMixin._calc_bearing(junction.point, far, projector)

    # ------------------------------------------------------------------
    # Ship-property extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_speed_ms(speed_arr: Any, loa_i: int, type_j: int) -> float:
        """Extract speed in m/s for a ship cell from a nested speed array.

        A missing speed (``inf`` from an AIS cell without observations, or
        <= 0) falls back to 10 kn like ``powered_model._extract_positive``:
        every encounter formula has ``Q / V`` times a relative speed, so one
        ``inf`` made the whole accident total NaN (KNOWN_ISSUES 14).
        ``traffic_gaps.fill_missing_speeds`` normally fills it first.
        """
        v_kts = 10.0
        if loa_i < len(speed_arr) and type_j < len(speed_arr[loa_i]):
            s_list = speed_arr[loa_i][type_j]
            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                raw = float(np.mean(s_list))
            elif isinstance(s_list, (int, float)):
                raw = float(s_list)
            else:
                raw = math.nan
            if math.isfinite(raw) and raw > 0.0:
                v_kts = raw
        return v_kts * 1852.0 / 3600.0

    @staticmethod
    def _extract_beam(
        beam_arr: Any, loa_i: int, type_j: int,
        default_beam: float,
    ) -> float:
        """Extract beam (m) for a ship cell; falls back to ``default_beam``.

        A beam that is not finite or not above 0 is missing (0 is IWRAP's
        "not given" and what an import carries); it used to be taken as a
        0 m beam (KNOWN_ISSUES 29).
        """
        if loa_i < len(beam_arr) and type_j < len(beam_arr[loa_i]):
            b_list = beam_arr[loa_i][type_j]
            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                b_raw = float(np.mean(b_list))
                if np.isfinite(b_raw) and b_raw > 0:
                    return b_raw
            elif isinstance(b_list, (int, float)) and np.isfinite(float(b_list)) and float(b_list) > 0:
                return float(b_list)
        return default_beam

    # ------------------------------------------------------------------
    # Collision-type sub-methods
    # ------------------------------------------------------------------

    def _calc_head_on_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        leg_length_m: float,
        pc_headon: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None = None,
    ) -> float:
        """Calculate head-on collision frequency for a single leg.

        Head-on collisions arise from ships travelling in opposite directions
        on the same leg.

        If ``by_cell`` is provided, per-(ship_type, length_idx) annual-frequency
        contributions are accumulated into it.  Each pair (i, j) vs (k, l)
        splits its contribution 50/50 between the two participating cells, so
        single-direction sums add up to the leg total.

        Returns the total head-on collision frequency for this leg.
        """
        dir_keys = list(leg_dirs.keys())
        if len(dir_keys) < 2:
            return 0.0

        dir1, dir2 = dir_keys[0], dir_keys[1]
        arrays1 = self._dir_arrays(leg_dirs.get(dir1, {}))
        arrays2 = self._dir_arrays(leg_dirs.get(dir2, {}))
        # The whole lateral mixture of each direction, component by
        # component (KNOWN_ISSUES 28: one collapsed normal per direction
        # smeared a two-lane leg into the middle).
        comps1 = lateral_mixture(seg_info, 0)
        comps2 = lateral_mixture(seg_info, 1)
        if not comps1 or not comps2:
            return 0.0

        return self._head_on_outer_loop(
            arrays1, arrays2, comps1, comps2,
            leg_length_m, pc_headon, length_intervals, by_cell,
        )

    @staticmethod
    def _dir_arrays(dir_data: dict[str, Any], speed_at: str | None = None) -> tuple[Any, Any, Any]:
        """Extract (freq, speed, beam) numpy arrays from a direction data dict.

        ``speed_at`` (``'start'`` / ``'end'``) takes the speed at that
        waypoint instead of the one along the leg; they differ only on a
        leg in a speed restriction zone.
        """
        return (
            np.array(dir_data.get('Frequency (ships/year)', [])),
            np.array(speed_matrix(dir_data, speed_at)),
            np.array(dir_data.get('Ship Beam (meters)', [])),
        )

    def _head_on_outer_loop(
        self,
        arrays1: tuple[Any, Any, Any],
        arrays2: tuple[Any, Any, Any],
        comps1: list, comps2: list,
        leg_length_m: float, pc_headon: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None,
    ) -> float:
        """Outer loop over dir1 ship cells for head-on collisions.

        Rows of the traffic matrices are ship types, columns length
        classes (``by_cell`` keys ``"{type}_{length}"``); the historical
        names ``loa_i`` / ``type_j`` are the other way round.
        """
        freq1, speed1, beam1 = arrays1
        overlap_cache: dict[float, float] = {}
        total = 0.0
        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                if q1 <= 0 or not np.isfinite(q1):
                    continue
                v1_ms = self._extract_speed_ms(speed1, loa_i, type_j)
                b1 = self._extract_beam(
                    beam1, loa_i, type_j,
                    self.estimate_beam(self.get_loa_midpoint(type_j, length_intervals)),
                )
                total += self._head_on_inner_loop(
                    arrays2[0], arrays2[1], arrays2[2],
                    q1, v1_ms, b1, loa_i, type_j,
                    comps1, comps2, overlap_cache,
                    leg_length_m, pc_headon, length_intervals, by_cell,
                )
        return total

    def _head_on_inner_loop(
        self,
        freq2: Any, speed2: Any, beam2: Any,
        q1: float, v1_ms: float, b1: float,
        loa_i: int, type_j: int,
        comps1: list, comps2: list, overlap_cache: dict[float, float],
        leg_length_m: float, pc_headon: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None,
    ) -> float:
        """Inner loop over dir2 ship categories for head-on collisions."""
        scale = self._type_scales()
        total = 0.0
        for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
            for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                if q2 <= 0 or not np.isfinite(q2):
                    continue
                v2_ms = self._extract_speed_ms(speed2, loa_k, type_l)
                b2 = self._extract_beam(
                    beam2, loa_k, type_l,
                    self.estimate_beam(self.get_loa_midpoint(type_l, length_intervals)),
                )
                p_g = self._overlap(overlap_cache, comps1, comps2, b1, b2)
                n_g_headon = head_on_candidates_from_overlap(q1, q2, v1_ms, v2_ms, p_g, leg_length_m)
                contrib = n_g_headon * pc_headon * self._pair_scale(scale, loa_i, loa_k)
                total += contrib
                if by_cell is not None and contrib > 0.0:
                    half = 0.5 * contrib
                    by_cell[f"{loa_i}_{type_j}"] = by_cell.get(f"{loa_i}_{type_j}", 0.0) + half
                    by_cell[f"{loa_k}_{type_l}"] = by_cell.get(f"{loa_k}_{type_l}", 0.0) + half
        return total

    def _calc_overtaking_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        leg_length_m: float,
        pc_overtaking: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None = None,
    ) -> float:
        """Calculate overtaking collision frequency for a single leg.

        Overtaking collisions occur between ships travelling in the same
        direction at different speeds.

        If ``by_cell`` is provided, per-(ship_type, length_idx) contributions
        are accumulated into it (50/50 split between the overtaking and
        overtaken cell).

        Returns the total overtaking collision frequency for this leg.
        """
        leg_overtaking = 0.0
        dir_keys = list(leg_dirs.keys())

        scale = self._type_scales()
        for dir_idx, dir_key in enumerate(dir_keys):
            freq, speed, beam = self._dir_arrays(leg_dirs.get(dir_key, {}))
            # Both ships come from this direction's whole mixture (KNOWN_ISSUES 28).
            comps = lateral_mixture(seg_info, dir_idx)
            if not comps:
                continue
            overlap_cache: dict[float, float] = {}
            ship_cells = self._collect_ship_cells(freq, speed, beam, length_intervals)

            for i, (loa_i, type_i, q_fast, v_fast, b_fast) in enumerate(ship_cells):
                for j, (loa_j, type_j, q_slow, v_slow, b_slow) in enumerate(ship_cells):
                    if i == j or v_fast <= v_slow:
                        continue

                    p_g = self._overlap(overlap_cache, comps, comps, b_fast, b_slow)
                    n_g_overtaking = overtaking_candidates_from_overlap(
                        q_fast, q_slow, v_fast, v_slow, p_g, leg_length_m)
                    contrib = n_g_overtaking * pc_overtaking * self._pair_scale(scale, loa_i, loa_j)
                    leg_overtaking += contrib
                    if by_cell is not None and contrib > 0.0:
                        half = 0.5 * contrib
                        by_cell[f"{loa_i}_{type_i}"] = by_cell.get(f"{loa_i}_{type_i}", 0.0) + half
                        by_cell[f"{loa_j}_{type_j}"] = by_cell.get(f"{loa_j}_{type_j}", 0.0) + half

        return leg_overtaking

    def _collect_ship_cells(
        self,
        freq: Any, speed: Any, beam: Any,
        length_intervals: list[dict],
    ) -> list[tuple[int, int, float, float, float]]:
        """Build a list of (loa_i, type_j, freq, speed_ms, beam) for non-zero cells."""
        ship_cells: list[tuple[int, int, float, float, float]] = []
        for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
            for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                if q <= 0 or not np.isfinite(q):
                    continue
                v_ms = self._extract_speed_ms(speed, loa_i, type_j)
                b = self._extract_beam(
                    beam, loa_i, type_j,
                    self.estimate_beam(self.get_loa_midpoint(type_j, length_intervals)),
                )
                ship_cells.append((loa_i, type_j, q, v_ms, b))
        return ship_cells

    # Arrival courses closer than this to parallel (overtaking-like) or to
    # opposite (head-on-like) give no waypoint encounter: the crossing
    # formula divides by |sin theta|.  ``pc['min_encounter_angle_deg']``
    # (compute/encounter_angle.py, default 10 degrees = Friis-Hansen's range
    # of validity and IWRAP's limit), set per run by run_ship_collision_model.
    def _min_route_angle_rad(self) -> float:
        return float(getattr(self, '_route_angle_limit_rad', None) or min_angle_rad(None))

    def _calc_crossing_collisions(
        self,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        leg_keys: list[str],
        pc_crossing: float,
        pc_merging: float,
        length_intervals: list[dict],
        by_waypoint: dict[tuple[float, float], dict[str, float]] | None = None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None = None,
        by_cell_crossing: dict[str, float] | None = None,
        by_cell_merging: dict[str, float] | None = None,
        junctions: dict[str, Junction] | None = None,
        pc_bend: float = 0.0,
        by_cell_bend: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Crossing, merging and bend at waypoints, counted per pair of *routes*.

        A route at a waypoint is ``in-leg -> out-leg``: the traffic of the
        in-leg's direction that arrives at the waypoint (speed at the
        waypoint), times the junction matrix share ``T[in][out]``.  Every
        waypoint where legs meet gets a matrix: the stored one (AIS / user)
        or the geometric default (``refresh_junction_registry``).  Each pair
        of routes is counted once:

        * same in-leg: the two sail in one lane -- no waypoint encounter
          (overtaking on the leg already counts them);
        * same out-leg: **merging** (``pc['merging']``);
        * four different legs whose paths interleave around the waypoint:
          **crossing** (``pc['crossing']``);
        * the two reverse routes of one turn (``A -> B`` and ``B -> A``):
          **bend** (``pc['bend']``), with every ship of both routes, at the
          angle between their arrival courses (180 degrees minus the turn);
        * anything else (one route leaves on the leg the other arrives on)
          is head-on on that leg, already counted there.

        The angle is the one between the two arrival courses, taken in the
        model's projection (``model_projector``, as the powered models;
        raw lon/lat degrees until 2026-11-12, KNOWN_ISSUES 36); courses within
        ``pc['min_encounter_angle_deg']`` (default 10 degrees,
        ``compute/encounter_angle.py``) of parallel or opposite are skipped,
        so a turn smaller than that gives no bend.  Pedersen's formula and the
        inputs are unchanged.

        Bend (2026-11-10): Friis-Hansen (2008, section 4.4) only says that
        at a bend a ship that does not change course becomes a collision
        candidate, with an omission probability of 0.01; it gives no
        formula.  The ship that does not turn meets the ships coming the
        other way on the next leg, which is the reverse route.  IWRAP counts
        that pair with the crossing formula and every ship, without the
        omission probability; the user's decision was to do the same.  The
        bend used to be the 1 % non-turners against the turners of their
        own route, at the turn angle, per leg at its End_Point (KNOWN_ISSUES
        31): those ships move apart after the waypoint.

        Until 2026-11-09 every pair of legs at a waypoint was taken with all
        four direction combinations, weighted only by ``1 - T12 T21`` and
        classified by the acute angle between the legs: one X-crossing
        counted in all four leg pairs, flows that both leave the waypoint
        counted, and consecutive legs of one route "merged" with themselves
        (KNOWN_ISSUES 30).  ``MERGING_ANGLE_DEG`` no longer classifies.

        Returns ``{'crossing': float, 'merging': float, 'bend': float}``;
        ``by_waypoint`` (key ``(lon, lat)``), ``by_leg_pair`` (key = the two
        in-legs, in ``leg_keys`` order) and the ``by_cell`` buckets (50/50
        per cell pair) are filled as before.
        """
        totals: dict[str, float] = {'crossing': 0.0, 'merging': 0.0, 'bend': 0.0}
        order = {str(k): i for i, k in enumerate(leg_keys)}
        registry = self._route_registry(junctions, segment_data)
        projector = model_projector_or_none(segment_data)
        n_junctions = max(1, len(registry))
        pcs = {'crossing': pc_crossing, 'merging': pc_merging, 'bend': pc_bend}
        buckets = {'crossing': by_cell_crossing, 'merging': by_cell_merging, 'bend': by_cell_bend}
        self._report_progress('cascade', 0.0, "Calculating crossing collisions...")
        for done, junction in enumerate(registry.values(), start=1):
            for kind, contrib in self._junction_route_encounters(
                junction, traffic_data, segment_data, order, pcs, length_intervals,
                by_waypoint, by_leg_pair, buckets, projector=projector,
            ):
                totals[kind] += contrib
            self._report_progress('cascade', done / n_junctions,
                                  f"Crossing / merging / bend at waypoint {done}/{n_junctions}...")
        return totals

    @staticmethod
    def _route_registry(
        junctions: dict[str, Junction] | None, segment_data: dict[str, Any],
    ) -> dict[str, Junction]:
        """Every waypoint where legs meet, with its matrix (stored or geometric)."""
        try:
            return refresh_junction_registry(dict(junctions or {}), segment_data)
        except Exception:  # nosec B110 - malformed geometry: no waypoint encounters
            return {}

    def _junction_routes(
        self, junction: Junction, traffic_data: dict[str, Any], segment_data: dict[str, Any],
        projector: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """``(routes, outward bearings)`` at one waypoint.

        Each route: ``in``, ``out``, ``share``, ``course`` (arrival course,
        degrees) and the in-leg's arriving ``arrays`` (freq x share, speed at
        the waypoint, beam).  Bearings are taken in ``projector`` (default:
        the model's, :func:`~geometries.model_projection.model_projector`).
        """
        if projector is None:
            projector = model_projector_or_none(segment_data)
        far: dict[str, float] = {}
        for leg_id in junction.legs:
            b = self._junction_outward_bearing(junction, leg_id, segment_data, projector)
            if b is not None:
                far[str(leg_id)] = b
        routes: list[dict[str, Any]] = []
        for in_leg, side in junction.legs.items():
            in_leg = str(in_leg)
            if in_leg not in far:
                continue
            blocks = [v for v in (traffic_data.get(in_leg) or {}).values() if isinstance(v, dict)]
            dir_idx = 0 if side == 'end' else 1        # the drawn direction arrives at End_Point
            if dir_idx >= len(blocks):
                continue
            freq, speed, beam = self._dir_arrays(blocks[dir_idx], side)
            if freq.size == 0:
                continue
            for out_leg, share in (junction.transitions.get(in_leg) or {}).items():
                out_leg = str(out_leg)
                share = float(share)
                if out_leg == in_leg or share <= 0.0 or out_leg not in far:
                    continue
                routes.append({
                    'in': in_leg, 'out': out_leg, 'share': share,
                    'course': (far[in_leg] + 180.0) % 360.0,
                    'arrays': (freq.astype(float) * share, speed, beam),
                })
        return routes, far

    @staticmethod
    def _paths_interleave(a: float, b: float, c: float, d: float) -> bool:
        """Do chords (a, b) and (c, d) -- bearings around the waypoint -- cross?"""
        def between(x: float, lo: float, hi: float) -> bool:
            return 0.0 < (x - lo) % 360.0 < (hi - lo) % 360.0
        return between(c, a, b) != between(d, a, b)

    def _route_pair_kind(self, r1: dict[str, Any], r2: dict[str, Any], far: dict[str, float]) -> str | None:
        """``'merging'``, ``'crossing'``, ``'bend'`` or ``None`` (no waypoint encounter) for two routes."""
        if r1['in'] == r2['in']:
            return None
        if r1['out'] == r2['out']:
            return 'merging'
        if r1['in'] == r2['out'] and r1['out'] == r2['in']:
            return 'bend'
        legs = {r1['in'], r1['out'], r2['in'], r2['out']}
        if len(legs) == 4 and self._paths_interleave(far[r1['in']], far[r1['out']], far[r2['in']], far[r2['out']]):
            return 'crossing'
        return None

    def _junction_route_encounters(
        self,
        junction: Junction,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        order: dict[str, int],
        pcs: dict[str, float],
        length_intervals: list[dict],
        by_waypoint: dict[tuple[float, float], dict[str, float]] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        buckets: dict[str, dict[str, float] | None],
        projector: Any = None,
    ):
        """Yield ``(kind, frequency)`` per pair of routes meeting at ``junction``.

        ``pcs`` / ``buckets``: causation factor and ``by_cell`` bucket per kind;
        ``projector``: the projection the bearings are taken in.
        """
        routes, far = self._junction_routes(junction, traffic_data, segment_data, projector)
        shared_pt = (float(junction.point[0]), float(junction.point[1]))
        limit = self._min_route_angle_rad()
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                r1, r2 = routes[a], routes[b]
                kind = self._route_pair_kind(r1, r2, far)
                if kind is None or not pcs.get(kind):
                    continue
                theta = abs(r1['course'] - r2['course']) % 360.0
                theta = math.radians(360.0 - theta if theta > 180.0 else theta)
                if not angle_counts(theta, limit):
                    continue
                if order.get(r2['in'], 0) < order.get(r1['in'], 0):
                    r1, r2 = r2, r1
                freq1, speed1, beam1 = r1['arrays']
                freq2, speed2, beam2 = r2['arrays']
                contrib = self._crossing_ship_cell_loop(
                    freq1, speed1, beam1, freq2, speed2, beam2,
                    theta, math.degrees(theta), 1.0,
                    pcs[kind], length_intervals,
                    shared_pt, kind, by_waypoint, by_leg_pair,
                    buckets.get('crossing'), buckets.get('merging'),
                    r1['in'], r2['in'], by_cell_bend=buckets.get('bend'),
                )
                yield kind, contrib

    def _crossing_ship_cell_loop(
        self,
        freq1: Any, speed1: Any, beam1_arr: Any,
        freq2: Any, speed2: Any, beam2_arr: Any,
        crossing_angle_rad: float,
        crossing_angle: float,
        conflict_factor: float,
        pc_crossing: float,
        length_intervals: list[dict],
        shared_pt: tuple[float, float],
        kind: str,
        by_waypoint: dict[tuple[float, float], dict[str, float]] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        leg1_key: str,
        leg2_key: str,
        by_cell_bend: dict[str, float] | None = None,
    ) -> float:
        """Iterate over all (leg1 cell) x (leg2 cell) combinations for one direction pair."""
        scale = self._type_scales()
        total = 0.0
        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                if q1 <= 0 or not np.isfinite(q1):
                    continue
                v1_ms = self._extract_speed_ms(speed1, loa_i, type_j)
                l1 = self.get_loa_midpoint(type_j, length_intervals)
                b1 = self._extract_beam(beam1_arr, loa_i, type_j, self.estimate_beam(l1))

                for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                    freq2_k_len = len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0
                    for type_l in range(freq2_k_len):
                        q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                        if q2 <= 0 or not np.isfinite(q2):
                            continue
                        v2_ms = self._extract_speed_ms(speed2, loa_k, type_l)
                        l2 = self.get_loa_midpoint(type_l, length_intervals)
                        b2 = self._extract_beam(beam2_arr, loa_k, type_l, self.estimate_beam(l2))

                        n_g_crossing = get_crossing_collision_candidates(
                            Q1=q1, Q2=q2, V1=v1_ms, V2=v2_ms,
                            L1=l1, L2=l2, B1=b1, B2=b2,
                            theta=crossing_angle_rad
                        )
                        contrib = (n_g_crossing * pc_crossing * conflict_factor
                                   * self._pair_scale(scale, loa_i, loa_k))
                        total += contrib
                        self._accumulate_crossing_contrib(
                            contrib, kind, shared_pt,
                            loa_i, type_j, loa_k, type_l,
                            crossing_angle,
                            by_waypoint, by_leg_pair,
                            by_cell_crossing, by_cell_merging,
                            leg1_key, leg2_key, by_cell_bend=by_cell_bend,
                        )
        return total

    @staticmethod
    def _accumulate_crossing_contrib(
        contrib: float,
        kind: str,
        shared_pt: tuple[float, float],
        loa_i: int, type_j: int,
        loa_k: int, type_l: int,
        crossing_angle: float,
        by_waypoint: dict[tuple[float, float], dict[str, float]] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        leg1_key: str,
        leg2_key: str,
        by_cell_bend: dict[str, float] | None = None,
    ) -> None:
        """Update all accumulator dicts for one crossing / merging / bend cell-pair contribution."""
        if by_waypoint is not None:
            rec_wp = by_waypoint.setdefault(
                shared_pt, {'crossing': 0.0, 'merging': 0.0, 'bend': 0.0},
            )
            rec_wp[kind] = rec_wp.get(kind, 0.0) + contrib
        if by_leg_pair is not None:
            pair_key = (str(leg1_key), str(leg2_key))
            rec = by_leg_pair.setdefault(
                pair_key,
                {'crossing': 0.0, 'merging': 0.0, 'bend': 0.0,
                 'waypoint': shared_pt, 'angle_deg': float(crossing_angle)},
            )
            rec[kind] = rec.get(kind, 0.0) + contrib
        bucket = {'merging': by_cell_merging, 'bend': by_cell_bend}.get(kind, by_cell_crossing)
        if bucket is not None and contrib > 0.0:
            half = 0.5 * contrib
            bucket[f"{loa_i}_{type_j}"] = bucket.get(f"{loa_i}_{type_j}", 0.0) + half
            bucket[f"{loa_k}_{type_l}"] = bucket.get(f"{loa_k}_{type_l}", 0.0) + half

    # ------------------------------------------------------------------
    # Main orchestrator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_collision_report() -> dict[str, Any]:
        """Build the skeleton collision report and by-cell accumulators."""
        result: dict[str, float] = {
            'head_on': 0.0, 'overtaking': 0.0,
            'crossing': 0.0, 'merging': 0.0, 'bend': 0.0, 'total': 0.0,
        }
        by_cell: dict[str, dict[str, float]] = {
            'head_on': {}, 'overtaking': {}, 'bend': {},
            'crossing': {}, 'merging': {},
        }
        return result, by_cell

    def _process_single_leg(
        self,
        leg_key: str,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        pc_headon: float,
        pc_overtaking: float,
        length_intervals: list[dict],
        by_cell_head_on: dict[str, float],
        by_cell_overtaking: dict[str, float],
    ) -> dict[str, float]:
        """Compute head-on and overtaking frequencies for one leg.

        ``bend`` starts at 0; the waypoint loop adds half of every bend
        encounter to each of its two legs (:meth:`_split_bend_over_legs`).
        """
        leg_dirs = traffic_data.get(leg_key, {})
        seg_info = segment_data.get(leg_key, {})
        leg_length_m = float(seg_info.get('line_length', 1000.0))

        leg_head_on = self._calc_head_on_collisions(
            leg_dirs, seg_info, leg_length_m, pc_headon, length_intervals,
            by_cell=by_cell_head_on,
        )
        leg_overtaking = self._calc_overtaking_collisions(
            leg_dirs, seg_info, leg_length_m, pc_overtaking, length_intervals,
            by_cell=by_cell_overtaking,
        )
        return {'head_on': leg_head_on, 'overtaking': leg_overtaking, 'bend': 0.0}

    @staticmethod
    def _split_bend_over_legs(
        by_leg: dict[str, dict[str, float]],
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        """Give each leg half of every bend encounter it is part of.

        A bend is a waypoint encounter of two routes (``A -> B`` / ``B -> A``);
        the per-leg tables (breakdowns, Compare) still show a bend per leg.
        The halves add up to the bend total.
        """
        for (a, b), rec in crossing_by_pair.items():
            bend = float(rec.get('bend', 0.0) or 0.0)
            if bend <= 0.0:
                continue
            for leg in (a, b):
                by_leg.setdefault(leg, {'head_on': 0.0, 'overtaking': 0.0, 'bend': 0.0})
                by_leg[leg]['bend'] = by_leg[leg].get('bend', 0.0) + 0.5 * bend

    @staticmethod
    def _bend_by_pair(
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """``"A->B": {bend, waypoint}`` for the collision breakdown's Bend table."""
        out: dict[str, dict[str, Any]] = {}
        for (a, b), rec in crossing_by_pair.items():
            bend = float(rec.get('bend', 0.0) or 0.0)
            if bend <= 0.0:
                continue
            wp = rec.get('waypoint')
            out[f"{a}->{b}"] = {
                'bend': bend,
                'waypoint': f"{wp[0]:.6f} {wp[1]:.6f}" if wp is not None else '',
                'angle_deg': rec.get('angle_deg', 0.0),
            }
        return out

    @staticmethod
    def _serialize_crossing_by_pair(
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Convert tuple-keyed crossing_by_pair to JSON-friendly string keys."""
        return {
            f"{a}->{b}": {
                'crossing': rec.get('crossing', 0.0),
                'merging': rec.get('merging', 0.0),
                'bend': rec.get('bend', 0.0),
                'angle_deg': rec.get('angle_deg', 0.0),
                'waypoint': (
                    f"{rec['waypoint'][0]:.6f} {rec['waypoint'][1]:.6f}"
                    if rec.get('waypoint') is not None else ''
                ),
            }
            for (a, b), rec in crossing_by_pair.items()
        }

    def _update_collision_ui(self, result: dict[str, float]) -> None:
        """Push collision totals into the UI line-edit widgets.

        One widget per accident type.  ``LEPMergingCollision`` used to be
        fed ``result['bend']``; merging and bend are separate accident
        types since v0.14.0 and each has its own widget.
        """
        mw = self.p.main_widget
        for attr, key in (
            ('LEPHeadOnCollision', 'head_on'),
            ('LEPOvertakingCollision', 'overtaking'),
            ('LEPCrossingCollision', 'crossing'),
            ('LEPMergingCollision', 'merging'),
            ('LEPBendCollision', 'bend'),
        ):
            widget = getattr(mw, attr, None)
            if widget is None:
                continue
            try:
                widget.setText(f"{result.get(key, 0.0):.3e}")
            except Exception:  # nosec B110 B112
                pass

    def _finalize_collision_layers(
        self,
        segment_data: dict[str, Any],
    ) -> None:
        """Create result line/point layers from the finalized collision_report."""
        try:
            from geometries.result_layers import create_collision_layers
            self.collision_line_layer, self.collision_point_layer = (
                create_collision_layers(
                    self.collision_report, segment_data, add_to_project=False,
                )
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Failed to create ship-collision layers: {e}"
            )

    @staticmethod
    def _extract_pc_and_intervals(
        data: dict[str, Any], pc_vals: dict[str, Any],
    ) -> tuple[float, float, float, float, float, list[dict]]:
        """Return the five causation factors plus ``length_intervals``.

        Order: ``(headon, overtaking, crossing, merging, bend)``.
        """
        pc_headon = float(pc_vals.get('headon', 4.9e-5))
        pc_overtaking = float(pc_vals.get('overtaking', 1.1e-4))
        pc_crossing = float(pc_vals.get('crossing', 1.3e-4))
        pc_merging = float(pc_vals.get('merging', 1.3e-4))
        pc_bend = float(pc_vals.get('bend', 1.3e-4))
        length_intervals = data.get('ship_categories', {}).get('length_intervals', [])
        return (
            pc_headon, pc_overtaking, pc_crossing, pc_merging, pc_bend,
            length_intervals,
        )

    def _run_per_leg_loop(
        self,
        leg_keys: list[str],
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        pc_headon: float, pc_overtaking: float,
        length_intervals: list[dict],
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[str, float], dict[str, float],
    ]:
        """Run head-on / overtaking calculations for every leg."""
        by_leg: dict[str, dict[str, float]] = {}
        by_cell_ho: dict[str, float] = {}
        by_cell_ot: dict[str, float] = {}
        total_legs = len(leg_keys)
        for processed, leg_key in enumerate(leg_keys, 1):
            by_leg[leg_key] = self._process_single_leg(
                leg_key, traffic_data, segment_data,
                pc_headon, pc_overtaking, length_intervals,
                by_cell_ho, by_cell_ot,
            )
            self._report_progress(
                'spatial', processed / total_legs * 0.8,
                f"Processing leg {leg_key} ({processed}/{total_legs})..."
            )
        return by_leg, by_cell_ho, by_cell_ot

    def _run_crossing_and_merge_waypoints(
        self,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        leg_keys: list[str],
        pc_crossing: float,
        pc_merging: float,
        length_intervals: list[dict],
        junctions: dict[str, Junction] | None,
        by_waypoint: dict[tuple[float, float], dict[str, float]],
        pc_bend: float = 0.0,
        by_cell_bend: dict[str, float] | None = None,
    ) -> tuple[
        dict[str, float],
        dict[tuple[str, str], dict[str, Any]],
        dict[str, float],
        dict[str, float],
    ]:
        """Run crossing + merging + bend collisions and fold into by_waypoint."""
        by_cell_cr: dict[str, float] = {}
        by_cell_mg: dict[str, float] = {}
        crossing_by_wp: dict[tuple[float, float], dict[str, float]] = {}
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        totals = self._calc_crossing_collisions(
            traffic_data, segment_data, leg_keys,
            pc_crossing, pc_merging, length_intervals,
            by_waypoint=crossing_by_wp, by_leg_pair=crossing_by_pair,
            by_cell_crossing=by_cell_cr, by_cell_merging=by_cell_mg,
            junctions=junctions, pc_bend=pc_bend, by_cell_bend=by_cell_bend,
        )
        for pt, rec_wp in crossing_by_wp.items():
            rec = by_waypoint.setdefault(
                pt, {'crossing': 0.0, 'merging': 0.0, 'bend': 0.0},
            )
            for kind in ('crossing', 'merging', 'bend'):
                rec[kind] += float(rec_wp.get(kind, 0.0) or 0.0)
        return totals, crossing_by_pair, by_cell_cr, by_cell_mg

    def _assemble_collision_report(
        self,
        result: dict[str, float],
        by_leg: dict[str, dict[str, float]],
        by_waypoint: dict[tuple[float, float], dict[str, float]],
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]],
        by_cell_head_on: dict[str, float],
        by_cell_overtaking: dict[str, float],
        by_cell_bend: dict[str, float],
        by_cell_crossing: dict[str, float],
        by_cell_merging: dict[str, float],
        pc_headon: float, pc_overtaking: float,
        pc_crossing: float, pc_merging: float, pc_bend: float,
        segment_data: dict[str, Any],
        leg_keys: list[str],
    ) -> None:
        """Populate self.collision_report from all accumulators."""
        by_waypoint_ser = {
            f"{pt[0]:.6f} {pt[1]:.6f}": rec for pt, rec in by_waypoint.items()
        }
        self.collision_report = {
            'totals': result,
            'by_leg': by_leg,
            'by_waypoint': by_waypoint_ser,
            'by_leg_pair': self._serialize_crossing_by_pair(crossing_by_pair),
            'bend_by_pair': self._bend_by_pair(crossing_by_pair),
            'by_cell': {
                'head_on': by_cell_head_on, 'overtaking': by_cell_overtaking,
                'bend': by_cell_bend, 'crossing': by_cell_crossing,
                'merging': by_cell_merging,
            },
            'causation_factors': {
                'headon': pc_headon, 'overtaking': pc_overtaking,
                'crossing': pc_crossing, 'merging': pc_merging,
                'bend': pc_bend,
            },
            'min_encounter_angle_deg': math.degrees(self._min_route_angle_rad()),
        }

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def run_ship_collision_model(self, data: dict[str, Any]) -> dict[str, float]:
        """
        Run ship-ship collision calculations.

        Calculates head-on, overtaking, crossing, merging and bend collision
        frequencies based on the traffic data and leg geometries.

        Args:
            data: Dictionary containing traffic_data, segment_data, pc (causation factors),
                  and ship_categories

        Returns:
            dict with keys: 'head_on', 'overtaking', 'crossing',
            'merging', 'bend', 'total'
        """
        result, by_cell_init = self._init_collision_report()
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        junctions = self._coerce_junctions(data.get('junctions'))
        self._collision_type_scale = collision_type_scale(data, pc_vals)
        self._route_angle_limit_rad = min_angle_rad(pc_vals)

        if not traffic_data or not segment_data:
            self.ship_collision_prob = 0.0
            self.collision_report = {'totals': result, 'by_leg': {}, 'by_cell': by_cell_init}
            return result

        (
            pc_ho, pc_ot, pc_cr, pc_mg, pc_be, length_intervals,
        ) = self._extract_pc_and_intervals(data, pc_vals)
        leg_keys = list(traffic_data.keys())
        self._report_progress('spatial', 0.0, "Starting ship collision calculations...")

        by_leg, by_cell_ho, by_cell_ot = self._run_per_leg_loop(
            leg_keys, traffic_data, segment_data,
            pc_ho, pc_ot, length_intervals,
        )
        by_waypoint: dict[tuple[float, float], dict[str, float]] = {}
        by_cell_be: dict[str, float] = {}
        crossing_totals, crossing_by_pair, by_cell_cr, by_cell_mg = (
            self._run_crossing_and_merge_waypoints(
                traffic_data, segment_data, leg_keys, pc_cr, pc_mg,
                length_intervals, junctions, by_waypoint,
                pc_bend=pc_be, by_cell_bend=by_cell_be,
            )
        )
        self._split_bend_over_legs(by_leg, crossing_by_pair)

        self._fill_result_totals(result, by_leg, crossing_totals)
        self.ship_collision_prob = result['total']
        self._assemble_collision_report(
            result, by_leg, by_waypoint, crossing_by_pair,
            by_cell_ho, by_cell_ot, by_cell_be, by_cell_cr, by_cell_mg,
            pc_ho, pc_ot, pc_cr, pc_mg, pc_be, segment_data, leg_keys,
        )
        self._report_progress('layers', 1.0, "Ship collision calculation complete")
        self._finalize_collision_layers(segment_data)
        self._update_collision_ui(result)
        return result

    #: Accident types summed into ``result['total']``.
    TOTAL_KEYS: tuple[str, ...] = (
        'head_on', 'overtaking', 'crossing', 'merging', 'bend',
    )

    @classmethod
    def _fill_result_totals(
        cls,
        result: dict[str, float],
        by_leg: dict[str, dict[str, float]],
        crossing_totals: dict[str, float],
    ) -> None:
        """Write summed per-type frequencies into the result dict in-place."""
        result['head_on'] = sum(v['head_on'] for v in by_leg.values())
        result['overtaking'] = sum(v['overtaking'] for v in by_leg.values())
        result['crossing'] = float(crossing_totals.get('crossing', 0.0) or 0.0)
        result['merging'] = float(crossing_totals.get('merging', 0.0) or 0.0)
        result['bend'] = (float(crossing_totals['bend']) if 'bend' in crossing_totals
                          else sum(v.get('bend', 0.0) for v in by_leg.values()))
        result['total'] = sum(result[k] for k in cls.TOTAL_KEYS)
