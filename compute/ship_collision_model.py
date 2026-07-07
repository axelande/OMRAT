"""Ship-ship collision model mixin.

Extracted from ``run_calculations.py`` to improve maintainability.
The :class:`ShipCollisionModelMixin` provides :meth:`run_ship_collision_model`
decomposed into smaller, testable helper methods for each collision type
(head-on, overtaking, bend, crossing).
"""

from typing import Any

import numpy as np

from compute.basic_equations import (
    get_head_on_collision_candidates,
    get_overtaking_collision_candidates,
    get_crossing_collision_candidates,
    get_bend_collision_candidates,
)
from compute.data_preparation import get_distribution
from geometries.junctions import (
    Junction,
    deflection_deg,
    junction_id_for_point,
)


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
    def get_loa_midpoint(loa_idx: int, length_intervals: list[dict]) -> float:
        """Get midpoint of LOA category for length estimates."""
        if loa_idx < len(length_intervals):
            interval = length_intervals[loa_idx]
            try:
                min_val = float(interval.get('min', 50))
                max_val = float(interval.get('max', 100))
                return (min_val + max_val) / 2.0
            except (ValueError, TypeError):
                pass
        # Default midpoints for typical LOA categories
        default_midpoints = [25.0, 75.0, 150.0, 250.0, 350.0]
        return default_midpoints[loa_idx] if loa_idx < len(default_midpoints) else 150.0

    @staticmethod
    def estimate_beam(loa: float) -> float:
        """Estimate beam from LOA using typical ship ratios (L/B ~ 6-7)."""
        return loa / 6.5

    @staticmethod
    def _get_weighted_mu_sigma(
        seg_info: dict[str, Any], direction: int
    ) -> tuple[float, float]:
        """Extract weighted mean and std from segment lateral distributions.

        Returns (mu, sigma) in meters.  Raises ValueError when
        segment_data has no distribution information.
        """
        dists, wgts = get_distribution(seg_info, direction)

        w = np.array(wgts, dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError(
                f"No lateral distribution weights found for direction {direction} "
                f"in segment data (keys: {list(seg_info.keys())})"
            )
        w = w / w_sum

        # Weighted mean: E[X] = sum w_i * mu_i
        weighted_mu = float(sum(
            wi * dist.mean() for dist, wi in zip(dists, w) if wi > 0
        ))
        # Total variance: Var[X] = sum w_i*(sigma_i^2 + mu_i^2) - E[X]^2
        weighted_var = float(sum(
            wi * (dist.var() + dist.mean() ** 2)
            for dist, wi in zip(dists, w) if wi > 0
        )) - weighted_mu ** 2
        weighted_sigma = float(np.sqrt(max(weighted_var, 0.0)))

        if weighted_sigma < 1.0:
            raise ValueError(
                f"Lateral distribution sigma too small ({weighted_sigma:.4f} m) "
                f"for direction {direction} -- check distribution data"
            )

        return weighted_mu, weighted_sigma

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
        start: tuple[float, float], end: tuple[float, float]
    ) -> float:
        """Calculate bearing in degrees (0=N, CW) from start to end (lon/lat)."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
        return bearing

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
    def _junction_conflict_factor(
        junction: Junction | None,
        leg1_id: str,
        leg2_id: str,
    ) -> float:
        """Fraction of the leg-pair (q1, q2) flow that conflicts at the junction.

        Defined as ``1 - T[leg1->leg2] * T[leg2->leg1]``.

        Reasoning: when 100% of leg1 traffic continues onto leg2 *and*
        100% of leg2 traffic continues onto leg1, the only encounter at
        the junction is a head-on/overtaking exchange that the per-leg
        models already capture; so the crossing/merging adjustment is 0.
        Any divergence (some traffic going elsewhere) re-introduces a
        junction-level conflict, hence the multiplicative complement.
        Returns 1.0 when no junction is registered (legacy behaviour).
        """
        if junction is None or not junction.transitions:
            return 1.0
        l1, l2 = str(leg1_id), str(leg2_id)
        m12 = float(junction.transitions.get(l1, {}).get(l2, 0.0))
        m21 = float(junction.transitions.get(l2, {}).get(l1, 0.0))
        f = 1.0 - m12 * m21
        return max(0.0, min(1.0, f))

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
    ) -> float | None:
        """Bearing from the junction toward leg's far endpoint."""
        seg = (segment_data or {}).get(leg_id) or {}
        sp = ShipCollisionModelMixin._parse_point(seg.get('Start_Point', ''))
        ep = ShipCollisionModelMixin._parse_point(seg.get('End_Point', ''))
        if sp is None or ep is None:
            return None
        side = junction.legs.get(str(leg_id))
        far = ep if side == 'start' else sp if side == 'end' else None
        if far is None:
            return None
        return ShipCollisionModelMixin._calc_bearing(junction.point, far)

    # ------------------------------------------------------------------
    # Ship-property extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_speed_ms(speed_arr: Any, loa_i: int, type_j: int) -> float:
        """Extract speed in m/s for a ship cell from a nested speed array."""
        v_kts = 10.0
        if loa_i < len(speed_arr) and type_j < len(speed_arr[loa_i]):
            s_list = speed_arr[loa_i][type_j]
            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                v_kts = float(np.mean(s_list))
            elif isinstance(s_list, (int, float)):
                v_kts = float(s_list)
        return v_kts * 1852.0 / 3600.0

    @staticmethod
    def _extract_beam(
        beam_arr: Any, loa_i: int, type_j: int,
        default_beam: float,
    ) -> float:
        """Extract beam (m) for a ship cell; falls back to ``default_beam``."""
        if loa_i < len(beam_arr) and type_j < len(beam_arr[loa_i]):
            b_list = beam_arr[loa_i][type_j]
            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                b_raw = float(np.mean(b_list))
                if np.isfinite(b_raw):
                    return b_raw
            elif isinstance(b_list, (int, float)) and np.isfinite(float(b_list)):
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
        mu1_lat, sigma1_lat = self._get_weighted_mu_sigma(seg_info, 0)
        mu2_lat, sigma2_lat = self._get_weighted_mu_sigma(seg_info, 1)

        return self._head_on_outer_loop(
            arrays1, arrays2,
            mu1_lat, sigma1_lat, mu2_lat, sigma2_lat,
            leg_length_m, pc_headon, length_intervals, by_cell,
        )

    @staticmethod
    def _dir_arrays(dir_data: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Extract (freq, speed, beam) numpy arrays from a direction data dict."""
        return (
            np.array(dir_data.get('Frequency (ships/year)', [])),
            np.array(dir_data.get('Speed (knots)', [])),
            np.array(dir_data.get('Ship Beam (meters)', [])),
        )

    def _head_on_outer_loop(
        self,
        arrays1: tuple[Any, Any, Any],
        arrays2: tuple[Any, Any, Any],
        mu1_lat: float, sigma1_lat: float,
        mu2_lat: float, sigma2_lat: float,
        leg_length_m: float, pc_headon: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None,
    ) -> float:
        """Outer loop over dir1 ship cells for head-on collisions."""
        freq1, speed1, beam1 = arrays1
        total = 0.0
        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                if q1 <= 0 or not np.isfinite(q1):
                    continue
                v1_ms = self._extract_speed_ms(speed1, loa_i, type_j)
                b1 = self._extract_beam(
                    beam1, loa_i, type_j,
                    self.estimate_beam(self.get_loa_midpoint(loa_i, length_intervals)),
                )
                total += self._head_on_inner_loop(
                    arrays2[0], arrays2[1], arrays2[2],
                    q1, v1_ms, b1, loa_i, type_j,
                    mu1_lat, sigma1_lat, mu2_lat, sigma2_lat,
                    leg_length_m, pc_headon, length_intervals, by_cell,
                )
        return total

    def _head_on_inner_loop(
        self,
        freq2: Any, speed2: Any, beam2: Any,
        q1: float, v1_ms: float, b1: float,
        loa_i: int, type_j: int,
        mu1_lat: float, sigma1_lat: float,
        mu2_lat: float, sigma2_lat: float,
        leg_length_m: float, pc_headon: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None,
    ) -> float:
        """Inner loop over dir2 ship categories for head-on collisions."""
        total = 0.0
        for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
            for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                if q2 <= 0 or not np.isfinite(q2):
                    continue
                v2_ms = self._extract_speed_ms(speed2, loa_k, type_l)
                b2 = self._extract_beam(
                    beam2, loa_k, type_l,
                    self.estimate_beam(self.get_loa_midpoint(loa_k, length_intervals)),
                )
                n_g_headon = get_head_on_collision_candidates(
                    Q1=q1, Q2=q2,
                    V1=v1_ms, V2=v2_ms,
                    mu1=mu1_lat, mu2=mu2_lat,
                    sigma1=sigma1_lat, sigma2=sigma2_lat,
                    B1=b1, B2=b2,
                    L_w=leg_length_m
                )
                contrib = n_g_headon * pc_headon
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

        for dir_idx, dir_key in enumerate(dir_keys):
            freq, speed, beam = self._dir_arrays(leg_dirs.get(dir_key, {}))
            mu_ot, sigma_ot = self._get_weighted_mu_sigma(seg_info, dir_idx)
            ship_cells = self._collect_ship_cells(freq, speed, beam, length_intervals)

            for i, (loa_i, type_i, q_fast, v_fast, b_fast) in enumerate(ship_cells):
                for j, (loa_j, type_j, q_slow, v_slow, b_slow) in enumerate(ship_cells):
                    if i == j or v_fast <= v_slow:
                        continue

                    n_g_overtaking = get_overtaking_collision_candidates(
                        Q_fast=q_fast, Q_slow=q_slow,
                        V_fast=v_fast, V_slow=v_slow,
                        mu_fast=mu_ot, mu_slow=mu_ot,
                        sigma_fast=sigma_ot, sigma_slow=sigma_ot,
                        B_fast=b_fast, B_slow=b_slow,
                        L_w=leg_length_m
                    )
                    contrib = n_g_overtaking * pc_overtaking
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
                    self.estimate_beam(self.get_loa_midpoint(loa_i, length_intervals)),
                )
                ship_cells.append((loa_i, type_j, q, v_ms, b))
        return ship_cells

    def _calc_bend_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        pc_bend: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None = None,
        junctions: dict[str, Junction] | None = None,
        leg_id: str | None = None,
        segment_data: dict[str, Any] | None = None,
    ) -> float:
        """Calculate bend collision frequency for a single leg.

        Two paths:

        * **Legacy** — when no junction is registered at the leg's end,
          uses the optional ``bend_angle`` field on ``seg_info`` (default
          0.0) and a single-bend calculation.

        * **Junction-aware** — when a junction registry is provided and
          a junction is registered at the leg's end, iterates over the
          junction's transition rows for this leg and accumulates bend
          frequency for each (this_leg -> other_leg) continuation whose
          deflection exceeds 5 degrees.  Per-pair traffic is the
          leg-total flow times the matrix share for that continuation.

        If ``by_cell`` is provided, the bend total is distributed across the
        cells contributing traffic on this leg in proportion to their share
        of total frequency.
        """
        avg_freq, avg_length, avg_beam, cell_freqs = self._aggregate_bend_flows(
            leg_dirs, length_intervals,
        )

        if avg_freq <= 0:
            return 0.0

        p_no_turn = 0.01
        end_pt = self._parse_point(seg_info.get('End_Point', ''))
        junction = self._junction_at_point(junctions, end_pt)

        if (
            junction is not None
            and leg_id is not None
            and segment_data is not None
            and junction.transitions
        ):
            leg_bend = self._bend_junction_path(
                junction, leg_id, segment_data,
                avg_freq, avg_length, avg_beam, p_no_turn, pc_bend,
            )
            if by_cell is not None and leg_bend > 0.0:
                for cell_key, cf in cell_freqs.items():
                    share_cell = leg_bend * (cf / avg_freq)
                    by_cell[cell_key] = by_cell.get(cell_key, 0.0) + share_cell
            return leg_bend

        return self._bend_legacy_path(
            seg_info, avg_freq, avg_length, avg_beam, p_no_turn,
            pc_bend, cell_freqs, by_cell,
        )

    def _aggregate_bend_flows(
        self,
        leg_dirs: dict[str, Any],
        length_intervals: list[dict],
    ) -> tuple[float, float, float, dict[str, float]]:
        """Aggregate per-leg traffic into scalar avg_freq, avg_length, avg_beam and per-cell freqs."""
        avg_freq = 0.0
        avg_length = 150.0
        avg_beam = 25.0
        count = 0
        cell_freqs: dict[str, float] = {}
        for dir_key in leg_dirs:
            freq, _, _ = self._dir_arrays(leg_dirs.get(dir_key, {}))
            for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
                for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                    q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                    if q > 0:
                        avg_freq += q
                        loa_mid = self.get_loa_midpoint(loa_i, length_intervals)
                        avg_length = (avg_length * count + loa_mid) / (count + 1)
                        avg_beam = (avg_beam * count + self.estimate_beam(loa_mid)) / (count + 1)
                        count += 1
                        cell_key = f"{loa_i}_{type_j}"
                        cell_freqs[cell_key] = cell_freqs.get(cell_key, 0.0) + q
        return avg_freq, avg_length, avg_beam, cell_freqs

    def _bend_junction_path(
        self,
        junction: Junction,
        leg_id: str,
        segment_data: dict[str, Any],
        avg_freq: float,
        avg_length: float,
        avg_beam: float,
        p_no_turn: float,
        pc_bend: float,
    ) -> float:
        """Compute bend frequency via the junction transition matrix."""
        leg_bend = 0.0
        row = junction.transitions.get(str(leg_id), {})
        in_bearing = self._junction_outward_bearing(junction, str(leg_id), segment_data)
        if in_bearing is None or not row:
            return leg_bend
        for out_leg, share in row.items():
            if share <= 0 or out_leg == str(leg_id):
                continue
            out_bearing = self._junction_outward_bearing(junction, str(out_leg), segment_data)
            if out_bearing is None:
                continue
            bend_angle_deg = deflection_deg(in_bearing, out_bearing)
            if bend_angle_deg <= 5.0:
                continue
            bend_angle_rad = bend_angle_deg * np.pi / 180.0
            q_pair = avg_freq * float(share)
            n_g_bend = get_bend_collision_candidates(
                Q=q_pair, P_no_turn=p_no_turn,
                L=avg_length, B=avg_beam, theta=bend_angle_rad,
            )
            leg_bend += n_g_bend * pc_bend
        return leg_bend

    def _bend_legacy_path(
        self,
        seg_info: dict[str, Any],
        avg_freq: float,
        avg_length: float,
        avg_beam: float,
        p_no_turn: float,
        pc_bend: float,
        cell_freqs: dict[str, float],
        by_cell: dict[str, float] | None,
    ) -> float:
        """Compute bend frequency using the static ``bend_angle`` field."""
        bend_angle_deg = float(seg_info.get('bend_angle', 0.0))
        if bend_angle_deg <= 5.0:
            return 0.0
        bend_angle_rad = bend_angle_deg * np.pi / 180.0
        n_g_bend = get_bend_collision_candidates(
            Q=avg_freq, P_no_turn=p_no_turn,
            L=avg_length, B=avg_beam, theta=bend_angle_rad,
        )
        leg_bend = n_g_bend * pc_bend
        if by_cell is not None and leg_bend > 0.0 and avg_freq > 0.0:
            for cell_key, cf in cell_freqs.items():
                share = leg_bend * (cf / avg_freq)
                by_cell[cell_key] = by_cell.get(cell_key, 0.0) + share
        return leg_bend

    def _calc_crossing_collisions(
        self,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        leg_keys: list[str],
        pc_crossing: float,
        length_intervals: list[dict],
        by_waypoint: dict[tuple[float, float], float] | None = None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None = None,
        by_cell_crossing: dict[str, float] | None = None,
        by_cell_merging: dict[str, float] | None = None,
        junctions: dict[str, Junction] | None = None,
    ) -> float:
        """Calculate crossing collision frequency between all leg pairs.

        Crossing collisions occur where two different legs share a waypoint
        and meet at a non-trivial angle.

        Returns the total crossing collision frequency.

        If ``by_waypoint`` is provided, per-waypoint contributions are
        accumulated into it (key = ``(lon, lat)``).  This lets the result
        layer place a point at each shared waypoint with the summed
        crossing probability for legs meeting there.
        """
        total_crossing = 0.0
        total_legs = len(leg_keys)
        crossing_pairs_processed = 0
        total_pairs = total_legs * (total_legs - 1) // 2

        self._report_progress('cascade', 0.0, "Calculating crossing collisions...")

        for i, leg1_key in enumerate(leg_keys):
            for j, leg2_key in enumerate(leg_keys):
                if j <= i:
                    continue

                contrib = self._crossing_pair_contribution(
                    leg1_key, leg2_key,
                    traffic_data, segment_data,
                    pc_crossing, length_intervals,
                    by_waypoint, by_leg_pair,
                    by_cell_crossing, by_cell_merging,
                    junctions,
                )
                total_crossing += contrib

                crossing_pairs_processed += 1
                if total_pairs > 0:
                    self._report_progress(
                        'cascade',
                        crossing_pairs_processed / total_pairs,
                        f"Processing crossing pair {leg1_key}-{leg2_key}..."
                    )

        return total_crossing

    def _crossing_pair_contribution(
        self,
        leg1_key: str,
        leg2_key: str,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        pc_crossing: float,
        length_intervals: list[dict],
        by_waypoint: dict[tuple[float, float], float] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        junctions: dict[str, Junction] | None,
    ) -> float:
        """Compute crossing contribution for one leg pair; returns 0.0 when not applicable."""
        seg1 = segment_data.get(leg1_key, {})
        seg2 = segment_data.get(leg2_key, {})

        shared_pt, conflict_factor, crossing_angle_rad = self._crossing_geometry(
            leg1_key, leg2_key, seg1, seg2, junctions,
        )
        if shared_pt is None or crossing_angle_rad is None:
            return 0.0

        crossing_angle = crossing_angle_rad * 180.0 / np.pi
        kind = 'merging' if crossing_angle <= 30.0 else 'crossing'

        leg1_dirs = traffic_data.get(leg1_key, {})
        leg2_dirs = traffic_data.get(leg2_key, {})

        pair_total = self._sum_crossing_pair_traffic(
            leg1_dirs, leg2_dirs,
            crossing_angle_rad, conflict_factor, pc_crossing, length_intervals,
            shared_pt, kind,
            by_waypoint, by_leg_pair,
            by_cell_crossing, by_cell_merging,
            leg1_key, leg2_key,
        )
        return pair_total

    def _crossing_geometry(
        self,
        leg1_key: str,
        leg2_key: str,
        seg1: dict[str, Any],
        seg2: dict[str, Any],
        junctions: dict[str, Junction] | None,
    ) -> tuple[tuple[float, float] | None, float, float | None]:
        """Check if two legs share a waypoint and compute crossing angle.

        Returns ``(shared_pt, conflict_factor, crossing_angle_rad)`` or
        ``(None, 1.0, None)`` when the pair is not a valid crossing.
        """
        s1_start = self._parse_point(seg1.get('Start_Point', ''))
        s1_end = self._parse_point(seg1.get('End_Point', ''))
        s2_start = self._parse_point(seg2.get('Start_Point', ''))
        s2_end = self._parse_point(seg2.get('End_Point', ''))

        shared_pt: tuple[float, float] | None = None
        if self._points_match(s1_start, s2_start) or self._points_match(s1_start, s2_end):
            shared_pt = s1_start
        elif self._points_match(s1_end, s2_start) or self._points_match(s1_end, s2_end):
            shared_pt = s1_end
        if shared_pt is None:
            return None, 1.0, None

        junction = self._junction_at_point(junctions, shared_pt)
        conflict_factor = self._junction_conflict_factor(junction, leg1_key, leg2_key)
        if conflict_factor <= 0.0:
            return None, 0.0, None

        bearing1 = self._seg_bearing(seg1, s1_start, s1_end)
        bearing2 = self._seg_bearing(seg2, s2_start, s2_end)
        if bearing1 is None or bearing2 is None:
            return None, 1.0, None

        crossing_angle = abs(bearing1 - bearing2) % 180.0
        if crossing_angle > 90:
            crossing_angle = 180 - crossing_angle
        crossing_angle_rad = crossing_angle * np.pi / 180.0

        if crossing_angle_rad < 0.1:
            return None, 1.0, None

        return shared_pt, conflict_factor, crossing_angle_rad

    def _seg_bearing(
        self,
        seg: dict[str, Any],
        start_pt: tuple[float, float] | None,
        end_pt: tuple[float, float] | None,
    ) -> float | None:
        """Return bearing for a segment, preferring stored value over computed."""
        if 'bearing' in seg and seg['bearing']:
            return float(seg['bearing'])
        if start_pt and end_pt:
            return self._calc_bearing(start_pt, end_pt)
        return None

    def _sum_crossing_pair_traffic(
        self,
        leg1_dirs: dict[str, Any],
        leg2_dirs: dict[str, Any],
        crossing_angle_rad: float,
        conflict_factor: float,
        pc_crossing: float,
        length_intervals: list[dict],
        shared_pt: tuple[float, float],
        kind: str,
        by_waypoint: dict[tuple[float, float], float] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        leg1_key: str,
        leg2_key: str,
    ) -> float:
        """Sum crossing contributions over all direction x ship-cell combinations."""
        crossing_angle = crossing_angle_rad * 180.0 / np.pi
        pair_total = 0.0
        for dir1_key in leg1_dirs:
            dir1_data = leg1_dirs.get(dir1_key, {})
            freq1 = np.array(dir1_data.get('Frequency (ships/year)', []))
            speed1 = np.array(dir1_data.get('Speed (knots)', []))
            beam1_arr = np.array(dir1_data.get('Ship Beam (meters)', []))

            for dir2_key in leg2_dirs:
                dir2_data = leg2_dirs.get(dir2_key, {})
                freq2 = np.array(dir2_data.get('Frequency (ships/year)', []))
                speed2 = np.array(dir2_data.get('Speed (knots)', []))
                beam2_arr = np.array(dir2_data.get('Ship Beam (meters)', []))

                pair_total += self._crossing_ship_cell_loop(
                    freq1, speed1, beam1_arr,
                    freq2, speed2, beam2_arr,
                    crossing_angle_rad, crossing_angle, conflict_factor,
                    pc_crossing, length_intervals,
                    shared_pt, kind,
                    by_waypoint, by_leg_pair,
                    by_cell_crossing, by_cell_merging,
                    leg1_key, leg2_key,
                )
        return pair_total

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
        by_waypoint: dict[tuple[float, float], float] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        leg1_key: str,
        leg2_key: str,
    ) -> float:
        """Iterate over all (leg1 cell) x (leg2 cell) combinations for one direction pair."""
        total = 0.0
        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                if q1 <= 0 or not np.isfinite(q1):
                    continue
                v1_ms = self._extract_speed_ms(speed1, loa_i, type_j)
                l1 = self.get_loa_midpoint(loa_i, length_intervals)
                b1 = self._extract_beam(beam1_arr, loa_i, type_j, self.estimate_beam(l1))

                for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                    for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                        q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                        if q2 <= 0 or not np.isfinite(q2):
                            continue
                        v2_ms = self._extract_speed_ms(speed2, loa_k, type_l)
                        l2 = self.get_loa_midpoint(loa_k, length_intervals)
                        b2 = self._extract_beam(beam2_arr, loa_k, type_l, self.estimate_beam(l2))

                        n_g_crossing = get_crossing_collision_candidates(
                            Q1=q1, Q2=q2, V1=v1_ms, V2=v2_ms,
                            L1=l1, L2=l2, B1=b1, B2=b2,
                            theta=crossing_angle_rad
                        )
                        contrib = n_g_crossing * pc_crossing * conflict_factor
                        total += contrib
                        self._accumulate_crossing_contrib(
                            contrib, kind, shared_pt,
                            loa_i, type_j, loa_k, type_l,
                            crossing_angle,
                            by_waypoint, by_leg_pair,
                            by_cell_crossing, by_cell_merging,
                            leg1_key, leg2_key,
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
        by_waypoint: dict[tuple[float, float], float] | None,
        by_leg_pair: dict[tuple[str, str], dict[str, Any]] | None,
        by_cell_crossing: dict[str, float] | None,
        by_cell_merging: dict[str, float] | None,
        leg1_key: str,
        leg2_key: str,
    ) -> None:
        """Update all accumulator dicts for one crossing cell-pair contribution."""
        if by_waypoint is not None:
            by_waypoint[shared_pt] = by_waypoint.get(shared_pt, 0.0) + contrib
        if by_leg_pair is not None:
            pair_key = (str(leg1_key), str(leg2_key))
            rec = by_leg_pair.setdefault(
                pair_key,
                {'crossing': 0.0, 'merging': 0.0, 'waypoint': shared_pt, 'angle_deg': float(crossing_angle)},
            )
            rec[kind] += contrib
        bucket = by_cell_merging if kind == 'merging' else by_cell_crossing
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
            'crossing': 0.0, 'bend': 0.0, 'total': 0.0,
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
        pc_bend: float,
        length_intervals: list[dict],
        by_cell_head_on: dict[str, float],
        by_cell_overtaking: dict[str, float],
        by_cell_bend: dict[str, float],
        junctions: dict[str, Junction] | None,
        by_waypoint: dict[tuple[float, float], dict[str, float]],
    ) -> dict[str, float]:
        """Compute head-on, overtaking, and bend frequencies for one leg."""
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
        leg_bend = self._calc_bend_collisions(
            leg_dirs, seg_info, pc_bend, length_intervals,
            by_cell=by_cell_bend,
            junctions=junctions, leg_id=leg_key, segment_data=segment_data,
        )

        if leg_bend > 0.0:
            end_pt = self._parse_point(seg_info.get('End_Point', ''))
            if end_pt is not None:
                rec = by_waypoint.setdefault(end_pt, {'crossing': 0.0, 'bend': 0.0})
                rec['bend'] += leg_bend

        return {'head_on': leg_head_on, 'overtaking': leg_overtaking, 'bend': leg_bend}

    def _build_bend_by_pair(
        self,
        by_leg: dict[str, dict[str, float]],
        segment_data: dict[str, Any],
        leg_keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Build the bend-by-pair label -> {bend, waypoint} mapping."""
        bend_by_pair: dict[str, dict[str, Any]] = {}
        for leg_key, vals in by_leg.items():
            bend = float(vals.get('bend', 0.0) or 0.0)
            if bend <= 0.0:
                continue
            seg = segment_data.get(leg_key, {})
            end_pt = self._parse_point(seg.get('End_Point', ''))
            next_leg_id = self._find_next_leg(end_pt, leg_key, leg_keys, segment_data)
            label = f"{leg_key}->{next_leg_id}" if next_leg_id else f"{leg_key}->"
            wp_text = f"{end_pt[0]:.6f} {end_pt[1]:.6f}" if end_pt is not None else ''
            bend_by_pair[label] = {'bend': bend, 'waypoint': wp_text}
        return bend_by_pair

    def _find_next_leg(
        self,
        end_pt: tuple[float, float] | None,
        leg_key: str,
        leg_keys: list[str],
        segment_data: dict[str, Any],
    ) -> str:
        """Return the leg_key whose Start_Point matches end_pt, or ''."""
        if end_pt is None:
            return ''
        for other_key in leg_keys:
            if other_key == leg_key:
                continue
            other_seg = segment_data.get(other_key, {})
            other_start = self._parse_point(other_seg.get('Start_Point', ''))
            if other_start is not None and self._points_match(end_pt, other_start):
                return str(other_key)
        return ''

    @staticmethod
    def _serialize_crossing_by_pair(
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Convert tuple-keyed crossing_by_pair to JSON-friendly string keys."""
        return {
            f"{a}->{b}": {
                'crossing': rec.get('crossing', 0.0),
                'merging': rec.get('merging', 0.0),
                'angle_deg': rec.get('angle_deg', 0.0),
                'waypoint': (
                    f"{rec['waypoint'][0]:.6f} {rec['waypoint'][1]:.6f}"
                    if rec.get('waypoint') is not None else ''
                ),
            }
            for (a, b), rec in crossing_by_pair.items()
        }

    def _update_collision_ui(self, result: dict[str, float]) -> None:
        """Push collision totals into the UI line-edit widgets."""
        try:
            self.p.main_widget.LEPHeadOnCollision.setText(f"{result['head_on']:.3e}")
            self.p.main_widget.LEPOvertakingCollision.setText(f"{result['overtaking']:.3e}")
            self.p.main_widget.LEPCrossingCollision.setText(f"{result['crossing']:.3e}")
            self.p.main_widget.LEPMergingCollision.setText(f"{result['bend']:.3e}")
        except Exception:
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
    ) -> tuple[float, float, float, float, list[dict]]:
        """Return (pc_headon, pc_overtaking, pc_crossing, pc_bend, length_intervals)."""
        pc_headon = float(pc_vals.get('headon', 4.9e-5))
        pc_overtaking = float(pc_vals.get('overtaking', 1.1e-4))
        pc_crossing = float(pc_vals.get('crossing', 1.3e-4))
        pc_bend = float(pc_vals.get('bend', 1.3e-4))
        length_intervals = data.get('ship_categories', {}).get('length_intervals', [])
        return pc_headon, pc_overtaking, pc_crossing, pc_bend, length_intervals

    def _run_per_leg_loop(
        self,
        leg_keys: list[str],
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        pc_headon: float, pc_overtaking: float, pc_bend: float,
        length_intervals: list[dict],
        junctions: dict[str, Junction] | None,
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[str, float], dict[str, float], dict[str, float],
        dict[tuple[float, float], dict[str, float]],
    ]:
        """Run head-on / overtaking / bend calculations for every leg."""
        by_leg: dict[str, dict[str, float]] = {}
        by_cell_ho: dict[str, float] = {}
        by_cell_ot: dict[str, float] = {}
        by_cell_be: dict[str, float] = {}
        by_waypoint: dict[tuple[float, float], dict[str, float]] = {}
        total_legs = len(leg_keys)
        for processed, leg_key in enumerate(leg_keys, 1):
            by_leg[leg_key] = self._process_single_leg(
                leg_key, traffic_data, segment_data,
                pc_headon, pc_overtaking, pc_bend, length_intervals,
                by_cell_ho, by_cell_ot, by_cell_be,
                junctions, by_waypoint,
            )
            self._report_progress(
                'spatial', processed / total_legs * 0.8,
                f"Processing leg {leg_key} ({processed}/{total_legs})..."
            )
        return by_leg, by_cell_ho, by_cell_ot, by_cell_be, by_waypoint

    def _run_crossing_and_merge_waypoints(
        self,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        leg_keys: list[str],
        pc_crossing: float,
        length_intervals: list[dict],
        junctions: dict[str, Junction] | None,
        by_waypoint: dict[tuple[float, float], dict[str, float]],
    ) -> tuple[
        float,
        dict[tuple[str, str], dict[str, Any]],
        dict[str, float],
        dict[str, float],
    ]:
        """Run crossing collisions and merge results into by_waypoint."""
        by_cell_cr: dict[str, float] = {}
        by_cell_mg: dict[str, float] = {}
        crossing_by_wp: dict[tuple[float, float], float] = {}
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        total_crossing = self._calc_crossing_collisions(
            traffic_data, segment_data, leg_keys, pc_crossing, length_intervals,
            by_waypoint=crossing_by_wp, by_leg_pair=crossing_by_pair,
            by_cell_crossing=by_cell_cr, by_cell_merging=by_cell_mg,
            junctions=junctions,
        )
        for pt, contrib in crossing_by_wp.items():
            rec = by_waypoint.setdefault(pt, {'crossing': 0.0, 'bend': 0.0})
            rec['crossing'] += contrib
        return total_crossing, crossing_by_pair, by_cell_cr, by_cell_mg

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
        pc_crossing: float, pc_bend: float,
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
            'bend_by_pair': self._build_bend_by_pair(by_leg, segment_data, leg_keys),
            'by_cell': {
                'head_on': by_cell_head_on, 'overtaking': by_cell_overtaking,
                'bend': by_cell_bend, 'crossing': by_cell_crossing,
                'merging': by_cell_merging,
            },
            'causation_factors': {
                'headon': pc_headon, 'overtaking': pc_overtaking,
                'crossing': pc_crossing, 'bend': pc_bend,
            },
        }

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def run_ship_collision_model(self, data: dict[str, Any]) -> dict[str, float]:
        """
        Run ship-ship collision calculations.

        Calculates head-on, overtaking, crossing, and bend collision frequencies
        based on the traffic data and leg geometries.

        Args:
            data: Dictionary containing traffic_data, segment_data, pc (causation factors),
                  and ship_categories

        Returns:
            dict with keys: 'head_on', 'overtaking', 'crossing', 'bend', 'total'
        """
        result, by_cell_init = self._init_collision_report()
        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}
        junctions = self._coerce_junctions(data.get('junctions'))

        if not traffic_data or not segment_data:
            self.ship_collision_prob = 0.0
            self.collision_report = {'totals': result, 'by_leg': {}, 'by_cell': by_cell_init}
            return result

        pc_ho, pc_ot, pc_cr, pc_be, length_intervals = self._extract_pc_and_intervals(
            data, pc_vals,
        )
        leg_keys = list(traffic_data.keys())
        self._report_progress('spatial', 0.0, "Starting ship collision calculations...")

        by_leg, by_cell_ho, by_cell_ot, by_cell_be, by_waypoint = self._run_per_leg_loop(
            leg_keys, traffic_data, segment_data,
            pc_ho, pc_ot, pc_be, length_intervals, junctions,
        )
        total_crossing, crossing_by_pair, by_cell_cr, by_cell_mg = (
            self._run_crossing_and_merge_waypoints(
                traffic_data, segment_data, leg_keys, pc_cr,
                length_intervals, junctions, by_waypoint,
            )
        )

        self._fill_result_totals(result, by_leg, total_crossing)
        self.ship_collision_prob = result['total']
        self._assemble_collision_report(
            result, by_leg, by_waypoint, crossing_by_pair,
            by_cell_ho, by_cell_ot, by_cell_be, by_cell_cr, by_cell_mg,
            pc_ho, pc_ot, pc_cr, pc_be, segment_data, leg_keys,
        )
        self._report_progress('layers', 1.0, "Ship collision calculation complete")
        self._finalize_collision_layers(segment_data)
        self._update_collision_ui(result)
        return result

    @staticmethod
    def _fill_result_totals(
        result: dict[str, float],
        by_leg: dict[str, dict[str, float]],
        total_crossing: float,
    ) -> None:
        """Write summed per-type frequencies into the result dict in-place."""
        result['head_on'] = sum(v['head_on'] for v in by_leg.values())
        result['overtaking'] = sum(v['overtaking'] for v in by_leg.values())
        result['crossing'] = total_crossing
        result['bend'] = sum(v['bend'] for v in by_leg.values())
        result['total'] = sum(result[k] for k in ('head_on', 'overtaking', 'crossing', 'bend'))
