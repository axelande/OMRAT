"""Ship-ship collision model mixin.

Extracted from ``run_calculations.py`` to improve maintainability.
The :class:`ShipCollisionModelMixin` provides :meth:`run_ship_collision_model`
decomposed into smaller, testable helper methods for each collision type
(head-on, overtaking, bend, crossing).
"""

from typing import Any

import numpy as np
from numpy import exp  # noqa: F401 – kept for parity with original imports

from compute.basic_equations import (
    get_head_on_collision_candidates,
    get_overtaking_collision_candidates,
    get_crossing_collision_candidates,
    get_bend_collision_candidates,
)
from compute.data_preparation import get_distribution


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
    # Collision-type sub-methods
    # ------------------------------------------------------------------

    def _calc_head_on_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        leg_length_m: float,
        pc_headon: float,
        length_intervals: list[dict],
    ) -> float:
        """Calculate head-on collision frequency for a single leg.

        Head-on collisions arise from ships travelling in opposite directions
        on the same leg.

        Returns the total head-on collision frequency for this leg.
        """
        leg_head_on = 0.0
        dir_keys = list(leg_dirs.keys())

        if len(dir_keys) < 2:
            return leg_head_on

        dir1, dir2 = dir_keys[0], dir_keys[1]
        data1 = leg_dirs.get(dir1, {})
        data2 = leg_dirs.get(dir2, {})

        freq1 = np.array(data1.get('Frequency (ships/year)', []))
        freq2 = np.array(data2.get('Frequency (ships/year)', []))
        speed1 = np.array(data1.get('Speed (knots)', []))
        speed2 = np.array(data2.get('Speed (knots)', []))
        beam1 = np.array(data1.get('Ship Beam (meters)', []))
        beam2 = np.array(data2.get('Ship Beam (meters)', []))

        # Get lateral distribution parameters from loaded data
        mu1_lat, sigma1_lat = self._get_weighted_mu_sigma(seg_info, 0)
        mu2_lat, sigma2_lat = self._get_weighted_mu_sigma(seg_info, 1)

        # Iterate ship categories (LOA x Type)
        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                if q1 <= 0 or not np.isfinite(q1):
                    continue

                # Get speed for dir1 ships
                v1_kts = 10.0  # Default
                if loa_i < len(speed1) and type_j < len(speed1[loa_i]):
                    s_list = speed1[loa_i][type_j]
                    if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                        v1_kts = float(np.mean(s_list))
                    elif isinstance(s_list, (int, float)):
                        v1_kts = float(s_list)
                v1_ms = v1_kts * 1852.0 / 3600.0  # Convert knots to m/s

                # Get beam for dir1 ships
                b1 = self.estimate_beam(self.get_loa_midpoint(loa_i, length_intervals))
                if loa_i < len(beam1) and type_j < len(beam1[loa_i]):
                    b_list = beam1[loa_i][type_j]
                    if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                        b1 = float(np.mean(b_list))
                    elif isinstance(b_list, (int, float)):
                        b1 = float(b_list)

                # Iterate over dir2 ship categories
                for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                    for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                        q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                        if q2 <= 0 or not np.isfinite(q2):
                            continue

                        # Get speed for dir2 ships
                        v2_kts = 10.0
                        if loa_k < len(speed2) and type_l < len(speed2[loa_k]):
                            s_list = speed2[loa_k][type_l]
                            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                v2_kts = float(np.mean(s_list))
                            elif isinstance(s_list, (int, float)):
                                v2_kts = float(s_list)
                        v2_ms = v2_kts * 1852.0 / 3600.0

                        # Get beam for dir2 ships
                        b2 = self.estimate_beam(self.get_loa_midpoint(loa_k, length_intervals))
                        if loa_k < len(beam2) and type_l < len(beam2[loa_k]):
                            b_list = beam2[loa_k][type_l]
                            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                b2 = float(np.mean(b_list))
                            elif isinstance(b_list, (int, float)):
                                b2 = float(b_list)

                        # Calculate head-on collision candidates using loaded lateral distributions
                        n_g_headon = get_head_on_collision_candidates(
                            Q1=q1, Q2=q2,
                            V1=v1_ms, V2=v2_ms,
                            mu1=mu1_lat, mu2=mu2_lat,
                            sigma1=sigma1_lat, sigma2=sigma2_lat,
                            B1=b1, B2=b2,
                            L_w=leg_length_m
                        )
                        leg_head_on += n_g_headon * pc_headon

        return leg_head_on

    def _calc_overtaking_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        leg_length_m: float,
        pc_overtaking: float,
        length_intervals: list[dict],
    ) -> float:
        """Calculate overtaking collision frequency for a single leg.

        Overtaking collisions occur between ships travelling in the same
        direction at different speeds.

        Returns the total overtaking collision frequency for this leg.
        """
        leg_overtaking = 0.0
        dir_keys = list(leg_dirs.keys())

        for dir_idx, dir_key in enumerate(dir_keys):
            dir_data = leg_dirs.get(dir_key, {})
            freq = np.array(dir_data.get('Frequency (ships/year)', []))
            speed = np.array(dir_data.get('Speed (knots)', []))
            beam = np.array(dir_data.get('Ship Beam (meters)', []))

            # Get lateral distribution for this direction from loaded data
            mu_ot, sigma_ot = self._get_weighted_mu_sigma(seg_info, dir_idx)

            # Collect all ship cells in this direction
            ship_cells: list[tuple[int, int, float, float, float]] = []  # (loa_i, type_j, freq, speed_ms, beam)
            for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
                for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                    q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                    if q <= 0 or not np.isfinite(q):
                        continue

                    v_kts = 10.0
                    if loa_i < len(speed) and type_j < len(speed[loa_i]):
                        s_list = speed[loa_i][type_j]
                        if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                            v_kts = float(np.mean(s_list))
                        elif isinstance(s_list, (int, float)):
                            v_kts = float(s_list)
                    v_ms = v_kts * 1852.0 / 3600.0

                    b = self.estimate_beam(self.get_loa_midpoint(loa_i, length_intervals))
                    if loa_i < len(beam) and type_j < len(beam[loa_i]):
                        b_list = beam[loa_i][type_j]
                        if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                            b = float(np.mean(b_list))
                        elif isinstance(b_list, (int, float)):
                            b = float(b_list)

                    ship_cells.append((loa_i, type_j, q, v_ms, b))

            # Pairwise overtaking between all ship cells in same direction
            for i, (loa_i, type_i, q_fast, v_fast, b_fast) in enumerate(ship_cells):
                for j, (loa_j, type_j, q_slow, v_slow, b_slow) in enumerate(ship_cells):
                    if i == j:
                        continue
                    if v_fast <= v_slow:
                        continue  # No overtaking if not faster

                    n_g_overtaking = get_overtaking_collision_candidates(
                        Q_fast=q_fast, Q_slow=q_slow,
                        V_fast=v_fast, V_slow=v_slow,
                        mu_fast=mu_ot, mu_slow=mu_ot,
                        sigma_fast=sigma_ot, sigma_slow=sigma_ot,
                        B_fast=b_fast, B_slow=b_slow,
                        L_w=leg_length_m
                    )
                    leg_overtaking += n_g_overtaking * pc_overtaking

        return leg_overtaking

    def _calc_bend_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        pc_bend: float,
        length_intervals: list[dict],
    ) -> float:
        """Calculate bend collision frequency for a single leg.

        Bend collisions occur at waypoints where a ship fails to turn.
        Only calculated when ``bend_angle > 5 degrees``.

        Returns the total bend collision frequency for this leg.
        """
        leg_bend = 0.0
        dir_keys = list(leg_dirs.keys())

        # Simplified: use average ship dimensions and traffic for this leg
        avg_freq = 0.0
        avg_length = 150.0
        avg_beam = 25.0
        count = 0
        for dir_key in dir_keys:
            dir_data = leg_dirs.get(dir_key, {})
            freq = np.array(dir_data.get('Frequency (ships/year)', []))
            for loa_i in range(len(freq) if hasattr(freq, '__len__') else 0):
                for type_j in range(len(freq[loa_i]) if loa_i < len(freq) and hasattr(freq[loa_i], '__len__') else 0):
                    q = float(freq[loa_i][type_j]) if loa_i < len(freq) and type_j < len(freq[loa_i]) else 0.0
                    if q > 0:
                        avg_freq += q
                        avg_length = (avg_length * count + self.get_loa_midpoint(loa_i, length_intervals)) / (count + 1)
                        avg_beam = (avg_beam * count + self.estimate_beam(self.get_loa_midpoint(loa_i, length_intervals))) / (count + 1)
                        count += 1

        # Bend collisions should only be calculated when there's an actual bend
        # at a waypoint between consecutive legs. Default to 0 (no bend).
        # Only calculate if segment_data explicitly specifies a bend_angle > 5 degrees.
        bend_angle_deg = float(seg_info.get('bend_angle', 0.0))
        bend_angle_rad = bend_angle_deg * np.pi / 180.0

        # Only calculate bend collision if there's a meaningful angle change (>5 degrees)
        if avg_freq > 0 and bend_angle_deg > 5.0:
            p_no_turn = 0.01  # Probability of failing to turn at bend
            n_g_bend = get_bend_collision_candidates(
                Q=avg_freq,
                P_no_turn=p_no_turn,
                L=avg_length,
                B=avg_beam,
                theta=bend_angle_rad
            )
            leg_bend += n_g_bend * pc_bend

        return leg_bend

    def _calc_crossing_collisions(
        self,
        traffic_data: dict[str, Any],
        segment_data: dict[str, Any],
        leg_keys: list[str],
        pc_crossing: float,
        length_intervals: list[dict],
    ) -> float:
        """Calculate crossing collision frequency between all leg pairs.

        Crossing collisions occur where two different legs share a waypoint
        and meet at a non-trivial angle.

        Returns the total crossing collision frequency.
        """
        total_crossing = 0.0
        total_legs = len(leg_keys)
        crossing_pairs_processed = 0
        total_pairs = total_legs * (total_legs - 1) // 2

        self._report_progress('cascade', 0.0, "Calculating crossing collisions...")

        for i, leg1_key in enumerate(leg_keys):
            for j, leg2_key in enumerate(leg_keys):
                if j <= i:
                    continue  # Avoid double counting

                seg1 = segment_data.get(leg1_key, {})
                seg2 = segment_data.get(leg2_key, {})

                # Parse endpoints
                s1_start = self._parse_point(seg1.get('Start_Point', ''))
                s1_end = self._parse_point(seg1.get('End_Point', ''))
                s2_start = self._parse_point(seg2.get('Start_Point', ''))
                s2_end = self._parse_point(seg2.get('End_Point', ''))

                # Only compute crossing if the legs share a waypoint
                shares_waypoint = (
                    self._points_match(s1_start, s2_start) or self._points_match(s1_start, s2_end) or
                    self._points_match(s1_end, s2_start) or self._points_match(s1_end, s2_end)
                )
                if not shares_waypoint:
                    crossing_pairs_processed += 1
                    continue

                # Calculate bearing from Start_Point/End_Point if not in segment_data
                if 'bearing' in seg1 and seg1['bearing']:
                    bearing1 = float(seg1['bearing'])
                elif s1_start and s1_end:
                    bearing1 = self._calc_bearing(s1_start, s1_end)
                else:
                    crossing_pairs_processed += 1
                    continue

                if 'bearing' in seg2 and seg2['bearing']:
                    bearing2 = float(seg2['bearing'])
                elif s2_start and s2_end:
                    bearing2 = self._calc_bearing(s2_start, s2_end)
                else:
                    crossing_pairs_processed += 1
                    continue

                crossing_angle = abs(bearing1 - bearing2) % 180.0
                if crossing_angle > 90:
                    crossing_angle = 180 - crossing_angle
                crossing_angle_rad = crossing_angle * np.pi / 180.0

                if crossing_angle_rad < 0.1:  # Nearly parallel, not a crossing
                    crossing_pairs_processed += 1
                    continue

                # Get traffic from both legs
                leg1_dirs = traffic_data.get(leg1_key, {})
                leg2_dirs = traffic_data.get(leg2_key, {})

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

                        # Iterate ship categories
                        for loa_i in range(len(freq1) if hasattr(freq1, '__len__') else 0):
                            for type_j in range(len(freq1[loa_i]) if loa_i < len(freq1) and hasattr(freq1[loa_i], '__len__') else 0):
                                q1 = float(freq1[loa_i][type_j]) if loa_i < len(freq1) and type_j < len(freq1[loa_i]) else 0.0
                                if q1 <= 0 or not np.isfinite(q1):
                                    continue

                                v1_kts = 10.0
                                if loa_i < len(speed1) and type_j < len(speed1[loa_i]):
                                    s_list = speed1[loa_i][type_j]
                                    if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                        v1_kts = float(np.mean(s_list))
                                    elif isinstance(s_list, (int, float)):
                                        v1_kts = float(s_list)
                                v1_ms = v1_kts * 1852.0 / 3600.0

                                l1 = self.get_loa_midpoint(loa_i, length_intervals)
                                b1 = self.estimate_beam(l1)
                                if loa_i < len(beam1_arr) and type_j < len(beam1_arr[loa_i]):
                                    b_list = beam1_arr[loa_i][type_j]
                                    if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                        b1 = float(np.mean(b_list))
                                    elif isinstance(b_list, (int, float)):
                                        b1 = float(b_list)

                                for loa_k in range(len(freq2) if hasattr(freq2, '__len__') else 0):
                                    for type_l in range(len(freq2[loa_k]) if loa_k < len(freq2) and hasattr(freq2[loa_k], '__len__') else 0):
                                        q2 = float(freq2[loa_k][type_l]) if loa_k < len(freq2) and type_l < len(freq2[loa_k]) else 0.0
                                        if q2 <= 0 or not np.isfinite(q2):
                                            continue

                                        v2_kts = 10.0
                                        if loa_k < len(speed2) and type_l < len(speed2[loa_k]):
                                            s_list = speed2[loa_k][type_l]
                                            if isinstance(s_list, (list, np.ndarray)) and len(s_list) > 0:
                                                v2_kts = float(np.mean(s_list))
                                            elif isinstance(s_list, (int, float)):
                                                v2_kts = float(s_list)
                                        v2_ms = v2_kts * 1852.0 / 3600.0

                                        l2 = self.get_loa_midpoint(loa_k, length_intervals)
                                        b2 = self.estimate_beam(l2)
                                        if loa_k < len(beam2_arr) and type_l < len(beam2_arr[loa_k]):
                                            b_list = beam2_arr[loa_k][type_l]
                                            if isinstance(b_list, (list, np.ndarray)) and len(b_list) > 0:
                                                b2 = float(np.mean(b_list))
                                            elif isinstance(b_list, (int, float)):
                                                b2 = float(b_list)

                                        n_g_crossing = get_crossing_collision_candidates(
                                            Q1=q1, Q2=q2,
                                            V1=v1_ms, V2=v2_ms,
                                            L1=l1, L2=l2,
                                            B1=b1, B2=b2,
                                            theta=crossing_angle_rad
                                        )
                                        total_crossing += n_g_crossing * pc_crossing

                crossing_pairs_processed += 1
                if total_pairs > 0:
                    self._report_progress(
                        'cascade',
                        crossing_pairs_processed / total_pairs,
                        f"Processing crossing pair {leg1_key}-{leg2_key}..."
                    )

        return total_crossing

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
        result: dict[str, float] = {
            'head_on': 0.0,
            'overtaking': 0.0,
            'crossing': 0.0,
            'bend': 0.0,
            'total': 0.0,
        }

        traffic_data = data.get('traffic_data', {})
        segment_data = data.get('segment_data', {})
        pc_vals = data.get('pc', {}) if isinstance(data.get('pc', {}), dict) else {}

        if not traffic_data or not segment_data:
            self.ship_collision_prob = 0.0
            self.collision_report = {'totals': result, 'by_leg': {}}
            return result

        # Get causation factors
        pc_headon = float(pc_vals.get('headon', 4.9e-5))
        pc_overtaking = float(pc_vals.get('overtaking', 1.1e-4))
        pc_crossing = float(pc_vals.get('crossing', 1.3e-4))
        pc_bend = float(pc_vals.get('bend', 1.3e-4))

        # Get ship categories for LOA estimates
        ship_categories = data.get('ship_categories', {})
        length_intervals = ship_categories.get('length_intervals', [])

        # Report structures
        by_leg: dict[str, dict[str, float]] = {}
        total_head_on = 0.0
        total_overtaking = 0.0
        total_crossing = 0.0
        total_bend = 0.0

        leg_keys = list(traffic_data.keys())
        total_legs = len(leg_keys)
        processed = 0

        self._report_progress('spatial', 0.0, "Starting ship collision calculations...")

        # Iterate through legs for head-on, overtaking, and bend (same-leg collisions)
        for leg_key in leg_keys:
            leg_dirs = traffic_data.get(leg_key, {})
            seg_info = segment_data.get(leg_key, {})
            leg_length_m = float(seg_info.get('line_length', 1000.0))

            leg_head_on = self._calc_head_on_collisions(
                leg_dirs, seg_info, leg_length_m, pc_headon, length_intervals
            )

            leg_overtaking = self._calc_overtaking_collisions(
                leg_dirs, seg_info, leg_length_m, pc_overtaking, length_intervals
            )

            leg_bend = self._calc_bend_collisions(
                leg_dirs, seg_info, pc_bend, length_intervals
            )

            # Store leg results
            by_leg[leg_key] = {
                'head_on': leg_head_on,
                'overtaking': leg_overtaking,
                'bend': leg_bend,
            }

            total_head_on += leg_head_on
            total_overtaking += leg_overtaking
            total_bend += leg_bend

            processed += 1
            self._report_progress(
                'spatial',
                processed / total_legs * 0.8,
                f"Processing leg {leg_key} ({processed}/{total_legs})..."
            )

        # Crossing collisions between different legs
        total_crossing = self._calc_crossing_collisions(
            traffic_data, segment_data, leg_keys, pc_crossing, length_intervals
        )

        # Compile results
        result['head_on'] = total_head_on
        result['overtaking'] = total_overtaking
        result['crossing'] = total_crossing
        result['bend'] = total_bend
        result['total'] = total_head_on + total_overtaking + total_crossing + total_bend

        self.ship_collision_prob = result['total']
        self.collision_report = {
            'totals': result,
            'by_leg': by_leg,
            'causation_factors': {
                'headon': pc_headon,
                'overtaking': pc_overtaking,
                'crossing': pc_crossing,
                'bend': pc_bend,
            },
        }

        self._report_progress('layers', 1.0, "Ship collision calculation complete")

        # Update UI with collision results
        try:
            self.p.main_widget.LEPHeadOnCollision.setText(f"{result['head_on']:.3e}")
            self.p.main_widget.LEPOvertakingCollision.setText(f"{result['overtaking']:.3e}")
            self.p.main_widget.LEPCrossingCollision.setText(f"{result['crossing']:.3e}")
            self.p.main_widget.LEPMergingCollision.setText(f"{result['bend']:.3e}")
        except Exception as e:
            pass  # UI update failed, but calculation succeeded

        return result
