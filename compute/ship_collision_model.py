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
                        contrib = n_g_headon * pc_headon
                        leg_head_on += contrib
                        if by_cell is not None and contrib > 0.0:
                            half = 0.5 * contrib
                            k1 = f"{loa_i}_{type_j}"
                            k2 = f"{loa_k}_{type_l}"
                            by_cell[k1] = by_cell.get(k1, 0.0) + half
                            by_cell[k2] = by_cell.get(k2, 0.0) + half

        return leg_head_on

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
                    contrib = n_g_overtaking * pc_overtaking
                    leg_overtaking += contrib
                    if by_cell is not None and contrib > 0.0:
                        half = 0.5 * contrib
                        k1 = f"{loa_i}_{type_i}"
                        k2 = f"{loa_j}_{type_j}"
                        by_cell[k1] = by_cell.get(k1, 0.0) + half
                        by_cell[k2] = by_cell.get(k2, 0.0) + half

        return leg_overtaking

    def _calc_bend_collisions(
        self,
        leg_dirs: dict[str, Any],
        seg_info: dict[str, Any],
        pc_bend: float,
        length_intervals: list[dict],
        by_cell: dict[str, float] | None = None,
    ) -> float:
        """Calculate bend collision frequency for a single leg.

        Bend collisions occur at waypoints where a ship fails to turn.
        Only calculated when ``bend_angle > 5 degrees``.

        If ``by_cell`` is provided, the bend total is distributed across the
        cells contributing traffic on this leg in proportion to their share
        of total frequency.

        Returns the total bend collision frequency for this leg.
        """
        leg_bend = 0.0
        dir_keys = list(leg_dirs.keys())

        # Simplified: use average ship dimensions and traffic for this leg
        avg_freq = 0.0
        avg_length = 150.0
        avg_beam = 25.0
        count = 0
        # Cell-level traffic shares for downstream apportionment.  Keys are
        # ``"{loa_i}_{type_j}"``, values are total annual frequency on this
        # leg (summed over both directions).
        cell_freqs: dict[str, float] = {}
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
                        cell_key = f"{loa_i}_{type_j}"
                        cell_freqs[cell_key] = cell_freqs.get(cell_key, 0.0) + q

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
                    continue  # Avoid double counting

                seg1 = segment_data.get(leg1_key, {})
                seg2 = segment_data.get(leg2_key, {})

                # Parse endpoints
                s1_start = self._parse_point(seg1.get('Start_Point', ''))
                s1_end = self._parse_point(seg1.get('End_Point', ''))
                s2_start = self._parse_point(seg2.get('Start_Point', ''))
                s2_end = self._parse_point(seg2.get('End_Point', ''))

                # Only compute crossing if the legs share a waypoint.
                # We also remember WHICH waypoint they share so the
                # per-waypoint result layer can attribute the crossing
                # contribution to the correct point.
                shared_pt: tuple[float, float] | None = None
                if self._points_match(s1_start, s2_start) or self._points_match(s1_start, s2_end):
                    shared_pt = s1_start
                elif self._points_match(s1_end, s2_start) or self._points_match(s1_end, s2_end):
                    shared_pt = s1_end
                if shared_pt is None:
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
                                        contrib = n_g_crossing * pc_crossing
                                        total_crossing += contrib
                                        if by_waypoint is not None and shared_pt is not None:
                                            by_waypoint[shared_pt] = by_waypoint.get(shared_pt, 0.0) + contrib
                                        kind = (
                                            'merging'
                                            if crossing_angle <= 30.0
                                            else 'crossing'
                                        )
                                        if by_leg_pair is not None:
                                            # Threshold from IWRAP to
                                            # split a "small-angle"
                                            # merge from a true crossing.
                                            pair_key = (
                                                str(leg1_key), str(leg2_key),
                                            )
                                            rec = by_leg_pair.setdefault(
                                                pair_key,
                                                {
                                                    'crossing': 0.0,
                                                    'merging': 0.0,
                                                    'waypoint': shared_pt,
                                                    'angle_deg': float(
                                                        crossing_angle,
                                                    ),
                                                },
                                            )
                                            rec[kind] += contrib
                                        # Per-cell apportionment, split
                                        # 50/50 between the two legs'
                                        # ship cells.  Uses kind to route
                                        # contributions to crossing vs
                                        # merging accumulators so the
                                        # consequence calculation can
                                        # treat them as separate accident
                                        # categories.
                                        bucket = (
                                            by_cell_merging
                                            if kind == 'merging'
                                            else by_cell_crossing
                                        )
                                        if bucket is not None and contrib > 0.0:
                                            half = 0.5 * contrib
                                            k1 = f"{loa_i}_{type_j}"
                                            k2 = f"{loa_k}_{type_l}"
                                            bucket[k1] = bucket.get(k1, 0.0) + half
                                            bucket[k2] = bucket.get(k2, 0.0) + half

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
            self.collision_report = {
                'totals': result, 'by_leg': {},
                'by_cell': {
                    'head_on': {}, 'overtaking': {}, 'bend': {},
                    'crossing': {}, 'merging': {},
                },
            }
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
        # Per-(ship_type_idx, length_idx) annual-frequency contributions per
        # accident sub-type.  Pair-wise collisions split 50/50 between the
        # two cells; bend distributes proportionally to per-cell traffic.
        by_cell_head_on: dict[str, float] = {}
        by_cell_overtaking: dict[str, float] = {}
        by_cell_bend: dict[str, float] = {}
        by_cell_crossing: dict[str, float] = {}
        by_cell_merging: dict[str, float] = {}

        leg_keys = list(traffic_data.keys())
        total_legs = len(leg_keys)
        processed = 0

        self._report_progress('spatial', 0.0, "Starting ship collision calculations...")

        # Per-waypoint accumulator -- crossing + bend land here so the
        # result-layer factory can place a single point per shared
        # waypoint with the combined probability.
        # Keys are (lon, lat) tuples; values are dicts.
        by_waypoint: dict[tuple[float, float], dict[str, float]] = {}
        crossing_by_wp: dict[tuple[float, float], float] = {}

        # Iterate through legs for head-on, overtaking, and bend (same-leg collisions)
        for leg_key in leg_keys:
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
            )

            # Store leg results
            by_leg[leg_key] = {
                'head_on': leg_head_on,
                'overtaking': leg_overtaking,
                'bend': leg_bend,
            }

            # Place bend at the leg's END waypoint (where the ship would
            # fail to turn).  Skip if the END point is unparseable or
            # the bend is zero -- otherwise we'd litter the map with
            # zero-value points.
            if leg_bend > 0.0:
                end_pt = self._parse_point(seg_info.get('End_Point', ''))
                if end_pt is not None:
                    rec = by_waypoint.setdefault(
                        end_pt, {'crossing': 0.0, 'bend': 0.0}
                    )
                    rec['bend'] += leg_bend

            total_head_on += leg_head_on
            total_overtaking += leg_overtaking
            total_bend += leg_bend

            processed += 1
            self._report_progress(
                'spatial',
                processed / total_legs * 0.8,
                f"Processing leg {leg_key} ({processed}/{total_legs})..."
            )

        # Crossing collisions between different legs (per-waypoint accum).
        # ``by_leg_pair`` lets the View buttons render
        # "leg_a -> leg_b" rows separately for crossing vs merging.
        crossing_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        total_crossing = self._calc_crossing_collisions(
            traffic_data, segment_data, leg_keys, pc_crossing, length_intervals,
            by_waypoint=crossing_by_wp,
            by_leg_pair=crossing_by_pair,
            by_cell_crossing=by_cell_crossing,
            by_cell_merging=by_cell_merging,
        )
        for pt, contrib in crossing_by_wp.items():
            rec = by_waypoint.setdefault(pt, {'crossing': 0.0, 'bend': 0.0})
            rec['crossing'] += contrib

        # Compile results
        result['head_on'] = total_head_on
        result['overtaking'] = total_overtaking
        result['crossing'] = total_crossing
        result['bend'] = total_bend
        result['total'] = total_head_on + total_overtaking + total_crossing + total_bend

        self.ship_collision_prob = result['total']
        # Drop the waypoint-key tuples to JSON-friendly strings; the
        # result-layer factory parses them back.
        by_waypoint_serialisable = {
            f"{pt[0]:.6f} {pt[1]:.6f}": rec
            for pt, rec in by_waypoint.items()
        }
        # Serialise the per-leg-pair accumulator with string keys so it
        # round-trips through any JSON / dict-copy step.
        by_leg_pair_serialisable = {
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
        # Bend is a same-leg phenomenon attributed to the leg's end
        # waypoint -- key it as ``leg -> next_leg`` if any other leg
        # starts at that point, otherwise fall back to ``leg ->`` so
        # the dialog still has a useful label.
        bend_by_pair: dict[str, dict[str, Any]] = {}
        for leg_key, vals in by_leg.items():
            bend = float(vals.get('bend', 0.0) or 0.0)
            if bend <= 0.0:
                continue
            seg = segment_data.get(leg_key, {})
            end_pt = self._parse_point(seg.get('End_Point', ''))
            next_leg_id = ''
            if end_pt is not None:
                for other_key in leg_keys:
                    if other_key == leg_key:
                        continue
                    other_seg = segment_data.get(other_key, {})
                    other_start = self._parse_point(
                        other_seg.get('Start_Point', '')
                    )
                    if (
                        other_start is not None
                        and self._points_match(end_pt, other_start)
                    ):
                        next_leg_id = str(other_key)
                        break
            label = f"{leg_key}->{next_leg_id}" if next_leg_id else f"{leg_key}->"
            wp_text = (
                f"{end_pt[0]:.6f} {end_pt[1]:.6f}" if end_pt is not None else ''
            )
            bend_by_pair[label] = {'bend': bend, 'waypoint': wp_text}

        self.collision_report = {
            'totals': result,
            'by_leg': by_leg,
            'by_waypoint': by_waypoint_serialisable,
            'by_leg_pair': by_leg_pair_serialisable,
            'bend_by_pair': bend_by_pair,
            'by_cell': {
                'head_on': by_cell_head_on,
                'overtaking': by_cell_overtaking,
                'bend': by_cell_bend,
                'crossing': by_cell_crossing,
                'merging': by_cell_merging,
            },
            'causation_factors': {
                'headon': pc_headon,
                'overtaking': pc_overtaking,
                'crossing': pc_crossing,
                'bend': pc_bend,
            },
        }

        self._report_progress('layers', 1.0, "Ship collision calculation complete")

        # Build the per-leg line layer + per-waypoint point layer.
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

        # Update UI with collision results
        try:
            self.p.main_widget.LEPHeadOnCollision.setText(f"{result['head_on']:.3e}")
            self.p.main_widget.LEPOvertakingCollision.setText(f"{result['overtaking']:.3e}")
            self.p.main_widget.LEPCrossingCollision.setText(f"{result['crossing']:.3e}")
            self.p.main_widget.LEPMergingCollision.setText(f"{result['bend']:.3e}")
        except Exception as e:
            pass  # UI update failed, but calculation succeeded

        return result
