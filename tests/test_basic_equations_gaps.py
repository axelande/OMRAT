"""Gap-filler tests for compute/basic_equations.py.

Covers the top-level formula helpers that weren't exercised by the
existing test_basic_equations.py / test_powered_calculations.py /
test_ship_ship_collisions.py suites.
"""
from __future__ import annotations

import sys
from math import exp, pi
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.basic_equations import (
    get_Fcoll,
    get_drifting_prob,
    get_drift_time,
    repairtime_function,
    powered_na,
    get_not_repaired,
    squat_m,
    get_overtaking_collision_candidates,
    get_crossing_collision_candidates,
    get_powered_grounding_cat1,
    get_powered_grounding_cat2,
    default_blackout_by_ship_type,
    SHIP_TYPE_NAMES,
)


# ---------------------------------------------------------------------------
# Simple scalar formulas
# ---------------------------------------------------------------------------

class TestScalarFormulas:
    def test_get_Fcoll(self):
        assert get_Fcoll(na=100.0, pc=1.1e-4) == pytest.approx(0.011, abs=1e-12)

    def test_get_drifting_prob_positive(self):
        """Short transit over low drift frequency: probability is tiny."""
        p = get_drifting_prob(Fb=0.1, line_length=1000.0, ship_speed=10.0)
        assert 0 < p < 1

    def test_get_drifting_prob_is_monotone_in_Fb(self):
        p_low = get_drifting_prob(Fb=0.01, line_length=10000.0, ship_speed=10.0)
        p_high = get_drifting_prob(Fb=1.0, line_length=10000.0, ship_speed=10.0)
        assert p_low < p_high

    def test_get_drift_time(self):
        assert get_drift_time(distance=3600.0, drift_speed=2.0) == 1800.0

    def test_powered_na(self):
        """exp(-distance / (mean_time * speed))."""
        distance = 1000.0
        mean_time = 30.0
        speed = 10.0
        ai = mean_time * speed
        assert powered_na(distance, mean_time, speed) == pytest.approx(exp(-distance/ai), abs=1e-12)


# ---------------------------------------------------------------------------
# repairtime_function (dict-based)
# ---------------------------------------------------------------------------

class TestRepairTimeFunction:
    def test_lognormal_path(self):
        from scipy import stats
        data = {
            'active_window': 0,
            'std': 1.0, 'loc': 0.0, 'scale': 1.0,
        }
        x = 2.0
        expected = float(stats.lognorm(1.0, 0.0, 1.0).cdf(x))
        assert repairtime_function(data, x) == pytest.approx(expected, abs=1e-12)

    def test_user_function_path(self):
        data = {
            'active_window': 1,
            'func': 'x / 2',
        }
        assert repairtime_function(data, 10.0) == 5.0


# ---------------------------------------------------------------------------
# get_not_repaired (dict-based, normal / lognormal / eval)
# ---------------------------------------------------------------------------

class TestGetNotRepaired:
    def test_lognormal_flag(self):
        """use_lognormal==1 picks the lognorm CDF path."""
        from scipy import stats
        data = {
            'use_lognormal': 1,
            'std': 1.0, 'loc': 0.0, 'scale': 1.0,
            'func': 'this.should.not.be.eval',
        }
        # drift_time = 7200 / 2 / 3600 = 1.0 h
        got = get_not_repaired(data, drift_speed=2.0, dist=7200.0)
        expected = 1.0 - float(stats.lognorm(1.0, 0.0, 1.0).cdf(1.0))
        assert got == pytest.approx(expected, abs=1e-12)

    def test_user_func_path(self):
        """use_lognormal==0 evaluates `data['func']` with variable `x`."""
        data = {'use_lognormal': 0, 'func': '0.5 * x'}
        # drift_time = 7200/2/3600 = 1 h; func returns 0.5; p_nr = 0.5
        assert get_not_repaired(data, 2.0, 7200.0) == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# get_squat
# ---------------------------------------------------------------------------

class TestGetSquat:
    def test_zero_speed_returns_zero(self):
        assert squat_m(speed_kts=0.0, ship_type=18) == 0.0
        assert squat_m(speed_kts=-1.0, ship_type=18) == 0.0

    def test_default_cb_for_unknown_ship_type(self):
        """An unrecognised ship_type falls back to Cb = 0.75."""
        s = squat_m(speed_kts=10.0, ship_type=9999)
        # Cb * V^2 / 100 = 0.75 * 100 / 100 = 0.75
        assert s == pytest.approx(0.75, abs=1e-9)

    def test_explicit_cb_overrides_lookup(self):
        s = squat_m(speed_kts=10.0, ship_type=18, cb=0.9)
        # 0.9 * 100 / 100 = 0.9
        assert s == pytest.approx(0.9, abs=1e-9)

    def test_negative_ship_type_uses_default(self):
        s = squat_m(speed_kts=10.0, ship_type=-1)
        # -1 is treated as "no lookup" -> 0.75 default
        assert s == pytest.approx(0.75, abs=1e-9)


# ---------------------------------------------------------------------------
# Early-return branches
# ---------------------------------------------------------------------------

class TestEarlyReturnBranches:
    def test_overtaking_zero_fast_speed_returns_zero(self):
        """V_fast = V_slow -> overtaking impossible -> early 0.0 exit
        from the V_fast <= V_slow guard at the top of the function.
        """
        result = get_overtaking_collision_candidates(
            Q_fast=100, Q_slow=100,
            V_fast=5.0, V_slow=5.0,  # equal -> not overtaking
            mu_fast=0.0, mu_slow=0.0,
            sigma_fast=50.0, sigma_slow=50.0,
            B_fast=20.0, B_slow=20.0, L_w=1000.0,
        )
        assert result == 0.0

    def test_overtaking_negative_v_slow_hits_second_guard(self):
        """Negative V_slow passes the V_fast > V_slow check but trips the
        later divide-by-zero guard at L323."""
        result = get_overtaking_collision_candidates(
            Q_fast=100, Q_slow=100,
            V_fast=5.0, V_slow=-1.0,  # pathological but defensible
            mu_fast=0.0, mu_slow=0.0,
            sigma_fast=50.0, sigma_slow=50.0,
            B_fast=20.0, B_slow=20.0, L_w=1000.0,
        )
        assert result == 0.0

    def test_crossing_zero_speed_returns_zero(self):
        """V1 = 0 -> divide-by-zero guard -> 0.0."""
        # Use a non-degenerate angle to bypass the sin(theta)=0 guard.
        result = get_crossing_collision_candidates(
            Q1=100, Q2=100, V1=0.0, V2=5.0,
            L1=100.0, L2=100.0, B1=20.0, B2=20.0, theta=pi / 4,
        )
        assert result == 0.0

    def test_powered_grounding_cat2_zero_speed_returns_zero(self):
        result = get_powered_grounding_cat2(
            Q=100, Pc=1e-4, prob_at_position=0.1,
            distance_to_obstacle=1000.0,
            position_check_interval=30.0, ship_speed=0.0,
        )
        assert result == 0.0


# ---------------------------------------------------------------------------
# get_powered_grounding_cat1 (the uncovered one-liner at line 565)
# ---------------------------------------------------------------------------

class TestPoweredGroundingCat1:
    def test_formula(self):
        """Cat I = Pc * Q * prob_in_obstacle."""
        val = get_powered_grounding_cat1(Q=100, Pc=1e-4, prob_in_obstacle=0.2)
        assert val == pytest.approx(100 * 1e-4 * 0.2, abs=1e-12)

    def test_zero_probability_gives_zero(self):
        assert get_powered_grounding_cat1(Q=100, Pc=1e-4, prob_in_obstacle=0.0) == 0.0


# ---------------------------------------------------------------------------
# default_blackout_by_ship_type + SHIP_TYPE_NAMES
# ---------------------------------------------------------------------------

class TestBlackoutDefaults:
    def test_defaults_covers_all_ship_type_indices(self):
        d = default_blackout_by_ship_type()
        for idx in SHIP_TYPE_NAMES:
            assert idx in d, f"blackout default missing for ship type index {idx}"

    def test_passenger_and_roro_have_lower_rate(self):
        """IWRAP convention: Passenger/RoRo/RoPax get 0.1; others 1.0."""
        d = default_blackout_by_ship_type()
        passenger_like = [8, 9, 10, 11]  # indices that map to Passenger-variants
        for i in passenger_like:
            assert d[i] == 0.1

    def test_generic_ship_types_have_default_one(self):
        d = default_blackout_by_ship_type()
        # Cargo (18), Tanker (19) and Other (20) should be 1.0.
        assert d[18] == 1.0 and d[19] == 1.0 and d[20] == 1.0
