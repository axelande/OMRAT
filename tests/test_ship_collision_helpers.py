"""Unit tests for the static helpers in compute/ship_collision_model.py.

The mixin's big methods (``_calc_head_on_collisions`` etc.) need full
scene setup and are exercised by integration tests; these cover the
small purely-static helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.ship_collision_model import ShipCollisionModelMixin as M


# ---------------------------------------------------------------------------
# get_loa_midpoint
# ---------------------------------------------------------------------------

class TestGetLoaMidpoint:
    def test_valid_interval_returns_midpoint(self):
        intervals = [{'min': 100.0, 'max': 200.0, 'label': '100-200'}]
        assert M.get_loa_midpoint(0, intervals) == 150.0

    def test_default_midpoint_for_out_of_range_index(self):
        # 0-25 default midpoint is 25.0.
        assert M.get_loa_midpoint(0, []) == 25.0
        assert M.get_loa_midpoint(3, []) == 250.0
        # Beyond the 5-element default list falls back to 150.0.
        assert M.get_loa_midpoint(99, []) == 150.0

    def test_malformed_min_max_uses_interval_defaults(self):
        """Non-numeric min/max fall back to the hardcoded 50/100 defaults."""
        intervals = [{'min': 'foo', 'max': 'bar', 'label': ''}]
        # The fallback defaults (min=50, max=100) live inside the try
        # block so invalid values fall through to the default_midpoints
        # list (index 0 -> 25.0).
        assert M.get_loa_midpoint(0, intervals) == 25.0


# ---------------------------------------------------------------------------
# estimate_beam
# ---------------------------------------------------------------------------

class TestEstimateBeam:
    def test_beam_ratio_is_one_sixth_five(self):
        assert M.estimate_beam(130.0) == pytest.approx(20.0, abs=1e-9)

    def test_zero_loa_gives_zero_beam(self):
        assert M.estimate_beam(0.0) == 0.0


# ---------------------------------------------------------------------------
# _parse_point
# ---------------------------------------------------------------------------

class TestParsePoint:
    def test_parses_space_separated(self):
        assert M._parse_point('14.5 55.3') == (14.5, 55.3)

    def test_extra_whitespace_tolerated(self):
        assert M._parse_point('  14.5   55.3 ') == (14.5, 55.3)

    def test_none_or_empty_returns_none(self):
        assert M._parse_point('') is None
        assert M._parse_point(None) is None

    def test_invalid_floats_return_none(self):
        assert M._parse_point('x y') is None

    def test_fewer_than_two_tokens_returns_none(self):
        assert M._parse_point('14.5') is None


# ---------------------------------------------------------------------------
# _calc_bearing
# ---------------------------------------------------------------------------

class TestCalcBearing:
    def test_bearing_due_north(self):
        b = M._calc_bearing((0.0, 0.0), (0.0, 1.0))
        assert b == pytest.approx(0.0, abs=1e-9)

    def test_bearing_due_east(self):
        b = M._calc_bearing((0.0, 0.0), (1.0, 0.0))
        assert b == pytest.approx(90.0, abs=1e-9)

    def test_bearing_due_south(self):
        b = M._calc_bearing((0.0, 0.0), (0.0, -1.0))
        assert b == pytest.approx(180.0, abs=1e-9)

    def test_bearing_due_west(self):
        b = M._calc_bearing((0.0, 0.0), (-1.0, 0.0))
        assert b == pytest.approx(270.0, abs=1e-9)

    def test_bearing_always_in_zero_to_360(self):
        for start in [(0, 0), (5, 5), (-2, 3)]:
            for end in [(1, 1), (-3, 2), (4, -6)]:
                b = M._calc_bearing(start, end)
                assert 0.0 <= b < 360.0


# ---------------------------------------------------------------------------
# _points_match
# ---------------------------------------------------------------------------

class TestPointsMatch:
    def test_exact_match(self):
        assert M._points_match((1.0, 2.0), (1.0, 2.0))

    def test_within_tolerance(self):
        assert M._points_match((1.0, 2.0), (1.0 + 1e-9, 2.0 - 1e-9))

    def test_outside_tolerance(self):
        assert not M._points_match((1.0, 2.0), (1.1, 2.0))

    def test_either_none_returns_false(self):
        assert not M._points_match(None, (1.0, 2.0))
        assert not M._points_match((1.0, 2.0), None)
        assert not M._points_match(None, None)

    def test_custom_tolerance(self):
        assert M._points_match((0, 0), (0.5, 0), tol=1.0)
        assert not M._points_match((0, 0), (0.5, 0), tol=0.1)


# ---------------------------------------------------------------------------
# _get_weighted_mu_sigma
# ---------------------------------------------------------------------------

class TestGetWeightedMuSigma:
    def _seg(self, **overrides):
        base = {
            'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
            'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
            'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
            'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 1.0,
            'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
            'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
            'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
            'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
        }
        base.update(overrides)
        return base

    def test_single_gaussian(self):
        mu, sigma = M._get_weighted_mu_sigma(self._seg(), direction=0)
        assert mu == pytest.approx(0.0, abs=1e-9)
        assert sigma == pytest.approx(100.0, abs=1e-6)

    def test_zero_weights_raises(self):
        seg = self._seg(weight1_1=0.0)
        with pytest.raises(ValueError):
            M._get_weighted_mu_sigma(seg, direction=0)

    def test_tiny_sigma_raises(self):
        """Bugs that feed a near-zero sigma should fail loudly."""
        seg = self._seg(std1_1=0.1, weight1_1=1.0)
        with pytest.raises(ValueError, match='sigma too small'):
            M._get_weighted_mu_sigma(seg, direction=0)

    def test_two_component_mixture_gives_mixture_variance(self):
        import numpy as np
        seg = self._seg(
            mean1_1=0.0, std1_1=50.0, weight1_1=1.0,
            mean1_2=200.0, std1_2=50.0, weight1_2=1.0,
        )
        mu, sigma = M._get_weighted_mu_sigma(seg, direction=0)
        assert mu == pytest.approx(100.0, abs=1e-6)  # midway
        # Total variance = 50^2 + (100)^2 = 12500
        assert sigma == pytest.approx(np.sqrt(12500.0), rel=1e-6)
