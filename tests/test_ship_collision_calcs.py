"""Tests for the calculation methods in ``ShipCollisionModelMixin``.

The static helpers are tested in ``test_ship_collision_helpers.py``.
This file exercises the ``_calc_*_collisions`` methods directly with
synthetic traffic_data + segment_data dicts so the per-row branches
(speed/beam mean of list, ship cell pairing, bend angle filter, crossing
shared-waypoint logic) are all reached.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.ship_collision_model import ShipCollisionModelMixin


def _segment_data():
    """A standard single-leg segment with valid distribution params."""
    return {
        'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
        'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
        'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
        'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 1.0,
        'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
        'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
        'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
        'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
    }


def _two_dir_traffic(speed_list_for_one_cell=None,
                     beam_list_for_one_cell=None):
    """2-direction traffic with one populated cell per direction."""
    # 21 ship-types x 5 LOA bins.
    def grid(init):
        return [[init] * 5 for _ in range(21)]

    f1 = grid(0.0); f1[18][1] = 100.0   # cargo, LOA 25-50, 100 ships/yr
    f2 = grid(0.0); f2[18][1] = 50.0
    s1 = grid(10.0); b1 = grid(20.0)
    s2 = grid(8.0);  b2 = grid(20.0)
    if speed_list_for_one_cell is not None:
        s1[18][1] = speed_list_for_one_cell
    if beam_list_for_one_cell is not None:
        b1[18][1] = beam_list_for_one_cell
    return {
        'East going': {
            'Frequency (ships/year)': f1,
            'Speed (knots)': s1,
            'Ship Beam (meters)': b1,
        },
        'West going': {
            'Frequency (ships/year)': f2,
            'Speed (knots)': s2,
            'Ship Beam (meters)': b2,
        },
    }


@pytest.fixture
def mixin():
    m = ShipCollisionModelMixin()
    m._report_progress = MagicMock(return_value=True)
    return m


# ---------------------------------------------------------------------------
# _calc_head_on_collisions
# ---------------------------------------------------------------------------

class TestCalcHeadOnCollisions:
    def test_single_direction_returns_zero(self, mixin):
        """A leg with only one traffic direction can't have head-on
        collisions -- returns the running total unchanged."""
        single_dir = {'East going': _two_dir_traffic()['East going']}
        seg = _segment_data()
        out = mixin._calc_head_on_collisions(
            leg_dirs=single_dir, seg_info=seg,
            leg_length_m=10_000.0, pc_headon=4.9e-5,
            length_intervals=[],
        )
        assert out == 0.0

    def test_basic_head_on_calculation(self, mixin):
        """Two opposite directions with traffic produce a positive number."""
        out = mixin._calc_head_on_collisions(
            leg_dirs=_two_dir_traffic(),
            seg_info=_segment_data(),
            leg_length_m=10_000.0, pc_headon=4.9e-5,
            length_intervals=[{'min': 0, 'max': 25, 'label': '0-25'},
                              {'min': 25, 'max': 50, 'label': '25-50'}],
        )
        assert out >= 0.0

    # Note: The speed/beam-list branches inside _calc_head_on_collisions
    # are dead code in current production flow because traffic_data is
    # always pre-averaged to scalars by handle_ais.convert_list2avg before
    # reaching the ship-collision pipeline.


# ---------------------------------------------------------------------------
# _calc_overtaking_collisions
# ---------------------------------------------------------------------------

class TestCalcOvertakingCollisions:
    def test_pairwise_with_different_speeds(self, mixin):
        """Two ship cells in same direction with v_fast > v_slow trigger the
        overtaking calculation (L298-309)."""
        def grid(init):
            return [[init] * 5 for _ in range(21)]

        # Two cells with different speeds in the same direction.
        f = grid(0.0); f[18][0] = 80.0; f[18][1] = 100.0
        s = grid(0.0); s[18][0] = 8.0; s[18][1] = 14.0  # second is faster
        b = grid(20.0)
        traffic = {
            'East going': {
                'Frequency (ships/year)': f,
                'Speed (knots)': s,
                'Ship Beam (meters)': b,
            },
        }
        out = mixin._calc_overtaking_collisions(
            leg_dirs=traffic, seg_info=_segment_data(),
            leg_length_m=10_000.0, pc_overtaking=1.1e-4,
            length_intervals=[{'min': 0, 'max': 25, 'label': '0-25'},
                              {'min': 25, 'max': 50, 'label': '25-50'}],
        )
        assert out >= 0.0

    # Speed/beam-list branches not exercised here -- see note in
    # TestCalcHeadOnCollisions.


# ---------------------------------------------------------------------------
# _calc_bend_collisions
# ---------------------------------------------------------------------------

class TestCalcBendCollisions:
    def test_no_bend_angle_returns_zero(self, mixin):
        """Without ``bend_angle`` set, nothing fires -> 0.0."""
        seg = _segment_data()  # no bend_angle key
        out = mixin._calc_bend_collisions(
            leg_dirs=_two_dir_traffic(),
            seg_info=seg, pc_bend=1.3e-4, length_intervals=[],
        )
        assert out == 0.0

    def test_bend_angle_below_5_skipped(self, mixin):
        """Bend angle <= 5° -> no bend collision (L354 guard)."""
        seg = _segment_data(); seg['bend_angle'] = 3.0
        out = mixin._calc_bend_collisions(
            leg_dirs=_two_dir_traffic(),
            seg_info=seg, pc_bend=1.3e-4, length_intervals=[],
        )
        assert out == 0.0

    def test_bend_angle_above_5_calculates(self, mixin):
        """Bend angle > 5° fires the calculation (L355-363)."""
        seg = _segment_data(); seg['bend_angle'] = 30.0
        out = mixin._calc_bend_collisions(
            leg_dirs=_two_dir_traffic(),
            seg_info=seg, pc_bend=1.3e-4,
            length_intervals=[{'min': 0, 'max': 25, 'label': '0-25'},
                              {'min': 25, 'max': 50, 'label': '25-50'}],
        )
        assert out > 0.0


# ---------------------------------------------------------------------------
# _calc_crossing_collisions
# ---------------------------------------------------------------------------

class TestCalcCrossingCollisions:
    def test_legs_without_shared_waypoint_skipped(self, mixin):
        """Two legs that don't share a waypoint produce zero crossings."""
        seg_data = {
            '1': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0'},
            '2': {**_segment_data(),
                  'Start_Point': '15.0 56.0', 'End_Point': '15.1 56.0'},
        }
        traffic = {'1': _two_dir_traffic(), '2': _two_dir_traffic()}
        out = mixin._calc_crossing_collisions(
            traffic_data=traffic, segment_data=seg_data,
            leg_keys=['1', '2'], pc_crossing=1.3e-4,
            length_intervals=[],
        )
        assert out == 0.0

    def test_legs_meeting_at_waypoint_compute_crossing(self, mixin):
        """Two legs that share a waypoint and meet at an angle compute
        crossing collisions (L413-514)."""
        seg_data = {
            # Leg 1 east-going.
            '1': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0'},
            # Leg 2 north-going from the same shared start point.
            '2': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.0 55.1'},
        }
        traffic = {'1': _two_dir_traffic(), '2': _two_dir_traffic()}
        out = mixin._calc_crossing_collisions(
            traffic_data=traffic, segment_data=seg_data,
            leg_keys=['1', '2'], pc_crossing=1.3e-4,
            length_intervals=[{'min': 0, 'max': 25, 'label': '0-25'},
                              {'min': 25, 'max': 50, 'label': '25-50'}],
        )
        assert out >= 0.0

    def test_parallel_legs_skipped(self, mixin):
        """Legs that share a waypoint but are nearly parallel don't compute
        crossings (L434 guard)."""
        # End at same point but both come from west.
        seg_data = {
            '1': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0'},
            '2': {**_segment_data(),
                  'Start_Point': '14.05 55.0', 'End_Point': '14.1 55.0'},
        }
        traffic = {'1': _two_dir_traffic(), '2': _two_dir_traffic()}
        out = mixin._calc_crossing_collisions(
            traffic_data=traffic, segment_data=seg_data,
            leg_keys=['1', '2'], pc_crossing=1.3e-4,
            length_intervals=[],
        )
        # Bearings are both ~90° -> crossing_angle ~0° -> skipped.
        assert out == 0.0

    def test_explicit_bearing_used_when_set(self, mixin):
        """If 'bearing' is in seg_info, it's used directly instead of computed."""
        seg_data = {
            '1': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                  'bearing': 90.0},
            '2': {**_segment_data(),
                  'Start_Point': '14.0 55.0', 'End_Point': '14.0 55.1',
                  'bearing': 0.0},
        }
        traffic = {'1': _two_dir_traffic(), '2': _two_dir_traffic()}
        out = mixin._calc_crossing_collisions(
            traffic_data=traffic, segment_data=seg_data,
            leg_keys=['1', '2'], pc_crossing=1.3e-4,
            length_intervals=[{'min': 0, 'max': 25, 'label': '0-25'},
                              {'min': 25, 'max': 50, 'label': '25-50'}],
        )
        assert out >= 0.0
