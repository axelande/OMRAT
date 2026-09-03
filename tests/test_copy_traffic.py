"""Pure tests for ``omrat_utils.copy_traffic`` (copy traffic between legs,
AIS lock)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.copy_traffic import (  # noqa: E402
    LOCK_KEY,
    SOURCE_KEY,
    copy_leg_traffic,
    describe_targets,
    is_locked,
    locked_legs,
    set_locked,
    split_locked,
)


def _block(freq: float) -> dict:
    return {
        'Frequency (ships/year)': [[freq, freq + 1], [freq + 2, freq + 3]],
        'Speed (knots)': [[10.0, 11.0], [12.0, 13.0]],
        'Scaling (%)': [[100.0, 100.0], [100.0, 100.0]],
    }


@pytest.fixture
def data():
    traffic = {
        'a': {'East going': _block(100), 'West going': _block(200)},
        'd': {'East going': _block(1), 'West going': _block(2)},
    }
    segs = {
        'a': {
            'Leg_name': 'LEG_5_12_a', 'Dirs': ['East going', 'West going'],
            'mean1_1': 50.0, 'std1_1': 20.0, 'weight1_1': 100, 'mean1_2': 0, 'std1_2': 0, 'weight1_2': 0,
            'mean1_3': 0, 'std1_3': 0, 'weight1_3': 0,
            'mean2_1': -60.0, 'std2_1': 25.0, 'weight2_1': 100, 'mean2_2': 0, 'std2_2': 0, 'weight2_2': 0,
            'mean2_3': 0, 'std2_3': 0, 'weight2_3': 0,
            'u_min1': -100.0, 'u_max1': 300.0, 'u_p1': 5, 'ai1': 180,
            'u_min2': -400.0, 'u_max2': 100.0, 'u_p2': 7, 'ai2': 120,
            'dist1': np.array([40.0, 60.0]), 'dist2': np.array([-70.0, -50.0]),
        },
        'd': {
            'Leg_name': 'LEG_5_12_d', 'Dirs': ['East going', 'West going'],
            'mean1_1': 1.0, 'std1_1': 1.0, 'mean2_1': 2.0, 'std2_1': 2.0,
            'dist1': np.array([1.0]), 'dist2': np.array([2.0]),
        },
        'n': {'Leg_name': 'LEG_1_1_c', 'Dirs': ['North going', 'South going']},
    }
    return traffic, segs


class TestCopy:
    def test_copies_matrices_per_direction_index(self, data):
        traffic, segs = data
        dirs = copy_leg_traffic(traffic, segs, 'a', 'd')
        assert dirs == ['East going', 'West going']
        assert traffic['d']['East going']['Frequency (ships/year)'] == [[100, 101], [102, 103]]
        assert traffic['d']['West going']['Frequency (ships/year)'] == [[200, 201], [202, 203]]

    def test_copy_is_deep(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'd')
        traffic['d']['East going']['Frequency (ships/year)'][0][0] = 999
        assert traffic['a']['East going']['Frequency (ships/year)'][0][0] == 100

    def test_target_direction_labels_come_from_target(self, data):
        """Leg n is north/south: its own labels are kept, filled by index."""
        traffic, segs = data
        dirs = copy_leg_traffic(traffic, segs, 'a', 'n')
        assert dirs == ['North going', 'South going']
        assert traffic['n']['North going']['Frequency (ships/year)'][0][0] == 100
        assert traffic['n']['South going']['Frequency (ships/year)'][0][0] == 200

    def test_distributions_copied_and_locked(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'd')
        d = segs['d']
        assert d['mean1_1'] == 50.0 and d['std1_1'] == 20.0 and d['mean2_1'] == -60.0
        assert d['u_min1'] == -100.0 and d['u_max1'] == 300.0 and d['u_p1'] == 5 and d['ai1'] == 180
        assert d['ai2'] == 120
        np.testing.assert_allclose(d['dist1'], [40.0, 60.0])
        np.testing.assert_allclose(d['dist2'], [-70.0, -50.0])
        assert d[LOCK_KEY] is True
        assert d[SOURCE_KEY] == 'a'

    def test_distributions_not_copied_when_disabled(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'd', copy_distributions=False)
        assert segs['d']['mean1_1'] == 1.0
        np.testing.assert_allclose(segs['d']['dist1'], [1.0])

    def test_lock_optional(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'd', lock=False)
        assert LOCK_KEY not in segs['d']

    def test_swap_dirs_exchanges_and_mirrors(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'd', swap_dirs=True)
        # Direction 1 of d gets direction 2 of a, and vice versa.
        assert traffic['d']['East going']['Frequency (ships/year)'][0][0] == 200
        assert traffic['d']['West going']['Frequency (ships/year)'][0][0] == 100
        d = segs['d']
        assert d['mean1_1'] == 60.0          # -(-60)
        assert d['mean2_1'] == -50.0         # -(50)
        assert d['std1_1'] == 25.0 and d['std2_1'] == 20.0
        assert d['ai1'] == 120 and d['ai2'] == 180
        # Uniform bounds mirrored: [-400, 100] -> [-100, 400]
        assert d['u_min1'] == -100.0 and d['u_max1'] == 400.0
        assert d['u_min2'] == -300.0 and d['u_max2'] == 100.0
        np.testing.assert_allclose(d['dist1'], [70.0, 50.0])
        np.testing.assert_allclose(d['dist2'], [-40.0, -60.0])

    def test_dist_lists_become_arrays(self, data):
        traffic, segs = data
        segs['a']['dist1'] = [1.0, 2.0]
        copy_leg_traffic(traffic, segs, 'a', 'd')
        assert isinstance(segs['d']['dist1'], np.ndarray)

    def test_same_leg_rejected(self, data):
        traffic, segs = data
        with pytest.raises(ValueError):
            copy_leg_traffic(traffic, segs, 'a', 'a')

    def test_source_without_traffic_rejected(self, data):
        traffic, segs = data
        with pytest.raises(KeyError):
            copy_leg_traffic(traffic, segs, 'n', 'd')

    def test_unknown_target_segment_is_created(self, data):
        traffic, segs = data
        copy_leg_traffic(traffic, segs, 'a', 'zz')
        assert segs['zz'][LOCK_KEY] is True
        assert 'East going' in traffic['zz']


class TestLock:
    def test_lock_helpers(self, data):
        _traffic, segs = data
        assert not is_locked(segs, 'a')
        assert set_locked(segs, 'a', True)
        assert is_locked(segs, 'a')
        assert locked_legs(segs) == ['a']
        assert set_locked(segs, 'a', False)
        assert not is_locked(segs, 'a')
        assert not set_locked(segs, 'missing', True)

    def test_non_bool_truthy_is_not_locked(self, data):
        _traffic, segs = data
        segs['a'][LOCK_KEY] = 'yes'
        assert not is_locked(segs, 'a')

    def test_split_locked(self, data):
        _traffic, segs = data
        set_locked(segs, 'd', True)
        legs = {'a': {'Width': '5000'}, 'd': {'Width': '5000'}, 'n': {'Width': '5000'}}
        keep, skipped = split_locked(legs, segs)
        assert list(keep) == ['a', 'n']
        assert skipped == ['d']

    def test_describe_targets(self, data):
        _traffic, segs = data
        assert describe_targets(['a', 'q'], segs) == 'LEG_5_12_a (a), q'
