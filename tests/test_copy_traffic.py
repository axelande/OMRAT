"""Pure tests for ``omrat_utils.copy_traffic`` (copy traffic between legs,
AIS lock)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.copy_traffic import (  # noqa: E402
    LOCK_KEY,
    MAX_ALIGNED_DEG,
    SOURCE_KEY,
    aligned_swaps,
    angle_between,
    copy_angle,
    copy_leg_traffic,
    describe_misaligned,
    describe_targets,
    is_locked,
    locked_legs,
    misaligned_targets,
    release_copy,
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


class TestReleaseCopy:
    """Unticking the AIS lock releases the copy: lock off, link gone."""

    def test_unlocks_and_drops_the_link(self):
        segs = {'d': {LOCK_KEY: True, SOURCE_KEY: 'a'}}
        assert release_copy(segs, 'd') == 'a'
        assert segs['d'][LOCK_KEY] is False
        assert SOURCE_KEY not in segs['d']

    def test_plain_locked_leg_has_nothing_to_release(self):
        segs = {'d': {LOCK_KEY: True}}
        assert release_copy(segs, 'd') is None
        assert segs['d'][LOCK_KEY] is False

    def test_unknown_leg(self):
        assert release_copy({}, 'x') is None

    def test_junction_link_goes_with_it(self):
        from geometries.junctions import traffic_link_root
        segs = {'a': {}, 'd': {LOCK_KEY: True, SOURCE_KEY: 'a'}}
        assert traffic_link_root(segs, 'd') == 'a'
        release_copy(segs, 'd')
        assert traffic_link_root(segs, 'd') == 'd'


def _leg(bearing: float, name: str, lon: float = 12.0, lat: float = 56.0, length: float = 0.05) -> dict:
    """A leg drawn on compass ``bearing`` (same flat-earth formula as
    ``direction_bearing``, so the angles come out exact)."""
    dy = length * math.cos(math.radians(bearing))
    dx = length * math.sin(math.radians(bearing)) / math.cos(math.radians(lat + dy / 2.0))
    return {
        'Leg_name': name,
        'Start_Point': f'{lon:.6f} {lat:.6f}',
        'End_Point': f'{lon + dx:.6f} {lat + dy:.6f}',
    }


class TestDirectionCheck:
    """Bearing check behind the Copy traffic direction warning."""

    @pytest.mark.parametrize('a, b, expected', [
        (10, 350, 20), (350, 10, 20), (0, 180, 180), (90, 270, 180), (45, 46, 1), (0, 360, 0),
    ])
    def test_angle_between(self, a, b, expected):
        assert angle_between(a, b) == pytest.approx(expected)

    @pytest.mark.parametrize('offset', [0, 30, 60, 85, 89, -89, -60])
    def test_aligned_target_not_flagged(self, offset):
        segs = {'s': _leg(70, 'S'), 't': _leg(70 + offset, 'T')}
        assert copy_angle(segs, 's', 't') == pytest.approx(abs(offset), abs=0.05)
        assert misaligned_targets(segs, 's', ['t']) == []

    @pytest.mark.parametrize('offset', [91, 95, 120, 180, -91, -135])
    def test_reversed_target_flagged(self, offset):
        segs = {'s': _leg(70, 'S'), 't': _leg(70 + offset, 'T')}
        flagged = misaligned_targets(segs, 's', ['t'])
        assert [t for t, _ in flagged] == ['t']
        assert flagged[0][1] > MAX_ALIGNED_DEG

    @pytest.mark.parametrize('offset, flagged', [(0, True), (89, True), (91, False), (180, False)])
    def test_swap_box_reverses_the_check(self, offset, flagged):
        segs = {'s': _leg(70, 'S'), 't': _leg(70 + offset, 'T')}
        assert bool(misaligned_targets(segs, 's', ['t'], swap_dirs=True)) is flagged

    def test_direction_names_are_not_compared(self):
        # 44 deg -> "North going", 46 deg -> "East going": same way, no warning.
        segs = {'s': {**_leg(44, 'S'), 'Dirs': ['North going', 'South going']},
                't': {**_leg(46, 'T'), 'Dirs': ['East going', 'West going']}}
        assert misaligned_targets(segs, 's', ['t']) == []
        # 44 deg vs 226 deg -> "North going" vs "West going": reversed, warns.
        segs['t'] = {**_leg(226, 'T'), 'Dirs': ['West going', 'East going']}
        assert [t for t, _ in misaligned_targets(segs, 's', ['t'])] == ['t']

    def test_across_north(self):
        segs = {'s': _leg(350, 'S'), 'same': _leg(10, 'A'), 'rev': _leg(190, 'B')}
        assert misaligned_targets(segs, 's', ['same', 'rev']) == [('rev', pytest.approx(160, abs=0.05))]

    def test_mixed_selection_swaps_only_the_reversed_legs(self):
        segs = {'s': _leg(70, 'S'), 'same': _leg(100, 'A'), 'rev': _leg(260, 'B')}
        expected = {'same': False, 'rev': True}
        # Whatever the Swap box says, the per-leg answer is the same.
        assert aligned_swaps(segs, 's', ['same', 'rev'], swap_dirs=False) == expected
        assert aligned_swaps(segs, 's', ['same', 'rev'], swap_dirs=True) == expected
        for t, swap in expected.items():
            assert copy_angle(segs, 's', t, swap) <= MAX_ALIGNED_DEG

    def test_legs_without_geometry_keep_the_box(self):
        segs = {'s': _leg(70, 'S'), 'x': {'Leg_name': 'X'}}
        assert copy_angle(segs, 's', 'x') is None
        assert misaligned_targets(segs, 's', ['x']) == []
        assert aligned_swaps(segs, 's', ['x'], swap_dirs=True) == {'x': True}

    def test_source_in_targets_is_ignored(self):
        segs = {'s': _leg(70, 'S')}
        assert misaligned_targets(segs, 's', ['s'], swap_dirs=True) == []

    def test_message(self):
        segs = {'s': _leg(70, 'S'), 'same': _leg(100, 'A'), 'rev': _leg(250, 'B')}
        flagged = misaligned_targets(segs, 's', ['same', 'rev'])
        text = describe_misaligned(segs, 's', flagged, n_targets=2)
        assert 'B (rev): 180 deg apart' in text
        assert 'A (same)' not in text
        assert 'fine as chosen' in text
        assert 'fine as chosen' not in describe_misaligned(segs, 's', flagged, n_targets=1)

    def test_swapped_copy_lands_on_the_right_direction(self):
        # Source drawn east, target drawn west: per-leg swap puts the
        # east-going traffic on the target's direction 2 (east going).
        traffic = {'s': {'East going': _block(100), 'West going': _block(200)}}
        segs = {'s': {**_leg(90, 'S'), 'Dirs': ['East going', 'West going'], 'mean1_1': 50.0, 'mean2_1': -60.0},
                't': {**_leg(270, 'T'), 'Dirs': ['West going', 'East going']}}
        swap = aligned_swaps(segs, 's', ['t'])['t']
        assert swap is True
        copy_leg_traffic(traffic, segs, 's', 't', swap_dirs=swap)
        assert traffic['t']['East going']['Frequency (ships/year)'][0][0] == 100
        assert traffic['t']['West going']['Frequency (ships/year)'][0][0] == 200
        assert segs['t']['mean2_1'] == -50.0 and segs['t']['mean1_1'] == 60.0
