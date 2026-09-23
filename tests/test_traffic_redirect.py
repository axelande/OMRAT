"""Pure tests for ``compute.traffic_redirect`` (suppress a leg, move its traffic)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.traffic_redirect import (  # noqa: E402
    after_split,
    apply_traffic_redirects,
    auto_target_dir,
    drop_legs_from_junctions,
    merge_direction,
    normalise_redirect,
    set_suppressed,
    total_frequency,
)

FREQ = 'Frequency (ships/year)'
SPEED = 'Speed (knots)'


def _dir(freq: float, speed: float = 10.0) -> dict:
    """One direction block: a 1 x 2 matrix, the second cell empty."""
    return {
        FREQ: [[freq, 0.0]],
        SPEED: [[speed, 0.0]],
        'Draught (meters)': [[8.0, 0.0]],
        'Ship heights (meters)': [[30.0, 0.0]],
        'Ship Beam (meters)': [[20.0, 0.0]],
    }


def _leg(seg_id: str, start: str, end: str, dirs: list[str]) -> dict:
    return {'Segment_Id': seg_id, 'Leg_name': f'LEG_1_{seg_id}', 'Start_Point': start,
            'End_Point': end, 'Dirs': dirs}


@pytest.fixture
def data():
    """Leg A (north-south, drawn northwards) and two east-west legs B, C.

    A carries 100 ships north and 40 south; B and C carry 10 each way.
    """
    ns = ['North going', 'South going']
    ew = ['East going', 'West going']
    return {
        'segment_data': {
            'A': _leg('A', '14.0 55.0', '14.0 55.2', ns),
            'B': _leg('B', '14.0 55.2', '14.2 55.2', ew),
            'C': _leg('C', '14.2 55.2', '14.2 55.4', ns),
        },
        'traffic_data': {
            'A': {'North going': _dir(100, 14.0), 'South going': _dir(40, 12.0)},
            'B': {'East going': _dir(10), 'West going': _dir(10)},
            'C': {'North going': _dir(10), 'South going': _dir(10)},
        },
    }


def _suppress(data, redirect):
    set_suppressed(data['segment_data'], 'A', True, redirect)


def _ships(data, leg, direction):
    return total_frequency(data['traffic_data'][leg][direction])


class TestMoveTraffic:
    def test_full_move_conserves_ships(self, data):
        before = sum(total_frequency(v) for blk in data['traffic_data'].values() for v in blk.values())
        _suppress(data, [
            {'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100},
            {'from_dir': 1, 'leg': 'C', 'dir': 1, 'share': 100},
        ])
        apply_traffic_redirects(data)
        after = sum(total_frequency(v) for blk in data['traffic_data'].values() for v in blk.values())
        assert after == pytest.approx(before)
        assert _ships(data, 'C', 'North going') == pytest.approx(110)
        assert _ships(data, 'C', 'South going') == pytest.approx(50)

    def test_suppressed_leg_is_removed_from_compute_data(self, data):
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        summary = apply_traffic_redirects(data)
        assert 'A' not in data['segment_data']
        assert 'A' not in data['traffic_data']
        assert summary['removed'] == ['A']

    def test_split_between_alternatives(self, data):
        # 80 % of the northbound ships take B, 20 % take C.
        _suppress(data, [
            {'from_dir': 0, 'leg': 'B', 'dir': 0, 'share': 80},
            {'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 20},
        ])
        apply_traffic_redirects(data)
        assert _ships(data, 'B', 'East going') == pytest.approx(10 + 80)
        assert _ships(data, 'C', 'North going') == pytest.approx(10 + 20)
        # Directions without a target are not moved anywhere.
        assert _ships(data, 'B', 'West going') == pytest.approx(10)
        assert _ships(data, 'C', 'South going') == pytest.approx(10)

    def test_detour_in_series_gives_every_leg_the_full_share(self, data):
        _suppress(data, [
            {'from_dir': 0, 'leg': 'B', 'dir': 0, 'share': 100},
            {'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100},
        ])
        apply_traffic_redirects(data)
        assert _ships(data, 'B', 'East going') == pytest.approx(110)
        assert _ships(data, 'C', 'North going') == pytest.approx(110)

    def test_speed_is_frequency_weighted(self, data):
        # C: 10 ships at 10 kn, moved: 100 ships at 14 kn -> (100 + 1400) / 110.
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        apply_traffic_redirects(data)
        speed = data['traffic_data']['C']['North going'][SPEED][0][0]
        assert speed == pytest.approx((10 * 10 + 100 * 14) / 110)

    def test_not_suppressed_redirect_is_ignored(self, data):
        original = copy.deepcopy(data)
        data['segment_data']['A']['traffic_redirect'] = [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}]
        summary = apply_traffic_redirects(data)
        assert summary['moves'] == []
        assert data['traffic_data'] == original['traffic_data']
        assert 'A' in data['segment_data']

    def test_suppressed_without_redirect_drops_the_ships(self, data):
        _suppress(data, [])
        apply_traffic_redirects(data)
        assert 'A' not in data['traffic_data']
        assert _ships(data, 'C', 'North going') == pytest.approx(10)

    def test_missing_or_suppressed_target_is_skipped_with_warning(self, data):
        data['segment_data']['B']['suppressed'] = True
        _suppress(data, [
            {'from_dir': 0, 'leg': 'B', 'dir': 0, 'share': 100},
            {'from_dir': 0, 'leg': 'gone', 'dir': 0, 'share': 100},
        ])
        summary = apply_traffic_redirects(data)
        assert summary['moves'] == []
        assert len(summary['warnings']) == 2
        assert set(summary['removed']) == {'A', 'B'}

    def test_target_without_traffic_gets_both_directions(self, data):
        del data['traffic_data']['C']
        _suppress(data, [{'from_dir': 1, 'leg': 'C', 'dir': 1, 'share': 50}])
        apply_traffic_redirects(data)
        assert set(data['traffic_data']['C']) == {'North going', 'South going'}
        assert _ships(data, 'C', 'South going') == pytest.approx(20)
        assert _ships(data, 'C', 'North going') == pytest.approx(0)


class TestMergeDirection:
    def test_empty_target_cell_takes_source_values(self):
        target, source = _dir(0.0, 0.0), _dir(30.0, 15.0)
        moved = merge_direction(target, source, 50)
        assert moved == pytest.approx(15)
        assert target[FREQ][0][0] == pytest.approx(15)
        assert target[SPEED][0][0] == pytest.approx(15.0)

    def test_zero_share_moves_nothing(self):
        target, source = _dir(5.0), _dir(30.0)
        assert merge_direction(target, source, 0) == 0.0
        assert target[FREQ][0][0] == 5.0


class TestDirections:
    def test_auto_dir_follows_bearing(self, data):
        segs = data['segment_data']
        assert auto_target_dir(segs, 'A', 0, 'C') == 0      # both drawn north
        assert auto_target_dir(segs, 'A', 1, 'C') == 1
        # A leg drawn southwards: northbound traffic is its direction 1.
        segs['D'] = _leg('D', '14.4 55.4', '14.4 55.2', ['South going', 'North going'])
        assert auto_target_dir(segs, 'A', 0, 'D') == 1

    def test_normalise_drops_bad_rows(self):
        rows = normalise_redirect([
            {'from_dir': 0, 'leg': 7, 'dir': 1, 'share': '80'},
            {'from_dir': 2, 'leg': '7', 'dir': 0, 'share': 10},
            {'from_dir': 0, 'dir': 0, 'share': 10},
            'junk',
        ])
        assert rows == [{'from_dir': 0, 'leg': '7', 'dir': 1, 'share': 80.0}]


class TestJunctions:
    def test_removed_leg_leaves_the_matrix(self, data):
        # Junction where A, B and a third leg X meet.
        segs = data['segment_data']
        segs['X'] = _leg('X', '14.0 55.2', '14.0 55.4', ['North going', 'South going'])
        junctions = {'j': {
            'point': [14.0, 55.2], 'legs': {'A': 'end', 'B': 'start', 'X': 'start'},
            'transitions': {'A': {'X': 0.9, 'B': 0.1}, 'B': {'A': 0.5, 'X': 0.5}, 'X': {'A': 1.0}},
            'source': 'user',
        }}
        del segs['A']
        out = drop_legs_from_junctions(junctions, {'A'}, segs)
        j = out['j']
        assert set(j['legs']) == {'B', 'X'}
        assert 'A' not in j['transitions']
        assert j['transitions']['B'] == pytest.approx({'X': 1.0})
        # X sent everything to A: falls back to the geometric default.
        assert j['transitions']['X'] == pytest.approx({'B': 1.0})
        assert j['source'] == 'user'

    def test_junction_with_one_leg_left_disappears(self, data):
        junctions = {'j': {'point': [14.0, 55.2], 'legs': {'A': 'end', 'B': 'start'},
                           'transitions': {'A': {'B': 1.0}, 'B': {'A': 1.0}}, 'source': 'geometry'}}
        _suppress(data, [])
        data['junctions'] = junctions
        apply_traffic_redirects(data)
        assert data['junctions'] == {}


class TestSplit:
    def test_split_suppressed_leg_moves_ships_once(self, data):
        from geometries.route_validation import split_leg_at_points
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        ids = iter(['A2'])
        split_leg_at_points(data['segment_data'], 'A', [(14.0, 55.1)], lambda: next(ids), data['traffic_data'])
        assert data['segment_data']['A2']['suppressed'] is True
        assert data['segment_data']['A2']['traffic_redirect'] == []
        apply_traffic_redirects(data)
        assert _ships(data, 'C', 'North going') == pytest.approx(110)

    def test_split_target_gets_the_share_on_every_sub_leg(self, data):
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 60}])
        after_split(data['segment_data'], 'C', ['C', 'C2'])
        entries = data['segment_data']['A']['traffic_redirect']
        assert [(e['leg'], e['share']) for e in entries] == [('C', 60), ('C2', 60)]


def test_schema_accepts_the_new_fields():
    from omrat_utils.validate_data import Segment
    seg = {k: 0 for k in (
        'mean1_1', 'std1_1', 'mean2_1', 'std2_1', 'weight1_1', 'weight2_1', 'mean1_2', 'mean1_3', 'std1_2',
        'std1_3', 'mean2_2', 'mean2_3', 'std2_2', 'std2_3', 'weight1_2', 'weight1_3', 'weight2_2', 'weight2_3',
        'u_min1', 'u_max1', 'u_p1', 'ai1', 'u_min2', 'u_max2', 'u_p2', 'ai2', 'Width', 'line_length', 'Route_Id')}
    seg.update({'Start_Point': '14 55', 'End_Point': '14 56', 'Dirs': ['N', 'S'], 'Leg_name': 'L',
                'Segment_Id': '1', 'suppressed': True,
                'traffic_redirect': [{'from_dir': 0, 'leg': '2', 'dir': 1, 'share': 80.0}]})
    model = Segment.model_validate(seg)
    assert model.suppressed is True
    assert model.traffic_redirect[0].share == 80.0


class TestExport:
    def test_export_copy_leaves_project_untouched(self, data):
        from compute.traffic_redirect import prepare_export_data
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        original = copy.deepcopy(data)
        prepared, summary = prepare_export_data(data)
        assert data == original
        assert 'A' not in prepared['segment_data']
        assert total_frequency(prepared['traffic_data']['C']['North going']) == pytest.approx(110)
        assert summary['removed'] == ['A']

    def test_iwrap_xml_has_no_suppressed_leg(self, data):
        import xml.etree.ElementTree as ET
        from compute.iwrap_convertion import generate_iwrap_xml
        from compute.traffic_redirect import prepare_export_data
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        prepared, _ = prepare_export_data(data)
        root = generate_iwrap_xml(prepared)
        names = {leg.get('name') for leg in root.iter('leg')}
        assert names == {'LEG_1_B', 'LEG_1_C'}
        assert ET.tostring(root)  # serialises

    def test_popup_text_names_legs_directions_and_ships(self, data):
        from compute.traffic_redirect import describe_redirect_summary
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        segs = copy.deepcopy(data['segment_data'])
        summary = apply_traffic_redirects(data)
        text = describe_redirect_summary(summary, segs)
        assert 'Suppressed: LEG_1_A (A)' in text
        assert 'LEG_1_A (A) North going -> LEG_1_C (C) North going: 100 % (100 ships/year)' in text


class TestSuppressedWith:
    """A whole route: one lead leg moves the ships, the others go with it."""

    @pytest.fixture
    def route(self, data):
        # Route 7 = A + A2 (same ships), detour = C.
        data['segment_data']['A2'] = _leg('A2', '14.0 55.2', '14.0 55.3', ['North going', 'South going'])
        data['traffic_data']['A2'] = copy.deepcopy(data['traffic_data']['A'])
        return data

    def test_route_ships_move_once(self, route):
        from compute.traffic_redirect import set_group
        _suppress(route, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        set_group(route['segment_data'], 'A', ['A2'])
        summary = apply_traffic_redirects(route)
        assert _ships(route, 'C', 'North going') == pytest.approx(10 + 100)   # not 10 + 200
        assert set(summary['removed']) == {'A', 'A2'}

    def test_member_redirect_is_ignored(self, route):
        from compute.traffic_redirect import set_group
        _suppress(route, [])
        route['segment_data']['A2']['traffic_redirect'] = [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}]
        set_group(route['segment_data'], 'A', ['A2'])
        apply_traffic_redirects(route)
        assert _ships(route, 'C', 'North going') == pytest.approx(10)

    def test_regroup_and_restore(self, route):
        from compute.traffic_redirect import group_members, is_suppressed, restore_group, set_group
        segs = route['segment_data']
        _suppress(route, [])
        set_group(segs, 'A', ['A2', 'B'])
        assert sorted(group_members(segs, 'A')) == ['A2', 'B']
        set_group(segs, 'A', ['A2'])                       # B unticked -> back in play
        assert not is_suppressed(segs, 'B') and 'suppressed_with' not in segs['B']
        assert sorted(restore_group(segs, 'A')) == ['A', 'A2']
        assert not is_suppressed(segs, 'A2') and not is_suppressed(segs, 'A')

    def test_split_lead_puts_sub_legs_in_its_group(self, data):
        from compute.traffic_redirect import group_lead
        from geometries.route_validation import split_leg_at_points
        _suppress(data, [{'from_dir': 0, 'leg': 'C', 'dir': 0, 'share': 100}])
        ids = iter(['A9'])
        split_leg_at_points(data['segment_data'], 'A', [(14.0, 55.1)], lambda: next(ids), data['traffic_data'])
        assert group_lead(data['segment_data'], 'A9') == 'A'
        assert group_lead(data['segment_data'], 'A') is None
