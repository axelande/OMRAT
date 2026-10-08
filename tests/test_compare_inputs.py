"""Pure tests for the Compare tab's input diff (``omrat_utils.compare_inputs``)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.compare_common import match_legs  # noqa: E402
from omrat_utils.compare_inputs import build_input_table, describe  # noqa: E402

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


def _snapshot():
    with EXAMPLE.open('r', encoding='utf-8') as f:
        return json.load(f)


def _rows(group, rows):
    return [r for r in rows if r[0] == group]


class TestNoDifference:
    def test_identical_snapshots_give_no_rows(self):
        snap = _snapshot()
        assert build_input_table(snap, copy.deepcopy(snap)) == []

    def test_empty_inputs(self):
        assert build_input_table({}, {}) == []
        assert build_input_table(None, None) == []


class TestLegs:
    def test_moved_endpoint_reports_distance(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['segment_data']['1']['End_Point'] = '14.200000 55.187230'
        rows = _rows('Legs', build_input_table(a, b))
        assert len(rows) == 1
        assert rows[0][1] == 'LEG_1_1 (1): end point'
        assert rows[0][2] == '14.158810 55.187230'
        assert rows[0][3].startswith('14.200000 55.187230 (moved 2,6')

    def test_same_id_other_name_is_not_matched(self):
        """Two layouts reuse ids for unrelated legs: list them, never diff them."""
        a = _snapshot()
        b = copy.deepcopy(a)
        b['segment_data']['2']['Leg_name'] = 'LEG_9_9'
        b['segment_data']['2']['Start_Point'] = '20.0 60.0'
        rows = _rows('Legs', build_input_table(a, b))
        assert [r[1] for r in rows] == ['Legs only in A (1)', 'Legs only in B (1)']
        assert rows[0][2] == 'LEG_2_1 (2)'
        assert rows[1][3] == 'LEG_9_9 (2)'
        matched, only_a, only_b = match_legs(a['segment_data'], b['segment_data'])
        assert '2' not in matched and only_a == ['2'] and only_b == ['2']

    def test_check_interval(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['segment_data']['1']['ai2'] = 0
        rows = _rows('Legs', build_input_table(a, b))
        assert rows == [['Legs', 'LEG_1_1 (1): Cat II check interval ai, West going (s)', '180', '0']]


class TestTraffic:
    def test_frequency_change_reports_ships_per_year(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        freq = b['traffic_data']['1']['East going']['Frequency (ships/year)']
        freq[0][0] = (freq[0][0] or 0) + 100
        rows = _rows('Traffic', build_input_table(a, b))
        assert len(rows) == 1
        assert rows[0][1] == 'LEG_1_1 (1) East going: ships/year'

    def test_other_field_change_is_listed(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        speed = b['traffic_data']['1']['West going']['Speed (knots)']
        speed[0][0] = 99.0
        rows = _rows('Traffic', build_input_table(a, b))
        assert rows == [['Traffic', 'LEG_1_1 (1) West going: fields that differ', '—', 'Speed (knots)']]


class TestDistributionAndRouting:
    def test_std_change(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['segment_data']['1']['std1_1'] = 450.0
        rows = _rows('Lateral distribution', build_input_table(a, b))
        assert len(rows) == 1
        assert rows[0][1] == 'LEG_1_1 (1) East going: std (m), component 1'
        assert rows[0][3] == '450'

    def test_suppression_and_redirect(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['segment_data']['1']['suppressed'] = True
        b['segment_data']['1']['traffic_redirect'] = [{'from_dir': 0, 'leg': '2', 'dir': 0, 'share': 100.0}]
        rows = _rows('Routing', build_input_table(a, b))
        items = {r[1]: r for r in rows}
        assert items['LEG_1_1 (1): suppressed'][2:] == ['no', 'yes']
        moved = items['LEG_1_1 (1): traffic moved to']
        assert moved[2] == '—'
        assert moved[3].startswith('East going -> LEG_2_1 (2)')
        assert moved[3].endswith('100 %')


class TestJunctions:
    @staticmethod
    def _with_junction(snap, source, shares):
        snap['junctions'] = {
            'j_1': {'point': [14.15881, 55.18723], 'legs': {'1': 'end', '2': 'start'},
                    'transitions': shares, 'source': source},
        }
        return snap

    def test_source_and_largest_share_change(self):
        a = self._with_junction(_snapshot(), 'geometry', {'1': {'2': 1.0}, '2': {'1': 1.0}})
        b = self._with_junction(_snapshot(), 'ais', {'1': {'2': 0.7}, '2': {'1': 1.0}})
        rows = _rows('Junctions', build_input_table(a, b))
        assert rows[0][1].endswith(': matrix source') and rows[0][2:] == ['geometry', 'ais']
        assert rows[1][1].endswith('share LEG_1_1 (1) -> LEG_2_1 (2)')
        assert rows[1][2:] == ['1.00', '0.70']

    def test_junction_only_in_one_run(self):
        a = _snapshot()
        b = self._with_junction(_snapshot(), 'geometry', {})
        rows = _rows('Junctions', build_input_table(a, b))
        assert rows[0][1] == 'Junctions only in B (1)'
        assert 'LEG_1_1 (1)' in rows[0][3]


class TestDepthsAndStructures:
    def test_duplicated_depth_polygon_keeps_the_area(self):
        """A layer loaded twice doubles the polygon count, not the area."""
        a = _snapshot()
        b = copy.deepcopy(a)
        b['depths'].append(list(b['depths'][0]))
        rows = _rows('Depths', build_input_table(a, b))
        assert rows[0] == ['Depths', 'Depth polygons', '17', '18']
        n_a, area_a = rows[1][2].split(' / ')
        n_b, area_b = rows[1][3].split(' / ')
        assert (n_a, n_b) == ('1', '2')
        assert area_a == area_b

    def test_structure_removed_and_height_changed(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['objects'] = [list(a['objects'][0])]
        b['objects'][0][1] = '25'
        rows = _rows('Structures', build_input_table(a, b))
        items = {r[1]: r for r in rows}
        assert items['Structure 1: height (m)'][2:] == ['10', '25']
        assert items['Structure 2'][3] == '—'
        assert items['Structure 2'][2].startswith('height 10 m, ')

    def test_describe_counts_groups(self):
        rows = [['Legs', 'x', '', ''], ['Legs', 'y', '', ''], ['Depths', 'z', '', '']]
        assert describe(rows) == 'Legs: 2, Depths: 1'
        assert describe([]) == 'none'
