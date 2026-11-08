"""Pure tests for ``omrat_utils.traffic_summary`` (the Traffic report's numbers and checks)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.traffic_summary import (  # noqa: E402
    build_traffic_summary, fmt_ships, legs_csv, short_type, types_csv,
)

INF = float('inf')
DIRS = ['East going', 'West going']


def _block(freq, speed=None, draught=None, scaling=None):
    shape = [[0.0] * len(freq[0]) for _ in freq]
    return {
        'Frequency (ships/year)': copy.deepcopy(freq),
        'Speed (knots)': copy.deepcopy(speed) if speed is not None else [[12.0] * len(r) for r in freq],
        'Draught (meters)': copy.deepcopy(draught) if draught is not None else [[6.0] * len(r) for r in freq],
        'Ship heights (meters)': copy.deepcopy(shape),
        'Ship Beam (meters)': copy.deepcopy(shape),
        'Scaling (%)': copy.deepcopy(scaling) if scaling is not None else [[100.0] * len(r) for r in freq],
    }


def _leg(seg, start, end, **extra):
    d = {'Segment_Id': seg, 'Leg_name': f'LEG_{seg}', 'Route_Id': '1', 'Start_Point': start, 'End_Point': end,
         'Dirs': list(DIRS)}
    d.update(extra)
    return d


@pytest.fixture
def project():
    """Two ship types x two length classes.

    1 -> 2 -> 3 run east and meet alone at their waypoints.  2 is a true
    copy of 1, 3 a copy of 1 that no longer matches it (and carries far
    more ships than 2).  4 is suppressed: East going moves to 5, West going
    has no target.  5 is scaled to 50 %.  6 has no traffic, 7 has ships one
    way only and no speed for them, 8 was copied from a removed leg.
    """
    a = {'East going': _block([[100, 0], [50, 0]]), 'West going': _block([[80, 0], [0, 20]])}
    segs = {
        '1': _leg('1', '14.0 55.0', '14.1 55.0'),
        '2': _leg('2', '14.1 55.0', '14.2 55.0', traffic_source='1', traffic_locked=True),
        '3': _leg('3', '14.2 55.0', '14.3 55.0', traffic_source='1'),
        '4': _leg('4', '16.0 57.0', '16.0 57.1', suppressed=True,
                  traffic_redirect=[{'from_dir': 0, 'leg': '5', 'dir': 0, 'share': 100}]),
        '5': _leg('5', '15.0 56.0', '15.1 56.0'),
        '6': _leg('6', '17.0 58.0', '17.1 58.0'),
        '7': _leg('7', '18.0 58.0', '18.1 58.0'),
        '8': _leg('8', '19.0 58.0', '19.1 58.0', traffic_source='99'),
    }
    traffic = {
        '1': copy.deepcopy(a),
        '2': copy.deepcopy(a),
        '3': {'East going': _block([[1000, 0], [0, 0]]), 'West going': _block([[1000, 0], [0, 0]])},
        '4': {'East going': _block([[40, 0], [0, 0]]), 'West going': _block([[0, 30], [0, 0]])},
        '5': {'East going': _block([[200, 0], [0, 0]], scaling=[[50.0, 50.0], [50.0, 50.0]]),
              'West going': _block([[0, 0], [0, 100]], scaling=[[50.0, 50.0], [50.0, 50.0]])},
        '7': {'East going': _block([[10, 0], [0, 0]], speed=[[INF, INF], [INF, INF]]),
              'West going': _block([[0, 0], [0, 0]])},
        '8': {'East going': _block([[5, 0], [0, 0]]), 'West going': _block([[5, 0], [0, 0]])},
    }
    return {
        'project_name': 'Test project', 'omrat_version': '9.9',
        'segment_data': segs, 'traffic_data': traffic,
        'ship_categories': {'types': ['Cargo, all ships of this type', 'Tanker, all ships of this type'],
                            'length_intervals': [{'min': 0, 'max': 100, 'label': '0 - 100'},
                                                 {'min': 100, 'max': 200, 'label': '100 - 200'}]},
        'traffic_scaling': {'global_percent': 100.0, 'follow_global': [True, True]},
    }


@pytest.fixture
def summary(project):
    return build_traffic_summary(project, project_path='C:/p/test.omrat')


def _leg_of(summary, seg):
    return next(leg for leg in summary['legs'] if leg['id'] == seg)


def _codes(summary, seg=None):
    return [c['code'] for c in summary['checks'] if seg is None or seg in c['legs']]


def test_input_is_not_changed(project):
    before = copy.deepcopy(project)
    build_traffic_summary(project)
    assert json.dumps(project, sort_keys=True, default=str) == json.dumps(before, sort_keys=True, default=str)


def test_stored_totals_per_direction(summary):
    leg = _leg_of(summary, '1')
    assert [d['ships'] for d in leg['dirs']] == [150, 100]
    assert [d['label'] for d in leg['dirs']] == DIRS
    assert leg['ships'] == 250
    assert leg['by_type'] == [180, 70]
    assert [(i, round(share, 3)) for i, _q, share in leg['top_types']] == [(0, 0.72), (1, 0.28)]
    assert leg['mean_speed'] == pytest.approx(12.0)
    assert leg['length_m'] == pytest.approx(6380, rel=0.01)   # 0.1 deg of longitude at 55 N


def test_calculation_view_scaling_and_redirect(summary):
    sup = _leg_of(summary, '4')
    assert sup['calc'] is None and sup['suppressed']
    target = _leg_of(summary, '5')
    # Scaled to 50 %: 100 + 50, plus 40 ships moved in East going.
    assert target['scaled'] == pytest.approx(150)
    assert [d['calc'] for d in target['dirs']] == [pytest.approx(140), pytest.approx(50)]
    assert target['calc'] == pytest.approx(190)
    assert target['dirs'][0]['received'] == pytest.approx(40)
    assert target['received_moves'][0]['src'] == '4'
    assert summary['calc_differs'] is True


def test_kinds_and_source_text(summary):
    assert (_leg_of(summary, '1')['kind'], _leg_of(summary, '1')['calc_kind']) == ('own', 'own')
    assert _leg_of(summary, '2')['kind'] == 'copy'
    assert _leg_of(summary, '2')['source_text'] == 'Copy of LEG_1 (locked)'
    assert _leg_of(summary, '4')['kind'] == 'suppressed'
    assert 'East 100 % -> LEG_5' in _leg_of(summary, '4')['source_text']
    assert _leg_of(summary, '5')['calc_kind'] == 'received'
    assert 'receives ships from suppressed LEG_4' in _leg_of(summary, '5')['source_text']
    assert _leg_of(summary, '6')['kind'] == 'none'
    assert _leg_of(summary, '6')['source_text'] == 'No traffic'


def test_checks(summary):
    assert 'no_traffic' in _codes(summary, '6')
    assert 'one_direction' in _codes(summary, '7')
    assert 'copy_stale' in _codes(summary, '3')
    assert 'copy_stale' not in _codes(summary, '2')        # a true copy is fine
    assert 'copy_source_missing' in _codes(summary, '8')
    assert 'suppressed' in _codes(summary, '4')
    assert 'suppressed_lost' in _codes(summary, '4')
    assert 'scaling' in _codes(summary, '5')
    assert 'locked' in _codes(summary, '2')
    gaps = [c for c in summary['checks'] if c['code'] == 'gaps']
    assert [c['legs'] for c in gaps] == [['7']]
    assert gaps[0]['level'] == 'warning'                  # all of leg 7's ships lack a speed
    assert 'speed' in gaps[0]['text']
    levels = [c['level'] for c in summary['checks']]
    assert levels == sorted(levels, key=lambda lv: lv != 'warning')   # warnings first


def test_jump_only_between_legs_meeting_alone(summary):
    jumps = [c for c in summary['checks'] if c['code'] == 'jump']
    assert [sorted(c['legs']) for c in jumps] == [['2', '3']]   # 1 and 2 carry the same ships
    assert '8.0 times' in jumps[0]['text']


def test_jump_ignores_small_differences(project):
    project['traffic_data']['3'] = copy.deepcopy(project['traffic_data']['1'])
    project['traffic_data']['3']['East going']['Frequency (ships/year)'][0][0] = 190   # 340 vs 250
    assert 'jump' not in _codes(build_traffic_summary(project))


def test_jump_skips_waypoints_with_more_legs(project):
    project['segment_data']['9'] = _leg('9', '14.2 55.0', '14.2 55.1')
    project['traffic_data']['9'] = copy.deepcopy(project['traffic_data']['1'])
    assert 'jump' not in _codes(build_traffic_summary(project), '3')


def test_few_ships_without_draught_is_a_note(project):
    block = project['traffic_data']['1']['East going']
    block['Draught (meters)'][1][0] = INF          # 50 of 250 ships -> warning
    s = build_traffic_summary(project)
    assert [c['level'] for c in s['checks'] if c['code'] == 'gaps' and 'without a draught' in c['text']] == ['warning']
    block['Frequency (ships/year)'][1][0] = 2      # 2 of 202 -> note
    s = build_traffic_summary(project)
    assert [c['level'] for c in s['checks'] if c['code'] == 'gaps' and 'without a draught' in c['text']] == ['note']


def test_copy_with_swapped_directions_matches(project):
    a = project['traffic_data']['1']
    project['traffic_data']['2'] = {'East going': copy.deepcopy(a['West going']),
                                    'West going': copy.deepcopy(a['East going'])}
    assert 'copy_stale' not in _codes(build_traffic_summary(project), '2')


def test_figures(summary):
    f = summary['figures']
    assert (f['legs'], f['in_calc'], f['suppressed']) == (8, 7, 1)
    assert f['busiest'] == ('3', 'LEG_3', 2000)
    assert f['no_traffic'] == 1
    assert (f['copies'], f['locked']) == (3, 1)
    assert f['warnings'] == sum(1 for c in summary['checks'] if c['level'] == 'warning')
    assert summary['type_index'] == [0, 1]
    assert summary['project']['name'] == 'Test project'


def test_checks_point_back_to_legs(summary):
    for leg in summary['legs']:
        for idx in leg['checks']:
            assert leg['id'] in summary['checks'][idx]['legs']


def test_no_calc_difference_without_scaling_or_suppression(project):
    project['segment_data']['4']['suppressed'] = False
    for block in project['traffic_data']['5'].values():
        block['Scaling (%)'] = [[100.0, 100.0], [100.0, 100.0]]
    assert build_traffic_summary(project)['calc_differs'] is False


def test_csv_follows_the_decimal_point(summary):
    comma = legs_csv(summary, decimal=',')
    head, first = comma.splitlines()[:2]
    assert head.startswith('Leg;Id;Route;Length (km)')
    assert first.startswith('LEG_1;1;1;6,')
    assert ';12,0;6,0;' in first
    dot = legs_csv(summary)
    assert dot.splitlines()[1].startswith('LEG_1,1,1,6.')
    assert 'Copy of LEG_1 (locked)' in dot
    types = types_csv(summary).splitlines()
    assert types[0] == 'Leg,Id,"Cargo, all ships of this type","Tanker, all ships of this type",Total'
    assert types[1] == 'LEG_1,1,180.0,70.0,250.0'


def test_formatting_helpers():
    assert fmt_ships(None) == '-'
    assert fmt_ships(0) == '0'
    assert fmt_ships(3.0) == '3'
    assert fmt_ships(2.5) == '2.5'
    assert fmt_ships(12345.6) == '12,346'
    assert short_type('Passenger, all ships of this type') == 'Passenger'


def test_empty_project():
    s = build_traffic_summary({})
    assert s['legs'] == [] and s['checks'] == [] and s['figures']['busiest'] is None


def test_example_project():
    data = json.loads((ROOT / 'tests' / 'example_data' / 'proj.omrat').read_text(encoding='utf-8'))
    s = build_traffic_summary(data)
    assert [leg['name'] for leg in s['legs']] == ['LEG_1_1', 'LEG_2_1', 'LEG_3_1', 'LEG_4_1']
    assert all(leg['ships'] > 0 for leg in s['legs'])
    assert s['figures']['busiest'][1] == 'LEG_4_1'
    jumps = [c for c in s['checks'] if c['code'] == 'jump']
    assert [sorted(c['legs']) for c in jumps] == [['1', '2']]
