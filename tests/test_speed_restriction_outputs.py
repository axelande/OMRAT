"""Speed restriction zones in the outputs (standalone, no QGIS).

Traffic report summary + HTML, Compare settings diff, "More info" input
groups, the run report section and the sensitivity parameter.
"""
from __future__ import annotations

import copy

import pytest

from compute.sensitivity import apply_perturbation, baseline_value, build_parameter_specs
from compute.speed_restriction import (
    ZONES_KEY, apply_speed_restrictions, new_rule, new_zone, settings_view, zone_area_km2,
)
from omrat_utils import compare_explain as ce
from omrat_utils.compare import build_settings_table
from omrat_utils.full_report import build_full_report_markdown
from omrat_utils.traffic_report_html import render_traffic_report_html
from omrat_utils.traffic_summary import CHECK_TEXT, build_traffic_summary


def _box(x0, y0, x1, y1):
    return f'Polygon (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))'


def _block(speed=12.0, q=100.0):
    return {'Frequency (ships/year)': [[q]], 'Speed (knots)': [[speed]], 'Draught (meters)': [[6.0]],
            'Ship heights (meters)': [[20.0]], 'Ship Beam (meters)': [[20.0]], 'Scaling (%)': [[100.0]]}


def _project(zones=()):
    seg = {'Segment_Id': '1', 'Route_Id': 1, 'Leg_name': 'LEG_1_1',
           'Start_Point': '0.000000 0.000000', 'End_Point': '0.100000 0.000000',
           'line_length': 11132.0, 'Dirs': ['East going', 'West going'],
           'mean1_1': 0.0, 'std1_1': 200.0, 'weight1_1': 1.0,
           'mean2_1': 0.0, 'std2_1': 200.0, 'weight2_1': 1.0, 'ai1': 180.0, 'ai2': 180.0}
    return {
        'segment_data': {'1': seg},
        'traffic_data': {'1': {'East going': _block(), 'West going': _block()}},
        'ship_categories': {'types': ['Cargo'], 'length_intervals': [{'min': 0, 'max': 100, 'label': '0-100'}]},
        'pc': {}, 'drift': {}, 'depths': [], 'objects': [],
        ZONES_KEY: [copy.deepcopy(z) for z in zones],
    }


HALF = new_zone(1, _box(0.05, -0.05, 0.2, 0.05), name='Anholt', limit_kn=8.0)


class TestTrafficReport:
    def test_leg_in_a_zone_shows_the_calculated_speed(self):
        summary = build_traffic_summary(_project([HALF]))
        (leg,) = summary['legs']
        assert leg['mean_speed'] == pytest.approx(12.0)
        assert leg['speed_zone']['zones'] == 'Zone 1 "Anholt" 50 %'
        assert leg['speed_zone']['calc_speed'] == pytest.approx(9.6, rel=1e-4)
        codes = [c['code'] for c in summary['checks']]
        assert 'speed_zone' in codes and 'speed_zone_error' not in codes
        assert set(CHECK_TEXT) >= {'speed_zone', 'speed_zone_error'}
        html = render_traffic_report_html(summary)
        assert '9.6 in the calculation' in html
        assert 'Speed restriction zone' in html

    def test_without_zones_nothing_changes(self):
        summary = build_traffic_summary(_project())
        assert summary['legs'][0]['speed_zone'] is None
        assert not [c for c in summary['checks'] if c['code'].startswith('speed_zone')]

    def test_overlap_is_a_warning(self):
        zones = [new_zone(1, _box(-1, -1, 1, 1)), new_zone(2, _box(0.5, -1, 2, 1))]
        summary = build_traffic_summary(_project(zones))
        (check,) = [c for c in summary['checks'] if c['code'] == 'speed_zone_error']
        assert check['level'] == 'warning' and 'cannot overlap' in check['text']
        assert summary['legs'][0]['speed_zone'] is None


class TestCompare:
    def test_settings_rows_by_zone_id(self):
        b = copy.deepcopy(HALF)
        b['limit_kn'] = 6.0
        b['rules'] = [new_rule(ship_type=0, restricted=False)]
        rows = build_settings_table(_project([HALF]), _project([b]))
        labels = {r[0]: (r[1], r[2]) for r in rows}
        assert labels['Speed restriction zones: Zone 1.limit_kn'] == ('8', '6')
        assert labels['Speed restriction zones: Zone 1.rules[0]'] == ('—', 'Cargo, all lengths: exempt')
        assert not any('wkt' in r[0] for r in rows)

    def test_zone_only_in_one_run(self):
        rows = build_settings_table(_project(), _project([HALF]))
        assert any(r[0] == 'Speed restriction zones: Zone 1.limit_kn' and r[1] == '—' for r in rows)

    def test_settings_view_summarises_the_polygon(self):
        view = settings_view([HALF], ['Cargo'])
        assert view['Zone 1']['polygon'].startswith('5 points, ')
        assert zone_area_km2(HALF) == pytest.approx(0.15 * 0.1 * 111.32 * 110.57, rel=0.01)

    def test_more_info_has_a_zone_group(self):
        a, b = _project(), _project([HALF])
        assert 'speed_zones' in ce.active_groups(a, b)
        mixed = ce.hybrid_data(a, b, ('speed_zones',))
        assert mixed[ZONES_KEY] == b[ZONES_KEY]
        assert ce.GROUP_LABELS['speed_zones'] == 'Speed restriction zones'


class TestRunReport:
    def test_section_only_with_zones(self):
        data = _project([HALF])
        summary = apply_speed_restrictions(copy.deepcopy(data))
        kwargs = dict(run_name='r', timestamp='t', data=data, drifting_md=None,
                      powered_grounding_report=None, powered_allision_report=None, collision_report=None)
        md = build_full_report_markdown(speed_restrictions=summary, **kwargs)
        assert '## Speed restriction zones' in md
        assert '| LEG_1_1 | East going |' in md
        assert '## Speed restriction zones' not in build_full_report_markdown(**kwargs)


class TestSensitivity:
    def test_limit_parameter(self):
        zone = copy.deepcopy(HALF)
        zone['rules'] = [new_rule(ship_type=0, limit_kn=6.0), new_rule(restricted=False)]
        data = _project([zone, dict(new_zone(2, _box(5, 5, 6, 6), limit_kn=4.0), enabled=False)])
        (spec,) = [s for s in build_parameter_specs(data) if s.key == 'zones.limit']
        assert spec.phases and spec.unit == 'knots'
        assert baseline_value(data, spec) == pytest.approx(8.0)       # enabled zones only
        out = apply_perturbation(data, spec, 1.1)
        z1, z2 = out[ZONES_KEY]
        assert z1['limit_kn'] == pytest.approx(8.8)
        assert z1['rules'][0]['limit_kn'] == pytest.approx(6.6)
        assert z1['rules'][1]['limit_kn'] is None
        assert z2['limit_kn'] == 4.0
        assert data[ZONES_KEY][0]['limit_kn'] == 8.0                  # input untouched

    def test_no_parameter_without_enabled_zones(self):
        data = _project([dict(HALF, enabled=False)])
        assert not [s for s in build_parameter_specs(data) if s.key == 'zones.limit']
