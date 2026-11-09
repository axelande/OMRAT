"""Speed restriction zones: pure logic (runs standalone, no QGIS)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from compute.basic_equations import (
    get_head_on_collision_candidates, get_overtaking_collision_candidates,
)
from compute.speed_restriction import (
    SPEED_AT_END_KEY, SPEED_AT_START_KEY, SPEED_KEY, ZONES_KEY, SpeedRestrictionError,
    apply_speed_restrictions, blocking_problems, cell_limit, describe_rule, describe_zone,
    find_overlaps, leg_exposure, new_rule, new_zone, normalise_zones, restricted_speed,
    rule_matches, speed_matrix, summary_log_lines, summary_markdown, time_equivalent_speed,
    zone_geometry, zone_leg_effects, zone_problems,
)

KN = 1852.0 / 3600.0

INTERVALS = [
    {'min': 0, 'max': 25, 'label': '0-25'},
    {'min': 25, 'max': 100, 'label': '25-100'},
    {'min': 100, 'max': 200, 'label': '100-200'},
    {'min': 200, 'max': '1000+', 'label': '200+'},
]
BOUNDS = [(0.0, 25.0), (25.0, 100.0), (100.0, 200.0), (200.0, math.inf)]
TYPES = ['Cargo', 'Tanker', 'Passenger']


def box(x0, y0, x1, y1):
    return f'Polygon (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))'


def matrix(value, rows=3, cols=4):
    return [[float(value)] * cols for _ in range(rows)]


def leg(start, end, name='LEG_1_1'):
    return {'Start_Point': f'{start[0]} {start[1]}', 'End_Point': f'{end[0]} {end[1]}',
            'Leg_name': name, 'Dirs': ['East going', 'West going']}


def block(speed=12.0, freq=100.0):
    return {'Frequency (ships/year)': matrix(freq), SPEED_KEY: matrix(speed)}


def project(zones, legs=None):
    legs = legs or {'1': leg((0.0, 0.0), (0.1, 0.0))}
    return {
        'segment_data': legs,
        'traffic_data': {k: {'East going': block(), 'West going': block()} for k in legs},
        'ship_categories': {'types': list(TYPES), 'length_intervals': list(INTERVALS)},
        ZONES_KEY: zones,
    }


# --------------------------------------------------------------------------
# Speeds
# --------------------------------------------------------------------------

class TestSpeeds:
    def test_full_compliance_sails_at_the_limit(self):
        assert restricted_speed(12.0, 8.0, 1.0) == pytest.approx(8.0)

    def test_partial_compliance_is_a_time_mix(self):
        # 80 % at 8 kn, 20 % at 12 kn, mixed by 1/V.
        assert restricted_speed(12.0, 8.0, 0.8) == pytest.approx(1.0 / (0.8 / 8.0 + 0.2 / 12.0))

    @pytest.mark.parametrize('speed,limit,compliance', [
        (8.0, 10.0, 1.0),        # already slower
        (12.0, 8.0, 0.0),        # nobody complies
        (math.inf, 8.0, 1.0),    # missing speed stays missing
        (0.0, 8.0, 1.0),
    ])
    def test_unchanged(self, speed, limit, compliance):
        out = restricted_speed(speed, limit, compliance)
        assert out == speed or (math.isinf(out) and math.isinf(speed))

    def test_half_the_leg_at_8_of_12_kn(self):
        assert time_equivalent_speed(12.0, {1: 0.5}, {1: 8.0}) == pytest.approx(9.6)

    def test_no_share_keeps_the_speed(self):
        assert time_equivalent_speed(12.0, {}, {}) == 12.0


# --------------------------------------------------------------------------
# Exactness: the time-equivalent speed gives what a split leg would give
# --------------------------------------------------------------------------

class TestExactAlongTheLeg:
    L = 10_000.0

    def test_head_on_equals_split_leg(self):
        q1, q2, v1, v2, lim = 1000.0, 800.0, 12.0, 15.0, 8.0
        args = dict(mu1=0.0, mu2=0.0, sigma1=200.0, sigma2=250.0, B1=20.0, B2=25.0)
        split = (get_head_on_collision_candidates(q1, q2, v1 * KN, v2 * KN, L_w=self.L / 2, **args)
                 + get_head_on_collision_candidates(q1, q2, lim * KN, lim * KN, L_w=self.L / 2, **args))
        e1 = time_equivalent_speed(v1, {1: 0.5}, {1: lim})
        e2 = time_equivalent_speed(v2, {1: 0.5}, {1: lim})
        whole = get_head_on_collision_candidates(q1, q2, e1 * KN, e2 * KN, L_w=self.L, **args)
        assert whole == pytest.approx(split, rel=1e-12)

    def test_overtaking_equals_split_leg(self):
        qf, qs, vf, vs, lim = 300.0, 900.0, 16.0, 12.0, 10.0
        args = dict(mu_fast=0.0, mu_slow=0.0, sigma_fast=200.0, sigma_slow=200.0, B_fast=20.0, B_slow=25.0)
        outside = get_overtaking_collision_candidates(qf, qs, vf * KN, vs * KN, L_w=self.L / 2, **args)
        inside = get_overtaking_collision_candidates(qf, qs, lim * KN, lim * KN, L_w=self.L / 2, **args)
        assert inside == 0.0   # everyone at the limit: nobody overtakes in the zone
        ef = time_equivalent_speed(vf, {1: 0.5}, {1: lim})
        es = time_equivalent_speed(vs, {1: 0.5}, {1: lim})
        whole = get_overtaking_collision_candidates(qf, qs, ef * KN, es * KN, L_w=self.L, **args)
        assert whole == pytest.approx(outside + inside, rel=1e-12)

    def test_drift_exposure_hours_are_exact(self):
        v, lim = 12.0, 8.0
        hours_split = (self.L / 2) / (v * 1852.0) + (self.L / 2) / (lim * 1852.0)
        e = time_equivalent_speed(v, {1: 0.5}, {1: lim})
        assert self.L / (e * 1852.0) == pytest.approx(hours_split, rel=1e-12)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

class TestGeometry:
    def test_share_and_end_zone(self):
        geom = zone_geometry(new_zone(1, box(0.05, -0.05, 0.2, 0.05)))
        exp = leg_exposure(leg((0.0, 0.0), (0.1, 0.0)), [(1, geom)])
        assert exp is not None
        assert exp.shares[1] == pytest.approx(0.5, abs=1e-6)
        assert exp.start_zone is None and exp.end_zone == 1

    def test_leg_outside(self):
        geom = zone_geometry(new_zone(1, box(1.0, 1.0, 1.1, 1.1)))
        assert leg_exposure(leg((0.0, 0.0), (0.1, 0.0)), [(1, geom)]) is None

    def test_waypoint_on_the_boundary_counts_as_inside(self):
        geom = zone_geometry(new_zone(1, box(0.1, -0.05, 0.2, 0.05)))
        exp = leg_exposure(leg((0.0, 0.0), (0.1, 0.0)), [(1, geom)])
        assert exp is not None and exp.end_zone == 1 and not exp.shares

    def test_shared_edge_is_not_an_overlap(self):
        zones = [new_zone(1, box(0, 0, 1, 1)), new_zone(2, box(1, 0, 2, 1))]
        assert find_overlaps(zones) == []

    def test_overlap_found(self):
        zones = [new_zone(1, box(0, 0, 1, 1)), new_zone(2, box(0.5, 0, 2, 1))]
        found = find_overlaps(zones)
        assert len(found) == 1 and found[0][:2] == (1, 2) and found[0][2] > 1e9

    def test_disabled_zone_is_not_checked(self):
        zones = [new_zone(1, box(0, 0, 1, 1)), dict(new_zone(2, box(0.5, 0, 2, 1)), enabled=False)]
        assert find_overlaps(zones) == []

    def test_bad_wkt(self):
        assert zone_geometry({'wkt': 'not wkt'}) is None
        assert zone_geometry({'wkt': 'Point (1 2)'}) is None


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

class TestRules:
    def zone(self):
        z = new_zone(1, box(0, 0, 1, 1), limit_kn=8.0, compliance_pct=90.0)
        z['rules'] = [
            new_rule(ship_type=2, restricted=False),                 # passenger exempt
            new_rule(length_min=100.0, limit_kn=6.0),               # big ships 6 kn
            new_rule(ship_type=0, length_min=25.0, length_max=100.0, limit_kn=12.0, compliance_pct=50.0),
        ]
        return z

    def test_first_match_wins(self):
        z = self.zone()
        assert cell_limit(z, 2, 3, BOUNDS) is None                  # rule 1 before rule 2
        assert cell_limit(z, 1, 2, BOUNDS) == (6.0, 0.9)            # rule 2, zone compliance
        assert cell_limit(z, 0, 1, BOUNDS) == (12.0, 0.5)           # rule 3
        assert cell_limit(z, 1, 0, BOUNDS) == (8.0, 0.9)            # no rule: zone default

    def test_open_top_class(self):
        rule = new_rule(length_min=100.0, length_max=200.0)
        assert rule_matches(rule, 0, 2, BOUNDS)
        assert not rule_matches(rule, 0, 3, BOUNDS)                 # 200+ is above 200
        assert rule_matches(new_rule(length_min=200.0), 0, 3, BOUNDS)

    def test_length_rule_without_classes_does_not_match(self):
        assert not rule_matches(new_rule(length_min=100.0), 0, 0, [])
        assert rule_matches(new_rule(ship_type=0), 0, 0, [])

    def test_problems(self):
        z = self.zone()
        assert zone_problems(z, TYPES, BOUNDS) == []
        z['rules'].append(new_rule(ship_type=7))
        z['rules'].append(new_rule(length_min=30.0))
        z['rules'].append(new_rule(length_min=100.0, length_max=25.0))
        z['limit_kn'] = 0.0
        problems = zone_problems(z, TYPES, BOUNDS)
        assert 'speed limit must be above 0 kn' in problems
        assert any('rule 4: ship type 7' in p for p in problems)
        assert any('rule 5: 30 m is not a length class boundary' in p for p in problems)
        assert any('rule 6: the length range is empty' in p for p in problems)

    def test_texts(self):
        z = self.zone()
        assert describe_rule(z['rules'][0], TYPES) == 'Passenger, all lengths: exempt'
        assert describe_rule(z['rules'][1], TYPES) == 'All ship types, >= 100 m: 6 kn'
        assert describe_rule(z['rules'][2], TYPES) == 'Cargo, 25-100 m: 12 kn, 50 % compliance'
        assert describe_zone(z) == 'Zone 1: 8 kn, 90 % compliance, 3 rules'


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

class TestNormalise:
    def test_defaults_and_unique_ids(self):
        zones = normalise_zones([{'id': 3, 'wkt': box(0, 0, 1, 1)}, {'id': 3}, 'junk', {}])
        assert [z['id'] for z in zones] == [3, 4, 5]
        assert zones[0]['limit_kn'] == 10.0 and zones[0]['compliance_pct'] == 100.0
        assert zones[0]['enabled'] is True and zones[0]['rules'] == []
        assert zones[1]['name'] == 'Zone 4'

    def test_rules_coerced(self):
        zone = normalise_zones([{'id': 1, 'rules': [{'ship_type': '2', 'limit_kn': '', 'restricted': 0}]}])[0]
        assert zone['rules'] == [new_rule(ship_type=2, restricted=False)]

    def test_not_a_list(self):
        assert normalise_zones(None) == [] and normalise_zones({'a': 1}) == []


# --------------------------------------------------------------------------
# Compute step
# --------------------------------------------------------------------------

class TestApply:
    def test_no_zones_changes_nothing(self):
        data = project([])
        before = [row[:] for row in data['traffic_data']['1']['East going'][SPEED_KEY]]
        summary = apply_speed_restrictions(data)
        assert summary == {'zones': [], 'legs': []}
        assert data['traffic_data']['1']['East going'][SPEED_KEY] == before
        assert SPEED_AT_END_KEY not in data['traffic_data']['1']['East going']

    def test_half_leg_in_zone(self):
        data = project([new_zone(1, box(0.05, -0.05, 0.2, 0.05), limit_kn=8.0)])
        summary = apply_speed_restrictions(data)
        blk = data['traffic_data']['1']['East going']
        assert blk[SPEED_KEY][0][0] == pytest.approx(9.6, rel=1e-5)
        assert blk[SPEED_AT_START_KEY][0][0] == pytest.approx(12.0)
        assert blk[SPEED_AT_END_KEY][0][0] == pytest.approx(8.0)
        (row,) = summary['legs']
        assert row['leg'] == '1' and row['end_zone'] == 1 and row['start_zone'] is None
        assert row['shares'][1] == pytest.approx(0.5, abs=1e-6)
        assert [d['dir'] for d in row['dirs']] == [0, 1]
        assert row['dirs'][0]['before'] == pytest.approx(12.0)
        assert row['dirs'][0]['leg'] == pytest.approx(9.6, rel=1e-5)

    def test_untouched_leg_keeps_its_data(self):
        legs = {'1': leg((0.0, 0.0), (0.1, 0.0)), '2': leg((0.0, 1.0), (0.1, 1.0), 'LEG_2_1')}
        data = project([new_zone(1, box(0.05, -0.05, 0.2, 0.05), limit_kn=8.0)], legs)
        summary = apply_speed_restrictions(data)
        assert [r['leg'] for r in summary['legs']] == ['1']
        assert SPEED_AT_START_KEY not in data['traffic_data']['2']['East going']

    def test_rules_per_cell(self):
        zone = new_zone(1, box(-1, -1, 1, 1), limit_kn=8.0)
        zone['rules'] = [new_rule(ship_type=2, restricted=False)]
        data = project([zone])
        apply_speed_restrictions(data)
        speed = data['traffic_data']['1']['East going'][SPEED_KEY]
        assert speed[0][0] == pytest.approx(8.0, rel=1e-6)       # whole leg inside
        assert speed[2][0] == pytest.approx(12.0)                # passenger exempt

    def test_numpy_matrices_stay_numpy(self):
        data = project([new_zone(1, box(-1, -1, 1, 1), limit_kn=8.0)])
        blk = data['traffic_data']['1']['East going']
        blk[SPEED_KEY] = np.array(blk[SPEED_KEY])
        apply_speed_restrictions(data)
        assert isinstance(blk[SPEED_KEY], np.ndarray)
        assert isinstance(blk[SPEED_AT_END_KEY], np.ndarray)

    def test_missing_speed_stays_missing(self):
        data = project([new_zone(1, box(-1, -1, 1, 1), limit_kn=8.0)])
        data['traffic_data']['1']['East going'][SPEED_KEY][0][0] = math.inf
        apply_speed_restrictions(data)
        assert math.isinf(data['traffic_data']['1']['East going'][SPEED_KEY][0][0])

    def test_overlap_stops_the_run(self):
        data = project([new_zone(1, box(0, -1, 1, 1)), new_zone(2, box(0.5, -1, 2, 1))])
        assert any('cannot overlap' in m for m in blocking_problems(data))
        with pytest.raises(SpeedRestrictionError, match='overlaps'):
            apply_speed_restrictions(data)

    def test_disabled_zone_ignored(self):
        data = project([dict(new_zone(1, box(-1, -1, 1, 1), limit_kn=8.0), enabled=False)])
        assert apply_speed_restrictions(data)['legs'] == []
        assert data['traffic_data']['1']['East going'][SPEED_KEY][0][0] == 12.0

    def test_invalid_enabled_zone_stops_the_run(self):
        data = project([new_zone(1, box(-1, -1, 1, 1), limit_kn=0.0)])
        with pytest.raises(SpeedRestrictionError, match='above 0 kn'):
            apply_speed_restrictions(data)

    def test_speed_matrix_fallback(self):
        blk = block()
        assert speed_matrix(blk, 'end') is blk[SPEED_KEY]
        blk[SPEED_AT_END_KEY] = matrix(8.0)
        assert speed_matrix(blk, 'end') is blk[SPEED_AT_END_KEY]
        assert speed_matrix(blk, 'start') is blk[SPEED_KEY]
        assert speed_matrix(blk) is blk[SPEED_KEY]

    def test_zone_leg_effects_does_not_touch_the_project(self):
        zone = dict(new_zone(1, box(0.05, -0.05, 0.2, 0.05), limit_kn=8.0), enabled=False)
        data = project([zone])
        rows = zone_leg_effects(data, zone)
        assert rows and rows[0]['leg'] == '1'
        assert data['traffic_data']['1']['East going'][SPEED_KEY][0][0] == 12.0

    def test_report_texts(self):
        data = project([new_zone(1, box(0.05, -0.05, 0.2, 0.05), name='Anholt', limit_kn=8.0)])
        summary = apply_speed_restrictions(data)
        md = '\n'.join(summary_markdown(summary, TYPES))
        assert '## Speed restriction zones' in md
        assert 'Zone 1 "Anholt": 8 kn, 100 % compliance, all ships' in md
        assert '| LEG_1_1 | East going | Zone 1 "Anholt" 50 % | 12.0 | 9.6 | 12.0 | 8.0 |' in md
        log = summary_log_lines(summary)
        assert log[0] == 'Speed restriction zones: 1 enabled, 1 leg(s) affected'
        assert summary_markdown({'zones': [], 'legs': []}) == []
