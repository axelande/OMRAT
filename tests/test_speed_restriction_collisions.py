"""Speed restriction zones in the real ship-ship collision model (standalone).

Four legs meet at junction J (the project of ``test_compare_explain.py``),
one ship cell per direction at 12 kn.  ``compare_explain.run_collision_model``
runs the same traffic steps as Run model (``compute/traffic_pipeline.py``),
so these tests cover the pipeline and the model together.

Every encounter scales as ``1/V`` when all speeds change together, so a
zone at 8 kn multiplies an encounter it covers by exactly 12 / 8 = 1.5.
"""
from __future__ import annotations

import copy

import pytest

from compute.speed_restriction import ZONES_KEY, SpeedRestrictionError, new_rule, new_zone
from geometries.junctions import apply_geometric_defaults, build_junctions, serialize_junctions
from omrat_utils.compare_common import haversine_m
from omrat_utils.compare_explain import run_collision_model

PTS = {'W': (14.0, 55.0), 'J': (14.5, 55.0), 'E': (15.0, 55.0), 'N': (14.5, 55.4), 'NE': (15.0, 55.15)}


def _xy(p):
    return f'{p[0]:.6f} {p[1]:.6f}'


def _seg(seg_id, name, a, b, dirs=('East going', 'West going')):
    return {
        'Segment_Id': seg_id, 'Leg_name': name, 'Route_Id': 1,
        'Start_Point': _xy(PTS[a]), 'End_Point': _xy(PTS[b]),
        'line_length': haversine_m(PTS[a], PTS[b]), 'Width': 5000, 'Dirs': list(dirs),
        'mean1_1': 0.0, 'std1_1': 500.0, 'weight1_1': 1.0,
        'mean2_1': 0.0, 'std2_1': 500.0, 'weight2_1': 1.0,
        'u_p1': 0, 'u_p2': 0, 'u_min1': 0, 'u_max1': 0, 'u_min2': 0, 'u_max2': 0,
        'ai1': 180.0, 'ai2': 180.0,
    }


def _dir(q, speed=12.0):
    return {
        'Frequency (ships/year)': [[q]], 'Speed (knots)': [[speed]],
        'Ship Beam (meters)': [[20.0]], 'Draught (meters)': [[6.0]],
        'Ship heights (meters)': [[20.0]], 'Scaling (%)': [[100.0]],
    }


def _project(zones=()):
    segs = {
        '1': _seg('1', 'LEG_1_1', 'W', 'J'),
        '2': _seg('2', 'LEG_1_2', 'J', 'E'),
        '3': _seg('3', 'LEG_2_1', 'J', 'N', dirs=('North going', 'South going')),
        '4': _seg('4', 'LEG_4_1', 'J', 'NE'),
    }
    q = {'1': 2000.0, '2': 1500.0, '3': 500.0, '4': 800.0}
    traffic = {k: {d: _dir(q[k]) for d in v['Dirs']} for k, v in segs.items()}
    junctions = build_junctions(segs)
    apply_geometric_defaults(junctions, segs)
    return {
        'segment_data': segs, 'traffic_data': traffic,
        'junctions': serialize_junctions(junctions),
        'pc': {'headon': 4.9e-5, 'overtaking': 1.1e-4, 'crossing': 1.3e-4,
               'merging': 1.3e-4, 'bend': 1.3e-4, 'grounding': 1.6e-4},
        'ship_categories': {'types': ['Cargo'], 'length_intervals': [{'min': 100, 'max': 150, 'label': '100-150'}]},
        'depths': [], 'objects': [],
        ZONES_KEY: list(zones),
    }


def _box(cx, cy, half):
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    return f'Polygon (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))'


def _totals(zones=()):
    report = run_collision_model(copy.deepcopy(_project(zones)))
    return report['totals'], report


BASE, _ = _totals()


def test_base_project_has_every_encounter():
    for key in ('head_on', 'crossing', 'merging', 'bend'):
        assert BASE[key] > 0.0, key


def test_zone_at_the_junction_scales_waypoint_encounters_by_v_ratio():
    # ~100 m box around J: the waypoint is inside, the legs hardly are.
    totals, _ = _totals([new_zone(1, _box(*PTS['J'], 0.001), limit_kn=8.0)])
    for key in ('crossing', 'merging', 'bend'):
        assert totals[key] == pytest.approx(BASE[key] * 1.5, rel=1e-9), key
    # Head-on only sees the ~0.2 % of each leg inside the box.
    assert BASE['head_on'] < totals['head_on'] < BASE['head_on'] * 1.01


def test_zone_over_everything_scales_head_on_too():
    totals, _ = _totals([new_zone(1, _box(14.5, 55.2, 1.0), limit_kn=8.0)])
    for key in ('head_on', 'crossing', 'merging', 'bend'):
        assert totals[key] == pytest.approx(BASE[key] * 1.5, rel=1e-9), key


def test_zone_away_from_the_junction_leaves_waypoint_encounters():
    # Box over the middle of LEG_1_1 only (W..J), not touching J.
    totals, report = _totals([new_zone(1, _box(14.25, 55.0, 0.05), limit_kn=8.0)])
    for key in ('crossing', 'merging', 'bend'):
        assert totals[key] == pytest.approx(BASE[key], rel=1e-12), key
    assert totals['head_on'] > BASE['head_on']
    by_leg = report['by_leg']
    assert by_leg['2']['head_on'] == pytest.approx(_totals()[1]['by_leg']['2']['head_on'], rel=1e-12)


def test_exempt_ships_keep_their_speed():
    zone = new_zone(1, _box(14.5, 55.2, 1.0), limit_kn=8.0)
    zone['rules'] = [new_rule(ship_type=0, restricted=False)]
    totals, _ = _totals([zone])
    for key in ('head_on', 'crossing', 'merging', 'bend'):
        assert totals[key] == pytest.approx(BASE[key], rel=1e-12), key


def test_disabled_zone_changes_nothing():
    zone = dict(new_zone(1, _box(14.5, 55.2, 1.0), limit_kn=8.0), enabled=False)
    totals, _ = _totals([zone])
    for key in ('head_on', 'crossing', 'merging', 'bend'):
        assert totals[key] == pytest.approx(BASE[key], rel=1e-12), key


def test_overlapping_zones_refuse_to_run():
    zones = [new_zone(1, _box(14.5, 55.0, 0.1)), new_zone(2, _box(14.55, 55.0, 0.1))]
    with pytest.raises(SpeedRestrictionError, match='cannot overlap'):
        _totals(zones)
