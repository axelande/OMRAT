"""Speed restriction zones in the powered model: Cat II at the turning point.

Direction 1 sails ``Start_Point -> End_Point`` and turns at the end;
direction 2 turns at the start (``compute.speed_restriction.turning_point``).
Cat II (``mass * exp(-d / (ai * V))``) must use the speed *there*; Cat I has
no speed term.  ``geometries/get_powered_overlap.py`` imports matplotlib's
Qt backend, so run this file with the QGIS conftest like
``test_cat1_start_zone.py``.
"""
from __future__ import annotations

import math

import pytest

from compute.powered_model import PoweredModelMixin
from compute.speed_restriction import (
    SPEED_AT_END_KEY, SPEED_AT_START_KEY, SPEED_KEY, ZONES_KEY, apply_speed_restrictions,
    new_zone, turning_point,
)
from geometries.get_powered_overlap import SimpleProjector, _build_leg_entry

KN = 1852.0 / 3600.0
AI = 180.0
D_MEAN = 2000.0


def _block(speed=12.0, q=100.0):
    return {'Frequency (ships/year)': [[q]], SPEED_KEY: [[speed]], 'Ship heights (meters)': [[0.0]]}


def _box(x0, y0, x1, y1):
    return f'Polygon (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))'


def _data(zone_wkt=None):
    seg = {'Start_Point': '0.000000 0.000000', 'End_Point': '0.100000 0.000000',
           'Leg_name': 'LEG_1_1', 'Dirs': ['East going', 'West going'], 'ai1': AI, 'ai2': AI,
           'mean1_1': 0.0, 'std1_1': 100.0, 'mean2_1': 0.0, 'std2_1': 100.0}
    data = {'segment_data': {'1': seg},
            'traffic_data': {'1': {'East going': _block(), 'West going': _block()}},
            'ship_categories': {'types': ['Cargo'], 'length_intervals': [{'min': 0, 'max': 100, 'label': '0-100'}]},
            ZONES_KEY: [new_zone(1, zone_wkt, limit_kn=8.0)] if zone_wkt else []}
    apply_speed_restrictions(data)
    return data


AT_END = _box(0.09, -0.01, 0.2, 0.01)      # holds End_Point only
AT_START = _box(-0.1, -0.01, 0.01, 0.01)   # holds Start_Point only


def test_turning_points():
    assert turning_point(0) == 'end' and turning_point(1) == 'start'


def test_zone_at_the_end_sets_the_end_speed_only():
    blk = _data(AT_END)['traffic_data']['1']['East going']
    assert blk[SPEED_AT_END_KEY][0][0] == pytest.approx(8.0)
    assert blk[SPEED_AT_START_KEY][0][0] == pytest.approx(12.0)


def test_ray_caster_takes_the_turning_point_speed():
    data = _data(AT_END)
    entry = _build_leg_entry('1', data['segment_data']['1'], data['traffic_data'], SimpleProjector(0.0, 0.0))
    east, west = entry['dirs']
    assert east['speed_kn'] == pytest.approx(8.0)     # turns at End_Point, inside the zone
    assert west['speed_kn'] == pytest.approx(12.0)    # turns at Start_Point, outside


class _Host(PoweredModelMixin):
    def __init__(self):  # the sums need no parent widget
        pass


def _comp(dir_idx):
    key = ('object', 'park')
    return {'seg_id': '1', 'dir_idx': dir_idx, 'dir_info': {'ai': AI},
            'summaries': {key: {'mass': 0.5, 'mean_dist': D_MEAN, 'edges': {}}},
            'cat1': {'summaries': {key: {'mass': 0.1, 'edges': {}}}}}


def _cat_totals(data, dir_idx):
    acc = _Host()._sum_allision_contribs([_comp(dir_idx)], data['traffic_data'], {}, 1.0, 1.0)
    report = acc.report('allision', 1.0, 1.0)
    return report['totals']['cat1'], report['totals']['cat2']


@pytest.mark.parametrize('zone,dir_idx,speed', [
    (None, 0, 12.0), (None, 1, 12.0),
    (AT_END, 0, 8.0), (AT_END, 1, 12.0),
    (AT_START, 0, 12.0), (AT_START, 1, 8.0),
])
def test_cat2_uses_the_turning_point_speed(zone, dir_idx, speed):
    cat1, cat2 = _cat_totals(_data(zone), dir_idx)
    assert cat1 == pytest.approx(100.0 * 0.1)                                  # no speed term
    assert cat2 == pytest.approx(100.0 * 0.5 * math.exp(-D_MEAN / (AI * speed * KN)))
