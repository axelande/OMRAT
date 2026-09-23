"""Pure tests for ``omrat_utils.traffic_links`` (labels + map links)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.traffic_links import (  # noqa: E402
    build_links, curve_points, leg_label, link_geometries, related_legs, status_suffix,
)


def _leg(seg_id, start, end, **extra):
    d = {'Leg_name': f'LEG_{seg_id}', 'Start_Point': start, 'End_Point': end,
         'Dirs': ['North going', 'South going']}
    d.update(extra)
    return d


@pytest.fixture
def segs():
    """7_3_b measured; 7_2 is a locked copy of it; 7_3_b is suppressed onto
    2_6 (both directions) and 7_3_a goes with it."""
    return {
        '1': _leg('7_3_b', '14.0 55.0', '14.0 55.1', suppressed=True, traffic_redirect=[
            {'from_dir': 0, 'leg': '3', 'dir': 0, 'share': 100},
            {'from_dir': 1, 'leg': '3', 'dir': 1, 'share': 80},
        ]),
        '2': _leg('7_2', '14.0 55.1', '14.0 55.2', traffic_locked=True, traffic_source='1'),
        '3': _leg('2_6', '14.2 55.0', '14.2 55.2'),
        '4': _leg('7_3_a', '14.0 54.9', '14.0 55.0', suppressed=True, suppressed_with='1'),
        '5': _leg('plain', '15.0 55.0', '15.0 55.1'),
    }


def test_links_of_every_kind(segs):
    links = {(lk.kind, lk.src, lk.dst): lk.label for lk in build_links(segs)}
    assert links == {
        ('copy', '1', '2'): 'copy (locked)',
        ('redirect', '1', '3'): 'N 100 %, S 80 %',
        ('with', '4', '1'): 'suppressed with',
    }


def test_labels_name_the_referenced_leg(segs):
    assert status_suffix('2', segs) == '  [locked, copy of LEG_7_3_b]'
    assert status_suffix('4', segs) == '  [suppressed with LEG_7_3_b]'
    assert status_suffix('1', segs) == '  [suppressed -> LEG_2_6, +1 leg(s) with it]'
    assert status_suffix('5', segs) == ''
    assert leg_label('5', segs) == 'LEG_plain  (id 5)'


def test_links_to_missing_legs_are_dropped(segs):
    segs['2']['traffic_source'] = 'gone'
    segs['1']['traffic_redirect'].append({'from_dir': 0, 'leg': 'gone', 'dir': 0, 'share': 50})
    kinds = sorted(lk.kind for lk in build_links(segs))
    assert kinds == ['redirect', 'with']


def test_related_legs_both_ways(segs):
    assert related_legs('1', segs) == {'2', '3', '4'}
    assert related_legs('3', segs) == {'1'}
    assert related_legs('5', segs) == set()


def test_curve_starts_and_ends_on_the_midpoints(segs):
    geoms = dict(((lk.kind, lk.src, lk.dst), pts) for lk, pts in link_geometries(segs))
    pts = geoms[('redirect', '1', '3')]
    assert pts[0] == pytest.approx((14.0, 55.05))
    assert pts[-1] == pytest.approx((14.2, 55.1))
    assert len(pts) == 17


def test_curve_bows_to_the_left():
    pts = curve_points((0.0, 0.0), (0.0, 1.0), bend=0.2)
    mid = pts[len(pts) // 2]
    assert mid[0] < 0          # travelling north, left = west
    assert mid[1] == pytest.approx(0.5)
