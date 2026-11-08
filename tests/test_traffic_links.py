"""Pure tests for ``omrat_utils.traffic_links`` (labels + map links)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.traffic_links import (  # noqa: E402
    VIEW_ALL, VIEW_LEG, build_links, curve_points, leg_label, link_geometries, link_groups, links_in_view,
    related_legs, resolve_view, status_suffix, step_view, view_entries,
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


# ---------------------------------------------------------------------------
# One group at a time
# ---------------------------------------------------------------------------

def test_every_link_is_in_exactly_one_group(segs):
    groups = link_groups(segs)
    assert [g.key for g in groups] == ['copy:1', 'suppress:1']   # same origin: copy first
    in_groups = [lk for g in groups for lk in g.links]
    assert sorted(in_groups, key=repr) == sorted(build_links(segs), key=repr)
    assert groups[0].title == 'LEG_7_3_b copied to LEG_7_2'
    assert groups[1].title == 'LEG_7_3_b suppressed -> LEG_2_6, +1 leg(s) with it'
    assert groups[1].legs() == ['1', '3', '4']                    # origin first


def test_copy_chain_is_one_group_named_after_its_root(segs):
    segs['6'] = _leg('7_1', '14.0 55.2', '14.0 55.3', traffic_source='2')
    segs['7'] = _leg('7_0', '14.0 55.3', '14.0 55.4', traffic_source='1')
    copy_group = link_groups(segs)[0]
    assert copy_group.key == 'copy:1'
    assert {(lk.src, lk.dst) for lk in copy_group.links} == {('1', '2'), ('2', '6'), ('1', '7')}
    assert copy_group.title == 'LEG_7_3_b copied to LEG_7_2, LEG_7_1 +1'


def test_groups_follow_the_route_table_order(segs):
    segs['5']['traffic_source'] = '3'          # 2_6 copied to "plain"
    keys = [g.key for g in link_groups(segs)]
    assert keys == ['copy:1', 'suppress:1', 'copy:3']


def test_suppressed_leg_without_targets_is_named_as_such(segs):
    segs['1']['traffic_redirect'] = []
    assert link_groups(segs)[1].title == 'LEG_7_3_b suppressed, +1 leg(s) with it'


def test_view_entries_all_selected_leg_then_numbered_groups(segs):
    entries = view_entries(segs)
    assert [key for _t, key in entries] == [VIEW_ALL, VIEW_LEG, 'copy:1', 'suppress:1']
    assert entries[0][0] == 'All links (3)'
    assert entries[2][0] == '1/2  LEG_7_3_b copied to LEG_7_2'
    assert view_entries({})[2:] == []


def test_step_wraps_through_the_groups_only():
    keys = [VIEW_ALL, VIEW_LEG, 'a', 'b', 'c']
    assert step_view(keys, VIEW_ALL, 1) == 'a'
    assert step_view(keys, VIEW_LEG, -1) == 'c'
    assert step_view(keys, 'b', 1) == 'c'
    assert step_view(keys, 'c', 1) == 'a'
    assert step_view(keys, 'a', -1) == 'c'
    assert step_view([VIEW_ALL, VIEW_LEG], VIEW_ALL, 1) == VIEW_ALL


def test_resolve_view_keeps_the_user_one_at_a_time():
    old = [VIEW_ALL, VIEW_LEG, 'a', 'b', 'c']
    assert resolve_view(old, old, 'b') == 'b'
    assert resolve_view(old, [VIEW_ALL, VIEW_LEG, 'a', 'c'], 'b') == 'c'    # the one now in its place
    assert resolve_view(old, [VIEW_ALL, VIEW_LEG, 'a'], 'c') == 'a'         # clamped
    assert resolve_view(old, [VIEW_ALL, VIEW_LEG], 'b') == VIEW_ALL         # nothing left
    assert resolve_view([], [VIEW_ALL, VIEW_LEG], VIEW_LEG) == VIEW_LEG
    assert resolve_view([], [VIEW_ALL, VIEW_LEG, 'x'], 'stale') == VIEW_ALL


def test_links_in_view(segs):
    assert len(links_in_view(segs, VIEW_ALL)) == 3
    assert [(lk.kind, lk.src, lk.dst) for lk in links_in_view(segs, VIEW_LEG, '3')] == [('redirect', '1', '3')]
    assert len(links_in_view(segs, VIEW_LEG, '1')) == 3
    assert links_in_view(segs, VIEW_LEG, None) == []
    assert links_in_view(segs, VIEW_LEG, '5') == []
    assert sorted(lk.kind for lk in links_in_view(segs, 'suppress:1')) == ['redirect', 'with']
    assert links_in_view(segs, 'copy:gone') == []


def test_link_geometries_of_a_subset(segs):
    subset = links_in_view(segs, 'copy:1')
    assert [lk for lk, _pts in link_geometries(segs, subset)] == subset
    assert link_geometries(segs, []) == []
