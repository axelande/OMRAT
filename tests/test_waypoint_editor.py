"""Pure tests for the waypoint editor operations: detach / attach a leg
end (``geometries.waypoints``), the crossing check
(``geometries.route_validation.crossings_of_leg``) and keeping
hand-edited junction matrices (``geometries.junctions.carry_user_matrices``)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.junctions import (  # noqa: E402
    build_junctions, carry_user_matrices, junction_id_for_point, refresh_junction_registry,
    apply_geometric_defaults,
)
from geometries.route_validation import (  # noqa: E402
    CLOSE_WAYPOINT_TOL_FRAC, crossings_of_leg, find_close_waypoint_pairs, haversine_m, parse_wkt_point,
)
from geometries.waypoints import (  # noqa: E402
    WP_START, WaypointEditError, attach_candidates, attach_leg_end, detach_leg_end,
    detach_limits, detach_point, incident_legs, points_equal, rebuild_waypoints, validate_waypoint_refs,
)

A = (11.000000, 57.000000)
B = (11.100000, 57.050000)
C = (11.200000, 57.000000)
D = (11.100000, 56.900000)
E = (11.300000, 57.100000)


def _leg(sid, sp, ep):
    return {
        'Segment_Id': str(sid), 'Route_Id': 1, 'Leg_name': f'LEG_1_{sid}',
        'Start_Point': f'{sp[0]:.6f} {sp[1]:.6f}', 'End_Point': f'{ep[0]:.6f} {ep[1]:.6f}',
        'Width': 5000, 'line_length': haversine_m(sp, ep), 'Dirs': ['East going', 'West going'],
    }


def _star():
    """Legs 1, 2, 3 meet at B; leg 4 joins C and D; leg 5 runs C -> E."""
    sd = {
        '1': _leg(1, A, B),
        '2': _leg(2, B, C),
        '3': _leg(3, B, D),
        '4': _leg(4, C, D),
        '5': _leg(5, C, E),
    }
    return sd, rebuild_waypoints(sd)


def _node_at(wps, xy):
    return next(k for k, v in wps.items() if points_equal(v, xy))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# ---------------------------------------------------------------- detach

def test_detach_moves_only_that_end_back_along_its_own_leg():
    sd, wps = _star()
    b = _node_at(wps, B)
    min_m, suggested, max_m = detach_limits(wps, sd, b, '2', 'start')
    assert min_m < suggested <= max_m
    new_id = detach_leg_end(wps, sd, b, '2', 'start', suggested)

    assert new_id not in (b,) and new_id in wps
    assert sd['2'][WP_START] == new_id
    new_xy = parse_wkt_point(sd['2']['Start_Point'])
    assert points_equal(new_xy, wps[new_id])
    # On the old leg B -> C, between the two ends, at the asked distance.
    assert abs(_cross(B, C, new_xy)) < 1e-9
    assert haversine_m(B, new_xy) == pytest.approx(suggested, rel=0.01, abs=1.0)
    assert haversine_m(new_xy, C) < haversine_m(B, C)
    assert sd['2']['line_length'] == pytest.approx(haversine_m(new_xy, C))
    # The others stay on B; B is still a node.
    assert {lid for lid, _s in incident_legs(sd, b)} == {'1', '3'}
    assert points_equal(wps[b], B)
    assert validate_waypoint_refs(sd, wps) == []


def test_detached_end_crosses_nothing_and_is_not_offered_for_merging_again():
    sd, wps = _star()
    b = _node_at(wps, B)
    before = {lid: crossings_of_leg(sd, lid) for lid in sd}
    _min_m, suggested, _max_m = detach_limits(wps, sd, b, '3', 'start')
    detach_leg_end(wps, sd, b, '3', 'start', suggested)
    assert {lid: crossings_of_leg(sd, lid) for lid in sd} == before
    assert find_close_waypoint_pairs(sd) == []


def test_minimum_gap_is_the_close_waypoint_threshold():
    sd, wps = _star()
    b = _node_at(wps, B)
    min_m, _s, max_m = detach_limits(wps, sd, b, '2', 'start')
    shortest = min(sd[k]['line_length'] for k in ('1', '2', '3'))
    assert min_m == pytest.approx(CLOSE_WAYPOINT_TOL_FRAC * shortest)
    assert max_m == pytest.approx(0.5 * sd['2']['line_length'])
    # Closer than the threshold would be merged again: refused.
    probe = copy.deepcopy(sd)
    xy = detach_point(probe, '2', 'start', 0.5 * min_m)
    probe['2']['Start_Point'] = f'{xy[0]:.6f} {xy[1]:.6f}'
    assert find_close_waypoint_pairs(probe) != []
    with pytest.raises(WaypointEditError):
        detach_leg_end(wps, sd, b, '2', 'start', 0.5 * min_m)
    with pytest.raises(WaypointEditError):
        detach_leg_end(wps, sd, b, '2', 'start', 1.01 * max_m)


def test_detach_refuses_a_lone_end_and_a_wrong_waypoint():
    sd, wps = _star()
    a = _node_at(wps, A)
    b = _node_at(wps, B)
    with pytest.raises(WaypointEditError, match='only leg'):
        detach_leg_end(wps, sd, a, '1', 'start', 100.0)
    with pytest.raises(WaypointEditError, match='not on waypoint'):
        detach_leg_end(wps, sd, b, '1', 'start', 100.0)
    with pytest.raises(WaypointEditError, match='does not exist'):
        detach_leg_end(wps, sd, b, '99', 'start', 100.0)


def test_junction_loses_the_detached_leg():
    sd, wps = _star()
    b = _node_at(wps, B)
    jid = junction_id_for_point(B)
    assert set(build_junctions(sd)[jid].legs) == {'1', '2', '3'}
    detach_leg_end(wps, sd, b, '2', 'start', detach_limits(wps, sd, b, '2', 'start')[1])
    assert set(build_junctions(sd)[jid].legs) == {'1', '3'}


# ---------------------------------------------------------------- attach

def test_attach_candidates_skip_legs_on_the_node_and_sort_by_distance():
    sd, wps = _star()
    b = _node_at(wps, B)
    cands = attach_candidates(wps, sd, b)
    assert {lid for lid, _s, _d in cands} == {'4', '5'}
    dists = [d for _l, _s, d in cands]
    assert dists == sorted(dists)
    assert cands[0][2] == pytest.approx(min(haversine_m(p, B) for p in (C, D, E)))


def test_attach_moves_the_end_onto_the_node_and_leaves_the_old_neighbours():
    sd, wps = _star()
    b = _node_at(wps, B)
    c = _node_at(wps, C)
    old = attach_leg_end(wps, sd, '5', 'start', b)
    assert old == c
    assert sd['5'][WP_START] == b
    assert points_equal(parse_wkt_point(sd['5']['Start_Point']), B)
    assert sd['5']['line_length'] == pytest.approx(haversine_m(B, E))
    # C still carries legs 2 and 4, so the node stays and they did not move.
    assert c in wps
    assert points_equal(parse_wkt_point(sd['2']['End_Point']), C)
    assert {lid for lid, _s in incident_legs(sd, b)} == {'1', '2', '3', '5'}
    assert validate_waypoint_refs(sd, wps) == []


def test_attach_drops_an_old_node_nothing_else_uses():
    sd, wps = _star()
    a = _node_at(wps, A)
    d = _node_at(wps, D)
    old = attach_leg_end(wps, sd, '1', 'start', d)
    assert old == a and a not in wps
    assert validate_waypoint_refs(sd, wps) == []


def test_attach_refuses_zero_length_and_no_op():
    sd, wps = _star()
    b = _node_at(wps, B)
    c = _node_at(wps, C)
    with pytest.raises(WaypointEditError, match='vanish'):
        attach_leg_end(wps, sd, '2', 'start', c)    # leg 2 ends at C
    with pytest.raises(WaypointEditError, match='already'):
        attach_leg_end(wps, sd, '2', 'start', b)
    with pytest.raises(WaypointEditError, match='does not exist'):
        attach_leg_end(wps, sd, '2', 'start', '999')


def test_detach_then_attach_restores_the_junction():
    sd, wps = _star()
    b = _node_at(wps, B)
    new_id = detach_leg_end(wps, sd, b, '2', 'start', detach_limits(wps, sd, b, '2', 'start')[1])
    attach_leg_end(wps, sd, '2', 'start', b)
    assert new_id not in wps
    assert points_equal(parse_wkt_point(sd['2']['Start_Point']), B)
    assert set(build_junctions(sd)[junction_id_for_point(B)].legs) == {'1', '2', '3'}


# ---------------------------------------------------------------- crossings

def test_crossings_of_leg_ignores_shared_endpoints():
    sd = {
        '1': _leg(1, (11.0, 57.0), (11.2, 57.2)),
        '2': _leg(2, (11.0, 57.2), (11.2, 57.0)),   # X with leg 1
        '3': _leg(3, (11.2, 57.2), (11.4, 57.2)),   # shares an end with leg 1
    }
    assert crossings_of_leg(sd, '1') == ['2']
    assert crossings_of_leg(sd, '2') == ['1']
    assert crossings_of_leg(sd, '3') == []
    assert crossings_of_leg(sd, '99') == []


# ---------------------------------------------------------------- matrices

def _registry(sd):
    reg = build_junctions(sd)
    apply_geometric_defaults(reg, sd)
    return reg


def test_user_matrix_is_trimmed_when_a_leg_leaves():
    sd, wps = _star()
    b = _node_at(wps, B)
    jid = junction_id_for_point(B)
    reg = _registry(sd)
    reg[jid].transitions = {
        '1': {'2': 0.6, '3': 0.4},
        '2': {'1': 0.5, '3': 0.5},
        '3': {'1': 0.2, '2': 0.8},
    }
    reg[jid].source = 'user'
    before = copy.deepcopy(reg)
    detach_leg_end(wps, sd, b, '2', 'start', detach_limits(wps, sd, b, '2', 'start')[1])
    after = refresh_junction_registry(reg, sd)
    assert after[jid].source == 'geometry'          # what the plain refresh does
    out = carry_user_matrices(before, after, sd)
    assert out == {'trimmed': [jid], 'extended': []}
    assert after[jid].source == 'user'
    assert after[jid].transitions == {'1': {'3': 1.0}, '3': {'1': 1.0}}


def test_user_matrix_gets_a_joining_leg_at_zero_percent():
    sd, wps = _star()
    b = _node_at(wps, B)
    jid = junction_id_for_point(B)
    reg = _registry(sd)
    user = {
        '1': {'2': 0.6, '3': 0.4},
        '2': {'1': 0.5, '3': 0.5},
        '3': {'1': 0.2, '2': 0.8},
    }
    reg[jid].transitions = copy.deepcopy(user)
    reg[jid].source = 'user'
    before = copy.deepcopy(reg)
    attach_leg_end(wps, sd, '5', 'start', b)
    after = refresh_junction_registry(reg, sd)
    out = carry_user_matrices(before, after, sd)
    assert out == {'trimmed': [], 'extended': [jid]}
    j = after[jid]
    assert j.source == 'user'
    for in_leg, row in user.items():
        for out_leg, share in row.items():
            assert j.transitions[in_leg][out_leg] == pytest.approx(share)
        assert j.transitions[in_leg]['5'] == 0.0
    assert sum(j.transitions['5'].values()) == pytest.approx(1.0)
    assert '5' not in j.transitions['5']


def test_non_user_junctions_are_left_to_the_refresh():
    sd, wps = _star()
    b = _node_at(wps, B)
    jid = junction_id_for_point(B)
    reg = _registry(sd)
    reg[jid].source = 'ais'
    before = copy.deepcopy(reg)
    detach_leg_end(wps, sd, b, '2', 'start', detach_limits(wps, sd, b, '2', 'start')[1])
    after = refresh_junction_registry(reg, sd)
    assert carry_user_matrices(before, after, sd) == {'trimmed': [], 'extended': []}
    assert after[jid].source == 'geometry'   # leg map changed: AIS must be counted again
