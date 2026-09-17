"""Pure tests for ``geometries.waypoints`` -- the node registry legs
hang on (v0.15.2)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.route_validation import (  # noqa: E402
    CloseWaypointPair, apply_waypoint_merge, split_leg_at_points, _make_id_provider,
)
from geometries.waypoints import (  # noqa: E402
    WP_END, WP_START, ensure_waypoint, find_waypoint_at, find_waypoint_near, incident_legs,
    merge_waypoints, move_waypoint, new_waypoint_id, points_equal, prune_unused_waypoints,
    rebuild_waypoints, sync_endpoints_from_waypoints, validate_waypoint_refs,
    waypoints_from_serializable, waypoints_to_serializable,
)

A = (11.000000, 57.000000)
B = (11.100000, 57.050000)
C = (11.200000, 57.000000)
D = (11.100000, 56.900000)


def _leg(sid, sp, ep, **extra):
    d = {
        'Segment_Id': str(sid), 'Route_Id': 1, 'Leg_name': f'LEG_1_{sid}',
        'Start_Point': f'{sp[0]:.6f} {sp[1]:.6f}', 'End_Point': f'{ep[0]:.6f} {ep[1]:.6f}',
        'Width': 5000, 'line_length': 10000.0, 'Dirs': ['East going', 'West going'],
    }
    d.update(extra)
    return d


def _star():
    """Three legs meeting at B, one leg elsewhere."""
    return {
        '1': _leg(1, A, B),
        '2': _leg(2, B, C),
        '3': _leg(3, B, D),
        '4': _leg(4, C, D),
    }


# ---------------------------------------------------------------- rebuild

def test_rebuild_shares_one_node_per_coincident_endpoint():
    sd = _star()
    wps = rebuild_waypoints(sd)
    assert len(wps) == 4  # A, B, C, D
    b_id = sd['1'][WP_END]
    assert sd['2'][WP_START] == b_id and sd['3'][WP_START] == b_id
    assert sd['2'][WP_END] == sd['4'][WP_START]  # C
    assert sd['3'][WP_END] == sd['4'][WP_END]    # D
    assert points_equal(wps[b_id], B)
    assert validate_waypoint_refs(sd, wps) == []


def test_rebuild_keeps_ids_of_unmoved_nodes_and_drops_orphans():
    sd = _star()
    wps = rebuild_waypoints(sd)
    b_id = sd['1'][WP_END]
    # Leg 4 removed -> C and D still used by legs 2 / 3; add a stray node.
    del sd['4']
    wps['99'] = (12.0, 58.0)
    wps2 = rebuild_waypoints(sd, wps)
    assert sd['1'][WP_END] == b_id
    assert '99' not in wps2
    assert set(wps2) == {sd['1'][WP_START], b_id, sd['2'][WP_END], sd['3'][WP_END]}


def test_rebuild_prefers_the_legs_own_ref_when_coordinates_match():
    sd = {'1': _leg(1, A, B, start_wp='7', end_wp='8')}
    wps = rebuild_waypoints(sd, {'7': A, '8': B})
    assert sd['1'][WP_START] == '7' and sd['1'][WP_END] == '8'
    assert set(wps) == {'7', '8'}


def test_rebuild_ignores_stale_ref_whose_node_moved_elsewhere():
    # Leg says start_wp=7 but node 7 is somewhere else: coordinates win.
    sd = {'1': _leg(1, A, B, start_wp='7', end_wp='8')}
    wps = rebuild_waypoints(sd, {'7': C, '8': B})
    assert sd['1'][WP_START] != '7'
    assert points_equal(wps[sd['1'][WP_START]], A)
    assert '7' not in wps


def test_rebuild_tolerates_junk_and_unparseable_legs():
    sd = {'1': _leg(1, A, B), 'x': 'junk', '2': {'Start_Point': 'bad', 'End_Point': None}}
    wps = rebuild_waypoints(sd)
    assert len(wps) == 2
    assert WP_START not in sd['2']


# ---------------------------------------------------------------- registry-wins ops

def test_move_waypoint_moves_every_incident_leg_endpoint():
    sd = _star()
    wps = rebuild_waypoints(sd)
    b_id = sd['1'][WP_END]
    new_b = (11.123456, 57.054321)
    moved = move_waypoint(wps, sd, b_id, new_b)
    assert sorted(moved) == ['1', '2', '3']
    assert sd['1']['End_Point'] == '11.123456 57.054321'
    assert sd['2']['Start_Point'] == '11.123456 57.054321'
    assert sd['3']['Start_Point'] == '11.123456 57.054321'
    assert sd['4']['Start_Point'] == f'{C[0]:.6f} {C[1]:.6f}'  # untouched
    assert sd['1']['line_length'] != 10000.0  # refreshed
    assert validate_waypoint_refs(sd, wps) == []


def test_move_unknown_waypoint_is_noop():
    sd = _star()
    wps = rebuild_waypoints(sd)
    assert move_waypoint(wps, sd, '404', (0.0, 0.0)) == []


def test_merge_waypoints_repoints_legs_and_drops_node():
    sd = _star()
    wps = rebuild_waypoints(sd)
    b_id, c_id = sd['1'][WP_END], sd['2'][WP_END]
    moved = merge_waypoints(wps, sd, keep_id=c_id, drop_id=b_id)
    assert sorted(moved) == ['1', '2', '3']
    assert b_id not in wps
    assert sd['1'][WP_END] == c_id and sd['3'][WP_START] == c_id
    assert sd['1']['End_Point'] == f'{C[0]:.6f} {C[1]:.6f}'
    # Leg 2 collapsed onto C at both ends -- still consistent.
    assert validate_waypoint_refs(sd, wps) == []


def test_merge_same_or_unknown_is_noop():
    sd = _star()
    wps = rebuild_waypoints(sd)
    b_id = sd['1'][WP_END]
    assert merge_waypoints(wps, sd, b_id, b_id) == []
    assert merge_waypoints(wps, sd, '404', b_id) == []


def test_sync_writes_endpoints_from_nodes_only_where_known():
    sd = _star()
    wps = rebuild_waypoints(sd)
    b_id = sd['1'][WP_END]
    wps[b_id] = (11.5, 57.5)          # registry edited behind the legs' back
    sd['4'][WP_START] = 'nope'        # unknown node: left alone
    changed = sync_endpoints_from_waypoints(sd, wps)
    assert changed == 3
    assert sd['2']['Start_Point'] == '11.500000 57.500000'
    assert sd['4']['Start_Point'] == f'{C[0]:.6f} {C[1]:.6f}'


def test_prune_and_ensure_and_ids():
    wps = {}
    a = ensure_waypoint(wps, A)
    assert ensure_waypoint(wps, A) == a           # exact reuse
    assert ensure_waypoint(wps, (A[0] + 5e-8, A[1])) == a  # within tolerance
    b = ensure_waypoint(wps, B)
    assert b != a and new_waypoint_id(wps) == '3'
    sd = {'1': _leg(1, A, A, start_wp=a, end_wp=a)}
    assert prune_unused_waypoints(wps, sd) == [b]
    assert set(wps) == {a}
    assert find_waypoint_at(wps, B) is None
    assert incident_legs(sd, a) == [('1', 'start'), ('1', 'end')]


def test_find_waypoint_near_uses_metres_and_exclusions():
    wps = {'1': A, '2': (A[0] + 0.0001, A[1])}  # ~6 m apart
    hit = find_waypoint_near(wps, A, tol_m=10.0, exclude=('1',))
    assert hit is not None and hit[0] == '2' and 4 < hit[1] < 8
    assert find_waypoint_near(wps, A, tol_m=1.0, exclude=('1',)) is None


# ---------------------------------------------------------------- serialisation

def test_serialisation_round_trip_and_junk():
    wps = {'1': A, '2': B}
    block = waypoints_to_serializable(wps)
    assert block == {'1': [A[0], A[1]], '2': [B[0], B[1]]}
    assert waypoints_from_serializable(block) == wps
    assert waypoints_from_serializable({'1': {'x': 1.0, 'y': 2.0}, '2': 'junk', '3': [1]}) == {'1': (1.0, 2.0)}
    assert waypoints_from_serializable(None) == {}


# ---------------------------------------------------------------- with route_validation

def test_split_then_rebuild_gives_interior_node_fresh_id_and_keeps_outer_ids():
    sd = {'1': _leg(1, A, C)}
    wps = rebuild_waypoints(sd)
    a_id, c_id = sd['1'][WP_START], sd['1'][WP_END]
    mid = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2)
    ids = split_leg_at_points(sd, '1', [mid], _make_id_provider(sd))
    first, second = ids
    # The pure split drops the inherited ids on the interior ends...
    assert WP_END not in sd[first] and WP_START not in sd[second]
    # ...and the rebuild assigns one shared fresh node there.
    wps2 = rebuild_waypoints(sd, wps)
    assert sd[first][WP_START] == a_id and sd[second][WP_END] == c_id
    m_id = sd[first][WP_END]
    assert sd[second][WP_START] == m_id and m_id not in (a_id, c_id)
    assert points_equal(wps2[m_id], mid)
    assert validate_waypoint_refs(sd, wps2) == []


def test_close_waypoint_merge_then_rebuild_collapses_to_one_node():
    near_b = (11.10002, 57.05)  # ~1.2 m east of B, exactly representable after 6-decimal formatting
    sd = {'1': _leg(1, A, B), '2': _leg(2, near_b, C)}
    wps = rebuild_waypoints(sd)
    assert len(wps) == 4
    pair = CloseWaypointPair(
        point_a=B, point_b=near_b, distance_m=1.2, threshold_m=500.0,
        leg_endpoints={B: [('1', 'end')], near_b: [('2', 'start')]},
    )
    assert apply_waypoint_merge(sd, pair, B) == 2
    wps2 = rebuild_waypoints(sd, wps)
    assert len(wps2) == 3
    assert sd['1'][WP_END] == sd['2'][WP_START]
    assert validate_waypoint_refs(sd, wps2) == []


def test_rebuild_is_idempotent():
    sd = _star()
    wps = rebuild_waypoints(sd)
    snapshot = copy.deepcopy(sd)
    wps2 = rebuild_waypoints(sd, wps)
    assert wps2 == wps and sd == snapshot


def test_storage_normaliser_adds_waypoints_block_and_schema_accepts_it():
    from unittest.mock import MagicMock
    from omrat_utils.storage import Storage
    from omrat_utils.validate_data import RootModelSchema
    sd = _star()
    data = {
        'pc': {'p_pc': 0.00016, 'd_pc': 1},
        'drift': {'drift_p': 1, 'anchor_p': 0.5, 'anchor_d': 1, 'speed': 2.0,
                  'rose': {'0': 12.5}, 'repair': {'func': 'lognorm', 'std': 1, 'loc': 0,
                                                  'scale': 1, 'use_lognormal': True}},
        'segment_data': sd,
        'traffic_data': {k: {'East going': _block(), 'West going': _block()} for k in sd},
        'depths': [], 'objects': [],
    }
    out = Storage(MagicMock())._normalize_legacy_to_schema(data)
    RootModelSchema.model_validate(out)
    assert len(out['waypoints']) == 4
    b_id = out['segment_data']['1'][WP_END]
    assert out['waypoints'][b_id] == [B[0], B[1]]
    # Same file again: the block is stable.
    again = Storage(MagicMock())._normalize_legacy_to_schema(copy.deepcopy(out))
    assert again['waypoints'] == out['waypoints']


def _block():
    names = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)',
             'Ship heights (meters)', 'Ship Beam (meters)', 'Scaling (%)']
    return {n: [[0.0]] for n in names}
