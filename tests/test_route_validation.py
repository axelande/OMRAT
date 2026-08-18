"""Standalone tests for the pure-geometry route-validation primitives.

Run with:

    /mnt/c/OSGeo4W/apps/Python312/python.exe \
        -m pytest -p no:qgis --noconftest tests/test_route_validation.py
"""

from __future__ import annotations

import math

import pytest

from geometries.route_validation import (
    CloseWaypointPair,
    LegIntersection,
    apply_intersection_split,
    apply_waypoint_merge,
    find_close_waypoint_pairs,
    find_leg_intersections,
    format_wkt_point,
    haversine_m,
    parse_wkt_point,
    validate_routes,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _seg(start: str, end: str, length_m: float, **extra) -> dict:
    return {
        'Start_Point': start,
        'End_Point': end,
        'line_length': length_m,
        'Width': 5000,
        'Route_Id': 1,
        'Leg_name': extra.pop('name', 'LEG'),
        **extra,
    }


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def test_parse_wkt_point_accepts_three_formats():
    assert parse_wkt_point("14.5 55.3") == (14.5, 55.3)
    assert parse_wkt_point("POINT(14.5 55.3)") == (14.5, 55.3)
    assert parse_wkt_point("Point (14.5 55.3)") == (14.5, 55.3)


def test_parse_wkt_point_rejects_garbage():
    assert parse_wkt_point(None) is None
    assert parse_wkt_point("") is None
    assert parse_wkt_point("not a point") is None
    assert parse_wkt_point("only-one-token") is None


def test_format_wkt_point_uses_six_decimals():
    assert format_wkt_point(14.5, 55.3) == "14.500000 55.300000"


def test_haversine_matches_known_distance():
    # Stockholm -> Gothenburg (~395 km)
    d = haversine_m((18.0686, 59.3293), (11.9746, 57.7089))
    assert 390_000 < d < 400_000


# ---------------------------------------------------------------------------
# Close-waypoint detection
# ---------------------------------------------------------------------------


def test_no_close_pairs_when_endpoints_already_match():
    """Identical endpoints share a junction, not a close-pair candidate."""
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0 55.0", "16.0 55.0", 100_000),
    }
    assert find_close_waypoint_pairs(sd) == []


def test_close_pair_detected_at_default_tolerance():
    # Two endpoints ~200 m apart; legs are 100 km long → 5% = 5 km, easy hit.
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.001 55.0001", "16.0 55.0", 100_000),  # ~200 m offset
    }
    pairs = find_close_waypoint_pairs(sd)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.distance_m < 250
    # Threshold uses 5% of shortest incident leg.
    assert math.isclose(p.threshold_m, 0.05 * 100_000, rel_tol=1e-6)


def test_close_pair_skipped_when_distance_above_threshold():
    # Legs only 1 km long → 5% = 50 m; offset of ~200 m must be skipped.
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 1_000),
        '2': _seg("15.001 55.0001", "16.0 55.0", 1_000),
    }
    assert find_close_waypoint_pairs(sd) == []


def test_threshold_uses_shorter_incident_leg():
    """A long leg adjacent to a short one should not relax the snap radius."""
    sd = {
        # Two legs sharing an "almost-close" pair where one leg is short.
        'long': _seg("14.0 55.0", "20.0 55.0", 400_000),
        'short': _seg("20.001 55.0", "20.05 55.0", 3_000),
    }
    pairs = find_close_waypoint_pairs(sd)
    # Threshold = 5% × min(400 km, 3 km) = 150 m.  The two close points
    # are ~64 m apart, so they DO qualify.
    assert len(pairs) == 1
    assert pairs[0].threshold_m == pytest.approx(150.0, rel=0.01)


def test_multiple_pairs_sorted_by_distance():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0001 55.0", "16.0 55.0", 100_000),  # ~6 m
        '3': _seg("15.001 55.0", "17.0 55.0", 100_000),  # ~64 m
    }
    pairs = find_close_waypoint_pairs(sd)
    assert len(pairs) >= 2
    distances = [p.distance_m for p in pairs]
    assert distances == sorted(distances)


def test_close_pair_records_endpoint_references():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0001 55.0", "16.0 55.0", 100_000),
    }
    pair = find_close_waypoint_pairs(sd)[0]
    refs_a = pair.leg_endpoints[pair.point_a]
    refs_b = pair.leg_endpoints[pair.point_b]
    # Each location should map back to exactly the leg+side that owns it.
    assert ("1", "end") in refs_a or ("1", "end") in refs_b
    assert ("2", "start") in refs_a or ("2", "start") in refs_b


# ---------------------------------------------------------------------------
# Apply waypoint merge
# ---------------------------------------------------------------------------


def test_apply_merge_rewrites_endpoints_to_target():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0001 55.0", "16.0 55.0", 100_000),
    }
    pair = find_close_waypoint_pairs(sd)[0]
    target = pair.point_a  # snap to leg 1's existing endpoint
    moved = apply_waypoint_merge(sd, pair, target)
    assert moved == 2  # leg 1 end + leg 2 start
    assert parse_wkt_point(sd['1']['End_Point']) == target
    assert parse_wkt_point(sd['2']['Start_Point']) == target
    # And after the merge there are no close pairs left.
    assert find_close_waypoint_pairs(sd) == []


def test_apply_merge_to_midpoint_updates_both():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0002 55.0", "16.0 55.0", 100_000),
    }
    pair = find_close_waypoint_pairs(sd)[0]
    mid = pair.midpoint
    moved = apply_waypoint_merge(sd, pair, mid)
    assert moved == 2
    assert parse_wkt_point(sd['1']['End_Point']) == pytest.approx(mid)
    assert parse_wkt_point(sd['2']['Start_Point']) == pytest.approx(mid)


def test_apply_merge_recomputes_line_length_via_haversine():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0001 55.0", "16.0 55.0", 100_000),
    }
    pair = find_close_waypoint_pairs(sd)[0]
    apply_waypoint_merge(sd, pair, pair.point_a)
    # Both legs spanned ~1 degree of longitude at lat 55 ≈ 64 km.
    assert 60_000 < sd['1']['line_length'] < 70_000
    assert 60_000 < sd['2']['line_length'] < 70_000


def test_apply_merge_handles_three_legs_at_one_junction():
    """All three legs that touched the snapped location must move together."""
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0001 55.0", "16.0 55.0", 100_000),
        '3': _seg("17.0 55.0", "15.0 55.0", 100_000),  # already exact match for leg 1's end
    }
    pair = find_close_waypoint_pairs(sd)[0]
    moved = apply_waypoint_merge(sd, pair, pair.point_a)
    assert moved >= 2
    # Leg 3 already shared the location with leg 1, so it should remain
    # tied to it after the snap.
    assert parse_wkt_point(sd['3']['End_Point']) == pair.point_a


# ---------------------------------------------------------------------------
# Leg-intersection detection
# ---------------------------------------------------------------------------


def test_no_intersection_for_disjoint_legs():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("14.0 56.0", "15.0 56.0", 100_000),
    }
    assert find_leg_intersections(sd) == []


def test_no_intersection_when_only_endpoints_meet():
    """Shared endpoints are junctions, not crossings."""
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0 55.0", "16.0 56.0", 100_000),
    }
    assert find_leg_intersections(sd) == []


def test_x_crossing_detected():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    hits = find_leg_intersections(sd)
    assert len(hits) == 1
    h = hits[0]
    assert (h.leg1_id, h.leg2_id) == ('1', '2')
    # Mid-X point ~= (15, 55.5)
    assert h.point[0] == pytest.approx(15.0, abs=0.01)
    assert h.point[1] == pytest.approx(55.5, abs=0.01)
    assert 0 < h.t1 < 1 and 0 < h.t2 < 1


def test_parallel_legs_not_flagged():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("14.0 55.1", "15.0 55.1", 100_000),
    }
    assert find_leg_intersections(sd) == []


def test_collinear_overlapping_legs_not_flagged_as_x_crossing():
    """Collinear overlap is a different kind of bug — out of scope here."""
    sd = {
        '1': _seg("14.0 55.0", "16.0 55.0", 200_000),
        '2': _seg("15.0 55.0", "17.0 55.0", 200_000),
    }
    # Strict X-test: parallel/collinear → no result.
    assert find_leg_intersections(sd) == []


# ---------------------------------------------------------------------------
# Apply intersection split
# ---------------------------------------------------------------------------


def test_split_creates_four_sub_legs():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    hit = find_leg_intersections(sd)[0]
    out = apply_intersection_split(sd, hit)
    # First sub-leg keeps original id; new ids are integers > 2.
    assert out['1'][0] == '1'
    assert out['2'][0] == '2'
    new_ids = {out['1'][1], out['2'][1]}
    assert all(int(nid) > 2 for nid in new_ids)
    # Total of 4 legs.
    assert len(sd) == 4


def test_split_ends_meet_at_intersection_point():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    hit = find_leg_intersections(sd)[0]
    apply_intersection_split(sd, hit)
    # Leg 1's first half ends where leg 2's first half ends.
    leg1_end = parse_wkt_point(sd['1']['End_Point'])
    assert leg1_end is not None
    # All four sub-legs share the intersection node.
    nodes = []
    for leg_id, seg in sd.items():
        nodes.append(parse_wkt_point(seg['Start_Point']))
        nodes.append(parse_wkt_point(seg['End_Point']))
    # The intersection point should appear at least four times.
    matches = sum(1 for n in nodes if n == leg1_end)
    assert matches == 4


def test_split_inherits_traffic_into_both_subs():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    td = {
        '1': {'East going': {'Frequency (ships/year)': [[10.0]]}},
        '2': {'East going': {'Frequency (ships/year)': [[20.0]]}},
    }
    hit = find_leg_intersections(sd)[0]
    out = apply_intersection_split(sd, hit, traffic_data=td)
    # Both sub-legs of leg 1 see the same frequency.
    assert td['1']['East going']['Frequency (ships/year)'] == [[10.0]]
    new_id_1 = out['1'][1]
    assert td[new_id_1]['East going']['Frequency (ships/year)'] == [[10.0]]


def test_split_leg_lengths_sum_to_parent():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    hit = find_leg_intersections(sd)[0]
    out = apply_intersection_split(sd, hit)
    # Within rounding, sub-legs of leg 1 should sum to original length.
    parent_len = haversine_m((14.0, 55.0), (16.0, 56.0))
    sub_a = sd['1']['line_length']
    sub_b = sd[out['1'][1]]['line_length']
    assert (sub_a + sub_b) == pytest.approx(parent_len, rel=0.001)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


def test_validate_routes_bundles_both_detectors():
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
        '3': _seg("16.0 56.0", "16.0001 56.0", 100_000),  # close pair to leg 1's end
        '4': _seg("16.001 56.0", "17.0 56.0", 100_000),
    }
    rep = validate_routes(sd)
    assert not rep.empty
    assert len(rep.intersections) == 1
    assert len(rep.close_pairs) >= 1


def test_validate_routes_empty_for_clean_project():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0", 100_000),
        '2': _seg("15.0 55.0", "16.0 55.0", 100_000),
    }
    assert validate_routes(sd).empty


# ---------------------------------------------------------------------------
# split_leg_at_points — multi-crossing fix
# ---------------------------------------------------------------------------

from geometries.route_validation import split_leg_at_points, _make_id_provider


def _simple_sd(start: str, end: str, name: str = 'LEG_1_1') -> dict:
    return {
        '1': {
            'Start_Point': start, 'End_Point': end,
            'line_length': 100_000, 'Width': 5000,
            'Route_Id': 1, 'Segment_Id': '1', 'Leg_name': name,
        }
    }


class TestSplitLegAtPoints:
    def test_no_points_is_noop(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        result = split_leg_at_points(sd, '1', [], _make_id_provider(sd))
        assert result == ['1']
        assert len(sd) == 1

    def test_one_point_produces_two_sub_legs(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0", 'LEG_A')
        result = split_leg_at_points(sd, '1', [(15.0, 55.0)], _make_id_provider(sd))
        assert len(result) == 2
        assert result[0] == '1'               # original id kept for first sub
        assert len(sd) == 2
        assert sd['1']['Leg_name'] == 'LEG_A_a'
        assert sd[result[1]]['Leg_name'] == 'LEG_A_b'

    def test_two_points_produce_three_sub_legs_a_b_c(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0", 'LEG_5_13')
        result = split_leg_at_points(
            sd, '1', [(14.67, 55.0), (15.33, 55.0)], _make_id_provider(sd)
        )
        assert len(result) == 3
        assert result[0] == '1'
        assert sd['1']['Leg_name'] == 'LEG_5_13_a'
        assert sd[result[1]]['Leg_name'] == 'LEG_5_13_b'
        assert sd[result[2]]['Leg_name'] == 'LEG_5_13_c'

    def test_three_points_produce_four_sub_legs(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0", 'LEG_X')
        result = split_leg_at_points(
            sd, '1',
            [(14.5, 55.0), (15.0, 55.0), (15.5, 55.0)],
            _make_id_provider(sd),
        )
        assert len(result) == 4
        for i, letter in enumerate('abcd'):
            assert sd[result[i]]['Leg_name'] == f'LEG_X_{letter}'

    def test_sub_leg_endpoints_are_contiguous(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        result = split_leg_at_points(
            sd, '1', [(14.67, 55.0), (15.33, 55.0)], _make_id_provider(sd)
        )
        # End of each sub-leg should equal start of the next.
        for i in range(len(result) - 1):
            end_pt = parse_wkt_point(sd[result[i]]['End_Point'])
            start_pt = parse_wkt_point(sd[result[i + 1]]['Start_Point'])
            assert end_pt == start_pt

    def test_first_sub_leg_starts_at_original_start(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        result = split_leg_at_points(
            sd, '1', [(15.0, 55.0)], _make_id_provider(sd)
        )
        assert parse_wkt_point(sd[result[0]]['Start_Point']) == (14.0, 55.0)

    def test_last_sub_leg_ends_at_original_end(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        result = split_leg_at_points(
            sd, '1', [(15.0, 55.0)], _make_id_provider(sd)
        )
        assert parse_wkt_point(sd[result[-1]]['End_Point']) == (16.0, 55.0)

    def test_traffic_copied_to_all_sub_legs(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        td = {'1': {'N': [100.0], 'S': [80.0]}}
        result = split_leg_at_points(
            sd, '1', [(14.67, 55.0), (15.33, 55.0)], _make_id_provider(sd), td
        )
        for sub_id in result:
            assert sub_id in td
            assert td[sub_id]['N'] == [100.0]

    def test_original_leg_id_missing_is_noop(self):
        sd = _simple_sd("14.0 55.0", "16.0 55.0")
        result = split_leg_at_points(sd, '99', [(15.0, 55.0)], _make_id_provider(sd))
        assert result == ['99']

    def test_no_cascading_names_for_multiple_crossings(self):
        """The old per-split loop produced LEG_5_13_a_a_a etc. This must not happen."""
        sd = _simple_sd("14.0 55.0", "16.0 55.0", 'LEG_5_13')
        result = split_leg_at_points(
            sd, '1',
            [(14.5, 55.0), (15.0, 55.0), (15.5, 55.0)],
            _make_id_provider(sd),
        )
        for sub_id in result:
            name = sd[sub_id]['Leg_name']
            assert '_a_' not in name and name.count('_') <= name.count('LEG') + 3, (
                f"Cascading name detected: {name}"
            )

    def test_resplit_sub_leg_no_cascading_and_no_collision(self):
        """Re-splitting a sub-leg must not produce cascading grandchild names
        AND must not reuse a suffix letter already taken by a sibling."""
        # First split: LEG_1_1 → LEG_1_1_a (id='1'), LEG_1_1_b (id='2')
        sd = _simple_sd("14.0 55.0", "16.0 55.0", 'LEG_1_1')
        id_prov = _make_id_provider(sd)
        first = split_leg_at_points(sd, '1', [(15.0, 55.0)], id_prov)
        assert len(first) == 2
        sub_b_id = first[1]
        assert sd[sub_b_id]['Leg_name'] == 'LEG_1_1_b'

        # Second split: re-split LEG_1_1_b at another point.
        # LEG_1_1_a is already taken by a sibling, so sub-legs must be
        # LEG_1_1_b and LEG_1_1_c (not cascading _b_a/_b_b, and not
        # a duplicate _a which is already in use).
        second = split_leg_at_points(sd, sub_b_id, [(15.5, 55.0)], id_prov)
        assert len(second) == 2
        names = [sd[sid]['Leg_name'] for sid in second]
        # No cascading (no double-underscore letter sequences)
        for n in names:
            assert '_b_' not in n and '_a_' not in n, f"Cascading name: {n}"
        # No collision: LEG_1_1_a is still held by the original first sub-leg
        assert 'LEG_1_1_a' not in names, f"Duplicate suffix: {names}"
        # Should be the next available pair: _b and _c
        assert set(names) == {'LEG_1_1_b', 'LEG_1_1_c'}, (
            f"Expected LEG_1_1_b and LEG_1_1_c, got {names}"
        )
