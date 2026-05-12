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
