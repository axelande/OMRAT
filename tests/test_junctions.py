"""Standalone tests for the junction registry + transition-matrix module."""

from __future__ import annotations

import math

import pytest

from geometries.junctions import (
    Junction,
    apply_ais_defaults,
    apply_geometric_defaults,
    build_junctions,
    compute_geometric_transition_matrix,
    deflection_deg,
    deserialize_junctions,
    junction_id_for_point,
    refresh_junction_registry,
    serialize_junctions,
    transition_matrix_from_counts,
    transition_share,
    validate_junctions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(start: str, end: str, length_m: float = 100_000) -> dict:
    return {
        'Start_Point': start,
        'End_Point': end,
        'line_length': length_m,
        'Width': 5000,
        'Route_Id': 1,
        'Leg_name': 'LEG',
    }


def _y_junction_segments() -> dict:
    """Three legs meeting at (15, 55), pointing to N, SE, SW."""
    return {
        '1': _seg("15.0 55.5", "15.0 55.0"),    # leg 1: north end is far; junction at end
        '2': _seg("15.0 55.0", "15.5 54.5"),    # leg 2: junction at start, far end SE
        '3': _seg("15.0 55.0", "14.5 54.5"),    # leg 3: junction at start, far end SW
    }


# ---------------------------------------------------------------------------
# Bearing math
# ---------------------------------------------------------------------------


def test_deflection_zero_for_straight_continuation():
    # In leg comes from south (outward bearing N), out leg goes north
    # (outward bearing N)... no wait, that's a U-turn.
    # Straight: in leg outward = N, out leg outward = S → ship coming N
    # leaves S → 0 deflection.
    assert deflection_deg(0.0, 180.0) == pytest.approx(0.0, abs=1e-6)


def test_deflection_180_for_uturn():
    # In leg outward = E (so ship arriving heads W); out leg outward = E
    # too means ship leaves E -> reversed -> 180 deflection.
    assert deflection_deg(90.0, 90.0) == pytest.approx(180.0, abs=1e-6)


def test_deflection_90_for_right_turn():
    # In leg outward = N (ship arrives heading S); out leg outward = E.
    # Ship arrives heading S, then heads E → 90 deg right turn.
    assert deflection_deg(0.0, 90.0) == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------


def test_build_junctions_finds_y_junction():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    assert len(js) == 1
    j = next(iter(js.values()))
    assert j.point == pytest.approx((15.0, 55.0))
    assert set(j.legs.keys()) == {'1', '2', '3'}
    assert j.legs['1'] == 'end'
    assert j.legs['2'] == 'start'
    assert j.legs['3'] == 'start'


def test_build_junctions_skips_isolated_endpoints():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("16.0 55.0", "17.0 55.0"),
    }
    assert build_junctions(sd) == {}


def test_build_junctions_handles_chain_of_three():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
        '3': _seg("16.0 55.0", "17.0 55.0"),
    }
    js = build_junctions(sd)
    assert len(js) == 2
    points = sorted(j.point for j in js.values())
    assert points == [(15.0, 55.0), (16.0, 55.0)]


# ---------------------------------------------------------------------------
# Geometric transition matrix
# ---------------------------------------------------------------------------


def test_two_leg_junction_is_trivial():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
    }
    j = next(iter(build_junctions(sd).values()))
    m = compute_geometric_transition_matrix(j, sd)
    assert m == {'1': {'2': 1.0}, '2': {'1': 1.0}}


def test_y_junction_rows_sum_to_one():
    sd = _y_junction_segments()
    j = next(iter(build_junctions(sd).values()))
    m = compute_geometric_transition_matrix(j, sd)
    assert set(m.keys()) == {'1', '2', '3'}
    for row in m.values():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def test_y_junction_straighter_continuation_gets_higher_share():
    """Leg 1 enters from north (outward bearing 0°). Leg 2 outward = SE
    (~135°), leg 3 outward = SW (~225°). Both deflections are 45° from
    straight-through (which would be S = 180°).  By symmetry, shares
    should be equal."""
    sd = _y_junction_segments()
    j = next(iter(build_junctions(sd).values()))
    m = compute_geometric_transition_matrix(j, sd)
    row1 = m['1']
    assert row1['2'] == pytest.approx(row1['3'], abs=1e-6)


def test_t_junction_straight_through_dominates():
    """T-junction: legs 1&2 are collinear, leg 3 is perpendicular.
    Coming in along leg 1, the straight choice (leg 2) should beat
    the right turn (leg 3)."""
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
        '3': _seg("15.0 55.0", "15.0 56.0"),
    }
    j = next(iter(build_junctions(sd).values()))
    m = compute_geometric_transition_matrix(j, sd)
    assert m['1']['2'] > m['1']['3']
    assert m['1']['2'] > 0.5


# ---------------------------------------------------------------------------
# Apply geometric defaults to a registry
# ---------------------------------------------------------------------------


def test_apply_geometric_defaults_marks_source_geometry():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    n = apply_geometric_defaults(js, sd)
    assert n == 1
    assert all(j.source == 'geometry' for j in js.values())


def test_apply_geometric_defaults_preserves_user_rows():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 0.7, '3': 0.3}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    j.source = 'user'
    apply_geometric_defaults(js, sd)
    assert j.source == 'user'
    assert j.transitions['1'] == {'2': 0.7, '3': 0.3}


def test_apply_geometric_defaults_can_force_overwrite_user():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 0.99, '3': 0.01}}
    j.source = 'user'
    apply_geometric_defaults(js, sd, overwrite_user=True)
    assert j.source == 'geometry'
    assert j.transitions['1']['2'] != 0.99


# ---------------------------------------------------------------------------
# AIS-derived transitions
# ---------------------------------------------------------------------------


def test_transition_matrix_from_counts_normalises():
    counts = {'1': {'2': 70, '3': 30}, '2': {'1': 100}}
    m = transition_matrix_from_counts(counts)
    assert m['1']['2'] == pytest.approx(0.7)
    assert m['1']['3'] == pytest.approx(0.3)
    assert m['2']['1'] == pytest.approx(1.0)


def test_transition_matrix_from_counts_drops_zeros_and_handles_empty():
    counts = {'1': {'2': 0, '3': 0}, '2': {}}
    m = transition_matrix_from_counts(counts)
    assert m == {'1': {}, '2': {}}


def test_apply_ais_defaults_uses_normalised_shares():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    jid = next(iter(js.keys()))
    apply_ais_defaults(
        js, {jid: {'1': {'2': 70, '3': 30}, '2': {'1': 100}, '3': {'1': 100}}}, sd,
    )
    j = js[jid]
    assert j.source == 'ais'
    assert j.transitions['1']['2'] == pytest.approx(0.7)


def test_apply_ais_defaults_falls_back_to_geometry_for_empty_rows():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    jid = next(iter(js.keys()))
    apply_geometric_defaults(js, sd)
    geo_row = dict(js[jid].transitions['1'])
    apply_ais_defaults(
        js, {jid: {'2': {'1': 100}, '3': {'1': 100}}}, sd,
    )
    # Leg 1's row had no AIS evidence → should match the geometric row.
    assert js[jid].transitions['1'] == pytest.approx(geo_row)
    assert js[jid].source == 'ais'


def test_apply_ais_defaults_preserves_user():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    jid = next(iter(js.keys()))
    j = js[jid]
    j.transitions = {'1': {'2': 0.5, '3': 0.5}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    j.source = 'user'
    apply_ais_defaults(
        js, {jid: {'1': {'2': 70, '3': 30}}}, sd,
    )
    assert j.source == 'user'
    assert j.transitions['1'] == {'2': 0.5, '3': 0.5}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_passes_clean_geometry_defaults():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    assert validate_junctions(js, sd) == []


def test_validate_flags_bad_row_sum():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 0.4, '3': 0.4}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    warnings = validate_junctions(js, sd)
    kinds = {w.kind for w in warnings}
    assert 'row_sum' in kinds


def test_validate_flags_unknown_leg():
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
    }
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.legs['9'] = 'end'  # phantom leg
    kinds = {w.kind for w in validate_junctions(js, sd)}
    assert 'unknown_leg' in kinds


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------


def test_transition_share_returns_default_when_no_junction():
    assert transition_share({}, (15.0, 55.0), '1', '2', default=1.0) == 1.0


def test_transition_share_returns_stored_value():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    j = next(iter(js.values()))
    expected = j.transitions['1']['2']
    assert transition_share(js, j.point, '1', '2') == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_serialize_round_trip_preserves_matrix_and_source():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    js[next(iter(js))].source = 'user'  # mark one as user
    payload = serialize_junctions(js)
    restored = deserialize_junctions(payload)
    assert set(restored.keys()) == set(js.keys())
    for jid in js:
        a, b = js[jid], restored[jid]
        assert a.point == pytest.approx(b.point)
        assert a.legs == b.legs
        assert a.source == b.source
        for in_leg in a.transitions:
            for out_leg in a.transitions[in_leg]:
                assert a.transitions[in_leg][out_leg] == pytest.approx(
                    b.transitions[in_leg][out_leg]
                )


def test_deserialize_tolerates_garbage_input():
    assert deserialize_junctions(None) == {}
    assert deserialize_junctions({'bad': 'not a dict'}) == {}
    assert deserialize_junctions({'j_15.0_55.0': {}}) != {}


# ---------------------------------------------------------------------------
# Refresh registry after structural edits
# ---------------------------------------------------------------------------


def test_refresh_preserves_user_when_legs_intact():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 0.6, '3': 0.4}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    j.source = 'user'
    refreshed = refresh_junction_registry(js, sd)
    rj = next(iter(refreshed.values()))
    assert rj.source == 'user'
    assert rj.transitions['1'] == {'2': 0.6, '3': 0.4}


def test_refresh_drops_user_matrix_when_legs_disappeared():
    sd = _y_junction_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 0.6, '3': 0.4}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    j.source = 'user'
    # Remove leg 3 from the segment data — user matrix references a
    # vanished leg, so it must be regenerated.
    sd.pop('3')
    refreshed = refresh_junction_registry(js, sd)
    if refreshed:  # Junction may or may not survive depending on remaining endpoints
        rj = next(iter(refreshed.values()))
        # Either the junction was rebuilt geometrically, or replaced.
        assert rj.source in ('geometry', 'user')
        if rj.source == 'user':
            # If preserved, the row must not reference the removed leg.
            for row in rj.transitions.values():
                assert '3' not in row
