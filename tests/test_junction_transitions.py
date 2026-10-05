"""Standalone tests for the AIS-derived junction transition module."""

from __future__ import annotations

import pytest

from compute.junction_transitions import (
    DEFAULT_TIME_WINDOW_S,
    Crossing,
    is_inbound,
    match_summary,
    normalise_counts_to_shares,
    transition_counts_from_crossings,
)

# Y junction: leg 1 ends at the junction, legs 2 and 3 start there.  A ship
# sailing 1 -> 2 crosses leg 1 in its drawn direction (towards the junction)
# and then leg 2 in its drawn direction (away from it).
SIDES_Y = {'1': 'end', '2': 'start', '3': 'start'}


def _c(mmsi, t, along, route=None):
    return Crossing(str(mmsi), route, float(t), along)


def _trip(mmsi, route, in_leg, out_leg, sides=SIDES_Y, t0=100.0, dt=600.0):
    """One trip sailing ``in_leg -> out_leg``: in at ``t0``, out at ``t0 + dt``.

    The direction flags follow from the junction sides, so the first
    crossing heads towards the junction and the second away from it.
    """
    return {
        in_leg: [_c(mmsi, t0, sides[in_leg] == 'end', route)],
        out_leg: [_c(mmsi, t0 + dt, sides[out_leg] == 'start', route)],
    }


def _merge(*parts):
    out: dict[str, list[Crossing]] = {}
    for part in parts:
        for leg, crossings in part.items():
            out.setdefault(leg, []).extend(crossings)
    return out


def test_is_inbound_follows_the_junction_side():
    assert is_inbound('end', True) is True      # drawn direction arrives at the end
    assert is_inbound('end', False) is False
    assert is_inbound('start', False) is True   # reciprocal arrives at the start
    assert is_inbound('start', True) is False
    assert is_inbound(None, True) is None


def test_empty_input_returns_empty_dict():
    assert transition_counts_from_crossings({}, {}) == {}


def test_single_leg_returns_empty_row():
    out = transition_counts_from_crossings({'1': [_c(1, 100, True, 5)]}, {'1': 'end'})
    assert out == {'1': {}}


def test_simple_transit_on_one_route_id():
    crossings = _trip(111, 5, '1', '2')
    out = transition_counts_from_crossings(crossings, SIDES_Y)
    assert out == {'1': {'2': 1}, '2': {}}


def test_reverse_flow_lands_in_the_other_row():
    """2 -> 1: reciprocal on leg 2 (inbound at its start), reciprocal on leg 1 (outbound at its end)."""
    crossings = {
        '2': [_c(111, 100, False, 5)],
        '1': [_c(111, 700, False, 5)],
    }
    out = transition_counts_from_crossings(crossings, {'1': 'end', '2': 'start'})
    assert out == {'2': {'1': 1}, '1': {}}


def test_outbound_then_outbound_is_not_a_transit():
    """Leaving on leg 1 and then crossing leg 2 away from the junction is no transit."""
    crossings = {
        '1': [_c(111, 100, False, 5)],   # heading away from the junction on leg 1
        '2': [_c(111, 700, True, 5)],    # heading away on leg 2
    }
    out = transition_counts_from_crossings(crossings, {'1': 'end', '2': 'start'})
    assert out == {'1': {}, '2': {}}


def test_split_at_y_junction():
    """7 of 10 trips continue from leg 1 to leg 2; 3 to leg 3."""
    crossings = _merge(
        *[_trip(f'm{i}', i, '1', '2') for i in range(7)],
        *[_trip(f'm{i}', i, '1', '3') for i in range(7, 10)],
    )
    out = transition_counts_from_crossings(crossings, SIDES_Y)
    assert out['1'] == {'2': 7, '3': 3}
    assert out['2'] == {} and out['3'] == {}
    assert normalise_counts_to_shares(out)['1'] == {'2': pytest.approx(0.7), '3': pytest.approx(0.3)}


def test_x_crossing_shares_over_matched_transits():
    """Route 1 (1a -> 1b) crossed by route 2 (2a -> 2b), split at the crossing."""
    sides = {'1a': 'end', '1b': 'start', '2a': 'end', '2b': 'start'}
    crossings = _merge(
        *[_trip(f'a{i}', i, '1a', '1b', sides) for i in range(700)],
        *[_trip(f'b{i}', i, '1a', '2a', sides) for i in range(40)],
        *[_trip(f'c{i}', i, '1a', '2b', sides) for i in range(60)],
    )
    shares = normalise_counts_to_shares(transition_counts_from_crossings(crossings, sides))
    assert shares['1a'] == {
        '1b': pytest.approx(0.875), '2a': pytest.approx(0.05), '2b': pytest.approx(0.075),
    }


def test_every_trip_of_a_vessel_counts():
    """A ferry on three port-to-port trips counts three transits, not one."""
    crossings = _merge(
        *[_trip('ferry', rid, '1', '2', t0=rid * 86400.0) for rid in (11, 12, 13)],
    )
    out = transition_counts_from_crossings(crossings, SIDES_Y)
    assert out['1'] == {'2': 3}


def test_route_id_trip_has_no_time_window():
    """Anchoring for longer than the window between the two crossings is still one trip."""
    crossings = _trip(111, 5, '1', '2', dt=DEFAULT_TIME_WINDOW_S * 5)
    out = transition_counts_from_crossings(crossings, SIDES_Y)
    assert out['1'] == {'2': 1}


def test_trip_passing_the_junction_twice_counts_twice():
    sides = {'1': 'end', '2': 'start'}
    crossings = {
        '1': [_c(111, 100, True, 5), _c(111, 5000, False, 5)],   # in on 1, ..., out on 1
        '2': [_c(111, 700, True, 5), _c(111, 4000, False, 5)],   # out on 2, ..., back in on 2
    }
    out = transition_counts_from_crossings(crossings, sides)
    assert out == {'1': {'2': 1}, '2': {'1': 1}}


def test_route_ids_of_different_vessels_are_not_merged():
    """route_id may be numbered per vessel: the trip key is (mmsi, route_id)."""
    crossings = {
        '1': [_c('aaa', 100, True, 1)],
        '2': [_c('bbb', 700, True, 1)],
    }
    out = transition_counts_from_crossings(crossings, {'1': 'end', '2': 'start'})
    assert out == {'1': {}, '2': {}}


def test_crossings_without_route_id_fall_back_to_mmsi_window():
    near = _trip(111, None, '1', '2', dt=600.0)
    far = _trip(222, None, '1', '3', dt=DEFAULT_TIME_WINDOW_S + 60.0)
    out = transition_counts_from_crossings(_merge(near, far), SIDES_Y)
    assert out['1'] == {'2': 1}


def test_time_window_overrides_default():
    crossings = _trip(111, None, '1', '2', dt=5.0)
    assert transition_counts_from_crossings(crossings, SIDES_Y, time_window_s=1.0)['1'] == {}
    assert transition_counts_from_crossings(crossings, SIDES_Y, time_window_s=10.0)['1'] == {'2': 1}


def test_u_turn_is_not_a_transit():
    """In on leg 1 and back out on leg 1: the ship never used the junction."""
    crossings = {'1': [_c(111, 100, True, 5), _c(111, 700, False, 5)], '2': []}
    out = transition_counts_from_crossings(crossings, {'1': 'end', '2': 'start'})
    assert out == {'1': {}, '2': {}}


def test_same_segment_crossing_both_lines_pairs_inbound_first():
    """A long straight segment crosses both passage lines: equal t, inbound sorts first."""
    crossings = {
        '2': [_c(111, 100, True, 5)],   # outbound, listed first
        '1': [_c(111, 100, True, 5)],   # inbound
    }
    out = transition_counts_from_crossings(crossings, {'1': 'end', '2': 'start'})
    assert out['1'] == {'2': 1}


def test_unmatched_crossings_do_not_change_the_shares():
    """Trips ending between the passage line and the junction stay out of the shares."""
    matched = _merge(
        *[_trip(f'm{i}', i, '1', '2') for i in range(7)],
        *[_trip(f'm{i}', i, '1', '3') for i in range(7, 10)],
    )
    unmatched = {'1': [_c(f'u{i}', 100, True, i) for i in range(5)]}
    crossings = _merge(matched, unmatched)
    counts = transition_counts_from_crossings(crossings, SIDES_Y)
    assert normalise_counts_to_shares(counts)['1'] == {'2': pytest.approx(0.7), '3': pytest.approx(0.3)}
    summary = match_summary(crossings, SIDES_Y, counts)
    assert (summary.matched, summary.inbound) == (10, 15)
    assert summary.with_route_id == 25   # 10 trips x 2 crossings + 5 unmatched


def test_match_summary_counts_crossings_without_route_id():
    crossings = _merge(_trip(1, None, '1', '2'), _trip(2, 9, '1', '3'))
    counts = transition_counts_from_crossings(crossings, SIDES_Y)
    summary = match_summary(crossings, SIDES_Y, counts)
    assert summary == (2, 2, 2)


def test_normalise_counts_handles_zero_rows():
    counts = {'1': {'2': 0, '3': 0}, '2': {}, '3': {'1': 5}}
    out = normalise_counts_to_shares(counts)
    assert out == {'1': {}, '2': {}, '3': {'1': 1.0}}


def test_normalise_drops_zero_entries():
    counts = {'1': {'2': 70, '3': 0, '4': 30}}
    out = normalise_counts_to_shares(counts)
    assert out['1'] == {'2': 0.7, '4': 0.3}
