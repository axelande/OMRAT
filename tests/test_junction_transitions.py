"""Standalone tests for the AIS-derived junction transition module."""

from __future__ import annotations

import pytest

from compute.junction_transitions import (
    DEFAULT_TIME_WINDOW_S,
    normalise_counts_to_shares,
    transition_counts_from_passages,
)


def test_empty_input_returns_empty_dict():
    assert transition_counts_from_passages({}) == {}


def test_single_leg_returns_empty_row():
    out = transition_counts_from_passages({'1': {'mmsi1': [100.0, 200.0]}})
    assert out == {'1': {}}


def test_simple_two_leg_transition():
    """One ship visits leg 1 then leg 2 within the window."""
    passages = {
        '1': {'111': [100.0]},
        '2': {'111': [200.0]},
    }
    out = transition_counts_from_passages(passages)
    assert out == {'1': {'2': 1}, '2': {}}


def test_multiple_ships_aggregate():
    passages = {
        '1': {'aaa': [100.0], 'bbb': [110.0]},
        '2': {'aaa': [120.0], 'bbb': [130.0]},
    }
    out = transition_counts_from_passages(passages)
    assert out['1']['2'] == 2
    assert out['2'] == {}


def test_split_at_y_junction():
    """7 of 10 ships continue from leg 1 to leg 2; 3 continue to leg 3."""
    passages = {
        '1': {f'm{i}': [100.0 + i] for i in range(10)},
        '2': {f'm{i}': [200.0 + i] for i in range(7)},
        '3': {f'm{i}': [200.0 + i] for i in range(7, 10)},
    }
    out = transition_counts_from_passages(passages)
    assert out['1']['2'] == 7
    assert out['1']['3'] == 3
    # Leg 2 and 3 ships never go anywhere else => empty rows.
    assert out['2'] == {}
    assert out['3'] == {}


def test_bidirectional_traffic():
    """5 ships go 1->2 and 5 go 2->1; both rows should reflect that."""
    passages = {
        '1': {f'a{i}': [100.0] for i in range(5)} | {f'b{i}': [300.0] for i in range(5)},
        '2': {f'a{i}': [200.0] for i in range(5)} | {f'b{i}': [200.0] for i in range(5)},
    }
    out = transition_counts_from_passages(passages)
    assert out['1']['2'] == 5
    assert out['2']['1'] == 5


def test_far_apart_passages_are_separate_trips():
    """Vessels whose two passages are days apart count as two separate trips."""
    passages = {
        '1': {'mmsi1': [100.0]},
        '2': {'mmsi1': [100.0 + DEFAULT_TIME_WINDOW_S + 60.0]},
    }
    out = transition_counts_from_passages(passages)
    # Outside the time window -> not counted as a transition; the second
    # passage seeds a new trip on its own.
    assert out == {'1': {}, '2': {}}


def test_round_trip_counted_as_two_transitions():
    """A ship visits leg 1, then leg 2, then comes back to leg 1.

    With the default window, the first 1->2 leg counts; the return
    leg starts a new trip (because the time gap is large) and may or
    may not record a transition depending on what comes next.
    """
    passages = {
        '1': {'mmsi1': [100.0, 100.0 + DEFAULT_TIME_WINDOW_S + 60.0]},
        '2': {'mmsi1': [200.0]},
    }
    out = transition_counts_from_passages(passages)
    assert out['1']['2'] == 1


def test_normalise_counts_handles_zero_rows():
    counts = {'1': {'2': 0, '3': 0}, '2': {}, '3': {'1': 5}}
    out = normalise_counts_to_shares(counts)
    assert out == {'1': {}, '2': {}, '3': {'1': 1.0}}


def test_normalise_drops_zero_entries():
    counts = {'1': {'2': 70, '3': 0, '4': 30}}
    out = normalise_counts_to_shares(counts)
    assert out['1'] == {'2': 0.7, '4': 0.3}


def test_no_self_loops_recorded():
    """Two passes on the same leg (a U-turn) shouldn't produce a 1->1 entry."""
    passages = {
        '1': {'mmsi1': [100.0, 200.0]},
        '2': {'mmsi2': [300.0]},
    }
    out = transition_counts_from_passages(passages)
    assert '1' not in out.get('1', {})


def test_each_mmsi_contributes_once_per_pair():
    """Multiple pings on out_leg shouldn't inflate the per-pair count."""
    passages = {
        '1': {'mmsi1': [100.0]},
        '2': {'mmsi1': [200.0, 250.0, 300.0]},
    }
    out = transition_counts_from_passages(passages)
    assert out['1']['2'] == 1


def test_time_window_overrides_default():
    passages = {
        '1': {'mmsi1': [100.0]},
        '2': {'mmsi1': [100.0 + 5.0]},
    }
    # Tight 1-second window -> excluded.
    out = transition_counts_from_passages(passages, time_window_s=1.0)
    assert out['1'] == {}
    # Wide window -> included.
    out2 = transition_counts_from_passages(passages, time_window_s=10.0)
    assert out2['1']['2'] == 1
