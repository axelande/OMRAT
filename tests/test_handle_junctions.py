"""Standalone tests for the ``Junctions`` plugin handler.

The handler talks to an OMRAT instance via duck-typed attribute access,
so we drive it with a minimal stand-in object.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geometries.junctions import Junction, junction_id_for_point
from omrat_utils.handle_junctions import Junctions


def _seg(start: str, end: str, length_m: float = 100_000) -> dict:
    return {
        'Start_Point': start,
        'End_Point': end,
        'line_length': length_m,
        'Width': 5000,
        'Route_Id': 1,
        'Leg_name': 'LEG',
    }


def _y_segments() -> dict:
    return {
        '1': _seg("15.0 55.5", "15.0 55.0"),
        '2': _seg("15.0 55.0", "15.5 54.5"),
        '3': _seg("15.0 55.0", "14.5 54.5"),
    }


def _make_handler(segment_data=None) -> Junctions:
    parent = SimpleNamespace(segment_data=segment_data or {})
    return Junctions(parent)


# ---------------------------------------------------------------------------
# rebuild_from_segments
# ---------------------------------------------------------------------------


def test_rebuild_seeds_geometric_defaults():
    h = _make_handler(_y_segments())
    h.rebuild_from_segments(prefer_user=False)
    assert len(h) == 1
    j = next(iter(h.registry.values()))
    assert j.source == 'geometry'
    for row in j.transitions.values():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def test_rebuild_preserves_user_when_legs_intact():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments(prefer_user=False)
    j = next(iter(h.registry.values()))
    # Simulate a user override of every row.
    h.set_matrix(j.junction_id, {
        '1': {'2': 0.7, '3': 0.3},
        '2': {'1': 1.0},
        '3': {'1': 1.0},
    })
    h.rebuild_from_segments(prefer_user=True)
    j2 = next(iter(h.registry.values()))
    assert j2.source == 'user'
    assert j2.transitions['1']['2'] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# load/save round-trip
# ---------------------------------------------------------------------------


def test_to_dict_and_load_round_trip():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments()
    j = next(iter(h.registry.values()))
    h.set_matrix(j.junction_id, {
        '1': {'2': 0.7, '3': 0.3},
        '2': {'1': 1.0},
        '3': {'1': 1.0},
    })
    payload = h.to_dict()

    h2 = _make_handler(sd)
    h2.load_from_dict(payload, sd)
    j2 = next(iter(h2.registry.values()))
    assert j2.source == 'user'
    assert j2.transitions['1']['2'] == pytest.approx(0.7)


def test_load_from_empty_payload_seeds_geometric_defaults():
    sd = _y_segments()
    h = _make_handler(sd)
    h.load_from_dict({}, sd)
    assert len(h) == 1
    j = next(iter(h.registry.values()))
    assert j.source == 'geometry'


def test_load_from_none_payload_handles_gracefully():
    sd = _y_segments()
    h = _make_handler(sd)
    h.load_from_dict(None, sd)
    assert len(h) == 1


# ---------------------------------------------------------------------------
# set_row
# ---------------------------------------------------------------------------


def test_set_row_normalises_to_one():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments()
    j = next(iter(h.registry.values()))
    assert h.set_row(j.junction_id, '1', {'2': 70, '3': 30})
    assert j.transitions['1']['2'] == pytest.approx(0.7)
    assert j.transitions['1']['3'] == pytest.approx(0.3)
    assert j.source == 'user'


def test_set_row_clamps_negatives_to_zero():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments()
    j = next(iter(h.registry.values()))
    h.set_row(j.junction_id, '1', {'2': 10, '3': -5})
    assert j.transitions['1']['3'] == pytest.approx(0.0)
    assert j.transitions['1']['2'] == pytest.approx(1.0)


def test_set_row_unknown_junction_returns_false():
    h = _make_handler(_y_segments())
    h.rebuild_from_segments()
    assert h.set_row('j_99_99', '1', {'2': 1.0}) is False


# ---------------------------------------------------------------------------
# share lookup
# ---------------------------------------------------------------------------


def test_share_returns_default_when_no_junction():
    h = _make_handler({})
    assert h.share((10.0, 50.0), '1', '2', default=1.0) == 1.0


def test_share_returns_stored_value():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments()
    j = next(iter(h.registry.values()))
    expected = j.transitions['1']['2']
    assert h.share(j.point, '1', '2') == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AIS counts wiring
# ---------------------------------------------------------------------------


def test_apply_ais_counts_passes_through_to_module_function():
    sd = _y_segments()
    h = _make_handler(sd)
    h.rebuild_from_segments()
    jid = next(iter(h.registry.keys()))
    n = h.apply_ais_counts({jid: {
        '1': {'2': 70, '3': 30},
        '2': {'1': 100},
        '3': {'1': 100},
    }})
    assert n == 1
    assert h.registry[jid].source == 'ais'
    assert h.registry[jid].transitions['1']['2'] == pytest.approx(0.7)
