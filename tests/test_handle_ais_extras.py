"""Extra AIS-import tests covering the audit-flagged gaps.

Focus on areas the audit reported as under-tested in
``omrat_utils.handle_ais``:

* NULL handling for the optional ``beam``/``draught``/``sog``/
  ``air_draught`` columns inside ``update_ais_data`` — those fall
  through ``if x is not None`` guards and must not raise.
* The new ``compute_junction_transitions`` plumbing on top of
  :func:`compute.junction_transitions.transition_counts_from_passages`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from compute.junction_transitions import (
    DEFAULT_TIME_WINDOW_S,
    transition_counts_from_passages,
)


# ---------------------------------------------------------------------------
# update_ais_data NULL-column handling (regression for audit gap)
# ---------------------------------------------------------------------------


def _stub_traffic_block(n_types: int = 21, n_loa: int = 5) -> dict:
    return {
        'Frequency (ships/year)': [[0] * n_loa for _ in range(n_types)],
        'Speed (knots)': [[[] for _ in range(n_loa)] for _ in range(n_types)],
        'Ship heights (meters)': [[[] for _ in range(n_loa)] for _ in range(n_types)],
        'Ship Beam (meters)': [[[] for _ in range(n_loa)] for _ in range(n_types)],
        'Draught (meters)': [[[] for _ in range(n_loa)] for _ in range(n_types)],
    }


@pytest.fixture
def stub_ais():
    """Return an :class:`AIS` instance with the heavy bits mocked away."""
    from omrat_utils.handle_ais import AIS
    with patch("omrat_utils.handle_ais.DB"), \
         patch("omrat_utils.handle_ais.AISConnectionWidget"):
        omrat = MagicMock()
        omrat.traffic = SimpleNamespace(
            traffic_data={
                'L1': {'East': _stub_traffic_block(), 'West': _stub_traffic_block()},
            },
        )
        ais = AIS(omrat)
        ais.max_deviation = 45.0
        ais.schema = "test"
        ais.year = 2023
        ais.months = []
        return ais


def test_update_ais_data_handles_null_beam(stub_ais):
    """beam=None must not append to the Ship Beam list (skipped via guard)."""
    row = [100, None, 70, 6.0, 'cargo', '2024-01-01', 12.0, 20.0, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['East', 'West'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Ship Beam (meters)'][18][3] == []


def test_update_ais_data_handles_null_sog(stub_ais):
    row = [100, 20, 70, 6.0, 'cargo', '2024-01-01', None, 20.0, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['East', 'West'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Speed (knots)'][18][3] == []
    assert td['Frequency (ships/year)'][18][3] == 1  # frequency still incremented


def test_update_ais_data_handles_null_draught_and_air_draught(stub_ais):
    row = [100, 20, 70, None, 'cargo', '2024-01-01', 12.0, None, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['East', 'West'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Draught (meters)'][18][3] == []
    assert td['Ship heights (meters)'][18][3] == []
    assert td['Frequency (ships/year)'][18][3] == 1


def test_update_ais_data_all_optional_columns_null(stub_ais):
    row = [None, None, None, None, None, '2024-01-01', None, None, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['East', 'West'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    # toc=None -> get_type returns 20 (Other Type); loa=None -> 100 -> bucket 3.
    assert td['Frequency (ships/year)'][20][3] == 1


# ---------------------------------------------------------------------------
# compute_junction_transitions wiring
# ---------------------------------------------------------------------------


def test_compute_junction_transitions_returns_empty_with_no_handler(stub_ais):
    stub_ais.omrat.junctions = None
    assert stub_ais.compute_junction_transitions() == {}


def test_compute_junction_transitions_returns_empty_with_no_db(stub_ais):
    """Even if junctions exist, no DB connection -> empty dict."""
    from omrat_utils.handle_junctions import Junctions
    handler = Junctions(stub_ais.omrat)
    handler.registry = {'j_x': MagicMock(legs={'1': 'end', '2': 'start'})}
    stub_ais.omrat.junctions = handler
    stub_ais.db = None
    assert stub_ais.compute_junction_transitions() == {}


def test_compute_junction_transitions_runs_pure_counter_per_junction(stub_ais):
    """Mock fetch_passages_for_leg so no real DB is consulted."""
    from omrat_utils.handle_junctions import Junctions
    handler = Junctions(stub_ais.omrat)
    junction = SimpleNamespace(legs={'1': 'end', '2': 'start'})
    handler.registry = {'j_x': junction}
    stub_ais.omrat.junctions = handler
    stub_ais.db = MagicMock()  # truthy; fetch is mocked below
    stub_ais.get_segment_data_from_table = MagicMock(return_value={
        '1': {'Start_Point': '14 55', 'End_Point': '15 55', 'Width': 5000},
        '2': {'Start_Point': '15 55', 'End_Point': '16 55', 'Width': 5000},
    })

    def fake_fetch(leg_d, near_radius_m=None):
        if leg_d['Start_Point'] == '14 55':
            return {'mmsi1': [100.0], 'mmsi2': [200.0]}
        return {'mmsi1': [150.0], 'mmsi2': [250.0]}

    stub_ais.fetch_passages_for_leg = fake_fetch
    out = stub_ais.compute_junction_transitions()
    assert 'j_x' in out
    # Both ships passed leg 1 first, then leg 2.
    assert out['j_x']['1']['2'] == 2


# ---------------------------------------------------------------------------
# Round-trip sanity: ``transition_counts_from_passages`` agrees with the wiring
# ---------------------------------------------------------------------------


def test_pure_counter_drives_handler_apply():
    """End-to-end: counts -> normalised matrix -> stored on Junction."""
    from geometries.junctions import Junction
    from omrat_utils.handle_junctions import Junctions
    counts = transition_counts_from_passages({
        # All ten ships start on leg 1; 7 continue to leg 2, 3 to leg 3.
        '1': {f'm{i}': [100.0] for i in range(10)},
        '2': {f'm{i}': [200.0] for i in range(7)},
        '3': {f'm{i}': [200.0] for i in range(7, 10)},
    })
    handler = Junctions()
    handler.registry = {
        'j_x': Junction(
            junction_id='j_x',
            point=(15.0, 55.0),
            legs={'1': 'end', '2': 'start', '3': 'start'},
        ),
    }
    n = handler.apply_ais_counts({'j_x': counts}, segment_data={})
    assert n == 1
    assert handler.registry['j_x'].source == 'ais'
    # 7/10 went to leg 2; 3/10 went to leg 3.
    assert handler.registry['j_x'].transitions['1']['2'] == pytest.approx(0.7)
    assert handler.registry['j_x'].transitions['1']['3'] == pytest.approx(0.3)
