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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from compute.junction_transitions import transition_counts_from_passages


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
        # Leg drawn westward (bearing 270): Dirs[0] = 'West' (the drawn
        # direction), Dirs[1] = 'East'.  The east-going test ship (cog 90)
        # therefore belongs in dirs[1] = 'East'.
        omrat.traffic = SimpleNamespace(
            traffic_data={
                'L1': {'West': _stub_traffic_block(), 'East': _stub_traffic_block()},
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
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Ship Beam (meters)'][18][3] == []


def test_update_ais_data_handles_null_sog(stub_ais):
    row = [100, 20, 70, 6.0, 'cargo', '2024-01-01', None, 20.0, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Speed (knots)'][18][3] == []
    assert td['Frequency (ships/year)'][18][3] == 1  # frequency still incremented


def test_update_ais_data_handles_null_draught_and_air_draught(stub_ais):
    row = [100, 20, 70, None, 'cargo', '2024-01-01', 12.0, None, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    td = stub_ais.omrat.traffic.traffic_data['L1']['East']
    assert td['Draught (meters)'][18][3] == []
    assert td['Ship heights (meters)'][18][3] == []
    assert td['Frequency (ships/year)'][18][3] == 1


def test_update_ais_data_all_optional_columns_null(stub_ais):
    row = [None, None, None, None, None, '2024-01-01', None, None, 0.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
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


# ---------------------------------------------------------------------------
# junction_pass_needed: skip the junction pass on a single-leg Update AIS
# ---------------------------------------------------------------------------


def _seg(start: str, end: str) -> dict:
    return {'Start_Point': start, 'End_Point': end, 'line_length': 100_000,
            'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG'}


def _y_segments() -> dict:
    return {
        '1': _seg("15.0 55.5", "15.0 55.0"),
        '2': _seg("15.0 55.0", "15.5 54.5"),
        '3': _seg("15.0 55.0", "14.5 54.5"),
    }


def _ais_handler(stub_ais, sd):
    from omrat_utils.handle_junctions import Junctions
    stub_ais.omrat.segment_data = sd
    handler = Junctions(stub_ais.omrat)
    handler.rebuild_from_segments(sd, prefer_user=False)
    stub_ais.omrat.junctions = handler
    return handler


def test_junction_pass_needed_without_handler(stub_ais):
    stub_ais.omrat.junctions = None
    stub_ais.omrat.segment_data = _y_segments()
    assert stub_ais.junction_pass_needed() is True


def test_junction_pass_needed_follows_registry_state(stub_ais):
    sd = _y_segments()
    handler = _ais_handler(stub_ais, sd)
    # Geometric defaults only: the pass has never run.
    assert stub_ais.junction_pass_needed() is True
    for j in handler.registry.values():
        j.source = 'ais'
    assert stub_ais.junction_pass_needed() is False
    # A new leg on the node needs a fresh count.
    sd['4'] = _seg("15.0 55.0", "15.0 54.5")
    assert stub_ais.junction_pass_needed() is True


def test_junction_pass_needed_after_passage_line_edit(stub_ais):
    sd = _y_segments()
    handler = _ais_handler(stub_ais, sd)
    for j in handler.registry.values():
        j.source = 'ais'
    assert stub_ais.junction_pass_needed() is False
    handler.invalidate_legs(['2'])
    assert stub_ais.junction_pass_needed() is True


# ---------------------------------------------------------------------------
# Custom ship-type mapping: the query's ship_type column wins over the AIS code
# ---------------------------------------------------------------------------


def test_update_ais_data_uses_mapped_ship_type(stub_ais):
    """Row layout: loa, beam, toc, draught, ship_type, date, sog, air_draught, dist, cog.
    toc 70 (Cargo -> 18) but the mapping says 19 (Tanker): bin as tanker."""
    row = [120, 20.0, 70, 7.0, 19, None, 12.0, None, 10.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    freq = stub_ais.omrat.traffic.traffic_data['L1']['East']['Frequency (ships/year)']
    assert freq[19][4] == 1
    assert freq[18][4] == 0


def test_update_ais_data_falls_back_to_ais_code_without_mapping(stub_ais):
    row = [120, 20.0, 70, 7.0, None, None, 12.0, None, 10.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    freq = stub_ais.omrat.traffic.traffic_data['L1']['East']['Frequency (ships/year)']
    assert freq[18][4] == 1


def test_update_ais_data_accepts_ais_code_in_mapped_column(stub_ais):
    """An external vessel table that stores the AIS code (80-89) still maps to Tanker."""
    row = [120, 20.0, 70, 7.0, 84, None, 12.0, None, 10.0, 90.0]
    stub_ais.update_ais_data('L1', [row], leg_bearing=270.0, dirs=['West', 'East'])
    freq = stub_ais.omrat.traffic.traffic_data['L1']['East']['Frequency (ships/year)']
    assert freq[19][4] == 1
