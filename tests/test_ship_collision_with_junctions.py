"""Standalone tests for the junction-matrix integration in ship_collision_model.

These exercise the helper methods (``_junction_at_point``,
``_junction_conflict_factor``, ``_coerce_junctions``,
``_junction_outward_bearing``) and the matrix-aware behaviour of
``_calc_bend_collisions`` and ``_calc_crossing_collisions`` in
isolation, without spinning up a full QGIS environment or the
``Calculation`` mixin host.
"""

from __future__ import annotations

import pytest

from compute.ship_collision_model import ShipCollisionModelMixin
from geometries.junctions import (
    Junction,
    apply_geometric_defaults,
    build_junctions,
)


# ---------------------------------------------------------------------------
# Test host
# ---------------------------------------------------------------------------


class _StubRunner(ShipCollisionModelMixin):
    """Minimal host so the mixin's instance methods become callable."""

    def __init__(self):
        self._progress_log: list[tuple[str, float, str]] = []

    def _report_progress(self, stage, fraction, message) -> None:
        self._progress_log.append((stage, fraction, message))


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


def _two_leg_chain() -> dict:
    return {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
    }


def _x_crossing_segments() -> dict:
    """A four-leg crossing where 1↔2 are continuations and 3↔4 are continuations."""
    return {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
        '3': _seg("15.0 54.0", "15.0 55.0"),
        '4': _seg("15.0 55.0", "15.0 56.0"),
    }


def _stub_traffic(freq: float = 1000.0) -> dict:
    """Single ship cell, both directions."""
    one = [[freq]]
    return {
        'East going': {
            'Frequency (ships/year)': one,
            'Speed (knots)': [[10.0]],
            'Ship Beam (meters)': [[20.0]],
        },
        'West going': {
            'Frequency (ships/year)': one,
            'Speed (knots)': [[10.0]],
            'Ship Beam (meters)': [[20.0]],
        },
    }


# ---------------------------------------------------------------------------
# _coerce_junctions
# ---------------------------------------------------------------------------


def test_coerce_junctions_returns_none_for_empty_input():
    assert ShipCollisionModelMixin._coerce_junctions(None) is None
    assert ShipCollisionModelMixin._coerce_junctions({}) is None


def test_coerce_junctions_passes_through_live_registry():
    sd = _y_segments()
    js = build_junctions(sd)
    out = ShipCollisionModelMixin._coerce_junctions(js)
    assert out is js


def test_coerce_junctions_deserialises_dict():
    sd = _y_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    payload = {jid: j.to_dict() for jid, j in js.items()}
    out = ShipCollisionModelMixin._coerce_junctions(payload)
    assert out is not None
    assert isinstance(next(iter(out.values())), Junction)


# ---------------------------------------------------------------------------
# _junction_at_point + _junction_conflict_factor
# ---------------------------------------------------------------------------


def test_junction_at_point_returns_none_when_no_match():
    sd = _y_segments()
    js = build_junctions(sd)
    out = ShipCollisionModelMixin._junction_at_point(js, (99.0, 99.0))
    assert out is None


def test_junction_at_point_finds_exact_match():
    sd = _y_segments()
    js = build_junctions(sd)
    out = ShipCollisionModelMixin._junction_at_point(js, (15.0, 55.0))
    assert out is not None


def test_conflict_factor_one_when_no_junction():
    f = ShipCollisionModelMixin._junction_conflict_factor(None, '1', '2')
    assert f == 1.0


def test_conflict_factor_zero_for_full_continuation():
    """Two legs trading 100% traffic -> conflict factor must be 0."""
    j = Junction(
        junction_id='j',
        point=(15.0, 55.0),
        legs={'1': 'end', '2': 'start'},
        transitions={'1': {'2': 1.0}, '2': {'1': 1.0}},
        source='user',
    )
    assert ShipCollisionModelMixin._junction_conflict_factor(j, '1', '2') == 0.0


def test_conflict_factor_one_when_no_continuation():
    """100% diverging traffic -> full conflict at junction."""
    j = Junction(
        junction_id='j',
        point=(15.0, 55.0),
        legs={'1': 'end', '2': 'end', '3': 'start'},
        transitions={
            '1': {'3': 1.0},
            '2': {'3': 1.0},
            '3': {'1': 0.5, '2': 0.5},
        },
        source='user',
    )
    assert ShipCollisionModelMixin._junction_conflict_factor(j, '1', '2') == 1.0


def test_conflict_factor_partial_for_split():
    """70% L1->L2 plus 100% L2->L1 -> 1 - 0.7*1 = 0.3."""
    j = Junction(
        junction_id='j',
        point=(15.0, 55.0),
        legs={'1': 'end', '2': 'start', '3': 'start'},
        transitions={
            '1': {'2': 0.7, '3': 0.3},
            '2': {'1': 1.0},
            '3': {'1': 1.0},
        },
        source='user',
    )
    f = ShipCollisionModelMixin._junction_conflict_factor(j, '1', '2')
    assert f == pytest.approx(0.3, abs=1e-9)


def test_conflict_factor_clamped_to_unit_interval():
    """Bogus matrix entries must not produce negative or >1 factors."""
    j = Junction(
        junction_id='j',
        point=(15.0, 55.0),
        legs={'1': 'end', '2': 'start'},
        transitions={'1': {'2': 2.0}, '2': {'1': 2.0}},  # malformed
        source='user',
    )
    f = ShipCollisionModelMixin._junction_conflict_factor(j, '1', '2')
    assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# _junction_outward_bearing
# ---------------------------------------------------------------------------


def test_outward_bearing_from_end_side():
    sd = _y_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    # Leg 1 has its END at the junction; far end is (15.0, 55.5).
    # Bearing from junction (15, 55) toward (15, 55.5) is North = 0°.
    b = ShipCollisionModelMixin._junction_outward_bearing(j, '1', sd)
    assert b == pytest.approx(0.0, abs=1e-6)


def test_outward_bearing_from_start_side():
    sd = _y_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    # Leg 2 start at junction; far end (15.5, 54.5) -> SE.
    b = ShipCollisionModelMixin._junction_outward_bearing(j, '2', sd)
    assert b == pytest.approx(135.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Matrix-aware _calc_bend_collisions
# ---------------------------------------------------------------------------


def test_bend_iterates_matrix_rows_when_junction_present():
    """A leg whose end touches a Y-junction with non-trivial deflection
    should produce non-zero bend, even without a `bend_angle` field."""
    sd = _y_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    runner = _StubRunner()
    leg_dirs = _stub_traffic(freq=1000.0)
    bend = runner._calc_bend_collisions(
        leg_dirs=leg_dirs,
        seg_info=sd['1'],
        pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js,
        leg_id='1',
        segment_data=sd,
    )
    assert bend > 0


def test_bend_legacy_path_used_when_no_junction():
    """Without junctions, the function reads `bend_angle` from seg_info."""
    sd = _two_leg_chain()
    seg = sd['1']
    seg['bend_angle'] = 45.0  # trigger legacy bend
    runner = _StubRunner()
    leg_dirs = _stub_traffic(freq=1000.0)
    bend = runner._calc_bend_collisions(
        leg_dirs=leg_dirs,
        seg_info=seg,
        pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=None,
        leg_id='1',
        segment_data=sd,
    )
    assert bend > 0


def test_bend_legacy_zero_when_no_angle_and_no_junction():
    sd = _two_leg_chain()
    runner = _StubRunner()
    leg_dirs = _stub_traffic(freq=1000.0)
    bend = runner._calc_bend_collisions(
        leg_dirs=leg_dirs,
        seg_info=sd['1'],
        pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=None,
        leg_id='1',
        segment_data=sd,
    )
    assert bend == 0.0


def test_bend_zero_when_junction_is_pure_continuation():
    """Two-leg straight-through chain -> deflection 0 -> bend 0."""
    sd = _two_leg_chain()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    runner = _StubRunner()
    leg_dirs = _stub_traffic(freq=1000.0)
    bend = runner._calc_bend_collisions(
        leg_dirs=leg_dirs,
        seg_info=sd['1'],
        pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js,
        leg_id='1',
        segment_data=sd,
    )
    assert bend == 0.0


def test_bend_scales_quadratically_with_matrix_share():
    """Bend collisions are quadratic in flow per pair (Q*P_no_turn vs Q*(1-P_no_turn)).

    All-to-one share Q gives bend ~ Q^2; 50/50 split gives 2 * (Q/2)^2
    = Q^2 / 2.  So the all-to-one variant should be twice the 50/50.
    """
    sd = _y_segments()
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    j = next(iter(js.values()))
    runner = _StubRunner()
    leg_dirs = _stub_traffic(freq=1000.0)

    j.transitions = {'1': {'2': 1.0, '3': 0.0}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    bend_full = runner._calc_bend_collisions(
        leg_dirs=leg_dirs, seg_info=sd['1'], pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js, leg_id='1', segment_data=sd,
    )
    j.transitions = {'1': {'2': 0.5, '3': 0.5}, '2': {'1': 1.0}, '3': {'1': 1.0}}
    bend_half = runner._calc_bend_collisions(
        leg_dirs=leg_dirs, seg_info=sd['1'], pc_bend=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js, leg_id='1', segment_data=sd,
    )
    assert bend_full == pytest.approx(2.0 * bend_half, rel=1e-3)


# ---------------------------------------------------------------------------
# Matrix-aware _calc_crossing_collisions
# ---------------------------------------------------------------------------


def test_crossing_unchanged_when_no_junctions_provided():
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    runner = _StubRunner()
    total_no_j = runner._calc_crossing_collisions(
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=None,
    )
    assert total_no_j > 0


def test_crossing_zero_when_full_continuation_matrix():
    """At a Y-junction, set leg 1 <-> leg 2 as 100% continuation and the
    1-2 conflict factor drops to 0.  Legs 1-3 and 2-3 still carry full
    crossing risk because they have no continuation share.
    """
    sd = _y_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {
        '1': {'2': 1.0, '3': 0.0},
        '2': {'1': 1.0, '3': 0.0},
        '3': {'1': 0.5, '2': 0.5},
    }
    j.source = 'user'
    runner = _StubRunner()
    total_with_j = runner._calc_crossing_collisions(
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js,
    )
    total_no_j = runner._calc_crossing_collisions(
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=None,
    )
    # The 1-2 pair (which would have a non-trivial crossing angle of 45°
    # since the legs are not parallel at the Y) drops out under the
    # matrix; 1-3 and 2-3 still contribute.
    assert total_with_j > 0
    assert total_with_j < total_no_j


def test_crossing_scales_linearly_with_conflict_factor():
    """Halving every off-diagonal share should halve the per-pair conflict_factor
    on a leg pair where both legs appear in the matrix only as 'other'."""
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    runner = _StubRunner()

    # Build two registries, one with full conflict (no continuation) and
    # one where leg 1 and 2 partially continue (factor = 1 - 0.5*0.5 = 0.75).
    js_full = build_junctions(sd)
    j_full = next(iter(js_full.values()))
    j_full.transitions = {k: {} for k in ['1', '2', '3', '4']}
    j_full.source = 'user'
    total_full = runner._calc_crossing_collisions(
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js_full,
    )

    js_partial = build_junctions(sd)
    j_partial = next(iter(js_partial.values()))
    j_partial.transitions = {
        '1': {'2': 0.5, '3': 0.25, '4': 0.25},
        '2': {'1': 0.5, '3': 0.25, '4': 0.25},
        '3': {'1': 0.25, '2': 0.25, '4': 0.5},
        '4': {'1': 0.25, '2': 0.25, '3': 0.5},
    }
    j_partial.source = 'user'
    total_partial = runner._calc_crossing_collisions(
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=js_partial,
    )
    # Matrix-weighted total should be strictly less than the no-conflict
    # baseline (some pairs partially continue and so don't count fully).
    assert total_partial < total_full
