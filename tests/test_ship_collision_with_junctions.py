"""Standalone tests for the junction-matrix integration in ship_collision_model.

These exercise the helper methods (``_junction_at_point``,
``_coerce_junctions``, ``_junction_outward_bearing``) and the
route-based ``_calc_crossing_collisions`` (KNOWN_ISSUES 30; bend since
2026-11-10) in isolation, without
spinning up a full QGIS environment or the ``Calculation`` mixin host.
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
# _junction_at_point
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


def test_outward_bearing_from_end_side():
    sd = _y_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    # Leg 1 has its END at the junction; far end is (15.0, 55.5).
    # Bearing from junction (15, 55) toward (15, 55.5) is North = 0°.
    b = ShipCollisionModelMixin._junction_outward_bearing(j, '1', sd)
    assert b == pytest.approx(0.0, abs=1e-6)


def _gc_bearing(p, q) -> float:
    """Great-circle initial bearing (degrees) from p to q (lon, lat)."""
    import math
    lon1, lat1, lon2, lat2 = map(math.radians, (*p, *q))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.degrees(math.atan2(y, x)) % 360.0


def test_outward_bearing_from_start_side():
    """Leg 2 starts at the junction (15, 55); far end (15.5, 54.5).

    The true bearing is ~150 degrees: half a degree of longitude at 55 N is
    only 0.57 of half a degree of latitude.  The raw lon/lat formula used
    until 2026-11-12 gave 135 (KNOWN_ISSUES 36).
    """
    sd = _y_segments()
    js = build_junctions(sd)
    j = next(iter(js.values()))
    b = ShipCollisionModelMixin._junction_outward_bearing(j, '2', sd)
    assert b == pytest.approx(_gc_bearing((15.0, 55.0), (15.5, 54.5)), abs=0.1)
    assert b == pytest.approx(150.0, abs=0.5)


def test_waypoint_turn_is_measured_in_the_projection_not_in_degrees():
    """At 57 N a turn that is 44 degrees on the chart was 28 in raw lon/lat degrees (KNOWN_ISSUES 36)."""
    import math
    wp = (11.0, 57.0)
    a = (wp[0] - 0.4, wp[1])                    # leg 1 arrives heading east
    far = (wp[0] + 0.264, wp[1] + 0.139)        # leg 2 leaves at ~46 degrees true
    d = abs((_gc_bearing(wp, a) + 180.0) - _gc_bearing(wp, far)) % 360.0
    true_turn = min(d, 360.0 - d)               # great circle, at the waypoint: ~44 degrees
    sd = {'1': _seg(f"{a[0]} {a[1]}", f"{wp[0]} {wp[1]}"), '2': _seg(f"{wp[0]} {wp[1]}", f"{far[0]} {far[1]}")}
    j = next(iter(build_junctions(sd).values()))
    b1 = ShipCollisionModelMixin._junction_outward_bearing(j, '1', sd)
    b2 = ShipCollisionModelMixin._junction_outward_bearing(j, '2', sd)
    turn = 180.0 - abs((b1 - b2 + 180.0) % 360.0 - 180.0)
    assert true_turn == pytest.approx(44.0, abs=2.0)
    assert turn == pytest.approx(true_turn, abs=0.15)
    raw = abs(math.degrees(math.atan2(far[0] - wp[0], far[1] - wp[1])
                           - math.atan2(wp[0] - a[0], wp[1] - a[1])))
    assert raw == pytest.approx(28.0, abs=1.0)   # what the old formula saw


# ---------------------------------------------------------------------------
# Bend = the two reverse routes of a turn (2026-11-10)
# ---------------------------------------------------------------------------


def _bend_segments(turn_deg: float) -> dict:
    """Leg 1 east into the waypoint, leg 2 leaving it ``turn_deg`` to port in lon/lat degrees.

    At the equator a degree of latitude is 0.7 % shorter than one of
    longitude, so the true turn is a little smaller (40 -> 39.8 degrees).
    """
    import math
    rad = math.radians(turn_deg)
    end = f"{math.cos(rad):.6f} {math.sin(rad):.6f}"
    return {'1': _seg("-1.0 0.0", "0.0 0.0"), '2': _seg("0.0 0.0", end)}


def _bend_kwargs(sd, td, js, **over):
    kw = dict(_kwargs(sd, td, js), pc_bend=1.3e-4)
    kw.update(over)
    return kw


def test_bend_is_the_crossing_formula_between_the_reverse_routes():
    """Leg 1 -> 2 and 2 -> 1, every ship of both, at 180 - turn; counted once (no omission probability)."""
    import math
    from compute.basic_equations import get_crossing_collision_candidates
    sd = _bend_segments(40.0)
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    runner = _StubRunner()
    by_cell: dict = {}
    totals = runner._calc_crossing_collisions(**_bend_kwargs(sd, td, None, by_cell_bend=by_cell))
    js = runner._route_registry(None, sd)
    j = next(iter(js.values()))
    b1 = runner._junction_outward_bearing(j, '1', sd)
    b2 = runner._junction_outward_bearing(j, '2', sd)
    theta = math.radians(abs((b1 - b2 + 180.0) % 360.0 - 180.0))   # = 180 - turn
    # 40 degrees in lon/lat at the equator; the ellipsoid (a degree of latitude
    # is 0.7 % shorter than one of longitude there) makes the turn 39.8.
    assert math.degrees(theta) == pytest.approx(140.0, abs=0.3)
    v = 10.0 * 1852.0 / 3600.0
    expected = get_crossing_collision_candidates(1000.0, 1000.0, v, v, 100.0, 100.0, 20.0, 20.0, theta) * 1.3e-4
    assert totals['bend'] == pytest.approx(expected, rel=1e-9)
    assert totals['crossing'] == 0.0 and totals['merging'] == 0.0
    assert sum(by_cell.values()) == pytest.approx(expected, rel=1e-9)


def test_bend_zero_when_junction_is_pure_continuation():
    """Straight through: the reverse routes are head-on on the legs, not a bend."""
    sd = _two_leg_chain()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    assert _StubRunner()._calc_crossing_collisions(**_bend_kwargs(sd, td, None))['bend'] == 0.0


def test_bend_skipped_within_the_angle_limit():
    """Default limit 10 degrees: an 8-degree turn gives no bend, a 12-degree turn does."""
    td = {k: _stub_traffic(freq=1000.0) for k in ('1', '2')}
    for turn in (3.0, 8.0):
        sd = _bend_segments(turn)
        assert _StubRunner()._calc_crossing_collisions(**_bend_kwargs(sd, td, None))['bend'] == 0.0
    sd = _bend_segments(12.0)
    assert _StubRunner()._calc_crossing_collisions(**_bend_kwargs(sd, td, None))['bend'] > 0.0


def test_angle_limit_follows_the_setting():
    """``pc['min_encounter_angle_deg']`` = 5: the 8-degree turn now counts, the 3-degree one still not."""
    from compute.encounter_angle import min_angle_rad
    td = {k: _stub_traffic(freq=1000.0) for k in ('1', '2')}
    runner = _StubRunner()
    runner._route_angle_limit_rad = min_angle_rad({'min_encounter_angle_deg': 5.0})
    assert runner._calc_crossing_collisions(**_bend_kwargs(_bend_segments(8.0), td, None))['bend'] > 0.0
    assert runner._calc_crossing_collisions(**_bend_kwargs(_bend_segments(3.0), td, None))['bend'] == 0.0


class _RunHost(_StubRunner):
    """Runs the whole collision model without layers or a dock."""

    def _finalize_collision_layers(self, segment_data) -> None:
        pass

    def _update_collision_ui(self, result) -> None:
        pass


@pytest.mark.parametrize('pc, expected_deg, has_bend', [
    ({}, 10.0, False),
    ({'min_encounter_angle_deg': 5.0}, 5.0, True),
])
def test_run_reads_the_angle_from_pc(pc, expected_deg, has_bend):
    sd = _bend_segments(8.0)
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    host = _RunHost()
    data = {'traffic_data': td, 'segment_data': sd, 'pc': dict(pc, bend=1.3e-4),
            'ship_categories': {'length_intervals': [{'min': 50.0, 'max': 150.0}]}}
    result = host.run_ship_collision_model(data)
    assert host.collision_report['min_encounter_angle_deg'] == pytest.approx(expected_deg)
    assert (result['bend'] > 0.0) is has_bend


def test_bend_follows_both_route_shares():
    """Y-junction: bend(1, 2) is proportional to T[1][2] x T[2][1]."""
    sd = _y_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    j = next(iter(js.values()))

    def pair_bend(t1, t2):
        j.transitions = {'1': t1, '2': t2, '3': {'1': 1.0}}
        j.source = 'user'   # a geometry matrix is re-derived by the registry refresh
        pairs: dict = {}
        _StubRunner()._calc_crossing_collisions(**_bend_kwargs(sd, td, js, by_leg_pair=pairs))
        return pairs.get(('1', '2'), {}).get('bend', 0.0)

    full = pair_bend({'2': 1.0, '3': 0.0}, {'1': 1.0})
    assert full > 0
    assert pair_bend({'2': 0.5, '3': 0.5}, {'1': 1.0}) == pytest.approx(0.5 * full, rel=1e-9)
    assert pair_bend({'2': 0.5, '3': 0.5}, {'1': 0.5, '3': 0.5}) == pytest.approx(0.25 * full, rel=1e-9)


def test_bend_split_half_to_each_leg_sums_to_the_total():
    by_leg = {'1': {'head_on': 1.0, 'overtaking': 0.0, 'bend': 0.0},
              '2': {'head_on': 0.0, 'overtaking': 0.0, 'bend': 0.0}}
    pairs = {('1', '2'): {'crossing': 0.0, 'merging': 0.0, 'bend': 4e-4}}
    ShipCollisionModelMixin._split_bend_over_legs(by_leg, pairs)
    assert by_leg['1']['bend'] == pytest.approx(2e-4) and by_leg['2']['bend'] == pytest.approx(2e-4)
    result = {}
    ShipCollisionModelMixin._fill_result_totals(result, by_leg, {'crossing': 0.0, 'merging': 0.0, 'bend': 4e-4})
    assert result['bend'] == pytest.approx(4e-4)
    assert ShipCollisionModelMixin._bend_by_pair(pairs)['1->2']['bend'] == pytest.approx(4e-4)


# ---------------------------------------------------------------------------
# Matrix-aware _calc_crossing_collisions
# ---------------------------------------------------------------------------

def _crossing_sum(runner, **kwargs) -> float:
    """Combined crossing + merging total.

    ``_calc_crossing_collisions`` returns the two kinds separately;
    these tests exercise the junction conflict factor, which applies
    identically to both, so the sum is the quantity of interest.
    """
    totals = runner._calc_crossing_collisions(**kwargs)
    return float(totals['crossing'] + totals['merging'])


def test_crossing_unchanged_when_no_junctions_provided():
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    runner = _StubRunner()
    total_no_j = _crossing_sum(
        runner,
        traffic_data=td, segment_data=sd,
        leg_keys=list(sd.keys()), pc_crossing=1.3e-4, pc_merging=1.3e-4,
        length_intervals=[{'min': 50, 'max': 150}],
        junctions=None,
    )
    assert total_no_j > 0


def _single_pair(theta_rad: float, q1=1000.0, q2=1000.0, pc=1.3e-4) -> float:
    """One pair of routes, one ship cell each (L = 100 m, B = 20 m, 10 kn)."""
    from compute.basic_equations import get_crossing_collision_candidates
    v = 10.0 * 1852.0 / 3600.0
    return get_crossing_collision_candidates(q1, q2, v, v, 100.0, 100.0, 20.0, 20.0, theta_rad) * pc


def _x_route_angles(sd) -> dict:
    """Angle between the arrival courses of each crossing route pair of the X, from the model's bearings.

    A chord along a parallel is not exactly perpendicular to a meridian
    (the great circle bows towards the pole), so at 55 N with 1-degree legs
    the pairs are 90 +- 0.41 degrees -- the WGS84 geodesic azimuths agree
    to 0.003 degrees.
    """
    import math
    j = next(iter(build_junctions(sd).values()))
    b = {leg: ShipCollisionModelMixin._junction_outward_bearing(j, leg, sd) for leg in sd}
    course = {('1', '2'): b['1'] + 180.0, ('2', '1'): b['2'] + 180.0,
              ('3', '4'): b['3'] + 180.0, ('4', '3'): b['4'] + 180.0}
    out = {}
    for r1 in (('1', '2'), ('2', '1')):
        for r2 in (('3', '4'), ('4', '3')):
            d = abs(course[r1] - course[r2]) % 360.0
            out[(r1, r2)] = math.radians(360.0 - d if d > 180.0 else d)
    return out


def _straight_through(js):
    j = next(iter(js.values()))
    j.transitions = {'1': {'2': 1.0}, '2': {'1': 1.0}, '3': {'4': 1.0}, '4': {'3': 1.0}}
    j.source = 'user'
    return js


def _kwargs(sd, td, js):
    return dict(traffic_data=td, segment_data=sd, leg_keys=list(sd.keys()),
                pc_crossing=1.3e-4, pc_merging=1.3e-4,
                length_intervals=[{'min': 50, 'max': 150}], junctions=js)


def test_two_leg_chain_has_no_waypoint_encounter():
    """Consecutive legs of one route never cross or merge with themselves (KNOWN_ISSUES 30)."""
    sd = _two_leg_chain()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    assert _crossing_sum(_StubRunner(), **_kwargs(sd, td, None)) == 0.0


def test_x_crossing_counts_each_pair_of_routes_once():
    """Straight through an X: the four crossing route pairs, each once.

    The leg-pair method took the four leg pairs x four direction pairs,
    i.e. the same encounters 16 times.
    """
    import math
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    totals = _StubRunner()._calc_crossing_collisions(**_kwargs(sd, td, _straight_through(build_junctions(sd))))
    angles = _x_route_angles(sd)
    assert all(abs(math.degrees(t) - 90.0) < 0.5 for t in angles.values())
    assert totals['merging'] == 0.0
    assert totals['crossing'] == pytest.approx(sum(_single_pair(t) for t in angles.values()), rel=1e-9)
    assert totals['crossing'] == pytest.approx(4 * _single_pair(math.pi / 2), rel=1e-4)


def test_route_share_scales_its_pairs_linearly():
    import math
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    js = _straight_through(build_junctions(sd))
    next(iter(js.values())).transitions['1'] = {'2': 0.5}
    totals = _StubRunner()._calc_crossing_collisions(**_kwargs(sd, td, js))
    # Route 1 -> 2 is in two of the four crossing pairs.
    expected = sum((0.5 if r1 == ('1', '2') else 1.0) * _single_pair(t) for (r1, _), t in _x_route_angles(sd).items())
    assert totals['crossing'] == pytest.approx(expected, rel=1e-9)
    assert totals['crossing'] == pytest.approx(3 * _single_pair(math.pi / 2), rel=1e-4)


def test_routes_into_one_leg_merge():
    """Two routes leaving on the same leg merge, at the angle between their arrival courses."""
    import math
    sd = _y_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    js = build_junctions(sd)
    j = next(iter(js.values()))
    j.transitions = {'2': {'1': 1.0}, '3': {'1': 1.0}, '1': {}}
    j.source = 'user'
    totals = _StubRunner()._calc_crossing_collisions(**_kwargs(sd, td, js))
    b2 = ShipCollisionModelMixin._calc_bearing((15.0, 55.0), (15.5, 54.5))
    b3 = ShipCollisionModelMixin._calc_bearing((15.0, 55.0), (14.5, 54.5))
    theta = abs(b2 - b3) % 360.0
    theta = math.radians(360.0 - theta if theta > 180.0 else theta)
    assert totals['crossing'] == 0.0
    assert totals['merging'] == pytest.approx(_single_pair(theta), rel=1e-9)


def test_waypoint_without_stored_matrix_gets_the_geometric_default():
    sd = _x_crossing_segments()
    td = {k: _stub_traffic(freq=1000.0) for k in sd}
    without = _StubRunner()._calc_crossing_collisions(**_kwargs(sd, td, None))
    js = build_junctions(sd)
    apply_geometric_defaults(js, sd)
    with_geometric = _StubRunner()._calc_crossing_collisions(**_kwargs(sd, td, js))
    assert without == pytest.approx(with_geometric, rel=1e-12)
    assert without['crossing'] > 0.0
