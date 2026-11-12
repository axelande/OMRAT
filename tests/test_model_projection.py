"""Bearings measured in the model projection (KNOWN_ISSUES 36, 2026-11-12).

Waypoint angles used to be ``atan2(dlon, dlat)`` on raw degrees.  At 57 N a
degree of longitude is 0.54 of a degree of latitude, so turns came out
up to 30 degrees wrong.  Runs standalone.
"""
import math

import pytest

from geometries.junctions import Junction, _leg_outward_bearing, compute_geometric_transition_matrix
from geometries.model_projection import (
    ModelProjector, model_projector, model_projector_or_none, projected_bearing,
)


def _gc(p, q) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*p, *q))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.degrees(math.atan2(y, x)) % 360.0


def _seg(start, end):
    return {'Start_Point': f"{start[0]} {start[1]}", 'End_Point': f"{end[0]} {end[1]}"}


@pytest.mark.parametrize("end, expected", [((0.0, 1.0), 0.0), ((1.0, 0.0), 90.0),
                                           ((0.0, -1.0), 180.0), ((-1.0, 0.0), 270.0)])
def test_cardinal_directions(end, expected):
    assert projected_bearing((0.0, 0.0), end) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("end", [(11.3, 57.2), (10.6, 56.8), (11.5, 56.9), (10.9, 57.4)])
def test_true_bearing_at_high_latitude(end):
    """Centred on the start point the bearing is the true one (great circle within 0.1 degree)."""
    start = (11.0, 57.0)
    assert projected_bearing(start, end) == pytest.approx(_gc(start, end), abs=0.1)


def test_raw_degrees_were_wrong_at_57_north():
    start, end = (11.0, 57.0), (11.5, 56.5)
    raw = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
    assert raw == pytest.approx(135.0)
    assert projected_bearing(start, end) == pytest.approx(_gc(start, end), abs=0.1)
    assert projected_bearing(start, end) == pytest.approx(151.0, abs=0.2)


def test_angle_between_bearings_does_not_depend_on_the_centre():
    """Conformal: grid north moves with the centre, the angle at a point does not (within 0.05 degree)."""
    wp, a, b = (11.0, 57.0), (10.6, 56.8), (11.4, 57.3)
    angles = []
    for centre in ((11.0, 57.0), (12.0, 56.5), (10.0, 57.5)):
        proj = ModelProjector(*centre)
        d = abs(projected_bearing(wp, a, proj) - projected_bearing(wp, b, proj)) % 360.0
        angles.append(min(d, 360.0 - d))
    assert max(angles) - min(angles) < 0.05


def test_model_projector_or_none():
    assert model_projector_or_none({}) is None
    proj = model_projector_or_none({'1': _seg((10.0, 56.0), (12.0, 58.0))})
    assert (proj.lon_ref, proj.lat_ref) == (11.0, 57.0)
    with pytest.raises(ValueError):
        model_projector({'1': {'Start_Point': 'x'}})


def test_junction_default_bearings_are_true_bearings():
    """The deflection heuristic behind the geometric default shares measures in the projection too."""
    wp = (11.0, 57.0)
    sd = {'1': _seg((10.7, 56.8), wp), '2': _seg(wp, (11.4, 57.1)), '3': _seg(wp, (11.1, 57.3))}
    j = Junction(junction_id='j', point=wp, legs={'1': 'end', '2': 'start', '3': 'start'})
    for leg, far in (('1', (10.7, 56.8)), ('2', (11.4, 57.1)), ('3', (11.1, 57.3))):
        assert _leg_outward_bearing(j, leg, sd, model_projector(sd)) == pytest.approx(_gc(wp, far), abs=0.15)
    matrix = compute_geometric_transition_matrix(j, sd)
    assert sum(matrix['1'].values()) == pytest.approx(1.0)


def _turn_at(wp, a, far) -> float:
    """True turn at ``wp`` for a ship arriving from ``a`` and leaving towards ``far`` (great circle)."""
    d = abs((_gc(wp, a) + 180.0) - _gc(wp, far)) % 360.0
    return min(d, 360.0 - d)


def test_iwrap_import_bend_angle_is_the_true_turn():
    from compute.iwrap_convertion import _compute_bend_angles
    wp, a, far = (11.0, 57.0), (10.7, 56.85), (11.3, 57.05)
    sd = {'1': dict(_seg(a, wp), Leg_name='LEG_1_1'), '2': dict(_seg(wp, far), Leg_name='LEG_1_2')}
    _compute_bend_angles(sd, debug=False)
    raw = abs(math.degrees(math.atan2(far[0] - wp[0], far[1] - wp[1]) - math.atan2(wp[0] - a[0], wp[1] - a[1])))
    assert sd['1']['bend_angle'] == pytest.approx(_turn_at(wp, a, far), abs=0.15)
    assert abs(sd['1']['bend_angle'] - raw) > 5.0
