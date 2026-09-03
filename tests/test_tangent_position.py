"""Pure tests for ``geometries.tangent_position``.

The movable tangent line stores a fraction ``Tangent_Pos`` along the leg.
Both the canvas drawing code and the PostGIS passage-line builder derive
their anchor point from these helpers, so the maths is pinned here
without QGIS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.tangent_position import (  # noqa: E402
    DEFAULT_TANGENT_POS,
    clamp_fraction,
    fraction_from_percent,
    normalize_tangent_pos,
    percent_from_fraction,
    perpendicular_through_point,
    point_along,
    project_fraction,
)


class TestNormalize:
    def test_default_is_midpoint(self):
        assert DEFAULT_TANGENT_POS == 0.5

    @pytest.mark.parametrize('raw', [None, 'abc', '', float('nan'), float('inf')])
    def test_garbage_falls_back_to_default(self, raw):
        assert normalize_tangent_pos(raw) == 0.5

    @pytest.mark.parametrize('raw, expected', [
        (0.25, 0.25), ('0.25', 0.25), (1, 1.0), (-0.3, 0.0), (7, 1.0),
    ])
    def test_numeric_input_is_clamped(self, raw, expected):
        assert normalize_tangent_pos(raw) == expected

    def test_clamp_fraction_bounds(self):
        assert clamp_fraction(-1) == 0.0
        assert clamp_fraction(2) == 1.0
        assert clamp_fraction(0.4) == 0.4


class TestPercentRoundTrip:
    @pytest.mark.parametrize('t, text', [(0.5, '50'), (0.3, '30'), (1 / 3, '33.3'), (0.0, '0'), (1.0, '100')])
    def test_percent_text(self, t, text):
        assert percent_from_fraction(t) == text

    @pytest.mark.parametrize('text, expected', [
        ('30', 0.3), ('30 %', 0.3), ('  45.5', 0.455), ('33,3', 0.333), (25, 0.25), ('150', 1.0), ('-5', 0.0),
    ])
    def test_parse_percent(self, text, expected):
        assert fraction_from_percent(text) == pytest.approx(expected)

    @pytest.mark.parametrize('text', ['', '   ', 'abc', None, 'nan'])
    def test_unparsable_percent_returns_none(self, text):
        assert fraction_from_percent(text) is None


class TestGeometry:
    def test_point_along(self):
        assert point_along((0.0, 0.0), (10.0, 0.0), 0.5) == (5.0, 0.0)
        assert point_along((0.0, 0.0), (10.0, 20.0), 0.25) == (2.5, 5.0)
        # Clamped.
        assert point_along((0.0, 0.0), (10.0, 0.0), 3.0) == (10.0, 0.0)

    def test_project_fraction_on_axis(self):
        s, e = (0.0, 0.0), (100.0, 0.0)
        assert project_fraction(s, e, (30.0, 999.0)) == pytest.approx(0.3)
        # Lateral offset is ignored; only the along-track component counts.
        assert project_fraction(s, e, (30.0, -5.0)) == pytest.approx(0.3)

    def test_project_fraction_clamps_beyond_ends(self):
        s, e = (0.0, 0.0), (100.0, 0.0)
        assert project_fraction(s, e, (-40.0, 0.0)) == 0.0
        assert project_fraction(s, e, (400.0, 0.0)) == 1.0

    def test_project_fraction_degenerate_leg(self):
        assert project_fraction((1.0, 1.0), (1.0, 1.0), (0.0, 0.0)) is None

    def test_project_fraction_diagonal(self):
        s, e = (0.0, 0.0), (100.0, 100.0)
        # (50, 50) is exactly halfway; (60, 40) projects to the same point.
        assert project_fraction(s, e, (50.0, 50.0)) == pytest.approx(0.5)
        assert project_fraction(s, e, (60.0, 40.0)) == pytest.approx(0.5)

    def test_perpendicular_default_is_midpoint(self):
        ends = perpendicular_through_point((0.0, 0.0), (1000.0, 0.0), 250.0)
        assert ends is not None
        (ax, ay), (bx, by) = ends
        assert ax == pytest.approx(500.0) and bx == pytest.approx(500.0)
        assert sorted((ay, by)) == [-250.0, 250.0]

    def test_perpendicular_at_fraction(self):
        ends = perpendicular_through_point((0.0, 0.0), (1000.0, 0.0), 250.0, 0.2)
        assert ends is not None
        (ax, ay), (bx, by) = ends
        assert ax == pytest.approx(200.0) and bx == pytest.approx(200.0)
        assert sorted((ay, by)) == [-250.0, 250.0]

    def test_perpendicular_is_orthogonal_to_leg(self):
        s, e = (0.0, 0.0), (300.0, 400.0)
        ends = perpendicular_through_point(s, e, 50.0, 0.75)
        assert ends is not None
        (ax, ay), (bx, by) = ends
        dot = (bx - ax) * (e[0] - s[0]) + (by - ay) * (e[1] - s[1])
        assert dot == pytest.approx(0.0, abs=1e-9)
        # Anchored on the leg at 75 %.
        mx, my = (ax + bx) / 2, (ay + by) / 2
        assert (mx, my) == pytest.approx((225.0, 300.0))

    def test_perpendicular_degenerate_leg(self):
        assert perpendicular_through_point((1.0, 1.0), (1.0, 1.0), 10.0) is None

    def test_drag_round_trip(self):
        """Dragging a tangent drawn at 50 % to 20 % (with a lateral wobble)
        and re-projecting recovers 20 %: the snap-back is idempotent."""
        s, e = (0.0, 0.0), (1000.0, 0.0)
        dragged = perpendicular_through_point(s, e, 100.0, 0.2)
        assert dragged is not None
        (ax, ay), (bx, by) = dragged
        wobble_mid = ((ax + bx) / 2, (ay + by) / 2 + 37.0)
        assert project_fraction(s, e, wobble_mid) == pytest.approx(0.2)
