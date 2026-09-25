"""Pure tests for ``geometries.distribution_curves`` (IWRAP-style lateral
distribution curves along the tangent line)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.distribution_curves import (  # noqa: E402
    HEIGHT_FRACTION, curve_points, curve_profiles, mixture, pdf, signature,
)


def _seg(**kw):
    """A leg with one normal per direction: dir 1 starboard, dir 2 port."""
    seg = {
        'mean1_1': -1000.0, 'std1_1': 400.0, 'weight1_1': 100,
        'mean2_1': 800.0, 'std2_1': 800.0, 'weight2_1': 100,
        'u_p1': 0, 'u_min1': 0, 'u_max1': 0, 'u_p2': 0, 'u_min2': 0, 'u_max2': 0,
    }
    seg.update(kw)
    return seg


class TestMixture:
    def test_weights_are_normalised_and_empty_components_skipped(self):
        seg = _seg(weight1_2=300, mean1_2=0.0, std1_2=0.0,        # std 0: skipped
                   weight1_3=100, mean1_3=500.0, std1_3=100.0, u_p1=100, u_min1=-50.0, u_max1=50.0)
        comps = mixture(seg, 0)
        assert [c[0] for c in comps] == ['n', 'n', 'u']
        assert sum(c[3] for c in comps) == pytest.approx(1.0)
        assert [c[3] for c in comps] == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_no_distribution(self):
        assert mixture({'weight1_1': 0, 'std1_1': 100, 'mean1_1': 0}, 0) == []
        assert curve_profiles({}, 5000) == {}

    def test_density_integrates_to_one(self):
        comps = mixture(_seg(u_p1=50, u_min1=-200.0, u_max1=300.0), 0)
        step = 5.0
        area = sum(pdf(comps, -6000 + (i + 0.5) * step) * step for i in range(int(12000 / step)))   # midpoint rule
        assert area == pytest.approx(1.0, abs=1e-3)


class TestProfiles:
    def test_shared_scale_and_ends_on_the_tangent(self):
        prof = curve_profiles(_seg(), 6000)
        assert set(prof) == {0, 1}
        top = max(h for pts in prof.values() for _x, h in pts)
        assert top == pytest.approx(HEIGHT_FRACTION * 6000)           # taller peak = 1/4 width
        # Direction 1 is narrower (std 400 vs 800) so it has the tall peak.
        assert max(h for _x, h in prof[0]) == pytest.approx(top)
        assert max(h for _x, h in prof[1]) == pytest.approx(top / 2, rel=1e-3)
        for pts in prof.values():
            assert pts[0] == (-3000, 0.0) and pts[-1] == (3000, 0.0)
            assert all(-3000 <= x <= 3000 for x, _h in pts)

    def test_narrow_peak_on_a_wide_leg_keeps_its_height(self):
        prof = curve_profiles(_seg(std1_1=100.0, mean2_1=0.0, weight2_1=0), 20000)
        assert set(prof) == {0}
        x_peak, h_peak = max(prof[0], key=lambda p: p[1])
        assert x_peak == pytest.approx(-1000.0)
        assert h_peak == pytest.approx(HEIGHT_FRACTION * 20000)

    def test_signature_changes_with_the_distribution(self):
        a = signature(_seg(), 5000, 0.5, (0, 0, 1, 1))
        assert a == signature(_seg(), 5000, 0.5, (0, 0, 1, 1))
        assert a != signature(_seg(mean1_1=-900.0), 5000, 0.5, (0, 0, 1, 1))
        assert a != signature(_seg(), 6000, 0.5, (0, 0, 1, 1))


class TestPoints:
    """A leg drawn eastwards: unit (1, 0), port (x > 0) = north."""

    def test_offsets_use_the_starboard_negative_axis(self):
        pts = curve_points((0.0, 0.0), (1.0, 0.0), [(-1000.0, 0.0), (500.0, 0.0)], 0)
        assert pts[0] == pytest.approx((0.0, -1000.0))      # starboard of east-going = south
        assert pts[1] == pytest.approx((0.0, 500.0))

    def test_each_direction_bulges_towards_where_it_sails(self):
        prof = [(0.0, 200.0)]
        (x0, _y0), = curve_points((0.0, 0.0), (1.0, 0.0), prof, 0)
        (x1, _y1), = curve_points((0.0, 0.0), (1.0, 0.0), prof, 1)
        assert x0 == pytest.approx(200.0)     # direction 1 sails to End_Point (east)
        assert x1 == pytest.approx(-200.0)    # direction 2 sails to Start_Point (west)
