"""Standalone tests for ``compute/ray_sampling.py`` and the ray-span ticks in
``geometries/distribution_curves.py``.  Run with ``-p no:qgis --noconftest``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.ray_sampling import (  # noqa: E402
    N_RAYS, RAY_SPAN_SIGMA, describe_ray_sampling, ray_offsets, ray_sampling_settings, span_bounds,
)
from geometries.distribution_curves import (  # noqa: E402
    TICK_FRACTION, curve_points, span_marks, span_tick_profiles, tick_profile,
)


def test_constants_and_offsets():
    assert RAY_SPAN_SIGMA == 6.0 and N_RAYS == 1000
    assert span_bounds(-3526, 1100) == pytest.approx((-10126, 3074))
    offs = ray_offsets(-3526, 1100)
    assert offs.shape == (1000,)
    assert offs[0] == pytest.approx(-10126) and offs[-1] == pytest.approx(3074)
    # Spacing is finer than the old 4 sigma / 500 rays (0.016 sigma).
    assert (offs[1] - offs[0]) / 1100 == pytest.approx(12 / 999)
    assert (offs[1] - offs[0]) / 1100 < 0.016
    # The sampled normal still integrates to ~1 (mass outside 6 sigma is 2e-9).
    from scipy.stats import norm
    assert float((norm.pdf(offs, -3526, 1100) * (offs[1] - offs[0])).sum()) == pytest.approx(1.0, abs=2e-3)
    assert ray_offsets(0.0, 10.0, n_rays=5, span_sigma=1.0).tolist() == pytest.approx([-10, -5, 0, 5, 10])


def test_settings_and_description():
    s = ray_sampling_settings()
    assert s == {'span_sigma': 6.0, 'n_rays': 1000}
    assert describe_ray_sampling() == '+/- 6 sigma, 1000 rays'
    assert describe_ray_sampling({'span_sigma': 4, 'n_rays': 500}) == '+/- 4 sigma, 500 rays'


SEG = {'mean1_1': -400.0, 'std1_1': 150.0, 'weight1_1': 100,
       'mean2_1': 300.0, 'std2_1': 250.0, 'weight2_1': 100}


def test_span_marks_use_the_first_component_like_the_caster():
    assert span_marks(SEG, 0) == pytest.approx((-1300.0, 500.0))
    assert span_marks(SEG, 1) == pytest.approx((-1200.0, 1800.0))
    assert span_marks(SEG, 0, span_sigma=4) == pytest.approx((-1000.0, 200.0))
    assert span_marks({'mean1_1': 0.0, 'std1_1': 0.0}, 0) is None
    assert span_marks({}, 0) is None
    assert span_marks(None, 0) is None
    # A second component does not move the marks: the caster never reads it.
    seg2 = dict(SEG, mean1_2=-3000.0, std1_2=500.0, weight1_2=50)
    assert span_marks(seg2, 0) == span_marks(SEG, 0)


def test_tick_profiles_inside_and_beyond_the_lane():
    half, h = 1000.0, 125.0
    assert tick_profile(500.0, half, h) == [(500.0, 0.0), (500.0, h)]
    assert tick_profile(-1300.0, half, h) == [(-1000.0, 0.0), (-1300.0, 0.0), (-1300.0, h)]
    assert tick_profile(1800.0, half, h) == [(1000.0, 0.0), (1800.0, 0.0), (1800.0, h)]
    prof = span_tick_profiles(SEG, 2000.0)
    assert set(prof) == {0, 1}
    assert prof[0][0][0] == (-1000.0, 0.0)              # low tick of dir 1 starts at the lane end
    assert prof[0][1] == [(500.0, 0.0), (500.0, TICK_FRACTION * 2000.0)]
    assert prof[1][1][0] == (1000.0, 0.0)               # high tick of dir 2 (1800 m) beyond the lane
    assert span_tick_profiles(SEG, 0) == {}
    assert span_tick_profiles({'mean1_1': -400.0, 'std1_1': 150.0}, 2000.0).keys() == {0}


def test_ticks_rise_towards_the_side_the_ships_sail_to():
    # Leg west -> east along x; lateral axis points north (port), ticks for
    # direction 0 rise eastwards (towards End_Point), direction 1 westwards.
    mid, unit = (0.0, 0.0), (1.0, 0.0)
    prof = span_tick_profiles(SEG, 2000.0)
    p0 = curve_points(mid, unit, prof[0][1], 0)          # dir 0, high tick at +500 m (port = north)
    assert p0[0] == pytest.approx((0.0, 500.0)) and p0[1][0] > 0 and p0[1][1] == pytest.approx(500.0)
    p1 = curve_points(mid, unit, prof[1][0], 1)          # dir 1, low tick at -1200 m (starboard = south)
    assert p1[0] == pytest.approx((0.0, -1000.0))        # starts at the south end of the tangent line
    assert p1[1] == pytest.approx((0.0, -1200.0))
    assert p1[2][0] < 0                                  # rises westwards
    assert np.isclose(p1[2][1], -1200.0)
