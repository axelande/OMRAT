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
    assert s == {'span_sigma': 6.0, 'n_rays': 1000, 'max_rays': 3000, 'mixture': True}
    assert describe_ray_sampling() == ('whole lateral distribution, +/- 6 sigma of every component, '
                                       '1000-3000 rays')
    # Reports written before 2026-11-09 (first component only) keep their wording.
    assert describe_ray_sampling({'span_sigma': 4, 'n_rays': 500}) == '+/- 4 sigma, 500 rays'


SEG = {'mean1_1': -400.0, 'std1_1': 150.0, 'weight1_1': 100,
       'mean2_1': 300.0, 'std2_1': 250.0, 'weight2_1': 100}


def test_span_marks_cover_the_whole_mixture_like_the_caster():
    assert span_marks(SEG, 0) == pytest.approx((-1300.0, 500.0))
    assert span_marks(SEG, 1) == pytest.approx((-1200.0, 1800.0))
    assert span_marks(SEG, 0, span_sigma=4) == pytest.approx((-1000.0, 200.0))
    assert span_marks({'mean1_1': 0.0, 'std1_1': 0.0}, 0) is None
    assert span_marks({}, 0) is None
    assert span_marks(None, 0) is None
    # A second component widens the marks: the caster casts it too (KNOWN_ISSUES 25).
    seg2 = dict(SEG, mean1_2=-3000.0, std1_2=500.0, weight1_2=50)
    assert span_marks(seg2, 0) == pytest.approx((-6000.0, 500.0))
    # So does the uniform part.
    seg3 = dict(SEG, u_min1=-200.0, u_max1=2500.0, u_p1=5)
    assert span_marks(seg3, 0) == pytest.approx((-1300.0, 2500.0))
    # No weights: the first component, as the caster falls back to it.
    assert span_marks({'mean1_1': -400.0, 'std1_1': 150.0}, 0) == pytest.approx((-1300.0, 500.0))


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


# ---------------------------------------------------------------------------
# mixture_rays: the powered model's rays over the whole lateral distribution
# ---------------------------------------------------------------------------

def test_one_component_is_sampled_exactly_as_before():
    from scipy.stats import norm
    from compute.ray_sampling import mixture_rays
    offs, pdf, masses = mixture_rays([('n', -3526.0, 1100.0, 1.0)])
    old = ray_offsets(-3526, 1100)
    assert offs.shape == old.shape and np.allclose(offs, old)
    assert np.allclose(masses, norm.pdf(old, -3526, 1100) * (old[1] - old[0]), rtol=1e-12)


def test_mixture_rays_carry_every_component_with_its_weight():
    from compute.ray_sampling import MAX_RAYS, mixture_rays
    comps = [('n', -300.0, 350.0, 0.8), ('n', -2100.0, 350.0, 0.2)]     # LEG_1_6 direction 1
    offs, pdf, masses = mixture_rays(comps)
    assert offs[0] == pytest.approx(-2100.0 - 6 * 350.0) and offs[-1] == pytest.approx(-300.0 + 6 * 350.0)
    spacing = offs[1] - offs[0]
    assert spacing <= 12 * 350.0 / 999 + 1e-9 and len(offs) <= MAX_RAYS     # same density as one component
    assert masses.sum() == pytest.approx(1.0, abs=1e-6)
    from scipy.stats import norm
    lo, hi = -2100.0 - 3 * 350.0, -2100.0 + 3 * 350.0
    second_lane = (offs > lo) & (offs < hi)
    expected = (0.2 * (norm.cdf(hi, -2100, 350) - norm.cdf(lo, -2100, 350))
                + 0.8 * (norm.cdf(hi, -300, 350) - norm.cdf(lo, -300, 350)))
    assert masses[second_lane].sum() == pytest.approx(expected, abs=1e-3)   # ~0.013 with component 1 only


def test_mixture_rays_with_a_uniform_part_and_a_narrow_component():
    from compute.ray_sampling import MAX_RAYS, mixture_rays
    comps = [('n', 0.0, 20.0, 0.9), ('u', -5000.0, 5000.0, 0.1)]
    offs, pdf, masses = mixture_rays(comps)
    assert len(offs) == MAX_RAYS                     # the narrow peak asks for more rays than allowed
    assert offs[0] == pytest.approx(-5000.0) and offs[-1] == pytest.approx(5000.0)
    assert masses.sum() == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        mixture_rays([('n', 0.0, 0.0, 1.0)])
