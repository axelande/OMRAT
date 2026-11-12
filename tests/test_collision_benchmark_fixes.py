"""Ship-ship collision fixes from the Kattegat benchmark (KNOWN_ISSUES 19, 28, 29).

* C: head-on and overtaking integrate the lateral mixture component by
  component (``lateral_overlap_probability``) instead of one collapsed normal.
* Length index: ship lengths come from the length class (matrix column),
  not the ship type (row).
* E2: the passenger ship / fast ferry reduction factor.

Standalone: ``-p no:qgis --noconftest``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import passenger_reduction  # noqa: E402
from compute.basic_equations import (  # noqa: E402
    get_head_on_collision_candidates, get_overtaking_collision_candidates, head_on_candidates_from_overlap,
    lateral_overlap_probability, overtaking_candidates_from_overlap,
)
from compute.ship_collision_model import ShipCollisionModelMixin, collision_type_scale  # noqa: E402

TWO_LANE_LEG = {  # a two-lane leg with a minority component in each direction
    'mean1_1': -300.0, 'std1_1': 350.0, 'weight1_1': 80.0,
    'mean1_2': -2100.0, 'std1_2': 350.0, 'weight1_2': 20.0,
    'mean2_1': 1100.0, 'std2_1': 450.0, 'weight2_1': 80.0,
    'mean2_2': -1200.0, 'std2_2': 300.0, 'weight2_2': 20.0,
}


def _numeric_overlap(pdf1, pdf2, half_width, lo=-8000.0, hi=8000.0, n=16001):
    """P(|X - Y| <= B) by brute-force quadrature."""
    x = np.linspace(lo, hi, n)
    dx = x[1] - x[0]
    f1, f2 = pdf1(x), pdf2(x)
    cdf2 = np.cumsum(f2) * dx
    upper = np.interp(x + half_width, x, cdf2)
    lower = np.interp(x - half_width, x, cdf2)
    return float(np.sum(f1 * (upper - lower)) * dx)


# ---------------------------------------------------------------------------
# lateral_overlap_probability
# ---------------------------------------------------------------------------

class TestLateralOverlap:
    def test_single_normals_equal_hansen_p_g(self):
        p = lateral_overlap_probability([('n', 100.0, 200.0, 1.0)], [('n', -50.0, 300.0, 1.0)], 15.0)
        s = math.hypot(200.0, 300.0)
        assert p == pytest.approx(norm.cdf((150 + 15) / s) - norm.cdf((150 - 15) / s), rel=1e-12)

    def test_normal_against_uniform_matches_quadrature(self):
        comps1, comps2 = [('n', 300.0, 250.0, 1.0)], [('u', -400.0, 900.0, 1.0)]
        exact = _numeric_overlap(lambda x: norm.pdf(x, 300, 250),
                                 lambda x: np.where((x >= -400) & (x <= 900), 1 / 1300, 0.0), 20.0)
        assert lateral_overlap_probability(comps1, comps2, 20.0) == pytest.approx(exact, rel=2e-3)
        assert lateral_overlap_probability(comps2, comps1, 20.0) == pytest.approx(exact, rel=2e-3)

    def test_uniform_against_uniform_is_exact(self):
        comps1, comps2 = [('u', -500.0, 500.0, 1.0)], [('u', 200.0, 1500.0, 1.0)]
        # Covered length ramps 0 -> 50 m for x in [175, 225], then 50 m to 500:
        # (1250 + 13750) / (1000 * 1300).
        assert lateral_overlap_probability(comps1, comps2, 25.0) == pytest.approx(15000 / 1.3e6, rel=1e-12)

    def test_two_lane_mixture_is_3_9_times_below_the_collapsed_normal(self):
        from geometries.distribution_curves import mixture

        def collapsed(comps):
            # the model before 2026-11-09: one normal, mixture mean and total variance
            mu = sum(w * m for _k, m, _s, w in comps)
            var = sum(w * (s * s + m * m) for _k, m, s, w in comps) - mu * mu
            return mu, math.sqrt(var)

        mix = lateral_overlap_probability(mixture(TWO_LANE_LEG, 0), mixture(TWO_LANE_LEG, 1), 10.0)
        mu1, s1 = collapsed(mixture(TWO_LANE_LEG, 0))
        mu2, s2 = collapsed(mixture(TWO_LANE_LEG, 1))
        s = math.hypot(s1, s2)
        collapsed = norm.cdf((mu1 - mu2 + 10) / s) - norm.cdf((mu1 - mu2 - 10) / s)
        assert collapsed / mix == pytest.approx(3.9, abs=0.05)

    def test_candidates_from_overlap_match_the_old_formulas_for_one_normal(self):
        old = get_head_on_collision_candidates(1000, 800, 6.0, 7.0, 100.0, -200.0, 250.0, 300.0, 20.0, 30.0, 5000.0)
        s = math.hypot(250.0, 300.0)
        p_g = norm.cdf((300 + 25) / s) - norm.cdf((300 - 25) / s)
        assert head_on_candidates_from_overlap(1000, 800, 6.0, 7.0, p_g, 5000.0) == pytest.approx(old, rel=1e-12)
        old_ot = get_overtaking_collision_candidates(1000, 800, 8.0, 5.0, 0.0, 0.0, 300.0, 300.0, 20.0, 30.0, 5000.0)
        p_ot = norm.cdf(25 / math.hypot(300, 300)) - norm.cdf(-25 / math.hypot(300, 300))
        assert overtaking_candidates_from_overlap(1000, 800, 8.0, 5.0, p_ot, 5000.0) == pytest.approx(old_ot, rel=1e-12)
        assert overtaking_candidates_from_overlap(1000, 800, 5.0, 8.0, p_ot, 5000.0) == 0.0


# ---------------------------------------------------------------------------
# Head-on / overtaking on a leg use the mixture
# ---------------------------------------------------------------------------

def _block(n_types=21, n_len=15, cells=((18, 5, 1000.0),), speed=12.0, beam=20.0):
    freq = [[0.0] * n_len for _ in range(n_types)]
    for row, col, q in cells:
        freq[row][col] = q
    return {'Frequency (ships/year)': freq, 'Speed (knots)': [[speed] * n_len for _ in range(n_types)],
            'Ship Beam (meters)': [[beam] * n_len for _ in range(n_types)]}


class TestLegModels:
    def test_head_on_uses_the_mixture(self):
        from geometries.distribution_curves import mixture
        m = ShipCollisionModelMixin()
        leg = {'East going': _block(), 'West going': _block()}
        got = m._calc_head_on_collisions(leg, TWO_LANE_LEG, 10_000.0, 1.0, [])
        v = 12.0 * 1852 / 3600
        p_g = lateral_overlap_probability(mixture(TWO_LANE_LEG, 0), mixture(TWO_LANE_LEG, 1), 20.0)
        assert got == pytest.approx(head_on_candidates_from_overlap(1000, 1000, v, v, p_g, 10_000.0), rel=1e-9)

    def test_no_weights_no_head_on(self):
        m = ShipCollisionModelMixin()
        leg = {'East going': _block(), 'West going': _block()}
        assert m._calc_head_on_collisions(leg, {'mean1_1': 0.0, 'std1_1': 100.0}, 10_000.0, 1.0, []) == 0.0


# ---------------------------------------------------------------------------
# Ship length = length class (column), not ship type (row)
# ---------------------------------------------------------------------------

INTERVALS = [{'min': float(lo), 'max': float(lo + 25)} for lo in range(0, 350, 25)] + [{'min': 350.0, 'max': 400.0}]


class TestLengthIndex:
    def test_missing_beam_is_estimated_from_the_length_class(self):
        m = ShipCollisionModelMixin()
        pleasure = 6                                        # row 6 of the AIS taxonomy, 0-25 m column
        freq = np.zeros((21, 15))
        freq[pleasure][0] = 100.0
        cells = m._collect_ship_cells(freq, np.full((21, 15), 8.0), np.zeros((21, 15)), INTERVALS)
        (_row, _col, _q, _v, beam), = cells
        assert beam == pytest.approx(12.5 / 6.5)            # was 162.5 / 6.5 (row 6 read as 150-175 m)

    def test_crossing_uses_the_length_class(self):
        m = ShipCollisionModelMixin()
        f1 = np.zeros((21, 15))
        f1[6][0] = 100.0                                   # pleasure craft, 0-25 m
        speed, beam = np.full((21, 15), 8.0), np.full((21, 15), 4.0)
        small = m._crossing_ship_cell_loop(f1, speed, beam, f1, speed, beam, math.pi / 2, 90.0, 1.0, 1.0,
                                           INTERVALS, (0.0, 0.0), 'crossing', None, None, None, None, 'a', 'b')
        f2 = np.zeros((21, 15))
        f2[6][6] = 100.0                                   # same ship type, 150-175 m
        big = m._crossing_ship_cell_loop(f2, speed, beam, f2, speed, beam, math.pi / 2, 90.0, 1.0, 1.0,
                                         INTERVALS, (0.0, 0.0), 'crossing', None, None, None, None, 'a', 'b')
        assert big > 3.0 * small                           # identical before the fix


# ---------------------------------------------------------------------------
# E2  passenger ship / fast ferry reduction
# ---------------------------------------------------------------------------

TYPES = ['Fishing', 'Towing', 'Dredging or underwater ops', 'Diving ops', 'Military ops', 'Sailing',
         'Pleasure Craft', 'High speed craft (HSC)', 'Pilot Vessel', 'Search and Rescue vessel', 'Tug',
         'Port Tender', 'Anti-pollution equipment', 'Law Enforcement', 'Spare', 'Medical Transport',
         'Noncombatant ship according to RR Resolution No. 18', 'Passenger, all ships of this type',
         'Cargo, all ships of this type', 'Tanker, all ships of this type', 'Other Type, all ships of this type']
PASSENGER, CARGO, HSC = 17, 18, 7
NO_REDUCTION = {passenger_reduction.PASSENGER_KEY: 1.0, passenger_reduction.FAST_FERRY_KEY: 1.0}


class TestPassengerReduction:
    def test_type_classes_and_factors(self):
        pc = {passenger_reduction.PASSENGER_KEY: 20.0, passenger_reduction.FAST_FERRY_KEY: 10.0}
        f = passenger_reduction.type_factors(TYPES, pc)
        assert f[PASSENGER] == 20.0 and f[HSC] == 10.0 and f[CARGO] == 1.0
        assert passenger_reduction.ship_type_class('Ro-pax') == 'passenger'
        assert passenger_reduction.ship_type_class('Ro-ro') is None
        assert passenger_reduction.collision_scales(TYPES, pc)[PASSENGER] == pytest.approx(1 / math.sqrt(20))
        assert passenger_reduction.powered_scales(TYPES, pc)[HSC] == pytest.approx(0.1)
        key = passenger_reduction.PASSENGER_KEY
        assert passenger_reduction.factor({key: 0.5}, key) == 1.0          # below 1 = no reduction
        assert not passenger_reduction.is_active(NO_REDUCTION)

    def test_default_is_iwrap_20(self):
        """A project without the keys gets IWRAP's default 20 for both (user's decision 2026-11-09)."""
        assert passenger_reduction.DEFAULTS == {passenger_reduction.PASSENGER_KEY: 20.0,
                                                passenger_reduction.FAST_FERRY_KEY: 20.0}
        assert passenger_reduction.factor({}, passenger_reduction.PASSENGER_KEY) == 20.0
        assert passenger_reduction.factor({}, passenger_reduction.FAST_FERRY_KEY) == 20.0
        assert passenger_reduction.is_active({})
        f = passenger_reduction.type_factors(TYPES, {})
        assert f[PASSENGER] == 20.0 and f[HSC] == 20.0 and f[CARGO] == 1.0

    def _head_on(self, pc, row1, row2):
        m = ShipCollisionModelMixin()
        leg = {'E': _block(cells=((row1, 5, 1000.0),)), 'W': _block(cells=((row2, 5, 1000.0),))}
        data = {'ship_categories': {'types': TYPES}, 'traffic_data': {'1': leg}}
        m._collision_type_scale = collision_type_scale(data, pc)
        seg = {'mean1_1': 0.0, 'std1_1': 300.0, 'weight1_1': 100.0,
               'mean2_1': 0.0, 'std2_1': 300.0, 'weight2_1': 100.0}
        return m._calc_head_on_collisions(leg, seg, 10_000.0, 1.0, [])

    def test_collision_causation_reduced_per_ship(self):
        pc = {passenger_reduction.PASSENGER_KEY: 20.0}
        base = self._head_on(NO_REDUCTION, CARGO, CARGO)
        assert self._head_on(pc, CARGO, CARGO) == pytest.approx(base)
        assert self._head_on(pc, PASSENGER, CARGO) == pytest.approx(base / math.sqrt(20))
        assert self._head_on(pc, PASSENGER, PASSENGER) == pytest.approx(base / 20)
        assert collision_type_scale({'ship_categories': {'types': TYPES}}, NO_REDUCTION) is None
        # no keys = IWRAP's default 20
        assert collision_type_scale({'ship_categories': {'types': TYPES}}, {})[PASSENGER] == pytest.approx(
            1 / math.sqrt(20))

    def test_powered_frequency_divided_by_f(self):
        from compute.powered_model import powered_type_scale
        data = {'pc': {passenger_reduction.PASSENGER_KEY: 20.0}, 'ship_categories': {'types': TYPES},
                'traffic_data': {'1': {'E': _block()}}}
        scale = powered_type_scale(data)
        assert len(scale) == 21 and scale[PASSENGER] == pytest.approx(0.05) and scale[CARGO] == 1.0
        assert powered_type_scale(dict(data, pc=dict(NO_REDUCTION))) is None
        assert powered_type_scale(dict(data, pc={}))[PASSENGER] == pytest.approx(0.05)   # default 20
