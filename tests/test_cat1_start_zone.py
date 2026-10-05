"""Tests for the Category I start-zone setting.

``compute/cat1_start_zone.py`` (keys, defaults, fraction) and its effect
on the ray caster in ``geometries/get_powered_overlap.py``.  The caster
module imports matplotlib's Qt backend, so run this file with the QGIS
conftest (plain ``pytest tests/test_cat1_start_zone.py``), like
``test_get_powered_overlap.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.cat1_start_zone import (  # noqa: E402
    CAT1_START_ZONE_DEFAULTS, KEY_ENABLED, KEY_PCT, describe_start_zone,
    start_zone_enabled, start_zone_fraction, start_zone_pct, start_zone_settings,
)
from compute.iwrap_defaults import IWRAP_PC_DEFAULTS, default_pc_values  # noqa: E402
from geometries.get_powered_overlap import (  # noqa: E402
    _build_hit_and_edge_matrices, _compute_cat1_in_lane, _compute_cat2_with_shadows,
    _run_all_computations,
)

ORIGIN = np.array([0.0, 0.0])
ALONG = np.array([1.0, 0.0])
PERP = np.array([0.0, 1.0])
L = 100_000.0   # 100 km leg


def _obs(geom, oid='park'):
    return [({'id': oid, 'geom': geom}, 'object')]


def _mass(obstacles, frac=0.0, mean=0.0, sigma=1000.0):
    summaries, _, _, _ = _compute_cat1_in_lane(ORIGIN, ALONG, PERP, mean, sigma, L, obstacles, start_zone_frac=frac)
    return sum(s['mass'] for s in summaries.values())


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

class TestSettings:
    def test_defaults_are_on_ten_percent(self):
        assert CAT1_START_ZONE_DEFAULTS == {KEY_ENABLED: True, KEY_PCT: 10.0}
        assert start_zone_fraction(None) == pytest.approx(0.10)
        assert start_zone_fraction({}) == pytest.approx(0.10)
        assert start_zone_settings({}) == {'enabled': True, 'pct': 10.0}

    def test_default_pc_values_carry_the_setting_but_iwrap_defaults_do_not(self):
        d = default_pc_values()
        assert d[KEY_ENABLED] is True and d[KEY_PCT] == 10.0
        assert KEY_ENABLED not in IWRAP_PC_DEFAULTS
        for k, v in IWRAP_PC_DEFAULTS.items():
            assert d[k] == v

    def test_off_and_custom_pct(self):
        assert start_zone_fraction({KEY_ENABLED: False, KEY_PCT: 25}) == 0.0
        assert start_zone_fraction({KEY_ENABLED: True, KEY_PCT: 25}) == pytest.approx(0.25)
        assert start_zone_enabled({KEY_ENABLED: 'false'}) is False
        assert start_zone_enabled({KEY_ENABLED: 'true'}) is True

    def test_pct_is_clamped_and_robust(self):
        assert start_zone_pct({KEY_PCT: -5}) == 0.0
        assert start_zone_pct({KEY_PCT: 150}) == 90.0
        assert start_zone_pct({KEY_PCT: 'abc'}) == 10.0
        assert start_zone_pct({KEY_PCT: float('nan')}) == 10.0

    def test_describe(self):
        assert 'off' in describe_start_zone({KEY_ENABLED: False})
        assert '12.5 %' in describe_start_zone({KEY_PCT: 12.5})


# ---------------------------------------------------------------------------
# Ray caster
# ---------------------------------------------------------------------------

class TestHitMatrixMinRange:
    def test_min_range_skips_near_crossings(self):
        offsets = np.array([0.0])
        near = _obs(box(2_000, -500, 4_000, 500))          # 2-4 km along the leg
        hit, edge = _build_hit_and_edge_matrices(offsets, near, ORIGIN, ALONG, PERP, max_range=L)
        assert hit[0, 0] == pytest.approx(2_000)
        hit, edge = _build_hit_and_edge_matrices(offsets, near, ORIGIN, ALONG, PERP, max_range=L, min_range=5_000)
        assert np.isinf(hit[0, 0]) and edge[0, 0] == -1
        # Straddling obstacle: the first crossing beyond min_range is its far edge.
        strad = _obs(box(2_000, -500, 8_000, 500))
        hit, edge = _build_hit_and_edge_matrices(offsets, strad, ORIGIN, ALONG, PERP, max_range=L, min_range=5_000)
        assert hit[0, 0] == pytest.approx(8_000)


class TestCat1StartZone:
    def test_obstacle_inside_start_zone_is_dropped(self):
        park = box(3_000, -3_000, 8_000, 3_000)   # 3-8 km of a 100 km leg: inside the first 10 %
        assert _mass(_obs(park)) > 0.99                      # off: every ray meets it
        assert _mass(_obs(park), frac=0.10) == 0.0           # on: nothing
        assert _mass(_obs(park), frac=0.05) > 0.99           # zone ends at 5 km: rays leave the box at 8 km -> kept

    def test_zone_boundary_is_measured_from_the_origin(self):
        park = box(6_000, -3_000, 8_000, 3_000)
        assert _mass(_obs(park), frac=0.05) > 0.99           # zone ends at 5 km, park starts at 6 km
        assert _mass(_obs(park), frac=0.07) > 0.99           # zone ends at 7 km: exit edge at 8 km still counts
        assert _mass(_obs(park), frac=0.09) == 0.0           # zone ends at 9 km: both edges inside -> dropped

    def test_straddling_obstacle_keeps_rays_crossing_beyond_the_zone(self):
        # 8-15 km along, 10 % zone ends at 10 km: rays are inside the box at
        # 10 km and are credited where they leave it (15 km), so the mass stays.
        park = box(8_000, -3_000, 15_000, 3_000)
        assert _mass(_obs(park), frac=0.10) > 0.99
        # Lateral partial: only part of the distribution is inside the box at all.
        half = box(8_000, 0.0, 15_000, 3_000)
        m = _mass(_obs(half), frac=0.10)
        assert 0.45 < m < 0.55

    def test_obstacle_beyond_the_zone_is_unchanged(self):
        park = box(20_000, -3_000, 30_000, 3_000)
        assert _mass(_obs(park)) == pytest.approx(_mass(_obs(park), frac=0.10))

    def test_zero_fraction_is_identity_and_fraction_is_clamped(self):
        park = box(3_000, -3_000, 8_000, 3_000)
        assert _mass(_obs(park), frac=0.0) == pytest.approx(_mass(_obs(park)))
        assert _mass(_obs(park), frac=5.0) == 0.0            # clamped to 1.0: whole leg skipped

    def test_cat2_is_untouched(self):
        park = box(3_000, -3_000, 8_000, 3_000)
        a = _compute_cat2_with_shadows(ORIGIN, ALONG, PERP, 0.0, 1000.0, 180.0, 5.0, _obs(park))[0]
        b = _compute_cat2_with_shadows(ORIGIN, ALONG, PERP, 0.0, 1000.0, 180.0, 5.0, _obs(park))[0]
        assert a == b and a[('object', 'park')]['mass'] > 0.99


class TestRunAllComputations:
    @staticmethod
    def _legs():
        return {
            '1': {
                'start': np.array([0.0, 0.0]), 'end': np.array([L, 0.0]),
                'dirs': [
                    {'mean': 0.0, 'std': 1000.0, 'ai': 180.0, 'speed_ms': 5.0},
                    {'mean': 0.0, 'std': 1000.0, 'ai': 180.0, 'speed_ms': 5.0},
                ],
            }
        }

    def test_only_the_direction_leaving_the_obstacle_is_filtered(self):
        park = box(3_000, -3_000, 8_000, 3_000)   # next to the start waypoint
        off = {(c['dir_idx']): sum(s['mass'] for s in c['cat1']['summaries'].values())
               for c in _run_all_computations(self._legs(), _obs(park))}
        on = {(c['dir_idx']): sum(s['mass'] for s in c['cat1']['summaries'].values())
              for c in _run_all_computations(self._legs(), _obs(park), cat1_start_zone_frac=0.10)}
        assert off[0] > 0.99 and off[1] > 0.99
        assert 0 not in on                       # dir 0 (origin = start): dropped entirely
        assert on[1] > 0.99                      # dir 1 (origin = end): the park sits at 92-97 % -> kept
        comp = [c for c in _run_all_computations(self._legs(), _obs(park), cat1_start_zone_frac=0.10)][0]
        assert comp['cat1']['start_zone_m'] == pytest.approx(0.10 * L)
