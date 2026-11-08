"""Powered model with missing ship values (KNOWN_ISSUES items 4 and 9).

Needs the QGIS conftest: ``compute.powered_model`` imports the ray caster,
which imports matplotlib's Qt backend.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compute.powered_model as pm  # noqa: E402


class TestExtractPositive:
    @pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan, 0, -3, '', 'inf'])
    def test_missing_values_give_the_default(self, value):
        assert pm._extract_positive([[value]], 0, 0, 5.0) == 5.0

    @pytest.mark.parametrize('value, expected', [(7.2, 7.2), ('7.2', 7.2), (3, 3.0)])
    def test_valid_values_pass(self, value, expected):
        assert pm._extract_positive([[value]], 0, 0, 5.0) == expected


@pytest.mark.parametrize('height, counted', [(0.0, True), (math.inf, True), (math.nan, True),
                                             (25.0, True), (5.0, False)])
def test_clearance_check_needs_a_known_ship_height(monkeypatch, height, counted):
    # One leg, one cell, one hit on object '1' of height 10 m.
    monkeypatch.setattr(pm, '_iter_hit_probs', lambda comp, decay: iter([('cat1', ('object', '1'), 0.5, None)]))
    comp = {'seg_id': '1', 'dir_idx': 0, 'dir_info': {'ai': 180.0}}
    traffic = {'1': {'East going': {
        'Frequency (ships/year)': [[100.0]],
        'Speed (knots)': [[10.0]],
        'Ship heights (meters)': [[height]],
    }}}
    acc = pm.PoweredModelMixin._sum_allision_contribs(None, [comp], traffic, {'1': 10.0}, 1e-4, 1e-4)
    assert (acc.total > 0) is counted


def test_known_height():
    assert pm.known_height(12.0)
    assert not pm.known_height(0.0)
    assert not pm.known_height(math.inf)
