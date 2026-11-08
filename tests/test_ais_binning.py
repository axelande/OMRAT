"""Length binning of AIS pings (``omrat_utils/ais_binning.py``) -- standalone.

KNOWN_ISSUES item 1: the AIS update sized its matrices from the Traffic
table (Total row/column included) and binned on hard-coded 25 m steps, so
ships longer than the last class landed in the Total column.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.ais_binning import (  # noqa: E402
    UNKNOWN_LOA_M,
    interval_bounds,
    length_bin,
)

STEPS_25 = [{'min': i * 25.0, 'max': i * 25.0 + 25.0, 'label': ''} for i in range(14)]


class TestIntervalBounds:
    def test_numeric(self):
        assert interval_bounds(STEPS_25)[:2] == [(0.0, 25.0), (25.0, 50.0)]

    def test_strings_from_the_table_and_open_top(self):
        bounds = interval_bounds([{'min': '0', 'max': '50'}, {'min': '50', 'max': '1000+'}])
        assert bounds[0] == (0.0, 50.0)
        assert bounds[1][0] == 50.0 and math.isinf(bounds[1][1])

    def test_empty(self):
        assert interval_bounds(None) == []


class TestLengthBin:
    @pytest.mark.parametrize('loa, col', [(1, 0), (25, 0), (25.5, 1), (100, 3), (350, 13)])
    def test_classes_are_min_exclusive_max_inclusive(self, loa, col):
        assert length_bin(loa, interval_bounds(STEPS_25)) == col

    def test_longer_than_the_last_class_goes_to_the_last_column(self):
        # Never a column past the last class (the old Total-column bug).
        assert length_bin(400, interval_bounds(STEPS_25)) == 13

    def test_uses_the_project_classes_not_25_m_steps(self):
        bounds = interval_bounds([{'min': 0, 'max': 50}, {'min': 50, 'max': 200}, {'min': 200, 'max': 400}])
        assert length_bin(120, bounds) == 1
        assert length_bin(399, bounds) == 2

    @pytest.mark.parametrize('loa', [None, 0, -5, 'n/a', float('nan')])
    def test_unknown_length_counts_as_default(self, loa):
        expected = length_bin(UNKNOWN_LOA_M, interval_bounds(STEPS_25))
        assert length_bin(loa, interval_bounds(STEPS_25)) == expected

    def test_fallback_without_classes_is_25_m_over_the_columns(self):
        assert length_bin(60, [], n_cols=5) == 2
        assert length_bin(999, [], n_cols=5) == 4


class TestObservations:
    @pytest.mark.parametrize('sog, expected', [(12.4, 12.4), (0.0, 0.0), (102.2, 102.2), (102.3, None),
                                               (None, None), (-1, None), (float('inf'), None)])
    def test_speed_not_available_is_dropped(self, sog, expected):
        from omrat_utils.ais_binning import valid_sog
        assert valid_sog(sog) == expected

    @pytest.mark.parametrize('value, expected', [(6.5, 6.5), (0, None), (None, None), ('x', None)])
    def test_zero_dimension_is_unknown(self, value, expected):
        from omrat_utils.ais_binning import valid_dimension
        assert valid_dimension(value) == expected

    def test_add_observations_skips_missing_values(self):
        from omrat_utils.ais_binning import add_observations
        keys = ('Speed (knots)', 'Ship heights (meters)', 'Ship Beam (meters)', 'Draught (meters)')
        cells = {k: [[[]]] for k in keys}
        add_observations(cells, 0, 0, sog=102.3, air_draught=0, beam=32.0, draught=0.0)
        add_observations(cells, 0, 0, sog=11.0, air_draught=None, beam=0, draught=9.5)
        assert cells['Speed (knots)'][0][0] == [11.0]
        assert cells['Ship heights (meters)'][0][0] == []
        assert cells['Ship Beam (meters)'][0][0] == [32.0]
        assert cells['Draught (meters)'][0][0] == [9.5]
