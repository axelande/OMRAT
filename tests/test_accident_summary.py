"""Pure tests for ``omrat_utils.accident_summary`` (no QGIS needed)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.accident_summary import (  # noqa: E402
    ACCIDENT_TOTAL_KEYS, SUMMARY_ROWS, format_probability, parse_probability,
    summary_values,
)


class TestRowVocabulary:
    def test_nine_accident_keys(self):
        assert len(ACCIDENT_TOTAL_KEYS) == 9

    def test_summary_labels_and_order(self):
        assert [label for label, _ in SUMMARY_ROWS] == [
            'All grounding', 'All allision', 'All collisions',
        ]

    def test_every_summary_key_is_an_accident_key(self):
        for _label, keys in SUMMARY_ROWS:
            for k in keys:
                assert k in ACCIDENT_TOTAL_KEYS

    def test_every_accident_key_is_summed_exactly_once(self):
        seen = [k for _label, keys in SUMMARY_ROWS for k in keys]
        assert sorted(seen) == sorted(ACCIDENT_TOTAL_KEYS)


class TestSummaryValues:
    def test_sums_components(self):
        totals = {
            'drift_grounding': 1e-3, 'powered_grounding': 2e-3,
            'drift_allision': 4e-4, 'powered_allision': 6e-4,
            'overtaking': 1e-5, 'head_on': 2e-5, 'crossing': 3e-5,
            'merging': 4e-5, 'bend': 5e-5,
        }
        grounding, allision, collisions = summary_values(totals)
        assert abs(grounding - 3e-3) < 1e-12
        assert abs(allision - 1e-3) < 1e-12
        assert abs(collisions - 1.5e-4) < 1e-12

    def test_missing_component_counts_as_zero(self):
        grounding, allision, collisions = summary_values({'drift_grounding': 1e-3})
        assert abs(grounding - 1e-3) < 1e-12
        assert allision is None
        assert collisions is None

    def test_accepts_formatted_strings(self):
        grounding, _a, _c = summary_values({
            'drift_grounding': '1.000e-03', 'powered_grounding': '',
        })
        assert abs(grounding - 1e-3) < 1e-12

    def test_all_missing_gives_none(self):
        assert summary_values({}) == [None, None, None]


class TestFormatting:
    def test_parse(self):
        assert parse_probability('1.5e-03') == 1.5e-3
        assert parse_probability('') is None
        assert parse_probability(None) is None
        assert parse_probability('—') is None

    def test_format(self):
        assert format_probability(None) == ''
        assert format_probability(0) == '0'
        assert format_probability(1.23456e-4) == '1.235e-04'


# ---------------------------------------------------------------------------
# Display mode: frequency vs. years between incidents
# ---------------------------------------------------------------------------

from omrat_utils.accident_summary import (  # noqa: E402
    DISPLAY_FREQUENCY, DISPLAY_MODES, DISPLAY_RETURN_PERIOD, INFINITY_TEXT,
    catastrophe_header, format_result, format_years, normalize_display_mode,
    result_header, return_period_years,
)


class TestDisplayMode:
    def test_modes_and_normalisation(self):
        assert DISPLAY_MODES == ('frequency', 'return_period')
        assert normalize_display_mode(None) == DISPLAY_FREQUENCY
        assert normalize_display_mode('') == DISPLAY_FREQUENCY
        assert normalize_display_mode('garbage') == DISPLAY_FREQUENCY
        assert normalize_display_mode(' Return_Period ') == DISPLAY_RETURN_PERIOD

    def test_return_period_is_reciprocal(self):
        assert return_period_years(1e-3) == 1000.0
        assert return_period_years(None) is None
        assert return_period_years('x') is None
        assert return_period_years(0.0) == float('inf')
        assert return_period_years(-1.0) == float('inf')

    def test_format_years_buckets(self):
        assert format_years(None) == ''
        assert format_years(float('inf')) == INFINITY_TEXT
        assert format_years(12345.6) == '12,346'
        assert format_years(100.0) == '100'
        assert format_years(45.67) == '45.7'
        assert format_years(2.345) == '2.35'

    def test_format_result_frequency_matches_lep_format(self):
        assert format_result(1.23456e-4, DISPLAY_FREQUENCY) == '1.235e-04'
        assert format_result('1.23456e-4') == '1.235e-04'
        assert format_result(None, DISPLAY_FREQUENCY) == ''
        assert format_result(0, DISPLAY_FREQUENCY) == '0'

    def test_format_result_return_period(self):
        assert format_result(1e-3, DISPLAY_RETURN_PERIOD) == '1,000'
        assert format_result('2.000e-02', DISPLAY_RETURN_PERIOD) == '50.0'
        assert format_result(0.5, DISPLAY_RETURN_PERIOD) == '2.00'
        assert format_result(0, DISPLAY_RETURN_PERIOD) == INFINITY_TEXT
        assert format_result(None, DISPLAY_RETURN_PERIOD) == ''
        assert format_result('—', DISPLAY_RETURN_PERIOD) == ''

    def test_headers(self):
        assert result_header(DISPLAY_FREQUENCY) == 'Probability'
        assert result_header(DISPLAY_RETURN_PERIOD) == 'Years between incidents'
        assert catastrophe_header(DISPLAY_FREQUENCY) == 'Exceedance (events/year)'
        assert catastrophe_header(DISPLAY_RETURN_PERIOD) == 'Years between exceedances'
