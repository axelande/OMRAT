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
