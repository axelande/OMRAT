"""Run columns of the Run Analysis result tables (pure module, standalone).

Layout (user's design, 2026-11-09): the baseline once with values only,
then every other selected run with its value and its change against the
baseline; no column for the live model; the change in the unit shown;
numbers copied to Excel with the locale's decimal point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.accident_summary import (  # noqa: E402
    ACCIDENT_TOTAL_KEYS, DISPLAY_FREQUENCY, DISPLAY_RETURN_PERIOD, SUMMARY_ROWS,
)
from omrat_utils.result_compare import (  # noqa: E402
    BASELINE_SUFFIX, CHANGE, CHANGE_HEADER, DASH, VALUE, RunValues,
    accident_values, build_compare_columns, catastrophe_rows, change_pct,
    change_text, compare_hint, copy_number, exceedance_values,
    value_copy_text, value_text,
)


class TestChangePct:
    def test_frequency_change(self):
        assert change_pct(2e-4, 1e-4) == pytest.approx(100.0)
        assert change_pct(5e-5, 1e-4) == pytest.approx(-50.0)
        assert change_pct(0.0, 1e-4) == pytest.approx(-100.0)

    def test_years_change_is_the_change_of_one_over_f(self):
        # Frequency doubles -> years halve.
        assert change_pct(2e-4, 1e-4, DISPLAY_RETURN_PERIOD) == pytest.approx(-50.0)
        assert change_pct(5e-5, 1e-4, DISPLAY_RETURN_PERIOD) == pytest.approx(100.0)

    @pytest.mark.parametrize('value, base, mode', [
        (None, 1e-4, DISPLAY_FREQUENCY),
        (1e-4, None, DISPLAY_FREQUENCY),
        (1e-4, 0.0, DISPLAY_FREQUENCY),           # no base to divide by
        (0.0, 1e-4, DISPLAY_RETURN_PERIOD),       # infinite years
        (1e-4, 0.0, DISPLAY_RETURN_PERIOD),
        ('n/a', 1e-4, DISPLAY_FREQUENCY),
    ])
    def test_undefined(self, value, base, mode):
        assert change_pct(value, base, mode) is None

    def test_text(self):
        assert change_text(12.345) == '+12.3%'
        assert change_text(-50.0) == '-50.0%'
        assert change_text(0.0) == '+0.0%'
        assert change_text(None) == DASH


class TestBuildColumns:
    base = RunValues('base', [1e-4, 2e-4], run_id=1)
    run_a = RunValues('a', [2e-4, 2e-4], run_id=2)
    run_b = RunValues('b', [5e-5, None], run_id=3)

    def test_baseline_once_then_value_and_change_per_run(self):
        cols = build_compare_columns(self.base, [self.run_a, self.run_b], 2)
        assert [(c.kind, c.header) for c in cols] == [
            (VALUE, 'base' + BASELINE_SUFFIX),
            (VALUE, 'a'), (CHANGE, CHANGE_HEADER),
            (VALUE, 'b'), (CHANGE, CHANGE_HEADER),
        ]
        assert cols[0].is_baseline and not cols[1].is_baseline
        assert cols[2].cells == pytest.approx([100.0, 0.0])
        assert cols[4].cells[0] == pytest.approx(-50.0)
        assert cols[4].cells[1] is None       # b did not calculate row 2
        assert '"a"' in cols[2].tooltip and '"base"' in cols[2].tooltip

    def test_selected_baseline_is_not_repeated(self):
        cols = build_compare_columns(self.base, [self.base, self.run_a], 2)
        assert [c.header for c in cols] == ['base' + BASELINE_SUFFIX, 'a', CHANGE_HEADER]

    def test_only_the_baseline(self):
        cols = build_compare_columns(self.base, [], 2)
        assert [c.header for c in cols] == ['base' + BASELINE_SUFFIX]

    def test_no_baseline_values_only(self):
        cols = build_compare_columns(None, [self.run_a, self.run_b], 2)
        assert [(c.kind, c.header) for c in cols] == [(VALUE, 'a'), (VALUE, 'b')]

    def test_nothing(self):
        assert build_compare_columns(None, [], 2) == []

    def test_change_follows_the_display_mode(self):
        cols = build_compare_columns(self.base, [self.run_a], 2, DISPLAY_RETURN_PERIOD)
        assert cols[2].cells == pytest.approx([-50.0, 0.0])

    def test_cells_padded_and_truncated_to_the_rows(self):
        cols = build_compare_columns(None, [RunValues('x', [1.0])], 3)
        assert cols[0].cells == [1.0, None, None]
        cols = build_compare_columns(None, [RunValues('x', [1.0, 2.0, 3.0])], 2)
        assert cols[0].cells == [1.0, 2.0]

    def test_run_without_data(self):
        cols = build_compare_columns(self.base, [RunValues('old', None, run_id=9, note='why')], 2)
        assert not cols[1].has_data
        assert cols[1].tooltip == 'why'
        assert cols[2].cells == [None, None]


class TestTexts:
    def test_value_text(self):
        assert value_text(1.234e-4) == '1.234e-04'
        assert value_text(1e-3, DISPLAY_RETURN_PERIOD) == '1,000'
        assert value_text(0.0) == '0'
        assert value_text(0.0, DISPLAY_RETURN_PERIOD) == '∞'
        assert value_text(None) == ''
        assert value_text(None, missing=DASH) == DASH
        assert value_text(1e-4, has_data=False) == DASH

    def test_copy_number_uses_the_decimal_point(self):
        assert copy_number(1.23456789e-5) == '1.23457e-05'
        assert copy_number(1.23456789e-5, ',') == '1,23457e-05'
        assert copy_number(0.5, ',') == '0,5'
        assert copy_number(50.0, ',') == '50'
        assert copy_number(None) == ''
        assert copy_number(float('inf')) == '∞'

    def test_value_copy_text_in_the_unit_shown(self):
        assert value_copy_text(4e-3, DISPLAY_FREQUENCY, ',') == '0,004'
        assert value_copy_text(4e-3, DISPLAY_RETURN_PERIOD, ',') == '250'
        assert value_copy_text(3e-3, DISPLAY_RETURN_PERIOD, ',') == '333,333'
        assert value_copy_text(0.0, DISPLAY_RETURN_PERIOD) == '∞'
        assert value_copy_text(None) == ''

    def test_hint(self):
        assert 'Select runs' in compare_hint(None, 0)
        assert 'Tick Baseline' in compare_hint(None, 2)
        assert compare_hint('Anna', 0).endswith('Anna')
        assert compare_hint('Anna', 3).endswith('Anna')


class TestRowValues:
    def test_accident_values_rows_and_summaries(self):
        totals = {k: 1e-4 for k in ACCIDENT_TOTAL_KEYS}
        totals['bend'] = None
        values = accident_values(totals)
        assert len(values) == len(ACCIDENT_TOTAL_KEYS) + len(SUMMARY_ROWS)
        assert values[ACCIDENT_TOTAL_KEYS.index('bend')] is None
        n = len(ACCIDENT_TOTAL_KEYS)
        assert values[n] == pytest.approx(2e-4)        # All grounding
        assert values[n + 2] == pytest.approx(4e-4)    # All collisions without bend

    def test_accident_values_empty(self):
        assert accident_values(None) == [None] * (len(ACCIDENT_TOTAL_KEYS) + len(SUMMARY_ROWS))

    def test_catastrophe_rows_union_in_order(self):
        a = {'levels': [{'name': 'Minor', 'quantity': 50.0}, {'name': 'Major', 'quantity': 500.0}]}
        b = {'levels': [{'name': 'Major', 'quantity': 999.0}, {'name': 'Extreme', 'quantity': 5e4}]}
        assert catastrophe_rows([None, a, b]) == [('Minor', 50.0), ('Major', 500.0), ('Extreme', 5e4)]

    def test_exceedance_values_by_name(self):
        res = {'levels': [{'name': 'Major', 'exceedance': 2e-4}, {'name': 'Minor', 'exceedance': 1e-3}]}
        assert exceedance_values(res, ['Minor', 'Major', 'Extreme']) == [1e-3, 2e-4, None]
        assert exceedance_values(None, ['Minor']) is None
