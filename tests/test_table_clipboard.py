"""Pure tests for ``omrat_utils.table_clipboard`` (Traffic table <-> Excel)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.table_clipboard import (  # noqa: E402
    format_number, parse_number, parse_tsv, plan_paste, strip_headers, to_tsv,
)

SV = (',', ' ')      # Swedish: decimal comma, non-breaking space groups
EN = ('.', ',')


class TestParseNumber:
    @pytest.mark.parametrize('text, expected', [
        ('12,5', 12.5), ('12.5', 12.5), ('1 234,5', 1234.5), ('1 234,5', 1234.5),
        ('1.234,5', 1234.5), ('1,234.5', 1234.5), ('1,234', 1.234), ('7', 7.0),
        (' 3 ', 3.0), ('130 %', 130.0), ('-2,5', -2.5), ('0,0', 0.0),
    ])
    def test_swedish_locale(self, text, expected):
        assert parse_number(text, *SV) == pytest.approx(expected)

    @pytest.mark.parametrize('text, expected', [
        ('12.5', 12.5), ('1,234', 1234.0), ('1,234,567', 1234567.0), ('12,5', 12.5),
        ('1,234.5', 1234.5), ('1.234,5', 1234.5),
    ])
    def test_english_locale(self, text, expected):
        assert parse_number(text, *EN) == pytest.approx(expected)

    @pytest.mark.parametrize('text', ['', '   ', ' '])
    def test_blank_is_none(self, text):
        assert parse_number(text, *SV) is None

    @pytest.mark.parametrize('text', ['abc', '12a', 'inf', 'nan', '1,2,3,4x'])
    def test_garbage_raises(self, text):
        with pytest.raises(ValueError):
            parse_number(text, *SV)


class TestFormatNumber:
    @pytest.mark.parametrize('value, expected', [
        (3, '3'), (3.0, '3'), (12.5, '12,5'), (12.3456, '12,35'), (14.25, '14,25'),
        (0.001, '0'), (-0.001, '0'), (1234.5, '1234,5'), (100.0, '100'),
    ])
    def test_swedish(self, value, expected):
        assert format_number(value, ',') == expected

    @pytest.mark.parametrize('value', ['', None, math.inf, -math.inf, [], [1.0, 2.0]])
    def test_no_observation_is_blank(self, value):
        assert format_number(value, ',') == ''

    def test_round_trip(self):
        for v in (0, 7, 12.5, 1234.75, 99.99, 100000):
            assert parse_number(format_number(v, ','), *SV) == pytest.approx(v)
            assert parse_number(format_number(v, '.'), *EN) == pytest.approx(v)


class TestTsv:
    def test_round_trip(self):
        rows = [['1', '2'], ['', '4,5']]
        assert parse_tsv(to_tsv(rows)) == rows

    def test_excel_line_ends_and_trailing_newline(self):
        assert parse_tsv('1\t2\r\n3\t4\r\n') == [['1', '2'], ['3', '4']]
        assert parse_tsv('1\r2') == [['1'], ['2']]

    def test_quoted_cell(self):
        assert parse_tsv('"Cargo"\t3') == [['Cargo', '3']]
        assert parse_tsv('"say ""hi"""\t3') == [['say "hi"', '3']]

    def test_empty(self):
        assert parse_tsv('') == [] and parse_tsv('\r\n') == []


class TestStripHeaders:
    def test_labelled_block(self):
        rows = parse_tsv('LEG_1 - East - Freq\t0 - 25\t25 - 50\nCargo\t3\t4\nTanker\t5\t6')
        assert strip_headers(rows, *SV) == ([['3', '4'], ['5', '6']], True, True)

    def test_excel_block_with_blank_corner(self):
        rows = parse_tsv('\t0 - 25\t25 - 50\nCargo\t3\t4')
        assert strip_headers(rows, *SV) == ([['3', '4']], True, True)

    def test_plain_block_untouched(self):
        rows = parse_tsv('1\t2\n3\t4')
        assert strip_headers(rows, *SV) == (rows, False, False)

    @pytest.mark.parametrize('text', ['1\tx\n-3\t4', 'x\t1\n2\t3', '1\tx\ty'])
    def test_typos_are_not_headers(self, text):
        rows = parse_tsv(text)
        assert strip_headers(rows, *SV)[1:] == (False, False)

    def test_column_labels_only(self):
        rows = parse_tsv('\t0 - 25\t25 - 50\n\t3\t4')
        out, had_row, had_col = strip_headers(rows, *SV)
        assert had_row and not had_col
        assert out == [['', '3', '4']]

    def test_row_labels_only(self):
        rows = parse_tsv('Cargo\t3\t4\nTanker\t5\t6')
        assert strip_headers(rows, *SV) == ([['3', '4'], ['5', '6']], False, True)


class TestPlanPaste:
    def test_block_at_anchor_with_clipping(self):
        plan = plan_paste([['1', '2'], ['3', '4']], 2, 2, (1, 1))
        assert plan.cells == [(1, 1, '1')]
        assert plan.clipped == 3

    def test_single_value_fills_selection(self):
        plan = plan_paste([['9']], 3, 3, (0, 0), [(0, 0), (2, 2), (5, 5)])
        assert plan.cells == [(0, 0, '9'), (2, 2, '9')]

    def test_single_value_single_cell(self):
        plan = plan_paste([['9']], 3, 3, (1, 2), [(1, 2)])
        assert plan.cells == [(1, 2, '9')]

    def test_ragged_rows(self):
        plan = plan_paste([['1'], ['2', '3']], 2, 2, (0, 0))
        assert plan.cells == [(0, 0, '1'), (1, 0, '2'), (1, 1, '3')]

    def test_empty(self):
        assert plan_paste([], 2, 2, (0, 0)).cells == []
