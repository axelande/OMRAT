"""Missing draughts and speeds in the traffic data (``compute/traffic_gaps.py``) -- standalone.

KNOWN_ISSUES item 9: an AIS update writes ``inf`` for a cell with ships but
no draught observation, and the grounding models read that as unlimited
draught (run axel_run2: half of powered grounding came from it).
KNOWN_ISSUES item 14: the same ``inf`` as a speed made the head-on,
overtaking and merging totals NaN (Salvorev, 29 Oct 2026).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.traffic_gaps import (  # noqa: E402
    DEFAULT_DRAUGHT_M,
    DEFAULT_SPEED_KN,
    FILL_RULE,
    cell_gaps,
    describe_fill,
    describe_gap_cell,
    describe_new_gaps,
    fill_missing_draughts,
    fill_missing_speeds,
    fill_traffic_gaps,
    gap_cells,
    new_gaps,
    preview_traffic_gaps,
)

INF = math.inf


def _block(freq, draught):
    return {'Frequency (ships/year)': freq, 'Draught (meters)': draught}


def _data(*blocks):
    return {'traffic_data': {str(i): {'East going': b} for i, b in enumerate(blocks)}}


class TestFillMissingDraughts:
    def test_same_type_and_length_on_other_legs_first(self):
        data = _data(
            _block([[10, 0]], [[4.0, INF]]),
            _block([[30, 0]], [[INF, INF]]),
            _block([[30, 0]], [[6.0, INF]]),
        )
        summary = fill_missing_draughts(data)
        # weighted mean of 4.0 (10 ships) and 6.0 (30 ships)
        assert data['traffic_data']['1']['East going']['Draught (meters)'][0][0] == pytest.approx(5.5)
        assert summary['ships'] == 30 and summary['cells'] == 1
        assert summary['by_source']['type_and_length'] == 30

    def test_then_same_type_then_same_length_then_default(self):
        data = _data(
            # type 0: draught only in length 0; type 1: nothing; type 2: nothing at all
            _block([[10, 5], [0, 0], [0, 0]], [[3.0, INF], [INF, INF], [INF, INF]]),
            _block([[0, 0], [7, 0], [0, 0]], [[INF, INF], [8.0, INF], [INF, INF]]),
            _block([[0, 0], [2, 0], [0, 4]], [[INF, INF], [INF, INF], [INF, INF]]),
        )
        summary = fill_missing_draughts(data)
        td = data['traffic_data']
        assert td['0']['East going']['Draught (meters)'][0][1] == pytest.approx(3.0)   # same type
        assert td['2']['East going']['Draught (meters)'][1][0] == pytest.approx(8.0)   # same type+length
        assert td['2']['East going']['Draught (meters)'][2][1] == DEFAULT_DRAUGHT_M      # nothing to go on
        assert summary['by_source'] == {'type_and_length': 2, 'type': 5, 'length': 0, 'default': 4}

    def test_zero_and_blank_count_as_missing(self):
        data = _data(_block([[10, 10]], [[0.0, '']]), _block([[10, 0]], [[7.0, INF]]))
        fill_missing_draughts(data)
        assert data['traffic_data']['0']['East going']['Draught (meters)'][0] == [7.0, 7.0]

    def test_cells_without_ships_and_valid_draughts_are_untouched(self):
        data = _data(_block([[0, 12]], [[INF, 9.1]]))
        summary = fill_missing_draughts(data)
        assert data['traffic_data']['0']['East going']['Draught (meters)'] == [[INF, 9.1]]
        assert summary['cells'] == 0
        assert describe_fill(summary) == ''

    def test_describe(self):
        data = _data(_block([[10, 5]], [[4.0, INF]]))
        text = describe_fill(fill_missing_draughts(data))
        assert text.startswith('5 ships/year in 1 traffic cells had no draught')
        assert 'same type: 5' in text

    def test_no_traffic_block(self):
        assert fill_missing_draughts({})['cells'] == 0


def _speed_block(freq, speed, draught=None):
    block = {'Frequency (ships/year)': freq, 'Speed (knots)': speed}
    if draught is not None:
        block['Draught (meters)'] = draught
    return block


class TestFillMissingSpeeds:
    def test_same_type_and_length_on_other_legs_first(self):
        # Salvorev LEG_9_1_c: hand-edited frequencies put ships in cells
        # whose AIS speed was ``inf``.
        data = _data(
            _speed_block([[10, 0]], [[8.0, INF]]),
            _speed_block([[30, 0]], [[INF, INF]]),
            _speed_block([[30, 0]], [[12.0, INF]]),
        )
        summary = fill_missing_speeds(data)
        # weighted mean of 8 kn (10 ships) and 12 kn (30 ships)
        assert data['traffic_data']['1']['East going']['Speed (knots)'][0][0] == pytest.approx(11.0)
        assert summary['ships'] == 30 and summary['cells'] == 1
        assert summary['by_source']['type_and_length'] == 30

    def test_zero_blank_and_nan_count_as_missing_then_default(self):
        data = _data(_speed_block([[10, 10, 10]], [[0.0, '', math.nan]]))
        summary = fill_missing_speeds(data)
        assert data['traffic_data']['0']['East going']['Speed (knots)'] == [[DEFAULT_SPEED_KN] * 3]
        assert summary['by_source']['default'] == 30

    def test_cells_without_ships_keep_inf(self):
        data = _data(_speed_block([[0, 12]], [[INF, 9.1]]))
        assert fill_missing_speeds(data)['cells'] == 0
        assert data['traffic_data']['0']['East going']['Speed (knots)'] == [[INF, 9.1]]

    def test_describe_names_speed_and_its_default(self):
        data = _data(_speed_block([[10, 5]], [[INF, INF]]))
        text = describe_fill(fill_missing_speeds(data))
        assert text.startswith('15 ships/year in 2 traffic cells had no speed')
        assert 'mean speed of the other legs' in text
        assert f'default {DEFAULT_SPEED_KN:g} kn: 15' in text

    def test_preview_counts_without_filling(self):
        data = _data(
            _speed_block([[10, 4]], [[INF, 9.0]], draught=[[INF, 6.0]]),
            _speed_block([[10, 0]], [[14.0, INF]], draught=[[7.0, INF]]),
        )
        preview = preview_traffic_gaps(data)
        assert data['traffic_data']['0']['East going']['Speed (knots)'] == [[INF, 9.0]]
        assert [(s['quantity'], s['ships'], s['cells']) for s in preview] == [('draught', 10, 1), ('speed', 10, 1)]
        assert preview == fill_traffic_gaps(data)

    def test_fill_traffic_gaps_fills_both_and_reports_each(self):
        data = _data(
            _speed_block([[10]], [[INF]], draught=[[INF]]),
            _speed_block([[10]], [[14.0]], draught=[[7.0]]),
        )
        draught, speed = fill_traffic_gaps(data)
        block = data['traffic_data']['0']['East going']
        assert block['Draught (meters)'] == [[7.0]] and block['Speed (knots)'] == [[14.0]]
        assert describe_fill(draught).startswith('10 ships/year in 1 traffic cells had no draught')
        assert describe_fill(speed).startswith('10 ships/year in 1 traffic cells had no speed')


class TestTrafficTabGaps:
    """What the Traffic tab warns about: cells an edit leaves with ships but
    no speed / draught, and the one wording of how a run fills them."""

    @staticmethod
    def _block(freq, speed, draught):
        return {'Frequency (ships/year)': freq, 'Speed (knots)': speed, 'Draught (meters)': draught}

    def test_gap_cells_only_where_there_are_ships(self):
        block = self._block([[5, 0, 3], [2, 0, 0]],
                            [[INF, INF, 12.0], [0.0, INF, INF]],
                            [[6.0, INF, []], [7.0, INF, INF]])
        assert gap_cells(block) == {'speed': {(0, 0), (1, 0)}, 'draught': {(0, 2)}}
        assert cell_gaps(block, 0, 0) == ['speed']
        assert cell_gaps(block, 0, 1) == []          # no ships, nothing to fill
        assert cell_gaps(block, 9, 9) == []          # outside the matrix
        assert gap_cells({}) == {'speed': set(), 'draught': set()}

    def test_new_gaps_ignores_the_ones_already_there(self):
        before = {'speed': {(0, 0)}, 'draught': set()}
        after = {'speed': {(0, 0), (1, 2)}, 'draught': {(1, 2)}}
        assert new_gaps(before, after) == {'speed': {(1, 2)}, 'draught': {(1, 2)}}

    def test_warning_names_the_cells_and_how_the_run_fills_them(self):
        new = {'speed': {(0, 1), (1, 0)}, 'draught': {(0, 1)}}
        text = describe_new_gaps(new, ['Fishing', 'Cargo'], ['0 - 25', '25 - 50'])
        assert text.startswith('2 cell(s) now have ships but no speed or draught '
                               '(Fishing 25 - 50, Cargo 0 - 25). ')
        assert FILL_RULE in text
        assert text.endswith('Enter Speed (knots) / Draught (meters) for these cells to use your own values.')
        assert describe_new_gaps({'speed': set(), 'draught': set()}) == ''

    def test_warning_lists_three_cells_then_counts(self):
        new = {'speed': {(0, j) for j in range(5)}, 'draught': set()}
        text = describe_new_gaps(new, ['Fishing'], ['a', 'b'])
        assert '(Fishing a, Fishing b, Fishing length class 3 and 2 more)' in text
        assert 'no speed (' in text and 'Enter Speed (knots) for' in text

    def test_fill_rule_states_the_order_and_defaults(self):
        assert 'same ship type and length on the other legs' in FILL_RULE
        assert f'{DEFAULT_SPEED_KN:g} kn speed / {DEFAULT_DRAUGHT_M:g} m draught' in FILL_RULE
        assert describe_gap_cell(['speed']) == f'These ships have no speed. {FILL_RULE} Type a value to use your own.'
        assert describe_gap_cell([]) == ''


# ---------------------------------------------------------------------------
# Negative mixture weights (KNOWN_ISSUES 6)
# ---------------------------------------------------------------------------

def test_negative_weights_are_listed_by_leg_name():
    from compute.traffic_gaps import describe_negative_weights, negative_weights
    segs = {'7': {'Leg_name': 'LEG_4_5_a', 'weight1_1': 50, 'weight1_2': -4900, 'u_p2': -1},
            '8': {'weight1_1': 100, 'weight2_1': '100'}}
    found = negative_weights(segs)
    assert found == [('LEG_4_5_a', 'weight1_2', -4900.0), ('LEG_4_5_a', 'u_p2', -1.0)]
    text = describe_negative_weights(found)
    assert text.startswith('2 lateral-distribution weight(s) are negative') and 'LEG_4_5_a weight1_2 = -4900' in text
    assert describe_negative_weights([]) == ''
    assert negative_weights(None) == []
