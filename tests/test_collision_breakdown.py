"""Pure tests for ``compute.collision_breakdown`` (the per-leg View table)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.collision_breakdown import build_breakdown_rows  # noqa: E402


def _report():
    return {
        'totals': {'overtaking': 4e-3, 'crossing': 0.0},
        'by_leg': {
            '1': {'overtaking': 1e-3, 'head_on': 5e-4},
            '2': {'overtaking': 3e-3, 'head_on': 0.0},
            '3': {'overtaking': 0.0},
            'bad': 'not-a-dict',
        },
        'by_leg_pair': {
            '1 -> 2': {'crossing': 2e-4, 'waypoint': '12.0 56.0', 'angle_deg': 45.0},
            '2 -> 3': {'crossing': 6e-4, 'waypoint': '12.1 56.1', 'angle_deg': 90.0},
        },
        'bend_by_pair': {
            '1 -> 2': {'bend': 1e-5, 'waypoint': '12.0 56.0'},
        },
    }


class TestSingleLeg:
    def test_headers_end_with_probability_and_share(self):
        headers, _rows = build_breakdown_rows(_report(), 'overtaking')
        assert headers == ['Leg', 'Probability', '% of total']

    def test_rows_sorted_desc_with_share_of_model_total(self):
        _h, rows = build_breakdown_rows(_report(), 'overtaking')
        assert [r[0] for r in rows] == ['2', '1']
        assert rows[0][1] == '3.000e-03'
        assert rows[0][2] == '75.0%'
        assert rows[1][2] == '25.0%'

    def test_zero_rows_are_dropped(self):
        _h, rows = build_breakdown_rows(_report(), 'head_on')
        assert [r[0] for r in rows] == ['1']
        # No head_on total in the report -> share of listed rows.
        assert rows[0][2] == '100.0%'


class TestLegPairs:
    def test_crossing_falls_back_to_row_sum_when_total_zero(self):
        headers, rows = build_breakdown_rows(_report(), 'crossing')
        assert headers == ['Leg pair', 'Waypoint (lon lat)', 'Angle°', 'Probability', '% of total']
        assert [r[0] for r in rows] == ['2 -> 3', '1 -> 2']
        assert rows[0][2] == '90.0'
        assert rows[0][4] == '75.0%'
        assert rows[1][4] == '25.0%'

    def test_bend(self):
        headers, rows = build_breakdown_rows(_report(), 'bend')
        assert headers == ['Leg pair', 'Waypoint (lon lat)', 'Probability', '% of total']
        assert rows == [['1 -> 2', '12.0 56.0', '1.000e-05', '100.0%']]


class TestEdgeCases:
    def test_empty_report(self):
        headers, rows = build_breakdown_rows(None, 'overtaking')
        assert rows == []
        assert headers[-2:] == ['Probability', '% of total']

    def test_shares_sum_to_100(self):
        _h, rows = build_breakdown_rows(_report(), 'overtaking')
        total = sum(float(r[-1].rstrip('%')) for r in rows)
        assert abs(total - 100.0) < 0.2
