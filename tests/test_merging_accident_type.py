# -*- coding: utf-8 -*-
"""Merging as a first-class accident type (v0.14.0).

Before this change, leg pairs meeting at a shallow angle were classified
as *merging* internally but their frequency was summed into the crossing
total, and the Run Analysis row labelled "Merging collision" was fed the
*bend* number instead.  These tests lock in the separation:

* ``_calc_crossing_collisions`` returns crossing and merging separately,
* each kind uses its own causation factor,
* the consequence module routes merging and bend to distinct rows.
"""
import pytest
from numpy import isclose

from compute.ship_collision_model import ShipCollisionModelMixin
from omrat_utils.consequence_defaults import (
    ACCIDENT_KEYS,
    ACCIDENT_TYPES,
    default_spill_fraction,
    default_spill_probability,
)


class _StubRunner(ShipCollisionModelMixin):
    """Minimal host so the mixin's instance methods become callable."""

    def __init__(self):
        self._progress_log: list[tuple[str, float, str]] = []

    def _report_progress(self, stage, fraction, message) -> None:
        self._progress_log.append((stage, fraction, message))


def _seg(start: str, end: str, length_m: float = 100_000) -> dict:
    return {
        'Start_Point': start,
        'End_Point': end,
        'line_length': length_m,
        'Width': 5000,
        'Route_Id': 1,
        'Leg_name': 'LEG',
    }


def _traffic(freq: float = 1000.0) -> dict:
    one = [[freq]]
    return {
        'East going': {
            'Frequency (ships/year)': one,
            'Speed (knots)': [[10.0]],
            'Ship Beam (meters)': [[20.0]],
        },
    }


def _shallow_pair() -> dict:
    """Two legs sharing a start point, diverging by about 17 degrees.

    Well inside ``MERGING_ANGLE_DEG`` (30), so the pair classifies as
    merging rather than crossing.
    """
    return {
        '1': _seg("15.0 55.0", "15.0 56.0"),      # due north
        '2': _seg("15.0 55.0", "15.30 55.97"),    # north-north-east
    }


def _steep_pair() -> dict:
    """Two legs sharing a start point and meeting at right angles."""
    return {
        '1': _seg("15.0 55.0", "15.0 56.0"),   # due north
        '2': _seg("15.0 55.0", "16.0 55.0"),   # due east
    }


LI = [{'min': 50, 'max': 150}]


def _run(segments, *, pc_crossing=1.3e-4, pc_merging=1.3e-4):
    runner = _StubRunner()
    traffic = {k: _traffic() for k in segments}
    return runner._calc_crossing_collisions(
        traffic_data=traffic, segment_data=segments,
        leg_keys=list(segments.keys()),
        pc_crossing=pc_crossing, pc_merging=pc_merging,
        length_intervals=LI,
    )


# ---------------------------------------------------------------------------
# Angle classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_shallow_angle_counts_as_merging_only(self):
        totals = _run(_shallow_pair())
        assert totals['merging'] > 0.0
        assert totals['crossing'] == 0.0

    def test_right_angle_counts_as_crossing_only(self):
        totals = _run(_steep_pair())
        assert totals['crossing'] > 0.0
        assert totals['merging'] == 0.0

    def test_threshold_is_the_documented_30_degrees(self):
        assert ShipCollisionModelMixin.MERGING_ANGLE_DEG == 30.0


# ---------------------------------------------------------------------------
# Independent causation factors
# ---------------------------------------------------------------------------

class TestCausationFactors:
    def test_merging_uses_its_own_factor(self):
        """Doubling pc_merging doubles the merging total and leaves
        crossing untouched."""
        base = _run(_shallow_pair(), pc_merging=1.3e-4)
        doubled = _run(_shallow_pair(), pc_merging=2.6e-4)
        assert isclose(doubled['merging'] / base['merging'], 2.0, rtol=1e-12)

    def test_crossing_factor_does_not_affect_merging(self):
        base = _run(_shallow_pair(), pc_crossing=1.3e-4)
        other = _run(_shallow_pair(), pc_crossing=9.9e-3)
        assert isclose(base['merging'], other['merging'], rtol=1e-12)

    def test_explicit_merging_factor_is_respected(self):
        data = {'ship_categories': {'length_intervals': LI}}
        pc_vals = {'crossing': 1.3e-4, 'merging': 5.0e-5}
        (
            _ho, _ot, pc_cr, pc_mg, _be, _li,
        ) = ShipCollisionModelMixin._extract_pc_and_intervals(data, pc_vals)
        assert pc_cr == 1.3e-4
        assert pc_mg == 5.0e-5


# ---------------------------------------------------------------------------
# Totals bookkeeping
# ---------------------------------------------------------------------------

class TestTotals:
    def test_merging_is_summed_into_the_grand_total(self):
        result = {
            'head_on': 1.0, 'overtaking': 2.0, 'crossing': 4.0,
            'merging': 8.0, 'bend': 16.0, 'total': 0.0,
        }
        by_leg = {
            '1': {'head_on': 1.0, 'overtaking': 2.0, 'bend': 16.0},
        }
        ShipCollisionModelMixin._fill_result_totals(
            result, by_leg, {'crossing': 4.0, 'merging': 8.0},
        )
        assert result['merging'] == 8.0
        assert result['total'] == 1.0 + 2.0 + 4.0 + 8.0 + 16.0

    def test_merging_is_not_double_counted_in_crossing(self):
        totals = _run(_shallow_pair())
        # The whole pair went to merging; nothing leaked into crossing.
        assert totals['crossing'] == 0.0


# ---------------------------------------------------------------------------
# Consequence wiring
# ---------------------------------------------------------------------------

class TestConsequenceRouting:
    def test_bend_is_its_own_accident_type(self):
        assert 'bend' in ACCIDENT_KEYS
        assert 'Bend collision' in ACCIDENT_TYPES
        assert len(ACCIDENT_KEYS) == len(ACCIDENT_TYPES)

    def test_merging_and_bend_keys_are_distinct_rows(self):
        assert ACCIDENT_KEYS.index('merging') != ACCIDENT_KEYS.index('bend')

    @pytest.mark.parametrize('key', ['merging', 'bend'])
    def test_by_cell_routing_does_not_mix_merging_and_bend(self, key):
        """``_by_cell_for_accident`` used to add the bend cells into the
        merging bucket.  Each must now return only its own cells."""
        from compute.consequence import _by_cell_for_accident

        collision_report = {
            'by_cell': {
                'merging': {'0_0': 1.0},
                'bend': {'1_1': 2.0},
            },
        }
        out = _by_cell_for_accident(key, None, None, None, collision_report)
        expected = {'merging': {'0_0': 1.0}, 'bend': {'1_1': 2.0}}[key]
        assert out == expected

    def test_default_spill_matrices_have_a_row_per_accident(self):
        assert len(default_spill_probability()) == len(ACCIDENT_TYPES)
        assert len(default_spill_fraction()) == len(ACCIDENT_TYPES)

    def test_every_spill_probability_row_still_sums_to_100(self):
        for label, row in zip(ACCIDENT_TYPES, default_spill_probability()):
            assert isclose(sum(row), 100.0), label


# ---------------------------------------------------------------------------
# Row-order invariant
# ---------------------------------------------------------------------------

class TestRowOrderInvariant:
    """``ACCIDENT_TYPES`` indexes the spill matrices *by position*, so its
    order must match ``AccidentResultsMixin._ACCIDENT_ROWS`` exactly.

    ``accident_results_mixin`` imports qgis, so the tuple is read out of
    the source with ``ast`` rather than imported.
    """

    @staticmethod
    def _accident_row_labels() -> list[str]:
        import ast
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'omrat_utils' / 'accident_results_mixin.py'
        ).read_text(encoding='utf-8')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if getattr(target, 'id', None) != '_ACCIDENT_ROWS':
                continue
            rows = ast.literal_eval(node.value)
            return [row[0] for row in rows]
        raise AssertionError('_ACCIDENT_ROWS not found')

    def test_labels_match_accident_types_in_order(self):
        assert self._accident_row_labels() == list(ACCIDENT_TYPES)

    def test_bend_row_exists_in_the_ui_table(self):
        assert 'Bend collision' in self._accident_row_labels()
