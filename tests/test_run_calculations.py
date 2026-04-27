"""Unit tests for compute/run_calculations.py -- the Calculation facade.

Covers the progress-callback plumbing and get_no_ship_h computation.
The individual mixin methods are tested by their respective test files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.run_calculations import Calculation


@pytest.fixture
def calc():
    return Calculation(MagicMock())


class TestProgressCallback:
    def test_set_and_report_progress_calls_callback(self, calc):
        received = []
        calc.set_progress_callback(lambda c, t, m: received.append((c, t, m)) or True)
        calc._report_progress('spatial', 0.5, 'hello')
        assert len(received) == 1
        # 'spatial' phase is 0-40%, 0.5 progress -> 20% overall
        completed, total, msg = received[0]
        assert completed == 20 and total == 100 and msg == 'hello'

    def test_progress_without_callback_returns_true(self, calc):
        # No callback set -- _report_progress returns True silently.
        assert calc._report_progress('cascade', 0.5, 'x') is True

    def test_cancellation_propagates(self, calc):
        calc.set_progress_callback(lambda c, t, m: False)  # cancel immediately
        assert calc._report_progress('spatial', 0.1, 'x') is False

    @pytest.mark.parametrize("phase, progress, expected_pct", [
        ('spatial', 0.0, 0),
        ('spatial', 1.0, 40),
        ('shadow', 0.0, 40),
        ('shadow', 1.0, 60),
        ('cascade', 0.5, 75),  # midpoint of 60-90
        ('layers', 1.0, 100),
    ])
    def test_phase_weights_correct(self, calc, phase, progress, expected_pct):
        out = []
        calc.set_progress_callback(lambda c, t, m: out.append(c) or True)
        calc._report_progress(phase, progress, '')
        assert out[0] == expected_pct

    def test_phase_progress_clamped_to_0_1(self, calc):
        out = []
        calc.set_progress_callback(lambda c, t, m: out.append(c) or True)
        calc._report_progress('cascade', -0.5, '')  # clamped to 0 -> 60%
        calc._report_progress('cascade', 2.0, '')   # clamped to 1 -> 90%
        assert out == [60, 90]

    def test_unknown_phase_defaults_to_0_100(self, calc):
        out = []
        calc.set_progress_callback(lambda c, t, m: out.append(c) or True)
        calc._report_progress('unknown', 0.5, '')
        assert out[0] == 50  # default (0.0, 1.0) range


class TestGetNoShipH:
    def test_simple_one_leg_one_direction(self, calc):
        data = {
            'traffic_data': {
                '1': {
                    'East going': {
                        'Frequency (ships/year)': [[100]],  # 100 ships
                        'Speed (knots)': [[10]],           # 10 kts
                    },
                },
            },
            'segment_data': {'1': {'line_length': 18520.0}},  # 18.52 km
        }
        out = calc.get_no_ship_h(data)
        # The function computes h = line_length / (speed * 1852/3600) in
        # seconds (despite the "h" naming convention).  For this input
        # the transit time is exactly 1 hour = 3600 s; total = 360_000.
        assert len(out) == 1
        assert out[0] == pytest.approx(360_000.0, abs=1e-3)

    def test_multiple_legs_and_directions(self, calc):
        data = {
            'traffic_data': {
                '1': {
                    'East going': {'Frequency (ships/year)': [[10]], 'Speed (knots)': [[10]]},
                    'West going': {'Frequency (ships/year)': [[5]], 'Speed (knots)': [[5]]},
                },
                '2': {
                    'East going': {'Frequency (ships/year)': [[20]], 'Speed (knots)': [[10]]},
                },
            },
            'segment_data': {
                '1': {'line_length': 18520.0},
                '2': {'line_length': 18520.0},
            },
        }
        out = calc.get_no_ship_h(data)
        assert len(out) == 3  # one entry per (leg, direction)

    def test_zero_freq_gives_zero(self, calc):
        data = {
            'traffic_data': {
                '1': {'East going': {'Frequency (ships/year)': [[0]], 'Speed (knots)': [[10]]}},
            },
            'segment_data': {'1': {'line_length': 1000.0}},
        }
        out = calc.get_no_ship_h(data)
        assert out == [0.0]


class TestCalculationInit:
    def test_default_attributes_set(self, calc):
        assert calc.drifting_report is None
        assert calc.allision_result_layer is None
        assert calc.grounding_result_layer is None
        assert calc.ship_collision_prob == 0.0
        assert calc.drifting_allision_prob == 0.0
        assert calc.drifting_grounding_prob == 0.0

    def test_progress_callback_starts_as_none(self, calc):
        assert calc._progress_callback is None
