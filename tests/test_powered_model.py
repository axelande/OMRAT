"""Unit tests for ``compute.powered_model.PoweredModelMixin``.

Covers the two public entry points ``run_powered_grounding_model`` and
``run_powered_allision_model`` end-to-end on a tiny synthetic project,
plus the early-return branches (empty traffic / missing segments /
missing obstacles) and the widget-missing exception swallowing.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.powered_model import PoweredModelMixin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _traffic_cell(freq_val: float = 100.0, speed_kts: float = 10.0,
                  draught: float = 13.0, height: float = 20.0) -> dict:
    """21x2 traffic grid with one populated cell."""
    def grid(init=0.0):
        return [[init, init] for _ in range(21)]

    f, s, d, h = grid(), grid(), grid(), grid()
    f[18][1] = freq_val
    s[18][1] = speed_kts
    d[18][1] = draught
    h[18][1] = height
    return {
        'Frequency (ships/year)': f,
        'Speed (knots)': s,
        'Draught (meters)': d,
        'Ship heights (meters)': h,
    }


def _minimal_powered_data(
    *, include_depths: bool = True, include_objects: bool = True,
    include_traffic: bool = True,
) -> dict:
    """Synthetic project data wired specifically for the powered model."""
    data: dict = {
        'pc': {
            'p_pc': 0.00016, 'd_pc': 1e-4,
            'grounding': 1.6e-4, 'allision': 1.9e-4,
        },
        'segment_data': {
            '1': {
                'Start_Point': '14.0 55.2',
                'End_Point': '14.2 55.2',
                'Dirs': ['East going', 'West going'],
                'Width': 1000,
                'line_length': 10_000.0,
                'Route_Id': 1, 'Leg_name': 'leg 1', 'Segment_Id': '1',
                'mean1_1': 0.0, 'std1_1': 200.0, 'weight1_1': 1.0,
                'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
                'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
                'mean2_1': 0.0, 'std2_1': 200.0, 'weight2_1': 1.0,
                'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
                'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
                'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
                'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
                'ai1': 180.0, 'ai2': 180.0,
                'dist1': [], 'dist2': [],
            },
        },
    }
    if include_traffic:
        data['traffic_data'] = {
            '1': {
                'East going': _traffic_cell(),
                'West going': _traffic_cell(),
            },
        }
    if include_depths:
        # 12 m depth polygon east of the leg end (so ships continuing
        # straight past the bend hit it).
        data['depths'] = [[
            'd1', '12',
            'POLYGON((14.21 55.195, 14.24 55.195, 14.24 55.205, '
            '14.21 55.205, 14.21 55.195))',
        ]]
    if include_objects:
        data['objects'] = [[
            's1', '20',
            'POLYGON((14.21 55.195, 14.24 55.195, 14.24 55.205, '
            '14.21 55.205, 14.21 55.195))',
        ]]
    return data


def _make_mixin_host(widget_name: str) -> PoweredModelMixin:
    """Compose a trivial host class with the mock main_widget."""
    class Host(PoweredModelMixin):
        def __init__(self):
            self.p = MagicMock()
            self.p.main_widget = MagicMock()
            setattr(self.p.main_widget, widget_name, MagicMock())
    return Host()


@pytest.fixture
def grounding_host():
    return _make_mixin_host('LEPPoweredGrounding')


@pytest.fixture
def allision_host():
    return _make_mixin_host('LEPPoweredAllision')


# ---------------------------------------------------------------------------
# run_powered_grounding_model
# ---------------------------------------------------------------------------

class TestRunPoweredGrounding:
    def test_empty_traffic_returns_zero(self, grounding_host):
        """No traffic -> early return with 0.0 and widget set to '0.000e+00'."""
        data = _minimal_powered_data(include_traffic=False)
        assert grounding_host.run_powered_grounding_model(data) == 0.0
        grounding_host.p.main_widget.LEPPoweredGrounding.setText.assert_called_with('0.000e+00')

    def test_empty_depths_returns_zero(self, grounding_host):
        data = _minimal_powered_data(include_depths=False)
        assert grounding_host.run_powered_grounding_model(data) == 0.0

    def test_empty_segment_data_returns_zero(self, grounding_host):
        data = _minimal_powered_data()
        data['segment_data'] = {}
        assert grounding_host.run_powered_grounding_model(data) == 0.0

    def test_happy_path_returns_non_negative_probability(self, grounding_host):
        data = _minimal_powered_data()
        result = grounding_host.run_powered_grounding_model(data)
        assert result >= 0.0
        assert result < 1.0
        # Widget was updated with scientific-notation text.
        call = grounding_host.p.main_widget.LEPPoweredGrounding.setText.call_args
        assert call is not None
        text = call.args[0]
        assert 'e' in text  # e.g. '1.234e-05'

    def test_scales_with_traffic_frequency(self, grounding_host):
        """Doubling freq doubles the result (exposure factor is linear)."""
        data1 = _minimal_powered_data()
        r1 = grounding_host.run_powered_grounding_model(data1)

        data2 = _minimal_powered_data()
        for leg in data2['traffic_data'].values():
            for di in leg.values():
                f = di['Frequency (ships/year)']
                di['Frequency (ships/year)'] = [[v * 2 for v in row] for row in f]
        r2 = grounding_host.run_powered_grounding_model(data2)
        if r1 > 0.0:
            assert r2 == pytest.approx(2 * r1, rel=0.01)

    def test_missing_widget_setter_swallowed(self):
        """If main_widget raises on setText, the method still returns cleanly."""
        class Host(PoweredModelMixin):
            def __init__(self):
                self.p = MagicMock()
                # Accessing LEPPoweredGrounding.setText raises.
                self.p.main_widget.LEPPoweredGrounding.setText.side_effect = \
                    RuntimeError("no widget")

        host = Host()
        data = _minimal_powered_data(include_traffic=False)
        # Doesn't raise.
        assert host.run_powered_grounding_model(data) == 0.0

    def test_malformed_start_point_returns_zero(self, grounding_host):
        """Bad Start_Point string -> _parse_point raises, early return with 0."""
        data = _minimal_powered_data()
        data['segment_data']['1']['Start_Point'] = 'not-a-coord'
        assert grounding_host.run_powered_grounding_model(data) == 0.0

    def test_progress_callback_invoked(self, grounding_host):
        """When host exposes `_progress_callback`, it fires per depth bin."""
        calls = []

        def cb(cur, total, msg):
            calls.append((cur, total, msg))

        grounding_host._progress_callback = cb
        data = _minimal_powered_data()
        grounding_host.run_powered_grounding_model(data)
        # At least one progress call expected, with a "Powered grounding" message.
        assert any('Powered grounding' in msg for _, _, msg in calls)


# ---------------------------------------------------------------------------
# run_powered_allision_model
# ---------------------------------------------------------------------------

class TestRunPoweredAllision:
    def test_empty_traffic_returns_zero(self, allision_host):
        data = _minimal_powered_data(include_traffic=False)
        assert allision_host.run_powered_allision_model(data) == 0.0
        allision_host.p.main_widget.LEPPoweredAllision.setText.assert_called_with('0.000e+00')

    def test_empty_objects_returns_zero(self, allision_host):
        data = _minimal_powered_data(include_objects=False)
        assert allision_host.run_powered_allision_model(data) == 0.0

    def test_empty_segment_data_returns_zero(self, allision_host):
        data = _minimal_powered_data()
        data['segment_data'] = {}
        assert allision_host.run_powered_allision_model(data) == 0.0

    def test_happy_path_returns_non_negative_probability(self, allision_host):
        data = _minimal_powered_data()
        result = allision_host.run_powered_allision_model(data)
        assert result >= 0.0
        assert result < 1.0

    def test_height_filter_drops_short_ships(self, allision_host):
        """Ships shorter than the structure pass under (zero contribution)."""
        data_tall = _minimal_powered_data()
        r_tall = allision_host.run_powered_allision_model(data_tall)

        # Now make the ship height 0 -> always below a 20m structure.
        data_short = _minimal_powered_data()
        for leg in data_short['traffic_data'].values():
            for di in leg.values():
                h = di['Ship heights (meters)']
                di['Ship heights (meters)'] = [[0.0 for _ in row] for row in h]
        r_short = allision_host.run_powered_allision_model(data_short)
        if r_tall > 0.0:
            assert r_short < r_tall
            # Completely shadowed: should be exactly zero for a single obstacle.
            assert r_short == 0.0

    def test_malformed_start_point_returns_zero(self, allision_host):
        data = _minimal_powered_data()
        data['segment_data']['1']['Start_Point'] = 'not-a-coord'
        assert allision_host.run_powered_allision_model(data) == 0.0

    def test_missing_widget_setter_swallowed(self):
        class Host(PoweredModelMixin):
            def __init__(self):
                self.p = MagicMock()
                self.p.main_widget.LEPPoweredAllision.setText.side_effect = \
                    RuntimeError("no widget")

        host = Host()
        data = _minimal_powered_data(include_traffic=False)
        assert host.run_powered_allision_model(data) == 0.0

    def test_malformed_object_height_ignored(self, allision_host):
        """Garbage in objects list is skipped, run still returns a number."""
        data = _minimal_powered_data()
        data['objects'].append(['bad', 'not-a-float', 'POLYGON((...))'])
        # Should not raise.
        result = allision_host.run_powered_allision_model(data)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Edge cases shared by both methods
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_speed_cells_skipped(self, grounding_host):
        """Cells where speed is 0 should not add to the total."""
        data = _minimal_powered_data()
        for leg in data['traffic_data'].values():
            for di in leg.values():
                s = di['Speed (knots)']
                di['Speed (knots)'] = [[0.0 for _ in row] for row in s]
        # Without valid speeds no cells contribute.
        result = grounding_host.run_powered_grounding_model(data)
        assert result == 0.0

    def test_zero_frequency_cells_skipped(self, grounding_host):
        """Cells where frequency is 0 don't add to the total."""
        data = _minimal_powered_data()
        for leg in data['traffic_data'].values():
            for di in leg.values():
                f = di['Frequency (ships/year)']
                di['Frequency (ships/year)'] = [[0.0 for _ in row] for row in f]
        result = grounding_host.run_powered_grounding_model(data)
        assert result == 0.0

    def test_draught_fallback_when_missing(self, grounding_host):
        """When draught array is empty we still get a valid result (5 m default)."""
        data = _minimal_powered_data()
        for leg in data['traffic_data'].values():
            for di in leg.values():
                di['Draught (meters)'] = []
        result = grounding_host.run_powered_grounding_model(data)
        # Default draught (5m) is shallower than 12m depth, so no depth
        # qualifies as a grounding hazard.
        assert result == 0.0
