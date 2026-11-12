"""Unit tests for ``compute.visualization.VisualizationMixin``.

The three ``run_*_visualization`` methods each:
  1. Guard against missing traffic/segment data with an early return.
  2. Create a ``ShowGeomRes`` dialog.
  3. Delegate to the ``*OverlapVisualizer.show_in_dialog`` helper.
  4. Call ``dialog.exec()`` (blocking).

These tests exercise (1) and (3) with the dialog / visualizer helpers
patched to avoid popping a real Qt window.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.visualization import VisualizationMixin  # noqa: E402


@pytest.fixture
def host():
    class Host(VisualizationMixin):
        def __init__(self):
            self.p = MagicMock()
            self.canvas = MagicMock()
    return Host()


def _minimal_powered_data() -> dict:
    return {
        'traffic_data': {'1': {'East going': {}}},
        'segment_data': {
            '1': {'Start_Point': '14.0 55.2', 'End_Point': '14.2 55.2'},
        },
        'max_draft': 15.0,
    }


def _minimal_drift_data() -> dict:
    return {
        'traffic_data': {'1': {'East going': {}}},
        'segment_data': {
            '1': {'Start_Point': '14.0 55.2', 'End_Point': '14.2 55.2'},
        },
        'objects': [
            ['s1', '20',
             'POLYGON((14.05 55.22, 14.10 55.22, 14.10 55.23, '
             '14.05 55.23, 14.05 55.22))'],
        ],
    }


# ---------------------------------------------------------------------------
# run_drift_visualization
# ---------------------------------------------------------------------------

class TestRunDriftVisualization:
    def test_missing_traffic_returns_without_dialog(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_drift_visualization({})
            MockDlg.assert_not_called()

    def test_empty_traffic_returns_without_dialog(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_drift_visualization({'traffic_data': {}})
            MockDlg.assert_not_called()

    def test_happy_path_builds_dialog_and_calls_show(self, host):
        from shapely.geometry import LineString, box
        data = _minimal_drift_data()
        # Extra ingredients run_drift_visualization's prepare_traffic_lists
        # expects; stub them via patch rather than a real project.
        fake_lines = [LineString([(0, 0), (1000, 0)])]
        fake_objs = [box(100, 10, 200, 20)]
        with patch('compute.visualization.ShowGeomRes') as MockDlg, \
             patch('compute.visualization.prepare_traffic_lists',
                   return_value=(fake_lines, [], [], ['leg 1'])), \
             patch('compute.visualization.transform_to_utm',
                   return_value=(fake_lines, fake_objs, MagicMock())), \
             patch('compute.visualization.DriftingOverlapVisualizer') as MockViz:
            dialog_instance = MockDlg.return_value
            host.run_drift_visualization(data)
            MockDlg.assert_called_once_with(host.p.main_widget)
            MockViz.show_in_dialog.assert_called_once()
            # PyQt6 (QGIS 4) dropped ``exec_``; production calls ``exec``.
            dialog_instance.exec.assert_called_once()


# ---------------------------------------------------------------------------
# run_powered_allision_visualization
# ---------------------------------------------------------------------------

class TestRunPoweredAllisionVisualization:
    def test_missing_traffic_returns_early(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_powered_allision_visualization({})
            MockDlg.assert_not_called()

    def test_missing_segment_data_returns_early(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_powered_allision_visualization(
                {'traffic_data': {'1': {}}})
            MockDlg.assert_not_called()

    def test_happy_path_invokes_visualizer(self, host):
        data = _minimal_powered_data()
        with patch('compute.visualization.ShowGeomRes') as MockDlg, \
             patch('compute.visualization.PoweredOverlapVisualizer') as MockViz:
            dialog_instance = MockDlg.return_value
            host.run_powered_allision_visualization(data)
            MockDlg.assert_called_once_with(host.p.main_widget)
            MockViz.show_in_dialog.assert_called_once()
            # Confirm mode/max_draft threaded through.
            call = MockViz.show_in_dialog.call_args
            assert call.kwargs.get('mode') == 'allision'
            assert call.kwargs.get('max_draft') == 15.0
            dialog_instance.exec.assert_called_once()

    def test_invalid_max_draft_falls_back_to_15(self, host):
        data = _minimal_powered_data()
        data['max_draft'] = 'nonsense'
        with patch('compute.visualization.ShowGeomRes'), \
             patch('compute.visualization.PoweredOverlapVisualizer') as MockViz:
            host.run_powered_allision_visualization(data)
            assert MockViz.show_in_dialog.call_args.kwargs['max_draft'] == 15.0


# ---------------------------------------------------------------------------
# run_powered_grounding_visualization
# ---------------------------------------------------------------------------

class TestRunPoweredGroundingVisualization:
    def test_missing_traffic_returns_early(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_powered_grounding_visualization({})
            MockDlg.assert_not_called()

    def test_missing_segment_data_returns_early(self, host):
        with patch('compute.visualization.ShowGeomRes') as MockDlg:
            host.run_powered_grounding_visualization(
                {'traffic_data': {'1': {}}})
            MockDlg.assert_not_called()

    def test_happy_path_shows_the_deepest_draught_bin(self, host):
        """Not a fixed 15 m any more: the deepest depth bin the model casts
        for the traffic (a 9 m draught + squat grounds on the 8 m class)."""
        data = _minimal_powered_data()
        data['traffic_data'] = {'1': {'East going': {
            'Frequency (ships/year)': [[100.0]], 'Speed (knots)': [[10.0]], 'Draught (meters)': [[9.0]]}}}
        poly = 'POLYGON((14.3 55.3, 14.31 55.3, 14.31 55.31, 14.3 55.31, 14.3 55.3))'
        data['depths'] = [['1', '3', poly], ['2', '8', poly], ['3', '12', poly]]
        with patch('compute.visualization.ShowGeomRes') as MockDlg, \
             patch('compute.visualization.PoweredOverlapVisualizer') as MockViz:
            dialog_instance = MockDlg.return_value
            host.run_powered_grounding_visualization(data)
            call = MockViz.show_in_dialog.call_args
            assert call.kwargs.get('mode') == 'grounding'
            assert call.kwargs.get('max_draft') == 8.0
            dialog_instance.exec.assert_called_once()

    def test_no_ship_can_ground_shows_no_depth_obstacles(self, host):
        data = _minimal_powered_data()   # no depth areas at all
        with patch('compute.visualization.ShowGeomRes'), \
             patch('compute.visualization.PoweredOverlapVisualizer') as MockViz:
            host.run_powered_grounding_visualization(data)
            assert MockViz.show_in_dialog.call_args.kwargs['max_draft'] == -1.0
