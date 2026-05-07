"""Regression test for the stale-traffic-reference bug in GatherData.populate.

The bug: `Traffic.__init__` captures `self.traffic_data =
self.omrat.traffic_data` as a reference at construction time.  When the
user loaded a project file, `gather.populate` rebound
`self.p.traffic_data = data['traffic_data']` but the `Traffic`
instance's reference still pointed to the original (empty) dict.
Result: after loading a project, the direction dropdown
(`cbTrafficDirectionSelect`) stayed empty because
`Traffic.update_direction_select` read the stale dict.

The fix syncs `self.p.traffic.traffic_data` to the loaded dict too.
This test simulates the load path with a minimal stub that mirrors the
plugin's attribute layout and asserts the references end up aligned.

Run standalone:
    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \\
        --noconftest tests/test_regression_gather_data.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def plugin_stub():
    """A minimal stub of the OMRAT plugin object that gather.populate
    touches.  Uses MagicMock for widgets, real attributes for the dicts
    whose *reference identity* we need to verify.
    """
    original_traffic_data: dict = {}  # the "old" empty dict held by Traffic
    plugin = SimpleNamespace()
    plugin.traffic_data = original_traffic_data
    plugin.segment_data = {}
    plugin.drift_values = None
    plugin.drift_settings = SimpleNamespace(drift_values=None)
    # ``CausationF.data`` on the real plugin is a dict of factor presets
    # that ``populate`` merges the loaded ``pc`` block over.  The stub
    # only needs ``.update`` to be callable, so a plain dict suffices.
    plugin.causation_f = SimpleNamespace(data={})
    plugin.main_widget = MagicMock()
    plugin.main_widget.cbTrafficSelectSeg.clear = MagicMock()
    plugin.main_widget.cbTrafficSelectSeg.addItem = MagicMock()
    plugin.main_widget.cbTrafficSelectSeg.currentText = MagicMock(
        return_value="1"
    )
    plugin.main_widget.twDepthList = MagicMock()
    plugin.main_widget.twObjectList = MagicMock()
    plugin.main_widget.leNormMean1_1 = MagicMock()
    plugin.main_widget.leNormMean1_1.setText = MagicMock()
    plugin.distributions = MagicMock()
    plugin.ship_cat = SimpleNamespace(
        scw=SimpleNamespace(
            cvTypes=MagicMock(rowCount=MagicMock(return_value=0)),
            twLengths=MagicMock(rowCount=MagicMock(return_value=0)),
        )
    )
    plugin.object = MagicMock()
    plugin.object._apply_depth_graduated_style = MagicMock()
    # The Traffic instance that captures a stale reference at __init__.
    plugin.traffic = SimpleNamespace(
        traffic_data=original_traffic_data,
        set_table_headings=MagicMock(),
    )
    plugin.load_lines = MagicMock()
    plugin._original_traffic_dict = original_traffic_data
    return plugin


@pytest.fixture
def loaded_data():
    """A small, valid `data` dict that mimics what storage.load_from_path
    yields after the legacy-normalisation step.
    """
    return {
        'traffic_data': {
            '1': {
                'East going': {
                    'Frequency (ships/year)': [[0]],
                    'Speed (knots)': [[0]],
                    'Draught (meters)': [[0]],
                    'Ship heights (meters)': [[0]],
                    'Ship Beam (meters)': [[0]],
                },
                'West going': {
                    'Frequency (ships/year)': [[0]],
                    'Speed (knots)': [[0]],
                    'Draught (meters)': [[0]],
                    'Ship heights (meters)': [[0]],
                    'Ship Beam (meters)': [[0]],
                },
            },
        },
        'segment_data': {
            '1': {
                'Start_Point': [14.0, 55.0],
                'End_Point': [14.5, 55.2],
                'Leg_name': 'leg 1',
                'Width': 1000,
                'dist1': [],
                'dist2': [],
            },
        },
        'drift': {
            'drift_p': 1.0,
            'anchor_p': 0.7,
            'anchor_d': 7,
            'speed': 1.94,
            'rose': {
                '0': 0.125, '45': 0.125, '90': 0.125, '135': 0.125,
                '180': 0.125, '225': 0.125, '270': 0.125, '315': 0.125,
            },
            'repair': {
                'func': "1", 'std': 0.95, 'loc': 0.2, 'scale': 0.85,
                'use_lognormal': False,
            },
        },
        'pc': {},
        'depths': [],
        'objects': [],
        'ship_categories': {'types': [], 'length_intervals': []},
    }


def test_gather_populate_syncs_traffic_reference(plugin_stub, loaded_data):
    """After populate() runs, `plugin.traffic.traffic_data` must point
    to the NEWLY-loaded dict (not the stale original held by Traffic).
    """
    from omrat_utils.gather_data import GatherData

    gd = GatherData(plugin_stub)

    # Patch out the heavy branches so we can isolate the traffic-data
    # sync.  The .populate_tbl and shapely-using load_area() paths
    # require real QTableWidgets / shapely WKT; they aren't relevant
    # to the reference-sync check.
    with patch.object(GatherData, 'populate_segment_tbl'), \
         patch.object(GatherData, 'populate_cbTrafficSelectSeg'), \
         patch.object(GatherData, 'populate_tbl'), \
         patch.object(GatherData, 'normalize_depths_for_ui',
                      return_value=[]), \
         patch.object(GatherData, 'normalize_objects_for_ui',
                      return_value=[]), \
         patch.object(GatherData, 'populate_ship_categories'):
        gd.populate(loaded_data)

    # Reference identity: plugin.traffic_data is the loaded dict
    assert plugin_stub.traffic_data is loaded_data['traffic_data']
    # ...AND plugin.traffic.traffic_data now points to the same dict
    assert plugin_stub.traffic.traffic_data is loaded_data['traffic_data']
    # ...AND crucially, it is no longer the stale original empty dict
    assert plugin_stub.traffic.traffic_data is not plugin_stub._original_traffic_dict


def test_gather_populate_exposes_loaded_directions(plugin_stub, loaded_data):
    """Without the reference sync, iterating `plugin.traffic.traffic_data`
    would yield 0 segments / 0 directions, explaining the empty
    direction dropdown the user reported.  With the sync, the directions
    come through.
    """
    from omrat_utils.gather_data import GatherData

    gd = GatherData(plugin_stub)
    with patch.object(GatherData, 'populate_segment_tbl'), \
         patch.object(GatherData, 'populate_cbTrafficSelectSeg'), \
         patch.object(GatherData, 'populate_tbl'), \
         patch.object(GatherData, 'normalize_depths_for_ui',
                      return_value=[]), \
         patch.object(GatherData, 'normalize_objects_for_ui',
                      return_value=[]), \
         patch.object(GatherData, 'populate_ship_categories'):
        gd.populate(loaded_data)

    seg_key = '1'
    directions = list(plugin_stub.traffic.traffic_data.get(seg_key, {}).keys())
    assert directions == ['East going', 'West going'], (
        f"Expected ['East going', 'West going'] after loading proj_3_3-"
        f"like data, got {directions!r} -- the Traffic instance is "
        f"holding a stale reference to the original empty dict."
    )
