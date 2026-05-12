"""Regression tests for the three issues found while testing the
junction-validation feature in the live plugin:

1. Schema bug: ``TrafficLeg`` rejected any direction label other than
   ``East going`` / ``West going``, so a project with a north-south
   leg silently failed to load (Pydantic raised, the load aborted in
   the ``except ValidationError``).

2. Data-loss bug: ``apply_waypoint_merge`` updated ``segment_data``
   in memory, but ``GatherData.get_segment_tbl`` reads endpoints back
   from ``twRouteList`` on save, overwriting the merge.

3. UI bug: the OMRAT dock covered the canvas while the merge prompts
   were on screen, so the user could not see the candidate waypoints.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fix 1 -- arbitrary direction labels in TrafficLeg
# ---------------------------------------------------------------------------


def _minimal_block(directions=("North going", "South going")):
    """Build a smallest-possible valid project block to exercise the schema."""
    one_traffic = {
        "Frequency (ships/year)": [[0]],
        "Speed (knots)": [[0]],
        "Draught (meters)": [[0]],
        "Ship heights (meters)": [[0]],
        "Ship Beam (meters)": [[0]],
    }
    return {
        "pc": {"p_pc": 0.0, "d_pc": 0.0},
        "drift": {
            "drift_p": 0,
            "anchor_p": 0.7,
            "anchor_d": 0,
            "speed": 1.0,
            "rose": {"0": 1.0},
            "repair": {
                "func": "x",
                "std": 1.0,
                "loc": 0.0,
                "scale": 1.0,
                "use_lognormal": False,
            },
        },
        "traffic_data": {
            "1": {directions[0]: one_traffic, directions[1]: one_traffic},
        },
        "segment_data": {
            "1": {
                "Start_Point": "14 55", "End_Point": "14 56",
                "Dirs": list(directions), "Width": 5000, "line_length": 1000.0,
                "Route_Id": 1, "Leg_name": "L1", "Segment_Id": "1",
                "mean1_1": 0.0, "std1_1": 0.0, "mean2_1": 0.0, "std2_1": 0.0,
                "weight1_1": 0.0, "weight2_1": 0.0,
                "mean1_2": 0.0, "mean1_3": 0.0, "std1_2": 0.0, "std1_3": 0.0,
                "mean2_2": 0.0, "mean2_3": 0.0, "std2_2": 0.0, "std2_3": 0.0,
                "weight1_2": 0.0, "weight1_3": 0.0,
                "weight2_2": 0.0, "weight2_3": 0.0,
                "u_min1": 0.0, "u_max1": 0.0, "u_p1": 0, "ai1": 0.0,
                "u_min2": 0.0, "u_max2": 0.0, "u_p2": 0, "ai2": 0.0,
            },
        },
        "depths": [],
        "objects": [],
    }


@pytest.mark.parametrize("dirs", [
    ("North going", "South going"),
    ("South going", "North going"),
    ("East going", "West going"),
    ("West going", "East going"),
])
def test_traffic_leg_schema_accepts_all_compass_pairs(dirs):
    """All four direction-label combinations the bearing logic emits must validate."""
    from omrat_utils.validate_data import RootModelSchema
    block = _minimal_block(directions=dirs)
    RootModelSchema.model_validate(block)  # raises if it fails


def test_traffic_leg_schema_accepts_arbitrary_label():
    """Even a non-canonical label should pass -- the schema can no longer
    block legitimate user-drawn legs because of label mismatch."""
    from omrat_utils.validate_data import RootModelSchema
    block = _minimal_block(directions=("Something going", "Other going"))
    RootModelSchema.model_validate(block)


# ---------------------------------------------------------------------------
# Fix 2 -- canvas/table rebuild after validation pass
# ---------------------------------------------------------------------------


def test_validation_pass_calls_qgis_geoms_reload_after_merge():
    """run_validation_pass must call reload_legs_from_segment_data so the
    canvas and twRouteList catch up with the merged segment_data."""
    from geometries.route_validation import parse_wkt_point
    from omrat_utils.route_validation_ui import (
        run_validation_pass, _MergeChoice,
    )
    sd = {
        '1': {
            'Start_Point': "14.0 55.0", 'End_Point': "15.0 55.0",
            'line_length': 100_000, 'Width': 5000, 'Route_Id': 1, 'Leg_name': 'L1',
        },
        '2': {
            'Start_Point': "15.0001 55.0", 'End_Point': "16.0 55.0",
            'line_length': 100_000, 'Width': 5000, 'Route_Id': 1, 'Leg_name': 'L2',
        },
    }
    qgis_geoms = MagicMock()
    omrat = SimpleNamespace(
        segment_data=sd, traffic_data={}, main_widget=None, iface=None,
        junctions=None, qgis_geoms=qgis_geoms,
    )
    run_validation_pass(
        omrat,
        show_dialog=lambda kind, payload: (
            _MergeChoice(target=payload.point_a) if kind == 'merge' else False
        ),
    )
    qgis_geoms.reload_legs_from_segment_data.assert_called_once()
    # And the dict actually got merged.
    assert parse_wkt_point(sd['2']['Start_Point']) == (14.0, 55.0) or \
           parse_wkt_point(sd['2']['Start_Point']) == (15.0, 55.0)


def test_validation_pass_skips_reload_when_nothing_changed():
    """No mutations -> no canvas teardown."""
    from omrat_utils.route_validation_ui import run_validation_pass
    sd = {
        '1': {
            'Start_Point': "14.0 55.0", 'End_Point': "15.0 55.0",
            'line_length': 100_000, 'Width': 5000, 'Route_Id': 1, 'Leg_name': 'L1',
        },
    }
    qgis_geoms = MagicMock()
    omrat = SimpleNamespace(
        segment_data=sd, traffic_data={}, main_widget=None, iface=None,
        junctions=None, qgis_geoms=qgis_geoms,
    )
    run_validation_pass(omrat, show_dialog=lambda kind, payload: None)
    qgis_geoms.reload_legs_from_segment_data.assert_not_called()


def test_validation_pass_calls_reload_after_split():
    from omrat_utils.route_validation_ui import run_validation_pass
    sd = {
        '1': {
            'Start_Point': "14.0 55.0", 'End_Point': "16.0 56.0",
            'line_length': 200_000, 'Width': 5000, 'Route_Id': 1, 'Leg_name': 'L1',
        },
        '2': {
            'Start_Point': "14.0 56.0", 'End_Point': "16.0 55.0",
            'line_length': 200_000, 'Width': 5000, 'Route_Id': 1, 'Leg_name': 'L2',
        },
    }
    qgis_geoms = MagicMock()
    omrat = SimpleNamespace(
        segment_data=sd, traffic_data={}, main_widget=None, iface=None,
        junctions=None, qgis_geoms=qgis_geoms,
    )
    run_validation_pass(
        omrat,
        show_dialog=lambda kind, payload: True if kind == 'split' else None,
    )
    qgis_geoms.reload_legs_from_segment_data.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 3 -- dock visibility around prompts
# ---------------------------------------------------------------------------


def test_dock_hidden_during_prompts_and_restored_after():
    """When run_validation_pass prompts the user, the OMRAT dock must
    be hidden so the canvas is visible, then restored at the end."""
    from omrat_utils.route_validation_ui import (
        _hide_dock_for_prompts, _restore_dock,
    )
    dock = MagicMock()
    dock.isVisible.return_value = True
    omrat = SimpleNamespace(main_widget=dock)
    was_visible = _hide_dock_for_prompts(omrat)
    assert was_visible is True
    dock.hide.assert_called_once()
    _restore_dock(omrat, was_visible)
    dock.show.assert_called_once()
    dock.raise_.assert_called_once()


def test_dock_not_restored_when_originally_hidden():
    from omrat_utils.route_validation_ui import (
        _hide_dock_for_prompts, _restore_dock,
    )
    dock = MagicMock()
    dock.isVisible.return_value = False
    omrat = SimpleNamespace(main_widget=dock)
    was_visible = _hide_dock_for_prompts(omrat)
    assert was_visible is False
    dock.hide.assert_not_called()
    _restore_dock(omrat, was_visible)
    dock.show.assert_not_called()


def test_dock_handlers_tolerate_missing_main_widget():
    """A bare omrat-like object without a dock must not crash."""
    from omrat_utils.route_validation_ui import (
        _hide_dock_for_prompts, _restore_dock,
    )
    omrat = SimpleNamespace(main_widget=None)
    was_visible = _hide_dock_for_prompts(omrat)
    assert was_visible is None
    _restore_dock(omrat, was_visible)  # no exception


# ---------------------------------------------------------------------------
# End-to-end: the user-supplied broken file must now load through the
# full normaliser + validator chain.
# ---------------------------------------------------------------------------


def test_user_supplied_t1_omrat_validates_after_schema_fix():
    """The repro file the user reported must validate against the
    updated RootModelSchema."""
    import os
    path = os.path.join(
        os.path.dirname(__file__), '..', 't1.omrat',
    )
    if not os.path.exists(path):
        pytest.skip("t1.omrat not present in repo")
    from omrat_utils.validate_data import RootModelSchema
    with open(path) as f:
        data = json.load(f)
    RootModelSchema.model_validate(data)
